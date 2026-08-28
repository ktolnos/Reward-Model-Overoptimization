"""OpenAICompatibleBackend: hosted LLM judges over an OpenAI-compatible API.

One backend serves every provider (OpenRouter, the Vector Institute proxy, any
self-hosted OpenAI-compatible server), so these tests cover both the shared
machinery and the per-provider deviations that ``OPENAI_PROVIDERS`` encodes.

Everything runs against a local stub server: no API key, no network, no model
grant. That pins the parts of the backend that would otherwise only be exercised
-- and only be discoverable as broken -- in the middle of a real eval:

  - the reasoning dialect sent per provider, model family and thinking mode,
  - recovery from the failure modes a shared/hosted API actually produces
    (429 rate limiting, and a 400 rejecting the reasoning fields),
  - the reasoning-channel fallback for models that leave ``content`` empty,
  - the whole Batch API round trip (upload -> submit -> poll -> demux),
  - end-to-end judging through ``LLMJudge``, including the position swap.
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy_eval.judges import (
    OPENAI_PROVIDERS,
    JudgeAccessError,
    JudgeGenParams,
    LLMJudge,
    OpenAICompatibleBackend,
    _RateLimiter,
)


# ---------------------------------------------------------------------------
# Stub proxy
# ---------------------------------------------------------------------------

class StubProxy:
    """Minimal OpenAI-compatible server: chat/completions, files and batches.

    The stub judges by answer length -- it declares whichever of assistant A/B
    wrote more the winner. The tests pair a long "chosen" answer against a short
    "rejected" one, so a correctly wired judge must return a clean sweep for the
    chosen response, and any mix-up in the position swap shows up as a 0.5.
    """

    def __init__(self, **modes):
        self.n_chat = 0
        self.n_429 = 0
        self.n_400 = 0
        self.bodies = []
        self.files = {}
        self.batches = {}
        self._next = 0
        self._lock = threading.Lock()
        self.reject_extras = modes.get("reject_extras", False)
        self.rate_limit_first = modes.get("rate_limit_first", 0)
        self.reasoning_only = modes.get("reasoning_only", False)
        # HTTP status for a permanent rejection (bad key / no model grant).
        self.deny_with = modes.get("deny_with", None)
        self._srv = HTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._srv.server_port}/v1"

    def close(self):
        self._srv.shutdown()

    def _new_id(self, kind):
        with self._lock:
            self._next += 1
            return f"{kind}-{self._next}"

    def _verdict(self, body):
        user = body["messages"][-1]["content"]
        pat = (r"<\|The Start of Assistant {0}'s Answer\|>\n(.*?)\n"
               r"<\|The End of Assistant {0}'s Answer\|>")
        a = re.search(pat.format("A"), user, re.S)
        b = re.search(pat.format("B"), user, re.S)
        assert a and b, "stub could not find both answers in the judge prompt"
        return "[[A>>B]]" if len(a.group(1)) > len(b.group(1)) else "[[B>>A]]"

    def _choice(self, body):
        text = f"My final verdict is {self._verdict(body)}"
        if self.reasoning_only:
            # Verdict only in the reasoning channel, content empty.
            return {"message": {"content": "", "reasoning_content": text},
                    "finish_reason": "stop"}
        return {"message": {"content": text, "reasoning_content": "analysis..."},
                "finish_reason": "stop"}

    def _handler(stub):  # noqa: N805 — closure over the stub instance
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, raw=False):
                payload = obj if raw else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if code == 429:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))

                if self.path.endswith("/chat/completions"):
                    body = json.loads(raw)
                    with stub._lock:
                        stub.n_chat += 1
                        idx = stub.n_chat
                        stub.bodies.append(body)
                    if stub.deny_with:
                        return self._send(stub.deny_with,
                                          {"detail": "Project does not have access to this model"})
                    extras = "reasoning_effort" in body or "chat_template_kwargs" in body
                    if stub.reject_extras and extras:
                        stub.n_400 += 1
                        return self._send(400, {"detail": "unrecognized field"})
                    if idx <= stub.rate_limit_first:
                        stub.n_429 += 1
                        return self._send(429, {"detail": "Rate limit exceeded (RPM)",
                                                "limit_type": "rpm"})
                    return self._send(200, {"choices": [stub._choice(body)]})

                if self.path.endswith("/files"):
                    boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
                    content = None
                    for part in raw.split(b"--" + boundary):
                        if b"filename=" in part:
                            content = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                            break
                    assert content is not None, "no file part in upload"
                    fid = stub._new_id("file")
                    stub.files[fid] = content
                    return self._send(200, {"id": fid, "object": "file"})

                if self.path.endswith("/batches"):
                    body = json.loads(raw)
                    out = []
                    for line in stub.files[body["input_file_id"]].decode().splitlines():
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        out.append(json.dumps({
                            "custom_id": rec["custom_id"],
                            "response": {"status_code": 200,
                                         "body": {"choices": [stub._choice(rec["body"])]}},
                        }))
                    oid = stub._new_id("file")
                    stub.files[oid] = ("\n".join(out) + "\n").encode()
                    bid = stub._new_id("batch")
                    # polls==0 now; the first status poll reports in_progress and
                    # the second completed, so the poll loop is genuinely exercised.
                    stub.batches[bid] = {"polls": 0, "out": oid, "n": len(out)}
                    return self._send(200, {"id": bid, "status": "validating"})

                return self._send(404, {"detail": "not found"})

            def do_GET(self):
                m = re.match(r".*/batches/([^/]+)$", self.path)
                if m:
                    b = stub.batches[m.group(1)]
                    b["polls"] += 1
                    done = b["polls"] >= 2
                    return self._send(200, {
                        "id": m.group(1),
                        "status": "completed" if done else "in_progress",
                        "output_file_id": b["out"] if done else None,
                        "request_counts": {"total": b["n"],
                                           "completed": b["n"] if done else 0},
                    })
                m = re.match(r".*/files/([^/]+)/content$", self.path)
                if m:
                    return self._send(200, stub.files[m.group(1)], raw=True)
                return self._send(404, {"detail": "not found"})

        return Handler


@pytest.fixture
def stub():
    s = StubProxy()
    yield s
    s.close()


# A long "chosen" answer vs a short "rejected" one, as in the preference dataset.
PROMPTS = [[{"role": "user", "content": f"Q{i}: how do I sort a list in Python?"}]
           for i in range(4)]
CHOSEN = ["Use sorted(lst) for a new list, or lst.sort() to sort in place. "
          "Pass key= for a custom ordering and reverse=True to descend. " * 3] * 4
REJECTED = ["idk just sort it"] * 4


def _params(thinking=False, max_tokens=512):
    return JudgeGenParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                          enable_thinking=thinking)


def _backend(stub, **kw):
    kw.setdefault("requests_per_minute", 0)  # no pacing: keeps tests fast
    return OpenAICompatibleBackend("gpt-oss-120b", provider="vector",
                                   api_key="test-key", base_url=stub.base_url,
                                   max_parallel=4, **kw)


def _judge_all(stub, **kw):
    judge = LLMJudge(_backend(stub, **kw), gen_params=_params())
    return judge.score_pairs(PROMPTS, CHOSEN, REJECTED, ctx=None)


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thinking,expected", [(True, "high"), (False, "low")])
def test_gpt_oss_maps_thinking_to_reasoning_effort(thinking, expected):
    """gpt-oss always reasons (harmony); only the effort dial exists."""
    body = OpenAICompatibleBackend("gpt-oss-120b", provider="vector")._body([], _params(thinking))
    assert body["reasoning_effort"] == expected
    assert "chat_template_kwargs" not in body


@pytest.mark.parametrize("thinking", [True, False])
def test_non_gpt_oss_uses_chat_template_kwargs(thinking):
    """vLLM-served models take the Qwen-style chat-template toggle instead."""
    body = OpenAICompatibleBackend("Qwen3_8-27B", provider="vector")._body([], _params(thinking))
    assert body["chat_template_kwargs"] == {"enable_thinking": thinking}
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("thinking,expected", [
    # OpenRouter has a native toggle, so only the "off" case sends anything --
    # leaving the model's own default reasoning behaviour untouched otherwise.
    (True, None),
    (False, {"enabled": False}),
])
def test_openrouter_uses_its_native_reasoning_toggle(thinking, expected):
    body = OpenAICompatibleBackend("openai/gpt-4.1", provider="openrouter")._body(
        [], _params(thinking))
    assert body.get("reasoning") == expected
    # The Vector-only dialects must never leak onto OpenRouter.
    assert "reasoning_effort" not in body
    assert "chat_template_kwargs" not in body


def test_openrouter_forwards_an_explicit_effort():
    b = OpenAICompatibleBackend("openai/gpt-4.1", provider="openrouter",
                                reasoning_effort="high")
    assert b._body([], _params(thinking=False))["reasoning"] == {"effort": "high"}


def test_providers_carry_their_own_endpoint_key_and_pacing():
    orouter = OpenAICompatibleBackend("m", provider="openrouter")
    vector = OpenAICompatibleBackend("m", provider="vector")
    assert orouter.api_key_env == "OPENROUTER_API_KEY"
    assert vector.api_key_env == "VECTOR_INFERENCE_API_KEY"
    assert "openrouter.ai" in orouter.base_url
    assert "vectorinstitute.ai" in vector.base_url
    # Vector publishes a shared project RPM budget; OpenRouter is unpaced.
    assert vector.requests_per_minute == 100.0
    assert orouter.requests_per_minute == 0.0


def test_explicit_rpm_overrides_the_provider_default():
    b = OpenAICompatibleBackend("m", provider="vector", requests_per_minute=7)
    assert b.requests_per_minute == 7


def test_base_url_can_be_overridden():
    b = OpenAICompatibleBackend("m", provider="vector",
                                base_url="http://localhost:8000/v1/")
    assert b.base_url == "http://localhost:8000/v1"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        OpenAICompatibleBackend("m", provider="nope")


def test_batch_api_rejected_for_providers_without_one():
    """OpenRouter has no Batch API; fail at construction, not mid-run."""
    assert OPENAI_PROVIDERS["openrouter"].supports_batch is False
    with pytest.raises(ValueError, match="no Batch API"):
        OpenAICompatibleBackend("m", provider="openrouter", use_batch_api=True)


def test_explicit_reasoning_effort_overrides_thinking_flag():
    b = OpenAICompatibleBackend("gpt-oss-120b", provider="vector", reasoning_effort="medium")
    assert b._body([], _params(thinking=False))["reasoning_effort"] == "medium"


def test_reasoning_effort_none_omits_the_field():
    b = OpenAICompatibleBackend("gpt-oss-120b", provider="vector", reasoning_effort="none")
    assert "reasoning_effort" not in b._body([], _params())


# ---------------------------------------------------------------------------
# End-to-end judging
# ---------------------------------------------------------------------------

def test_judges_chosen_over_rejected(stub):
    """A clean sweep for the longer (chosen) answer, and a correct position swap.

    game0 is A=rejected/B=chosen and game1 is A=chosen/B=rejected, so the two
    games must return mirrored labels; identical labels would mean the swap was
    dropped and the battle scores would average to 0.5.
    """
    battles, details = _judge_all(stub)
    flat = [b for bs in battles for b in bs]
    assert flat and all(b == 1.0 for b in flat)
    assert details.game0_labels == ["B>>A"] * 4
    assert details.game1_labels == ["A>>B"] * 4
    assert details.n_dropped_prompts == 0
    assert details.n_generation_failures == 0
    assert details.n_parse_failures == 0


def test_recovers_from_rate_limiting(stub):
    """429s are retried (honouring Retry-After) rather than dropped."""
    stub.rate_limit_first = 5
    battles, details = _judge_all(stub)
    assert stub.n_429 == 5
    assert details.n_dropped_prompts == 0
    assert all(b == 1.0 for bs in battles for b in bs)


def test_strips_extras_after_a_400_and_stops_resending_them(stub):
    """A server rejecting reasoning_effort costs one 400, not the whole run."""
    stub.reject_extras = True
    backend = _backend(stub)
    judge = LLMJudge(backend, gen_params=_params())
    battles, details = judge.score_pairs(PROMPTS, CHOSEN, REJECTED, ctx=None)
    assert details.n_dropped_prompts == 0
    assert all(b == 1.0 for bs in battles for b in bs)
    assert backend._strip_extras is True
    # Whatever the concurrent workers raced through before the flag was set, the
    # run must not have kept resending the doomed field for all 8 games.
    assert stub.n_400 < 8
    assert not any("reasoning_effort" in b for b in stub.bodies[-4:])


def test_falls_back_to_reasoning_content(stub):
    """A verdict emitted only in the reasoning channel still parses."""
    stub.reasoning_only = True
    battles, details = _judge_all(stub)
    assert details.n_generation_failures == 0
    assert all(b == 1.0 for bs in battles for b in bs)


def test_returns_empty_generation_when_the_server_never_recovers(stub):
    """Exhausted retries degrade to a generation failure, not an exception."""
    stub.rate_limit_first = 10_000
    backend = _backend(stub, max_retries=2)
    gens = backend.generate([[{"role": "user", "content": "x"}]], _params())
    assert len(gens) == 1 and gens[0].text == ""


@pytest.mark.parametrize("code", [401, 403])
def test_access_errors_fail_fast_instead_of_dropping_every_prompt(stub, code):
    """401/403 is the same for every request, so it must raise, not retry.

    Retrying would spend the full backoff schedule on all 2N games and then hand
    back empty verdicts, reporting a misconfigured key or a missing model grant
    as "the judge dropped every prompt".
    """
    stub.deny_with = code
    backend = _backend(stub, max_retries=6)
    t0 = time.monotonic()
    with pytest.raises(JudgeAccessError, match="gpt-oss-120b"):
        backend.generate([[{"role": "user", "content": "x"}]], _params())
    assert time.monotonic() - t0 < 2.0, "must not have run the backoff schedule"
    assert stub.n_chat == 1, "must not retry a permanent rejection"


def test_access_error_propagates_out_of_the_judge(stub):
    stub.deny_with = 403
    judge = LLMJudge(_backend(stub), gen_params=_params())
    with pytest.raises(JudgeAccessError):
        judge.score_pairs(PROMPTS, CHOSEN, REJECTED, ctx=None)


def test_missing_api_key_is_reported_clearly(monkeypatch):
    monkeypatch.delenv("VECTOR_INFERENCE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VECTOR_INFERENCE_API_KEY"):
        OpenAICompatibleBackend("gpt-oss-120b", provider="vector").generate([], _params())


def test_results_are_returned_in_request_order(stub):
    """Concurrent fan-out must not permute results relative to conversations."""
    convs = [[{"role": "user", "content":
               f"<|The Start of Assistant A's Answer|>\n{'x' * (i + 1)}\n"
               f"<|The End of Assistant A's Answer|>\n\n"
               f"<|The Start of Assistant B's Answer|>\n{'y' * (20 - i)}\n"
               f"<|The End of Assistant B's Answer|>"}] for i in range(12)]
    gens = _backend(stub).generate(convs, _params())
    expected = ["[[A>>B]]" if (i + 1) > (20 - i) else "[[B>>A]]" for i in range(12)]
    assert [g.text.split("verdict is ")[1] for g in gens] == expected


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def test_batch_api_round_trip(stub):
    """Upload, submit, poll to completion, then demux results by custom_id."""
    battles, details = _judge_all(stub, use_batch_api=True, batch_poll_seconds=0.01)
    assert details.game0_labels == ["B>>A"] * 4
    assert details.game1_labels == ["A>>B"] * 4
    assert all(b == 1.0 for bs in battles for b in bs)
    # One batch, and the poll loop really waited for a non-terminal status.
    (batch,) = stub.batches.values()
    assert batch["polls"] >= 2
    assert stub.n_chat == 0, "batch mode must not hit chat/completions"


def test_batch_uploads_one_jsonl_request_per_game(stub):
    _judge_all(stub, use_batch_api=True, batch_poll_seconds=0.01)
    uploaded = min(stub.files.items(), key=lambda kv: int(kv[0].split("-")[1]))[1]
    lines = [json.loads(x) for x in uploaded.decode().splitlines() if x.strip()]
    assert len(lines) == 8  # 4 prompts x 2 position-swapped games
    assert [r["custom_id"] for r in lines] == [str(i) for i in range(8)]
    assert all(r["url"] == "/v1/chat/completions" for r in lines)
    assert all(r["body"]["model"] == "gpt-oss-120b" for r in lines)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_paces_concurrent_callers():
    """N acquires under a cap take at least (N-1) intervals, even across threads."""
    import concurrent.futures as cf

    rpm, n = 1200, 10  # 0.05s apart
    lim = _RateLimiter(rpm)
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: lim.acquire(), range(n)))
    elapsed = time.monotonic() - t0
    assert elapsed >= (n - 1) * (60.0 / rpm) * 0.9


def test_rate_limiter_disabled_by_non_positive_rpm():
    lim = _RateLimiter(0)
    t0 = time.monotonic()
    for _ in range(50):
        lim.acquire()
    assert time.monotonic() - t0 < 0.1


# ---------------------------------------------------------------------------
# Phase contract
# ---------------------------------------------------------------------------

def test_backend_is_deferred():
    """Deferred is what makes --judge_selected_checkpoint_only and
    --load_generations apply to this judge (both act on deferred evaluators)."""
    assert OpenAICompatibleBackend.phase == "deferred"
    judge = LLMJudge(OpenAICompatibleBackend("gpt-oss-120b", provider="vector"),
                     gen_params=_params())
    assert judge.phase == "deferred"
    assert judge.name == "gpt-oss-120b"
