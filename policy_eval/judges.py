"""Pairwise judges: compare two answers to a prompt and emit battle outcomes.

A judge decides which of (answer_a, answer_b) is better for a given prompt.
Two implementations:

    - ``RMJudge``: score each answer with a reward model; winner has the higher
      score. One battle per prompt.
    - ``LLMJudge``: ask an LLM -- locally via ``VLLMBackend``, or over any
      OpenAI-compatible API via ``OpenAICompatibleBackend`` (one backend for
      every hosted provider; ``OPENAI_PROVIDERS`` holds the per-provider
      endpoint, key env var, reasoning dialect and Batch-API support) -- using
      the Arena-Hard-Auto v2.0 prompt template. Runs 2 games per prompt with
      position swap; each game's label is mapped (with
      weighting) to one or more {0.0, 0.5, 1.0} battles, matching ``show_result.py``
      exactly so that scores produced here match the official leaderboard when the
      same judge/baseline are used.

Both return ``(battles_per_prompt, extras)`` where ``battles_per_prompt`` is a
list of lists: ``battles_per_prompt[i]`` = battle scores for prompt ``i`` from
answer_a's perspective (1.0 = A wins, 0.0 = A loses, 0.5 = tie). The
``PairwiseEvaluator`` aggregates these into Arena scores and bootstrap CIs.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import requests

from .rewards import score_responses_with_rm


@dataclass
class RMBattleDetails:
    """Per-prompt reward-model scores behind each RM-judge battle.

    ``policy_scores``/``baseline_scores`` are aligned to the prompts passed to
    ``score_pairs`` (answer_a = policy, answer_b = baseline)."""
    policy_scores: np.ndarray
    baseline_scores: np.ndarray


@dataclass
class LLMBattleDetails:
    """Per-prompt position-swapped game labels behind each LLM-judge battle.

    ``game0_labels`` is the A=baseline/B=policy game, ``game1_labels`` the
    A=policy/B=baseline game; entries are ``None`` when a game failed to parse.
    ``game0_texts``/``game1_texts`` hold the raw judge generations when the judge
    captures them (both backends do).

    Failure counts are over the ``2 * n_prompts`` games unless noted:
    ``n_generation_failures`` = empty judge output (API HTTP failure after retries,
    or an empty local generation); ``n_parse_failures`` = non-empty output with no
    parsable verdict; ``n_dropped_prompts`` = prompts excluded from metrics because
    at least one of their two games had no usable verdict."""
    game0_labels: List[Optional[str]]
    game1_labels: List[Optional[str]]
    game0_texts: Optional[List[str]] = None
    game1_texts: Optional[List[str]] = None
    n_generation_failures: int = 0
    n_truncation_failures: int = 0
    n_parse_failures: int = 0
    n_dropped_prompts: int = 0


@dataclass
class JudgeGeneration:
    """One judge completion: the raw ``text`` and whether it hit the token limit.

    ``truncated`` lets the judge attribute an unparsable verdict to truncation
    (the model ran out of tokens) rather than a genuine parse failure."""
    text: str
    truncated: bool = False


# ---------------------------------------------------------------------------
# Arena-Hard-Auto v2.0 default prompt/template/regex (from utils/judge_utils.py
# + config/arena-hard-v2.0.yaml in lmarena-ai/arena-hard-auto).
# ---------------------------------------------------------------------------

ARENA_HARD_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by "
    "two AI assistants to the user prompt displayed below. You will be given assistant A's "
    "answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your "
    "answers before judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. "
    "You must identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means "
    "the answer correctly responds to the prompt or follows the instructions. Note when user "
    "prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate "
    "to ask for clarifications or more information from the user than providing an answer based "
    "on assumptions. Relevant means all parts of the response closely connect or are appropriate "
    "to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
    "identify any missing important information in the assistants' answers that would be "
    "beneficial to include when responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your "
    "final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

# No-thinking variant. The canonical prompt above is a chain-of-thought judge: it
# instructs the model to generate its own answer and a written explanation before
# the verdict. That is right for the thinking path but incoherent when thinking is
# off -- there the verdict is prefilled, so none of that text is ever produced and
# the instructions contradict what the model is allowed to do. This variant instead
# asks the model to do all reasoning internally and emit only the verdict, and lists
# the label first so it lands immediately after the "My final verdict is [[" prefill.
ARENA_HARD_NO_THINKING_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by "
    "two AI assistants to the user prompt displayed below. You will be given assistant A's "
    "answer and assistant B's answer. Your job is to decide which assistant's answer is better.\n\n"
    "Think about what an ideal answer to the prompt would be and compare both assistants' answers "
    "against it, without writing any of this out. Identify any mistakes or inaccurate information. "
    "Consider whether each answer is helpful (correctly responds to the prompt or follows the "
    "instructions; note that when the prompt is ambiguous or has more than one interpretation it "
    "is more helpful to ask for clarification than to answer on an assumption), relevant, and "
    "concise, its creativity and novelty where needed, and whether it omits any important "
    "information.\n\n"
    "Respond with only your final verdict, stated as one of the following labels:\n\n"
    "1. [[A>>B]], Assistant A is significantly better\n"
    "2. [[A>B]], Assistant A is slightly better\n"
    "3. [[A=B]], a tie (relatively the same)\n"
    "4. [[B>A]], Assistant B is slightly better\n"
    "5. [[B>>A]], Assistant B is significantly better\n\n"
    'Example output: "My final verdict is [[A=B]], a tie".'
)

ARENA_HARD_PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n"
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
)

ARENA_HARD_REGEX_PATTERNS = [
    r"\[\[([AB<>=]+)\]\]",
    r"\[([AB<>=]+)\]",
]


def render_judge_question(messages: List[dict]) -> str:
    """Render a prompt conversation for the judge's ``<|User Prompt|>`` slot.

    Single-turn prompts pass through verbatim, matching upstream Arena-Hard
    byte-for-byte. Multi-turn conversations become a role-labelled transcript,
    so the judge sees the same full dialog the reward models score with
    (RM judges re-template the complete conversation; without this the LLM
    judge saw only the final user turn).
    """
    if len(messages) == 1:
        return messages[0]["content"]
    return "\n\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)

# label_to_score from show_result.py. ``weight`` multiplies the strong labels,
# expanding each prompt into multiple battles. With weight=3 the final
# per-prompt battle count is 2 games × (weight or 1 depending on label
# strength). Bootstrap is over this expanded battle list.
def _label_to_score(weight: int = 3) -> Dict[str, List[float]]:
    return {
        "A>B":  [1.0],
        "A>>B": [1.0] * weight,
        "A=B":  [0.5],
        "A<<B": [0.0] * weight,
        "A<B":  [0.0],
        "B>A":  [0.0],
        "B>>A": [0.0] * weight,
        "B=A":  [0.5],
        "B<<A": [1.0] * weight,
        "B<A":  [1.0],
    }


def battles_from_game_labels(
    game0_labels: List[Optional[str]],
    game1_labels: List[Optional[str]],
    *,
    weight: int = 3,
) -> Tuple[List[List[float]], int]:
    """Apply the Arena-Hard ``show_result.py`` formula to two swapped games.

    ``game0`` is the A=baseline/B=policy game, ``game1`` the A=policy/B=baseline
    game. Returns ``(battles_per_prompt, dropped)`` where each per-prompt entry is
    a list of battle scores in {0.0, 0.5, 1.0} from the policy's perspective
    (decisive verdicts expand to ``weight`` copies); an empty list marks a prompt
    where either game failed to parse. Mirrors:

        scores = label_to_score[game1] + [1 - s for s in label_to_score[game0]]
    """
    label_score = _label_to_score(weight)
    battles: List[List[float]] = []
    dropped = 0
    for lbl0, lbl1 in zip(game0_labels, game1_labels):
        if (lbl0 is None or lbl1 is None
                or lbl0 not in label_score or lbl1 not in label_score):
            dropped += 1
            battles.append([])
            continue
        ms1 = label_score[lbl1]
        ms0 = [1.0 - s for s in label_score[lbl0]]
        battles.append(ms1 + ms0)
    return battles, dropped


def positional_bias_metrics(details: "LLMBattleDetails") -> Dict[str, float]:
    """Controversial (position-flipped) rate from the two swapped games.

    A prompt is *controversial* when the judge names opposite decisive winners in
    the two position-swapped games (game0 = A:baseline/B:policy, game1 =
    A:policy/B:baseline): it picked the same answer *position* both times, so the
    pairwise decision reflects answer order rather than answer content. This is the
    positional-bias signature — with no bias the judge names the same actual answer
    regardless of order. ``prefers_first``/``prefers_second`` split the flips by
    which position the judge favored (A = first, B = second).

    Rate is over prompts both of whose games parsed (same denominator as
    ``win_rate``). Ties in either game are not counted as a flip.
    """
    label_score = _label_to_score(weight=1)  # single value per label: direction only
    n_judged = n_controversial = n_prefers_first = n_prefers_second = 0
    for lbl0, lbl1 in zip(details.game0_labels, details.game1_labels):
        if lbl0 not in label_score or lbl1 not in label_score:
            continue
        n_judged += 1
        # Policy-perspective winner per game: +1 policy, 0 tie, -1 baseline.
        s1 = label_score[lbl1][0]
        s0 = 1.0 - label_score[lbl0][0]
        w1 = (s1 > 0.5) - (s1 < 0.5)
        w0 = (s0 > 0.5) - (s0 < 0.5)
        if w0 != 0 and w1 != 0 and w0 != w1:
            n_controversial += 1
            # Opposite winners => same position won both games (w1 > 0 <=> the
            # judge picked A/first in both, w1 < 0 <=> B/second in both).
            if w1 > 0:
                n_prefers_first += 1
            else:
                n_prefers_second += 1
    return {
        "n_controversial": n_controversial,
        "controversial_rate": n_controversial / n_judged if n_judged else 0.0,
        "n_prefers_first": n_prefers_first,
        "n_prefers_second": n_prefers_second,
    }


# ---------------------------------------------------------------------------
# RM judge
# ---------------------------------------------------------------------------

class RMJudge:
    """Score both answers with a reward model; higher score wins.

    Produces exactly one battle per prompt.
    """
    kind = "rm"

    def __init__(self, rm_label: str):
        self.rm_label = rm_label
        self.name = f"rm_{rm_label}"

    def score_pairs(
        self,
        prompts_messages: List[list],
        answers_a: List[str],
        answers_b: List[str],
        ctx,
        *,
        baseline_cache_key: Optional[str] = None,
    ) -> Tuple[List[List[float]], RMBattleDetails]:
        rms = ctx.loaded_rms
        entry = rms.get(self.rm_label) if rms else None
        if entry is None:
            raise ValueError(f"RMJudge needs RM '{self.rm_label}' loaded")
        model, tokenizer = entry

        # Always re-score the policy side: its responses change every checkpoint.
        scores_a = score_responses_with_rm(
            answers_a, prompts_messages, model, tokenizer,
            batch_size=ctx.args.batch_size, device=ctx.args.device,
            checkpoint_num=ctx.checkpoint_num,
        )
        # Baseline side: cached on disk when caller supplies a key (the cache
        # is keyed only on the baseline content + RM, not on the policy).
        if baseline_cache_key is not None:
            scores_b = rms.score_with_cache(
                self.rm_label, answers_b, prompts_messages,
                cache_namespace="baseline_score_cache",
                cache_key=baseline_cache_key,
                log_label=f"Baseline:{self.rm_label}",
            )
        else:
            scores_b = score_responses_with_rm(
                answers_b, prompts_messages, model, tokenizer,
                batch_size=ctx.args.batch_size, device=ctx.args.device,
                checkpoint_num=f"{ctx.checkpoint_num}:pair_b",
            )
        battles = []
        for sa, sb in zip(scores_a, scores_b):
            if sa > sb:
                battles.append([1.0])
            elif sa == sb:
                battles.append([0.5])
            else:
                battles.append([0.0])
        return battles, RMBattleDetails(policy_scores=scores_a, baseline_scores=scores_b)


# ---------------------------------------------------------------------------
# Generative LLM judge (Arena-Hard-Auto v2.0 compatible)
#
# One judge, pluggable generation backend. ``LLMJudge`` owns everything that is
# backend-independent (prompt construction, the two position-swapped games, label
# parsing, weighted battle scoring). A backend only turns chat conversations +
# shared ``JudgeGenParams`` into completion strings, so switching between a remote
# API and a local vLLM model changes only the backend — not the protocol, the
# metrics, or how generation params are set.
# ---------------------------------------------------------------------------

def _parse_arena_label(text: str, patterns: List[str]) -> Optional[str]:
    """Find the last [[...]] label in ``text``. Matches Arena-Hard's get_score."""
    up = text.upper()
    for pat in patterns:
        matches = re.findall(pat, up)
        matches = [m for m in matches if m]
        if matches:
            return matches[-1].strip("\n")
    return None


@dataclass
class JudgeGenParams:
    """Generation controls applied identically across judge backends.

    Fields are required (no defaults) on purpose: the canonical default values
    live in ``ScriptArguments`` (the ``--llm_judge_*`` flags), and callers build
    this from there, so a default can't drift between config and code.

    ``enable_thinking`` is honored in full by the local vLLM backend (chat
    template + ``"My final verdict is [["`` prefill); ``OpenAICompatibleBackend``
    maps it to whatever reasoning dialect its provider speaks.
    """
    temperature: float
    top_p: float
    max_tokens: int
    enable_thinking: bool


class JudgeAccessError(RuntimeError):
    """A judge API rejected the credentials or the model grant (HTTP 401/403)."""


@dataclass(frozen=True)
class OpenAIProvider:
    """The only four things that differ between OpenAI-compatible endpoints."""
    name: str
    base_url: str
    api_key_env: str
    # How reasoning is requested or suppressed; see ``_reasoning_fields``:
    #   "toggle" - OpenRouter's native {"reasoning": {"enabled"|"effort": ...}}
    #   "model"  - derived from the model: gpt-oss (harmony) takes
    #              reasoning_effort, vLLM-served models take
    #              chat_template_kwargs.enable_thinking
    reasoning_style: str
    # Whether the provider implements the async OpenAI Batch API
    # (/v1/files + /v1/batches). OpenRouter does not.
    supports_batch: bool = False
    # Default client-side pacing. 0 = unpaced; set only where the provider
    # actually enforces a request-per-minute budget.
    default_rpm: float = 0.0


OPENAI_PROVIDERS: Dict[str, OpenAIProvider] = {
    "openrouter": OpenAIProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        reasoning_style="toggle",
    ),
    "vector": OpenAIProvider(
        name="vector",
        base_url="https://proxy.vectorinstitute.ai/v1",
        api_key_env="VECTOR_INFERENCE_API_KEY",
        # The proxy fronts on-prem vLLM servers and hosted models alike, so the
        # reasoning control depends on the model rather than the endpoint.
        reasoning_style="model",
        supports_batch=True,
        # The proxy enforces a project-wide RPM budget that is SHARED with
        # everyone else on the project; 100 leaves headroom under the observed
        # 120 cap.
        default_rpm=100.0,
    ),
}


class _RateLimiter:
    """Thread-safe pacer keeping request starts under a per-minute cap.

    Pacing beats discovering a published RPM ceiling through 429s: a rejected
    request still costs a round-trip and its retry lands in the same saturated
    window. ``requests_per_minute<=0`` disables pacing.
    """

    def __init__(self, requests_per_minute: Optional[float]):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            # Reserve this slot before releasing the lock so concurrent workers
            # queue up behind each other rather than all sleeping to the same
            # instant and then firing as one burst.
            self._next_at = max(now, self._next_at) + self.min_interval
        if wait:
            time.sleep(wait)


class OpenAICompatibleBackend:
    """Generate judge completions from any OpenAI ``/chat/completions`` endpoint.

    The endpoint is a ``provider`` entry (``OPENAI_PROVIDERS``); everything else
    -- fan-out in request order, pacing, retry/backoff, verdict extraction -- is
    shared, so a fix reaches every provider at once.

    Non-obvious behaviour:

      - Reasoning is controlled per provider (``_reasoning_fields``). A server
        that rejects those extras with HTTP 400 is retried once without them:
        losing the thinking toggle beats losing the run.
      - Verdict text falls back to the reasoning channel when ``content`` is
        empty, so a model that stops inside its reasoning still parses.
      - 401/403 raises ``JudgeAccessError`` instead of retrying -- it is identical
        for every request, so retrying only delays a doomed run and would report a
        bad key as "every prompt dropped".

    Unlike ``VLLMBackend`` the no-thinking path cannot prefill the assistant turn,
    so the model emits the whole ``"My final verdict is [[X]]"`` line itself --
    which is what ``ARENA_HARD_NO_THINKING_SYSTEM_PROMPT`` asks for.

    ``phase`` is "deferred" (= after all generation) despite needing no GPU:
    ``--judge_selected_checkpoint_only`` trims the *deferred* cache and
    ``--load_generations`` runs *only* deferred evaluators, so an online judge
    would silently judge every checkpoint instead of the selected one.
    """
    phase = "deferred"

    def __init__(
        self,
        model_name: str,
        *,
        provider: "str | OpenAIProvider" = "openrouter",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_parallel: int = 8,
        max_retries: int = 5,
        timeout: int = 600,
        requests_per_minute: Optional[float] = None,
        reasoning_effort: str = "auto",
        use_batch_api: bool = False,
        batch_poll_seconds: float = 30.0,
        batch_completion_window: str = "24h",
    ):
        if isinstance(provider, str):
            if provider not in OPENAI_PROVIDERS:
                raise ValueError(
                    f"Unknown provider {provider!r}. Known: {sorted(OPENAI_PROVIDERS)}."
                )
            provider = OPENAI_PROVIDERS[provider]
        self.provider = provider
        self.model_name = model_name
        self.label = model_name.replace("/", "_")
        self._api_key = api_key
        self.api_key_env = provider.api_key_env
        self.base_url = (base_url or provider.base_url).rstrip("/")
        self.max_parallel = max_parallel
        self.max_retries = max_retries
        self.timeout = timeout
        # None = the provider's own default; 0 = explicitly unpaced.
        self.requests_per_minute = (
            provider.default_rpm if requests_per_minute is None else requests_per_minute
        )
        # "auto" derives the effort from JudgeGenParams.enable_thinking; an
        # explicit low/medium/high pins it regardless of the thinking flag.
        self.reasoning_effort = reasoning_effort
        if use_batch_api and not provider.supports_batch:
            raise ValueError(
                f"Provider {provider.name!r} has no Batch API; drop --llm_judge_use_batch_api "
                f"or use a provider that supports it "
                f"({sorted(p for p, v in OPENAI_PROVIDERS.items() if v.supports_batch)})."
            )
        self.use_batch_api = use_batch_api
        self.batch_poll_seconds = batch_poll_seconds
        self.batch_completion_window = batch_completion_window
        self._limiter = _RateLimiter(self.requests_per_minute)
        # Set once a request proves the server rejects our reasoning/thinking
        # extras, so the remaining ~2N requests skip the doomed first attempt.
        self._strip_extras = False

    # -- request construction ------------------------------------------------

    def _resolve_key(self) -> str:
        key = self._api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"The {self.provider.name} judge needs an API key: pass "
                f"--llm_judge_api_key or set {self.api_key_env}."
            )
        return key

    def _headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _is_harmony(self) -> bool:
        """gpt-oss models always reason; only the effort dial exists."""
        return "gpt-oss" in self.model_name.lower()

    def _reasoning_fields(self, params: "JudgeGenParams") -> dict:
        """Provider-specific fields controlling the judge's reasoning."""
        style = self.provider.reasoning_style
        if self._strip_extras:
            return {}
        effort = self.reasoning_effort

        if style == "toggle":
            if effort not in ("auto", "none"):
                return {"reasoning": {"effort": effort}}
            # Only deviate from the model's default reasoning behaviour when
            # thinking is explicitly disabled (a no-op for models without it).
            return {} if params.enable_thinking else {"reasoning": {"enabled": False}}

        if style == "model":
            if self._is_harmony():
                if effort == "auto":
                    effort = "high" if params.enable_thinking else "low"
                return {} if effort == "none" else {"reasoning_effort": effort}
            fields: dict = {
                "chat_template_kwargs": {"enable_thinking": params.enable_thinking}
            }
            if effort not in ("auto", "none"):
                fields["reasoning_effort"] = effort
            return fields

        raise ValueError(f"Unknown reasoning_style {style!r} for {self.provider.name}.")

    def _body(self, messages: List[dict], params: "JudgeGenParams") -> dict:
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        body.update(self._reasoning_fields(params))
        return body

    @staticmethod
    def _text_from_choice(choice: dict) -> str:
        """Verdict text, falling back to the reasoning channel when empty.

        Reasoning models return the analysis separately (``reasoning_content`` on
        vLLM, ``reasoning`` on OpenRouter); a model that stops inside it leaves
        ``content`` empty, and the verdict may still be in there.
        """
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if not text.strip():
            text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return text

    # -- live (non-batch) path -----------------------------------------------

    def _sleep_for_retry(self, resp, attempt: int) -> None:
        """Back off before a retry, honouring a server-provided ``Retry-After``."""
        delay = min(60.0, 2 ** attempt)
        if resp is not None:
            hdr = resp.headers.get("Retry-After")
            if hdr:
                try:
                    delay = max(delay, float(hdr))
                except ValueError:
                    pass
        time.sleep(delay)

    def _post_chat(
        self, messages: List[dict], params: "JudgeGenParams", api_key: str,
    ) -> "JudgeGeneration":
        last_err = None
        for attempt in range(self.max_retries):
            resp = None
            try:
                self._limiter.acquire()
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(api_key),
                    json=self._body(messages, params),
                    timeout=self.timeout,
                )
                if resp.status_code in (401, 403):
                    raise JudgeAccessError(
                        f"The {self.provider.name} endpoint refused the request for "
                        f"model {self.model_name!r} (HTTP {resp.status_code}): "
                        f"{resp.text[:300]}. Check that {self.api_key_env} holds a valid "
                        f"key and that it is entitled to this model (on the Vector "
                        f"proxy, models are granted per project at "
                        f"inference.vectorinstitute.ai)."
                    )
                if resp.status_code == 400 and not self._strip_extras and \
                        self._reasoning_fields(params):
                    # Most likely the server rejected the reasoning fields. Drop
                    # them for this and every later request, then retry at once --
                    # losing the thinking toggle beats losing the run.
                    print(
                        f"[{self.provider.name}] HTTP 400 from {self.model_name}; "
                        f"retrying without the reasoning/thinking fields. "
                        f"Body: {resp.text[:300]}"
                    )
                    self._strip_extras = True
                    continue
                if resp.status_code == 429:
                    last_err = f"429 {resp.text[:200]}"
                    self._sleep_for_retry(resp, attempt)
                    continue
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                return JudgeGeneration(
                    text=self._text_from_choice(choice),
                    truncated=choice.get("finish_reason") == "length",
                )
            except JudgeAccessError:
                raise
            except Exception as e:
                last_err = e
                self._sleep_for_retry(resp, attempt)
        print(f"[{self.provider.name}] giving up on a game after "
              f"{self.max_retries} retries: {last_err}")
        return JudgeGeneration(text="", truncated=False)

    def generate(
        self, conversations: List[List[dict]], params: "JudgeGenParams",
    ) -> List["JudgeGeneration"]:
        api_key = self._resolve_key()
        if self.use_batch_api:
            return self._generate_batch(conversations, params, api_key)

        import concurrent.futures as cf

        started = time.monotonic()
        gens: List[Optional[JudgeGeneration]] = [None] * len(conversations)
        with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futs = {
                pool.submit(self._post_chat, conv, params, api_key): i
                for i, conv in enumerate(conversations)
            }
            for fut in cf.as_completed(futs):
                gens[futs[fut]] = fut.result()
        elapsed = time.monotonic() - started
        n = max(1, len(conversations))
        print(
            f"[{self.provider.name}] {len(conversations)} requests to "
            f"{self.model_name} in {elapsed:.1f}s ({elapsed / n:.2f}s/request "
            f"wall-clock, max_parallel={self.max_parallel}, "
            f"rpm_cap={self.requests_per_minute})"
        )
        return gens

    # -- batch path ----------------------------------------------------------

    def _generate_batch(
        self, conversations: List[List[dict]], params: "JudgeGenParams", api_key: str,
    ) -> List["JudgeGeneration"]:
        """Upload one JSONL request per game, poll, demux results by ``custom_id``.

        Trades latency for throughput: the batch runs against a separate quota
        rather than the live RPM budget. Dropped requests come back as empty
        generations, which ``score_pairs`` counts as generation failures.
        """
        auth = {"Authorization": f"Bearer {api_key}"}
        tag = f"[{self.provider.name}:batch]"

        # 1. Upload the request set as a JSONL file.
        lines = [
            json.dumps({
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self._body(conv, params),
            })
            for i, conv in enumerate(conversations)
        ]
        payload = ("\n".join(lines) + "\n").encode()
        print(f"{tag} uploading {len(lines)} requests ({len(payload) / 1e6:.1f} MB)")
        up = requests.post(
            f"{self.base_url}/files",
            headers=auth,
            files={"file": ("judge_batch.jsonl", io.BytesIO(payload), "application/jsonl")},
            data={"purpose": "batch"},
            timeout=self.timeout,
        )
        if not up.ok:
            # A provider can advertise the Batch API (GET /v1/files and
            # /v1/batches answer) while its upstream implements no file
            # endpoint, so the surface looks present until the first upload.
            # Measured on the Vector proxy: 502 "Upstream file upload failed:
            # 404". Say so plainly -- the raw HTTPError lands after the whole
            # generation phase and reads like a transient blip.
            raise RuntimeError(
                f"{self.provider.name} rejected the batch file upload "
                f"(HTTP {up.status_code}: {up.text[:200]}). Its Batch API is not "
                f"usable, whatever GET /v1/batches reports. Drop "
                f"--llm_judge_use_batch_api and use the live path; cached "
                f"generations can be re-judged with --llm_judge_on_cached instead "
                f"of regenerating."
            )
        input_file_id = up.json()["id"]

        # 2. Submit the batch.
        sub = requests.post(
            f"{self.base_url}/batches",
            headers=self._headers(api_key),
            json={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": self.batch_completion_window,
            },
            timeout=self.timeout,
        )
        sub.raise_for_status()
        batch = sub.json()
        batch_id = batch["id"]
        print(f"{tag} {batch_id} submitted (window={self.batch_completion_window})")

        # 3. Poll until it settles.
        started = time.monotonic()
        terminal = {"completed", "failed", "expired", "cancelled"}
        while batch.get("status") not in terminal:
            time.sleep(self.batch_poll_seconds)
            st = requests.get(
                f"{self.base_url}/batches/{batch_id}",
                headers=self._headers(api_key), timeout=self.timeout,
            )
            st.raise_for_status()
            batch = st.json()
            print(f"{tag} {batch_id} status={batch.get('status')} "
                  f"counts={batch.get('request_counts') or {}} "
                  f"elapsed={time.monotonic() - started:.0f}s", flush=True)

        gens = [JudgeGeneration(text="", truncated=False) for _ in conversations]
        if batch.get("status") != "completed":
            print(f"{tag} {batch_id} ended as {batch.get('status')}: "
                  f"{str(batch.get('errors'))[:500]}")
            return gens

        # 4. Download and demux results by custom_id.
        out_id = batch.get("output_file_id")
        if not out_id:
            print(f"{tag} {batch_id} completed with no output file")
            return gens
        content = requests.get(
            f"{self.base_url}/files/{out_id}/content", headers=auth, timeout=self.timeout,
        )
        content.raise_for_status()
        n_ok = 0
        for line in content.text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                choice = rec["response"]["body"]["choices"][0]
                gens[int(rec["custom_id"])] = JudgeGeneration(
                    text=self._text_from_choice(choice),
                    truncated=choice.get("finish_reason") == "length",
                )
                n_ok += 1
            except Exception as e:
                print(f"{tag} unparsable batch result line: {e!r}")
        print(f"{tag} {batch_id} returned {n_ok}/{len(conversations)} results "
              f"in {time.monotonic() - started:.0f}s")
        return gens


class VLLMBackend:
    """Generate judge completions from a local open-weight model via vLLM.

    Deferred: the model loads lazily on the first ``generate`` (after the policy
    vLLM is torn down so the GPU is free) and is reused across checkpoints, then
    released by ``teardown``. Applies the chat template locally so it can honor
    ``enable_thinking`` and prefill ``"My final verdict is [["`` when thinking is off.
    """
    phase = "deferred"
    # With thinking off the assistant turn is prefilled up to the opening of the
    # verdict token. The system prompt states the label first ("[[A=B]], a tie"),
    # so from this prefill the model emits the label and its closing brackets
    # immediately -- e.g. "A>>B]]" -- and nothing else is needed to parse a verdict.
    # This matches the prompt's example output verbatim.
    PREFILL = "My final verdict is [["
    # The label completion ("A>>B]]", the longest of the five) is exactly 4 tokens
    # across the Llama-3.2, Gemma-3, Gemma-4, Qwen3 and Qwen3.5 tokenizers (measured),
    # so 4 fits every verdict. No EOS headroom is needed: a label that fills the
    # budget with no room for EOS, or is followed by the descriptor text, hits
    # finish_reason=="length" yet still parses -- and LLMJudge.score_pairs only counts
    # truncation as a failure when NO label parses, so that is never treated as
    # truncation. (Only applies to the prefilled path; the thinking path keeps the
    # full --llm_judge_max_new_tokens budget.)
    PREFILL_MAX_TOKENS = 4

    def __init__(
        self,
        model_name: str,
        *,
        max_model_len: int,
        gpu_memory_utilization: float,
        quantization: Optional[str] = None,
    ):
        self.model_name = model_name
        # Reflect the quant method in the label so metrics/artifacts distinguish
        # e.g. the fp8 judge from the bf16 one when both run in a sweep.
        self.label = model_name.replace("/", "_")
        if quantization:
            self.label = f"{self.label}_{quantization}"
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        # None -> load the checkpoint's native dtype (bf16). "fp8" -> weight-only
        # fp8 quantized in-flight from the bf16 weights (Marlin kernels on Ampere;
        # ~half the VRAM). Any vLLM-recognized quant string is forwarded verbatim.
        self.quantization = quantization
        self._llm = None
        self._tokenizer = None
        self._stop_token_ids = None

    def _ensure(self) -> None:
        if self._llm is not None:
            return
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        from .generation import wait_for_gpu_memory
        from data_utils import get_generation_stop_token_ids

        print(
            f"[VLLMBackend] loading judge {self.model_name} "
            f"(max_model_len={self.max_model_len}, "
            f"gpu_memory_utilization={self.gpu_memory_utilization})"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        # Resolve the stop set once, here: vLLM does not fold the model's
        # generation_config eos list into its stops by default, so a judge whose
        # turn-end token lives only there (gemma-4's <turn|>) would never stop and
        # would burn the full max_tokens on every game. The shared helper covers
        # both that list and the known turn-end token strings.
        self._stop_token_ids = get_generation_stop_token_ids(
            self._tokenizer, model_name_or_path=self.model_name,
        )
        print(
            f"[VLLMBackend] judge stop_token_ids={self._stop_token_ids} "
            f"({[self._tokenizer.decode([i]) for i in self._stop_token_ids]})"
        )
        # The just-torn-down policy engine frees its GPU memory asynchronously
        # (its EngineCore process exits after teardown_vllm returns); wait for that
        # so vLLM's startup free-memory check doesn't race the reclaim and fail.
        wait_for_gpu_memory(self.gpu_memory_utilization)
        self._llm = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            # vLLM's sampler warmup builds a [max_num_seqs, vocab] logits tensor,
            # and large-vocab judges (gemma: 262k) with logit soft-capping need a
            # second copy of it -- ~256 MiB per 256 seqs. With a big judge model
            # filling the GPU, that transient can be what tips warmup into OOM.
            # A large judge is KV-bound to only a few concurrent full-length
            # sequences regardless, so capping concurrency costs no real
            # throughput while keeping the warmup transient small.
            max_num_seqs=64,
            trust_remote_code=True,
            language_model_only=True,
            quantization=self.quantization,
        )

    def teardown(self) -> None:
        from .generation import teardown_vllm
        teardown_vllm(self._llm)
        self._llm = None
        self._tokenizer = None

    def _render(self, messages: List[dict], enable_thinking: bool) -> str:
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Chat template doesn't accept the enable_thinking kwarg.
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        if not enable_thinking:
            text += self.PREFILL
        return text

    def generate(self, conversations: List[List[dict]], params: "JudgeGenParams") -> List["JudgeGeneration"]:
        from vllm import SamplingParams

        self._ensure()
        prompts = [self._render(c, params.enable_thinking) for c in conversations]
        # The prefill (see PREFILL_MAX_TOKENS) only exists on the no-thinking path,
        # where the model just completes the verdict tail; cap it there so it
        # commits immediately instead of running out the full budget. The thinking
        # path needs the reasoning room, so it keeps params.max_tokens.
        max_tokens = params.max_tokens if params.enable_thinking else self.PREFILL_MAX_TOKENS
        sampling = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=max_tokens,
            n=1,
            stop_token_ids=self._stop_token_ids,
        )
        outputs = self._llm.generate(prompts, sampling)
        gens = []
        for o in outputs:
            comp = o.outputs[0]
            text = comp.text
            # Restore the prefill so saved text and the parser see the full verdict.
            if not params.enable_thinking:
                text = self.PREFILL + text
            gens.append(JudgeGeneration(text=text, truncated=comp.finish_reason == "length"))
        return gens


class LLMJudge:
    """Arena-Hard-Auto-compatible pairwise judge with a pluggable generation backend.

    Backend-independent: builds the system+user judge prompt, runs the two
    position-swapped games per prompt, parses the 5-point verdict, and maps the
    labels to weighted battles via ``battles_from_game_labels`` (matching
    ``show_result.py``). The ``backend`` (``OpenAICompatibleBackend`` or
    ``VLLMBackend``) only turns chat conversations + ``JudgeGenParams`` into
    completion strings, so switching API <-> local vLLM leaves the protocol,
    metrics, and generation-param handling unchanged. ``phase`` is inherited from
    the backend so deferred (GPU-loading) backends run after the policy vLLM.
    """
    kind = "llm"

    def __init__(
        self,
        backend,
        *,
        gen_params: JudgeGenParams,
        system_prompt: Optional[str] = None,
        prompt_template: str = ARENA_HARD_PROMPT_TEMPLATE,
        regex_patterns: Optional[List[str]] = None,
        weight: int = 3,
    ):
        self.backend = backend
        # The prompt is tied to the reasoning mode: the canonical chain-of-thought
        # Arena-Hard prompt for the thinking path, the direct-verdict variant for the
        # no-thinking path (see ARENA_HARD_NO_THINKING_SYSTEM_PROMPT). An explicit
        # system_prompt overrides this selection.
        if system_prompt is None:
            system_prompt = (
                ARENA_HARD_SYSTEM_PROMPT if gen_params.enable_thinking
                else ARENA_HARD_NO_THINKING_SYSTEM_PROMPT
            )
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.regex_patterns = regex_patterns or ARENA_HARD_REGEX_PATTERNS
        self.weight = weight
        self.gen_params = gen_params
        self.phase = getattr(backend, "phase", "online")
        self.name = backend.label

    def teardown(self) -> None:
        if hasattr(self.backend, "teardown"):
            self.backend.teardown()

    def _conversation(self, question: str, answer_a: str, answer_b: str) -> List[dict]:
        user_prompt = self.prompt_template.format(
            QUESTION=question, ANSWER_A=answer_a, ANSWER_B=answer_b,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def score_pairs(
        self,
        prompts_messages: List[list],
        answers_a: List[str],
        answers_b: List[str],
        ctx,
    ) -> Tuple[List[List[float]], LLMBattleDetails]:
        n = len(answers_a)
        questions = [render_judge_question(m) for m in prompts_messages]

        # Two position-swapped games per prompt, flattened [p0/g0, p0/g1, ...].
        # game0 = A:baseline/B:policy, game1 = A:policy/B:baseline (upstream swap).
        conversations: List[List[dict]] = []
        for i in range(n):
            conversations.append(self._conversation(questions[i], answers_b[i], answers_a[i]))
            conversations.append(self._conversation(questions[i], answers_a[i], answers_b[i]))

        gens = self.backend.generate(conversations, self.gen_params)
        game0_gens = [gens[2 * i] for i in range(n)]
        game1_gens = [gens[2 * i + 1] for i in range(n)]
        game0_texts = [g.text for g in game0_gens]
        game1_texts = [g.text for g in game1_gens]
        game0_labels = [_parse_arena_label(t, self.regex_patterns) for t in game0_texts]
        game1_labels = [_parse_arena_label(t, self.regex_patterns) for t in game1_texts]

        battles, dropped = battles_from_game_labels(
            game0_labels, game1_labels, weight=self.weight,
        )

        # Attribute each unparsable game (label is None) to a single cause, in
        # priority order, so the reason for any drops is visible in wandb:
        #   empty output -> generation failure (API HTTP failure / empty gen)
        #   truncated    -> truncation failure (ran out of tokens before verdict)
        #   otherwise    -> parse failure (had output, no recognizable verdict)
        all_gens = game0_gens + game1_gens
        all_labels = game0_labels + game1_labels
        n_generation_failures = n_truncation_failures = n_parse_failures = 0
        for g, lbl in zip(all_gens, all_labels):
            if lbl is not None:
                continue
            if g.text == "":
                n_generation_failures += 1
            elif g.truncated:
                n_truncation_failures += 1
            else:
                n_parse_failures += 1
        if dropped:
            print(
                f"[LLMJudge:{self.name}] dropped {dropped}/{n} prompts "
                f"({n_generation_failures} generation, {n_truncation_failures} truncation, "
                f"{n_parse_failures} parse failures over {2 * n} games)"
            )
        return battles, LLMBattleDetails(
            game0_labels=game0_labels, game1_labels=game1_labels,
            game0_texts=game0_texts, game1_texts=game1_texts,
            n_generation_failures=n_generation_failures,
            n_truncation_failures=n_truncation_failures,
            n_parse_failures=n_parse_failures,
            n_dropped_prompts=dropped,
        )


