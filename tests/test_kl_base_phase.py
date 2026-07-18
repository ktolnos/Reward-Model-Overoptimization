"""Tests for the two-phase KLEvaluator: online policy pass + one-swap base pass.

Contract: ``evaluate`` stashes per-checkpoint state and returns no metrics;
``run_kl_base_phase`` swaps the base weights in exactly once, scores every
stashed checkpoint, and returns ``{checkpoint_num: metrics}``.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import policy_eval.evaluators as evaluators_mod
from policy_eval.eval_utils import run_kl_base_phase
from policy_eval.evaluators import KLEvaluator
from policy_eval.types import EvalContext, Example, GenerationResult


class _Recorder:
    def __init__(self):
        self.columns = {}

    def add_response_column(self, name, values):
        self.columns[name] = list(values)


def _generation(token_ids):
    return GenerationResult(
        responses=["r"] * len(token_ids),
        raw_responses=["r"] * len(token_ids),
        finish_reasons=["stop"] * len(token_ids),
        n_responses_per_example=1,
        full_ids_list=[list(ids) for ids in token_ids],
        prompt_lens_list=[1] * len(token_ids),
    )


def _examples(n):
    return [
        Example(prompt_messages=[{"role": "user", "content": f"q{i}"}], metadata={})
        for i in range(n)
    ]


def _patch_engine(monkeypatch, logprobs_by_call, swap_log):
    """Stub the vLLM-touching pieces; logprobs_by_call is popped per TF pass."""
    def fake_tf(llm, full_ids_list, prompt_lens_list):
        token_lp = logprobs_by_call.pop(0)
        mean_lp = [sum(t) / len(t) if t else 0.0 for t in token_lp]
        return mean_lp, token_lp

    monkeypatch.setattr(evaluators_mod, "teacher_forced_response_logprobs", fake_tf)
    monkeypatch.setattr(
        evaluators_mod, "update_vllm_weights",
        lambda llm, path: swap_log.append(path),
    )


def test_base_pass_swaps_once_and_returns_per_checkpoint_metrics(
    monkeypatch, tmp_path,
):
    from evaluate_policy import ScriptArguments

    args = ScriptArguments(
        checkpoints_dir=str(tmp_path),
        per_example_dir=str(tmp_path / "per_example"),
        per_example_format="jsonl",
    )
    benchmark = SimpleNamespace(
        name="preference", metric_key=lambda k: k, evaluators=[],
    )
    ev = KLEvaluator("base/model")
    benchmark.evaluators = [ev]

    swap_log = []
    # 2 online policy passes (ckpt 10, 20) + 2 base passes, in that order.
    _patch_engine(monkeypatch, [
        [[-1.0, -1.0], [-2.0]],   # ckpt 10 policy
        [[-1.5, -0.5], [-2.5]],   # ckpt 20 policy
        [[-2.0, -2.0], [-3.0]],   # ckpt 10 base
        [[-1.5, -0.5], [-2.5]],   # ckpt 20 base (identical -> KL 0)
    ], swap_log)

    examples = _examples(2)
    for ckpt in (10, 20):
        ctx = EvalContext(
            args=args, checkpoint_num=ckpt, checkpoint_path=None, llm=None,
            policy_tokenizer=None, loaded_rms=None,
        )
        ctx.recorder = _Recorder()
        metrics = ev.evaluate(benchmark, examples, _generation([[5, 6, 7], [8, 9]]), ctx)
        # Online phase records policy logprobs but emits no metrics yet.
        assert metrics == {}
        assert "policy_mean_logprob" in ctx.recorder.columns

    result = run_kl_base_phase([benchmark], llm=None, args=args)

    # Exactly one weight swap, to the base model, for both checkpoints.
    assert swap_log == ["base/model"]
    assert sorted(result) == [10, 20]
    # ckpt 10: policy - base = +1.0 per token on both samples.
    assert result[10]["kl/mean"] == 1.0
    # ckpt 20: identical logprobs -> zero KL under both estimators.
    assert result[20]["kl/mean"] == 0.0
    assert result[20]["kl/grpo_mean"] == 0.0

    # Per-response KL persisted, one file per checkpoint, joinable by prompt_uid.
    for ckpt in (10, 20):
        path = tmp_path / "per_example" / f"preference__checkpoint-{ckpt}__kl.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        assert len(rows) == 2
        assert {"prompt_uid", "kl__k1", "kl__grpo", "base_mean_logprob"} <= set(rows[0])

    # The stash is consumed: a second base phase is a no-op (no extra swap).
    assert run_kl_base_phase([benchmark], llm=None, args=args) == {}
    assert swap_log == ["base/model"]


def test_evaluate_skips_when_generation_has_no_token_ids(monkeypatch):
    ev = KLEvaluator("base/model")
    gen = GenerationResult(
        responses=["r"], raw_responses=["r"], finish_reasons=["stop"],
        n_responses_per_example=1,
    )
    ctx = EvalContext(
        args=None, checkpoint_num=1, checkpoint_path=None, llm=None,
        policy_tokenizer=None, loaded_rms=None,
    )
    assert ev.evaluate(SimpleNamespace(name="b"), _examples(1), gen, ctx) == {}
    assert ev.run_base_phase(llm=None, benchmark=None, args=None) == {}
