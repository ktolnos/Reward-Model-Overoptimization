"""Deferred-phase metrics must reach the result rows / selection summary (L6).

``run_deferred_phase`` returns {checkpoint: metrics}; evaluate_policy folds
them into ``results_rows`` before the CSV write and ``report_selection``.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import SamplingParams

from policy_eval.eval_utils import run_deferred_phase
from policy_eval.selection import _judge_metric_keys
from policy_eval.types import Benchmark, GenerationConfig, GenerationResult


class StubDeferredJudgeEvaluator:
    """Minimal deferred evaluator emitting judge-style metric keys."""
    phase = "deferred"
    requires_logprobs = False
    name = "pairwise_llm_stub"

    def __init__(self):
        self.torn_down = False

    def teardown(self):
        self.torn_down = True

    def evaluate(self, benchmark, examples, generation, ctx):
        return {
            benchmark.metric_key(f"llm_stub/chosen/arena_score"):
                float(ctx.checkpoint_num),
        }


def _preference_benchmark(evaluator):
    return Benchmark(
        name="preference",
        load_examples=lambda args: [],
        format_prompt=lambda ex, tokenizer, thinking: "",
        generation_config=GenerationConfig(
            sampling_params=SamplingParams(max_tokens=1),
        ),
        evaluators=[evaluator],
        metric_prefix="",
    )


def _generation():
    return GenerationResult(
        responses=["r"], raw_responses=["r"], finish_reasons=["stop"],
    )


def test_run_deferred_phase_returns_per_checkpoint_metrics():
    evaluator = StubDeferredJudgeEvaluator()
    bench = _preference_benchmark(evaluator)
    cache = {("preference", 100): _generation(), ("preference", 200): _generation()}

    out = run_deferred_phase([bench], {"preference": []}, cache, args=None,
                             loaded_rms=None)

    assert out == {
        100: {"llm_stub/chosen/arena_score": 100.0},
        200: {"llm_stub/chosen/arena_score": 200.0},
    }
    assert evaluator.torn_down


def test_merged_judge_metrics_are_picked_up_by_selection_summary():
    # After the main-loop rows are updated with deferred metrics, the selection
    # report's judge-key detection must find them.
    row = {
        "checkpoint": 100,
        "select/sibling_rm/mean": 1.0,
        "llm_stub/chosen/arena_score": 100.0,
        "arena_hard/llm_stub/hard_prompt/sc_score": 55.0,
    }
    assert _judge_metric_keys(row) == [
        "arena_hard/llm_stub/hard_prompt/sc_score",
        "llm_stub/chosen/arena_score",
    ]
