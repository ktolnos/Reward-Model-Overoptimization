"""Orchestration helpers used by ``evaluate_policy.py``.

These are the glue bits that sit between the CLI entry point and the reusable
components in this package:

- ``list_checkpoints``: discover checkpoint-* subdirs (or single-checkpoint mode).
- ``rms_required_by``: scan benchmarks and return the set of RM labels to load.
- ``make_baseline_responses``: generate (or pluck from dataset) judge baselines.
- ``chosen_responses_as_generation``: wrap dataset chosen responses into a
  ``GenerationResult`` so the evaluator path is reused unchanged.
- ``run_chosen_only``: the ``--evaluate_chosen_responses`` code path.
- ``run_deferred_phase``: calls deferred evaluators after the policy vLLM is torn down.

Keeping these out of ``evaluate_policy.py`` makes the entry point easy to read
and these helpers testable in isolation.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_utils import get_generation_stop_token_ids

from . import wandb_utils
from .evaluators import RewardModelEvaluator
from .generation import teardown_vllm
from .types import Benchmark, EvalContext, Example, GenerationResult


def list_checkpoints(args) -> Tuple[List[str], Optional[str], str]:
    """Return (checkpoint_names, single_model_path, first_checkpoint_path).

    A ``checkpoints_dir`` that's itself a checkpoint (no ``checkpoint-*``
    subdirs) is treated as a single-checkpoint run.
    """
    if os.path.isdir(args.checkpoints_dir):
        names = sorted(
            [d for d in os.listdir(args.checkpoints_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
    else:
        names = []

    single_path = None
    if not names:
        basename = os.path.basename(args.checkpoints_dir.rstrip(os.sep))
        step = int(basename.split("-")[1]) if basename.startswith("checkpoint-") else 0
        single_path = args.checkpoints_dir
        names = [f"checkpoint-{step}"]

    first_path = single_path or os.path.join(args.checkpoints_dir, names[0])
    if args.debug:
        names = names[:1]
    return names, single_path, first_path


def rms_required_by(benchmarks: List[Benchmark]) -> set:
    """Scan benchmarks' evaluators for required RM labels."""
    labels = set()
    for b in benchmarks:
        for ev in b.evaluators:
            if isinstance(ev, RewardModelEvaluator):
                labels.add(ev.rm_label)
    return labels


def make_baseline_responses(
    args,
    preference_bench: Benchmark,
    examples: List[Example],
    tokenizer,
) -> Optional[List[str]]:
    """Generate (or pluck from dataset) baseline responses for the LLM judge."""
    if not args.evaluate_with_llm_judge:
        return None
    if args.use_dataset_response_as_baseline:
        print("[judge] using dataset 'chosen' as baseline")
        return [ex.metadata.get("chosen_response", "") for ex in examples]
    if not args.baseline_model_path:
        raise ValueError(
            "--baseline_model_path or --use_dataset_response_as_baseline required for LLM judge"
        )
    # Spin up a disposable vLLM for the baseline. This follows the original
    # script: it's wasteful per-invocation but avoids swapping weights on the
    # policy vLLM and then having to re-load the first checkpoint.
    from vllm import LLM, SamplingParams
    print(f"[judge] generating baseline responses from {args.baseline_model_path}")
    baseline_llm = LLM(
        model=args.baseline_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_length + args.max_new_tokens,
        trust_remote_code=True,
        language_model_only=True,
    )
    sampling = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, n=1,
        stop_token_ids=get_generation_stop_token_ids(tokenizer),
    )
    prompts = [preference_bench.format_prompt(ex, tokenizer, thinking=True) for ex in examples]
    outputs = baseline_llm.generate(prompts, sampling)
    responses = [o.outputs[0].text for o in outputs]
    teardown_vllm(baseline_llm)
    return responses


def chosen_responses_as_generation(examples: List[Example]) -> GenerationResult:
    """Build a GenerationResult from the dataset's chosen responses (no generation)."""
    responses = [ex.metadata.get("chosen_response", "") for ex in examples]
    return GenerationResult(
        responses=responses,
        raw_responses=responses,
        finish_reasons=["stop"] * len(responses),
        n_responses_per_example=1,
    )


def run_chosen_only(args, benchmarks, bench_examples, loaded_rms) -> None:
    """Score dataset chosen responses directly. No generation, single step=0."""
    preference_bench = next((b for b in benchmarks if b.name == "preference"), None)
    if preference_bench is None:
        raise ValueError("--evaluate_chosen_responses requires the 'preference' benchmark")

    examples = bench_examples["preference"]
    generation = chosen_responses_as_generation(examples)

    ctx = EvalContext(
        args=args, checkpoint_num=0, checkpoint_path=None,
        llm=None, policy_tokenizer=None, loaded_rms=loaded_rms,
        baseline_responses=None,
    )
    combined: Dict = {}
    for ev in preference_bench.online_evaluators:
        if ev.requires_logprobs:
            print(f"[chosen-only] skipping {ev.name} (needs logprobs)")
            continue
        if ev.name.startswith("llm_judge"):
            print(f"[chosen-only] skipping {ev.name} (judge needs baseline)")
            continue
        combined.update(ev.evaluate(preference_bench, examples, generation, ctx))

    wandb_utils.log_metrics(combined, checkpoint_num=0)
    if args.output_file:
        pd.DataFrame([{**combined, "checkpoint": 0}]).to_csv(args.output_file, index=False)
        print(f"Results saved to {args.output_file}")
    wandb_utils.finish()


def run_deferred_phase(
    benchmarks,
    bench_examples,
    deferred_cache: Dict[Tuple[str, int], GenerationResult],
    args,
    loaded_rms,
) -> None:
    """Run deferred evaluators (e.g. vLLM judge) across all checkpoints."""
    for bench in benchmarks:
        for ev in bench.deferred_evaluators:
            print(f"[deferred] running {ev.name} on benchmark '{bench.name}'")
            for (bname, ckpt_num), gen in sorted(deferred_cache.items()):
                if bname != bench.name:
                    continue
                ctx = EvalContext(
                    args=args, checkpoint_num=ckpt_num, checkpoint_path=None,
                    llm=None, policy_tokenizer=None, loaded_rms=loaded_rms,
                    baseline_responses=None,
                )
                metrics = ev.evaluate(bench, bench_examples[bench.name], gen, ctx)
                wandb_utils.log_metrics(metrics, checkpoint_num=ckpt_num)
