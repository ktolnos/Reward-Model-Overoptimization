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

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import wandb

from data_utils import get_generation_stop_token_ids

from . import wandb_utils
from .evaluators import PairwiseEvaluator, RewardModelEvaluator
from .judges import RMJudge
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


def fetch_training_history(
    checkpoints_dir: str, project: Optional[str]
) -> Optional[pd.DataFrame]:
    """Fetch the GRPO training run's history for re-logging into the eval run.

    The training run is identified by ``group == checkpoints_dir`` — grpo.sh
    sets ``WANDB_RUN_GROUP=${log_dir}`` and the eval is invoked with that same
    dir, so the group lookup is unambiguous for runs launched by the pipeline.

    Returns a DataFrame sorted by ``_step`` with scalar metric columns, or
    ``None`` if the run can't be found / wandb API fails / history is empty.
    """
    if not project or project.lower() == "none":
        return None
    try:
        api = wandb.Api()
        runs = list(api.runs(path=project, filters={"group": checkpoints_dir}))
    except Exception as e:
        print(f"[eval] could not query training runs in '{project}': {e}")
        return None

    if not runs:
        print(f"[eval] no training runs with group={checkpoints_dir} in '{project}'")
        return None
    if len(runs) > 1:
        # Pick the most recent. Stale duplicates (e.g. an aborted restart) sort below.
        runs.sort(key=lambda r: r.created_at, reverse=True)
        print(f"[eval] {len(runs)} training runs found; using most recent: {runs[0].name}")

    try:
        df = runs[0].history(samples=10000, pandas=True)
    except Exception as e:
        print(f"[eval] failed to fetch training history: {e}")
        return None

    if df.empty or "_step" not in df.columns:
        print(f"[eval] training history empty or missing _step")
        return None

    df = df.dropna(subset=["_step"]).copy()
    df["_step"] = df["_step"].astype(int)
    df = df.sort_values("_step").reset_index(drop=True)
    print(f"[eval] mirroring {len(df.columns) - 1} training metrics from "
          f"'{runs[0].name}' ({len(df)} rows)")
    return df


def lookup_train_metrics(
    history_df: Optional[pd.DataFrame], target_step: int
) -> Dict[str, float]:
    """Return scalar train metrics at the ``_step`` nearest to ``target_step``.

    Non-scalar columns (tables, histograms, NaNs) and wandb-internal ``_*``
    fields are dropped so they don't pollute the eval run.
    """
    if history_df is None or history_df.empty:
        return {}
    idx = (history_df["_step"] - target_step).abs().idxmin()
    row = history_df.loc[idx]
    metrics: Dict[str, float] = {}
    for k, v in row.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float)) and pd.notna(v):
            metrics[k] = float(v)
    return metrics


def rms_required_by(benchmarks: List[Benchmark]) -> set:
    """Scan benchmarks' evaluators for required RM labels.

    When adding a new evaluator type that consumes a reward model, extend this
    scan so ``LoadedRewardModels`` knows to load it upfront.
    """
    labels = set()
    for b in benchmarks:
        for ev in b.evaluators:
            if isinstance(ev, RewardModelEvaluator):
                labels.add(ev.rm_label)
            elif isinstance(ev, PairwiseEvaluator) and isinstance(ev.judge, RMJudge):
                labels.add(ev.judge.rm_label)
    return labels


# Fail the run if more than this fraction of generated baseline responses are
# truncated — a cut-off baseline answer corrupts every judge comparison.
_BASELINE_TRUNCATION_TOLERANCE = 0.10


def judge_baseline_label(args) -> str:
    """The judge's baseline slot/model name. ``"chosen"`` when comparing against
    the dataset response, else a filesystem-safe slug of ``--baseline_model_path``.

    Used both to inject the baseline responses into example metadata and to tell
    the judge which slot to read, so the two always agree."""
    if args.use_dataset_response_as_baseline:
        return "chosen"
    if not args.baseline_model_path:
        raise ValueError(
            "--baseline_model_path or --use_dataset_response_as_baseline required for LLM judge"
        )
    return (
        args.baseline_model_path.replace("/", "_").replace("\\", "_").replace(":", "_")
    )


def _baseline_cache_key(prompts: List[str], max_new_tokens: int) -> str:
    h = hashlib.sha256()
    h.update(str(max_new_tokens).encode())
    h.update(b"\x00")
    for p in prompts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def make_baseline_responses(
    args,
    preference_bench: Benchmark,
    examples: List[Example],
    tokenizer,
) -> Tuple[str, List[str]]:
    """Return ``(baseline_label, responses)`` — the judge's reference answers.

    ``--use_dataset_response_as_baseline`` plucks the dataset 'chosen' answers
    (label ``"chosen"``). Otherwise responses are generated from
    ``--baseline_model_path``: cached on disk (keyed on model + prompts +
    max_new_tokens) so re-runs don't regenerate, and the run fails early if more
    than 10% are truncated (a cut-off baseline corrupts the comparison)."""
    label = judge_baseline_label(args)
    if args.use_dataset_response_as_baseline:
        print("[judge] using dataset 'chosen' as baseline")
        return label, [ex.metadata.get("chosen_response", "") for ex in examples]

    prompts = [preference_bench.format_prompt(ex, tokenizer, thinking=True) for ex in examples]
    # Run-independent cache: responses are deterministic (temperature 0) given
    # (model, prompts, max_new_tokens), so share the cache across all runs.
    cache_dir = args.baseline_cache_dir
    cache_file = os.path.join(
        cache_dir, f"{label}__{_baseline_cache_key(prompts, args.max_new_tokens)}.json"
    )
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("num_samples") == len(prompts) and "responses" in cached:
                print(f"[judge] loaded {len(prompts)} cached baseline responses from {cache_file}")
                return label, cached["responses"]
        except Exception as e:
            print(f"[judge] failed to read baseline cache {cache_file}: {e}")

    # Spin up a disposable vLLM for the baseline. Called before the policy engine
    # is initialized, so the two never run concurrently.
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
    outputs = baseline_llm.generate(prompts, sampling)
    responses = [o.outputs[0].text for o in outputs]
    n_truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
    teardown_vllm(baseline_llm)

    frac = n_truncated / len(responses) if responses else 0.0
    if frac > _BASELINE_TRUNCATION_TOLERANCE:
        raise ValueError(
            f"{n_truncated}/{len(responses)} ({frac:.0%}) baseline responses from "
            f"{args.baseline_model_path} were truncated at max_new_tokens="
            f"{args.max_new_tokens}; raise --max_new_tokens. A truncated baseline "
            f"corrupts the judge comparison."
        )
    if n_truncated:
        print(f"[judge] warning: {n_truncated}/{len(responses)} ({frac:.0%}) baseline responses truncated")

    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(cache_file, "w") as f:
            json.dump({
                "baseline_model": args.baseline_model_path,
                "max_new_tokens": args.max_new_tokens,
                "num_samples": len(responses),
                "num_truncated": n_truncated,
                "responses": responses,
            }, f)
        print(f"[judge] cached baseline responses to {cache_file}")
    except Exception as e:
        print(f"[judge] failed to write baseline cache {cache_file}: {e}")
    return label, responses


def chosen_responses_as_generation(examples: List[Example]) -> GenerationResult:
    """Build a GenerationResult from the dataset's chosen responses (no generation)."""
    responses = [ex.metadata.get("chosen_response", "") for ex in examples]
    return GenerationResult(
        responses=responses,
        raw_responses=responses,
        finish_reasons=["stop"] * len(responses),
        n_responses_per_example=1,
        response_token_lens=[None] * len(responses),
    )


def run_chosen_only(args, benchmarks, bench_examples, loaded_rms,
                    *, per_example_dir: str) -> None:
    """Score dataset chosen responses directly. No generation, single step=0."""
    from .persistence import (
        PerExampleRecorder, init_base_columns, write_recorder,
    )

    preference_bench = next((b for b in benchmarks if b.name == "preference"), None)
    if preference_bench is None:
        raise ValueError("--evaluate_chosen_responses requires the 'preference' benchmark")

    examples = bench_examples["preference"]
    generation = chosen_responses_as_generation(examples)

    recorder = PerExampleRecorder(
        benchmark_name="preference", checkpoint_num=0,
        n_responses_per_example=1, n_examples=len(examples),
    )
    init_base_columns(recorder, examples, generation,
                      response_token_budget=args.response_token_budget)

    ctx = EvalContext(
        args=args, checkpoint_num=0, checkpoint_path=None,
        llm=None, policy_tokenizer=None, loaded_rms=loaded_rms,
        recorder=recorder,
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

    write_recorder(recorder, per_example_dir, fmt=args.per_example_format)

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
    """Run deferred evaluators (e.g. the vLLM LLM judge) across all checkpoints.

    A deferred evaluator that holds a GPU model (the vLLM judge) loads it lazily
    on its first ``evaluate`` and reuses it across checkpoints. Evaluators that
    share one backend (e.g. the preference and arena_hard judges naming the same
    model) load that model once: the shared backend is freed only after the last
    evaluator using it finishes, so different judges never sit in GPU memory at
    the same time and the same judge isn't reloaded per benchmark.
    """
    from collections import Counter

    pairs = [(bench, ev) for bench in benchmarks for ev in bench.deferred_evaluators]

    def _resource(ev):
        # The GPU resource to free is the judge's backend (shared across judges
        # naming the same model); fall back to the evaluator itself.
        backend = getattr(getattr(ev, "judge", None), "backend", None)
        if backend is not None and hasattr(backend, "teardown"):
            return backend
        return ev if hasattr(ev, "teardown") else None

    remaining = Counter(id(_resource(ev)) for _, ev in pairs if _resource(ev) is not None)

    for bench, ev in pairs:
        print(f"[deferred] running {ev.name} on benchmark '{bench.name}'")
        for (bname, ckpt_num), gen in sorted(deferred_cache.items()):
            if bname != bench.name:
                continue
            ctx = EvalContext(
                args=args, checkpoint_num=ckpt_num, checkpoint_path=None,
                llm=None, policy_tokenizer=None, loaded_rms=loaded_rms,
            )
            metrics = ev.evaluate(bench, bench_examples[bench.name], gen, ctx)
            wandb_utils.log_metrics(metrics, checkpoint_num=ckpt_num)
        res = _resource(ev)
        if res is not None:
            remaining[id(res)] -= 1
            if remaining[id(res)] == 0:
                res.teardown()
