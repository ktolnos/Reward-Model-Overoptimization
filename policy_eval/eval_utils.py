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

import dataclasses
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import wandb
from transformers import HfArgumentParser

from data_utils import get_generation_stop_token_ids, read_run_manifest

from . import wandb_utils
from .evaluators import KLEvaluator, PairwiseEvaluator, RewardModelEvaluator
from .judges import RMJudge
from .generation import teardown_vllm
from .types import Benchmark, EvalContext, Example, GenerationResult


def _explicit_cli_fields(args, argv: List[str]) -> Set[str]:
    """Names of the dataclass fields explicitly set on the command line.

    Re-parses ``argv`` with a probe parser built from the same dataclass, with
    every default replaced by a sentinel: whatever ends up non-sentinel must
    have been bound from ``argv``. Delegating to argparse means ``--field=value``,
    dashed aliases (``--dataset-name``), and unambiguous prefix abbreviations
    (``--eval_temp``) are all resolved exactly as the real parse resolved them,
    so explicitness detection can never disagree with the parsed args.
    """
    probe = HfArgumentParser(type(args))
    missing = object()
    probe.set_defaults(**{f.name: missing for f in dataclasses.fields(args)})
    namespace, _ = probe.parse_known_args(argv)
    return {name for name, value in vars(namespace).items() if value is not missing}


def apply_run_manifest_defaults(args, argv: Optional[List[str]] = None) -> None:
    """Default eval args from the training run's ``run_manifest.json``.

    ``my_grpo.py`` writes the manifest into the checkpoints dir, so the eval
    configuration derives from what the training run actually used instead of
    hardcoded shell defaults that can go stale. Precedence:

        explicit CLI flag  >  run manifest  >  ScriptArguments default

    A flag counts as explicit when argparse binds a value to it from ``argv``
    (see ``_explicit_cli_fields``). Legacy runs without a manifest are left
    untouched.
    """
    explicit = _explicit_cli_fields(args, sys.argv[1:] if argv is None else argv)
    manifest = read_run_manifest(args.checkpoints_dir)
    if manifest is None:
        print(f"[run-manifest] none found for {args.checkpoints_dir!r}; "
              "using CLI values / defaults")
        return

    def apply(field: str, value) -> None:
        if field in explicit or value is None:
            return
        setattr(args, field, value)
        print(f"[run-manifest] {field} = {value!r}")

    apply("dataset_name", manifest.get("dataset_path"))
    apply("kl_base_model_path", manifest.get("model_name_or_path"))
    # eval_temperature is deliberately NOT derived from the manifest: higher
    # sampling temperature degrades the metrics regardless of the training
    # regime, so eval uses a fixed temperature (its CLI value / default)
    # rather than matching the training temperature.
    reward_model_paths = manifest.get("reward_model_paths") or []
    if reward_model_paths:
        if len(reward_model_paths) > 1 and "training_rm_path" not in explicit:
            print(f"[run-manifest] training used {len(reward_model_paths)} "
                  "reward models; taking the first as training_rm_path")
        apply("training_rm_path", reward_model_paths[0])


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

    The training run is resolved from the run manifest's ``wandb_run_id`` when
    present (exact — ``my_grpo.py`` initializes wandb and writes the manifest
    at train start).
    Legacy manifest-less runs fall back to ``group == checkpoints_dir`` —
    grpo.sh sets ``WANDB_RUN_GROUP=${log_dir}`` and the eval is invoked with
    that same dir, so the group lookup is unambiguous for pipeline runs.

    Returns a DataFrame sorted by ``_step`` with scalar metric columns, or
    ``None`` if the run can't be found / wandb API fails / history is empty.
    """
    if not project or project.lower() == "none":
        return None
    try:
        # Eagerly verifies the API key / connectivity, so it can raise even
        # when wandb.init succeeded (e.g. offline mode). History mirroring is
        # best-effort — never abort the eval over it.
        api = wandb.Api()
    except Exception as e:
        print(f"[eval] wandb API unavailable ({e}); skipping training history")
        return None

    run = None
    manifest = read_run_manifest(checkpoints_dir) or {}
    run_id = manifest.get("wandb_run_id")
    if run_id:
        run_path = f"{manifest.get('wandb_project') or project}/{run_id}"
        try:
            run = api.run(run_path)
            print(f"[eval] training run from manifest: {run.name} ({run_path})")
        except Exception as e:
            print(f"[eval] manifest wandb run '{run_path}' not fetchable ({e}); "
                  "falling back to group lookup")

    if run is None:
        try:
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
        run = runs[0]

    try:
        # scan_history returns every logged row (history(samples=N) downsamples
        # long runs, so nearest-_step matching in lookup_train_metrics could land
        # far from the checkpoint step).
        df = pd.DataFrame(run.scan_history())
    except Exception as e:
        print(f"[eval] failed to fetch training history: {e}")
        return None

    if df.empty or "_step" not in df.columns:
        print(f"[eval] training history empty or missing _step")
        return None

    df = df.dropna(subset=["_step"]).copy()
    df["_step"] = df["_step"].astype(int)
    df = df.sort_values("_step").reset_index(drop=True)
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


# =============================================================================
# Checkpoint-0 backfill from the base model's own eval run
# =============================================================================
#
# The base model a GRPO run started from (``model_name_or_path`` in the run
# manifest, i.e. ``base_model_name`` in grpo.sh) is "checkpoint 0" of the run's
# trajectory. Its own eval run — which we run independently for every base model
# anyway — already holds every run-independent metric. Rather than re-generate
# and re-score the base model in every GRPO eval, we look that run up in wandb
# and log its metrics at checkpoint 0, giving every eval plot a shared origin.

# Metric families that must NOT be carried over verbatim from an independent
# base eval: ``training_rm`` scored a *different* RM (each GRPO run has its own),
# and ``kl`` is measured against the base model itself so it is 0 at checkpoint 0
# by construction — re-injected below rather than copied.
_BASE_CKPT_DROP_SUBSTRINGS = ("training_rm",)
_BASE_CKPT_KL_ZERO = {
    "kl/mean": 0.0, "kl/std": 0.0, "kl/grpo_mean": 0.0, "kl/grpo_std": 0.0,
}


def _norm_identity(value):
    """Normalise a path/name for equality checks (trailing slash, whitespace)."""
    if value is None:
        return None
    return str(value).strip().rstrip("/")


def _base_eval_compatible(run, args) -> bool:
    """Whether a candidate base eval run was generated the same way as this run.

    Only the generation-level identity keys are required to match — differing
    reward models are handled per-metric in ``_rm_family_incompatible`` (so a
    base eval with a different secondary/sibling RM still contributes its gold /
    IFEval / Arena-Hard numbers)."""
    cfg = run.config or {}
    if getattr(run, "job_type", None) != "evaluation":
        return False
    if cfg.get("evaluate_chosen_responses"):
        return False  # scored dataset chosen responses, not a generated policy
    if _norm_identity(cfg.get("dataset_name")) != _norm_identity(args.dataset_name):
        return False
    if cfg.get("split") != args.split:
        return False
    if cfg.get("eval_temperature") != args.eval_temperature:
        return False
    if cfg.get("max_new_tokens") != args.max_new_tokens:
        return False
    if cfg.get("max_length") != args.max_length:
        return False
    return True


def _rm_family_incompatible(metric_key: str, base_cfg: dict, args) -> bool:
    """Whether a base-eval metric depends on an RM that differs from this run's.

    Gold / secondary / sibling RM scores are only comparable when the base eval
    used the *same* RM. This gates each family on the matching config key so a
    partially-compatible base eval still contributes the families that do match.
    """
    if "gold_rm" in metric_key:
        return _norm_identity(base_cfg.get("gold_rm_name")) != _norm_identity(args.gold_rm_name)
    if "secondary_rm" in metric_key:
        return _norm_identity(base_cfg.get("secondary_rm_name")) != _norm_identity(args.secondary_rm_name)
    if "sibling_rm" in metric_key or metric_key.startswith("select"):
        return _norm_identity(base_cfg.get("sibling_rm_path")) != _norm_identity(args.sibling_rm_path)
    return False


def _extract_single_checkpoint_metrics(run) -> Optional[Tuple[int, Dict[str, float]]]:
    """Return ``(checkpoint_num, scalar_metrics)`` for a single-checkpoint run.

    A base eval evaluates exactly one model, but its ``checkpoint`` axis value is
    0 for a plain model dir / HF id and the embedded step for a ``checkpoint-N``
    dir (e.g. an SFT checkpoint). We take the run's single distinct ``checkpoint``
    value rather than assuming 0, and refuse to guess if the run turns out to
    hold more than one — that means it isn't the base eval we expect.
    """
    try:
        df = pd.DataFrame(run.scan_history())
    except Exception as e:
        print(f"[base-ckpt] failed to fetch base eval history: {e}")
        return None
    if df.empty or "checkpoint" not in df.columns:
        print("[base-ckpt] base eval history empty or missing 'checkpoint' axis")
        return None
    ckpts = sorted({int(c) for c in df["checkpoint"].dropna().tolist()})
    if len(ckpts) != 1:
        print(f"[base-ckpt] expected exactly one checkpoint in base eval run "
              f"'{run.name}', found {ckpts}; refusing to guess. "
              f"Pin the run with --base_eval_run_id.")
        return None
    rows = df[df["checkpoint"] == ckpts[0]]
    merged: Dict[str, float] = {}
    for _, row in rows.iterrows():
        for k, v in row.items():
            if k.startswith("_") or k == "checkpoint":
                continue
            if isinstance(v, (int, float)) and pd.notna(v):
                merged[k] = float(v)
    return ckpts[0], merged


def fetch_base_checkpoint_metrics(args) -> Optional[Dict[str, float]]:
    """Fetch the base model's eval metrics to log as this run's checkpoint 0.

    Returns the metrics dict (ready to log at ``checkpoint_num=0``) or ``None``
    when disabled, the base model can't be resolved, wandb is unavailable, or no
    compatible base eval run exists.

    Lookup correctness (the two things that must be right):
      - **Run.** Eval runs are grouped by ``checkpoints_dir``; a base eval's
        group is the base-model path exactly. Among runs with that group we keep
        only those whose generation config matches (``_base_eval_compatible``)
        and take the most recent. ``--base_eval_run_id`` pins one and skips the
        search entirely.
      - **Checkpoint.** Taken as the run's single distinct ``checkpoint`` value
        (see ``_extract_single_checkpoint_metrics``), not assumed to be 0.

    Run-specific families are handled rather than copied blindly: ``training_rm``
    is dropped, ``kl`` is dropped and re-injected as 0, and gold/secondary/sibling
    families are dropped when the base eval's RM differs from this run's.
    """
    if not getattr(args, "prepend_base_checkpoint", True):
        return None

    manifest = read_run_manifest(args.checkpoints_dir) or {}
    base_path = _norm_identity(manifest.get("model_name_or_path") or args.kl_base_model_path)
    if not base_path:
        print("[base-ckpt] base model path unknown (no manifest model_name_or_path "
              "or --kl_base_model_path); skipping checkpoint-0 backfill")
        return None

    try:
        # Eagerly verifies API key / connectivity — can raise even when
        # wandb.init succeeded (offline mode). Backfill is best-effort.
        api = wandb.Api()
    except Exception as e:
        print(f"[base-ckpt] wandb API unavailable ({e}); skipping checkpoint-0 backfill")
        return None

    run = None
    if getattr(args, "base_eval_run_id", None):
        run_path = f"{args.wandb_project}/{args.base_eval_run_id}"
        try:
            run = api.run(run_path)
        except Exception as e:
            print(f"[base-ckpt] --base_eval_run_id '{run_path}' not fetchable ({e}); "
                  "skipping checkpoint-0 backfill")
            return None
        if not _base_eval_compatible(run, args):
            print(f"[base-ckpt] warning: pinned base eval run '{run.name}' has a "
                  "generation config that differs from this run (dataset/split/"
                  "temperature/lengths); backfilling anyway as requested.")
    else:
        try:
            candidates = list(api.runs(path=args.wandb_project, filters={"group": base_path}))
        except Exception as e:
            print(f"[base-ckpt] could not query base eval runs in "
                  f"'{args.wandb_project}': {e}")
            return None
        candidates = [r for r in candidates if _base_eval_compatible(r, args)]
        if not candidates:
            print(f"[base-ckpt] no compatible base eval run found "
                  f"(group={base_path!r}, dataset={args.dataset_name!r}, "
                  f"split={args.split!r}, temp={args.eval_temperature}); "
                  "checkpoint 0 will be absent. Run the base model through "
                  "evaluate_policy.py with the same settings, or pass "
                  "--base_eval_run_id.")
            return None
        candidates.sort(key=lambda r: r.created_at, reverse=True)
        if len(candidates) > 1:
            print(f"[base-ckpt] {len(candidates)} compatible base eval runs; "
                  f"using most recent: {candidates[0].name}")
        run = candidates[0]

    extracted = _extract_single_checkpoint_metrics(run)
    if extracted is None:
        return None
    base_ckpt_num, raw_metrics = extracted

    base_cfg = dict(run.config or {})
    metrics: Dict[str, float] = {}
    dropped_rm = 0
    for k, v in raw_metrics.items():
        if any(s in k for s in _BASE_CKPT_DROP_SUBSTRINGS) or k.startswith("kl/"):
            continue
        if _rm_family_incompatible(k, base_cfg, args):
            dropped_rm += 1
            continue
        metrics[k] = v
    metrics.update(_BASE_CKPT_KL_ZERO)

    print(f"[base-ckpt] base eval '{run.name}' ({run.id}) evaluated at checkpoint "
          f"{base_ckpt_num}; backfilling {len(metrics)} metrics at checkpoint 0 "
          f"(dropped {dropped_rm} RM-mismatched)")
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
        if isinstance(ev, PairwiseEvaluator):
            print(f"[chosen-only] skipping {ev.name} (judge needs baseline)")
            continue
        combined.update(ev.evaluate(preference_bench, examples, generation, ctx))

    write_recorder(recorder, per_example_dir, fmt=args.per_example_format)

    wandb_utils.log_metrics(combined, checkpoint_num=0)
    if args.output_file:
        pd.DataFrame([{**combined, "checkpoint": 0}]).to_csv(args.output_file, index=False)
        print(f"Results saved to {args.output_file}")
    wandb_utils.finish()


def run_kl_base_phase(benchmarks, llm, args) -> Dict[int, dict]:
    """Run the KL base-model pass for every checkpoint in one weight swap.

    Called after the checkpoint loop while the policy engine is still alive:
    all generation is done, so the engine can be left holding the base
    weights. This costs one base-weight load per eval instead of the two
    extra full loads per checkpoint that an in-loop swap+restore would.
    Returns ``{checkpoint_num: metrics}``, merged into the result rows the
    same way as the deferred metrics. Empty when no benchmark has a KLEvaluator
    (or nothing was stashed).
    """
    metrics: Dict[int, dict] = {}
    for bench in benchmarks:
        for ev in bench.evaluators:
            if isinstance(ev, KLEvaluator):
                for ckpt_num, m in ev.run_base_phase(llm, bench, args).items():
                    metrics.setdefault(ckpt_num, {}).update(m)
    return metrics


def run_deferred_phase(
    benchmarks,
    bench_examples,
    deferred_cache: Dict[Tuple[str, int], GenerationResult],
    args,
    loaded_rms,
) -> Dict[int, dict]:
    """Run deferred evaluators (e.g. the vLLM LLM judge) across all checkpoints.

    A deferred evaluator that holds a GPU model (the vLLM judge) loads it lazily
    on its first ``evaluate`` and reuses it across checkpoints. Evaluators that
    share one backend (e.g. the preference and arena_hard judges naming the same
    model) load that model once: the shared backend is freed only after the last
    evaluator using it finishes, so different judges never sit in GPU memory at
    the same time and the same judge isn't reloaded per benchmark.

    Metrics are logged to wandb here and also returned as
    ``{checkpoint_num: metrics}`` so the caller can fold them into the result
    rows (CSV + checkpoint-selection summary), which were built before this
    phase ran.
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

    deferred_metrics: Dict[int, dict] = {}
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
            deferred_metrics.setdefault(ckpt_num, {}).update(metrics)
        res = _resource(ev)
        if res is not None:
            remaining[id(res)] -= 1
            if remaining[id(res)] == 0:
                res.teardown()
    return deferred_metrics
