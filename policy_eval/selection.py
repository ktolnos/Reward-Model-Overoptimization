"""Checkpoint selection + headline metric reporting.

The ``select`` benchmark scores every checkpoint with a held-out sibling RM on
the selection split (``select/sibling_rm/mean``). After the main loop has
produced per-checkpoint metrics on the validation/test split, this module:

    1. picks the checkpoint with the highest sibling-RM selection score, and
    2. reports that checkpoint's main metrics (the numbers that go in a paper):
       per-category Arena-Hard win_rate/sc_score, the macro-averaged Arena-Hard
       sc_score, the official strict IFEval accuracies, and the preference-split
       RM win-rate / style-controlled scores.

Selection signal and reported metrics come from different splits by design —
the sibling RM never touches the validation/test prompts it is selecting for.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .arena_hard_upstream import CATEGORY_BASELINES

SELECTION_METRIC = "select/sibling_rm/mean"

# Categories reported individually in the summary (the rest still feed the
# macro-average, but these two are the ones called out explicitly).
_HEADLINE_CATEGORIES = ["hard_prompt", "creative_writing"]


def _first_rm_judge_label(args) -> str:
    """Metric-key label for the first reward-model Arena-Hard judge.

    ``rm:gold_rm`` -> ``rm_gold_rm`` (matches RMJudge.name). Falls back to the
    gold RM judge label, which is the default.
    """
    for tok in (t.strip() for t in (args.arena_hard_judges or "").split(",") if t.strip()):
        if tok.startswith("rm:"):
            return f"rm_{tok[3:]}"
    return "rm_gold_rm"


def compute_aggregate_metrics(metrics: dict, args) -> Dict[str, float]:
    """Derive combined headline scores from a checkpoint's per-benchmark metrics.

    Computed in the main loop so every checkpoint logs them (wandb + CSV):
      - ``arena_hard/aggregate/sc_score``: macro-average of the per-category
        style-controlled scores (matches the Arena-Hard-Auto v2.0 overall).
      - ``ifeval/aggregate/strict_acc``: mean of the official prompt-level and
        instance-level strict accuracies.

    Returns only the aggregates whose inputs are present, so it is a no-op for
    runs that didn't include arena_hard / ifeval.
    """
    judge = _first_rm_judge_label(args)
    out: Dict[str, float] = {}

    sc_per_category = [
        float(metrics[f"arena_hard/{judge}/{cat}/sc_score"])
        for cat in CATEGORY_BASELINES
        if metrics.get(f"arena_hard/{judge}/{cat}/sc_score") is not None
    ]
    if sc_per_category:
        out["arena_hard/aggregate/sc_score"] = sum(sc_per_category) / len(sc_per_category)

    strict_accs = [
        float(metrics[k])
        for k in ("ifeval/prompt_strict_acc", "ifeval/inst_strict_acc")
        if metrics.get(k) is not None
    ]
    if strict_accs:
        out["ifeval/aggregate/strict_acc"] = sum(strict_accs) / len(strict_accs)

    return out


def select_best_checkpoint(
    results_rows: List[dict], metric_key: str = SELECTION_METRIC,
) -> Optional[Tuple[int, float, dict]]:
    """Return (checkpoint, selection_score, row) for the argmax selection score.

    Returns None if no row carries the selection metric (e.g. the 'select'
    benchmark wasn't run).
    """
    best: Optional[Tuple[int, float, dict]] = None
    for row in results_rows:
        if metric_key not in row or row[metric_key] is None:
            continue
        val = float(row[metric_key])
        if best is None or val > best[1]:
            best = (int(row["checkpoint"]), val, row)
    return best


def _summary_keys(args) -> List[str]:
    """The headline metric keys to lift out for the selected checkpoint.

    Aggregates (``arena_hard/aggregate/sc_score``, ``ifeval/aggregate/strict_acc``)
    are computed per-checkpoint in the main loop via ``compute_aggregate_metrics``;
    here we only read them back by key.
    """
    judge = _first_rm_judge_label(args)
    keys: List[str] = []
    for cat in _HEADLINE_CATEGORIES:
        keys.append(f"arena_hard/{judge}/{cat}/win_rate")
        keys.append(f"arena_hard/{judge}/{cat}/sc_score")
    keys.append("arena_hard/aggregate/sc_score")
    keys += ["ifeval/prompt_strict_acc", "ifeval/inst_strict_acc",
             "ifeval/aggregate/strict_acc"]
    for label in ("secondary_rm", "gold_rm"):
        keys += [f"{label}/sc_score", f"{label}/win_rate_vs_chosen"]
    return keys


def build_selected_summary(row: dict, args) -> Dict[str, float]:
    """Lift the headline main metrics out of a single checkpoint's metric row.

    Pure key-extraction — the aggregates are already in ``row`` (computed in the
    main loop). Missing metrics are simply absent (their benchmark wasn't run).
    """
    return {
        k: float(row[k])
        for k in _summary_keys(args)
        if k in row and row[k] is not None
    }


def report_selection(results_rows: List[dict], args) -> Optional[Dict]:
    """Pick the best checkpoint and report its headline metrics.

    Logs the summary to the wandb run summary (prefixed ``selected/``), prints a
    table, and writes ``<output_stem>_selected_summary.json``. No-op (returns
    None) when the selection metric is absent from every row.
    """
    picked = select_best_checkpoint(results_rows)
    if picked is None:
        print(f"[selection] '{SELECTION_METRIC}' not found in any row; "
              f"skipping checkpoint selection (was the 'select' benchmark run?).")
        return None

    ckpt, score, row = picked
    summary = build_selected_summary(row, args)

    print("\n" + "=" * 70)
    print(f"[selection] selected checkpoint-{ckpt} "
          f"({SELECTION_METRIC}={score:.4f}, argmax over "
          f"{sum(1 for r in results_rows if SELECTION_METRIC in r)} checkpoints)")
    print("-" * 70)
    print("Main metrics for the selected checkpoint:")
    width = max((len(k) for k in summary), default=0)
    for k in sorted(summary):
        print(f"  {k:<{width}}  {summary[k]:.4f}")
    print("=" * 70 + "\n")

    # wandb run summary: single headline values for the selected checkpoint.
    import wandb
    if wandb.run is not None:
        wandb.run.summary["selected/checkpoint"] = ckpt
        wandb.run.summary[f"selected/{SELECTION_METRIC}"] = score
        for k, v in summary.items():
            wandb.run.summary[f"selected/{k}"] = v

    payload = {
        "selected_checkpoint": ckpt,
        "selection_metric": SELECTION_METRIC,
        "selection_score": score,
        "split": getattr(args, "split", None),
        "selection_split": getattr(args, "selection_split", None),
        "metrics": summary,
    }
    out_stem = os.path.splitext(args.output_file)[0] if args.output_file else "evaluation_results"
    summary_path = f"{out_stem}_selected_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[selection] summary written to {summary_path}")
    except Exception as e:
        print(f"[selection] failed to write summary {summary_path}: {e}")

    return payload
