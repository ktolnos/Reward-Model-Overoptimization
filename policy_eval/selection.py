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

# The benchmark producing the selection signal, and the per-example column its
# RM evaluator writes (``RewardModelEvaluator("sibling_rm").name`` is
# ``rm_sibling_rm``). The column lets a later run recompute the selection argmax
# straight from cached per-example logs, without loading the sibling RM.
SELECTION_BENCHMARK = "select"
SELECTION_SCORE_COLUMN = "score__rm_sibling_rm"

# Categories reported individually in the summary (the rest still feed the
# macro-average, but these two are the ones called out explicitly).
_HEADLINE_CATEGORIES = ["hard_prompt", "creative_writing"]


def _base_fingerprint(config) -> Dict[str, object]:
    """Structural fingerprint of a (reward) model's base, from its HF config.

    Two checkpoints that are different-seed siblings of the same base model share
    all of these; a different base (e.g. 0.6B vs 4B, or a different family) does
    not. The classifier head / exact weights are intentionally ignored.
    """
    sub = getattr(config, "text_config", None) or config
    return {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(sub, "hidden_size", getattr(config, "hidden_size", None)),
        "num_hidden_layers": getattr(sub, "num_hidden_layers",
                                     getattr(config, "num_hidden_layers", None)),
        "num_attention_heads": getattr(sub, "num_attention_heads",
                                       getattr(config, "num_attention_heads", None)),
        "vocab_size": getattr(sub, "vocab_size", getattr(config, "vocab_size", None)),
    }


def assert_sibling_base_matches_training(args) -> None:
    """Fail fast unless the sibling RM shares the training RM's base model.

    The sibling RM must be an independently-seeded RM of the *same* base model as
    the training RM, so comparing on the same split is meaningful. This catches
    the common footgun of leaving the sibling path at a default that belongs to a
    different base than the training RM actually used.

    No-op when the training RM path is unset (nothing to compare against).
    """
    from transformers import AutoConfig

    training_path = getattr(args, "training_rm_path", "")
    sibling_path = getattr(args, "sibling_rm_path", "")
    if not training_path or training_path.lower() == "none":
        print("[selection] --training_rm_path unset; skipping sibling/training "
              "base-model compatibility check.")
        return

    train_cfg = AutoConfig.from_pretrained(training_path, trust_remote_code=True)
    sib_cfg = AutoConfig.from_pretrained(sibling_path, trust_remote_code=True)
    train_fp = _base_fingerprint(train_cfg)
    sib_fp = _base_fingerprint(sib_cfg)
    if train_fp != sib_fp:
        raise ValueError(
            "Sibling RM base model does not match the training RM base model — the "
            "sibling must be a different-seed counterpart of the same base.\n"
            f"  training_rm ({training_path}): {train_fp}\n"
            f"  sibling_rm  ({sibling_path}): {sib_fp}\n"
            "Set --sibling_rm_path to the same-base, different-seed RM (or drop "
            "'select' from --benchmarks)."
        )
    print(f"[selection] sibling/training base match OK ({train_fp['model_type']}, "
          f"hidden_size={train_fp['hidden_size']}, layers={train_fp['num_hidden_layers']}).")


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


# Headline LLM-judge metrics surfaced for the selected checkpoint.
_JUDGE_SUMMARY_METRICS = ("arena_score", "sc_score", "win_rate")


def _llm_judge_labels(args) -> set:
    """Metric-key segments for the configured LLM judges (preference + arena_hard).

    An LLM judge's metric-key segment is its sanitized model name (no prefix),
    e.g. ``openai/gpt-4.1`` -> ``openai_gpt-4.1``, mirroring ``LLMJudge.name``
    (which is just ``backend.label``). Covers the preference judge
    (``--llm_judge_model_name``) and every ``llm:<model>`` in ``arena_hard_judges``.
    """
    labels = set()
    if args.evaluate_with_llm_judge and args.llm_judge_model_name:
        labels.add(args.llm_judge_model_name.replace("/", "_"))
    for tok in (t.strip() for t in (args.arena_hard_judges or "").split(",") if t.strip()):
        if tok.startswith("llm:"):
            labels.add(tok[4:].replace("/", "_"))
    return labels


def _judge_metric_keys(row: dict, args) -> List[str]:
    """Headline LLM-judge metric keys present in a checkpoint's row.

    Matches the preference judge (``<judge>/<slot>/<metric>``) and the arena_hard
    LLM judge (``arena_hard/<judge>/<cat>/<metric>``) by detecting a path segment
    equal to one of the configured LLM-judge labels and a headline final metric,
    so it works regardless of the judge model/backend in use.
    """
    labels = _llm_judge_labels(args)
    keys = []
    for k in row:
        parts = k.split("/")
        if parts[-1] in _JUDGE_SUMMARY_METRICS and any(p in labels for p in parts):
            keys.append(k)
    return sorted(keys)


def build_selected_summary(row: dict, args) -> Dict[str, float]:
    """Lift the headline main metrics out of a single checkpoint's metric row.

    Pure key-extraction — the aggregates are already in ``row`` (computed in the
    main loop). Includes the RM-panel metrics, IFEval, Arena-Hard, and any LLM
    judge metrics present. Missing metrics are simply absent (benchmark not run).
    """
    keys = list(dict.fromkeys(_summary_keys(args) + _judge_metric_keys(row, args)))
    return {
        k: float(row[k])
        for k in keys
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
