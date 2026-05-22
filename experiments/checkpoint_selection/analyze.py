"""Per-run + aggregate metrics for the checkpoint-selection experiment.

Inputs (all under --output_dir, which is the same directory the rescore
script writes to):
    scores.parquet           - long-form RM scores per (run, ckpt, prompt, rm)
    train_reward.parquet     - per-step training reward from wandb (local fetch)
    prompt_partition.json    - {prompt_hash: "A"|"B"}

Outputs:
    per_run_strategy_metrics.csv
    aggregate.json
    plots/  (mean gold@B by strategy; cross-run scatter)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow direct ``python experiments/checkpoint_selection/analyze.py``.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from experiments.checkpoint_selection import manifest as M  # noqa: E402
else:
    from . import manifest as M


# ---------------------------------------------------------------------------
# Strategy names
# ---------------------------------------------------------------------------

STRATEGIES_DEPLOYABLE = [
    "first",
    "last",
    "random",
    "train_reward",
    "training_rm",       # training_rm @ test_A
    "sibling_rm",        # sibling_rm  @ test_A
    "secondary_rm",      # secondary_rm @ test_A
]
STRATEGIES_DIAGNOSTIC = [
    "gold_at_A",         # diagnostic
    "gold_at_B",         # oracle verdict; reported as "oracle"
]
ALL_STRATEGIES = STRATEGIES_DEPLOYABLE + STRATEGIES_DIAGNOSTIC


# ---------------------------------------------------------------------------
# Loading + reshaping
# ---------------------------------------------------------------------------

@dataclass
class Tables:
    """Pivoted tables consumed by the strategy logic."""
    # (run_idx, ckpt, rm_label) → mean score on the slice
    score_A: pd.DataFrame   # MultiIndex (run_idx, checkpoint), cols=rm_labels
    score_B: pd.DataFrame
    # (run_idx, ckpt) → train_reward (NaN where missing)
    train_reward: pd.Series
    # All checkpoints by run_idx (sorted).
    checkpoints_per_run: Dict[int, List[int]]


def _aggregate_by_slice(scores: pd.DataFrame, partition: Dict[str, str], slice_label: str) -> pd.DataFrame:
    """Compute mean score per (run, checkpoint, rm_label) on the given slice."""
    mask = scores["prompt_hash"].map(partition).eq(slice_label)
    sub = scores[mask]
    if sub.empty:
        raise RuntimeError(f"no scores fall in slice {slice_label}")
    agg = sub.groupby(["grpo_run_idx", "checkpoint", "rm_label"], as_index=False)["score"].mean()
    pivoted = agg.pivot_table(
        index=["grpo_run_idx", "checkpoint"],
        columns="rm_label",
        values="score",
    )
    return pivoted


def _align_train_reward(
    train_df: pd.DataFrame,
    runs_idx_to_id: Dict[int, str],
    saved_checkpoints: Dict[int, List[int]],
) -> pd.Series:
    """For each saved checkpoint step, pick the closest train_reward step.

    Returns a Series indexed by (run_idx, checkpoint).
    """
    if train_df is None or train_df.empty:
        idx = pd.MultiIndex.from_tuples(
            [(r, c) for r, cs in saved_checkpoints.items() for c in cs],
            names=["grpo_run_idx", "checkpoint"],
        )
        return pd.Series(np.nan, index=idx, name="train_reward")

    pieces = []
    for run_idx, ckpts in saved_checkpoints.items():
        wandb_id = runs_idx_to_id[run_idx]
        sub = train_df[train_df["grpo_run_id"] == wandb_id]
        if sub.empty:
            for c in ckpts:
                pieces.append((run_idx, c, np.nan))
            continue
        steps = sub["step"].to_numpy()
        rewards = sub["train_reward"].to_numpy()
        for c in ckpts:
            i = int(np.argmin(np.abs(steps - c)))
            pieces.append((run_idx, c, float(rewards[i])))
    df = pd.DataFrame(pieces, columns=["grpo_run_idx", "checkpoint", "train_reward"])
    return df.set_index(["grpo_run_idx", "checkpoint"])["train_reward"]


def load_tables(output_dir: str) -> Tables:
    scores = pd.read_parquet(os.path.join(output_dir, M.SCORES_FILENAME))
    with open(os.path.join(output_dir, M.PARTITION_FILENAME)) as f:
        partition: Dict[str, str] = json.load(f)
    train_path = os.path.join(output_dir, M.TRAIN_REWARD_FILENAME)
    train_df = pd.read_parquet(train_path) if os.path.isfile(train_path) else None
    if train_df is None:
        print(f"[analyze] WARNING: no {M.TRAIN_REWARD_FILENAME}; the train_reward strategy will be skipped.")

    score_A = _aggregate_by_slice(scores, partition, "A")
    score_B = _aggregate_by_slice(scores, partition, "B")

    saved_checkpoints: Dict[int, List[int]] = {
        int(run_idx): sorted({int(c) for c in df["checkpoint"]})
        for run_idx, df in scores.groupby("grpo_run_idx")
    }

    runs_idx_to_id = {r.idx: r.wandb_id for r in M.RUNS}
    train_aligned = _align_train_reward(train_df, runs_idx_to_id, saved_checkpoints)

    return Tables(
        score_A=score_A,
        score_B=score_B,
        train_reward=train_aligned,
        checkpoints_per_run=saved_checkpoints,
    )


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def _pick_argmax(values: pd.Series) -> Optional[int]:
    """Return the checkpoint with the highest value, NaN-safe."""
    v = values.dropna()
    if v.empty:
        return None
    return int(v.idxmax())


def per_run_selections(tables: Tables, n_random_seeds: int = 1000) -> pd.DataFrame:
    """For each (run, strategy), pick a checkpoint and read gold@test_B for it."""
    rows = []
    rng = np.random.default_rng(42)
    score_A = tables.score_A
    score_B = tables.score_B

    for run_idx, ckpts in tables.checkpoints_per_run.items():
        ckpts_sorted = sorted(ckpts)
        sub_A = score_A.loc[run_idx]                          # index=checkpoint
        sub_B = score_B.loc[run_idx]
        # gold@test_B per checkpoint — the verdict we score every strategy on.
        gold_B = sub_B["gold_rm"] if "gold_rm" in sub_B.columns else pd.Series(dtype=float)
        oracle_ckpt = _pick_argmax(gold_B)
        oracle_gold_B = float(gold_B.loc[oracle_ckpt]) if oracle_ckpt is not None else np.nan

        def _row(strategy: str, picked_ckpt: Optional[int], extra: Optional[dict] = None) -> dict:
            picked = picked_ckpt
            ach = float(gold_B.loc[picked]) if picked is not None and picked in gold_B.index else np.nan
            r = {
                "grpo_run_idx": run_idx,
                "grpo_run_id":  next((r.wandb_id for r in M.RUNS if r.idx == run_idx), None),
                "strategy":     strategy,
                "picked_checkpoint": picked,
                "gold_at_test_B":    ach,
                "oracle_gold_at_test_B": oracle_gold_B,
                "regret":            (oracle_gold_B - ach) if not (np.isnan(oracle_gold_B) or np.isnan(ach)) else np.nan,
            }
            if extra:
                r.update(extra)
            return r

        # 1. first / last (no signal needed)
        rows.append(_row("first", ckpts_sorted[0]))
        rows.append(_row("last", ckpts_sorted[-1]))

        # 2. random — average gold@B over uniform picks.
        if ckpts_sorted:
            picks = rng.choice(ckpts_sorted, size=n_random_seeds, replace=True)
            valid = [gold_B.loc[c] for c in picks if c in gold_B.index]
            mean_gold = float(np.mean(valid)) if valid else np.nan
            rows.append({
                "grpo_run_idx": run_idx,
                "grpo_run_id":  next((r.wandb_id for r in M.RUNS if r.idx == run_idx), None),
                "strategy":     "random",
                "picked_checkpoint": None,
                "gold_at_test_B":    mean_gold,
                "oracle_gold_at_test_B": oracle_gold_B,
                "regret":            oracle_gold_B - mean_gold if not (np.isnan(oracle_gold_B) or np.isnan(mean_gold)) else np.nan,
            })

        # 3. train_reward — argmax over wandb-aligned reward at saved steps.
        try:
            tr = tables.train_reward.loc[run_idx]
            picked = _pick_argmax(tr)
        except KeyError:
            picked = None
        rows.append(_row("train_reward", picked))

        # 4-6. RM-on-test_A argmax for each selector RM.
        for label, strat in [
            ("training_rm",  "training_rm"),
            ("sibling_rm",   "sibling_rm"),
            ("secondary_rm", "secondary_rm"),
        ]:
            picked = _pick_argmax(sub_A[label]) if label in sub_A.columns else None
            rows.append(_row(strat, picked))

        # 7. gold @ test_A — diagnostic.
        if "gold_rm" in sub_A.columns:
            picked = _pick_argmax(sub_A["gold_rm"])
            rows.append(_row("gold_at_A", picked))

        # 8. gold @ test_B — verdict oracle.
        rows.append(_row("gold_at_B", oracle_ckpt))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-run analysis
# ---------------------------------------------------------------------------

def cross_run_selection(tables: Tables) -> Dict[str, dict]:
    """For each strategy, pick the (run, ckpt) with highest selection signal.

    For RM-based strategies the signal is the RM's score on test_A. For
    ``train_reward`` it's the aligned wandb reward. For ``gold_at_B`` it's
    the oracle (gold_rm on test_B).
    """
    out: Dict[str, dict] = {}
    score_A = tables.score_A
    score_B = tables.score_B
    gold_B = score_B["gold_rm"] if "gold_rm" in score_B.columns else pd.Series(dtype=float)

    def _record(name: str, picked_idx, signal_value: Optional[float] = None):
        if picked_idx is None:
            out[name] = {"picked": None, "gold_at_test_B": np.nan, "signal": signal_value}
            return
        run_idx, ckpt = picked_idx
        ach = float(gold_B.loc[(run_idx, ckpt)]) if (run_idx, ckpt) in gold_B.index else np.nan
        out[name] = {
            "picked": {"grpo_run_idx": int(run_idx), "checkpoint": int(ckpt)},
            "gold_at_test_B": ach,
            "signal": signal_value,
        }

    # Oracle: argmax gold@B across all (run, ckpt).
    if not gold_B.empty:
        picked = gold_B.idxmax()
        _record("oracle_gold_at_test_B", picked, float(gold_B.loc[picked]))

    for label in ["training_rm", "sibling_rm", "secondary_rm"]:
        if label not in score_A.columns:
            continue
        s = score_A[label].dropna()
        if s.empty:
            continue
        picked = s.idxmax()
        _record(label, picked, float(s.loc[picked]))

    # train_reward: argmax over aligned reward across all (run, ckpt).
    tr = tables.train_reward.dropna()
    if not tr.empty:
        picked = tr.idxmax()
        _record("train_reward", picked, float(tr.loc[picked]))

    return out


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------

def aggregate(per_run: pd.DataFrame) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    grouped = per_run.groupby("strategy")
    for strategy, df in grouped:
        gold = df["gold_at_test_B"].dropna().to_numpy()
        regret = df["regret"].dropna().to_numpy()
        out[strategy] = {
            "n_runs": int(len(df)),
            "mean_gold_at_test_B": float(np.mean(gold)) if len(gold) else None,
            "mean_regret":         float(np.mean(regret)) if len(regret) else None,
            "boot_ci95_regret":    _paired_bootstrap_ci(regret) if len(regret) >= 2 else None,
        }
    return out


def _paired_bootstrap_ci(values: np.ndarray, n_boot: int = 5000) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(values[idx])))
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return (lo, hi)


def make_plots(per_run: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable: {e}; skipping plots")
        return

    plots_dir = os.path.join(output_dir, M.PLOTS_SUBDIR)
    os.makedirs(plots_dir, exist_ok=True)

    # Mean gold@test_B per strategy.
    summary = per_run.groupby("strategy")["gold_at_test_B"].agg(["mean", "sem"]).reindex(ALL_STRATEGIES).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary.index, summary["mean"], yerr=summary["sem"], capsize=3)
    ax.set_ylabel("Mean gold@test_B")
    ax.set_title("Checkpoint selection strategy — mean across runs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "mean_gold_by_strategy.png"), dpi=150)
    plt.close(fig)

    # Regret per strategy.
    regret = per_run.groupby("strategy")["regret"].agg(["mean", "sem"]).reindex(ALL_STRATEGIES).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(regret.index, regret["mean"], yerr=regret["sem"], capsize=3, color="C3")
    ax.set_ylabel("Regret vs oracle gold@test_B")
    ax.set_title("Per-strategy regret across runs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "regret_by_strategy.png"), dpi=150)
    plt.close(fig)

    print(f"[plots] wrote → {plots_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--n_random_seeds", type=int, default=1000)
    args = parser.parse_args()

    tables = load_tables(args.output_dir)

    print("\n=== per-run strategy selections ===")
    per_run = per_run_selections(tables, n_random_seeds=args.n_random_seeds)
    csv_path = os.path.join(args.output_dir, M.PER_RUN_METRICS_FILENAME)
    per_run.to_csv(csv_path, index=False)
    print(f"[analyze] wrote {len(per_run)} rows → {csv_path}")

    agg = aggregate(per_run)
    cross = cross_run_selection(tables)
    summary = {
        "per_strategy": agg,
        "cross_run": cross,
        "checkpoints_per_run": {str(k): v for k, v in tables.checkpoints_per_run.items()},
    }
    json_path = os.path.join(args.output_dir, M.AGGREGATE_FILENAME)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[analyze] wrote summary → {json_path}")

    make_plots(per_run, args.output_dir)
    print("\n[analyze] done")


if __name__ == "__main__":
    main()
