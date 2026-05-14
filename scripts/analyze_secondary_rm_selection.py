#!/usr/bin/env python3
"""Quantify how effective a held-out (secondary) RM is at selecting the
gold-best policy checkpoint from a GRPO run.

For each filtered run with >= --min-checkpoints evaluations, treat the
secondary RM as a checkpoint-selection rule and compare to:
  - gold-best   (oracle ceiling)
  - final       (no early stopping)
  - first       (training anchor)

Per-run history is fetched once from W&B and cached at
wandb_cache/history/<run_id>.json; subsequent invocations are offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "wandb_cache"
HISTORY_DIR = CACHE_DIR / "history"
SUMMARY_PATH = CACHE_DIR / "distill-llms_policy-evaluation.json"
DEFAULT_ENTITY_PROJECT = "distill-llms/policy-evaluation"

DEFAULT_DATASET = "ktolnos/helpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B"
DEFAULT_SECONDARY_RM = "Ray2333/GRM-Gemma-2B-sftreg"
DEFAULT_GOLD_METRIC = "gold_rm/mean"
DEFAULT_SECONDARY_METRIC = "secondary_rm/mean"


def filter_summary(runs, dataset, secondary_rm, gold_metric, secondary_metric):
    out = []
    for r in runs:
        cfg = r.get("config") or {}
        s = r.get("summary") or {}
        if cfg.get("dataset_name") != dataset:
            continue
        if cfg.get("secondary_rm_name") != secondary_rm:
            continue
        # Summary holds the last-logged values; require both metrics ever appeared.
        if gold_metric not in s or secondary_metric not in s:
            continue
        out.append(r)
    return out


def load_or_fetch_history(api, project, rid, gold_metric, secondary_metric, refetch=False):
    p = HISTORY_DIR / f"{rid}.json"
    if p.exists() and not refetch:
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    if api is None:
        return None
    run = api.run(f"{project}/{rid}")
    keys = ["checkpoint", gold_metric, secondary_metric]
    rows = [dict(row) for row in run.scan_history(keys=keys)]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))
    return rows


def collapse_history(rows, gold_metric, secondary_metric):
    """Reduce a history list to a DataFrame indexed by ascending checkpoint.

    Drops rows where checkpoint or either metric is missing. If the same
    checkpoint has multiple rows (e.g. re-evaluated), keep the last.
    """
    if not rows:
        return pd.DataFrame(columns=["checkpoint", "gold", "secondary"])
    df = pd.DataFrame(rows)
    needed = ["checkpoint", gold_metric, secondary_metric]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=["checkpoint", "gold", "secondary"])
    df = df[needed].rename(columns={gold_metric: "gold", secondary_metric: "secondary"})
    df = df.dropna()
    if df.empty:
        return df
    df["checkpoint"] = df["checkpoint"].astype(int)
    df = df.sort_values("checkpoint").drop_duplicates("checkpoint", keep="last")
    return df.reset_index(drop=True)


def per_run_metrics(df):
    g = df["gold"].to_numpy()
    s = df["secondary"].to_numpy()
    ckpt = df["checkpoint"].to_numpy()

    i_gold = int(np.argmax(g))
    i_sec = int(np.argmax(s))
    i_final = len(df) - 1
    i_first = 0

    # Ranks in gold-descending order (1 = perfect).
    gold_order = np.argsort(-g, kind="stable")
    rank_sec = int(np.where(gold_order == i_sec)[0][0]) + 1
    rank_final = int(np.where(gold_order == i_final)[0][0]) + 1

    rho, _ = spearmanr(g, s)
    rho = None if rho is None or np.isnan(rho) else float(rho)

    return {
        "n_ckpts": len(df),
        "first_ckpt": int(ckpt[i_first]),
        "final_ckpt": int(ckpt[i_final]),
        "gold_best_ckpt": int(ckpt[i_gold]),
        "secondary_best_ckpt": int(ckpt[i_sec]),
        "gold_at_first": float(g[i_first]),
        "gold_at_final": float(g[i_final]),
        "gold_at_gold_best": float(g[i_gold]),
        "gold_at_secondary_best": float(g[i_sec]),
        # Secondary-RM-best selection rule
        "rank_secondary": rank_sec,
        "exact_match_secondary": i_sec == i_gold,
        "top3_match_secondary": rank_sec <= 3,
        "top5_match_secondary": rank_sec <= 5,
        "ckpt_distance_secondary": int(abs(int(ckpt[i_sec]) - int(ckpt[i_gold]))),
        # Final-checkpoint selection rule (= no early stopping)
        "rank_final": rank_final,
        "exact_match_final": i_final == i_gold,
        "top3_match_final": rank_final <= 3,
        "top5_match_final": rank_final <= 5,
        "ckpt_distance_final": int(abs(int(ckpt[i_final]) - int(ckpt[i_gold]))),
        "spearman_gold_secondary": rho,
        # "Hacking observed" = gold peaks before the last evaluated checkpoint.
        "hacking_observed": i_gold != i_final,
    }


def aggregate(rows):
    df = pd.DataFrame(rows)

    # Fraction of oracle improvement recovered. Defined only for runs where
    # the gold-best is above the anchor by a non-trivial margin.
    denom = df["gold_at_gold_best"] - df["gold_at_first"]
    mask = denom > 1e-9
    if mask.any():
        frac_sec = ((df["gold_at_secondary_best"] - df["gold_at_first"])[mask]
                    / denom[mask]).clip(lower=-1.0, upper=2.0)
        frac_final = ((df["gold_at_final"] - df["gold_at_first"])[mask]
                      / denom[mask]).clip(lower=-1.0, upper=2.0)
        frac_sec_mean = float(frac_sec.mean())
        frac_final_mean = float(frac_final.mean())
    else:
        frac_sec_mean = frac_final_mean = float("nan")

    return {
        "n_runs": len(df),
        "n_ckpts_mean": float(df["n_ckpts"].mean()),
        "n_ckpts_median": float(df["n_ckpts"].median()),
        "n_ckpts_min": int(df["n_ckpts"].min()),
        "n_ckpts_max": int(df["n_ckpts"].max()),

        "hacking_observed_pct": 100.0 * float(df["hacking_observed"].mean()),

        "exact_match_secondary_pct": 100.0 * float(df["exact_match_secondary"].mean()),
        "top3_match_secondary_pct": 100.0 * float(df["top3_match_secondary"].mean()),
        "top5_match_secondary_pct": 100.0 * float(df["top5_match_secondary"].mean()),
        "rank_secondary_mean": float(df["rank_secondary"].mean()),
        "rank_secondary_median": float(df["rank_secondary"].median()),
        "ckpt_distance_secondary_mean": float(df["ckpt_distance_secondary"].mean()),
        "ckpt_distance_secondary_median": float(df["ckpt_distance_secondary"].median()),

        "exact_match_final_pct": 100.0 * float(df["exact_match_final"].mean()),
        "top3_match_final_pct": 100.0 * float(df["top3_match_final"].mean()),
        "top5_match_final_pct": 100.0 * float(df["top5_match_final"].mean()),
        "rank_final_mean": float(df["rank_final"].mean()),
        "rank_final_median": float(df["rank_final"].median()),
        "ckpt_distance_final_mean": float(df["ckpt_distance_final"].mean()),
        "ckpt_distance_final_median": float(df["ckpt_distance_final"].median()),

        "gold_at_first_mean": float(df["gold_at_first"].mean()),
        "gold_at_gold_best_mean": float(df["gold_at_gold_best"].mean()),
        "gold_at_secondary_best_mean": float(df["gold_at_secondary_best"].mean()),
        "gold_at_final_mean": float(df["gold_at_final"].mean()),

        "regret_secondary_mean": float((df["gold_at_gold_best"] - df["gold_at_secondary_best"]).mean()),
        "regret_final_mean": float((df["gold_at_gold_best"] - df["gold_at_final"]).mean()),

        "frac_oracle_improvement_secondary_mean": frac_sec_mean,
        "frac_oracle_improvement_final_mean": frac_final_mean,
        "n_runs_with_positive_gain": int(mask.sum()),

        "spearman_rho_mean": float(df["spearman_gold_secondary"].mean(skipna=True)),
        "spearman_rho_median": float(df["spearman_gold_secondary"].median(skipna=True)),
    }


def make_plots(per_run, agg, out_dir, args):
    """Save three slide-ready bar charts to out_dir."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_run)

    # Consistent palette across plots.
    C_ORACLE = "#2ca02c"
    C_SECONDARY = "#1f77b4"
    C_FINAL = "#d62728"
    C_FIRST = "#7f7f7f"

    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 13,
        "figure.dpi": 140,
        "savefig.dpi": 200,
    })

    subtitle = (f"N={agg['n_runs']} runs, "
                f"min {args.min_checkpoints} checkpoints/run, "
                f"{agg['hacking_observed_pct']:.0f}% show reward hacking")

    def finish(fig, ax, main_title):
        # Two-line title: bold main + smaller subtitle. Reserve top space.
        fig.suptitle(main_title, fontsize=15, fontweight="bold", y=0.98)
        ax.set_title(subtitle, fontsize=10, color="#555", pad=6)
        fig.tight_layout(rect=(0, 0, 1, 0.93))

    # --- Plot 1: fraction of oracle gold-improvement recovered -------------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    vals = [agg["frac_oracle_improvement_secondary_mean"],
            agg["frac_oracle_improvement_final_mean"]]
    bars = ax.bar(["secondary-RM\n(held-out)", "final checkpoint\n(no stopping)"],
                  vals, color=[C_SECONDARY, C_FINAL], width=0.55)
    ax.axhline(1.0, color=C_ORACLE, linewidth=2, linestyle="--",
               label="oracle (gold-best)")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of oracle gold-improvement recovered")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=13)
    ax.legend(loc="lower right", frameon=False)
    finish(fig, ax, "Held-out RM recovers near-oracle gold reward")
    fig.savefig(out_dir / "01_oracle_improvement_recovered.png")
    plt.close(fig)

    # --- Plot 2: mean gold reward by selection rule -----------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    labels = ["first\ncheckpoint", "final\ncheckpoint",
              "secondary-RM\nbest", "gold-best\n(oracle)"]
    vals = [agg["gold_at_first_mean"], agg["gold_at_final_mean"],
            agg["gold_at_secondary_best_mean"], agg["gold_at_gold_best_mean"]]
    colors = [C_FIRST, C_FINAL, C_SECONDARY, C_ORACLE]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_ylabel("Mean gold reward across runs")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08,
                f"{v:.2f}", ha="center", fontsize=13)
    ax.set_ylim(0, max(vals) * 1.18)
    finish(fig, ax, "Gold reward by checkpoint-selection rule")
    fig.savefig(out_dir / "02_gold_reward_by_rule.png")
    plt.close(fig)

    # --- Plot 3: selection accuracy (top-k hit rate) ----------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    groups = ["exact match\n(top-1)", "top-3 by gold", "top-5 by gold"]
    sec = [agg["exact_match_secondary_pct"],
           agg["top3_match_secondary_pct"],
           agg["top5_match_secondary_pct"]]
    fin = [agg["exact_match_final_pct"],
           agg["top3_match_final_pct"],
           agg["top5_match_final_pct"]]
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w / 2, sec, w, color=C_SECONDARY, label="secondary-RM-best")
    b2 = ax.bar(x + w / 2, fin, w, color=C_FINAL, label="final checkpoint")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("% of runs where rule picks a top-k gold checkpoint")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="upper left")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 2,
                    f"{v:.0f}%", ha="center", fontsize=12)
    finish(fig, ax, "Held-out RM picks a near-optimal checkpoint far more often")
    fig.savefig(out_dir / "03_selection_accuracy.png")
    plt.close(fig)

    print(f"[plots] wrote 3 PNGs to {out_dir}", file=sys.stderr)


def print_report(agg, args):
    p = print
    p("=== Secondary-RM checkpoint-selection effectiveness ===")
    p(f"dataset:          {args.dataset}")
    p(f"secondary RM:     {args.secondary_rm}")
    p(f"gold metric:      {args.gold_metric}")
    p(f"secondary metric: {args.secondary_metric}")
    p(f"min checkpoints:  {args.min_checkpoints}")
    p("")
    p(f"runs included: {agg['n_runs']}")
    p(f"checkpoints/run: mean {agg['n_ckpts_mean']:.1f}, "
      f"median {agg['n_ckpts_median']:.0f}, "
      f"min {agg['n_ckpts_min']}, max {agg['n_ckpts_max']}")
    p(f"runs where gold peaks before final ckpt: {agg['hacking_observed_pct']:.1f}%")
    p("")
    p("# Checkpoint-selection accuracy vs gold-best")
    p(f"  {'metric':<28} {'secondary':>12} {'final':>12}")
    p(f"  {'exact match':<28} "
      f"{agg['exact_match_secondary_pct']:>11.1f}% "
      f"{agg['exact_match_final_pct']:>11.1f}%")
    p(f"  {'in top-3 by gold':<28} "
      f"{agg['top3_match_secondary_pct']:>11.1f}% "
      f"{agg['top3_match_final_pct']:>11.1f}%")
    p(f"  {'in top-5 by gold':<28} "
      f"{agg['top5_match_secondary_pct']:>11.1f}% "
      f"{agg['top5_match_final_pct']:>11.1f}%")
    p(f"  {'mean gold-rank of pick':<28} "
      f"{agg['rank_secondary_mean']:>12.2f} "
      f"{agg['rank_final_mean']:>12.2f}")
    p(f"  {'median gold-rank of pick':<28} "
      f"{agg['rank_secondary_median']:>12.1f} "
      f"{agg['rank_final_median']:>12.1f}")
    p(f"  {'mean ckpt distance (steps)':<28} "
      f"{agg['ckpt_distance_secondary_mean']:>12.1f} "
      f"{agg['ckpt_distance_final_mean']:>12.1f}")
    p(f"  {'median ckpt distance':<28} "
      f"{agg['ckpt_distance_secondary_median']:>12.0f} "
      f"{agg['ckpt_distance_final_median']:>12.0f}")
    p("")
    p("# Gold reward, mean across runs")
    p(f"  at first checkpoint:        {agg['gold_at_first_mean']:.4f}")
    p(f"  at gold-best (oracle):      {agg['gold_at_gold_best_mean']:.4f}")
    p(f"  at secondary-best:          {agg['gold_at_secondary_best_mean']:.4f}")
    p(f"  at final checkpoint:        {agg['gold_at_final_mean']:.4f}")
    p("")
    p("# Regret vs oracle  (gold_at_gold_best - X)")
    p(f"  secondary-selected: {agg['regret_secondary_mean']:.4f}")
    p(f"  final-checkpoint:   {agg['regret_final_mean']:.4f}")
    p("")
    p(f"# Fraction of oracle gold-improvement recovered "
      f"(over {agg['n_runs_with_positive_gain']} runs with positive gain)")
    p(f"  secondary-selected: {agg['frac_oracle_improvement_secondary_mean']:.3f}")
    p(f"  final-checkpoint:   {agg['frac_oracle_improvement_final_mean']:.3f}")
    p("")
    p("# Spearman rho(gold curve, secondary curve), per run")
    p(f"  mean {agg['spearman_rho_mean']:.3f}, "
      f"median {agg['spearman_rho_median']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--secondary-rm", default=DEFAULT_SECONDARY_RM)
    parser.add_argument("--min-checkpoints", type=int, default=10)
    parser.add_argument("--gold-metric", default=DEFAULT_GOLD_METRIC)
    parser.add_argument("--secondary-metric", default=DEFAULT_SECONDARY_METRIC)
    parser.add_argument("--wandb-project", default=DEFAULT_ENTITY_PROJECT,
                        help="Full entity/project path on W&B.")
    parser.add_argument("--refetch", action="store_true",
                        help="Re-fetch history from W&B even if cached locally.")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Don't talk to W&B; only use cached history. "
                             "Runs without a cache are skipped.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to dump per-run details as CSV.")
    parser.add_argument("--plots-dir", type=Path, default=None,
                        help="If set, write slide-ready PNG bar charts here.")
    args = parser.parse_args()

    if not SUMMARY_PATH.exists():
        print(f"missing summary cache: {SUMMARY_PATH}\n"
              f"run scripts/download_run_data.py first.", file=sys.stderr)
        return 2

    summary_runs = json.loads(SUMMARY_PATH.read_text())
    filtered = filter_summary(summary_runs, args.dataset, args.secondary_rm,
                              args.gold_metric, args.secondary_metric)
    print(f"[filter] {len(filtered)} runs match dataset + secondary_rm + "
          f"have both metrics in summary", file=sys.stderr)
    if not filtered:
        return 1

    api = None
    if not args.no_fetch:
        import wandb
        api = wandb.Api(timeout=60)

    per_run = []
    skipped_short = 0
    skipped_err = 0
    for i, r in enumerate(filtered):
        rid = r["id"]
        try:
            rows = load_or_fetch_history(
                api, args.wandb_project, rid,
                args.gold_metric, args.secondary_metric,
                refetch=args.refetch,
            )
        except Exception as e:
            print(f"  ! {rid} ({r.get('name')}): {e}", file=sys.stderr)
            skipped_err += 1
            continue
        if rows is None:
            skipped_err += 1
            continue
        df = collapse_history(rows, args.gold_metric, args.secondary_metric)
        if len(df) < args.min_checkpoints:
            skipped_short += 1
            continue
        m = per_run_metrics(df)
        m["run_id"] = rid
        m["run_name"] = r.get("name")
        per_run.append(m)
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(filtered)}", file=sys.stderr)

    print(f"[keep] {len(per_run)} runs (skipped {skipped_short} short, "
          f"{skipped_err} errored)", file=sys.stderr)
    if not per_run:
        print("No runs survive the filter; nothing to report.")
        return 1

    agg = aggregate(per_run)
    print_report(agg, args)

    if args.csv:
        pd.DataFrame(per_run).to_csv(args.csv, index=False)
        print(f"\n[csv] wrote per-run details to {args.csv}", file=sys.stderr)
    if args.plots_dir:
        make_plots(per_run, agg, args.plots_dir, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
