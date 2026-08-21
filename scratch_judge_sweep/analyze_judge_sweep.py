#!/usr/bin/env python
"""Analyze a judge sweep: quality, agreement, checkpoint-ranking, speed.

Consumes ``<tag>_records.parquet`` + ``<tag>_stats.json`` from run_judge_sweep.py.

Per-pair verdict (policy perspective) = battle_mean thresholded at 0.5
(>0.5 win / ==0.5 tie / <0.5 loss), matching pairwise.compute_pairwise_metrics.
Gold verdict = sign(gold_policy - gold_baseline).

Metrics per config:
  - failures: parse / truncation / generation / dropped (from stats)
  - speed:    pairs/s and wall seconds (from stats)
  - gold agreement: fraction of decided pairs where judge direction == gold
    direction (ties on either side excluded); also with ties-count-as-half.
  - self-consistency: 1 - controversial_rate (position-swap agreement) --
    "agreement with itself" independent of temperature.
  - cross-config agreement: per-pair verdict match vs the reference config
    (default bf16 thinking), over pairs both configs decided.
  - checkpoint ranking: per-checkpoint policy win-rate; Spearman vs reference
    config and vs gold-RM win-rate.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def verdict(battle_mean):
    if pd.isna(battle_mean):
        return np.nan
    if battle_mean > 0.5:
        return 1.0
    if battle_mean < 0.5:
        return 0.0
    return 0.5


def gold_dir(gp, gb):
    if gp > gb:
        return 1.0
    if gp < gb:
        return 0.0
    return 0.5


def controversial_rate(sub):
    """Position-flip rate: judge named opposite decisive winners in the two
    swapped games => it followed answer position, not content."""
    label_dir = {"A>B": 1, "A>>B": 1, "B<A": 1, "B<<A": 1,
                 "A<B": -1, "A<<B": -1, "B>A": -1, "B>>A": -1,
                 "A=B": 0, "B=A": 0}
    n = c = 0
    for _, r in sub.iterrows():
        l0, l1 = r["game0_label"], r["game1_label"]
        if l0 not in label_dir or l1 not in label_dir:
            continue
        n += 1
        # policy-perspective winner per game: game1 A=policy so +dir; game0 A=baseline so -dir
        w1 = label_dir[l1]
        w0 = -label_dir[l0]
        if w0 and w1 and w0 != w1:
            c += 1
    return c / n if n else np.nan, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--reference", default="bf16__think=True")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(args.dir, f"{args.tag}_records.parquet"))
    with open(os.path.join(args.dir, f"{args.tag}_stats.json")) as f:
        stats = {s["config"]: s for s in json.load(f)}

    df["verdict"] = df["battle_mean"].map(verdict)
    df["gold"] = [gold_dir(p, b) for p, b in zip(df["gold_policy"], df["gold_baseline"])]

    configs = list(dict.fromkeys(df["config"]))
    ref = args.reference if args.reference in configs else configs[0]

    # Per-config, per-pair verdict pivot for cross-config agreement / ranking.
    key = ["checkpoint", "prompt_uid"]
    pivot = df.pivot_table(index=key, columns="config", values="verdict",
                           aggfunc="first")

    # Gold win-rate per checkpoint (reference ranking; identical across configs).
    gold_by_ck = df.groupby("checkpoint")["gold"].mean()

    rows = []
    for cfg in configs:
        sub = df[df["config"] == cfg]
        st = stats[cfg]
        n_pairs = st["n_pairs"]

        # gold agreement over decided pairs (both non-tie)
        dec = sub[(sub["verdict"].isin([0.0, 1.0])) & (sub["gold"].isin([0.0, 1.0]))]
        gold_agree = (dec["verdict"] == dec["gold"]).mean() if len(dec) else np.nan
        # soft agreement: |1 - |v-g|| averaged over pairs where verdict parsed
        parsed = sub[sub["verdict"].notna()]
        soft = (1 - (parsed["verdict"] - parsed["gold"]).abs()).mean() if len(parsed) else np.nan

        ctrl, n_ctrl = controversial_rate(sub)

        # cross-config agreement vs reference (pairs both decided, incl ties)
        if cfg == ref:
            xagree = 1.0
        else:
            both = pivot[[cfg, ref]].dropna()
            xagree = (both[cfg] == both[ref]).mean() if len(both) else np.nan

        # checkpoint ranking: policy win-rate per ckpt (ties=0.5)
        wr = sub.groupby("checkpoint")["verdict"].mean()
        common = wr.index.intersection(gold_by_ck.index)
        sp_gold = spearmanr(wr.loc[common], gold_by_ck.loc[common]).correlation
        if cfg == ref:
            sp_ref = 1.0
        else:
            wr_ref = df[df["config"] == ref].groupby("checkpoint")["verdict"].mean()
            cc = wr.index.intersection(wr_ref.index)
            sp_ref = spearmanr(wr.loc[cc], wr_ref.loc[cc]).correlation

        rows.append(dict(
            config=cfg,
            pairs_per_s=round(st["pairs_per_s"], 2),
            wall_s=round(st["elapsed_s"], 1),
            parse_fail=st["n_parse_failures"],
            trunc_fail=st["n_truncation_failures"],
            gen_fail=st["n_generation_failures"],
            dropped=st["n_dropped_prompts"],
            drop_rate=round(st["n_dropped_prompts"] / n_pairs, 3),
            gold_agree=round(gold_agree, 3),
            gold_soft=round(soft, 3),
            self_consist=round(1 - ctrl, 3) if not pd.isna(ctrl) else np.nan,
            agree_vs_ref=round(xagree, 3),
            rank_sp_gold=round(sp_gold, 3) if sp_gold == sp_gold else np.nan,
            rank_sp_ref=round(sp_ref, 3) if sp_ref == sp_ref else np.nan,
        ))

    out = pd.DataFrame(rows).set_index("config")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\nReference config: {ref}\n")
    print(out.to_string())

    print("\nPer-checkpoint policy win-rate by config:")
    ck_wr = df.pivot_table(index="checkpoint", columns="config", values="verdict")
    ck_wr["GOLD"] = gold_by_ck
    print(ck_wr.round(3).to_string())

    # Full pairwise cross-config agreement matrix (per-pair verdict match over
    # pairs both configs decided, ties included). Diagonal = 1.
    print("\nCross-config verdict-agreement matrix (fraction of shared pairs w/ same verdict):")
    mat = pd.DataFrame(index=configs, columns=configs, dtype=float)
    for a in configs:
        for b in configs:
            if a == b:
                mat.loc[a, b] = 1.0
                continue
            both = pivot[[a, b]].dropna()
            va, vb = both.iloc[:, 0], both.iloc[:, 1]
            mat.loc[a, b] = float((va == vb).mean()) if len(both) else np.nan
    print(mat.round(3).to_string())

    # Agreement with gold, restricted to the common thinking-subset prompts so
    # think/no-think are compared on the SAME pairs (fair when coverage differs).
    common_keys = pivot.dropna().index  # pairs every config decided
    print(f"\nGold agreement on the {len(common_keys)} pairs ALL configs decided:")
    dfk = df.set_index(key)
    for cfg in configs:
        s = dfk[dfk["config"] == cfg].loc[
            dfk[dfk["config"] == cfg].index.intersection(common_keys)]
        dec = s[(s["verdict"].isin([0.0, 1.0])) & (s["gold"].isin([0.0, 1.0]))]
        ga = (dec["verdict"] == dec["gold"]).mean() if len(dec) else np.nan
        print(f"  {cfg:24s} gold_agree={ga:.3f} (n_decided={len(dec)})")

    out.to_csv(os.path.join(args.dir, f"{args.tag}_summary.csv"))
    print(f"\nwrote {args.tag}_summary.csv")


if __name__ == "__main__":
    main()
