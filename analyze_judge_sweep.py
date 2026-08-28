"""Analyse a `vector_judge_probe.py` sweep: rank judges and test the winner.

Reads the per-game dumps a sweep wrote (``--dump_dir``) and answers the two
questions a judge choice turns on:

  1. Is each judge better than a coin flip at all?
  2. Is the best judge significantly better than *every* other candidate, and its
     chosen thinking mode better than the alternative?

Every model judges the same prompt sample, so comparisons are **paired**: each
bootstrap resample draws prompts, not judgments, which cancels prompt difficulty
and is far tighter than treating two runs as independent.

Comparing the winner against N rivals is N simultaneous tests, so the family-wise
error rate is controlled with a Holm-Bonferroni step-down over the paired
bootstrap p-values -- otherwise "significant vs all 5" is roughly a 1-in-4 coin
flip at alpha=0.05 rather than 1-in-20.

Usage:
    python analyze_judge_sweep.py --sweep_dir <dir with dumps/ and results.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from policy_eval.judges import battles_from_game_labels

N_BOOT = 20000
SEED = 0


def per_prompt_agreement(dump_path: str) -> np.ndarray:
    """Mean battle score per prompt; NaN where the judge gave no usable verdict.

    A battle score of 1.0 means the judge picked the dataset's ``chosen``
    response, so the mean over prompts is agreement with the human label. The
    prompt -- not the individual battle -- is the independent unit here, which is
    what makes the paired bootstrap below valid.
    """
    rows = [json.loads(l) for l in open(dump_path)]
    by_idx: Dict[int, Dict[int, Optional[str]]] = {}
    for r in rows:
        by_idx.setdefault(r["prompt_index"], {})[r["game"]] = r["label"]
    n = max(by_idx) + 1
    out = np.full(n, np.nan)
    for i in range(n):
        g0, g1 = by_idx.get(i, {}).get(0), by_idx.get(i, {}).get(1)
        battles, _ = battles_from_game_labels([g0], [g1], weight=3)
        if battles and battles[0]:
            out[i] = float(np.mean(battles[0]))
    return out


def prompt_languages(dataset_name: str, split: str, n: int) -> List[str]:
    """``language`` per prompt index, in the loader's order.

    Reproduces ``_load_preference_split``'s transformation exactly (dedupe ->
    shuffle(seed=42) -> select) so index i here is index i in the judge dumps.
    Safe only because that path filters no rows -- verify the probe reports the
    full n ("Loaded n pairs") before trusting the join.
    """
    from datasets import load_dataset

    from data_utils import dedupe_dataset_by_prompt

    ds = load_dataset(dataset_name)[split]
    ds = dedupe_dataset_by_prompt(ds).shuffle(seed=42).select(range(n))
    return list(ds["language"])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """Paired bootstrap of ``mean(a - b)`` over the prompts both judges decided."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = a[ok] - b[ok]
    if len(d) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": 1.0, "n": 0}
    boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided bootstrap p: how often the resampled difference crosses zero.
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, p)), "n": int(len(d))}


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone-enforced)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def bootstrap_ci(x: np.ndarray, rng) -> tuple:
    x = x[~np.isnan(x)]
    boot = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return float(x.mean()), *np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True,
                    help="directory holding results.jsonl and dumps/<model>/<mode>.jsonl")
    ap.add_argument("--mode", default="no_thinking",
                    help="mode to rank the models on")
    ap.add_argument("--by_language", action="store_true",
                    help="break agreement out by English vs non-English (HelpSteer3 "
                         "is largely non-English, and the eval runs on it)")
    ap.add_argument("--dataset", default="ktolnos/helpsteer3-qwen35_annotated_human",
                    help="dataset the sweep ran on (for --by_language)")
    ap.add_argument("--split", default="validation", help="split (for --by_language)")
    ap.add_argument("--thinking_model", default="",
                    help="model to run the thinking-vs-no-thinking comparison on "
                         "(default: the top-ranked model). The thinking run is often "
                         "done on one candidate only, since it costs ~10x.")
    args = ap.parse_args()

    results = [json.loads(l) for l in open(os.path.join(args.sweep_dir, "results.jsonl"))]
    by_key = {(r["model"], r["mode"]): r for r in results}
    rng = np.random.default_rng(SEED)

    models = [m for (m, mode) in by_key if mode == args.mode]
    scores = {m: per_prompt_agreement(
        os.path.join(args.sweep_dir, "dumps", m, f"{args.mode}.jsonl")) for m in models}
    models.sort(key=lambda m: -np.nanmean(scores[m]))
    best = models[0]
    n_prompts = len(scores[best])

    print(f"\n{'=' * 78}\nJUDGE SWEEP  ({args.mode}, {n_prompts} prompts, "
          f"{len(models)} models)\n{'=' * 78}\n")

    # ---- headline table -------------------------------------------------
    print(f"{'model':<32}{'agree':>7}{'95% CI':>18}{'flip':>7}{'drop':>8}{'med lat':>9}")
    for m in models:
        r = by_key[(m, args.mode)]
        mean, lo, hi = bootstrap_ci(scores[m], rng)
        print(f"{m:<32}{mean:>7.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{r['controversial_rate']:>7.3f}{r['dropped']:>5}/{r['n_prompts']}"
              f"{r['median_latency']:>9.1f}s")

    print("\nBetter than chance (agreement CI excludes 0.50)?")
    for m in models:
        mean, lo, hi = bootstrap_ci(scores[m], rng)
        print(f"  {m:<32}{'YES' if lo > 0.5 else 'NO -- judges at chance'}")

    # ---- winner vs every rival, Holm-corrected --------------------------
    rivals = models[1:]
    tests = [paired_bootstrap(scores[best], scores[m], rng) for m in rivals]
    adj = holm([t["p"] for t in tests])
    print(f"\n{'-' * 78}\nPaired bootstrap: {best} vs each rival "
          f"({N_BOOT} resamples, Holm-corrected)\n{'-' * 78}")
    print(f"{'rival':<32}{'delta':>8}{'95% CI':>18}{'p':>9}{'p_holm':>9}{'':>6}")
    all_sig = True
    for m, t, pa in zip(rivals, tests, adj):
        sig = pa < 0.05 and t["lo"] > 0
        all_sig &= sig
        ci = f"[{t['lo']:+.3f}, {t['hi']:+.3f}]"
        print(f"{m:<32}{t['delta']:>+8.3f}{ci:>18}"
              f"{t['p']:>9.4f}{pa:>9.4f}{'  SIG' if sig else '  n.s.':>6}")
    print(f"\n=> {best} is {'SIGNIFICANTLY better than ALL rivals' if all_sig else 'NOT significantly better than every rival'} "
          f"(Holm, alpha=0.05).")

    # ---- English vs non-English -----------------------------------------
    # HelpSteer3 is largely non-English and the eval runs on it, so a judge that
    # is strong only in English would be the wrong pick regardless of its
    # aggregate score.
    if args.by_language:
        langs = np.array(prompt_languages(args.dataset, args.split, n_prompts))
        is_en = langs == "english"
        print(f"\n{'-' * 78}\nEnglish ({is_en.sum()}) vs non-English "
              f"({(~is_en).sum()}) agreement\n{'-' * 78}")
        print(f"{'model':<32}{'English':>10}{'non-Eng':>10}{'delta':>9}{'95% CI':>18}")
        for m in models:
            s = scores[m]
            en, non = s[is_en], s[~is_en]
            # Unpaired here: different prompts, so bootstrap each side.
            d = np.nanmean(en) - np.nanmean(non)
            be = np.array([np.nanmean(en[rng.integers(0, len(en), len(en))])
                           for _ in range(2000)])
            bn = np.array([np.nanmean(non[rng.integers(0, len(non), len(non))])
                           for _ in range(2000)])
            lo, hi = np.percentile(be - bn, [2.5, 97.5])
            print(f"{m:<32}{np.nanmean(en):>10.3f}{np.nanmean(non):>10.3f}"
                  f"{d:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>18}")

    # ---- thinking vs no-thinking for the winner -------------------------
    other = "thinking" if args.mode == "no_thinking" else "no_thinking"
    tm = args.thinking_model or best
    path = os.path.join(args.sweep_dir, "dumps", tm, f"{other}.jsonl")
    if os.path.exists(path) and tm in scores:
        alt = per_prompt_agreement(path)
        n = min(len(alt), len(scores[tm]))
        # The smaller run is a prefix of the larger (shuffle(seed=42) + select),
        # so truncating to n keeps the pairs aligned.
        t = paired_bootstrap(scores[tm][:n], alt[:n], rng)
        ra, rb = by_key[(tm, args.mode)], by_key.get((tm, other))
        print(f"\n{'-' * 78}\nThinking mode for {tm} (paired on {t['n']} prompts)\n{'-' * 78}")
        print(f"{'mode':<16}{'agree':>8}{'flip':>8}{'drop':>10}{'med lat':>10}{'games/min':>11}")
        for label, r, s in ((args.mode, ra, scores[tm][:n]), (other, rb, alt[:n])):
            if r is None:
                continue
            print(f"{label:<16}{np.nanmean(s):>8.3f}{r['controversial_rate']:>8.3f}"
                  f"{r['dropped']:>7}/{r['n_prompts']}{r['median_latency']:>9.1f}s"
                  f"{r['games_per_min']:>11.0f}")
        verdict = ("no significant difference" if t["lo"] <= 0 <= t["hi"]
                   else f"{args.mode} better" if t["delta"] > 0 else f"{other} better")
        print(f"\n  delta ({args.mode} - {other}) = {t['delta']:+.3f} "
              f"[{t['lo']:+.3f}, {t['hi']:+.3f}], p={t['p']:.4f}  =>  {verdict}")
        if rb is not None:
            speed = rb["median_latency"] / ra["median_latency"]
            print(f"  {other} costs {speed:.1f}x the latency of {args.mode}.")
    else:
        print(f"\n(no {other} run found for {tm})")


if __name__ == "__main__":
    main()
