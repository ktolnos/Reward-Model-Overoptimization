"""Fetch per-step training reward from wandb for each GRPO run.

Run this LOCALLY (it needs wandb credentials/cache) and commit/copy
``train_reward.parquet`` to the cluster before launching the sbatch.

Output schema (one row per (run, step)):
    grpo_run_id   : str  (wandb id, e.g. "q77u6k01")
    grpo_run_idx  : int  (1..8 from the manifest)
    step          : int  (wandb _step)
    train_reward  : float

The alignment to saved checkpoint steps is done by the analyzer, not here —
keep this script simple and resumable.

Usage:
    venv/bin/python -m experiments.checkpoint_selection.fetch_train_rewards \\
        [--output_dir experiments/checkpoint_selection]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd
import wandb

# Allow ``python experiments/checkpoint_selection/fetch_train_rewards.py``
# in addition to the ``-m`` form.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from experiments.checkpoint_selection import manifest as M  # noqa: E402
else:
    from . import manifest as M


def fetch_one(api: wandb.Api, run_spec) -> pd.DataFrame:
    print(
        f"[{run_spec.idx}] fetching wandb history for {run_spec.wandb_id} "
        f"({run_spec.name}) key={run_spec.reward_key}"
    )
    path = f"{M.WANDB_PROJECT}/{run_spec.wandb_id}"
    try:
        run = api.run(path)
    except Exception as e:
        raise RuntimeError(f"could not load wandb run {path}: {e}") from e

    df = run.history(keys=["_step", run_spec.reward_key], samples=100000, pandas=True)
    if df is None or df.empty:
        # Fall back to fetching all keys — some older runs need this.
        df = run.history(samples=100000, pandas=True)
    if df is None or df.empty:
        raise RuntimeError(f"empty wandb history for {path}")

    if run_spec.reward_key not in df.columns:
        raise RuntimeError(
            f"wandb run {path} has no column {run_spec.reward_key!r}; "
            f"available: {sorted(df.columns)[:20]}..."
        )

    df = df[["_step", run_spec.reward_key]].dropna()
    df["_step"] = df["_step"].astype(int)
    out = pd.DataFrame({
        "grpo_run_id": run_spec.wandb_id,
        "grpo_run_idx": run_spec.idx,
        "step": df["_step"].values,
        "train_reward": df[run_spec.reward_key].astype(float).values,
    }).sort_values("step").reset_index(drop=True)
    print(f"  → {len(out)} rows; reward range [{out['train_reward'].min():.4f}, {out['train_reward'].max():.4f}]")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Where to write train_reward.parquet (default: this script's directory).",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated wandb run ids; default = all 8.",
    )
    args = parser.parse_args()

    runs: List = list(M.RUNS)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        runs = [r for r in runs if r.wandb_id in wanted]
        if not runs:
            raise SystemExit(f"no runs matched --only={args.only!r}")

    api = wandb.Api()
    pieces = [fetch_one(api, r) for r in runs]
    df = pd.concat(pieces, ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, M.TRAIN_REWARD_FILENAME)
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df)} rows from {df['grpo_run_id'].nunique()} runs → {out_path}")


if __name__ == "__main__":
    main()
