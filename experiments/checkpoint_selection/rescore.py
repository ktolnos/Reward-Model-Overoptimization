"""Re-score preserved eval generations with each selector / gold RM.

Run on the cluster (needs the local RM checkpoints + the JSONL eval files +
a GPU). One RM is loaded at a time and ``scores.parquet`` is appended after
each RM finishes, so killed jobs are resumable.

Output schema (long-form):
    grpo_run_id   : str
    grpo_run_idx  : int
    checkpoint    : int
    prompt_hash   : str
    prompt        : str
    rm_label      : str  ∈ {training_rm, sibling_rm, secondary_rm, gold_rm}
    score         : float

Also writes ``prompt_partition.json`` ({prompt_hash: "A"|"B"}) on first run.

Usage (entry point for the sbatch wrapper):
    python -m experiments.checkpoint_selection.rescore \\
        --eval_root /nas/ucb/eop/Reward-Model-Overoptimization \\
        --output_dir experiments/checkpoint_selection \\
        --rms training_rm,sibling_rm,secondary_rm,gold_rm \\
        --batch_size 1
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer

# Allow `python experiments/checkpoint_selection/rescore.py` to work.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from experiments.checkpoint_selection import manifest as M  # noqa: E402
    from experiments.checkpoint_selection.ingest import (  # noqa: E402
        EvalRow,
        ab_bucket,
        build_prompt_messages_index,
        load_eval_rows,
    )
else:
    from . import manifest as M
    from .ingest import (
        EvalRow,
        ab_bucket,
        build_prompt_messages_index,
        load_eval_rows,
    )

from policy_eval.rewards import LoadedRewardModels, score_responses_with_rm


# ---------------------------------------------------------------------------
# Output assembly helpers
# ---------------------------------------------------------------------------

def write_partition(rows: List[EvalRow], path: str) -> None:
    partition: Dict[str, str] = {}
    for r in rows:
        if r.prompt_hash not in partition:
            partition[r.prompt_hash] = ab_bucket(r.prompt_hash)
    with open(path, "w") as f:
        json.dump(partition, f, indent=2, sort_keys=True)
    a = sum(v == "A" for v in partition.values())
    b = len(partition) - a
    print(f"[partition] wrote {len(partition)} prompts ({a} A / {b} B) → {path}")


def already_done_labels(scores_path: str) -> set:
    """Which rm_labels already have rows in scores.parquet."""
    if not os.path.isfile(scores_path):
        return set()
    try:
        df = pd.read_parquet(scores_path, columns=["rm_label"])
    except Exception as e:
        print(f"[rescore] failed to peek existing scores ({e}); ignoring")
        return set()
    return set(df["rm_label"].unique().tolist())


def append_parquet(df_new: pd.DataFrame, scores_path: str) -> None:
    if os.path.isfile(scores_path):
        df_old = pd.read_parquet(scores_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(scores_path, index=False)
    print(f"[rescore] wrote {len(df_new)} new rows; total now {len(df)} → {scores_path}")


# ---------------------------------------------------------------------------
# Per-RM scoring pass
# ---------------------------------------------------------------------------

@dataclass
class ScoringInputs:
    """Indexed view of (rows × hash → prompt_messages) for one scoring pass."""
    rows: List[EvalRow]
    prompt_messages_per_row: List[list]


def assemble_scoring_inputs(
    rows: List[EvalRow], hash_to_messages: Dict[str, list]
) -> ScoringInputs:
    keep: List[EvalRow] = []
    msgs: List[list] = []
    missing = 0
    for r in rows:
        m = hash_to_messages.get(r.prompt_hash)
        if m is None:
            missing += 1
            continue
        keep.append(r)
        msgs.append(m)
    if missing:
        print(
            f"[rescore] WARNING: {missing}/{len(rows)} eval rows had no matching "
            f"dataset prompt; these are the full-only prompts in runs 6/7 and "
            f"are dropped as per plan."
        )
    return ScoringInputs(rows=keep, prompt_messages_per_row=msgs)


def score_one_rm(
    label: str,
    rm_path: str,
    inputs: ScoringInputs,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    fake_args = SimpleNamespace(
        device=device,
        batch_size=batch_size,
        output_file="",
        gold_rm_name=rm_path,
        training_rm_path=rm_path,
        sibling_rm_path=rm_path,
        secondary_rm_name=rm_path,
    )
    loaded = LoadedRewardModels(fake_args, {label})
    try:
        model, tokenizer = loaded.get(label)
        # Score in one shot — score_responses_with_rm batches internally.
        responses = [r.response for r in inputs.rows]
        scores = score_responses_with_rm(
            responses,
            inputs.prompt_messages_per_row,
            model,
            tokenizer,
            batch_size=batch_size,
            device=device,
            checkpoint_num=f"rescore:{label}",
        )
        df = pd.DataFrame({
            "grpo_run_id":  [r.grpo_run_id for r in inputs.rows],
            "grpo_run_idx": [r.grpo_run_idx for r in inputs.rows],
            "checkpoint":   [r.checkpoint for r in inputs.rows],
            "prompt_hash":  [r.prompt_hash for r in inputs.rows],
            "prompt":       [r.prompt for r in inputs.rows],
            "rm_label":     label,
            "score":        scores.astype(float),
        })
        return df
    finally:
        loaded.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", default=M.CLUSTER_ROOT,
                        help="Directory holding the evaluation_dataset_*.jsonl files.")
    parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Where to write scores.parquet, prompt_partition.json.")
    parser.add_argument("--rms", default=",".join(M.RM_LOAD_ORDER),
                        help=f"Comma-separated subset of {list(M.RM_PATHS)}. "
                             f"Default: load in M.RM_LOAD_ORDER.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-RM forward batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy_tokenizer", default=M.POLICY_TOKENIZER_NAME,
                        help="Tokenizer used to re-format prompts to match eval-file prompt strings.")
    parser.add_argument("--only_runs", default=None,
                        help="Comma-separated wandb run ids; default = all 8.")
    parser.add_argument("--skip_done", action="store_true", default=True,
                        help="Skip RM labels that already have rows in scores.parquet.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, M.SCORES_FILENAME)
    partition_path = os.path.join(args.output_dir, M.PARTITION_FILENAME)

    # ---- select runs -------------------------------------------------------
    runs = list(M.RUNS)
    if args.only_runs:
        wanted = {x.strip() for x in args.only_runs.split(",")}
        runs = [r for r in runs if r.wandb_id in wanted]
        if not runs:
            raise SystemExit(f"no runs matched --only_runs={args.only_runs!r}")

    # ---- ingest preserved eval rows ----------------------------------------
    print(f"\n=== ingest eval rows from {args.eval_root} ===")
    rows = load_eval_rows(args.eval_root, runs)
    print(f"[ingest] total preference rows: {len(rows)}")

    # ---- prompt partition (written once) -----------------------------------
    if not os.path.isfile(partition_path):
        write_partition(rows, partition_path)
    else:
        print(f"[partition] reusing existing {partition_path}")

    # ---- build hash → prompt_messages from source HF datasets --------------
    print("\n=== build prompt-messages lookup from HF datasets ===")
    policy_tok = AutoTokenizer.from_pretrained(
        args.policy_tokenizer, trust_remote_code=True,
    )
    hash_to_messages = build_prompt_messages_index(policy_tok)
    print(f"[ingest] hash→messages lookup has {len(hash_to_messages)} entries")

    inputs = assemble_scoring_inputs(rows, hash_to_messages)
    print(f"[rescore] will score {len(inputs.rows)} rows × len(rms) RMs")

    # ---- per-RM scoring pass ----------------------------------------------
    selected = [x.strip() for x in args.rms.split(",") if x.strip()]
    for label in selected:
        if label not in M.RM_PATHS:
            raise SystemExit(f"unknown rm_label {label!r}; known={list(M.RM_PATHS)}")

    done = already_done_labels(scores_path) if args.skip_done else set()
    if done:
        print(f"[rescore] skipping already-done labels: {sorted(done)}")

    for label in selected:
        if label in done:
            continue
        print(f"\n=== rescore with {label} ({M.RM_PATHS[label]}) ===")
        df = score_one_rm(
            label=label,
            rm_path=M.RM_PATHS[label],
            inputs=inputs,
            batch_size=args.batch_size,
            device=args.device,
        )
        append_parquet(df, scores_path)

    print("\n[rescore] done")


if __name__ == "__main__":
    main()
