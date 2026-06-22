"""Hardcoded inputs for the checkpoint-selection experiment.

The 8 GRPO runs, their preserved eval-dataset JSONL files, and the RM
checkpoints used as selectors all live here as plain Python literals — see
i-have-a-lot-foamy-spindle.md for the source of these values.

Splitting between local and cluster runs:
- wandb fetch runs locally (uses RUN_IDS + REWARD_KEY).
- re-scoring runs on the cluster (uses EVAL_FILE_PATHS + RM_PATHS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


CLUSTER_ROOT = "/nas/ucb/eop/Reward-Model-Overoptimization"

DATASET_NAME = "ktolnos/helpsteer3-qwen35_annotated_human"
DATASET_NAME_25PCT = "ktolnos/helpsteer3-qwen35_annotated_human_25pct"
POLICY_TOKENIZER_NAME = "Qwen/Qwen3.5-4B-Base"

# Project under which the GRPO training runs live in wandb.
WANDB_PROJECT = "grpo"


@dataclass(frozen=True)
class RunSpec:
    idx: int
    name: str
    wandb_id: str
    variant: str             # "25pct" or "full"
    eval_file: str           # path under CLUSTER_ROOT
    reward_key: str          # wandb metric key for training-time mean reward


# rewards/batch_mean is the standard per-step mean reward; the 8-RM sequential
# run also logs an ensemble_mean which is equivalent for single-RM runs but
# semantically clearer when multiple RMs are active in the rotation.
_BATCH_MEAN = "rewards/batch_mean"
_ENSEMBLE_MEAN = "rewards/ensemble_mean"

RUNS: List[RunSpec] = [
    RunSpec(
        idx=1,
        name="dapo0.5-max1.5_KL0_1rms_sequential3x_1099677",
        wandb_id="q77u6k01",
        variant="25pct",
        eval_file="evaluation_dataset_20260416_153641_1099677_20260427_020109.jsonl",
        reward_key=_BATCH_MEAN,
    ),
    RunSpec(
        idx=2,
        name="0.6DAPO_squared_max1_KL0_1rms_sequential3x_1109219",
        wandb_id="qazyyejd",
        variant="25pct",
        eval_file="evaluation_dataset_20260424_172603_1109219_20260427_202454.jsonl",
        reward_key=_BATCH_MEAN,
    ),
    RunSpec(
        idx=3,
        name="0.6DAPO_linear_max1_KL0_1rms_sequential3x_1109220",
        wandb_id="a2gt0rey",
        variant="25pct",
        eval_file="evaluation_dataset_20260425_184521_1109220_20260426_210106.jsonl",
        reward_key=_BATCH_MEAN,
    ),
    RunSpec(
        idx=4,
        name="linear0.6-max1.5_KL0_1rms_sequential3x_1126524",
        wandb_id="0b18abv6",
        variant="25pct",
        eval_file="evaluation_dataset_20260430_193141_1126524_20260504_175120.jsonl",
        reward_key=_BATCH_MEAN,
    ),
    RunSpec(
        idx=5,
        name="0.6DAPO_max4_mask_KL0_1rms_sequential3x_1131216",
        wandb_id="x8j84qb5",
        variant="25pct",
        eval_file="evaluation_dataset_20260507_201409_1131216_20260508_130658.jsonl",
        reward_key=_BATCH_MEAN,
    ),
    RunSpec(
        idx=6,
        name="full_ds_max1024_KL0_1rms_sequential3x_1132879",
        wandb_id="m3o6zldy",
        variant="full",
        eval_file="evaluation_dataset_20260511_233813_1132879_20260513_002022.jsonl",
        reward_key=_BATCH_MEAN,
    ),
    RunSpec(
        idx=7,
        name="full_ds_5e-6lr_KL0_1rms_sequential3x_1136572",
        wandb_id="zkdx9lek",
        variant="full",
        eval_file="evaluation_dataset_20260514_184550_1136572_20260515_203006.jsonl",
        reward_key=_BATCH_MEAN,
    ),
    RunSpec(
        idx=8,
        name="same_seed_KL0_8rms_sequential3x_1129482",
        wandb_id="vu71m2rn",
        variant="25pct",
        # Run 8 cycles through 8 RM_19 checkpoints during training, so the
        # ensemble mean is the meaningful training-time signal.
        eval_file="evaluation_dataset_20260504_165538_1129482_20260505_153713.jsonl",
        reward_key=_ENSEMBLE_MEAN,
    ),
]


def runs_by_id() -> Dict[str, RunSpec]:
    return {r.wandb_id: r for r in RUNS}


def runs_by_idx() -> Dict[int, RunSpec]:
    return {r.idx: r for r in RUNS}


# ---------------------------------------------------------------------------
# Reward models used by the experiment.
# ---------------------------------------------------------------------------

# Label scheme matches policy_eval.rewards.LoadedRewardModels._LABEL_TO_ARG:
#   training_rm / sibling_rm / secondary_rm / gold_rm.
RM_PATHS: Dict[str, str] = {
    "training_rm": f"{CLUSTER_ROOT}/save_reward_models/19_Qwen3.5-4B-Base_len2048_fulltrain_2e-05_datahelpsteer3-qwen35_annotated_human/logs/checkpoint-3144",
    "sibling_rm":  f"{CLUSTER_ROOT}/save_reward_models/20_Qwen3.5-4B-Base_len2048_fulltrain_2e-05_datahelpsteer3-qwen35_annotated_human/logs/checkpoint-1179",
    "secondary_rm": "Ray2333/GRM-Gemma-2B-sftreg",
    "gold_rm":     "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
}

# Order matters: the rescoring script loads them one-at-a-time and the 8B gold
# RM dominates wall time, so unloading then loading it last is fine. The
# selector RMs (training/sibling) share a 4B base so they're roughly the same
# cost; secondary is the smallest.
RM_LOAD_ORDER: List[str] = ["secondary_rm", "sibling_rm", "training_rm", "gold_rm"]


# ---------------------------------------------------------------------------
# Output paths (all under experiments/checkpoint_selection/).
# ---------------------------------------------------------------------------

# These default to relative paths under the package directory but the
# rescoring/analyze entry points accept --output_dir overrides for the cluster.
DEFAULT_OUTPUT_SUBDIR = "experiments/checkpoint_selection"

SCORES_FILENAME = "scores.parquet"
TRAIN_REWARD_FILENAME = "train_reward.parquet"
PARTITION_FILENAME = "prompt_partition.json"
PER_RUN_METRICS_FILENAME = "per_run_strategy_metrics.csv"
AGGREGATE_FILENAME = "aggregate.json"
PLOTS_SUBDIR = "plots"


# Hash truncation length used everywhere we identify prompts.
PROMPT_HASH_LEN = 16
