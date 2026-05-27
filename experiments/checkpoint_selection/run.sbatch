#!/bin/bash

# Checkpoint-selection experiment: re-score preserved eval generations with
# four RMs, then run the analyzer. Assumes train_reward.parquet has already
# been produced locally (the cluster has no wandb cache, see
# experiments/checkpoint_selection/fetch_train_rewards.py).
#
# Usage:
#   sbatch experiments/checkpoint_selection/run.sbatch
#
# Override defaults via env vars:
#   EVAL_ROOT=/some/path RMS=gold_rm sbatch experiments/checkpoint_selection/run.sbatch

#SBATCH --job-name=ckpt_select
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --gres=gpu:1

set -euo pipefail

# --- repo root resolution (same convention as scripts/my_train_bt_rm_full.sh) -
REPO_ROOT=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/AGENTS.md" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${REPO_ROOT}" ]]; then
    if [[ -f "/nas/ucb/eop/Reward-Model-Overoptimization/AGENTS.md" ]]; then
        REPO_ROOT="/nas/ucb/eop/Reward-Model-Overoptimization"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
if [[ ! -f "${REPO_ROOT}/AGENTS.md" ]]; then
    echo "ERROR: Could not resolve repo root (got '${REPO_ROOT}')." >&2
    exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

EVAL_ROOT="${EVAL_ROOT:-/nas/ucb/eop/Reward-Model-Overoptimization}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/checkpoint_selection}"
RMS="${RMS:-secondary_rm,sibling_rm,training_rm,gold_rm}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RMS=${RMS}"

PYTHON="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python"
fi

# Cluster-side preflight: the wandb parquet must have been copied over.
if [[ ! -f "${OUTPUT_DIR}/train_reward.parquet" ]]; then
    echo "WARNING: ${OUTPUT_DIR}/train_reward.parquet not found. The"          \
         "train_reward strategy will be skipped. Run"                          \
         "fetch_train_rewards.py locally and copy the parquet over to enable." >&2
fi

# 1) Re-score every (run, ckpt, prompt) response under each RM (loads RMs one
#    at a time; appends to scores.parquet between RMs so it's resumable).
"${PYTHON}" -m experiments.checkpoint_selection.rescore  \
    --eval_root  "${EVAL_ROOT}"                         \
    --output_dir "${OUTPUT_DIR}"                        \
    --rms        "${RMS}"                               \
    --batch_size "${BATCH_SIZE}"

# 2) Analyze: per-run strategy CSV + aggregate.json + plots.
"${PYTHON}" -m experiments.checkpoint_selection.analyze \
    --output_dir "${OUTPUT_DIR}"

echo "Done. Outputs under ${OUTPUT_DIR}."
