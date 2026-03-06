#!/bin/bash
# Train a Pythia-44M proxy RM on AlpacaFarm preference data (Phase 1b).
# Paper hyperparams: LR 1e-5, 5 epochs, batch 32, BT loss.
# Accepts --seed argument (default: 42). Run 5 times with seeds 1-5 for ensemble.
#
# Usage:
#   sbatch train_paper_rm.sh --seed 1
#   sbatch train_paper_rm.sh --seed 2
#   ... etc.

#SBATCH --job-name=paper_rm
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --qos=high

set -euo pipefail

REPO_ROOT=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/AGENTS.md" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_ROOT}"

# Defaults
SEED=42

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Paper's published SFT'd 70M model (RM base)
base_model="tlc4418/pythia_70m_sft"
dataset_name="ktolnos/alpacafarm_paper_preference_messages"  # converted via convert_paper_dataset.py

log_dir="${REPO_ROOT}/save_reward_models/paper_comparison"
wandb_name="paper_rm_pythia44m_seed${SEED}_${SLURM_JOB_ID:-local}"

export PYTHONPATH="${PWD}/reward_models:${PWD}:${PYTHONPATH:-}"
export WANDB_PROJECT="paper-comparison-rm"

echo "Training RM with seed=${SEED}"
echo "Base model: ${base_model}"
echo "Dataset: ${dataset_name}"
echo "Log dir: ${log_dir}"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    reward_models/run_reward_models_train.py \
    --base_model "${base_model}" \
    --dataset "${dataset_name}" \
    --wandb_name "${wandb_name}" \
    --log_dir "${log_dir}" \
    --seed "${SEED}" \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --use_lora False \
    --report_to "wandb" \
    --save_strategy "epoch" \
    --eval_strategy steps --eval_steps 0.02 \
    --bf16 True \
    || exit 1

echo "RM training complete (seed=${SEED}). Checkpoints in ${log_dir}"
