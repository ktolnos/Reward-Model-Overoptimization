#!/bin/bash
# Evaluate GRPO checkpoints with AlpacaFarm 7B gold RM (Phase 3).
# Scores policy generations on the AlpacaFarm eval split and computes
# KL divergence from the SFT policy.
#
# Usage:
#   sbatch evaluate_paper.sh --checkpoint <checkpoints_dir> --run_name <name> --kl_base_model_path <model>

#SBATCH --job-name=paper_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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

# ---- Defaults ----
CHECKPOINTS_DIR=""
WANDB_RUN_NAME="paper_eval_$(date +%Y%m%d_%H%M%S)"
KL_BASE_MODEL_PATH="tlc4418/pythia_1.4b_sft_policy"

# Gold RM: AlpacaFarm 7B human preference reward model
GOLD_RM_NAME="alpaca_farm_models/reward-model-human"

# Eval dataset: AlpacaFarm val split (2K prompts), converted to messages format
DATASET_NAME="ktolnos/alpacafarm_paper_eval_prompts"

export LD_PRELOAD="/nas/ucb/eop/.local/lib/libsqlite3.so.0"

# ---- Parse arguments ----
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINTS_DIR="$2"; shift ;;
        --run_name) WANDB_RUN_NAME="$2"; shift ;;
        --kl_base_model_path) KL_BASE_MODEL_PATH="$2"; shift ;;
        --gold_rm) GOLD_RM_NAME="$2"; shift ;;
        --dataset) DATASET_NAME="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "${CHECKPOINTS_DIR}" ]]; then
    echo "ERROR: --checkpoint is required"
    exit 1
fi

OUTPUT_FILE="evaluation_paper_${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).csv"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "=== Paper Comparison Evaluation ==="
echo "Checkpoints: ${CHECKPOINTS_DIR}"
echo "Gold RM: ${GOLD_RM_NAME}"
echo "Eval dataset: ${DATASET_NAME}"
echo "KL base model: ${KL_BASE_MODEL_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo ""

python evaluate_policy.py \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --gold_rm_name "${GOLD_RM_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --output_file "${OUTPUT_FILE}" \
    --length_config "alpacafarm_paper" \
    --kl_base_model_path "${KL_BASE_MODEL_PATH}" \
    --batch_size 1 \
    --generation_batch_size 32 \
    --max_length 520 \
    --max_new_tokens 256 \
    --device "cuda" \
    --wandb_project "paper-comparison-eval" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --evaluate_with_training_rm False \
    --evaluate_with_llm_judge False \
    --use_dataset_response_as_baseline False \
    --secondary_rm_name "none" \
    --skip_validation True \
    --save_eval_dataset_path "evaluation_paper_data_${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).json" \
    || exit 1

echo "Evaluation complete. Results in ${OUTPUT_FILE}"
