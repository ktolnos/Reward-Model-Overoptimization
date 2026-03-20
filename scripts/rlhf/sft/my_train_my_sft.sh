#!/bin/bash

#SBATCH --job-name=train_sft
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=8:00:00
#SBATCH --qos=high

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
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    fi
fi
if [[ ! -f "${REPO_ROOT}/AGENTS.md" ]]; then
    echo "ERROR: Could not resolve repo root (got '${REPO_ROOT}')." >&2
    exit 1
fi

# Go to the root of the repo
cd "${REPO_ROOT}" || exit

log_dir="${REPO_ROOT}/scripts/rlhf/logs_sft/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
base_model_name="Qwen/Qwen3-1.7B-Base"
dataset_path="ktolnos/helpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

gpu=0

# Argument parsing
COMMIT_MSG=$(git log -1 --pretty=%s)
DEFAULT_WANDB_NAME="sft_${COMMIT_MSG// /_}_${SLURM_JOB_ID}"
wandb_name="${DEFAULT_WANDB_NAME}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) wandb_name="$2_${SLURM_JOB_ID}"; shift ;;
        *) ;;
    esac
    shift
done

# Port selection
PORT_SELECTOR_SCRIPT="${REPO_ROOT}/scripts/common/select_master_port.sh"

if ! MASTER_PORT="$(bash "${PORT_SELECTOR_SCRIPT}" 9900 9999)"; then
    exit 1
fi
export MASTER_PORT

export WANDB_PROJECT="sft"
export WANDB_RUN_NAME=${wandb_name}

# Create log directory
mkdir -p "${log_dir}"

echo "Logging to ${log_dir}"

CUDA_VISIBLE_DEVICES=${gpu} accelerate launch \
    --mixed_precision bf16 \
    --main_process_port ${MASTER_PORT} \
    rlhf/sft/my_sft.py \
    --model_name_or_path ${base_model_name} \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --eval_strategy "steps" \
    --eval_steps 0.05 \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_only_model True \
    --learning_rate 1e-5 \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --report_to "wandb" \
    --run_name ${wandb_name} \
    --length_config "default" \
    --trust_remote_code True || exit 1

echo "running evaluation script for checkpoints in ${log_dir}"
sbatch --export=ALL "${REPO_ROOT}/evaluate_policy.sh" --run_name "${wandb_name}" --kl_base_model_path "${base_model_name}" --checkpoint "${log_dir}"
