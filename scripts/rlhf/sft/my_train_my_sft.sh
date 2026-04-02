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
# base_model_name="Qwen/Qwen3-8B-Base"
base_model_name="Qwen/Qwen3.5-4B-Base"
dataset_path="ktolnos/helpsteer3-qwen35_annotated_human"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

gpu=0
use_lora=false
base_learning_rate="1e-5"
lora_lr_multiplier=5  # LoRA typically needs higher LR

# Argument parsing
COMMIT_MSG=$(git log -1 --pretty=%s)
DEFAULT_WANDB_NAME="sft_${COMMIT_MSG// /_}"
if [[ "${use_lora}" == "true" ]]; then
    DEFAULT_WANDB_NAME="${DEFAULT_WANDB_NAME}_lora"
fi
DEFAULT_WANDB_NAME="${DEFAULT_WANDB_NAME}_${SLURM_JOB_ID}"
wandb_name="${DEFAULT_WANDB_NAME}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) wandb_name="$2_${SLURM_JOB_ID}"; shift ;;
        *) ;;
    esac
    shift
done

# Compute effective learning rate
if [[ "${use_lora}" == "true" ]]; then
    learning_rate=$(python3 -c "print(f'{${base_learning_rate} * ${lora_lr_multiplier}:.0e}')")
else
    learning_rate="${base_learning_rate}"
fi

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
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --eval_strategy "steps" \
    --eval_steps 0.25 \
    --save_strategy "steps" \
    --save_steps 0.25 \
    --save_only_model True \
    --learning_rate ${learning_rate} \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --report_to "wandb" \
    --run_name ${wandb_name} \
    --length_config "default" \
    --skip_length_validation True \
    --trust_remote_code True \
    $(if [[ "${use_lora}" == "true" ]]; then echo "\
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --lora_target_modules all-linear"; fi) || exit 1

echo "running evaluation script for checkpoints in ${log_dir}"
sbatch --export=ALL "${REPO_ROOT}/evaluate_policy.sh" --run_name "${wandb_name}" --kl_base_model_path "${base_model_name}" --checkpoint "${log_dir}"
