#!/bin/bash

#SBATCH --job-name=train_dpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=24:00:00
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

log_dir="${REPO_ROOT}/scripts/rlhf/logs_dpo/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"

# ---- Model & Dataset ----
# Point to the SFT checkpoint (output of my_train_my_sft.sh)
base_model_name="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/20260402_233303_1082435"  # 3.5 4B-base human
dataset_path="ktolnos/helpsteer3-qwen35_annotated_human"

export PYTHONPATH="${REPO_ROOT}/rlhf/dpo:${REPO_ROOT}:$PYTHONPATH"

gpu=0
use_lora=false
base_learning_rate="5e-7"
lora_lr_multiplier=5  # LoRA typically needs higher LR

# ---- DPO Hyperparameters ----
# loss_type options: sigmoid (standard DPO), apo_zero, apo_down, ipo, hinge
loss_type="sigmoid"
beta="0.1"              # KL penalty strength (use 0.05 for apo_zero)

# ---- Argument parsing ----
if [ -n "$LAST_COMMIT_MESSAGE" ]; then
    COMMIT_MSG="$LAST_COMMIT_MESSAGE"
else
    COMMIT_MSG=$(git log -1 --pretty=%s)
fi
DEFAULT_WANDB_NAME="dpo_${loss_type}_${COMMIT_MSG// /_}"
if [[ "${use_lora}" == "true" ]]; then
    DEFAULT_WANDB_NAME="${DEFAULT_WANDB_NAME}_lora"
fi
DEFAULT_WANDB_NAME="${DEFAULT_WANDB_NAME}_${SLURM_JOB_ID}"
wandb_name="${DEFAULT_WANDB_NAME}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) wandb_name="$2_${SLURM_JOB_ID}"; shift ;;
        --loss_type) loss_type="$2"; shift ;;
        --beta) beta="$2"; shift ;;
        --learning_rate) base_learning_rate="$2"; shift ;;
        --model) base_model_name="$2"; shift ;;
        --dataset) dataset_path="$2"; shift ;;
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

export WANDB_PROJECT="dpo"
export WANDB_RUN_NAME=${wandb_name}

# Create log directory
mkdir -p "${log_dir}"

echo "Logging to ${log_dir}"
echo "Loss type: ${loss_type}, beta: ${beta}, lr: ${learning_rate}"

CUDA_VISIBLE_DEVICES=${gpu} accelerate launch \
    --mixed_precision bf16 \
    --main_process_port ${MASTER_PORT} \
    rlhf/dpo/my_dpo.py \
    --model_name_or_path ${base_model_name} \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir} \
    --loss_type "${loss_type}" \
    --beta ${beta} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --eval_strategy "steps" \
    --eval_steps 0.1 \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_only_model True \
    --learning_rate ${learning_rate} \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --max_grad_norm 1.0 \
    --disable_dropout True \
    --bf16 True \
    --report_to "wandb" \
    --run_name ${wandb_name} \
    --length_config "default" \
    --skip_length_validation True \
    --trust_remote_code True \
    --precompute_ref_log_probs True \
    $(if [[ "${use_lora}" == "true" ]]; then echo "\
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --lora_target_modules all-linear"; fi) || exit 1

echo "running evaluation script for checkpoints in ${log_dir}"
sbatch --export=ALL "${REPO_ROOT}/evaluate_policy.sh" --run_name "${wandb_name}" --kl_base_model_path "${base_model_name}" --checkpoint "${log_dir}"

# APO: psbt scripts/rlhf/dpo/dpo.sh --loss_type apo_zero --beta 0.05
