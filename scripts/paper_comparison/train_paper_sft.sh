#!/bin/bash
# Train Pythia-1.4B SFT on AlpacaFarm SFT split (Phase 1a).
# Paper hyperparams: LR 8e-6, 3 epochs, batch 4.

#SBATCH --job-name=paper_sft
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

log_dir="${REPO_ROOT}/scripts/paper_comparison/logs_sft/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"

# Paper's published SFT'd model (baseline, for reference):
#   tlc4418/pythia_1.4b_sft_policy
# Here we re-train from the base Pythia-1.4B for full pipeline validation.
base_model_name="EleutherAI/pythia-1.4b"
dataset_path="ktolnos/alpacafarm_paper_sft_messages"  # converted via convert_paper_dataset.py

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_PROJECT="paper-comparison-sft"

PORT_SELECTOR_SCRIPT="${REPO_ROOT}/scripts/common/select_master_port.sh"
if [[ -f "${PORT_SELECTOR_SCRIPT}" ]]; then
    MASTER_PORT="$(bash "${PORT_SELECTOR_SCRIPT}" 9900 9999)"
else
    MASTER_PORT=9901
fi
export MASTER_PORT

wandb_name="paper_sft_pythia1.4b_${SLURM_JOB_ID:-local}"
export WANDB_RUN_NAME="${wandb_name}"

mkdir -p "${log_dir}"
echo "Logging to ${log_dir}"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision bf16 \
    --main_process_port "${MASTER_PORT}" \
    rlhf/sft/my_sft.py \
    --model_name_or_path "${base_model_name}" \
    --dataset_path "${dataset_path}" \
    --output_dir "${log_dir}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 0.1 \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_only_model True \
    --learning_rate 8e-6 \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --report_to "wandb" \
    --run_name "${wandb_name}" \
    --trust_remote_code True \
    --length_config "alpacafarm_paper" \
    || exit 1

echo "SFT training complete. Checkpoints in ${log_dir}"
