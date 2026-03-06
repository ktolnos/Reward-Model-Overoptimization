#!/bin/bash
# GRPO training for the paper comparison (Phase 2).
# Matches the paper's PPO setup as closely as possible.
#
# Required arguments:
#   --ensemble_aggregation  mean|min|uwo|none  (none = single RM)
#   --beta                  KL penalty coefficient (0.1 or 0.01)
#
# Optional arguments:
#   --run_name              Custom WandB run name
#   --single_rm             Use only the first RM (for single-RM baseline runs)
#
# Usage examples:
#   # Single RM, high KL:
#   sbatch grpo_paper.sh --ensemble_aggregation none --beta 0.1 --single_rm
#
#   # WCO ensemble, low KL:
#   sbatch grpo_paper.sh --ensemble_aggregation min --beta 0.01
#
#   # UWO ensemble, high KL:
#   sbatch grpo_paper.sh --ensemble_aggregation uwo --beta 0.1

#SBATCH --job-name=paper_grpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
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
beta="0.1"
ensemble_aggregation="mean"
SINGLE_RM=0
wandb_name_base=""
uwo_lambda="0.5"

# ---- Parse arguments ----
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --beta) beta="$2"; shift ;;
        --ensemble_aggregation) ensemble_aggregation="$2"; shift ;;
        --single_rm) SINGLE_RM=1 ;;
        --run_name) wandb_name_base="$2"; shift ;;
        --uwo_lambda) uwo_lambda="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ---- Model and data paths ----
# Use the paper's published SFT policy as base (or replace with our re-trained SFT)
base_model_name="tlc4418/pythia_1.4b_sft_policy"
dataset_path="ktolnos/alpacafarm_paper_grpo_prompts"  # converted unlabeled split

# Proxy RM ensemble (5 seeds). Update these paths after running train_paper_rm.sh.
# TODO: Replace with actual trained RM checkpoint paths
reward_model_paths=(
    "${REPO_ROOT}/save_reward_models/paper_comparison/1_pythia_70m_sft_len776_fulltrain_1e-05_dataalpacafarm_paper_preference_messages/logs/LATEST_CHECKPOINT"
    "${REPO_ROOT}/save_reward_models/paper_comparison/2_pythia_70m_sft_len776_fulltrain_1e-05_dataalpacafarm_paper_preference_messages/logs/LATEST_CHECKPOINT"
    "${REPO_ROOT}/save_reward_models/paper_comparison/3_pythia_70m_sft_len776_fulltrain_1e-05_dataalpacafarm_paper_preference_messages/logs/LATEST_CHECKPOINT"
    "${REPO_ROOT}/save_reward_models/paper_comparison/4_pythia_70m_sft_len776_fulltrain_1e-05_dataalpacafarm_paper_preference_messages/logs/LATEST_CHECKPOINT"
    "${REPO_ROOT}/save_reward_models/paper_comparison/5_pythia_70m_sft_len776_fulltrain_1e-05_dataalpacafarm_paper_preference_messages/logs/LATEST_CHECKPOINT"
)

if [[ "${SINGLE_RM}" -eq 1 ]]; then
    reward_model_paths=("${reward_model_paths[0]}")
fi

num_rms=${#reward_model_paths[@]}

# ---- Construct run name ----
if [[ -z "${wandb_name_base}" ]]; then
    if [[ "${SINGLE_RM}" -eq 1 ]]; then
        wandb_name_base="paper_grpo_single"
    else
        wandb_name_base="paper_grpo_${ensemble_aggregation}"
    fi
fi
run_name="${wandb_name_base}_KL${beta}_${num_rms}rms_${SLURM_JOB_ID:-local}"

log_dir="${REPO_ROOT}/scripts/paper_comparison/logs_grpo/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
mkdir -p "${log_dir}"

export PYTHONPATH="${REPO_ROOT}/rlhf/grpo:${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT="paper-comparison-grpo"
export WANDB_RUN_NAME="${run_name}"
export WANDB_RUN_GROUP="${log_dir}"

# Port selection
PORT_SELECTOR_SCRIPT="${REPO_ROOT}/scripts/common/select_master_port.sh"
if [[ -f "${PORT_SELECTOR_SCRIPT}" ]]; then
    MASTER_PORT="$(bash "${PORT_SELECTOR_SCRIPT}" 9900 9999)"
else
    MASTER_PORT=9901
fi
export MASTER_PORT
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost

# ---- Determine aggregation flags ----
# For single RM, ensemble_aggregation is irrelevant; use "mean" as default.
actual_aggregation="${ensemble_aggregation}"
if [[ "${SINGLE_RM}" -eq 1 ]]; then
    actual_aggregation="mean"
fi

echo "=== Paper Comparison GRPO ==="
echo "Base model: ${base_model_name}"
echo "Dataset: ${dataset_path}"
echo "RMs: ${num_rms}"
echo "Aggregation: ${actual_aggregation}"
echo "Beta: ${beta}"
echo "UWO lambda: ${uwo_lambda}"
echo "Log dir: ${log_dir}"
echo "Run name: ${run_name}"
echo ""

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision bf16 \
    rlhf/grpo/my_grpo.py \
    --model_name_or_path "${base_model_name}" \
    --dataset_path "${dataset_path}" \
    --output_dir "${log_dir}" \
    --reward_model_paths "${reward_model_paths[@]}" \
    --ensemble_aggregation "${actual_aggregation}" \
    --rm_switch_strategy "ensemble" \
    --length_config "alpacafarm_paper" \
    --uwo_use_variance True \
    --uwo_lambda "${uwo_lambda}" \
    --beta "${beta}" \
    --num_generations 16 \
    --num_train_epochs 1 \
    --temperature 1.0 \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.1 \
    --vllm_mode "colocate" \
    --loss_type "dr_grpo" \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-6 \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --save_steps 200 \
    --save_only_model True \
    --logging_steps 10 \
    --log_completions True \
    --log_unique_prompts True \
    --disable_dropout True \
    --max_grad_norm 1.0 \
    --run_name "${run_name}" \
    --rm_subtract_mean_reward_per_model True \
    --rm_scale_reward_by_std_per_model True \
    --penalize_no_eos False \
    --reference_rewards False \
    --sigmoid_rewards False \
    --adv_rm_lambda 0.0 \
    --online_pet_enabled False \
    --trust_remote_code True \
    --save_generations_path "${log_dir}/generations.csv" \
    --gradient_checkpointing False \
    --scale_rewards "batch" \
    --mask_truncated_completions False \
    --epsilon_high 0.28 \
    || exit 1

echo "GRPO training complete. Checkpoints in ${log_dir}"

# Trigger evaluation
echo "Submitting evaluation job for ${log_dir}"
if [[ -f "${REPO_ROOT}/scripts/paper_comparison/evaluate_paper.sh" ]]; then
    sbatch --export=ALL "${REPO_ROOT}/scripts/paper_comparison/evaluate_paper.sh" \
        --run_name "${run_name}" \
        --kl_base_model_path "${base_model_name}" \
        --checkpoint "${log_dir}"
fi
