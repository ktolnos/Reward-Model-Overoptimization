#!/bin/bash

#SBATCH --job-name=train_rm
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=48:00:00
#SBATCH --qos=default

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
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
if [[ ! -f "${REPO_ROOT}/AGENTS.md" ]]; then
    echo "ERROR: Could not resolve repo root (got '${REPO_ROOT}')." >&2
    exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

devices=0
n_gpu=1
# export NCCL_P2P_DISABLE=1
# dataset_name='hendrydong/preference_700K'
#dataset_name='../experimental/data/helpsteer2_gold/'
dataset_name=(
    "ktolnos/helpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B"
#    'ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B'
#   'ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k'
#  'gagan3012/helpsteer2-preference-v2'
#  "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/Qwen3-8B-Embedding-Adv-RM-step_1"
#  "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/Qwen3-8B-Embedding-Adv-RM-step_2"
#  "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/Qwen3-8B-Embedding-Adv-RM-step_3"
)
# base_model='Qwen/Qwen3-4B-Instruct-2507'
base_model='Qwen/Qwen3-4B-Base'
# base_model='Qwen/Qwen3-4B'
# base_model='Qwen/Qwen3-4B-Instruct-2507'
seed=${1:-19}
save_last_only=${2:-False}
skip_optimizer=${3:-True}

save_total_limit_arg=""
if [ "$save_last_only" = "True" ] || [ "$save_last_only" = "true" ]; then
    save_total_limit_arg="--save_total_limit 1"
fi

save_only_model_arg=""
if [ "$skip_optimizer" = "True" ] || [ "$skip_optimizer" = "true" ]; then
    save_only_model_arg="--save_only_model True"
fi

echo "Running with seed: $seed, save_last_only: $save_last_only, skip_optimizer: $skip_optimizer"

wandb_name="${seed}_BT_RM_${base_model}_${SLURM_JOB_ID}_helpsteer3v2_annotated"
log_dir="${REPO_ROOT}/save_reward_models"

PORT_SELECTOR_SCRIPT="${REPO_ROOT}/scripts/common/select_master_port.sh"

if ! MASTER_PORT="$(bash "${PORT_SELECTOR_SCRIPT}" 9900 9999)"; then
    exit 1
fi
export MASTER_PORT

learning_rate=2e-5
num_train_epochs=8

evals_per_epoch=4
save_steps=$(echo "scale=10; 1 / $num_train_epochs" | bc)
eval_steps=$(echo "scale=10; 1 / ($num_train_epochs * $evals_per_epoch)" | bc)

gradient_accumulation_steps=64
per_device_train_batch_size=1
per_device_eval_batch_size=1

cd "${REPO_ROOT}/reward_models"
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${MASTER_PORT} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type "constant" \
    --dataset "${dataset_name[@]}" \
    --gradient_checkpointing False \
    --save_strategy steps --save_steps ${save_steps} \
    --eval_strategy steps --eval_steps ${eval_steps} \
    --seed ${seed} \
    ${save_total_limit_arg} \
    ${save_only_model_arg} \
