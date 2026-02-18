#!/bin/bash

#SBATCH --job-name=train_sft
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=8:00:00
#SBATCH --qos=high

# Go to the root of the repo
cd /nas/ucb/eop/Reward-Model-Overoptimization/ || exit
f
export HF_HOME="/nas/ucb/eop/cache"
export TMPDIR="/nas/ucb/eop/temp"
export TEMP="/nas/ucb/eop/temp"
export TMP="/nas/ucb/eop/temp"
export PYTHONPYCACHEPREFIX="/nas/ucb/eop/temp/pycache"
export TORCHINDUCTOR_CACHE_DIR="/nas/ucb/eop/temp/torchinductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE="/nas/ucb/eop/temp/fx_graph_cache"
export VLLM_CONFIG_ROOT="/nas/ucb/eop/cache/vllm_config"
export VLLM_DISABLE_COMPILE_CACHE="1"
export VLLM_CACHE_ROOT="/nas/ucb/eop/cache/"

export WANDB_DIR="/nas/ucb/eop/wandb"
export WANDB_CACHE_DIR="/nas/ucb/eop/cache/wandb"
export WANDB_DATA_DIR="/nas/ucb/eop/cache/wandb-data"
export WANDB_ARTIFACT_DIR="/nas/ucb/eop/cache/wandb-artifacts"

log_dir="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
base_model_name="Qwen/Qwen3-0.6B-Base"
dataset_path="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"

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
MIN_PORT=9900
MAX_PORT=9999

is_port_in_use() {
    ss -lntu | grep -q ":$1 "
}

IDEAL_PORT=$((MIN_PORT + (SLURM_JOB_ID % (MAX_PORT - MIN_PORT + 1))))

if ! is_port_in_use "${IDEAL_PORT}"; then
    export MASTER_PORT=${IDEAL_PORT}
else
    for port in $(seq ${MIN_PORT} ${MAX_PORT}); do
        if ! is_port_in_use "${port}"; then
            export MASTER_PORT=${port}
            break
        fi
    done
fi

if [[ -z "${MASTER_PORT}" ]]; then
    echo "ERROR: Could not find any free port in the range ${MIN_PORT}-${MAX_PORT}." >&2
    exit 1
fi

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
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 0.0625 \
    --save_strategy "steps" \
    --save_steps 0.0625 \
    --save_only_model True \
    --learning_rate 1e-5 \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --report_to "wandb" \
    --run_name ${wandb_name} \
    --max_prompt_length 1024 \
    --max_length_filter 2048 \
    --max_length 1024 \
    --trust_remote_code True || exit 1

echo "running evaluation script for checkpoints in ${log_dir}"
sbatch --export=ALL /nas/ucb/eop/Reward-Model-Overoptimization/evaluate_policy.sh --run_name "${wandb_name}" --kl_base_model_path "${base_model_name}" --checkpoint "${log_dir}"
