#!/bin/bash

#SBATCH --job-name=train_rm
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=8:00:00
#SBATCH --qos=default


export HF_HOME="/nas/ucb/eop/cache"
export PYTHONPATH="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/grpo/:/nas/ucb/eop/Reward-Model-Overoptimization/:$PYTHONPATH"
export TMPDIR="/nas/ucb/eop/temp"
export TEMP="/nas/ucb/eop/temp"
export TMP="/nas/ucb/eop/temp"
export PYTHONPYCACHEPREFIX="/nas/ucb/eop/temp/pycache"
export TORCHINDUCTOR_CACHE_DIR="/nas/ucb/eop/temp/torchinductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE="/nas/ucb/eop/temp/fx_graph_cache"
export VLLM_CONFIG_ROOT="/nas/ucb/eop/cache/vllm_config"
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_DIR="/nas/ucb/eop/wandb"
export WANDB_CACHE_DIR="/nas/ucb/eop/cache/wandb"
export WANDB_DATA_DIR="/nas/ucb/eop/cache/wandb-data"
export WANDB_ARTIFACT_DIR="/nas/ucb/eop/cache/wandb-artifacts"

devices=0
n_gpu=1
# export NCCL_P2P_DISABLE=1
# dataset_name='hendrydong/preference_700K'
#dataset_name='../experimental/data/helpsteer2_gold/'
dataset_name=(
   # 'ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B'
  'ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k'
#  'gagan3012/helpsteer2-preference-v2'
#  "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/Qwen3-8B-Embedding-Adv-RM-step_1"
#  "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/Qwen3-8B-Embedding-Adv-RM-step_2"
#  "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/Qwen3-8B-Embedding-Adv-RM-step_3"
)
base_model='Qwen/Qwen3-0.6B'
seed=${1:-19}
save_last_only=${2:-False}
skip_optimizer=${3:-False}

save_total_limit_arg=""
if [ "$save_last_only" = "True" ] || [ "$save_last_only" = "true" ]; then
    save_total_limit_arg="--save_total_limit 1"
fi

save_only_model_arg=""
if [ "$skip_optimizer" = "True" ] || [ "$skip_optimizer" = "true" ]; then
    save_only_model_arg="--save_only_model True"
fi

echo "Running with seed: $seed, save_last_only: $save_last_only, skip_optimizer: $skip_optimizer"

wandb_name="${seed}_BT_RM_${base_model}_${SLURM_JOB_ID}_helpsteer3_gold_10k"
log_dir='../save_reward_models'

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

learning_rate=2e-5
max_length=2000
num_train_epochs=2
gradient_accumulation_steps=16
per_device_train_batch_size=4
per_device_eval_batch_size=4

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${MASTER_PORT} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type "constant" \
    --dataset "${dataset_name[@]}" \
    --gradient_checkpointing False \
    --seed ${seed} \
    ${save_total_limit_arg} \
    ${save_only_model_arg} \
