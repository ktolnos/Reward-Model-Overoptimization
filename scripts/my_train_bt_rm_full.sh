#!/bin/bash

#SBATCH --job-name=train_rm
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=24:00:00

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
dataset_name='gagan3012/helpsteer2-preference-v2'
base_model='Qwen/Qwen3-Embedding-8B'
seed=42
wandb_name="${seed}_BT_RM_Qwen3-Embedding-8B_${SLURM_JOB_ID}"
log_dir='../save_reward_models'
main_process_port=9994

learning_rate=2e-5
max_length=3000
num_train_epochs=1
gradient_accumulation_steps=4
per_device_train_batch_size=16
per_device_eval_batch_size=16

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --seed ${seed} \
