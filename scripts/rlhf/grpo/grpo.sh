#!/bin/bash

#SBATCH --job-name=train_grpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=24:00:00

cd /nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/grpo
export HF_HOME="/nas/ucb/eop/cache"

log_dir="rlhf/logs_grpo/$(date +%Y%m%d_%H%M%S)"
base_model_name="Qwen/Qwen3-0.6B" # policy base model
dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold_QRM_Gemma2_27B_0_7748"

cd ../../../
gpu=0 #,1,2,3
#reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
reward_base_model="nicolinho/QRM-Gemma-2-27B"
learning_rate="5e-7"
per_device_train_batch_size=1
gradient_accumulation_steps=16
# shellcheck disable=SC2004
wandb_name="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)_lr${learning_rate}_batch$(($per_device_train_batch_size * $gradient_accumulation_steps))_rmQwen06B_Full_helpsteer2_gold"
#checkpoint="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40"
echo $SLURM_JOB_ID

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=9991
export WANDB_PROJECT="grpo"
export WANDB_RUN_NAME=${wandb_name}



CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch  \
    --mixed_precision bf16 \
    rlhf/grpo/my_grpo.py \
    --num_generations 8 \
    --num_train_epochs 2 \
    --temperature 0.9 \
    --max_prompt_length 512 \
    --max_completion_length 256 \
    --epsilon_high 0.28 \
    --mask_truncated_completions True \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.08 \
    --vllm_mode "colocate" \
    --beta 0.0 \
    --log_completions True \
    --loss_type "dr_grpo" \
    --mask_truncated_completions True \
    --wandb_log_unique_prompts True \
    --disable_dropout True \
    --bf16 True \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir}\
    --warmup_ratio=0.1 \
    --lr_scheduler_type=cosine \
    --model_name_or_path ${base_model_name} \
    --reward_model_path ${reward_base_model} \
    --save_steps 0.025 \
    --run_name ${wandb_name} \
    --logging_steps 0.005 \
    --learning_rate ${learning_rate} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --gradient_checkpointing True \
    --scale_rewards False \
    --trust_remote_code True \
#    --use_peft True \
#    --lora_r 32 \
#    --lora_alpha 64 \
#    --lora_target_modules 'all-linear' \
#    --resume_from_checkpoint True \
# 'q_proj' 'k_proj' 'v_proj' 'o_proj' \
