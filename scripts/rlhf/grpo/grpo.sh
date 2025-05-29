#!/bin/bash

#SBATCH --job-name=train_ppo
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=24:00:00

cd /nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/grpo


log_dir="rlhf/logs_grpo/$(date +%Y%m%d_%H%M%S)"
base_model_name="Qwen/Qwen3-0.6B" # policy base model
dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold"

cd ../../../
gpu=0 #,1,2,3
reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B-Base_BT_RM_len3000_fulltrain_5e-06_data/logs/checkpoint-1280/"
# reward_base_model="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
wandb_name="grpo_rmQwen06B_Full_lr5e-7_kl0.0_helpsteer2_gold"
#checkpoint="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40"
echo $SLURM_JOB_ID

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=9989
export WANDB_PROJECT="grpo"
export WANDB_RUN_NAME=${wandb_name}



CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch  \
    --mixed_precision bf16 \
    rlhf/grpo/my_grpo.py \
    --num_generations 8 \
    --temperature 0.9 \
    --max_completion_length 512 \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_mode "colocate" \
    --beta 0.01 \
    --log_completions True \
    --loss_type "dr_grpo" \
    --mask_truncated_completions True \
    --wandb_log_unique_prompts True \
    --disable_dropout True \
    --bf16 True \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir}\
    --learning_rate 1e-5 \
    --warmup_ratio=0.1 \
    --lr_scheduler_type=cosine \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path ${base_model_name} \
    --reward_model_path ${reward_base_model} \
    --save_steps 0.025 \
    --run_name ${wandb_name} \
    --use_peft True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules 'all-linear' \

#    --resume_from_checkpoint True \
# 'q_proj' 'k_proj' 'v_proj' 'o_proj' \
