#!/bin/bash

#SBATCH --job-name=train_ppo
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=24:00:00

cd /nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/ppo


log_dir="rlhf/logs_ppo/$(date +%Y%m%d_%H%M%S)"
base_model_name="Qwen/Qwen3-0.6B" # policy base model
dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold"

cd ../../../
gpu=0 #,1,2,3
reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B-Base_BT_RM_len3000_fulltrain_5e-06_data/logs/checkpoint-1280/"
# reward_base_model="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
wandb_name="ppo_rmQwen06B_Full_lr5e-7_kl0.0_helpsteer2_gold"
#checkpoint="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40"


CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch  \
    --mixed_precision bf16 \
    rlhf/ppo/my_ppo.py \
    --bf16 True \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir}\
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --learning_rate 5e-5 \
    --warmup_ratio=0.03 \
    --lr_scheduler_type=cosine \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --model_name_or_path ${base_model_name} \
    --sft_model_path ${base_model_name} \
    --reward_model_path ${reward_base_model} \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --whiten_rewards True \
    --save_steps 0.025 \
    --response_length 512 \
    --run_name ${wandb_name} \
    --exp_name ${wandb_name} \
    --num_sample_generations 40 \
    --use_peft True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' \
    --kl_coef 0.0 \
#    --resume_from_checkpoint True \

    