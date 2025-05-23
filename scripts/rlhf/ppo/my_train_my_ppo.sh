log_dir='rlhf/logs_ppo'
base_model_name="Qwen/Qwen3-0.6B" # policy base model
dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold"

cd ../../../
gpu=0 #,1,2,3
reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B-Base_BT_RM_len3000_fulltrain_5e-06_data/logs/checkpoint-1280/"
# reward_base_model="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
wandb_name="ppo_rmQwen06B_Full_lr5e-7_kl0.1_helpsteer2_gold"
checkpoint="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-30"


CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch  \
    --mixed_precision bf16 \
    rlhf/ppo/my_ppo.py \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir}\
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --warmup_ratio=0.03 \
    --lr_scheduler_type=cosine \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --model_name_or_path ${base_model_name} \
    --sft_model_path ${base_model_name} \
    --reward_model_path ${reward_base_model} \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --whiten_rewards True \
    --save_steps 0.025 \
    --response_length 512 \
    --run_name ${wandb_name} \
    --exp_name ${wandb_name} \
    --num_sample_generations 40 \
    --resume_from_checkpoint True \

    