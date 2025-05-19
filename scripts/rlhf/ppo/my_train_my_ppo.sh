
log_dir='rlhf/logs_ppo'
init_kl_coef=0.00
base_model_name="Qwen/Qwen3-0.6B-Base" # policy base model
dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold" # set the train dataset path, refer to the BoN experiments
eval_dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold" # set the eval dataset


cd ../../../

# 4 gpus for 2b rm
gpu=0 #,1,2,3
num_processes=1 #4
reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B-Base_BT_RM_len3000_fulltrain_5e-06_data/logs/checkpoint-1280/"
### you need set this path
#reward_peft_path='rlhf/save_reward_models/gemma-2b-it_BT_RM_seed2_len1024_lora32_1e-05_dataUnified-Feedback/logs/checkpoint-3536'
wandb_name="ppo_rmQwen06B_lr_kl0.005"
#CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9989 --num_processes ${num_processes} rlhf/ppo/ppo.py \
#    --base_model_name ${base_model_name} \
#    --reward_base_model ${reward_base_model} \
#    --dataset_path ${dataset_path}\
#    --eval_dataset_path ${eval_dataset_path}\
#    --init_kl_coef ${init_kl_coef}\
#    --log_dir ${log_dir} \
#    --wandb_name ${wandb_name} \
#    --normalize_rewards True \
#    --learning_rate 1e-5 \
#    #    --reward_peft_path "${reward_peft_path}" \


CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch rlhf/ppo/my_ppo.py \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir}\
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path ${base_model_name} \
    --sft_model_path ${base_model_name} \
    --reward_model_path ${reward_base_model} \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.05 \
    --save_steps 0.025 \
    --run_name ${wandb_name} \


# training 7B reward model requires 6 gpus and 4 process (other 2 gpus for reward inference)
#gpu='1,2,3,4,5,6'
#num_processes=4
#reward_base_model="mistralai/Mistral-7B-Instruct-v0.2"
#### just an example, you need set this path
#reward_peft_path='rlhf/save_reward_models/Mistral-7B-Instruct-v0.2_20kunifed_label_smooth/logs/checkpoint-1666'
#wandb_name="ppo_labelsmooth7B_lr1e-5_klreg0.0_normrewards"
#CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9989 --num_processes ${num_processes} ppo.py \
#    --base_model_name ${base_model_name} \
#    --reward_base_model ${reward_base_model} \
#    --reward_peft_path ${reward_peft_path} \
#    --dataset_path ${dataset_path}\
#    --eval_dataset_path ${eval_dataset_path}\
#    --init_kl_coef ${init_kl_coef}\
#    --log_dir ${log_dir} \
#    --wandb_name ${wandb_name} \
#    --normalize_rewards True \
#    --learning_rate 1e-5 \

   