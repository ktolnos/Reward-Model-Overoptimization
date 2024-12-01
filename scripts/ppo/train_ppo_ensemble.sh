gpu=6,7,8,9
base_model_name="google/gemma-2b-it"
reward_base_model="google/gemma-2b-it"
dataset_path="./data/unified_20k_gold_score" # set the train dataset path, refer to the BoN experiments
eval_dataset_path="./data/unified_1k" # set the eval dataset
init_kl_coef=0.0
log_dir='./log_ppo_noise'
eval_every=4

cd ../../rlhf/ppo

# you need set the path
reward_peft_path1='reward_models_noise/model_finetuned_proxy_rm_noise25/reward_models_20kunified_vanilla_1/718cb189da9c5b2e55abe86f2eeffee9b4ae0dad_gemma-2b-it_20kunifed_vanilla_1_1e-05/logs/checkpoint-3334'
reward_peft_path2='reward_models_noise/model_finetuned_proxy_rm_noise25/reward_models_20kunified_vanilla_2/718cb189da9c5b2e55abe86f2eeffee9b4ae0dad_gemma-2b-it_20kunifed_vanilla_2_1e-05/logs/checkpoint-3334'
reward_peft_path3='reward_models_noise/model_finetuned_proxy_rm_noise25/reward_models_20kunified_vanilla_3/718cb189da9c5b2e55abe86f2eeffee9b4ae0dad_gemma-2b-it_20kunifed_vanilla_3_1e-05/logs/checkpoint-3334'

ensemble_method='avg'
wandb_name="ppo_noise_ensemble_avg_lr1e-5_kl0.0_tau0.7_len512_normrewards_noadapkl"
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9991 ppo_ensemble.py \
    --base_model_name ${base_model_name} \
    --reward_base_model ${reward_base_model} \
    --reward_peft_path1 ${reward_peft_path1} \
    --reward_peft_path2 ${reward_peft_path2} \
    --reward_peft_path3 ${reward_peft_path3} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --eval_every ${eval_every} \
    --ensemble_method ${ensemble_method} \
    --normalize_rewards True \
    --learning_rate 1e-5 \



ensemble_method='min'
wandb_name="gemma-2b-it_ppo_ensemble_min_lr1e-5_kl0.0_tau0.7_len512_normrewards_noadapkl"
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9998 ppo_ensemble.py \
    --base_model_name ${base_model_name} \
    --reward_base_model ${reward_base_model} \
    --reward_peft_path1 ${reward_peft_path1} \
    --reward_peft_path2 ${reward_peft_path2} \
    --reward_peft_path3 ${reward_peft_path3} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --eval_every ${eval_every} \
    --ensemble_method ${ensemble_method} \
    --learning_rate 1e-5 \




   



