
log_dir='./logs_ppo'
init_kl_coef=0.00
base_model_name="google/gemma-2b-it" # policy base model
dataset_path="./data/unified_20k_gold_score" # set the train dataset path, refer to the BoN experiments
eval_dataset_path="./data/unified_1k" # set the eval dataset


cd ../../rlhf/ppo

# 4 gpus for 2b rm
gpu=4 #0,1,2,3 
num_processes=1
reward_base_model="google/gemma-2b-it"
### you need set this path
reward_peft_path='reward_models/model_finetuned_proxy_rm_2b/logs/checkpoint-1250' 
wandb_name="ppo_rm2B_lr1e-5_klreg0.0_normrewards"
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9989 --num_processes ${num_processes} ppo.py \
    --base_model_name ${base_model_name} \
    --reward_base_model ${reward_base_model} \
    --reward_peft_path ${reward_peft_path} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --normalize_rewards True \
    --learning_rate 1e-5 \


# # training 7B reward model requires 6 gpus and 4 process (other 2 gpus for reward inference)
# gpu='1,2,3,4,5,6'
# reward_base_model="mistralai/Mistral-7B-Instruct-v0.2"
# ### just an example, you need set this path
# reward_peft_path='reward_models/20kunified_label_smooth/Mistral-7B-Instruct-v0.2_20kunifed_label_smooth/logs/checkpoint-1666'
# wandb_name="ppo_labelsmooth7B_lr1e-5_klreg0.0_normrewards"
# CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9989 --num_processes 4 ppo.py \
#     --base_model_name ${base_model_name} \
#     --reward_base_model ${reward_base_model} \
#     --reward_peft_path ${reward_peft_path} \
#     --dataset_path ${dataset_path}\
#     --eval_dataset_path ${eval_dataset_path}\
#     --init_kl_coef ${init_kl_coef}\
#     --log_dir ${log_dir} \
#     --wandb_name ${wandb_name} \
#     --normalize_rewards True \
#     --learning_rate 1e-5 \

   