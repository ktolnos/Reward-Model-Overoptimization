devices=0
n_gpu=1
# export NCCL_P2P_DISABLE=1
# dataset_name='hendrydong/preference_700K'
dataset_name='../experimental/data/helpsteer2_gold/'
base_model='Qwen/Qwen3-0.6B-Base'
wandb_name="BT_RM"
log_dir='../save_reward_models'
main_process_port=9994

learning_rate=5e-6
max_length=3000
num_train_epochs=5
gradient_accumulation_steps=64

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name}