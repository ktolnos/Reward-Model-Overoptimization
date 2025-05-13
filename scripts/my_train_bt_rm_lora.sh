devices=0
n_gpu=1
dataset_name='../experimental/data/helpsteer2_subset_gold/'
base_model='google/gemma-2b-it' # Qwen/Qwen3-0.6B-Base
wandb_name="BT_RM_seed2"
log_dir='../save_reward_models'
main_process_port=9994
loss_type='bt'

learning_rate=1e-5
lora_r=32
lora_alpha=64
max_length=1024
num_train_epochs=2
gradient_accumulation_steps=4


cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora True \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} --loss_type ${loss_type} \
    --dataset ${dataset_name}