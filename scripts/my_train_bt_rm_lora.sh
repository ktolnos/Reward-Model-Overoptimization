devices=0
n_gpu=1
dataset_name='../experimental/data/helpsteer2_gold/'
base_model='google/gemma-2b-it' # Qwen/Qwen3-0.6B-Base
wandb_name="BT_RM_seed2"
log_dir='../save_reward_models'
main_process_port=9994

learning_rate=1e-5
lora_r=32
lora_alpha=64
num_train_epochs=5
gradient_accumulation_steps=4


cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --use_lora True \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name}