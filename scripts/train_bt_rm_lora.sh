devices=0,1,2,3,4,5
n_gpu=6
dataset_name='llm-blender/Unified-Feedback'
base_model='google/gemma-2b-it'
wandb_name="BT_RM_seed2"
log_dir='../save_reward_models'
main_process_port=9994

learning_rate=1e-5
lora_r=32
lora_alpha=64
num_train_epochs=2
gradient_accumulation_steps=4


cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --use_lora True \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --eval_strategy steps --eval_steps 0.02 \
    --dataset ${dataset_name}