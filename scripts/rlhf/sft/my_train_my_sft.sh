log_dir='rlhf/logs_sft_helpsteer2_gold'
base_model_name="Qwen/Qwen3-0.6B-Base" # base model to finetune
dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold"

cd ../../../
gpu=0 #,1,2,3
wandb_name="sft_Qwen06B_lr5e-5_helpsteer2_gold"

CUDA_VISIBLE_DEVICES=${gpu} accelerate launch \
    --mixed_precision bf16 \
    rlhf/sft/my_sft.py \
    --model_name_or_path ${base_model_name} \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to "wandb" \
    --run_name ${wandb_name} \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_length 1024 \
    --trust_remote_code True