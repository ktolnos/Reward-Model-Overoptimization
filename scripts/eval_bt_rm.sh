
gpu='0'
port=8991
max_length=1024
per_device_eval_batch_size=12
base_model="Qwen/Qwen3-0.6B-Base"
## If not use lora, make peft_name=''
#peft_name='../save_reward_models/gemma-2b-it_BT_RM_seed2_len1024_lora32_1e-05_data/logs/checkpoint-23'
peft_name=''
log_dir='./eval_BT_qwen0.6B_from_gemma2B'
save_all_data=True
freeze_pretrained=False # for freeze pretrained feature baseline

cd ../rm_eval
for task in '../experimental/data/helpsteer2_gold' 'hhh' 'mtbench' # 'unified' 'hhh'  'mtbench' '../experimental/data/helpsteer2_gold'
do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval.py \
                                        --base_model ${base_model} --peft_name ${peft_name} \
                                        --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                        --save_all_data ${save_all_data} --freeze_pretrained ${freeze_pretrained} \
                                        --task ${task}  --max_length ${max_length} --log_dir ${log_dir}
done



