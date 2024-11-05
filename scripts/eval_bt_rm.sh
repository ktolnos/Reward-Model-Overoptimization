
gpu='0,1,2,3'
port=8991
max_length=1024
per_device_eval_batch_size=8
base_model="google/gemma-2b-it"
## If not use lora, make peft_name=''
peft_name='./gemma-2b-it_gemma-2b-it_reward_unified_0.05datasset_lora32_2epoch/logs/checkpoint'
log_dir='./eval_BT'
save_all_data=False
freeze_pretrained=False

cd ../rm_eval
for task in 'unified' 'hhh'  'mtbench'
do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval.py \
                                        --base_model ${base_model} --peft_name ${peft_name} \
                                        --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                        --save_all_data ${save_all_data} --freeze_pretrained ${freeze_pretrained} \
                                        --task ${task}  --max_length ${max_length} --log_dir ${log_dir}
done



