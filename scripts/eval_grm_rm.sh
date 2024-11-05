
gpu='0,1,2,3'
port=8991
per_device_eval_batch_size=8
base_model="google/gemma-2b-it"
peft_name='./gemma-2b-it_reward_unified_0.05datasset_lora32_2epoch_nonlinear_sftreg0.01_1e-05/logs/checkpoint'
layer_type='mlp' # linear
num_layers=1
log_dir='./eval'
save_all_data=False

cd ../rm_eval
for task in 'unified' 'hhh'  'mtbench'
do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval_grm.py --base_model ${base_model} --peft_name ${peft_name} \
                                            --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${log_dir} --save_all_data ${save_all_data} \
                                              --task ${task} --layer_type ${layer_type} --num_layers ${num_layers} 

done



