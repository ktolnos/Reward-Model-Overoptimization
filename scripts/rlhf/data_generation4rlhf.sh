devices=0,1,2,3
n_gpu=4
main_process_port=9997


cd ../..
# Data Generation

python rlhf/data_generation/sample_dataset.py

# Step 1
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
   rlhf/data_generation/obtain_gold_score.py \
    --per_device_batch_size 8 \
    --max_length 1024 \
    --data_path "rlhf/data/unified_20k" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/bon/step1_obtain_gold_score" \
    --save_name "unified_20k_gold_score" \
    --mode "train" --debug False
    
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/data_generation/obtain_gold_score.py \
    --per_device_batch_size 8 \
    --max_length 1024 \
    --data_path "rlhf/data/unified_1k" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/bon/step1_obtain_gold_score" \
    --save_name "unified_1k_gold_score" \
    --mode "test" --debug False
