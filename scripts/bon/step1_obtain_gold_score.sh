devices=0,1,2,3
n_gpu=4
main_process_port=9994

# Step 1:
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step1_obtain_gold_score.py \
    --per_device_batch_size: 64 \
    --max_length: 1024 \
    --data_path: "rlhf/bon//data/unified_20k" \
    --model_path: "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path: "rlhf/bon/step1_obtain_gold_score" \
    --save_name: "unified_sampled_gold_score" \
    --mode: "train" \
    
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step1_obtain_gold_score.py \
    --per_device_batch_size: 64 \
    --max_length: 1024 \
    --data_path: "rlhf/bon/data/unified_1k" \
    --model_path: "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path: "rlhf/bon/step1_obtain_gold_score" \
    --save_name: "unified_sampled_gold_score" \
    --mode: "test" \
