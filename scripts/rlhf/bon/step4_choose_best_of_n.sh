# Step 4:
python rlhf/bon/step5_choose_best_of_n.py \
    --model_type grm \
    --data_path "rlhf/bon/step4_obtain_proxy_score/gemma-2b-it/grm" \
    --n_values_start 1 \
    --n_values_end 406 \
    --save_path "rlhf/bon/step5_choose_best_of_n/gemma-2b-it"
