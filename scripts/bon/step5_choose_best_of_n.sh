cd ../rlhf/bon

# Step 5:
python step5_choose_best_of_n.py \
    --model_type grm \
    --data_path "./step4_obtain_proxy_score/gemma-2b-it/grm" \
    --n_values_start 1 \
    --n_values_end 406 \
    --save_path "./step5_choose_best_of_n/gemma-2b-it"
