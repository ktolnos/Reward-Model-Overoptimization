cd ../rlhf/bon

# Step 7:
python step7_collect.py \
    --proxy_score_path: './step5_choose_best_of_n/gemma-2b-it/grm/proxy_score.csv' \
    --gold_score_path: './step6_obtain_bon_gold_score/gemma-2b-it/grm/gold_score.csv' \
    --output_path: './step7_collect/gemma-2b-it/grm' \
    --n_values_start: 1 \
    --n_values_end: 406 \
   
