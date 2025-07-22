#!/bin/bash

#SBATCH --job-name=evaluate_policy
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=shard:13
#SBATCH --time=12:00:00

export PYTHONPATH="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/grpo/:/nas/ucb/eop/Reward-Model-Overoptimization/:$PYTHONPATH"
export HF_HOME="/nas/ucb/eop/cache"

cd /nas/ucb/eop/Reward-Model-Overoptimization
source .bashrc

# Directory containing the checkpoints
CHECKPOINTS_DIR="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_grpo/20250722_101740" # Current directory with all checkpoints

# Path to the training reward model
TRAINING_RM_PATH="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-Embedding-8B_42_BT_RM_Qwen3-Embedding-8B_915487_len2000_fulltrain_2e-05_datahelpsteer2-preference-v2/logs/checkpoint-272"
#TRAINING_RM_PATH="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"

# Name of the gold reward model
#GOLD_RM_NAME="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
#GOLD_RM_NAME="LxzGordon/URM-LLaMa-3.1-8B"
GOLD_RM_NAME="nicolinho/QRM-Gemma-2-27B"

# Dataset name
#DATASET_NAME="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold_URM-LLaMa-3.1-8B_0_7951/"
DATASET_NAME="gagan3012/helpsteer2-preference-v2"


# Base model name (required for LoRA checkpoints)
# Uncomment and set this if evaluating LoRA checkpoints
#BASE_MODEL_NAME="Qwen/Qwen3-0.6B"

# Output file
OUTPUT_FILE="evaluation_results${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).json"

# WandB settings
WANDB_PROJECT="policy-evaluation"
WANDB_RUN_NAME="policy_evaluation_$(date +%Y%m%d_%H%M%S)"

# Debug mode flag (uncomment to enable)
#DEBUG_MODE="--debug"

# Run the evaluation script
python evaluate_policy.py \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --training_rm_path "$TRAINING_RM_PATH" \
    --gold_rm_name "$GOLD_RM_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 1 \
    --generation_batch_size 64 \
    --max_length 1024 \
    --device "cuda" \
    --num_responses_per_prompt 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --evaluate_with_training_rm True \
    --evaluate_with_llm_judge True \
    --llm_judge_model_name "deepseek/deepseek-r1-0528:free" \
    --baseline_model_path "Qwen/Qwen3-0.6B" \
    --use_dataset_response_as_baseline False \
    --save_eval_dataset_path "evaluation_dataset_${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).json" \
    --subsample_n 25 \
    ${DEBUG_MODE:-} \
    $([ ! -z "${BASE_MODEL_NAME:-}" ] && echo "--base_model_name $BASE_MODEL_NAME") \

# To disable wandb logging, add: --disable_wandb
# To enable debug mode, uncomment the DEBUG_MODE line above
# For LoRA models, uncomment and set BASE_MODEL_NAME above 