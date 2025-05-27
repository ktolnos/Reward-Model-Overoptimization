#!/bin/bash

# Directory containing the checkpoints
CHECKPOINTS_DIR="."  # Current directory with all checkpoints

# Path to the training reward model
TRAINING_RM_PATH="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40"

# Name of the gold reward model
GOLD_RM_NAME="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"

# Dataset name
DATASET_NAME="gagan3012/helpsteer2-gold"

# Base model name (required for LoRA checkpoints)
# Uncomment and set this if evaluating LoRA checkpoints
# BASE_MODEL_NAME="mistralai/Mistral-7B-v0.1"

# Output file
OUTPUT_FILE="evaluation_results.csv"

# WandB settings
WANDB_PROJECT="policy-evaluation"
WANDB_RUN_NAME="policy_evaluation_$(date +%Y%m%d_%H%M%S)"

# Debug mode flag (uncomment to enable)
DEBUG_MODE="--debug"

# Run the evaluation script
python evaluate_policy.py \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --training_rm_path "$TRAINING_RM_PATH" \
    --gold_rm_name "$GOLD_RM_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 8 \
    --max_length 1024 \
    --device "cuda" \
    --num_responses_per_prompt 5 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    ${DEBUG_MODE:-} \
    $([ ! -z "${BASE_MODEL_NAME:-}" ] && echo "--base_model_name $BASE_MODEL_NAME")

# To disable wandb logging, add: --disable_wandb
# To enable debug mode, uncomment the DEBUG_MODE line above
# For LoRA models, uncomment and set BASE_MODEL_NAME above 