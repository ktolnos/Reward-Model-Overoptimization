#!/bin/bash

#SBATCH --job-name=evaluate_policy
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=3:00:00
#SBATCH --qos=high

cd /nas/ucb/eop/Reward-Model-Overoptimization
source .bashrc

# Directory containing the checkpoints
CHECKPOINTS_DIR="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_grpo/20260202_183444_1035193"


# Path to the training reward model
TRAINING_RM_PATH="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen/Qwen3-0.6B_974219_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#TRAINING_RM_PATH="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"

# Name of the gold reward model
#GOLD_RM_NAME="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
#GOLD_RM_NAME="LxzGordon/URM-LLaMa-3.1-8B"
#GOLD_RM_NAME="Skywork/Skywork-Reward-V2-Qwen3-8B"
GOLD_RM_NAME="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

# Dataset name
#DATASET_NAME="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold_URM-LLaMa-3.1-8B_0_7951/"
DATASET_NAME="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"


# Base model name (required for LoRA checkpoints)
# Uncomment and set this if evaluating LoRA checkpoints
#BASE_MODEL_NAME="Qwen/Qwen3-0.6B"

# Output file
OUTPUT_FILE="evaluation_results${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).json"

# WandB settings
WANDB_PROJECT="policy-evaluation"
WANDB_RUN_NAME="policy_evaluation_$(date +%Y%m%d_%H%M%S)"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) WANDB_RUN_NAME="$2"; shift ;;
        --kl_base_model_path) KL_BASE_MODEL_PATH="$2"; shift ;;
        --checkpoint) CHECKPOINTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running evaluation with the following settings:"
echo "Checkpoints Directory: $CHECKPOINTS_DIR"
echo "Training RM Path: $TRAINING_RM_PATH"
echo "Gold RM Name: $GOLD_RM_NAME"
echo "Dataset Name: $DATASET_NAME"
echo "Output File: $OUTPUT_FILE"
echo "WandB Project: $WANDB_PROJECT"

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
    --generation_batch_size 32 \
    --max_length 1024 \
    --device "cuda" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --evaluate_with_training_rm False \
    --evaluate_with_llm_judge False \
    --llm_judge_model_name "tngtech/deepseek-r1t2-chimera:free" \
    --baseline_model_path "Qwen/Qwen3-0.6B" \
    --kl_base_model_path "${KL_BASE_MODEL_PATH:-/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/20260106_012931_1016814/checkpoint-158}" \
    --use_dataset_response_as_baseline False \
    --save_eval_dataset_path "evaluation_dataset_${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).json" \
    ${DEBUG_MODE:-} \
    $([ ! -z "${BASE_MODEL_NAME:-}" ] && echo "--base_model_name $BASE_MODEL_NAME") \

#     --subsample_n 25 \

# To disable wandb logging, add: --disable_wandb
# To enable debug mode, uncomment the DEBUG_MODE line above
# For LoRA models, uncomment and set BASE_MODEL_NAME above 