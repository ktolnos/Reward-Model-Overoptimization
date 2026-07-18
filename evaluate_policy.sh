#!/bin/bash

#SBATCH --job-name=evaluate_policy
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --time=5:00:00
#SBATCH --qos=high

# =============================================================================
# Usage
# =============================================================================
# Full eval (preference + ifeval):
#     sbatch evaluate_policy.sh
#
# Add IFEval to an existing wandb run (no re-generation of preference eval):
#     sbatch evaluate_policy.sh --run_id <WANDB_RUN_ID> --only_ifeval
#
# IFEval with both rule-based AND gold-RM scoring:
#     sbatch evaluate_policy.sh --ifeval_use_gold_rm
#
# Arena-Hard-Auto v2.0 (gold RM only):
#     sbatch evaluate_policy.sh --only_arena_hard [--run_id <WANDB_RUN_ID>]
#
# Arbitrary benchmark subset:
#     sbatch evaluate_policy.sh --benchmarks ifeval,preference,arena_hard
#
# Other overrides (all optional): --run_name, --checkpoint, --kl_base_model_path,
# --ifeval_thinking, --evaluate_chosen_responses, --no_secondary_rm,
# --with_training_rm, --with_llm_judge.
# =============================================================================

cd /nas/ucb/eop/Reward-Model-Overoptimization
source /home/eop/.bashrc
echo "$(ls -a /home/eop/)"

# Directory containing the checkpoints
CHECKPOINTS_DIR="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_grpo/20260202_183444_1035193"


# Training RM, dataset, KL base model, and eval temperature default from the
# run's run_manifest.json (written by my_grpo.py into the checkpoints dir), so
# eval always matches what the run actually trained with. Leave these empty to
# use the manifest; set them (here or via the --flags below) only to override
# it or to evaluate a legacy run that predates the manifest.
TRAINING_RM_PATH=""
#TRAINING_RM_PATH="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen/Qwen3-0.6B_974219_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#TRAINING_RM_PATH="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"

# Sibling RM: an independently-seeded RM from the training RM's base model, used
# by the 'select' benchmark to pick the best checkpoint (scored on the dataset's
# 'select' split). Validated as a near-oracle checkpoint selector.
SIBLING_RM_PATH="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/20_Qwen3.5-4B-Base_len2048_fulltrain_2e-05_datahelpsteer3-qwen35_annotated_human/logs/checkpoint-1179"

# Name of the gold reward model
#GOLD_RM_NAME="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
#GOLD_RM_NAME="LxzGordon/URM-LLaMa-3.1-8B"
#GOLD_RM_NAME="Skywork/Skywork-Reward-V2-Qwen3-8B"
GOLD_RM_NAME="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

# Open-weight LLM judge (deferred vLLM backend). Used for the preference
# benchmark and, together with the gold RM, for arena_hard — enabled by
# --with_llm_judge. The same model serves both benchmarks (loaded once).
LLM_JUDGE_BACKEND="vllm"
LLM_JUDGE_MODEL="google/gemma-4-31B-it"

# Dataset name (empty = from the run manifest)
DATASET_NAME=""
#DATASET_NAME="ktolnos/helpsteer3-qwen35_annotated_human"
# DATASET_NAME="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"


# Base model name (required for LoRA checkpoints)
# Uncomment and set this if evaluating LoRA checkpoints
#BASE_MODEL_NAME="Qwen/Qwen3-0.6B"

# Output file
OUTPUT_FILE="evaluation_results${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).csv"

# WandB settings
WANDB_PROJECT="policy-evaluation"
WANDB_RUN_NAME="policy_evaluation_$(date +%Y%m%d_%H%M%S)"

SKIP_VALIDATION=1
BENCHMARKS="select,preference,ifeval,arena_hard"

# Policy sampling temperature for all policy generations (BENCHMARK.md §8).
# Empty = the training temperature from the run manifest (falling back to the
# Python default 1.0 for manifest-less runs). Set only to override.
EVAL_TEMPERATURE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) WANDB_RUN_NAME="$2"; shift ;;
        --run_id) WANDB_RUN_ID="$2"; shift ;;
        --kl_base_model_path) KL_BASE_MODEL_PATH="$2"; shift ;;
        --training_rm_path) TRAINING_RM_PATH="$2"; shift ;;
        --dataset_name) DATASET_NAME="$2"; shift ;;
        --checkpoint) CHECKPOINTS_DIR="$2"; shift ;;
        --benchmarks) BENCHMARKS="$2"; shift ;;
        --eval_temperature) EVAL_TEMPERATURE="$2"; shift ;;
        --only_ifeval) BENCHMARKS="ifeval"; NO_SECONDARY_RM=1 ;;
        --only_preference) BENCHMARKS="preference" ;;
        --only_arena_hard) BENCHMARKS="arena_hard"; NO_SECONDARY_RM=1 ;;
        --skip_validation) SKIP_VALIDATION=1 ;;
        --evaluate_chosen_responses) EVALUATE_CHOSEN=1 ;;
        --ifeval_thinking) IFEVAL_THINKING=1 ;;
        --ifeval_use_gold_rm) IFEVAL_USE_GOLD_RM=1 ;;
        --no_ifeval) NO_IFEVAL=1 ;;
        --no_secondary_rm) NO_SECONDARY_RM=1 ;;
        --with_training_rm) WITH_TRAINING_RM=1 ;;
        --with_llm_judge) WITH_LLM_JUDGE=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# arena_hard is scored by the gold RM, plus the open-weight LLM judge when
# --with_llm_judge is passed (both judges run on the same arena_hard responses).
ARENA_HARD_JUDGES="rm:gold_rm"
if [ -n "${WITH_LLM_JUDGE:-}" ]; then
    ARENA_HARD_JUDGES="rm:gold_rm,llm:${LLM_JUDGE_MODEL}"
fi

echo "Running evaluation with the following settings:"
echo "Checkpoints Directory: $CHECKPOINTS_DIR"
echo "Training RM Path: ${TRAINING_RM_PATH:-<run manifest>}"
echo "Sibling RM Path: $SIBLING_RM_PATH"
echo "Gold RM Name: $GOLD_RM_NAME"
echo "Dataset Name: ${DATASET_NAME:-<run manifest>}"
echo "KL Base Model: ${KL_BASE_MODEL_PATH:-<run manifest>}"
echo "Eval Temperature: ${EVAL_TEMPERATURE:-<run manifest>}"
echo "Output File: $OUTPUT_FILE"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run Name: $WANDB_RUN_NAME"
echo "WandB Run ID (resume): ${WANDB_RUN_ID:-<new run>}"
echo "Benchmarks: $BENCHMARKS"
echo "Arena-Hard judges: $ARENA_HARD_JUDGES"
echo "LLM judge: ${WITH_LLM_JUDGE:+enabled ($LLM_JUDGE_BACKEND: $LLM_JUDGE_MODEL)}${WITH_LLM_JUDGE:-disabled}"

export LD_PRELOAD="/nas/ucb/eop/.local/lib/libsqlite3.so.0"

# Run the evaluation script. Manifest-covered settings (training RM, dataset,
# KL base, temperature) are passed only when explicitly set, so the run
# manifest supplies them otherwise.
python evaluate_policy.py \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --sibling_rm_path "$SIBLING_RM_PATH" \
    --gold_rm_name "$GOLD_RM_NAME" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 1 \
    --generation_batch_size 32 \
    --device "cuda" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --benchmarks "$BENCHMARKS" \
    --evaluate_with_training_rm "$([ -n "${WITH_TRAINING_RM:-}" ] && echo True || echo False)" \
    --evaluate_with_llm_judge "$([ -n "${WITH_LLM_JUDGE:-}" ] && echo True || echo False)" \
    --llm_judge_backend "$LLM_JUDGE_BACKEND" \
    --llm_judge_model_name "$LLM_JUDGE_MODEL" \
    --arena_hard_judges "$ARENA_HARD_JUDGES" \
    --baseline_model_path "Qwen/Qwen3-0.6B" \
    --use_dataset_response_as_baseline True \
    $([ -n "${TRAINING_RM_PATH:-}" ] && echo "--training_rm_path $TRAINING_RM_PATH") \
    $([ -n "${DATASET_NAME:-}" ] && echo "--dataset_name $DATASET_NAME") \
    $([ -n "${KL_BASE_MODEL_PATH:-}" ] && echo "--kl_base_model_path $KL_BASE_MODEL_PATH") \
    $([ -n "${EVAL_TEMPERATURE:-}" ] && echo "--eval_temperature $EVAL_TEMPERATURE") \
    --save_eval_dataset_path "evaluation_dataset_${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).jsonl" \
    ${DEBUG_MODE:-} \
    $([ ! -z "${BASE_MODEL_NAME:-}" ] && echo "--base_model_name $BASE_MODEL_NAME") \
    $([ ! -z "${SKIP_VALIDATION:-}" ] && echo "--skip_validation True") \
    $([ ! -z "${EVALUATE_CHOSEN:-}" ] && echo "--evaluate_chosen_responses True") \
    $([ ! -z "${IFEVAL_THINKING:-}" ] && echo "--ifeval_thinking True") \
    $([ ! -z "${IFEVAL_USE_GOLD_RM:-}" ] && echo "--ifeval_use_gold_rm True") \
    $([ ! -z "${NO_IFEVAL:-}" ] && echo "--evaluate_ifeval False") \
    $([ ! -z "${NO_SECONDARY_RM:-}" ] && echo "--secondary_rm_name none") \
    $([ ! -z "${WANDB_RUN_ID:-}" ] && echo "--wandb_run_id $WANDB_RUN_ID") \

#     --subsample_n 25 \

# Notes:
# - --only_ifeval combined with --run_id resumes an existing wandb run and
#   logs only IFEval metrics on the custom "checkpoint" step axis; the
#   preference benchmark (and its RM scoring) is skipped entirely.
# - To disable wandb logging, add: --disable_wandb
# - To enable debug mode, uncomment the DEBUG_MODE line above
# - For LoRA models, uncomment and set BASE_MODEL_NAME above
