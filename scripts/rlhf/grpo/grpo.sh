#!/bin/bash

#SBATCH --job-name=train_grpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:A100-PCI-80GB:2
#SBATCH --time=24:00:00
#SBATCH --qos=normal

cd /nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/grpo
export HF_HOME="/nas/ucb/eop/cache"
export TMPDIR="/nas/ucb/eop/temp"
export TEMP="/nas/ucb/eop/temp"
export TMP="/nas/ucb/eop/temp"
export PYTHONPYCACHEPREFIX="/nas/ucb/eop/temp/pycache"
export TORCHINDUCTOR_CACHE_DIR="/nas/ucb/eop/temp/torchinductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE="/nas/ucb/eop/temp/fx_graph_cache"
export VLLM_CONFIG_ROOT="/nas/ucb/eop/cache/vllm_config"
export VLLM_DISABLE_COMPILE_CACHE="1"
export VLLM_CACHE_ROOT="/nas/ucb/eop/cache/"

export WANDB_DIR="/nas/ucb/eop/wandb"
export WANDB_CACHE_DIR="/nas/ucb/eop/cache/wandb"
export WANDB_DATA_DIR="/nas/ucb/eop/cache/wandb-data"
export WANDB_ARTIFACT_DIR="/nas/ucb/eop/cache/wandb-artifacts"

#resume_from_checkpoint=""

log_dir="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_grpo/$(date +%Y%m%d_%H%M%S)"
if [[ -v resume_from_checkpoint ]]; then
  log_dir="${resume_from_checkpoint}"
fi
base_model_name="Qwen/Qwen3-0.6B-Base" # policy base model
#dataset_path="gagan3012/helpsteer2-preference-v2"
dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3-06B-Base_reward_Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3-06B_reward_Qwen-Embedding-8B-42"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/annotated_helpsteer2_Qwen06B-Base_policy_Qwen3-0.6B_42_BT_RM_Qwen3-0.6B_912840_len3000_fulltrain_4e-05_datahelpsteer2-preference-v2_reference"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3_06B_reward_Gemma2_2B_ray_gold_URM_LLama8B/"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold_QRM_Gemma2_27B_0_7748"
export PYTHONPATH="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/grpo/:/nas/ucb/eop/Reward-Model-Overoptimization/:$PYTHONPATH"

cd ../../../
#reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
#reward_base_model="nicolinho/QRM-Gemma-2-27B"
#reward_base_model="LxzGordon/URM-LLaMa-3.1-8B"
#reward_base_model="Ray2333/GRM-gemma2-2B-rewardmodel-ft"
learning_rate="1e-7"

# --- Batch Size Configuration ---
# -- Policy Training --
POLICY_EFFECTIVE_BATCH_SIZE=32
PER_DEVICE_POLICY_BATCH_SIZE=1
# -- RM Training (Adversarial) --
ADV_EFFECTIVE_BATCH_SIZE=32
PER_DEVICE_ADV_BATCH_SIZE=2
# -- RM Training (Preference) --
PREF_EFFECTIVE_BATCH_SIZE=32
PER_DEVICE_PREF_BATCH_SIZE=2
# --- End Batch Size Configuration ---

# --- Automatic Calculation of Accumulation Steps ---
NUM_GPUS=$(CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Policy
POLICY_GLOBAL_BATCH_SIZE=$(($PER_DEVICE_POLICY_BATCH_SIZE * $NUM_GPUS))
if [ $(($POLICY_EFFECTIVE_BATCH_SIZE % $POLICY_GLOBAL_BATCH_SIZE)) -ne 0 ]; then
    echo "Error: POLICY_EFFECTIVE_BATCH_SIZE ($POLICY_EFFECTIVE_BATCH_SIZE) is not divisible by the global batch size ($POLICY_GLOBAL_BATCH_SIZE)."
    exit 1
fi
POLICY_ACCUMULATION_STEPS=$(($POLICY_EFFECTIVE_BATCH_SIZE / $POLICY_GLOBAL_BATCH_SIZE))

# Adversarial
ADV_GLOBAL_BATCH_SIZE=$(($PER_DEVICE_ADV_BATCH_SIZE * $NUM_GPUS))
if [ $(($ADV_EFFECTIVE_BATCH_SIZE % $ADV_GLOBAL_BATCH_SIZE)) -ne 0 ]; then
    echo "Error: ADV_EFFECTIVE_BATCH_SIZE ($ADV_EFFECTIVE_BATCH_SIZE) is not divisible by the global batch size ($ADV_GLOBAL_BATCH_SIZE)."
    exit 1
fi
ADV_ACCUMULATION_STEPS=$(($ADV_EFFECTIVE_BATCH_SIZE / $ADV_GLOBAL_BATCH_SIZE))

# Preference
PREF_GLOBAL_BATCH_SIZE=$(($PER_DEVICE_PREF_BATCH_SIZE * $NUM_GPUS))
if [ $(($PREF_EFFECTIVE_BATCH_SIZE % $PREF_GLOBAL_BATCH_SIZE)) -ne 0 ]; then
    echo "Error: PREF_EFFECTIVE_BATCH_SIZE ($PREF_EFFECTIVE_BATCH_SIZE) is not divisible by the global batch size ($PREF_GLOBAL_BATCH_SIZE)."
    exit 1
fi
PREF_ACCUMULATION_STEPS=$(($PREF_EFFECTIVE_BATCH_SIZE / $PREF_GLOBAL_BATCH_SIZE))

echo "--- Calculated Training Configuration ---"
echo "  - Num GPUs: $NUM_GPUS"
echo "  - Policy Accumulation Steps: $POLICY_ACCUMULATION_STEPS"
echo "  - Adversarial Accumulation Steps: $ADV_ACCUMULATION_STEPS"
echo "  - Preference Accumulation Steps: $PREF_ACCUMULATION_STEPS"
echo "---------------------------------------"
# --- End Automatic Calculation ---


wandb_name="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)_lr${learning_rate}_batch${POLICY_EFFECTIVE_BATCH_SIZE}_rmQwen06B_Full_helpsteer2_gold"
echo $SLURM_JOB_ID

export WANDB_PROJECT="grpo"
export WANDB_RUN_NAME=${wandb_name}

reward_model_paths=(
     "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
)

export WANDB_RUN_GROUP=${log_dir}

accelerate launch --config_file scripts/accelerate_configs/accelerate_deepspeed_zero3.yaml \
    rlhf/grpo/my_grpo.py \
    --deepspeed "scripts/accelerate_configs/deepspeed_zero3.json" \
    --num_generations 16 \
    --num_train_epochs 1 \
    --temperature 0.9 \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --epsilon_high 0.28 \
    --mask_truncated_completions True \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.1 \
    --vllm_mode "colocate" \
    --beta 0.0 \
    --log_completions True \
    --loss_type "dr_grpo" \
    --wandb_log_unique_prompts True \
    --disable_dropout True \
    --bf16 True \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir}\
    --warmup_ratio=0.1 \
    --lr_scheduler_type=constant_with_warmup \
    --model_name_or_path ${base_model_name} \
    --reward_model_paths "${reward_model_paths[@]}" \
    --ensemble_aggregation "min" \
    --save_steps 0.0083333333 \
    --run_name ${wandb_name} \
    --logging_steps 0.01 \
    --learning_rate ${learning_rate} \
    --gradient_checkpointing False \
    --scale_rewards False \
    --trust_remote_code True \
    --reference_rewards False \
    --sigmoid_rewards False \
    --save_generations_path "${log_dir}/generations.csv" \
    --adv_rm_lambda 0.0 \
    --online_pet_enabled True \
    --preference_dataset_path "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3-06B-Base_reward_Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05" \
    --rm_gradient_checkpointing True \
    --pessimistic_loss_weight 10000.0 \
    --rm_update_steps 1 \
    --rm_update_learning_rate 4e-5 \
    --k_top_responses 16 \
    --rm_optimizer 'AdamW' \
    --rm_buffer_size 512 \
    --resume_from_checkpoint "${resume_from_checkpoint:-}" \
    --per_device_train_batch_size ${PER_DEVICE_POLICY_BATCH_SIZE} \
    --gradient_accumulation_steps ${POLICY_ACCUMULATION_STEPS} \
    --adversarial_batch_size ${PER_DEVICE_ADV_BATCH_SIZE} \
    --pessimistic_gradient_accumulation_steps ${ADV_ACCUMULATION_STEPS} \
    --preference_batch_size ${PER_DEVICE_PREF_BATCH_SIZE} \
    --bt_gradient_accumulation_steps ${PREF_ACCUMULATION_STEPS} \
    || exit 1

echo "running evaluation script for checkpoints in ${log_dir}"
sbatch --export=ALL,CHECKPOINTS_DIR_OVERRIDE="${log_dir}" /nas/ucb/eop/Reward-Model-Overoptimization/evaluate_policy.sh
