#!/bin/bash

#SBATCH --job-name=train_grpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --gres=gpu:1


REPO_ROOT=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/AGENTS.md" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${REPO_ROOT}" ]]; then
    if [[ -f "/nas/ucb/eop/Reward-Model-Overoptimization/AGENTS.md" ]]; then
        REPO_ROOT="/nas/ucb/eop/Reward-Model-Overoptimization"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    fi
fi
if [[ ! -f "${REPO_ROOT}/AGENTS.md" ]]; then
    echo "ERROR: Could not resolve repo root (got '${REPO_ROOT}')." >&2
    exit 1
fi

cd "${REPO_ROOT}/scripts/rlhf/grpo"

log_dir="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_grpo/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
# base_model_name="Qwen/Qwen3-0.6B"
# base_model_name="Qwen/Qwen3-0.6B-Base"
base_model_name="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/20260219_224557_1060185/checkpoint-740"
# base_model_name="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/20260106_012931_1016814/checkpoint-158"
dataset_path="ktolnos/helpsteer3v2_annotated_25pct"
# dataset_path="ktolnos/helpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B"
# dataset_path="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"
# dataset_path="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3-06B-Base_reward_Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3-06B_reward_Qwen-Embedding-8B-42"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/annotated_helpsteer2_Qwen06B-Base_policy_Qwen3-0.6B_42_BT_RM_Qwen3-0.6B_912840_len3000_fulltrain_4e-05_datahelpsteer2-preference-v2_reference"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3_06B_reward_Gemma2_2B_ray_gold_URM_LLama8B/"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold_QRM_Gemma2_27B_0_7748"
export PYTHONPATH="${REPO_ROOT}/rlhf/grpo:${REPO_ROOT}:$PYTHONPATH"

cd "${REPO_ROOT}"
gpu=0 #,1,2,3
#reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
#reward_base_model="nicolinho/QRM-Gemma-2-27B"
#reward_base_model="LxzGordon/URM-LLaMa-3.1-8B"
#reward_base_model="Ray2333/GRM-gemma2-2B-rewardmodel-ft"
learning_rate="1e-6"
per_device_train_batch_size=1
gradient_accumulation_steps=32
beta="0.005"
rm_switch_strategy="ensemble"
ensemble_aggregation="mean"
mix_strategy="disjoint"
mix_ensemble_size=10
if [ -n "$LAST_COMMIT_MESSAGE" ]; then
    COMMIT_MSG="$LAST_COMMIT_MESSAGE"
else
    # shellcheck disable=SC2004
    COMMIT_MSG=$(git log -1 --pretty=%s)
fi
DEFAULT_WANDB_NAME_BASE="${COMMIT_MSG// /_}"
wandb_name_base="$DEFAULT_WANDB_NAME_BASE"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) wandb_name_base="$2"; shift ;;
        *) ;;
    esac
    shift
done
#checkpoint="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40"
echo $SLURM_JOB_ID

PORT_SELECTOR_SCRIPT="${REPO_ROOT}/scripts/common/select_master_port.sh"

if ! MASTER_PORT="$(bash "${PORT_SELECTOR_SCRIPT}" 9900 9999)"; then
    exit 1
fi
export MASTER_PORT


export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export WANDB_PROJECT="grpo"

#  "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
#  "Ray2333/GRM-gemma2-2B-rewardmodel-ft"
# "Reward-Reasoning/RRM-7B"

reward_model_paths=(
#      "nicolinho/QRM-Llama3.1-8B-v2"
    # "Skywork/Skywork-Reward-V2-Qwen3-8B"
     ### helpsteer3v2 5ep
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/600_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/601_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/602_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/603_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/604_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/605_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/606_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/607_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/608_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/609_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
)

num_reward_models=${#reward_model_paths[@]}
run_name_suffix="KL${beta}_${rm_switch_strategy}_${num_reward_models}rms"
if [[ "${rm_switch_strategy}" == "ensemble" || "${rm_switch_strategy}" == "mix" ]]; then
    run_name_suffix="${run_name_suffix}_${ensemble_aggregation}"
fi
if [[ "${rm_switch_strategy}" == "mix" ]]; then
    run_name_suffix="${run_name_suffix}_${mix_strategy}_${mix_ensemble_size}-mixens"
fi
wandb_name="${wandb_name_base}_${run_name_suffix}_${SLURM_JOB_ID}"

export WANDB_RUN_NAME=${wandb_name}
export WANDB_RUN_GROUP=${log_dir}

CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch  \
    --mixed_precision bf16 \
    rlhf/grpo/my_grpo.py \
    --num_generations 16 \
    --num_train_epochs 1 \
    --temperature 1 \
    --max_completion_length 1024 \
    --epsilon_high 0.28 \
    --mask_truncated_completions False \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.1 \
    --vllm_mode "colocate" \
    --beta ${beta} \
    --log_completions True \
    --loss_type "dr_grpo" \
    --log_unique_prompts True \
    --disable_dropout True \
    --bf16 True \
    --dataset_path ${dataset_path} \
    --output_dir ${log_dir}\
    --warmup_ratio=0 \
    --lr_scheduler_type=constant \
    --model_name_or_path ${base_model_name} \
    --reward_model_paths "${reward_model_paths[@]}" \
    --ensemble_aggregation "${ensemble_aggregation}" \
    --save_steps 0.05 \
    --save_only_model True \
    --run_name ${wandb_name} \
    --logging_steps 0.01 \
    --learning_rate ${learning_rate} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --gradient_checkpointing False \
    --scale_rewards 'batch' \
    --trust_remote_code True \
    --reference_rewards False \
    --sigmoid_rewards False \
    --rm_subtract_mean_reward_per_model True \
    --save_generations_path "${log_dir}/generations.csv" \
    --adv_rm_lambda 0.0 \
    --online_pet_enabled False \
    --preference_dataset_path ${dataset_path} \
    --rm_gradient_checkpointing True \
    --move_rm_to_cpu False \
    --move_policy_to_cpu False \
    --pessimistic_loss_weight 0.005 \
    --cql_optimistic_loss_weight 0.005 \
    --rm_update_steps 1 \
    --rm_update_learning_rate 4e-5 \
    --k_top_responses 16 \
    --rm_optimizer 'AdamW' \
    --rm_buffer_size 'full' \
    --pessimistic_gradient_accumulation_steps 16 \
    --bt_gradient_accumulation_steps 16 \
    --adversarial_batch_size 2 \
    --preference_batch_size 2 \
    --rm_switches_multiplier 3 \
    --rm_switch_strategy "${rm_switch_strategy}" \
    --mix_ensemble_size ${mix_ensemble_size} \
    --mix_strategy "${mix_strategy}" \
    --penalize_no_eos True \
    --max_grad_norm 1.0 \
    --vllm_max_model_length 2048 \
    || exit 1

    # --rm_switch_strategy 'mix' or 'sequential' or 'ensemble'
#     --mix_strategy 'sliding' or 'disjoint' or 'random_disjoint'
#    --relu_chosen_reward_loss 0.1 \
#    --relu_chosen_use_rejected_baseline True \
#    --rm_switches_multiplier 50 \
#    --rm_switch_strategy 'sequential' \
#    --use_peft True \
#    --lora_r 32 \
#    --lora_alpha 64 \
#    --lora_target_modules 'all-linear' \
#    --resume_from_checkpoint True \
# 'q_proj' 'k_proj' 'v_proj' 'o_proj' \

# For Adv-RM:
#     --reference_rewards True \
#     --adv_rm_lambda 1.0 \
# Add second reward model

# For 27B:
#    --gradient_checkpointing True \
#    --max_completion_length 256 \
#     --vllm_gpu_memory_utilization 0.08 \
#    --max_prompt_length 512 \

# For RRM:
#    --reward_model_paths "Reward-Reasoning/RRM-7B" \
#    --mask_truncated_completions False \
#    #SBATCH --time=168:00:00

#     --report_to "none" \

echo "running evaluation script for checkpoints in ${log_dir}"
sbatch --export=ALL "${REPO_ROOT}/evaluate_policy.sh" --run_name "${wandb_name}" --kl_base_model_path "${base_model_name}" --checkpoint "${log_dir}"
