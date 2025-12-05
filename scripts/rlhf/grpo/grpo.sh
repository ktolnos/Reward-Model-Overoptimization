#!/bin/bash

#SBATCH --job-name=train_grpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=168:00:00
#SBATCH --qos=high

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

log_dir="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_grpo/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
#base_model_name="Qwen/Qwen3-0.6B" # policy base model
base_model_name="Qwen/Qwen3-0.6B-Base" # policy base model
dataset_path="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"
# dataset_path="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3-06B-Base_reward_Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3-06B_reward_Qwen-Embedding-8B-42"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/annotated_helpsteer2_Qwen06B-Base_policy_Qwen3-0.6B_42_BT_RM_Qwen3-0.6B_912840_len3000_fulltrain_4e-05_datahelpsteer2-preference-v2_reference"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer_anntoated_policy_Qwen3_06B_reward_Gemma2_2B_ray_gold_URM_LLama8B/"
#dataset_path="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer2_gold_QRM_Gemma2_27B_0_7748"
export PYTHONPATH="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/grpo/:/nas/ucb/eop/Reward-Model-Overoptimization/:$PYTHONPATH"

cd ../../../
gpu=0 #,1,2,3
#reward_base_model="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
#reward_base_model="nicolinho/QRM-Gemma-2-27B"
#reward_base_model="LxzGordon/URM-LLaMa-3.1-8B"
#reward_base_model="Ray2333/GRM-gemma2-2B-rewardmodel-ft"
learning_rate="2e-6"
per_device_train_batch_size=1
gradient_accumulation_steps=32
# shellcheck disable=SC2004
wandb_name="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)_lr${learning_rate}_batch$(($per_device_train_batch_size * $gradient_accumulation_steps))_rmQwen06B_Full_helpsteer2_gold"
#checkpoint="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40"
echo $SLURM_JOB_ID

MIN_PORT=9900
MAX_PORT=9999

is_port_in_use() {
    ss -lntu | grep -q ":$1 "
}

IDEAL_PORT=$((MIN_PORT + (SLURM_JOB_ID % (MAX_PORT - MIN_PORT + 1))))

if ! is_port_in_use "${IDEAL_PORT}"; then
    export MASTER_PORT=${IDEAL_PORT}
else
    for port in $(seq ${MIN_PORT} ${MAX_PORT}); do
        if ! is_port_in_use "${port}"; then
            export MASTER_PORT=${port}
            break
        fi
    done
fi

if [[ -z "${MASTER_PORT}" ]]; then
    echo "ERROR: Could not find any free port in the range ${MIN_PORT}-${MAX_PORT}." >&2
    exit 1
fi


export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export WANDB_PROJECT="grpo"
export WANDB_RUN_NAME=${wandb_name}

#  "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
#  "Ray2333/GRM-gemma2-2B-rewardmodel-ft"
# "Reward-Reasoning/RRM-7B"

reward_model_paths=(
#    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-Embedding-8B_43_BT_RM_Qwen3-Embedding-8B_917426_len2000_fulltrain_2e-05_datahelpsteer2-preference-v2/logs/checkpoint-660"
#    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-Embedding-8B_43_BT_RM_Qwen3-Embedding-8B_916704_len2000_fulltrain_2e-05_datahelpsteer2-preference-v2/logs/checkpoint-482"
#    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-Embedding-8B_43_BT_RM_Qwen3-Embedding-8B_916583_len2000_fulltrain_2e-05_datahelpsteer2-preference-v2/logs/checkpoint-290"
#    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-Embedding-8B_42_BT_RM_Qwen3-Embedding-8B_915487_len2000_fulltrain_2e-05_datahelpsteer2-preference-v2/logs/checkpoint-272"
#    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-Embedding-8B_43_BT_RM_Qwen3-Embedding-8B_915731_len2000_fulltrain_2e-05_datahelpsteer2-preference-v2/logs/checkpoint-272"
#     "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256/"
#     "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen3-0.6B_912840_len3000_fulltrain_4e-05_datahelpsteer2-preference-v2/logs/checkpoint-136/"
#      "nicolinho/QRM-Llama3.1-8B-v2"
    #  "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen3-0.6B-helpsteer3_963211_len2000_fulltrain_2e-05_datahelpsteer3-preference-chosenrrejected/logs/checkpoint-1142"
    #  "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen/Qwen3-0.6B_974219_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-1420"
    #  "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen/Qwen3-0.6B_974219_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_43_BT_RM_Qwen/Qwen3-0.6B_974244_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_44_BT_RM_Qwen/Qwen3-0.6B_974245_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_45_BT_RM_Qwen/Qwen3-0.6B_974246_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_46_BT_RM_Qwen/Qwen3-0.6B_974247_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_47_BT_RM_Qwen/Qwen3-0.6B_974254_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_48_BT_RM_Qwen/Qwen3-0.6B_974255_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_49_BT_RM_Qwen/Qwen3-0.6B_974256_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#      "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_50_BT_RM_Qwen/Qwen3-0.6B_974257_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
    #   "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen/Qwen3-0.6B_982417_helpsteer3_gold_full_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B/logs/checkpoint-569"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_10_BT_RM_Qwen/Qwen3-0.6B_994415_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_11_BT_RM_Qwen/Qwen3-0.6B_995139_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_12_BT_RM_Qwen/Qwen3-0.6B_995140_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_13_BT_RM_Qwen/Qwen3-0.6B_995141_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_14_BT_RM_Qwen/Qwen3-0.6B_995142_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_15_BT_RM_Qwen/Qwen3-0.6B_995143_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_16_BT_RM_Qwen/Qwen3-0.6B_995144_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_17_BT_RM_Qwen/Qwen3-0.6B_995145_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_18_BT_RM_Qwen/Qwen3-0.6B_995146_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_19_BT_RM_Qwen/Qwen3-0.6B_995148_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
)

export WANDB_RUN_GROUP=${log_dir}

CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch  \
    --mixed_precision bf16 \
    rlhf/grpo/my_grpo.py \
    --num_generations 16 \
    --num_train_epochs 1 \
    --temperature 1.05 \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --epsilon_high 0.28 \
    --mask_truncated_completions False \
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
    --warmup_ratio=0 \
    --lr_scheduler_type=constant \
    --model_name_or_path ${base_model_name} \
    --reward_model_paths "${reward_model_paths[@]}" \
    --ensemble_aggregation "mean" \
    --save_steps 0.05 \
    --run_name ${wandb_name} \
    --logging_steps 0.01 \
    --learning_rate ${learning_rate} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --gradient_checkpointing False \
    --scale_rewards True \
    --trust_remote_code True \
    --reference_rewards False \
    --sigmoid_rewards False \
    --rm_subtract_mean_reward_per_model True \
    --save_generations_path "${log_dir}/generations.csv" \
    --adv_rm_lambda 0.0 \
    --online_pet_enabled False \
    --preference_dataset_path ${dataset_path} \
    --rm_gradient_checkpointing True \
    --move_rm_to_cpu True \
    --move_policy_to_cpu True \
    --pessimistic_loss_weight 0.01 \
    --cql_optimistic_loss_weight 0.01 \
    --rm_update_steps 1 \
    --rm_update_learning_rate 4e-5 \
    --k_top_responses 16 \
    --rm_optimizer 'AdamW' \
    --rm_buffer_size 'full' \
    --pessimistic_gradient_accumulation_steps 16 \
    --bt_gradient_accumulation_steps 16 \
    --adversarial_batch_size 2 \
    --preference_batch_size 2 \
    --rm_switches_multiplier 5 \
    --rm_switch_strategy 'ensemble' \
    || exit 1
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
sbatch --export=ALL,CHECKPOINTS_DIR_OVERRIDE="${log_dir}" /nas/ucb/eop/Reward-Model-Overoptimization/evaluate_policy.sh
