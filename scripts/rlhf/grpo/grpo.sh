#!/bin/bash

#SBATCH --job-name=train_grpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=48:00:00
#SBATCH --qos=default

#SELECTGPU A100-SXM4-80GB, A100-PCI-80GB

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
base_model_name="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/20260106_012931_1016814/checkpoint-158"
dataset_path="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"
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
if [ -n "$LAST_COMMIT_MESSAGE" ]; then
    COMMIT_MSG="$LAST_COMMIT_MESSAGE"
else
    # shellcheck disable=SC2004
    COMMIT_MSG=$(git log -1 --pretty=%s)
fi
DEFAULT_WANDB_NAME="${COMMIT_MSG// /_}_${SLURM_JOB_ID}"
wandb_name="$DEFAULT_WANDB_NAME"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) wandb_name="$2_${SLURM_JOB_ID}"; shift ;;
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
    # "Skywork/Skywork-Reward-V2-Qwen3-8B"
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
    #######  
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_15_BT_RM_Qwen/Qwen3-0.6B_995143_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_16_BT_RM_Qwen/Qwen3-0.6B_995144_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_17_BT_RM_Qwen/Qwen3-0.6B_995145_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_18_BT_RM_Qwen/Qwen3-0.6B_995146_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_19_BT_RM_Qwen/Qwen3-0.6B_995148_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
  
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_10_BT_RM_Qwen/Qwen3-0.6B_994415_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_11_BT_RM_Qwen/Qwen3-0.6B_995139_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_12_BT_RM_Qwen/Qwen3-0.6B_995140_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_13_BT_RM_Qwen/Qwen3-0.6B_995141_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_14_BT_RM_Qwen/Qwen3-0.6B_995142_helpsteer3_gold_10k_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-284/"
    #####
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/100_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/101_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/102_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/103_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/104_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/105_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/106_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/107_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/108_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/109_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/110_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/111_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/112_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/113_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/114_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/115_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/116_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/117_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/118_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/119_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/120_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/121_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/122_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/123_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/124_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/125_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/126_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/127_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/128_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/129_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/130_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/131_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/132_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/133_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/134_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/135_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/136_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/137_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/138_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/139_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/140_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/141_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/142_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/143_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/144_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/145_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/146_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/147_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/148_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/149_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/150_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/151_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/152_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/153_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/154_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/155_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/156_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/157_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/158_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/159_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/160_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/161_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/162_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/163_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/164_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/165_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/166_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/167_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/168_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/169_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/170_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/171_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/172_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/173_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/174_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/175_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/176_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/177_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/178_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/179_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/180_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/181_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/182_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/183_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/184_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/185_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/186_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/187_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/188_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/189_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/190_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/191_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/192_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/193_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/194_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/195_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/196_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/197_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/198_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/199_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/200_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-218"
    ###
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/449_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/450_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/451_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/452_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/453_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/454_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/455_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/456_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/457_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/458_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/459_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/460_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/461_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/462_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/463_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/464_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/465_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/466_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/467_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/468_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/469_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/470_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/471_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/472_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/473_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/474_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/475_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/476_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/477_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/478_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/479_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/480_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/481_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/482_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/483_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/484_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/485_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/486_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/487_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/488_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/489_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/490_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/491_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/492_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/493_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/494_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/495_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/496_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/497_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/498_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/499_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/400_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/401_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/402_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/403_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/404_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/405_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/406_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/407_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/408_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/409_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/410_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/411_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/412_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/413_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/414_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/415_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/416_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/417_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/418_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/419_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/420_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/421_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/422_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/423_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/424_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/425_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/426_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/427_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/428_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/429_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/430_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/431_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/432_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/433_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/434_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/435_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/436_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/437_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/438_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/439_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/440_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/441_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/442_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/443_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/444_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/445_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/446_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/447_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/448_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-545"
)

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
    --beta 0.005 \
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
    --ensemble_aggregation "uwo" \
    --save_steps 0.05 \
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
    --rm_switch_strategy 'mix' \
    --mix_ensemble_size 10 \
    --mix_strategy 'disjoint' \
    --penalize_no_eos True \
    --max_grad_norm 1.0 \
    --vllm_max_model_length 2048 \
    || exit 1

    # --rm_switch_strategy 'mix' or 'sequential' or 'ensemble' or 'random_disjoint'
#     --mix_strategy 'sliding' or 'disjoint'
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
