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
# base_model_name="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/20260320_224539_1070739/checkpoint-740" # 3 4B-Base
base_model_name="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_sft/20260219_224557_1060185" # 3 0.6B-Base
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
learning_rate="1e-5"
per_device_train_batch_size=1
gradient_accumulation_steps=32
beta="0"
rm_switch_strategy="sequential" # "ensemble" or "sequential" or "mix"
ensemble_aggregation="mean" # "mean" or "min" or "uwo"
mix_strategy="disjoint" # "disjoint" or "sliding" or "random_disjoint"
mix_ensemble_size=10
uwo_lambda=10
rm_switches_multiplier=3
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
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/600_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/601_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/602_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/603_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/604_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/605_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/606_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/607_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/608_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/609_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
###
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/600_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/601_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/602_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/603_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/604_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/605_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/606_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/607_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/608_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/609_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/610_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/611_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/612_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/613_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/614_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/615_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/616_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/617_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/618_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/619_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/620_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/621_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/622_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/623_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/624_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/625_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/626_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/627_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/628_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/629_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/630_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/631_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/632_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/633_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/634_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/635_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/636_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/637_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/638_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/639_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/640_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/641_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/642_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/643_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/644_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/645_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/646_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/647_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/648_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/649_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/650_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/651_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/652_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/653_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/654_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/655_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/656_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/657_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/658_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/659_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/660_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/661_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/662_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/663_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/664_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/665_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/666_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/667_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/668_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/669_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/670_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/671_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/672_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/673_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/674_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/675_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/676_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/677_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/678_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/679_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/680_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/681_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/682_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/683_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/684_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/685_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/686_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/687_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/688_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/689_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/690_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/691_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/692_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/693_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/694_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/695_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/696_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/697_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/698_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/699_Qwen3-0.6B_len2000_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1564"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/1_Qwen3-4B-Instruct-2507_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-500"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/1_Qwen3-4B-Instruct-2507_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1000"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/1_Qwen3-4B-Instruct-2507_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1500"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/1_Qwen3-4B-Instruct-2507_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-2000"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/1_Qwen3-4B-Instruct-2507_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-2500"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/1_Qwen3-4B-Instruct-2507_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-3000"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/1_Qwen3-4B-Instruct-2507_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-3128"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/19_Qwen3-4B-Base_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-3128"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/19_Qwen3-4B_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-3128"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/19_Qwen3.5-4B_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
    # "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/19_Qwen3-8B_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-3128"  
    "/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/19_Qwen3.5-9B_len2048_fulltrain_2e-05_datahelpsteer3v2_annotated_Skywork-Skywork-Reward-V2-Llama-3-1-8B/logs/checkpoint-1173"
)

num_reward_models=${#reward_model_paths[@]}
run_name_suffix="KL${beta}_${num_reward_models}rms_${rm_switch_strategy}"
if [[ "${rm_switch_strategy}" == "mix" ]]; then
    run_name_suffix="${run_name_suffix}_${mix_strategy}_${mix_ensemble_size}-mixens"
fi
if [[ "${rm_switch_strategy}" == "sequential" ]]; then
    run_name_suffix="${run_name_suffix}${rm_switches_multiplier}x"
fi
if [[ "${rm_switch_strategy}" == "ensemble" || "${rm_switch_strategy}" == "mix" ]]; then
    run_name_suffix="${run_name_suffix}_${ensemble_aggregation}"
fi
if [[ "${ensemble_aggregation}" == "uwo" ]]; then
    run_name_suffix="${run_name_suffix}${uwo_lambda}"
fi

wandb_name="${wandb_name_base}_${run_name_suffix}_${SLURM_JOB_ID}"

export WANDB_RUN_NAME=${wandb_name}
export WANDB_RUN_GROUP=${log_dir}

CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch  \
    --mixed_precision bf16 \
    rlhf/grpo/my_grpo.py \
    --report_to wandb \
    --num_generations 16 \
    --num_train_epochs 1 \
    --temperature 1 \
    --epsilon_high 0.28 \
    --mask_truncated_completions False \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.2 \
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
    --gradient_checkpointing True \
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
    --rm_scale_reward_by_std_per_model True \
    --uwo_lambda ${uwo_lambda} \
    || exit 1
#     --clip_reward_max 3.0 \

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
