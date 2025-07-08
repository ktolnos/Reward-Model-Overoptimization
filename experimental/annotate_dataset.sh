#!/bin/bash

#SBATCH --job-name=annotate_dataset
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=12:00:00
#SBATCH --qos=high

cd /nas/ucb/eop/Reward-Model-Overoptimization/experimental/
export HF_HOME="/nas/ucb/eop/cache"
export PYTHONPATH="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/grpo/:/nas/ucb/eop/Reward-Model-Overoptimization/:$PYTHONPATH"
export TMPDIR="/nas/ucb/eop/temp"
export TEMP="/nas/ucb/eop/temp"
export TMP="/nas/ucb/eop/temp"
export PYTHONPYCACHEPREFIX="/nas/ucb/eop/temp/pycache"
export TORCHINDUCTOR_CACHE_DIR="/nas/ucb/eop/temp/torchinductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE="/nas/ucb/eop/temp/fx_graph_cache"
export VLLM_CONFIG_ROOT="/nas/ucb/eop/cache/vllm_config"
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_DIR "/nas/ucb/eop/wandb"
export WANDB_CACHE_DIR "/nas/ucb/eop/cache/wandb"
export WANDB_DATA_DIR "/nas/ucb/eop/cache/wandb-data"
export WANDB_ARTIFACT_DIR "/nas/ucb/eop/cache/wandb-artifacts"

python dataset_annotation.py
