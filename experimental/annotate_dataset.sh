#!/bin/bash

#SBATCH --job-name=annotate_dataset
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=4:00:00
#SBATCH --qos=high

cd /nas/ucb/eop/Reward-Model-Overoptimization/experimental/
export HF_HOME="/nas/ucb/eop/cache"
export PYTHONPATH="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/grpo/:/nas/ucb/eop/Reward-Model-Overoptimization/:$PYTHONPATH"

python dataset_annotation.py
