# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase for "Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs" (NeurIPS 2024). The repository implements training and evaluation of Generalizable Reward Models (GRM) and supports various RLHF algorithms including PPO, GRPO, and Best-of-N sampling.

Current experiments are focused on GRPO, assume GRPO unless specified otherwise

## Architecture

### Core Components

- **reward_models/** - Reward model training implementations
  - `run_reward_models_train.py` - Main training script for BT (Bradley-Terry) reward models -- currently used by the experiments by default
  - `run_grm_reward_train.py` - Training script for GRM (regularized) reward models
  - `grm_utils.py` / `utils.py` - Utilities for reward model training
  - `reward_trainer.py` / `grm_reward_trainer.py` - Custom trainers

- **rlhf/** - RLHF algorithm implementations
  - `ppo/` - Proximal Policy Optimization
    - `my_ppo.py` - Main PPO implementation
    - `ppo_grm.py` - PPO with GRM reward models
    - `ppo_rm_ensemble.py` - PPO with ensemble reward models
  - `grpo/` - Group Relative Policy Optimization
    - `my_grpo.py` - Main GRPO implementation
    - `grpo_utils.py` - GRPO utilities
  - `sft/` - Supervised Fine-Tuning
    - `my_sft.py` - SFT implementation
  - `bon/` - Best-of-N sampling (see `rlhf/bon/README.md`)
  - `data_generation/` - Dataset generation utilities for RLHF

- **rm_eval/** - Reward model evaluation
  - `eval.py` - Main evaluation script for BT models
  - `eval_grm.py` - Evaluation script for GRM models
  - `load_eval_datasets.py` - Dataset loading for evaluation

- **experimental/** - Experimental dataset annotation
  - `dataset_annotation.py` - Annotate datasets with reward model scores
  - `data/` - Preprocessed/annotated datasets

- **scripts/** - Shell scripts for running experiments
  - All scripts use SLURM by default (with `#SBATCH` directives)
  - Most scripts auto-detect available ports and set up distributed training

### Key Utilities

- `reward_utils.py` - Reward model utilities including reasoning reward models (e.g., Skywork prompts)
- `evaluate_policy.py` - Policy evaluation after RLHF training -- main evaluation script for the project.

## Common Commands

### Training Reward Models

Imprtant: Claude code runs not on the compute cluster, so running most scripts is impossible in the local environment (which has 12Gb cuda GPU). Don't try to run GRPO training and such locally. The following is just for claude to consult how the training is currenly using the python scripts.


### Training reward models
```bash
cd scripts
sbatch my_train_bt_rm_full.sh
```

### Training the base policy with SFT
```bash
cd scripts/rlhf/sft
sh my_train_my_sft.sh
```

### RL training
```bash
cd scripts/rlhf/grpo
sbatch grpo.sh
```

### Dataset Annotation

```bash
cd experimental
sh annotate_dataset.sh
```

This uses `dataset_annotation.py` to annotate datasets with reward model scores.

## Important Patterns and Conventions

### SLURM Integration

- All training scripts in `scripts/` include SLURM directives (`#SBATCH`)
- Scripts auto-detect free ports in range 9900-9999 for distributed training
- Scripts typically accept command-line arguments for seeds, wandb names, etc.
- Logs are saved with timestamps and SLURM job IDs in the directory names

### Model Paths

- Base models: HuggingFace model IDs (e.g., `Qwen/Qwen3-0.6B`)
- Trained models: Stored on the cluster. Trained models are unaccessible from local environments.

### Dataset Formats

- Preference datasets typically have `chosen` and `rejected` fields
- Common datasets:
  - `ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k`
  - `gagan3012/helpsteer2-preference-v2`

### Reward Model Types
- **BT (Bradley-Terry)**: Standard preference-based reward models
  - Training: `reward_models/run_reward_models_train.py`
  - Use classifier head for sequence classification

- **GRM (Generalizable Reward Model)**: With hidden state regularization
  - Training: `reward_models/run_grm_reward_train.py`
  - Better generalization via regularization

- **Reasoning Reward Models**: Use LLM-as-judge approach (e.g., Skywork)
  - Identified by `hasattr(model, 'lm_head')` in `reward_utils.py`
  - Use special prompts (see `Skywork_SYSTEM_PROMPT` in `reward_utils.py`)
  
# Current experiments

Current experiments focus on BT reward models.

The main pipeline for current experiments involves trainig multiple BT reward models, training SFT policy model, then experimenting with training the policy with GRPO starting from SFT checkpoing. Each GRPO run is automatically evaluated using evaluate_policy.sh.  

# Results

Using sequential switching of reward models allows to train Qwen3-0.6B on 10k datapoints from Helpsteer 3 dataset at very low KL penalty. Lower KL penalty allows to achive much higher gold reward than standard training. We have also successfully finetuned a base model using only RL to a decent gold reward. Out of all our experiments, big ensembles seem to be the best way to prevent reward hacking, with mixed strategy of sequentually switching the mean ensemble of 10 reward models being the best so far (uwo is still churning).

