#!/bin/bash

#SBATCH --job-name=annotate_dataset
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=12:00:00
#SBATCH --qos=high

source ~/.bashrc

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -f "${REPO_ROOT}/AGENTS.md" ]]; then
  echo "ERROR: Could not resolve repo root (got '${REPO_ROOT}')." >&2
  exit 1
fi
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/nas/ucb/eop/cache}"
export PYTHONPATH="${REPO_ROOT}/rlhf/grpo:${REPO_ROOT}:${PYTHONPATH:-}"

# Normalize token env names so huggingface_hub can read either convention.
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

python3 scripts/dataset_pipeline/stage3_annotate_and_upload.py "$@"
