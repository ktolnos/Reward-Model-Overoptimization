#!/bin/bash
#SBATCH --job-name=prep_stage4_subsample
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --qos=default

source ~/.bashrc

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="/nas/ucb/eop/cache"
export TMPDIR="/nas/ucb/eop/temp"
export TEMP="/nas/ucb/eop/temp"
export TMP="/nas/ucb/eop/temp"
export PYTHONPYCACHEPREFIX="/nas/ucb/eop/temp/pycache"
export PYTHONPATH="${REPO_ROOT}/rlhf/grpo:${REPO_ROOT}:${PYTHONPATH:-}"

SOURCE_DATASET=""
OUTPUT_DATASET=""
FRACTION=0.25
SEED=42
PRIVATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dataset) SOURCE_DATASET="$2"; shift 2 ;;
    --output-dataset) OUTPUT_DATASET="$2"; shift 2 ;;
    --fraction) FRACTION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --private) PRIVATE=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SOURCE_DATASET}" || -z "${OUTPUT_DATASET}" ]]; then
  echo "Both --source-dataset and --output-dataset are required." >&2
  exit 2
fi

ARGS=(
  --source-dataset "${SOURCE_DATASET}"
  --output-dataset "${OUTPUT_DATASET}"
  --fraction "${FRACTION}"
  --seed "${SEED}"
)
if [[ "${PRIVATE}" -eq 1 ]]; then
  ARGS+=(--private)
fi

echo "Running Stage 4 subsampling ${SOURCE_DATASET} -> ${OUTPUT_DATASET}"
python3 scripts/dataset_pipeline/stage4_subsample_upload.py "${ARGS[@]}"

echo "Stage 4 completed successfully."
