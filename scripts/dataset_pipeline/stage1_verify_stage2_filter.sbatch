#!/bin/bash
#SBATCH --job-name=prep_stage1_stage2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --qos=default

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
TOKENIZER_NAME="Qwen/Qwen3-0.6B"
SEED=42
TRAIN_RATIO=0.9
TEST_RATIO=0.05
HELDOUT_RATIO=0.05
MAX_PROMPT_TOKENS=1024
MAX_RESPONSE_TOKENS=1024
MAX_CONVERSATION_TOKENS=2048
MAX_ERRORS=20
PRIVATE=0
TRUST_REMOTE_CODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dataset) SOURCE_DATASET="$2"; shift 2 ;;
    --output-dataset) OUTPUT_DATASET="$2"; shift 2 ;;
    --tokenizer-name) TOKENIZER_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --test-ratio) TEST_RATIO="$2"; shift 2 ;;
    --heldout-ratio) HELDOUT_RATIO="$2"; shift 2 ;;
    --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
    --max-response-tokens) MAX_RESPONSE_TOKENS="$2"; shift 2 ;;
    --max-conversation-tokens) MAX_CONVERSATION_TOKENS="$2"; shift 2 ;;
    --max-errors) MAX_ERRORS="$2"; shift 2 ;;
    --private) PRIVATE=1; shift ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1; shift ;;
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

VERIFY_ARGS=(
  --source-dataset "${SOURCE_DATASET}"
  --tokenizer-name "${TOKENIZER_NAME}"
  --max-errors "${MAX_ERRORS}"
)
FILTER_ARGS=(
  --source-dataset "${SOURCE_DATASET}"
  --output-dataset "${OUTPUT_DATASET}"
  --tokenizer-name "${TOKENIZER_NAME}"
  --seed "${SEED}"
  --train-ratio "${TRAIN_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --heldout-ratio "${HELDOUT_RATIO}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS}"
  --max-conversation-tokens "${MAX_CONVERSATION_TOKENS}"
  --max-errors "${MAX_ERRORS}"
)

if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then
  VERIFY_ARGS+=(--trust-remote-code)
  FILTER_ARGS+=(--trust-remote-code)
fi
if [[ "${PRIVATE}" -eq 1 ]]; then
  FILTER_ARGS+=(--private)
fi

echo "Running Stage 1 verification for ${SOURCE_DATASET}"
python3 scripts/dataset_pipeline/stage1_verify_dataset.py "${VERIFY_ARGS[@]}"

echo "Running Stage 2 filtering/splitting/upload to ${OUTPUT_DATASET}"
python3 scripts/dataset_pipeline/stage2_filter_split_upload.py "${FILTER_ARGS[@]}"

echo "Stage 1+2 completed successfully."
