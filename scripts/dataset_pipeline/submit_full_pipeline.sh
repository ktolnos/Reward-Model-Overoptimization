#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_DATASET=""
REWARD_MODEL=""
PREFIX=""
NAMESPACE=""
TOKENIZER_NAME="Qwen/Qwen3-0.6B"
SEED=42
TRAIN_RATIO=0.9
TEST_RATIO=0.05
HELDOUT_RATIO=0.05
SUBSAMPLE_FRACTION=0.25
MAX_PROMPT_TOKENS=1000
MAX_RESPONSE_TOKENS=1000
MAX_CONVERSATION_TOKENS=2000
MAX_ERRORS=20
PRIVATE=0
TRUST_REMOTE_CODE=0
SKIP_STAGE12=0
SKIP_STAGE3=0

usage() {
  cat <<EOF
Usage: $0 \
  --source-dataset <hf_repo> \
  --reward-model <hf_or_local_model> \
  --prefix <name_prefix> \
  --namespace <hf_user_or_org> \
  [--tokenizer-name <hf_tokenizer>] \
  [--seed <int>] \
  [--train-ratio <float>] [--test-ratio <float>] [--heldout-ratio <float>] \
  [--subsample-fraction <float>] \
  [--max-prompt-tokens <int>] [--max-response-tokens <int>] [--max-conversation-tokens <int>] \
  [--max-errors <int>] [--private] [--trust-remote-code] \
  [--skip-stage12] [--skip-stage3]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dataset) SOURCE_DATASET="$2"; shift 2 ;;
    --reward-model) REWARD_MODEL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --tokenizer-name) TOKENIZER_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --test-ratio) TEST_RATIO="$2"; shift 2 ;;
    --heldout-ratio) HELDOUT_RATIO="$2"; shift 2 ;;
    --subsample-fraction) SUBSAMPLE_FRACTION="$2"; shift 2 ;;
    --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
    --max-response-tokens) MAX_RESPONSE_TOKENS="$2"; shift 2 ;;
    --max-conversation-tokens) MAX_CONVERSATION_TOKENS="$2"; shift 2 ;;
    --max-errors) MAX_ERRORS="$2"; shift 2 ;;
    --private) PRIVATE=1; shift ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1; shift ;;
    --skip-stage12) SKIP_STAGE12=1; shift ;;
    --skip-stage3) SKIP_STAGE3=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${SOURCE_DATASET}" || -z "${REWARD_MODEL}" || -z "${PREFIX}" || -z "${NAMESPACE}" ]]; then
  usage
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HF token is missing in the current shell." >&2
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before submitting the pipeline." >&2
  exit 2
fi

# Normalize token env names so downstream tools can read either convention.
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

SBATCH_EXPORT="ALL"
SBATCH_EXPORT+=",HF_TOKEN,HUGGINGFACE_HUB_TOKEN"

sanitize() {
  echo "$1" | sed -e 's#[/:]#-#g' -e 's#[^a-zA-Z0-9_-]#-#g' -e 's#--*#-#g' -e 's#^-##' -e 's#-$##'
}

REWARD_SUFFIX="$(sanitize "${REWARD_MODEL}")"

FILTERED_DATASET="${NAMESPACE}/$(sanitize "${PREFIX}")_filtered"
ANNOTATED_DATASET="${NAMESPACE}/$(sanitize "${PREFIX}")_annotated_${REWARD_SUFFIX}"
SUBSAMPLED_DATASET="${NAMESPACE}/$(sanitize "${PREFIX}")_annotated_25pct"

# Hugging Face SQL (run on the ANNOTATED_DATASET viewer) to check RM agreement accuracy:
# Overall accuracy across splits:
# SELECT AVG(CASE WHEN does_gold_agree_with_original THEN 1.0 ELSE 0.0 END) AS rm_accuracy
# FROM (
#   SELECT does_gold_agree_with_original FROM train
#   UNION ALL
#   SELECT does_gold_agree_with_original FROM test
#   UNION ALL
#   SELECT does_gold_agree_with_original FROM heldout
# );
#
# Per-split accuracy:
# SELECT split, COUNT(*) AS n_examples,
#        AVG(CASE WHEN does_gold_agree_with_original THEN 1.0 ELSE 0.0 END) AS rm_accuracy
# FROM (
#   SELECT 'train' AS split, does_gold_agree_with_original FROM train
#   UNION ALL
#   SELECT 'test' AS split, does_gold_agree_with_original FROM test
#   UNION ALL
#   SELECT 'heldout' AS split, does_gold_agree_with_original FROM heldout
# )
# GROUP BY split
# ORDER BY split;

echo "Source dataset:     ${SOURCE_DATASET}"
echo "Filtered dataset:   ${FILTERED_DATASET}"
echo "Annotated dataset:  ${ANNOTATED_DATASET}"
echo "Subsampled dataset: ${SUBSAMPLED_DATASET}"

COMMON_STAGE12_ARGS=(
  --source-dataset "${SOURCE_DATASET}"
  --output-dataset "${FILTERED_DATASET}"
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
if [[ "${PRIVATE}" -eq 1 ]]; then
  COMMON_STAGE12_ARGS+=(--private)
fi
if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then
  COMMON_STAGE12_ARGS+=(--trust-remote-code)
fi

JOB1=""
if [[ "${SKIP_STAGE12}" -eq 1 ]]; then
  echo "Skipping Stage 1+2 submission (--skip-stage12)."
  echo "Expected existing filtered dataset repo: ${FILTERED_DATASET}"
else
  JOB1="$(sbatch --parsable --export "${SBATCH_EXPORT}" --chdir "${REPO_ROOT}" scripts/dataset_pipeline/stage1_verify_stage2_filter.sbatch "${COMMON_STAGE12_ARGS[@]}")"
  echo "Submitted Stage 1+2 job: ${JOB1}"
fi

STAGE3_ARGS=(
  --source-dataset "${FILTERED_DATASET}"
  --output-dataset "${ANNOTATED_DATASET}"
  --reward-model "${REWARD_MODEL}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
  --max-conversation-tokens "${MAX_CONVERSATION_TOKENS}"
  --validation-tokenizer-name "${TOKENIZER_NAME}"
)
if [[ "${PRIVATE}" -eq 1 ]]; then
  STAGE3_ARGS+=(--private)
fi
if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then
  STAGE3_ARGS+=(--trust-remote-code)
fi

JOB2=""
if [[ "${SKIP_STAGE3}" -eq 1 ]]; then
  echo "Skipping Stage 3 submission (--skip-stage3)."
  echo "Expected existing annotated dataset repo: ${ANNOTATED_DATASET}"
else
  STAGE3_SBATCH_ARGS=(
    --parsable
    --export "${SBATCH_EXPORT}"
    --chdir "${REPO_ROOT}"
  )
  if [[ -n "${JOB1}" ]]; then
    STAGE3_SBATCH_ARGS+=(--dependency="afterok:${JOB1}")
  fi

  JOB2="$(sbatch "${STAGE3_SBATCH_ARGS[@]}" experimental/annotate_dataset.sh "${STAGE3_ARGS[@]}")"
  if [[ -n "${JOB1}" ]]; then
    echo "Submitted Stage 3 job: ${JOB2} (depends on ${JOB1})"
  else
    echo "Submitted Stage 3 job: ${JOB2}"
  fi
fi

STAGE4_ARGS=(
  --source-dataset "${ANNOTATED_DATASET}"
  --output-dataset "${SUBSAMPLED_DATASET}"
  --fraction "${SUBSAMPLE_FRACTION}"
  --seed "${SEED}"
)
if [[ "${PRIVATE}" -eq 1 ]]; then
  STAGE4_ARGS+=(--private)
fi

STAGE4_SBATCH_ARGS=(
  --parsable
  --export "${SBATCH_EXPORT}"
  --chdir "${REPO_ROOT}"
)
if [[ -n "${JOB2}" ]]; then
  STAGE4_SBATCH_ARGS+=(--dependency="afterok:${JOB2}")
fi

JOB3="$(sbatch "${STAGE4_SBATCH_ARGS[@]}" scripts/dataset_pipeline/stage4_subsample.sbatch "${STAGE4_ARGS[@]}")"
if [[ -n "${JOB2}" ]]; then
  echo "Submitted Stage 4 job: ${JOB3} (depends on ${JOB2})"
else
  echo "Submitted Stage 4 job: ${JOB3}"
fi

echo ""
echo "Pipeline submitted successfully."
echo "Filtered dataset repo:   ${FILTERED_DATASET}"
echo "Annotated dataset repo:  ${ANNOTATED_DATASET}"
echo "Subsampled dataset repo: ${SUBSAMPLED_DATASET}"

JOB_IDS=()
if [[ -n "${JOB1}" ]]; then
  JOB_IDS+=("${JOB1}")
fi
if [[ -n "${JOB2}" ]]; then
  JOB_IDS+=("${JOB2}")
fi
if [[ -n "${JOB3}" ]]; then
  JOB_IDS+=("${JOB3}")
fi

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
  JOB_LIST="$(IFS=,; echo "${JOB_IDS[*]}")"
  echo "Track jobs: squeue -j ${JOB_LIST}"
fi
