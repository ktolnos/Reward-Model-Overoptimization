#!/bin/bash
set -euo pipefail

# Local, single-process dataset pipeline runner.
#
# Runs the stages sequentially in THIS shell (no SLURM, no job dependencies):
#   Stage 1  verify dataset format
#   Stage 2  filter + four-way split (train/select/validation/test) + upload
#   Stage 3  annotate with a reward model + upload   (skipped with --skip-annotation)
#   Stage 4  subsample + upload
#
# Stage 3 annotation needs a GPU large enough for the reward model; the local box
# may not have one. For a human-preference-only dataset pass --skip-annotation
# (Stage 2 then writes directly to the annotated repo and Stage 3 is skipped),
# which needs no GPU. Any stage failure aborts the run (set -e).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Prefer the repo virtualenv locally; fall back to python3 (e.g. on a cluster).
if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/venv/bin/python"
else
  PYTHON="python3"
fi
export PYTHONPATH="${REPO_ROOT}/rlhf/grpo:${REPO_ROOT}:${PYTHONPATH:-}"

SOURCE_DATASET=""
REWARD_MODEL=""
PREFIX=""
NAMESPACE=""
TOKENIZER_NAME="Qwen/Qwen3-0.6B"
SEED=42
TRAIN_RATIO=0.85
SELECT_RATIO=0.05
VALIDATION_RATIO=0.05
TEST_RATIO=0.05
SUBSAMPLE_FRACTION=0.25
# deliberatly smaller to allow for other tokenizers
MAX_PROMPT_TOKENS=1000
MAX_RESPONSE_TOKENS=1000
MAX_CONVERSATION_TOKENS=2000
MAX_ERRORS=20
PRIVATE=0
TRUST_REMOTE_CODE=0
SKIP_STAGE12=0
SKIP_STAGE3=0
SKIP_STAGE4=0
SKIP_ANNOTATION=0
SKIP_PREFIX_CHECK=0
MERGE_SPLITS=0

usage() {
  cat <<EOF
Usage: $0 \
  --source-dataset <hf_repo> \
  --reward-model <hf_or_local_model> \
  --prefix <name_prefix> \
  --namespace <hf_user_or_org> \
  [--tokenizer-name <hf_tokenizer>] \
  [--seed <int>] \
  [--train-ratio <float>] [--select-ratio <float>] [--validation-ratio <float>] [--test-ratio <float>] \
  [--subsample-fraction <float>] \
  [--max-prompt-tokens <int>] [--max-response-tokens <int>] [--max-conversation-tokens <int>] \
  [--max-errors <int>] [--private] [--trust-remote-code] \
  [--merge-splits] [--skip-prefix-check] \
  [--skip-stage12] [--skip-stage3] [--skip-stage4] [--skip-annotation]

Runs locally in a single process (no SLURM). --merge-splits concatenates all
source splits into one pool before the four-way split (use when re-splitting an
existing derived dataset whose splits are a row-level partition of one population).
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
    --select-ratio) SELECT_RATIO="$2"; shift 2 ;;
    --validation-ratio) VALIDATION_RATIO="$2"; shift 2 ;;
    --test-ratio) TEST_RATIO="$2"; shift 2 ;;
    --subsample-fraction) SUBSAMPLE_FRACTION="$2"; shift 2 ;;
    --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
    --max-response-tokens) MAX_RESPONSE_TOKENS="$2"; shift 2 ;;
    --max-conversation-tokens) MAX_CONVERSATION_TOKENS="$2"; shift 2 ;;
    --max-errors) MAX_ERRORS="$2"; shift 2 ;;
    --private) PRIVATE=1; shift ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1; shift ;;
    --merge-splits) MERGE_SPLITS=1; shift ;;
    --skip-prefix-check) SKIP_PREFIX_CHECK=1; shift ;;
    --skip-stage12) SKIP_STAGE12=1; shift ;;
    --skip-stage3) SKIP_STAGE3=1; shift ;;
    --skip-stage4) SKIP_STAGE4=1; shift ;;
    --skip-annotation) SKIP_ANNOTATION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${SOURCE_DATASET}" || -z "${PREFIX}" || -z "${NAMESPACE}" ]]; then
  usage
  exit 2
fi

if [[ -z "${REWARD_MODEL}" && "${SKIP_ANNOTATION}" -eq 0 && "${SKIP_STAGE3}" -eq 0 ]]; then
  echo "ERROR: --reward-model is required unless --skip-annotation or --skip-stage3 is set." >&2
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HF token is missing in the current shell." >&2
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running the pipeline." >&2
  exit 2
fi

# Normalize token env names so downstream tools can read either convention.
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

sanitize() {
  echo "$1" | sed -e 's#[/:]#-#g' -e 's#[^a-zA-Z0-9_-]#-#g' -e 's#--*#-#g' -e 's#^-##' -e 's#-$##'
}

if [[ -n "${REWARD_MODEL}" ]]; then
  REWARD_SUFFIX="$(sanitize "${REWARD_MODEL}")"
elif [[ "${SKIP_ANNOTATION}" -eq 1 ]]; then
  REWARD_SUFFIX="human"
else
  REWARD_SUFFIX="no-rm"
fi

ANNOTATED_DATASET="${NAMESPACE}/$(sanitize "${PREFIX}")_annotated_${REWARD_SUFFIX}"
SUBSAMPLED_DATASET="${NAMESPACE}/$(sanitize "${PREFIX}")_annotated_${REWARD_SUFFIX}_25pct"

# When skipping annotation, Stage 2 writes directly to ANNOTATED_DATASET (no _filtered upload).
if [[ "${SKIP_ANNOTATION}" -eq 1 ]]; then
  FILTERED_DATASET="${ANNOTATED_DATASET}"
  SKIP_STAGE3=1
else
  FILTERED_DATASET="${NAMESPACE}/$(sanitize "${PREFIX}")_filtered"
fi

echo "Python interpreter:  ${PYTHON}"
echo "Source dataset:      ${SOURCE_DATASET}"
if [[ "${SKIP_ANNOTATION}" -eq 0 ]]; then
  echo "Filtered dataset:    ${FILTERED_DATASET}"
fi
echo "Annotated dataset:   ${ANNOTATED_DATASET}"
echo "Subsampled dataset:  ${SUBSAMPLED_DATASET}"

# ----------------------------------------------------------------------------
# Stage 1 + 2: verify, then filter / four-way split / upload
# ----------------------------------------------------------------------------
if [[ "${SKIP_STAGE12}" -eq 1 ]]; then
  echo ""
  echo "Skipping Stage 1+2 (--skip-stage12)."
  echo "Expected existing dataset repo: ${FILTERED_DATASET}"
else
  VERIFY_ARGS=(
    --source-dataset "${SOURCE_DATASET}"
    --tokenizer-name "${TOKENIZER_NAME}"
    --max-errors "${MAX_ERRORS}"
  )
  STAGE2_ARGS=(
    --source-dataset "${SOURCE_DATASET}"
    --output-dataset "${FILTERED_DATASET}"
    --tokenizer-name "${TOKENIZER_NAME}"
    --seed "${SEED}"
    --train-ratio "${TRAIN_RATIO}"
    --select-ratio "${SELECT_RATIO}"
    --validation-ratio "${VALIDATION_RATIO}"
    --test-ratio "${TEST_RATIO}"
    --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
    --max-response-tokens "${MAX_RESPONSE_TOKENS}"
    --max-conversation-tokens "${MAX_CONVERSATION_TOKENS}"
    --max-errors "${MAX_ERRORS}"
  )
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then
    VERIFY_ARGS+=(--trust-remote-code)
    STAGE2_ARGS+=(--trust-remote-code)
  fi
  if [[ "${PRIVATE}" -eq 1 ]]; then
    STAGE2_ARGS+=(--private)
  fi
  if [[ "${MERGE_SPLITS}" -eq 1 ]]; then
    STAGE2_ARGS+=(--merge-splits)
  fi
  if [[ "${SKIP_PREFIX_CHECK}" -eq 1 ]]; then
    STAGE2_ARGS+=(--skip-prefix-check)
  fi

  echo ""
  echo "=== Stage 1: verifying ${SOURCE_DATASET} ==="
  "${PYTHON}" scripts/dataset_pipeline/stage1_verify_dataset.py "${VERIFY_ARGS[@]}"

  echo ""
  echo "=== Stage 2: filter / split / upload -> ${FILTERED_DATASET} ==="
  "${PYTHON}" scripts/dataset_pipeline/stage2_filter_split_upload.py "${STAGE2_ARGS[@]}"
fi

# ----------------------------------------------------------------------------
# Stage 3: annotate with reward model + upload (needs GPU; skipped for --skip-annotation)
# ----------------------------------------------------------------------------
if [[ "${SKIP_STAGE3}" -eq 1 ]]; then
  echo ""
  echo "Skipping Stage 3 (annotation)."
  echo "Annotated dataset repo: ${ANNOTATED_DATASET}"
else
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

  echo ""
  echo "=== Stage 3: annotate ${FILTERED_DATASET} with ${REWARD_MODEL} -> ${ANNOTATED_DATASET} ==="
  echo "(annotation requires a GPU large enough for the reward model)"
  "${PYTHON}" scripts/dataset_pipeline/stage3_annotate_and_upload.py "${STAGE3_ARGS[@]}"
fi

# ----------------------------------------------------------------------------
# Stage 4: subsample + upload
# ----------------------------------------------------------------------------
if [[ "${SKIP_STAGE4}" -eq 1 ]]; then
  echo ""
  echo "Skipping Stage 4 (subsample)."
else
  STAGE4_ARGS=(
    --source-dataset "${ANNOTATED_DATASET}"
    --output-dataset "${SUBSAMPLED_DATASET}"
    --fraction "${SUBSAMPLE_FRACTION}"
    --seed "${SEED}"
  )
  if [[ "${PRIVATE}" -eq 1 ]]; then
    STAGE4_ARGS+=(--private)
  fi

  echo ""
  echo "=== Stage 4: subsample ${ANNOTATED_DATASET} -> ${SUBSAMPLED_DATASET} ==="
  "${PYTHON}" scripts/dataset_pipeline/stage4_subsample_upload.py "${STAGE4_ARGS[@]}"
fi

echo ""
echo "Pipeline completed successfully."
if [[ "${SKIP_ANNOTATION}" -eq 0 ]]; then
  echo "Filtered dataset repo:   ${FILTERED_DATASET}"
fi
echo "Annotated dataset repo:  ${ANNOTATED_DATASET}"
if [[ "${SKIP_STAGE4}" -eq 0 ]]; then
  echo "Subsampled dataset repo: ${SUBSAMPLED_DATASET}"
fi
