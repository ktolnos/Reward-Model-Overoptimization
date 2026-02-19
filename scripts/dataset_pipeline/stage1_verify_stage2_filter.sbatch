#!/bin/bash
#SBATCH --job-name=prep_stage1_stage2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --qos=default

source ~/.bashrc

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ ! -f "${REPO_ROOT}/AGENTS.md" ]]; then
  echo "ERROR: Could not resolve repo root (got '${REPO_ROOT}')." >&2
  exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/rlhf/grpo:${REPO_ROOT}:${PYTHONPATH:-}"

# Normalize token env names so huggingface_hub can read either convention.
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

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

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: No HF token found in job environment (HF_TOKEN/HUGGINGFACE_HUB_TOKEN)." >&2
  exit 2
fi

python3 - "${OUTPUT_DATASET}" <<'PY'
import os
import sys
from huggingface_hub import HfApi

output_dataset = sys.argv[1]
namespace = output_dataset.split("/", 1)[0] if "/" in output_dataset else None
token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if not token:
    raise SystemExit("ERROR: missing HF auth token in environment.")

api = HfApi(token=token)
try:
    who = api.whoami()
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"ERROR: HF auth failed before Stage 2 upload: {exc}") from exc

username = who.get("name")
orgs_raw = who.get("orgs", [])
org_names = set()
for org in orgs_raw:
    if isinstance(org, dict):
        name = org.get("name")
    else:
        name = str(org)
    if name:
        org_names.add(name)

allowed_namespaces = {username, *org_names}
if namespace and namespace not in allowed_namespaces:
    raise SystemExit(
        "ERROR: HF token user "
        f"'{username}' cannot write to namespace '{namespace}'. "
        f"Allowed namespaces from whoami: {sorted(allowed_namespaces)}"
    )

print(f"HF auth preflight OK: user={username}, target_namespace={namespace}")
PY

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
