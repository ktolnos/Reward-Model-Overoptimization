#!/usr/bin/env bash
# Compare the two most promising LLM-judge configs on cached policy generations.
#
# Conclusion from the 4-way sweep (think/no-think x bf16/fp8, gemma-4-31B judge):
#   * NO-THINKING dominates: 0% dropped, highest gold-RM agreement (0.69) and
#     self-consistency (0.91), and 3-5x faster than thinking. Gemma's thought
#     channel is verbose and truncates (5-58% dropped depending on prompt/budget)
#     while LOWERING agreement -- reasoning does not help preference judging here.
#   * fp8 ~= bf16 on quality (97.6% identical verdicts) but is SLOWER on this A100
#     (Ampere lacks native fp8 -> Marlin weight-only penalty on the prefill-bound
#     judge), and the 31B fits in bf16 on 80GB anyway.
#
# => The two configs worth comparing head-to-head:
#      1. bf16 no-think  (PRIMARY recommendation: fastest + best quality)
#      2. fp8  no-think  (fallback ONLY when VRAM-constrained, e.g. <48GB GPU)
#
# This script re-judges the SAME cached generations from a previous eval under
# both configs and reports quality / speed / agreement / checkpoint-ranking, so
# the choice can be confirmed at scale (esp. that fp8 preserves the ranking).
#
# Requires: the judge fixes in policy_eval/judges.py (fp8 `quantization` support
# + gemma turn-end stop tokens) -- already committed on this branch.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- config -----------------------------------------------------------------
# Per-example dir of a prior eval run holding preference__checkpoint-*.parquet
# (policy responses + chosen baseline + gold-RM scores). Auto-pick the newest.
PE_DIR="${PE_DIR:-$(ls -dt evaluation_results*_per_example 2>/dev/null | head -1)}"

# Checkpoints to judge. Default: all checkpoints present in PE_DIR (best ranking
# signal). Override with e.g. CKPTS=149,745,1341,2086,2975 for a quick pass.
if [ -z "${CKPTS:-}" ]; then
  CKPTS=$(ls "$PE_DIR"/preference__checkpoint-*.parquet 2>/dev/null \
    | grep -E 'checkpoint-[0-9]+\.parquet$' \
    | sed -E 's/.*checkpoint-([0-9]+)\.parquet/\1/' | sort -n | paste -sd, -)
fi

N_PROMPTS="${N_PROMPTS:-150}"     # preference prompts per checkpoint (150 ~= tight CIs)
GPU_MEM="${GPU_MEM:-0.92}"
MAX_LEN="${MAX_LEN:-7168}"
TAG="${TAG:-top2}"

echo "PE_DIR=$PE_DIR"
echo "CKPTS=$CKPTS"
echo "N_PROMPTS=$N_PROMPTS  (no-think only; thinking is excluded -- it lost the sweep)"

# ---- run --------------------------------------------------------------------
# Both promising configs are NO-THINK; sweep both quant levels, bf16 first.
VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1 python scratch_judge_sweep/run_judge_sweep.py \
  --per_example_dir "$PE_DIR" \
  --checkpoints "$CKPTS" \
  --n_prompts "$N_PROMPTS" \
  --quants none,fp8 \
  --thinks false \
  --max_model_len "$MAX_LEN" --gpu_mem "$GPU_MEM" \
  --tag "$TAG"

# ---- analyze ----------------------------------------------------------------
python scratch_judge_sweep/analyze_judge_sweep.py --tag "$TAG" --reference bf16__think=False
