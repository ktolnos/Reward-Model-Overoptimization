#!/bin/bash

#SBATCH --job-name=evaluate_policy
#SBATCH --cpus-per-task=8
#SBATCH --mem=56gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --time=24:00:00
#SBATCH --qos=high

# =============================================================================
# Usage
# =============================================================================
# Full eval (preference + ifeval):
#     sbatch evaluate_policy.sh
#
# Add IFEval to an existing wandb run (no re-generation of preference eval):
#     sbatch evaluate_policy.sh --run_id <WANDB_RUN_ID> --only_ifeval
#
# IFEval with both rule-based AND gold-RM scoring:
#     sbatch evaluate_policy.sh --ifeval_use_gold_rm
#
# Arena-Hard-Auto v2.0 (gold RM only):
#     sbatch evaluate_policy.sh --only_arena_hard [--run_id <WANDB_RUN_ID>]
#
# LLM-judge on already-generated answers from a previous eval (no regeneration,
# no reward models). Source generations are auto-discovered (latest per-example
# dir for this checkpoints_dir); pass --run_id to add the judge metrics onto the
# original eval's wandb run:
#     sbatch evaluate_policy.sh --llm_judge_on_cached [--run_id <WANDB_RUN_ID>]
#     sbatch evaluate_policy.sh --llm_judge_on_cached --load_generations_dir <DIR>
#
# Arbitrary benchmark subset:
#     sbatch evaluate_policy.sh --benchmarks ifeval,preference,arena_hard
#
# The LLM judge runs through the Vector Institute inference proxy by default (no
# GPU used by the judge; needs VECTOR_INFERENCE_API_KEY, exported from ~/.bashrc
# which this script sources), on gpt-oss-120b in non-thinking mode -- see
# LLM_JUDGE_MODEL below for why, and note thinking mode is measurably not better
# and much slower. Judge inline in THIS job instead of the auto-queued one, pick
# another proxy model, or fall back to a local vLLM judge:
#     sbatch evaluate_policy.sh --vector_judge
#     sbatch evaluate_policy.sh --vector_judge --judge_thinking
#     sbatch evaluate_policy.sh --judge_model Qwen3_5-122B-A10B
#     sbatch evaluate_policy.sh --vllm_judge --judge_model google/gemma-4-31B-it
#
# The proxy's RPM budget is shared across the whole project, so the judge paces
# itself (--judge_rpm, default 100 under the observed 120 cap). For a whole-run
# judge pass, the async Batch API trades latency for freedom from that budget:
#     sbatch evaluate_policy.sh --vector_judge --judge_batch_api
#
# Judge only cached generations through the proxy (no regeneration, no RMs):
#     sbatch evaluate_policy.sh --llm_judge_on_cached --vector_judge
#
# Other overrides (all optional): --run_name, --checkpoint, --kl_base_model_path,
# --ifeval_thinking, --evaluate_chosen_responses, --no_secondary_rm,
# --with_training_rm, --with_llm_judge, --vllm_judge, --judge_all_checkpoints,
# --judge_reasoning_effort, --judge_max_parallel, --judge_max_new_tokens.
#
# The LLM judge runs on the SELECTED checkpoint (sibling-RM argmax) plus the
# FINAL one by default, rather than on all ~20: those are the only checkpoints
# whose judge numbers are read, and the pair is what would expose an
# overoptimized gold RM. To narrow or widen that set:
#     sbatch evaluate_policy.sh --with_llm_judge --judge_no_final
#     sbatch evaluate_policy.sh --with_llm_judge --judge_all_checkpoints
#
# The judge is NOT run inside this job by default. This job queues
# judge_cached.sh -- GPU-free, inheriting this job's qos/partition/account,
# chained afterok on it, and serialized against other judge jobs -- so a plain
#     sbatch evaluate_policy.sh
# gets you generation + RM/IFEval/KL here and the judge in its own CPU job right
# after, with its metrics on this same wandb run. Suppress with --no_auto_judge;
# to judge inline in this job instead, pass --with_llm_judge (which also skips
# the auto-submission).
#
# Debug mode (subsamples examples, only the first checkpoint, and suffixes
# outputs / the wandb run name with _debug):
#     sbatch evaluate_policy.sh --debug
# =============================================================================

REPO_ROOT=/nas/ucb/eop/Reward-Model-Overoptimization
cd "$REPO_ROOT"
source /home/eop/.bashrc
echo "$(ls -a /home/eop/)"

# Directory containing the checkpoints
CHECKPOINTS_DIR="/nas/ucb/eop/Reward-Model-Overoptimization/scripts/rlhf/logs_grpo/20260202_183444_1035193"


# Training RM, dataset, KL base model, and eval temperature default from the
# run's run_manifest.json (written by my_grpo.py into the checkpoints dir), so
# eval always matches what the run actually trained with. Leave these empty to
# use the manifest; set them (here or via the --flags below) only to override
# it or to evaluate a legacy run that predates the manifest.
TRAINING_RM_PATH=""
#TRAINING_RM_PATH="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_42_BT_RM_Qwen/Qwen3-0.6B_974219_len2000_fulltrain_2e-05_datahelpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k/logs/checkpoint-142"
#TRAINING_RM_PATH="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"

# Sibling RM: an independently-seeded RM from the training RM's base model, used
# by the 'select' benchmark to pick the best checkpoint (scored on the dataset's
# 'select' split). Validated as a near-oracle checkpoint selector.
SIBLING_RM_PATH="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/20_Qwen3.5-4B-Base_len2048_fulltrain_2e-05_datahelpsteer3-qwen35_annotated_human/logs/checkpoint-1179"

# Name of the gold reward model
#GOLD_RM_NAME="Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
#GOLD_RM_NAME="LxzGordon/URM-LLaMa-3.1-8B"
#GOLD_RM_NAME="Skywork/Skywork-Reward-V2-Qwen3-8B"
GOLD_RM_NAME="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

# LLM judge. Used for the preference benchmark and, together with the gold RM,
# for arena_hard. The same model serves both benchmarks (loaded once).
#
# Two backends:
#   vector - Vector Institute inference proxy (https://proxy.vectorinstitute.ai/v1),
#            OpenAI-compatible. Needs VECTOR_INFERENCE_API_KEY in the environment
#            (it is exported from ~/.bashrc, which this script sources). THE
#            DEFAULT: no GPU is used by the judge at all, so it neither competes
#            with the policy for the node nor spends minutes loading a large
#            model, and the judge pass can run as its own GPU-free job.
#   vllm   - local open-weight model on this node's GPU (no API quota, but it
#            has to share the node and load a large model). Select with
#            --vllm_judge.
LLM_JUDGE_BACKEND="vector"

# The judge model, for whichever backend is selected -- there is deliberately no
# per-backend default, so switching backends never silently switches the judge
# too. This is a proxy model id; --vllm_judge needs an HF path, so pass
# --judge_model with it.
# See llm_judge_config_eval.md. On 500 deduped helpsteer3-qwen35_annotated_human
# validation prompts the top four models are statistically INDISTINGUISHABLE on
# agreement with the human labels, so the pick is made on the other axes:
# gpt-oss-120b drops 0/500 prompts, has the highest throughput, and -- decisively
# -- is from a different model family than the Qwen policies and RMs, so it
# cannot self-prefer policy rollouts the way a Qwen judge might. It is ~0.03
# weaker on non-English prompts than the Qwen judges, which is the known cost.
# Do NOT use Nemotron-3-Nano-Omni-30B-A3B: it decides 57% of prompts by answer
# position rather than content (and is fast with zero drops, so speed and
# failure counters both endorse it -- only the flip rate catches it).
LLM_JUDGE_MODEL="gpt-oss-120b"
# Client-side pacing, overriding the provider's default with --judge_rpm.
# Empty = the provider decides (Vector proxy: 100/min, under its observed
# 120 project-wide cap; OpenRouter: unpaced). 0 disables pacing.
LLM_JUDGE_RPM=""
# Measured (llm_judge_config_eval.md): the judge is concurrency-bound, not
# rate-limit-bound -- at 8 the 100 RPM budget is never reached. 8 -> 32 buys
# +10% throughput on a fast judge and +61% on a slow one. The cost is that the
# server queues: per-request latency grows ~linearly with concurrency once
# saturated (gpt-oss 5.4s -> 19.6s mean), and the budget is shared with the rest
# of the project. Lower this to 16 (most of the gain, half the latency) if the
# proxy is busy or requests start timing out.
LLM_JUDGE_MAX_PARALLEL="32"
# auto = derive from the thinking flag (high when thinking, low when not).
# gpt-oss models always reason (harmony format); 'low' is as close to
# non-thinking as the API allows.
LLM_JUDGE_REASONING_EFFORT="auto"

# Dataset name (empty = from the run manifest)
DATASET_NAME=""
#DATASET_NAME="ktolnos/helpsteer3-qwen35_annotated_human"
# DATASET_NAME="ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"


# Base model name (required for LoRA checkpoints)
# Uncomment and set this if evaluating LoRA checkpoints
#BASE_MODEL_NAME="Qwen/Qwen3-0.6B"

# Output file
OUTPUT_FILE="evaluation_results${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).csv"

# WandB settings
WANDB_PROJECT="policy-evaluation"
WANDB_RUN_NAME="policy_evaluation_$(date +%Y%m%d_%H%M%S)"

SKIP_VALIDATION=1
BENCHMARKS="select,preference,ifeval,arena_hard"

# Policy sampling temperature for all policy generations (BENCHMARK.md §8).
# Empty = the training temperature from the run manifest (falling back to the
# Python default 1.0 for manifest-less runs). Set only to override.
EVAL_TEMPERATURE=""

# Run the LLM-as-judge evaluators on the selected checkpoint only (the argmax of
# the sibling RM's selection score) instead of every checkpoint. On by default:
# the judge is the most expensive evaluator and only the selected checkpoint's
# judge numbers are reported. Cheap metrics (RM scores, IFEval, KL, the
# arena_hard gold-RM judge) still run for all checkpoints. Needs a selection
# signal — the 'select' benchmark here, or cached 'select' per-example logs
# under --load_generations; a judged run without one is rejected at startup.
# Turn off with --judge_all_checkpoints (e.g. --only_arena_hard --with_llm_judge).
JUDGE_SELECTED_ONLY=1

# Judge the FINAL checkpoint alongside the selected one (one extra judged
# checkpoint out of ~20). The pair is what separates "the sibling RM picked a
# good checkpoint" from "the gold RM was itself overoptimized": if the gold RM
# ranks the final checkpoint at or above the selected one but the LLM judge
# ranks it below, the gold signal degraded over training. No-op when the
# selected checkpoint already is the last. Turn off with --judge_no_final.
JUDGE_FINAL=1

# Auto-submit the GPU-free LLM-judge pass (judge_cached.sh) over this run's
# cached generations, chained afterok on this job. On by default so the normal
# workflow is "sbatch evaluate_policy.sh" and nothing else: the judge then runs
# in its own CPU-only job, inherits this job's qos/partition, and serializes
# against other judge jobs (--dependency=singleton) because the proxy's RPM
# budget is shared project-wide while the client-side pacing is per-process.
# Skipped automatically when the judge already ran inline (--with_llm_judge),
# when this IS the judge pass (--load_generations), or off slurm.
# Turn off with --no_auto_judge.
AUTO_JUDGE=1

# Debug mode: subsamples examples, uses only the first checkpoint, and
# suffixes outputs / the wandb run name with _debug. See --debug below.
DEBUG_MODE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) WANDB_RUN_NAME="$2"; shift ;;
        --run_id) WANDB_RUN_ID="$2"; shift ;;
        --base_eval_run_id) BASE_EVAL_RUN_ID="$2"; shift ;;
        --no_base_checkpoint) NO_BASE_CHECKPOINT=1 ;;
        --kl_base_model_path) KL_BASE_MODEL_PATH="$2"; shift ;;
        --training_rm_path) TRAINING_RM_PATH="$2"; shift ;;
        --dataset_name) DATASET_NAME="$2"; shift ;;
        --checkpoint) CHECKPOINTS_DIR="$2"; shift ;;
        --benchmarks) BENCHMARKS="$2"; shift ;;
        --eval_temperature) EVAL_TEMPERATURE="$2"; shift ;;
        --only_ifeval) BENCHMARKS="ifeval"; NO_SECONDARY_RM=1 ;;
        --only_preference) BENCHMARKS="preference" ;;
        --only_arena_hard) BENCHMARKS="arena_hard"; NO_SECONDARY_RM=1 ;;
        --skip_validation) SKIP_VALIDATION=1 ;;
        --evaluate_chosen_responses) EVALUATE_CHOSEN=1 ;;
        --ifeval_thinking) IFEVAL_THINKING=1 ;;
        --ifeval_use_gold_rm) IFEVAL_USE_GOLD_RM=1 ;;
        --no_ifeval) NO_IFEVAL=1 ;;
        --no_secondary_rm) NO_SECONDARY_RM=1 ;;
        --with_training_rm) WITH_TRAINING_RM=1 ;;
        --with_llm_judge) WITH_LLM_JUDGE=1 ;;
        # Judge through the Vector proxy (the default backend) in THIS job rather
        # than in the auto-queued one. Implies --with_llm_judge.
        --vector_judge) WITH_LLM_JUDGE=1; LLM_JUDGE_BACKEND="vector" ;;
        # Judge with a local open-weight model on this job's GPU instead of the
        # proxy. Implies --with_llm_judge, and needs --judge_model with an HF
        # path, since LLM_JUDGE_MODEL above is a proxy model id.
        --vllm_judge) WITH_LLM_JUDGE=1; LLM_JUDGE_BACKEND="vllm" ;;
        # Pick the judge model for whichever backend is selected, e.g.
        # --judge_model Qwen3_5-122B-A10B.
        --judge_model) WITH_LLM_JUDGE=1; LLM_JUDGE_MODEL="$2"; shift ;;
        --judge_thinking) JUDGE_THINKING=1 ;;
        --judge_reasoning_effort) LLM_JUDGE_REASONING_EFFORT="$2"; shift ;;
        --judge_rpm) LLM_JUDGE_RPM="$2"; shift ;;
        --judge_max_parallel) LLM_JUDGE_MAX_PARALLEL="$2"; shift ;;
        --judge_max_new_tokens) LLM_JUDGE_MAX_NEW_TOKENS="$2"; shift ;;
        # Async OpenAI-style Batch API instead of live requests: runs against the
        # batch quota rather than the live RPM budget, at the cost of latency.
        --judge_batch_api) JUDGE_BATCH_API=1 ;;
        --judge_all_checkpoints) JUDGE_SELECTED_ONLY="" ;;
        # Judge the selected checkpoint alone, without the final one.
        --judge_no_final) JUDGE_FINAL="" ;;
        --no_auto_judge) AUTO_JUDGE="" ;;
        --load_generations) LOAD_GENERATIONS=1 ;;
        --load_generations_dir) LOAD_GENERATIONS=1; LOAD_GENERATIONS_DIR="$2"; shift ;;
        # Convenience: run ONLY the LLM judge on already-generated answers from a
        # previous eval (auto-discovered), skipping regeneration and all RMs.
        --llm_judge_on_cached)
            LOAD_GENERATIONS=1
            WITH_LLM_JUDGE=1
            BENCHMARKS="preference,arena_hard"
            NO_SECONDARY_RM=1
            ;;
        --debug) DEBUG_MODE=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# arena_hard is scored by the gold RM, plus the LLM judge when --with_llm_judge
# is passed (both judges run on the same arena_hard responses).
# In --load_generations mode no RM is loaded, so arena_hard is judged by the LLM
# judge alone.
# A judge-only pass over cached generations loads no reward model and no policy
# vLLM, so unless the judge itself is local vLLM there is nothing to put on a
# GPU -- see judge_cached.sh, which submits exactly this without --gres.
DEVICE="cuda"
if [ -n "${LOAD_GENERATIONS:-}" ] && [ "$LLM_JUDGE_BACKEND" != "vllm" ]; then
    DEVICE="cpu"
fi

ARENA_HARD_JUDGES="rm:gold_rm"
if [ -n "${LOAD_GENERATIONS:-}" ]; then
    ARENA_HARD_JUDGES="llm:${LLM_JUDGE_MODEL}"
elif [ -n "${WITH_LLM_JUDGE:-}" ]; then
    ARENA_HARD_JUDGES="rm:gold_rm,llm:${LLM_JUDGE_MODEL}"
fi

echo "Running evaluation with the following settings:"
echo "Checkpoints Directory: $CHECKPOINTS_DIR"
echo "Training RM Path: ${TRAINING_RM_PATH:-<run manifest>}"
echo "Sibling RM Path: $SIBLING_RM_PATH"
echo "Gold RM Name: $GOLD_RM_NAME"
echo "Dataset Name: ${DATASET_NAME:-<run manifest>}"
echo "KL Base Model: ${KL_BASE_MODEL_PATH:-<run manifest>}"
echo "Eval Temperature: ${EVAL_TEMPERATURE:-<run manifest>}"
echo "Output File: $OUTPUT_FILE"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run Name: $WANDB_RUN_NAME"
echo "WandB Run ID (resume): ${WANDB_RUN_ID:-<new run>}"
echo "Benchmarks: $BENCHMARKS"
echo "Arena-Hard judges: $ARENA_HARD_JUDGES"
echo "LLM judge: $([ -n "${WITH_LLM_JUDGE:-}" ] && echo "enabled ($LLM_JUDGE_BACKEND: $LLM_JUDGE_MODEL)" || echo disabled)"
if [ "$LLM_JUDGE_BACKEND" = "vector" ] && [ -n "${WITH_LLM_JUDGE:-}" ]; then
    echo "  Vector proxy: rpm=${LLM_JUDGE_RPM:-<provider default: 100>} max_parallel=$LLM_JUDGE_MAX_PARALLEL reasoning_effort=$LLM_JUDGE_REASONING_EFFORT"
    echo "  Judge thinking: $([ -n "${JUDGE_THINKING:-}" ] && echo enabled || echo disabled)"
    echo "  Batch API: $([ -n "${JUDGE_BATCH_API:-}" ] && echo enabled || echo disabled)"
    # Only when the proxy judge runs in THIS process. A run that merely queues
    # the judge job must not die over a key it never uses; the queued job comes
    # back through here with --llm_judge_on_cached and fails its own check. The
    # auto-judge block below warns at submit time so that is not a surprise.
    if [ -z "${VECTOR_INFERENCE_API_KEY:-}" ]; then
        echo "ERROR: the vector judge needs VECTOR_INFERENCE_API_KEY in the environment." >&2
        exit 1
    fi
fi
echo "Judge selected checkpoint only: $([ -n "${JUDGE_SELECTED_ONLY:-}" ] && echo "enabled (+ final: $([ -n "${JUDGE_FINAL:-}" ] && echo yes || echo no))" || echo disabled)"
echo "Load cached generations: $([ -n "${LOAD_GENERATIONS:-}" ] && echo "enabled (${LOAD_GENERATIONS_DIR:-auto-discover})" || echo disabled)"
echo "Debug mode: ${DEBUG_MODE:+enabled}${DEBUG_MODE:-disabled}"

# Per-example log dir. Derived exactly as evaluate_policy.py would, but pinned
# here and passed explicitly so the auto-submitted judge job can be pointed at
# this run's generations with no chance of the two derivations drifting.
PER_EXAMPLE_DIR="${OUTPUT_FILE%.csv}$([ -n "${DEBUG_MODE:-}" ] && echo _debug)_per_example"

# ---------------------------------------------------------------------------
# Queue the GPU-free judge pass over this run's generations (see AUTO_JUDGE).
# Submitted BEFORE python starts so its job id can be logged into the eval's
# wandb run (slurm/judge/job_id) rather than only existing in this log.
# --kill-on-invalid-dep: if this eval fails, afterok is never satisfied and the
# queued judge job is cancelled instead of sitting pending forever.
# ---------------------------------------------------------------------------
JUDGE_SLURM_JOB_ID=""
if [ -n "${AUTO_JUDGE:-}" ] && [ -z "${LOAD_GENERATIONS:-}" ] \
   && [ -z "${WITH_LLM_JUDGE:-}" ] && [ -n "${SLURM_JOB_ID:-}" ] \
   && command -v sbatch >/dev/null 2>&1; then
    # The queued job judges through the proxy, so a key it cannot see is a
    # failure hours from now, after the GPU work is already paid for.
    if [ "$LLM_JUDGE_BACKEND" = "vector" ] && [ -z "${VECTOR_INFERENCE_API_KEY:-}" ]; then
        echo "WARNING: VECTOR_INFERENCE_API_KEY is not set; the queued judge job will" >&2
        echo "  fail on it. Export it in ~/.bashrc, or pass --no_auto_judge." >&2
    fi
    JUDGE_ARGS=(--checkpoint "$CHECKPOINTS_DIR" --load_generations_dir "$PER_EXAMPLE_DIR")
    [ -z "${JUDGE_SELECTED_ONLY:-}" ] && JUDGE_ARGS+=(--judge_all_checkpoints)
    [ -z "${JUDGE_FINAL:-}" ] && JUDGE_ARGS+=(--judge_no_final)
    [ -n "${DEBUG_MODE:-}" ] && JUDGE_ARGS+=(--debug)
    # Inherit this job's scheduling context so the judge lands in the same
    # place in the queue policy instead of falling back to judge_cached.sh's
    # defaults.
    SBATCH_OPTS=(--parsable
                 --dependency="singleton,afterok:${SLURM_JOB_ID}"
                 --kill-on-invalid-dep=yes)
    [ -n "${SLURM_JOB_QOS:-}" ] && SBATCH_OPTS+=(--qos="$SLURM_JOB_QOS")
    [ -n "${SLURM_JOB_PARTITION:-}" ] && SBATCH_OPTS+=(--partition="$SLURM_JOB_PARTITION")
    [ -n "${SLURM_JOB_ACCOUNT:-}" ] && SBATCH_OPTS+=(--account="$SLURM_JOB_ACCOUNT")
    JUDGE_SLURM_JOB_ID=$(sbatch "${SBATCH_OPTS[@]}" \
        "$REPO_ROOT/judge_cached.sh" "${JUDGE_ARGS[@]}") || JUDGE_SLURM_JOB_ID=""
    # --parsable returns "jobid" or "jobid;cluster" on a multi-cluster setup.
    JUDGE_SLURM_JOB_ID="${JUDGE_SLURM_JOB_ID%%;*}"
    if [ -n "$JUDGE_SLURM_JOB_ID" ]; then
        echo "Queued judge pass: slurm job $JUDGE_SLURM_JOB_ID (afterok:${SLURM_JOB_ID}, qos=${SLURM_JOB_QOS:-<default>})"
    else
        echo "WARNING: could not queue the judge pass; run it manually with" >&2
        echo "  sbatch judge_cached.sh --checkpoint $CHECKPOINTS_DIR --load_generations_dir $PER_EXAMPLE_DIR" >&2
    fi
fi
# Read by evaluate_policy.py (wandb_utils.eval_provenance_fields) so the eval
# run records the judge job that will extend it.
export JUDGE_SLURM_JOB_ID

export LD_PRELOAD="/nas/ucb/eop/.local/lib/libsqlite3.so.0"

# Run the evaluation script. Manifest-covered settings (training RM, dataset,
# KL base, temperature) are passed only when explicitly set, so the run
# manifest supplies them otherwise.
python evaluate_policy.py \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --sibling_rm_path "$SIBLING_RM_PATH" \
    --gold_rm_name "$GOLD_RM_NAME" \
    --output_file "$OUTPUT_FILE" \
    --per_example_dir "$PER_EXAMPLE_DIR" \
    --batch_size 1 \
    --generation_batch_size 32 \
    --device "$DEVICE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --benchmarks "$BENCHMARKS" \
    --evaluate_with_training_rm "$([ -n "${WITH_TRAINING_RM:-}" ] && echo True || echo False)" \
    --evaluate_with_llm_judge "$([ -n "${WITH_LLM_JUDGE:-}" ] && echo True || echo False)" \
    --llm_judge_backend "$LLM_JUDGE_BACKEND" \
    --llm_judge_model_name "$LLM_JUDGE_MODEL" \
    --llm_judge_enable_thinking "$([ -n "${JUDGE_THINKING:-}" ] && echo True || echo False)" \
    --llm_judge_max_parallel "$LLM_JUDGE_MAX_PARALLEL" \
    --llm_judge_reasoning_effort "$LLM_JUDGE_REASONING_EFFORT" \
    --llm_judge_use_batch_api "$([ -n "${JUDGE_BATCH_API:-}" ] && echo True || echo False)" \
    $([ -n "${LLM_JUDGE_MAX_NEW_TOKENS:-}" ] && echo "--llm_judge_max_new_tokens $LLM_JUDGE_MAX_NEW_TOKENS") \
    $([ -n "${LLM_JUDGE_RPM:-}" ] && echo "--llm_judge_requests_per_minute $LLM_JUDGE_RPM") \
    --arena_hard_judges "$ARENA_HARD_JUDGES" \
    --judge_selected_checkpoint_only "$([ -n "${JUDGE_SELECTED_ONLY:-}" ] && echo True || echo False)" \
    --judge_final_checkpoint "$([ -n "${JUDGE_FINAL:-}" ] && echo True || echo False)" \
    --baseline_model_path "Qwen/Qwen3-0.6B" \
    --use_dataset_response_as_baseline True \
    $([ -n "${TRAINING_RM_PATH:-}" ] && echo "--training_rm_path $TRAINING_RM_PATH") \
    $([ -n "${DATASET_NAME:-}" ] && echo "--dataset_name $DATASET_NAME") \
    $([ -n "${KL_BASE_MODEL_PATH:-}" ] && echo "--kl_base_model_path $KL_BASE_MODEL_PATH") \
    $([ -n "${EVAL_TEMPERATURE:-}" ] && echo "--eval_temperature $EVAL_TEMPERATURE") \
    --save_eval_dataset_path "evaluation_dataset_${CHECKPOINTS_DIR##*/}_$(date +%Y%m%d_%H%M%S).jsonl" \
    $([ -n "${DEBUG_MODE:-}" ] && echo "--debug True") \
    $([ ! -z "${BASE_MODEL_NAME:-}" ] && echo "--base_model_name $BASE_MODEL_NAME") \
    $([ ! -z "${SKIP_VALIDATION:-}" ] && echo "--skip_validation True") \
    $([ ! -z "${EVALUATE_CHOSEN:-}" ] && echo "--evaluate_chosen_responses True") \
    $([ ! -z "${IFEVAL_THINKING:-}" ] && echo "--ifeval_thinking True") \
    $([ ! -z "${IFEVAL_USE_GOLD_RM:-}" ] && echo "--ifeval_use_gold_rm True") \
    $([ ! -z "${NO_IFEVAL:-}" ] && echo "--evaluate_ifeval False") \
    $([ ! -z "${NO_SECONDARY_RM:-}" ] && echo "--secondary_rm_name none") \
    $([ ! -z "${WANDB_RUN_ID:-}" ] && echo "--wandb_run_id $WANDB_RUN_ID") \
    $([ -n "${BASE_EVAL_RUN_ID:-}" ] && echo "--base_eval_run_id $BASE_EVAL_RUN_ID") \
    $([ -n "${NO_BASE_CHECKPOINT:-}" ] && echo "--prepend_base_checkpoint False") \
    $([ -n "${LOAD_GENERATIONS:-}" ] && echo "--load_generations True") \
    $([ -n "${LOAD_GENERATIONS_DIR:-}" ] && echo "--load_generations_dir $LOAD_GENERATIONS_DIR") \

#     --subsample_n 25 \

# Notes:
# - --only_ifeval combined with --run_id resumes an existing wandb run and
#   logs only IFEval metrics on the custom "checkpoint" step axis; the
#   preference benchmark (and its RM scoring) is skipped entirely.
# - To disable wandb logging, add: --disable_wandb
# - To enable debug mode, add: --debug
# - For LoRA models, uncomment and set BASE_MODEL_NAME above
