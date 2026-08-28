#!/bin/bash

#SBATCH --job-name=policy_judge
# 8 CPUs: the judge itself is I/O-bound (max_parallel=32 HTTP requests in a
# ThreadPoolExecutor, so the GIL caps the useful parallelism well below the
# request count), but prompt construction / JSON decode / parquet reads run in
# that same process, and the Arena-Hard style-control fit is a 100-round torch
# bootstrap that does use several cores. More than this buys nothing -- the
# bottleneck is the proxy's queue, not local CPU (llm_judge_config_eval.md).
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --nodes=1
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --dependency=singleton
# qos/partition/account here are only the manual-submission defaults: when
# evaluate_policy.sh queues this job automatically it passes the eval job's own
# --qos/--partition/--account on the command line, which override them.
# NOTE: no --gres. A hosted judge (Vector proxy / OpenRouter) over cached
# generations loads no policy vLLM and no reward model, so the job needs no GPU.
# The nodelist is kept only for environment parity (/nas, the pyenv, ~/.bashrc).

# =============================================================================
# LLM-as-judge pass over a previous eval's cached generations. No GPU.
#
# Why this is a separate job rather than the tail of evaluate_policy.sh:
#
#   1. The proxy's RPM budget is shared project-wide and the client-side pacing
#      is PER-PROCESS (_RateLimiter is a backend instance attribute), so two
#      concurrent evals each assume the full budget and together blow past the
#      observed ~120 RPM cap. `#SBATCH --dependency=singleton` above serializes
#      every job submitted under this job name, so GPU evals can run in parallel
#      while their judge passes queue behind one another.
#   2. A proxy outage no longer costs the generation phase: judging cached
#      per-example logs is a free retry.
#   3. The GPU is released as soon as generation + RM/IFEval/KL scoring is done.
#
# Judge metrics land on the GENERATING run's wandb run: evaluate_policy.py
# stamps its live run id into the per-example `_manifest.json`, and this pass
# reads it back (resolve_load_generations_source). Pass --run_id to override.
#
# Usage
# -----
# NORMALLY YOU DO NOT SUBMIT THIS YOURSELF. evaluate_policy.sh queues it for you
# (AUTO_JUDGE, on by default), chained afterok on the eval job and pinned to the
# eval's per-example dir, so the everyday workflow stays:
#     sbatch evaluate_policy.sh
# Suppress that with `sbatch evaluate_policy.sh --no_auto_judge`.
#
# Submitting by hand -- judging the latest cached generations for the
# checkpoints dir configured in evaluate_policy.sh:
#     sbatch judge_cached.sh
#
# Chained by hand (keep BOTH dependencies -- a --dependency on the command line
# REPLACES the singleton directive above, it does not add to it):
#     GPU=$(sbatch --parsable evaluate_policy.sh --no_auto_judge)
#     sbatch --dependency=singleton,afterok:$GPU judge_cached.sh
#
# A specific run / generations dir / wandb run:
#     sbatch judge_cached.sh --checkpoint <CHECKPOINTS_DIR>
#     sbatch judge_cached.sh --load_generations_dir <PER_EXAMPLE_DIR>
#     sbatch judge_cached.sh --run_id <WANDB_RUN_ID>
#
# Judge-model / pacing overrides, and judging every cached checkpoint:
#     sbatch judge_cached.sh --judge_model Qwen3_8-27B
#     sbatch judge_cached.sh --judge_max_parallel 16 --judge_rpm 50
#     sbatch judge_cached.sh --judge_all_checkpoints   # ~28 min x n_checkpoints
#
# Every other evaluate_policy.sh flag is accepted and forwarded verbatim.
# =============================================================================

REPO=/nas/ucb/eop/Reward-Model-Overoptimization

# --llm_judge_on_cached implies: --load_generations, --with_llm_judge,
# benchmarks=preference,arena_hard, no secondary RM (see evaluate_policy.sh).
# The judge backend is the proxy by default, which is the whole premise of this
# job asking for no GPU -- pass --vllm_judge only if you also add --gres.
exec bash "$REPO/evaluate_policy.sh" --llm_judge_on_cached "$@"
