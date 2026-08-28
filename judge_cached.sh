#!/bin/bash

#SBATCH --job-name=policy_judge
# 8 CPUs: the judge is I/O-bound, but prompt construction / JSON decode /
# parquet reads share its process and the Arena-Hard style-control fit is a
# 100-round torch bootstrap. More buys nothing -- the bottleneck is the proxy's
# queue, not local CPU (llm_judge_config_eval.md).
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --nodes=1
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --dependency=singleton
# qos/partition/account are manual-submission defaults only: auto-submission
# passes the eval job's own, which override them. No --gres -- a hosted judge
# over cached generations loads no policy vLLM and no reward model. The nodelist
# is kept only for environment parity (/nas, the pyenv, ~/.bashrc).

# =============================================================================
# LLM-as-judge pass over a previous eval's cached generations. No GPU.
#
# Why a separate job rather than the tail of evaluate_policy.sh:
#
#   1. The proxy's RPM budget is project-wide but the pacing is PER-PROCESS, so
#      two concurrent evals each assume the full budget and together blow past
#      the observed ~120 cap. `--dependency=singleton` serializes every job under
#      this name, so GPU evals still run in parallel while their judges queue.
#   2. A proxy outage costs a retry, not the generation phase.
#   3. The GPU is released as soon as generation + RM/IFEval/KL is done.
#
# Metrics land on the GENERATING run: evaluate_policy.py stamps its live wandb id
# into the per-example `_manifest.json` and this pass reads it back
# (resolve_load_generations_source). Pass --run_id to override.
#
# Usage
# -----
# NORMALLY YOU DO NOT SUBMIT THIS YOURSELF. evaluate_policy.sh queues it
# (AUTO_JUDGE), chained afterok and pinned to the eval's per-example dir, so the
# everyday workflow stays `sbatch evaluate_policy.sh` (--no_auto_judge opts out).
#
# By hand -- judging the latest cached generations for the checkpoints dir
# configured in evaluate_policy.sh:
#     sbatch judge_cached.sh
#
# Chained by hand (keep BOTH dependencies -- a command-line --dependency
# REPLACES the singleton directive above rather than adding to it):
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

# --llm_judge_on_cached implies --load_generations, --with_llm_judge,
# benchmarks=preference,arena_hard and no secondary RM (see evaluate_policy.sh).
# The proxy backend is the default, which is why this job asks for no GPU --
# pass --vllm_judge only if you also add --gres.
exec bash "$REPO/evaluate_policy.sh" --llm_judge_on_cached "$@"
