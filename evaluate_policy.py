"""Policy evaluation entry point.

``ScriptArguments`` (the CLI) lives in this file. The reusable framework lives
in ``policy_eval/``:
    - ``policy_eval.types``: Benchmark/Evaluator/Example abstractions.
    - ``policy_eval.benchmarks``: built-in benchmarks + registry.
    - ``policy_eval.evaluators``: built-in evaluators (RM, IFEval-rule, KL, judges).
    - ``policy_eval.generation``: vLLM lifecycle.
    - ``policy_eval.rewards``: reward-model loading + chosen-score cache.
    - ``policy_eval.wandb_utils``: wandb init (with resume) + custom-step-axis logging.
    - ``policy_eval.eval_utils``: orchestration helpers used below.

Adding IFEval (or any benchmark) to an existing wandb run:
    python evaluate_policy.py \
        --checkpoints_dir ... \
        --wandb_run_id <run_id> \
        --benchmarks ifeval \
        --evaluate_with_training_rm False \
        --secondary_rm_name none

The custom ``checkpoint`` step axis (set up in wandb_utils) lets the new
metrics land at the correct checkpoint numbers even when logged out of order
or added to a run with a different original step axis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from data_utils import setup_tokenizer

from policy_eval import wandb_utils
from policy_eval.benchmarks import build_benchmarks
from policy_eval.eval_utils import (
    chosen_responses_as_generation,
    fetch_training_history,
    list_checkpoints,
    lookup_train_metrics,
    make_baseline_responses,
    rms_required_by,
    run_chosen_only,
    run_deferred_phase,
)
from policy_eval.generation import (
    generate_responses_vllm,
    resolve_vllm_base_model,
    update_vllm_weights,
    vllm_session,
)
from policy_eval.rewards import LoadedRewardModels
from policy_eval.types import Benchmark, EvalContext, Example, GenerationResult


# =============================================================================
# CLI arguments
# =============================================================================

@dataclass
class ScriptArguments:
    """CLI for policy evaluation.

    Flags group naturally:
      - Core inputs: --checkpoints_dir, --dataset_name.
      - Reward models: --gold_rm_name, --training_rm_path, --secondary_rm_name
        (set to "none" to disable).
      - Benchmark selection: --benchmarks (comma-separated). Add a new
        benchmark by registering it in ``policy_eval.benchmarks.BENCHMARK_BUILDERS``.
      - Wandb: --wandb_run_id resumes an existing run; new metrics land on the
        custom "checkpoint" step axis regardless of log order.

    Per-benchmark knobs (e.g. --ifeval_thinking, --ifeval_use_gold_rm) are read
    by the benchmark builders; benchmarks interpret them and attach the
    appropriate evaluators.
    """
    # ------------------------------------------------------------------
    # Core inputs
    # ------------------------------------------------------------------
    checkpoints_dir: str = field(
        default="", metadata={"help": "Directory containing policy checkpoints"}
    )
    dataset_name: str = field(
        default="ktolnos/helpsteer3-preference-chosenrrejected",
        metadata={"help": "Name of the preference dataset (for the 'preference' benchmark)"},
    )
    split: str = field(
        default="test",
        metadata={"help": "Preference dataset split to evaluate: 'validation' for "
                          "hyperparameter sweeps, 'test' for final/truth eval. "
                          "Raises if the split is absent."},
    )

    # ------------------------------------------------------------------
    # Reward models (shared across evaluators)
    # ------------------------------------------------------------------
    training_rm_path: str = field(
        default="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40",
        metadata={"help": "Path to the reward model used during training"},
    )
    gold_rm_name: str = field(
        default="Ray2333/GRM-Gemma2-2B-rewardmodel-ft",
        metadata={"help": "Name of the gold reward model"},
    )
    secondary_rm_name: Optional[str] = field(
        default="Ray2333/GRM-Gemma-2B-sftreg",
        metadata={"help": "Secondary reward model for cross-validation. Set to 'none' to disable."},
    )
    evaluate_with_training_rm: Optional[bool] = field(
        default=True,
        metadata={"help": "Attach the training RM evaluator to the preference benchmark"},
    )

    # ------------------------------------------------------------------
    # Benchmark selection
    # ------------------------------------------------------------------
    benchmarks: str = field(
        default="preference,ifeval,arena_hard",
        metadata={"help": "Comma-separated list of benchmarks to run."},
    )
    # Legacy flag: disable IFEval if False (equivalent to dropping it from --benchmarks).
    evaluate_ifeval: Optional[bool] = field(
        default=True,
        metadata={"help": "DEPRECATED: prefer --benchmarks. If False, drops 'ifeval' from --benchmarks."},
    )
    ifeval_thinking: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable thinking for IFEval generation (matches official leaderboard)."},
    )
    ifeval_use_gold_rm: Optional[bool] = field(
        default=True,
        metadata={"help": "Also score IFEval responses with the gold RM."},
    )
    arena_hard_dataset: Optional[str] = field(
        default="lmarena-ai/arena-hard-auto",
        metadata={"help": "HF dataset repo holding Arena-Hard-Auto. The v2.0 files live under data/arena-hard-v2.0/."},
    )
    arena_hard_baseline_models: Optional[str] = field(
        default="auto",
        metadata={"help": "Comma-separated list of baseline models. Special token 'auto' expands to the per-category baselines used by the official Arena-Hard-Auto v2.0 leaderboard (hard_prompt/coding/math → o3-mini-2025-01-31, creative_writing → gemini-2.0-flash-001) and switches evaluation to per-category mode. Mixing 'auto' with explicit baselines runs both."},
    )
    arena_hard_judges: Optional[str] = field(
        default="rm:gold_rm",
        metadata={"help": "Comma-separated judge specs. Use 'rm:<label>' for a reward model judge or 'llm:<model>' for an Arena-Hard-style OpenRouter API judge (e.g. 'llm:openai/gpt-4.1'). Multiple judges can be combined: 'rm:gold_rm,llm:openai/gpt-4.1'."},
    )
    # ------------------------------------------------------------------
    # LLM judge (attaches to the preference benchmark when enabled)
    # ------------------------------------------------------------------
    evaluate_with_llm_judge: Optional[bool] = field(
        default=False,
        metadata={"help": "Use an LLM judge to compare policy vs baseline on the preference benchmark"},
    )
    llm_judge_backend: str = field(
        default="api",
        metadata={"help": "'api' (OpenRouter) or 'vllm' (local model, deferred phase)"},
    )
    llm_judge_model_name: Optional[str] = field(
        default="google/gemma-7b-it",
        metadata={"help": "LLM judge model name (OpenRouter id or HF path)"},
    )
    openrouter_api_key: Optional[str] = field(
        default=None, metadata={"help": "OpenRouter API key (falls back to env)"},
    )
    baseline_model_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the baseline model for judge comparison"},
    )
    use_dataset_response_as_baseline: Optional[bool] = field(
        default=False, metadata={"help": "Use the dataset 'chosen' column as baseline"},
    )
    llm_judge_max_new_tokens: Optional[int] = field(
        default=2048, metadata={"help": "Max new tokens for the LLM judge"},
    )

    # ------------------------------------------------------------------
    # KL divergence (attaches to preference benchmark when kl_base_model_path set)
    # ------------------------------------------------------------------
    kl_base_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Base policy path for KL(policy || base). Enables KL evaluator."},
    )

    # ------------------------------------------------------------------
    # Generation parameters
    # ------------------------------------------------------------------
    max_length: Optional[int] = field(default=1024, metadata={"help": "Max prompt length"})
    max_new_tokens: Optional[int] = field(default=1024, metadata={"help": "Max new tokens"})
    num_responses_per_prompt: Optional[int] = field(
        default=1, metadata={"help": "Samples per prompt (LLM judge)"},
    )
    gpu_memory_utilization: float = field(
        default=0.3, metadata={"help": "vLLM GPU memory utilization"},
    )
    length_config: Optional[str] = field(
        default="default", metadata={"help": "Name from DATASET_LENGTH_CONFIGS"},
    )
    skip_validation: Optional[bool] = field(
        default=False, metadata={"help": "Skip prompt-length validation"},
    )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    batch_size: Optional[int] = field(default=8)
    generation_batch_size: Optional[int] = field(default=8)
    device: Optional[str] = field(default="cuda")
    output_file: Optional[str] = field(default="evaluation_results.csv")
    save_eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Save per-checkpoint responses to this jsonl path"},
    )
    base_model_name: Optional[str] = field(
        default=None, metadata={"help": "Base model for LoRA checkpoints"},
    )

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    wandb_project: Optional[str] = field(default="policy-evaluation")
    wandb_run_name: Optional[str] = field(default=None)
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Resume a prior wandb run by id (e.g. add IFEval to an existing run). "
            "When set, new metrics are logged on the custom 'checkpoint' step axis so "
            "they appear at the correct checkpoint numbers regardless of log order."
        },
    )
    disable_wandb: Optional[bool] = field(default=False)
    training_wandb_project: Optional[str] = field(
        default="grpo",
        metadata={
            "help": "Project containing the GRPO training run, used to mirror its "
            "metrics into this eval run under train/ on the checkpoint axis. "
            "The training run is found by group=checkpoints_dir. "
            "Set to 'none' to disable."
        },
    )

    # ------------------------------------------------------------------
    # Debug / sub-sampling
    # ------------------------------------------------------------------
    debug: Optional[bool] = field(default=False)
    subsample_n: Optional[int] = field(default=None)
    evaluate_chosen_responses: Optional[bool] = field(
        default=False,
        metadata={"help": "Score dataset chosen responses instead of generating from a policy"},
    )

    def selected_benchmarks(self) -> list:
        """Parse --benchmarks into a list, honouring legacy --evaluate_ifeval=False."""
        names = [n.strip() for n in (self.benchmarks or "").split(",") if n.strip()]
        if not self.evaluate_ifeval and "ifeval" in names:
            names.remove("ifeval")
        return names


# =============================================================================
# Main
# =============================================================================

def main():
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    wandb_utils.init_wandb(args)

    benchmarks = build_benchmarks(args)
    if not benchmarks:
        raise ValueError(
            f"No benchmarks selected. --benchmarks was {args.benchmarks!r}; "
            f"effective list was empty (evaluate_ifeval={args.evaluate_ifeval})."
        )
    print(f"Benchmarks: {[b.name for b in benchmarks]}")
    for b in benchmarks:
        print(f"  - {b.name}: evaluators={[e.name for e in b.evaluators]}")

    # ----- Lazily load the reward models actually used by evaluators --------
    loaded_rms: Optional[LoadedRewardModels] = None
    rm_labels_needed = rms_required_by(benchmarks)
    if rm_labels_needed:
        loaded_rms = LoadedRewardModels(args, rm_labels_needed)

    # ----- Load each benchmark's examples once ------------------------------
    bench_examples: Dict[str, List[Example]] = {
        b.name: b.load_examples(args) for b in benchmarks
    }

    # ----- Precompute chosen scores for the preference benchmark ------------
    preference_bench = next((b for b in benchmarks if b.name == "preference"), None)
    if loaded_rms is not None and preference_bench is not None and not args.evaluate_chosen_responses:
        prompt_messages = [ex.prompt_messages for ex in bench_examples["preference"]]
        fake_dataset = [
            {"chosen": list(ex.prompt_messages) + [
                {"role": "assistant", "content": ex.metadata.get("chosen_response", "")}
            ]}
            for ex in bench_examples["preference"]
        ]
        loaded_rms.precompute_chosen_scores(fake_dataset, prompt_messages, args)

    # ----- Chosen-responses-only path (no vLLM) -----------------------------
    if args.evaluate_chosen_responses:
        return run_chosen_only(args, benchmarks, bench_examples, loaded_rms)

    # ----- Resolve checkpoints + tokenizer ----------------------------------
    checkpoints, single_model_path, first_checkpoint_path = list_checkpoints(args)
    vllm_base = resolve_vllm_base_model(first_checkpoint_path)

    train_history = None
    if not args.disable_wandb:
        train_history = fetch_training_history(args.checkpoints_dir, args.training_wandb_project)

    print("Loading tokenizer...")
    policy_tokenizer = AutoTokenizer.from_pretrained(
        first_checkpoint_path, trust_remote_code=True,
    )
    setup_tokenizer(policy_tokenizer, model_name=vllm_base)

    # ----- Max model length: honor any benchmark's extra_max_model_len ------
    max_model_len = args.max_length + args.max_new_tokens
    for b in benchmarks:
        if b.generation_config.extra_max_model_len:
            max_model_len = max(max_model_len, b.generation_config.extra_max_model_len)

    # ----- Main loop --------------------------------------------------------
    results_rows: List[dict] = []
    deferred_cache: Dict[Tuple[str, int], GenerationResult] = {}
    full_eval_data: Dict[str, Dict[int, list]] = {b.name: {} for b in benchmarks}

    try:
        with vllm_session(
            base_model_path=vllm_base,
            initial_checkpoint_path=first_checkpoint_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as llm:

            baseline_responses = None
            if preference_bench is not None:
                baseline_responses = make_baseline_responses(
                    args, preference_bench, bench_examples["preference"],
                    policy_tokenizer,
                )

            for idx, ckpt_name in enumerate(tqdm(checkpoints, desc="Checkpoints")):
                ckpt_path = single_model_path or os.path.join(args.checkpoints_dir, ckpt_name)
                ckpt_num = int(ckpt_name.split("-")[1])
                print(f"\n=== {ckpt_name} ===")
                if idx != 0:
                    update_vllm_weights(llm, ckpt_path)

                ctx = EvalContext(
                    args=args,
                    checkpoint_num=ckpt_num,
                    checkpoint_path=ckpt_path,
                    llm=llm,
                    policy_tokenizer=policy_tokenizer,
                    loaded_rms=loaded_rms,
                    baseline_responses=baseline_responses,
                )

                combined_metrics: dict = {}
                for bench in benchmarks:
                    examples = bench_examples[bench.name]
                    prompts = [
                        bench.format_prompt(ex, policy_tokenizer, bench.generation_config.thinking)
                        for ex in examples
                    ]
                    print(f"[{bench.name}] generating {len(prompts)} prompts "
                          f"(thinking={bench.generation_config.thinking}, "
                          f"logprobs={bench.generation_config.collect_logprobs})")
                    generation = generate_responses_vllm(
                        llm, prompts, policy_tokenizer, bench.generation_config,
                    )

                    if bench.deferred_evaluators:
                        deferred_cache[(bench.name, ckpt_num)] = generation

                    if args.save_eval_dataset_path:
                        full_eval_data[bench.name][ckpt_num] = [
                            {"prompt": p, "response": r}
                            for p, r in zip(prompts, generation.responses)
                        ]

                    for ev in bench.online_evaluators:
                        metrics = ev.evaluate(bench, examples, generation, ctx)
                        combined_metrics.update(metrics)

                combined_metrics["checkpoint"] = ckpt_num
                train_metrics = lookup_train_metrics(train_history, ckpt_num)
                combined_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
                wandb_utils.log_metrics(
                    {k: v for k, v in combined_metrics.items() if k != "checkpoint"},
                    checkpoint_num=ckpt_num,
                )
                results_rows.append({
                    k: v for k, v in combined_metrics.items()
                    if not isinstance(v, wandb.Histogram)
                })

        # ----- Deferred phase (policy vLLM is torn down) --------------------
        if deferred_cache:
            run_deferred_phase(benchmarks, bench_examples, deferred_cache, args, loaded_rms)

    finally:
        if loaded_rms is not None:
            loaded_rms.unload()

    # ----- Save CSV + jsonl -------------------------------------------------
    if results_rows:
        out = args.output_file
        if args.debug and out.endswith(".csv"):
            out = out.replace(".csv", "_debug.csv")
        pd.DataFrame(results_rows).to_csv(out, index=False)
        print(f"\nResults saved to {out}")

    if args.save_eval_dataset_path and any(full_eval_data[b.name] for b in benchmarks):
        with open(args.save_eval_dataset_path, "w") as f:
            for bench_name, per_ckpt in full_eval_data.items():
                for ckpt_num, rows in per_ckpt.items():
                    for row in rows:
                        f.write(json.dumps({
                            "benchmark": bench_name,
                            "checkpoint": ckpt_num,
                            **row,
                        }) + "\n")
        print(f"Full evaluation data saved to {args.save_eval_dataset_path}")

    wandb_utils.finish()


if __name__ == "__main__":
    main()
