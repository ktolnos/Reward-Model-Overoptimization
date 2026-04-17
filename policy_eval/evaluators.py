"""Built-in ``Evaluator`` implementations.

Each class conforms to the ``Evaluator`` protocol in ``types.py``:
    - ``name``: metric prefix
    - ``phase``: "online" (runs while policy vLLM is loaded) or "deferred"
      (runs after the policy vLLM is torn down; used by evaluators that load
      their own vLLM models, e.g. LLMJudgeVLLMEvaluator).
    - ``requires_logprobs``: if True, forces ``collect_logprobs=True`` on the
      benchmark's generation config.

To add a new evaluator, add a class here (or in another module) and attach it
to the relevant benchmark in ``benchmarks.py``.
"""
from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import requests
import wandb

from .rewards import score_responses_with_rm
from .types import Benchmark, EvalContext, Example, GenerationResult
from .wandb_utils import log_artifact
from .generation import get_log_probs_from_ids


# =============================================================================
# Reward-model evaluator
# =============================================================================

class RewardModelEvaluator:
    """Score a benchmark's responses with one configured reward model.

    ``rm_label`` selects the RM from ``ctx.loaded_rms`` ("gold_rm", "training_rm",
    or "secondary_rm"). Metric keys use ``benchmark.metric_key(...)``, which
    drops the benchmark prefix entirely when ``benchmark.metric_prefix == ""``.
    """
    phase = "online"
    requires_logprobs = False

    def __init__(self, rm_label: str):
        self.rm_label = rm_label
        self.name = f"rm_{rm_label}"

    def evaluate(
        self,
        benchmark: Benchmark,
        examples: List[Example],
        generation: GenerationResult,
        ctx: EvalContext,
    ) -> dict:
        rms = ctx.loaded_rms
        entry = rms.get(self.rm_label) if rms else None
        if entry is None:
            print(f"[RewardModelEvaluator] RM '{self.rm_label}' not loaded, skipping.")
            return {}
        model, tokenizer = entry

        n = generation.n_responses_per_example
        prompt_messages_list = [ex.prompt_messages for ex in examples for _ in range(n)]

        scores = score_responses_with_rm(
            generation.responses, prompt_messages_list, model, tokenizer,
            batch_size=ctx.args.batch_size, device=ctx.args.device,
            checkpoint_num=ctx.checkpoint_num,
        )

        label = self.rm_label
        out: Dict = {
            benchmark.metric_key(f"{label}/mean"): float(np.mean(scores)),
            benchmark.metric_key(f"{label}/std"): float(np.std(scores)),
        }
        if not ctx.args.disable_wandb:
            out[benchmark.metric_key(f"{label}/scores_hist")] = wandb.Histogram(scores)

        # Win-rate vs chosen: only valid when this benchmark's examples carry a
        # chosen_response AND the cached chosen_scores array lines up with them.
        # The cache is populated by precompute_chosen_scores (preference-only);
        # without this guard, running the preference benchmark alongside another
        # benchmark that shares an RM evaluator would silently leak preference
        # chosen scores into the other benchmark's win-rate metric.
        has_chosen_metadata = all("chosen_response" in ex.metadata for ex in examples)
        chosen = rms.chosen_scores(label) if (rms and has_chosen_metadata) else None
        if chosen is not None and len(chosen) == len(examples) and len(chosen) > 0:
            if n > 1:
                scores_per_prompt = scores.reshape(-1, n).mean(axis=1)
            else:
                scores_per_prompt = scores
            wins = (scores_per_prompt > chosen).sum()
            ties = (scores_per_prompt == chosen).sum()
            out[benchmark.metric_key(f"{label}/win_rate_vs_chosen")] = float(wins) / len(chosen)
            out[benchmark.metric_key(f"{label}/win_or_tie_rate_vs_chosen")] = float(wins + ties) / len(chosen)

        return out


# =============================================================================
# IFEval rule-based evaluator (strict/loose instruction-following match)
# =============================================================================

class IfevalRuleEvaluator:
    """Official IFEval rule-based scoring.

    Expects each ``Example.metadata["ifeval"]`` to contain
    ``{key, instruction_id_list, prompt, kwargs}``.
    """
    phase = "online"
    requires_logprobs = False
    name = "ifeval_rule"

    def evaluate(
        self,
        benchmark: Benchmark,
        examples: List[Example],
        generation: GenerationResult,
        ctx: EvalContext,
    ) -> dict:
        from lm_eval.tasks.ifeval.utils import (
            InputExample,
            test_instruction_following_strict,
            test_instruction_following_loose,
        )

        n = generation.n_responses_per_example
        if n != 1:
            raise ValueError("IfevalRuleEvaluator currently supports n_responses_per_example=1")

        prompt_strict, prompt_loose = [], []
        inst_strict_all, inst_loose_all = [], []
        for example, response in zip(examples, generation.responses):
            meta = example.metadata["ifeval"]
            inp = InputExample(
                key=meta["key"],
                instruction_id_list=meta["instruction_id_list"],
                prompt=meta["prompt"],
                kwargs=meta["kwargs"],
            )
            out_strict = test_instruction_following_strict(inp, response)
            out_loose = test_instruction_following_loose(inp, response)
            prompt_strict.append(out_strict.follow_all_instructions)
            prompt_loose.append(out_loose.follow_all_instructions)
            inst_strict_all.extend(out_strict.follow_instruction_list)
            inst_loose_all.extend(out_loose.follow_instruction_list)

        n_truncated = sum(1 for fr in generation.finish_reasons if fr == "length")
        if n_truncated:
            print(f"[IFEval] WARNING: {n_truncated}/{len(generation.finish_reasons)} "
                  f"responses truncated by max_tokens")

        if not ctx.args.disable_wandb and wandb.run is not None:
            table = wandb.Table(columns=[
                "key", "prompt", "raw_response", "stripped_response", "finish_reason",
            ])
            for ex, raw, stripped, fr in zip(
                examples, generation.raw_responses, generation.responses,
                generation.finish_reasons,
            ):
                meta = ex.metadata["ifeval"]
                table.add_data(meta["key"], meta["prompt"], raw, stripped, fr)
            log_artifact({benchmark.metric_key(f"responses_{ctx.checkpoint_num}"): table})

        k = benchmark.metric_key
        results = {
            k("prompt_strict_acc"): sum(prompt_strict) / len(prompt_strict),
            k("prompt_loose_acc"): sum(prompt_loose) / len(prompt_loose),
            k("inst_strict_acc"): sum(inst_strict_all) / len(inst_strict_all),
            k("inst_loose_acc"): sum(inst_loose_all) / len(inst_loose_all),
            k("n_truncated"): n_truncated,
        }
        print(
            f"[IFEval] strict={results[k('prompt_strict_acc')]:.4f}  "
            f"loose={results[k('prompt_loose_acc')]:.4f}  "
            f"inst_strict={results[k('inst_strict_acc')]:.4f}  "
            f"inst_loose={results[k('inst_loose_acc')]:.4f}"
        )
        return results


# =============================================================================
# KL divergence evaluator (requires logprobs from generation)
# =============================================================================

class KLEvaluator:
    """KL(policy || base_policy) per-sample, logged as mean/std.

    Two estimators:
      - ``kl/mean``: mean-log-prob difference (simple per-sequence)
      - ``kl/grpo_mean``: per-token ``exp(diff) - diff - 1`` (matches GRPO's KL term)

    Needs ``collect_logprobs=True`` on the benchmark's generation config, which
    is set automatically via ``requires_logprobs = True``.
    """
    phase = "online"
    requires_logprobs = True
    name = "kl"

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self._base_model = None

    def _ensure_base_model(self, ctx: EvalContext):
        if self._base_model is not None:
            return self._base_model
        from data_utils import load_policy_and_tokenizer
        print(f"Loading base policy for KL from {self.base_model_path}...")
        model, _ = load_policy_and_tokenizer(self.base_model_path)
        self._base_model = model.to(ctx.args.device).eval()
        return self._base_model

    def evaluate(self, benchmark, examples, generation, ctx):
        if generation.full_ids_list is None:
            print("[KLEvaluator] generation did not collect logprobs; skipping")
            return {}
        base_model = self._ensure_base_model(ctx)

        _, base_mean_lp, base_token_lp_list = get_log_probs_from_ids(
            base_model, generation.full_ids_list, generation.prompt_lens_list,
            ctx.args.device, batch_size=4,
        )

        kl_per_sample = generation.policy_mean_logprobs - np.array(base_mean_lp)
        kl_mean = float(np.mean(kl_per_sample))
        kl_std = float(np.std(kl_per_sample))

        kl_grpo_per_sample = []
        for pol_lp, ref_lp in zip(generation.policy_token_logprobs, base_token_lp_list):
            min_len = min(len(pol_lp), len(ref_lp))
            pol = np.array(pol_lp[:min_len])
            ref = np.array(ref_lp[:min_len])
            diff = ref - pol
            per_token = np.exp(diff) - diff - 1
            kl_grpo_per_sample.append(np.mean(per_token) if len(per_token) else 0.0)

        k = benchmark.metric_key
        return {
            k("kl/mean"): kl_mean,
            k("kl/std"): kl_std,
            k("kl/grpo_mean"): float(np.mean(kl_grpo_per_sample)),
            k("kl/grpo_std"): float(np.std(kl_grpo_per_sample)),
        }


# =============================================================================
# LLM-as-judge (API backend: OpenRouter)
# =============================================================================

class LLMJudgeAPIEvaluator:
    """Pairwise LLM judge via OpenRouter API.

    Expects ``ctx.baseline_responses`` (one per example) set by the preference
    benchmark. Uses the Skywork judge template.

    NOTE: The original implementation has an unresolved bug around how
    chat-template-formatted prompts are plugged into the judge template. Kept
    raising NotImplementedError until that's fixed; structure preserved so the
    fix can land without changing wiring.
    """
    phase = "online"
    requires_logprobs = False
    name = "llm_judge_api"

    def __init__(self, model_name: str, max_new_tokens: int = 2048):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def evaluate(self, benchmark, examples, generation, ctx):
        if ctx.baseline_responses is None:
            raise ValueError("LLMJudgeAPIEvaluator needs ctx.baseline_responses")
        verdicts, judge_responses = _openrouter_judge(
            examples, generation.responses, ctx.baseline_responses,
            args=ctx.args, model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
        )
        return _verdict_metrics(verdicts, benchmark)


def _openrouter_judge(examples, policy_responses, baseline_responses, *, args, model_name, max_new_tokens):
    from reward_utils import (
        Skywork_PROMPT, Skywork_SYSTEM_PROMPT, Skywork_ASSISTANT_PROMPT,
        extract_reward_from_response,
    )
    raise NotImplementedError(
        "LLM judge API evaluation is not currently supported. "
        "Known issue: `prompts` passed here are chat-template-formatted strings "
        "(e.g. containing <|im_start|> tokens) but get plugged into the Skywork "
        "judge template as the raw 'question', corrupting judge input. "
        "Fix: extract raw user question from examples[i].prompt_messages[-1]['content']."
    )


# =============================================================================
# LLM-as-judge (vLLM backend: deferred, loads its own model)
# =============================================================================

class LLMJudgeVLLMEvaluator:
    """Pairwise LLM judge via a separately-loaded vLLM instance.

    This is a **deferred** evaluator: it runs after the policy vLLM is torn
    down. The main loop caches per-checkpoint responses and feeds them back in
    at the deferred phase, so the judge loads exactly once.

    Unimplemented stub for now — the abstraction is in place so this can be
    filled in without changing the main loop. Recipe:
        1. Load vLLM(judge_model_path).
        2. Build Skywork prompts from (prompt, policy_response, baseline_response).
        3. Call llm.generate(...) and parse the preference from each generation.
        4. Emit win/loss/tie metrics per checkpoint via log_metrics(...).
    """
    phase = "deferred"
    requires_logprobs = False
    name = "llm_judge_vllm"

    def __init__(self, judge_model_path: str, max_new_tokens: int = 2048):
        self.judge_model_path = judge_model_path
        self.max_new_tokens = max_new_tokens

    def evaluate(self, benchmark, examples, generation, ctx):
        raise NotImplementedError(
            "LLMJudgeVLLMEvaluator is a scaffold. Implement with a deferred "
            "entry point that receives {checkpoint_num: GenerationResult} and "
            "loads the judge vLLM exactly once."
        )


def _verdict_metrics(verdicts: List[int], benchmark: Benchmark) -> dict:
    wins = verdicts.count(1)
    losses = verdicts.count(-1)
    ties = verdicts.count(0)
    total = len(verdicts) or 1
    k = benchmark.metric_key
    return {
        k("llm_judge/win_rate"): wins / total,
        k("llm_judge/loss_rate"): losses / total,
        k("llm_judge/tie_rate"): ties / total,
        k("llm_judge/mean"): float(np.mean(verdicts)) if verdicts else 0.0,
        k("llm_judge/mean_no_tie"): (
            (wins - losses) / (wins + losses) if (wins + losses) > 0 else 0.0
        ),
        k("llm_judge/total_comparisons"): total,
    }
