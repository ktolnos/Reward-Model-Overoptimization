"""Built-in ``Evaluator`` implementations.

Each class conforms to the ``Evaluator`` protocol in ``types.py``:
    - ``name``: metric prefix
    - ``phase``: "online" (runs while policy vLLM is loaded) or "deferred"
      (runs after the policy vLLM is torn down; used by evaluators that load
      their own vLLM models, e.g. ``PairwiseEvaluator`` with an ``LLMJudge`` over
      a ``VLLMBackend``, whose phase is derived from the judge).
    - ``requires_logprobs``: if True, forces ``collect_logprobs=True`` on the
      benchmark's generation config.

To add a new evaluator, add a class here (or in another module) and attach it
to the relevant benchmark in ``benchmarks.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import wandb

from .arena_hard_upstream import CATEGORY_BASELINES
from .judges import LLMBattleDetails, RMBattleDetails, render_judge_question
from .pairwise import compute_pairwise_metrics
from .rewards import hash_responses, score_responses_with_rm
from .types import Benchmark, EvalContext, Example, GenerationResult
from .wandb_utils import log_artifact
from .generation import (
    kl_estimators_per_sample,
    teacher_forced_response_logprobs,
    update_vllm_weights,
)


# =============================================================================
# Reward-model evaluator
# =============================================================================

class RewardModelEvaluator:
    """Score a benchmark's responses with one configured reward model.

    ``rm_label`` selects the RM from ``ctx.loaded_rms`` ("gold_rm", "training_rm",
    or "secondary_rm"). Metric keys use ``benchmark.metric_key(...)``, which
    drops the benchmark prefix entirely when ``benchmark.metric_prefix == ""``.

    ``compare_vs_chosen`` additionally computes win-rate / arena / sc metrics
    against the dataset's chosen responses. Only the preference benchmark sets
    it: its examples are the ones ``precompute_chosen_scores`` scored, so the
    opt-in guarantees chosen scores are never paired with another benchmark's
    generations.
    """
    phase = "online"
    requires_logprobs = False

    def __init__(self, rm_label: str, *, compare_vs_chosen: bool = False):
        self.rm_label = rm_label
        self.compare_vs_chosen = compare_vs_chosen
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
        # Persist the policy-side score for every response.
        ctx.recorder.add_response_column(f"score__{self.name}", scores)

        out: Dict = {
            benchmark.metric_key(f"{label}/mean"): float(np.mean(scores)),
            benchmark.metric_key(f"{label}/std"): float(np.std(scores)),
        }
        if not ctx.args.disable_wandb:
            out[benchmark.metric_key(f"{label}/scores_hist")] = wandb.Histogram(scores)

        # Win-rate vs chosen: explicit opt-in (preference benchmark only), so
        # chosen scores precomputed on the preference split can never be paired
        # with another benchmark's generations. ``chosen_scores`` is None in the
        # --evaluate_chosen_responses path, where precompute is skipped.
        chosen = rms.chosen_scores(label) if self.compare_vs_chosen else None
        if chosen is not None:
            if len(chosen) != len(examples):
                raise ValueError(
                    f"[RewardModelEvaluator] chosen scores for '{label}' have "
                    f"{len(chosen)} entries but benchmark '{benchmark.name}' has "
                    f"{len(examples)} examples — chosen scores were precomputed "
                    f"on a different example set."
                )
            # Persist the reference (chosen) scores for offline win-rate (per-prompt),
            # plus the reference text once so style-controlled win-rate is recomputable.
            ctx.recorder.add_prompt_column(
                f"chosen_or_baseline_score__{label}", list(chosen)
            )
            if not ctx.recorder.has_column("reference_response_text"):
                ctx.recorder.add_prompt_column(
                    "reference_response_text",
                    [ex.metadata.get("chosen_response", "") for ex in examples],
                )
            # Frozen eval is single-sample (BENCHMARK.md §8); guard here so the
            # win-rate scores and the response texts feeding style control stay
            # aligned to the same sample (no silent first-sample slicing).
            assert n == 1, "win-rate vs chosen requires n_responses_per_example=1"
            scores_per_prompt = scores
            policy_responses_per_prompt = generation.responses
            chosen_responses = [ex.metadata["chosen_response"] for ex in examples]
            chosen_arr = np.asarray(chosen)
            battles_per_prompt = [
                [1.0 if p > c else (0.5 if p == c else 0.0)]
                for p, c in zip(scores_per_prompt, chosen_arr)
            ]
            metrics = compute_pairwise_metrics(
                battles_per_prompt,
                policy_responses_per_prompt, chosen_responses,
            )
            # Preserve legacy metric names for chart continuity.
            out[benchmark.metric_key(f"{label}/win_rate_vs_chosen")] = metrics["win_rate"]
            out[benchmark.metric_key(f"{label}/win_or_tie_rate_vs_chosen")] = metrics["win_or_tie_rate"]
            # Arena-style + style-controlled.
            for k in ("arena_score", "arena_score_ci_low", "arena_score_ci_high",
                      "sc_score", "sc_score_ci_low", "sc_score_ci_high"):
                if k in metrics:
                    out[benchmark.metric_key(f"{label}/{k}")] = metrics[k]
            for k, v in metrics.items():
                if k.startswith("sc_coef/"):
                    out[benchmark.metric_key(f"{label}/{k}")] = v

        return out


# =============================================================================
# Generic pairwise evaluator — swap any Judge (RM or LLM) without a new class
# =============================================================================

def _get_baseline_responses_for(examples: List[Example], baseline_name: str) -> List[str]:
    """Pull baseline responses out of Example metadata.

    Supports two conventions so the same evaluator works across benchmarks:
      - ``metadata["baselines"][baseline_name]`` (arena_hard: dict of model→answer)
      - ``metadata["chosen_response"]`` when baseline_name == "chosen" (preference)
    """
    if baseline_name == "chosen":
        return [ex.metadata.get("chosen_response", "") for ex in examples]
    return [ex.metadata.get("baselines", {}).get(baseline_name, "") for ex in examples]


def _judge_failure_metrics(details) -> Dict[str, int]:
    """Per-judge failure counts for wandb (empty for non-generative judges)."""
    if not isinstance(details, LLMBattleDetails):
        return {}
    return {
        "n_generation_failures": details.n_generation_failures,
        "n_truncation_failures": details.n_truncation_failures,
        "n_parse_failures": details.n_parse_failures,
        "n_dropped_prompts": details.n_dropped_prompts,
    }


class PairwiseEvaluator:
    """Compute Arena-Hard-style pairwise win metrics against one or more baselines.

    Pluggable via ``judge`` (``RMJudge`` or ``LLMJudge``). Same metric keys
    regardless of judge, so swapping judges/backends keeps charts comparable.

    Caching: for ``RMJudge`` only, the baseline-side RM scores are cached on
    disk per (RM, baseline slot, n, content hash) — independent of the policy
    being evaluated, since baseline responses don't change. The policy side is
    always re-scored. Generative judges (``LLMJudge``) skip the cache; their
    verdicts depend on the current policy responses.

    Two modes:
      - Global (default): each baseline in ``baselines`` is compared against
        every prompt. Metric key = ``{judge}/{baseline}/{metric}``.
      - Per-category (``per_category=True``): each prompt is compared against
        ``CATEGORY_BASELINES[prompt.metadata['category']]`` — matching
        upstream ``show_result.py --category``. Metric key =
        ``{judge}/{category}/{metric}``. Any baselines in ``baselines`` that
        aren't referenced by ``CATEGORY_BASELINES`` are silently ignored.
    """
    requires_logprobs = False

    def __init__(
        self,
        judge,
        baselines: List[str],
        per_category: bool = False,
    ):
        self.judge = judge
        self.baselines = baselines
        self.per_category = per_category
        self.name = f"pairwise_{judge.name}"
        # A judge that loads its own GPU model (LLMJudge over a VLLMBackend) runs
        # deferred, after the policy vLLM is torn down; API/RM judges run online.
        self.phase = getattr(judge, "phase", "online")

    def teardown(self) -> None:
        """Release judge-held resources (e.g. a deferred vLLM model)."""
        if hasattr(self.judge, "teardown"):
            self.judge.teardown()

    def _run_judge(
        self,
        slot_name: str,             # metric slot label ("baseline_model" or category)
        prompt_messages_list,
        policy_responses,
        baseline_responses,
        ctx,
    ) -> Optional[tuple]:
        if not all(baseline_responses):
            print(
                f"[pairwise:{self.judge.name}] slot '{slot_name}' has missing "
                f"baseline responses; skipping."
            )
            return None

        print(
            f"[pairwise:{self.judge.name}] judging {len(policy_responses)} prompts "
            f"for {slot_name} (ckpt {ctx.checkpoint_num})..."
        )

        # Cache strategy:
        #   - RMJudge: cache the baseline-side scores. Pure function of
        #     (RM, baseline content), so safe to reuse across runs.
        #   - LLMJudge etc.: no cache; the verdict depends on the policy.
        if getattr(self.judge, "kind", None) == "rm":
            safe_slot = slot_name.replace("/", "_")
            cache_key = (
                f"{safe_slot}__n{len(baseline_responses)}__"
                f"{hash_responses(baseline_responses)}"
            )
            battles, details = self.judge.score_pairs(
                prompt_messages_list, policy_responses, baseline_responses, ctx,
                baseline_cache_key=cache_key,
            )
        else:
            battles, details = self.judge.score_pairs(
                prompt_messages_list, policy_responses, baseline_responses, ctx,
            )
        return battles, details

    def _record_pairwise(self, recorder, slot_name, indices, battles, details,
                         baseline_responses) -> None:
        """Persist per-prompt judge signals into the recorder.

        ``indices`` are the global prompt indices these battles correspond to
        (the full range in global mode; a category's subset in per-category
        mode). We store the raw signals that determine the battle outcomes —
        for RM judges the policy + baseline scores, for LLM judges the two
        per-game labels — plus the baseline response text, so any pairwise
        metric (arena_score, sc_score, win-rate vs any reference) can be
        recomputed offline without re-loading the baseline answer files.
        """
        jn = self.judge.name
        safe = slot_name.replace("/", "_")
        battle_mean = [float(np.mean(b)) if len(b) else float("nan") for b in battles]
        recorder.add_sparse_prompt_column(
            f"battle_mean__{jn}__{safe}", indices, battle_mean, fill=float("nan"),
        )
        recorder.add_sparse_prompt_column(
            f"baseline_response_text__{safe}", indices, list(baseline_responses), fill="",
        )
        if isinstance(details, RMBattleDetails):
            # Policy-side score is independent of the baseline slot; accumulate
            # it into a single column (sparse-merge handles per-category).
            recorder.add_sparse_prompt_column(
                f"score__{jn}", indices, list(details.policy_scores), fill=float("nan"),
            )
            recorder.add_sparse_prompt_column(
                f"chosen_or_baseline_score__{jn}__{safe}", indices,
                list(details.baseline_scores), fill=float("nan"),
            )
        elif isinstance(details, LLMBattleDetails):
            recorder.add_sparse_prompt_column(
                f"judge_label_game0__{jn}__{safe}", indices,
                details.game0_labels, fill=None,
            )
            recorder.add_sparse_prompt_column(
                f"judge_label_game1__{jn}__{safe}", indices,
                details.game1_labels, fill=None,
            )
        else:
            raise TypeError(
                f"Unknown judge battle details type: {type(details).__name__}"
            )

    def _persist_pairwise(self, ctx, benchmark, examples, slot_name, indices,
                          battles, details, policy_responses, baseline_responses) -> None:
        """Persist per-prompt judge signals.

        Online judges have a ``ctx.recorder`` and stream columns into the shared
        per-example log. Deferred judges (vLLM) run after that log is written, so
        their raw verdicts go to a dedicated judge file instead.
        """
        if ctx.recorder is not None:
            self._record_pairwise(ctx.recorder, slot_name, indices, battles,
                                  details, baseline_responses)
        else:
            self._write_deferred_judge_file(
                ctx, benchmark, examples, slot_name, battles, details,
                policy_responses, baseline_responses,
            )

    def _write_deferred_judge_file(self, ctx, benchmark, examples, slot_name,
                                   battles, details, policy_responses,
                                   baseline_responses) -> None:
        """Write one row per prompt (question, both answers, both swapped-game
        raw judge texts + labels, battle outcomes) to a parquet/jsonl file next
        to the per-example logs, joinable via ``prompt_uid``."""
        import os
        import pandas as pd
        from .persistence import example_uid, resolve_per_example_dir

        g0_labels = getattr(details, "game0_labels", None)
        g1_labels = getattr(details, "game1_labels", None)
        g0_texts = getattr(details, "game0_texts", None)
        g1_texts = getattr(details, "game1_texts", None)
        rows = []
        for k, ex in enumerate(examples):
            b = battles[k]
            rows.append({
                "benchmark": benchmark.name,
                "checkpoint": ctx.checkpoint_num,
                "prompt_uid": example_uid(ex),
                "slot": slot_name,
                "question": render_judge_question(ex.prompt_messages),
                "policy_response": policy_responses[k],
                "baseline_response": baseline_responses[k],
                "game0_label": g0_labels[k] if g0_labels else None,
                "game1_label": g1_labels[k] if g1_labels else None,
                "game0_text": g0_texts[k] if g0_texts else None,
                "game1_text": g1_texts[k] if g1_texts else None,
                "battle_mean": float(np.mean(b)) if len(b) else float("nan"),
                "n_battles": len(b),
            })

        per_example_dir = resolve_per_example_dir(ctx.args)
        os.makedirs(per_example_dir, exist_ok=True)
        fmt = ctx.args.per_example_format
        ext = "jsonl" if fmt == "jsonl" else "parquet"
        safe_bench = benchmark.name.replace("/", "_")
        safe_slot = slot_name.replace("/", "_")
        path = os.path.join(
            per_example_dir,
            f"{safe_bench}__checkpoint-{ctx.checkpoint_num}__{self.name}__{safe_slot}.{ext}",
        )
        df = pd.DataFrame(rows)
        if fmt == "jsonl":
            df.to_json(path, orient="records", lines=True, force_ascii=False)
        else:
            df.to_parquet(path, index=False)
        print(f"[{self.name}] wrote {len(df)} judge rows -> {path}")

    def _eval_global(
        self, benchmark, examples, prompt_messages_list, policy_responses, ctx,
    ) -> Dict[str, float]:
        out: Dict = {}
        all_indices = list(range(len(examples)))
        for baseline_name in self.baselines:
            baseline_responses = _get_baseline_responses_for(examples, baseline_name)
            result = self._run_judge(
                baseline_name, prompt_messages_list, policy_responses,
                baseline_responses, ctx,
            )
            if result is None:
                continue
            battles, details = result
            self._persist_pairwise(ctx, benchmark, examples, baseline_name, all_indices,
                                   battles, details, policy_responses, baseline_responses)
            metrics = compute_pairwise_metrics(battles, policy_responses, baseline_responses)
            metrics.update(_judge_failure_metrics(details))
            for k, v in metrics.items():
                out[f"{self.judge.name}/{baseline_name}/{k}"] = v
        return out

    def _eval_per_category(
        self, benchmark, examples, prompt_messages_list, policy_responses, ctx,
    ) -> Dict[str, float]:
        """For each category, filter to its prompts and compare vs the upstream
        baseline for that category (CATEGORY_BASELINES)."""
        out: Dict = {}
        # Group prompt indices by category.
        by_category: Dict[str, List[int]] = {}
        for i, ex in enumerate(examples):
            cat = ex.metadata.get("category")
            if cat is None:
                continue
            by_category.setdefault(cat, []).append(i)

        for category, baseline_name in CATEGORY_BASELINES.items():
            indices = by_category.get(category, [])
            if not indices:
                continue
            sub_examples = [examples[i] for i in indices]
            sub_prompts = [prompt_messages_list[i] for i in indices]
            sub_policy = [policy_responses[i] for i in indices]
            sub_baseline = _get_baseline_responses_for(sub_examples, baseline_name)
            result = self._run_judge(
                f"{category}__vs_{baseline_name}",
                sub_prompts, sub_policy, sub_baseline, ctx,
            )
            if result is None:
                continue
            battles, details = result
            # Record under the category slot, scattered back to global indices.
            self._persist_pairwise(ctx, benchmark, sub_examples, category, indices,
                                   battles, details, sub_policy, sub_baseline)
            metrics = compute_pairwise_metrics(battles, sub_policy, sub_baseline)
            metrics.update(_judge_failure_metrics(details))
            # Metric slot is the category name (matches upstream's leaderboard format).
            for k, v in metrics.items():
                out[f"{self.judge.name}/{category}/{k}"] = v
            # Also record which baseline each category used (diagnostic only).
            out[f"{self.judge.name}/{category}/baseline_model"] = baseline_name
        return out

    def evaluate(
        self,
        benchmark: Benchmark,
        examples: List[Example],
        generation: GenerationResult,
        ctx: EvalContext,
    ) -> dict:
        n = generation.n_responses_per_example
        # Frozen eval is single-sample (BENCHMARK.md §8). Assert rather than
        # silently judging only responses[::n] (the first sample per prompt),
        # which previously diverged from the RM /mean that averages all samples.
        assert n == 1, "PairwiseEvaluator requires n_responses_per_example=1 (frozen eval)"
        policy_responses = generation.responses
        prompt_messages_list = [ex.prompt_messages for ex in examples]

        if self.per_category:
            raw = self._eval_per_category(benchmark, examples, prompt_messages_list, policy_responses, ctx)
        else:
            raw = self._eval_global(benchmark, examples, prompt_messages_list, policy_responses, ctx)

        return {benchmark.metric_key(k): v for k, v in raw.items()}


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

        # Persist per-prompt strict/loose follow-all flags.
        ctx.recorder.add_response_column(
            "ifeval_prompt_strict", [bool(x) for x in prompt_strict]
        )
        ctx.recorder.add_response_column(
            "ifeval_prompt_loose", [bool(x) for x in prompt_loose]
        )

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

@dataclass
class _PendingKL:
    """Per-checkpoint stash the KL base pass consumes after the loop."""
    full_ids_list: List[List[int]]
    prompt_lens_list: List[int]
    policy_token_logprobs: List[List[float]]
    prompt_uids: List[str]


class KLEvaluator:
    """KL(policy || base_policy) per-sample, logged as mean/std.

    Both sides are teacher-forced through the SAME vLLM engine — identical
    kernels/numerics, exact per-token alignment, and no second copy of any
    model in GPU memory. The two sides run at different times:

    - ``evaluate`` (online, per checkpoint): policy logprobs under the
      checkpoint weights that just generated (no weight load), plus a stash of
      the token ids + policy logprobs for the base pass.
    - ``run_base_phase`` (once, after the checkpoint loop, engine still
      alive): the base weights are swapped in a single time and every stashed
      checkpoint is scored. All generation is done by then, so the checkpoint
      weights never need restoring — one base-weight load per eval instead of
      two extra loads per checkpoint.

    ``evaluate`` therefore returns no metrics; they arrive keyed by checkpoint
    from ``run_base_phase`` (via ``eval_utils.run_kl_base_phase``) and are
    merged into the result rows with the deferred metrics.

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
        self._pending: Dict[int, _PendingKL] = {}

    def evaluate(self, benchmark, examples, generation, ctx):
        if generation.full_ids_list is None:
            print("[KLEvaluator] generation did not collect token ids; skipping")
            return {}

        from .persistence import example_uid

        print("[KLEvaluator] teacher-forcing policy logprobs...")
        policy_mean_lp, policy_token_lp = teacher_forced_response_logprobs(
            ctx.llm, generation.full_ids_list, generation.prompt_lens_list,
        )
        ctx.recorder.add_response_column("policy_mean_logprob", policy_mean_lp)
        self._pending[ctx.checkpoint_num] = _PendingKL(
            full_ids_list=generation.full_ids_list,
            prompt_lens_list=generation.prompt_lens_list,
            policy_token_logprobs=policy_token_lp,
            prompt_uids=[example_uid(ex) for ex in examples],
        )
        return {}

    def run_base_phase(self, llm, benchmark, args) -> Dict[int, dict]:
        """Score every stashed checkpoint against the base model.

        Swaps the base weights into the (still-alive) engine once and leaves
        them there — nothing generates after this phase. Returns
        ``{checkpoint_num: metrics}``.
        """
        if not self._pending:
            return {}
        print(f"[KLEvaluator] swapping in KL base weights ({self.base_model_path}) "
              f"to score {len(self._pending)} checkpoints...")
        update_vllm_weights(llm, self.base_model_path)

        k = benchmark.metric_key
        metrics_by_ckpt: Dict[int, dict] = {}
        for ckpt_num, pending in sorted(self._pending.items()):
            base_mean_lp, base_token_lp = teacher_forced_response_logprobs(
                llm, pending.full_ids_list, pending.prompt_lens_list,
            )
            kl_per_sample, kl_grpo_per_sample = kl_estimators_per_sample(
                pending.policy_token_logprobs, base_token_lp,
            )
            self._write_kl_file(args, benchmark, ckpt_num, pending,
                                kl_per_sample, kl_grpo_per_sample, base_mean_lp)
            metrics_by_ckpt[ckpt_num] = {
                k("kl/mean"): float(np.mean(kl_per_sample)),
                k("kl/std"): float(np.std(kl_per_sample)),
                k("kl/grpo_mean"): float(np.mean(kl_grpo_per_sample)),
                k("kl/grpo_std"): float(np.std(kl_grpo_per_sample)),
            }
        self._pending.clear()
        return metrics_by_ckpt

    def _write_kl_file(self, args, benchmark, ckpt_num, pending,
                       kl_per_sample, kl_grpo_per_sample, base_mean_lp) -> None:
        """Persist per-response KL + base logprobs, one file per checkpoint.

        The shared per-example log is already written by the time the base
        pass runs (``policy_mean_logprob`` lives there), so the KL columns go
        to a dedicated file joinable via ``prompt_uid`` — the same pattern as
        the deferred judge verdicts.
        """
        import os
        import pandas as pd
        from .persistence import resolve_per_example_dir

        per_example_dir = resolve_per_example_dir(args)
        os.makedirs(per_example_dir, exist_ok=True)
        fmt = args.per_example_format
        ext = "jsonl" if fmt == "jsonl" else "parquet"
        safe_bench = benchmark.name.replace("/", "_")
        path = os.path.join(
            per_example_dir, f"{safe_bench}__checkpoint-{ckpt_num}__kl.{ext}",
        )
        df = pd.DataFrame({
            "benchmark": benchmark.name,
            "checkpoint": ckpt_num,
            "prompt_uid": pending.prompt_uids,
            "kl__k1": kl_per_sample,
            "kl__grpo": kl_grpo_per_sample,
            "base_mean_logprob": base_mean_lp,
        })
        if fmt == "jsonl":
            df.to_json(path, orient="records", lines=True, force_ascii=False)
        else:
            df.to_parquet(path, index=False)
        print(f"[kl] wrote {len(df)} rows -> {path}")

