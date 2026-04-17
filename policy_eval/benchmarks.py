"""Built-in benchmarks.

A benchmark bundles (prompt set + generation config + evaluators). The main
loop iterates benchmarks × checkpoints and is benchmark-agnostic — adding a
new benchmark requires only a new factory function registered in
``BENCHMARK_BUILDERS``.

Current builtins:
    - ``preference``: generates on a preference dataset's prompts, scores with
      reward models, optionally computes KL + LLM-judge metrics.
    - ``ifeval``: generates on google/IFEval prompts, scores with rule-based
      strict/loose matching. Optionally attach the gold RM evaluator too via
      ``--ifeval_use_gold_rm``.

See ``Benchmark`` in ``types.py`` for the core shape.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from datasets import load_dataset
from vllm import SamplingParams

from data_utils import (
    _apply_chat_template_no_thinking,
    format_and_validate_preference_sample,
    get_generation_stop_token_ids,
)

from .evaluators import (
    IfevalRuleEvaluator,
    KLEvaluator,
    LLMJudgeAPIEvaluator,
    LLMJudgeVLLMEvaluator,
    RewardModelEvaluator,
)
from .types import Benchmark, Example, GenerationConfig


# ---------------------------------------------------------------------------
# Preference benchmark (HelpSteer-style dataset with chosen/rejected)
# ---------------------------------------------------------------------------

def _load_preference_examples(args) -> List[Example]:
    ds = load_dataset(args.dataset_name)
    split_priority = ("validation", "test", "train")
    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        split = next((s for s in split_priority if s in ds), splits[0])
        print(f"[preference] Using split '{split}' from {splits}")
        dataset = ds[split]
    else:
        dataset = ds

    if args.debug:
        dataset = dataset.select(range(min(100, len(dataset))))
    elif args.subsample_n is not None and args.subsample_n < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(args.subsample_n))
        print(f"[preference] Subsampling to {args.subsample_n} prompts.")

    if "chosen" not in dataset.column_names:
        raise ValueError("Preference dataset must have a 'chosen' column.")

    examples = []
    for ex in dataset:
        prompt_messages = ex["chosen"][:-1]
        chosen_resp = ex["chosen"][-1]["content"]
        examples.append(Example(
            prompt_messages=prompt_messages,
            metadata={
                "chosen_response": chosen_resp,
                "rejected_messages": ex.get("rejected"),
            },
        ))
    print(f"[preference] Loaded {len(examples)} examples")
    return examples


def _format_preference_prompt(example: Example, tokenizer, thinking: bool) -> str:
    chosen_messages = list(example.prompt_messages) + [
        {"role": "assistant", "content": example.metadata.get("chosen_response", "")}
    ]
    prompt_text, _, _ = format_and_validate_preference_sample(
        chosen_messages,
        tokenizer,
        rejected_messages=example.metadata.get("rejected_messages"),
        length_config="default",
        skip_validation=True,
        sample_id=0,
        context="Evaluation",
    )
    return prompt_text


def build_preference_benchmark(args) -> Benchmark:
    evaluators: List = [RewardModelEvaluator("gold_rm")]
    if args.evaluate_with_training_rm:
        evaluators.append(RewardModelEvaluator("training_rm"))
    if args.secondary_rm_name and args.secondary_rm_name.lower() != "none":
        evaluators.append(RewardModelEvaluator("secondary_rm"))
    if args.kl_base_model_path:
        evaluators.append(KLEvaluator(args.kl_base_model_path))
    if args.evaluate_with_llm_judge:
        if args.llm_judge_backend == "api":
            evaluators.append(LLMJudgeAPIEvaluator(
                model_name=args.llm_judge_model_name,
                max_new_tokens=args.llm_judge_max_new_tokens,
            ))
        elif args.llm_judge_backend == "vllm":
            evaluators.append(LLMJudgeVLLMEvaluator(
                judge_model_path=args.llm_judge_model_name,
                max_new_tokens=args.llm_judge_max_new_tokens,
            ))
        else:
            raise ValueError(f"Unknown llm_judge_backend: {args.llm_judge_backend}")

    # Temperature 0 by default; LLM judge needs diversity, so bump to 0.7 if attached.
    has_judge = any(e.name.startswith("llm_judge") for e in evaluators)
    sampling_params = SamplingParams(
        temperature=0.7 if has_judge else 0,
        top_p=0.9 if has_judge else 1.0,
        max_tokens=args.max_new_tokens,
        n=args.num_responses_per_prompt or 1,
        logprobs=1 if any(e.requires_logprobs for e in evaluators) else None,
        stop_token_ids=None,  # filled in by generate_responses_vllm
    )
    gen_config = GenerationConfig(
        sampling_params=sampling_params,
        thinking=True,  # preference benchmark keeps thinking (matches original behavior)
        n_responses_per_example=args.num_responses_per_prompt or 1,
        collect_logprobs=any(e.requires_logprobs for e in evaluators),
    )
    return Benchmark(
        name="preference",
        load_examples=_load_preference_examples,
        format_prompt=_format_preference_prompt,
        generation_config=gen_config,
        evaluators=evaluators,
        # Empty prefix keeps preference metric keys identical to the pre-refactor
        # layout (e.g. ``gold_rm/mean``, ``kl/grpo_mean``), so wandb charts
        # from existing runs stay valid when resuming via --wandb_run_id.
        metric_prefix="",
    )


# ---------------------------------------------------------------------------
# IFEval benchmark (google/IFEval)
# ---------------------------------------------------------------------------

_IFEVAL_MAX_NEW_TOKENS_NO_THINK = 1280
_IFEVAL_MAX_NEW_TOKENS_THINK = 1280 + 32768


def _load_ifeval_examples(args) -> List[Example]:
    ds = load_dataset("google/IFEval", split="train")
    print(f"[ifeval] Loaded {len(ds)} prompts (thinking={args.ifeval_thinking})")
    if args.debug:
        ds = ds.select(range(min(50, len(ds))))
    examples = []
    for ex in ds:
        examples.append(Example(
            prompt_messages=[{"role": "user", "content": ex["prompt"]}],
            metadata={
                "ifeval": {
                    "key": ex["key"],
                    "instruction_id_list": ex["instruction_id_list"],
                    "prompt": ex["prompt"],
                    "kwargs": ex["kwargs"],
                },
            },
        ))
    return examples


def _format_ifeval_prompt(example: Example, tokenizer, thinking: bool) -> str:
    messages = example.prompt_messages
    if thinking:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return _apply_chat_template_no_thinking(
        tokenizer, messages, add_generation_prompt=True,
    )


def build_ifeval_benchmark(args) -> Benchmark:
    evaluators: List = [IfevalRuleEvaluator()]
    if args.ifeval_use_gold_rm:
        evaluators.append(RewardModelEvaluator("gold_rm"))

    thinking = args.ifeval_thinking
    ifeval_max_tokens = (
        _IFEVAL_MAX_NEW_TOKENS_THINK if thinking else _IFEVAL_MAX_NEW_TOKENS_NO_THINK
    )
    # Match the legacy behaviour that respects the user's --max_new_tokens floor.
    ifeval_max_tokens = max(ifeval_max_tokens, args.max_new_tokens)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=ifeval_max_tokens,
        stop_token_ids=None,
    )
    gen_config = GenerationConfig(
        sampling_params=sampling_params,
        thinking=thinking,
        n_responses_per_example=1,
        collect_logprobs=False,
        extra_max_model_len=_IFEVAL_MAX_NEW_TOKENS_THINK if thinking else None,
    )
    return Benchmark(
        name="ifeval",
        load_examples=_load_ifeval_examples,
        format_prompt=_format_ifeval_prompt,
        generation_config=gen_config,
        evaluators=evaluators,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_BUILDERS: Dict[str, Callable] = {
    "preference": build_preference_benchmark,
    "ifeval": build_ifeval_benchmark,
}


def build_benchmarks(args) -> List[Benchmark]:
    names = args.selected_benchmarks()
    out = []
    for n in names:
        if n not in BENCHMARK_BUILDERS:
            raise ValueError(
                f"Unknown benchmark '{n}'. Known: {list(BENCHMARK_BUILDERS)}. "
                f"Register a new one in BENCHMARK_BUILDERS."
            )
        out.append(BENCHMARK_BUILDERS[n](args))
    return out
