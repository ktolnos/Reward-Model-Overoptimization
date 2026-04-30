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

import os
from typing import Callable, Dict, List

from datasets import load_dataset
from vllm import SamplingParams

from data_utils import (
    _apply_chat_template_no_thinking,
    format_and_validate_preference_sample,
    get_generation_stop_token_ids,
)

from .arena_hard_upstream import CATEGORY_BASELINES
from .evaluators import (
    IfevalRuleEvaluator,
    KLEvaluator,
    LLMJudgeVLLMEvaluator,
    PairwiseEvaluator,
    RewardModelEvaluator,
)
from .judges import LLMAPIJudge, RMJudge
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
            # Route API judges through the unified PairwiseEvaluator against
            # the dataset's ``chosen`` response. Produces the same Arena-style
            # metrics (arena_score, sc_score, CIs) as arena_hard — directly
            # comparable across benchmarks.
            evaluators.append(PairwiseEvaluator(
                judge=LLMAPIJudge(
                    args.llm_judge_model_name,
                    max_tokens=args.llm_judge_max_new_tokens,
                ),
                baselines=["chosen"],
            ))
        elif args.llm_judge_backend == "vllm":
            evaluators.append(LLMJudgeVLLMEvaluator(
                judge_model_path=args.llm_judge_model_name,
                max_new_tokens=args.llm_judge_max_new_tokens,
            ))
        else:
            raise ValueError(f"Unknown llm_judge_backend: {args.llm_judge_backend}")

    # Temperature 0 by default; LLM judge needs diversity, so bump to 0.7 if attached.
    has_judge = any(e.name.startswith(("pairwise_llm", "llm_judge")) for e in evaluators)
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
# Arena-Hard-Auto v2.0 benchmark
#
# Prompts from lmarena-ai/arena-hard-auto (data/arena-hard-v2.0/question.jsonl),
# baseline responses from a pre-generated model answer file bundled in the
# same HF dataset (default: Qwen3-4B). Scoring + win-rate against that baseline
# is done by a single ArenaHardRMEvaluator, which caches baseline RM scores on
# disk so baselines are only scored once per (RM, dataset) pair.
# ---------------------------------------------------------------------------

_ARENA_HARD_MAX_NEW_TOKENS = 2048
# Prompt budget (in tokens) reserved on top of generation budget when sizing
# the engine's max_model_len. Some Arena-Hard prompts exceed 2k tokens.
_ARENA_HARD_PROMPT_BUDGET = 4096


def _extract_baseline_answer_text(messages: List[dict]) -> str:
    """Pull the assistant's answer text out of an Arena-Hard answer record.

    Upstream stores the assistant turn's ``content`` as either a plain string or
    (when thinking is on) a dict ``{"thought": ..., "answer": ...}``.
    """
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if isinstance(content, dict):
        # Thinking-mode answer record.
        return content.get("answer") or content.get("response") or ""
    if isinstance(content, str):
        return content
    return ""


_AUTO_BASELINES = "auto"


def _parse_baseline_models(raw: str) -> List[str]:
    """Split a comma-separated baseline list.

    Recognizes the special token ``auto`` which expands into the set of models
    referenced by upstream's ``JUDGE_SETTINGS`` (per-category baselines). When
    ``auto`` is present the evaluator is configured to score per-category
    using ``CATEGORY_BASELINES`` — exactly what ``show_result.py --category``
    does on the leaderboard.
    """
    if not raw:
        return []
    tokens = [m.strip() for m in raw.split(",") if m.strip()]
    out: List[str] = []
    for tok in tokens:
        if tok == _AUTO_BASELINES:
            for base in dict.fromkeys(CATEGORY_BASELINES.values()):  # preserve order, dedupe
                if base not in out:
                    out.append(base)
        elif tok not in out:
            out.append(tok)
    return out


def _uses_auto_baselines(raw: str) -> bool:
    return _AUTO_BASELINES in [t.strip() for t in (raw or "").split(",") if t.strip()]


def _download_arena_hard_files(repo_id: str, baseline_models: List[str]) -> tuple:
    """Fetch question + each baseline's model_answer JSONL files from HF."""
    from huggingface_hub import snapshot_download

    q_rel = "data/arena-hard-v2.0/question.jsonl"
    answer_rels = [
        f"data/arena-hard-v2.0/model_answer/{m}.jsonl" for m in baseline_models
    ]
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[q_rel] + answer_rels,
    )
    q_path = os.path.join(local_dir, q_rel)
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Arena-Hard question file not found at {q_path}")
    answer_paths = {}
    for model, rel in zip(baseline_models, answer_rels):
        p = os.path.join(local_dir, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Baseline answer file not found at {p}. Check that '{model}' "
                f"exists under data/arena-hard-v2.0/model_answer/ in {repo_id}."
            )
        answer_paths[model] = p
    return q_path, answer_paths


def _load_arena_hard_examples(args) -> List[Example]:
    import json

    baseline_models = _parse_baseline_models(args.arena_hard_baseline_models)
    if not baseline_models:
        raise ValueError(
            "arena_hard_baseline_models is empty; provide at least one baseline."
        )

    q_path, answer_paths = _download_arena_hard_files(
        args.arena_hard_dataset, baseline_models
    )

    questions = []
    with open(q_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    baselines_by_uid: Dict[str, Dict[str, str]] = {q["uid"]: {} for q in questions}
    for model, path in answer_paths.items():
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                uid = rec.get("uid")
                if uid is None or uid not in baselines_by_uid:
                    continue
                baselines_by_uid[uid][model] = _extract_baseline_answer_text(
                    rec.get("messages", [])
                )

    print(
        f"[arena_hard] Loaded {len(questions)} prompts; baselines={baseline_models}"
    )
    for model in baseline_models:
        n_have = sum(1 for uid in baselines_by_uid if model in baselines_by_uid[uid])
        if n_have != len(questions):
            print(
                f"[arena_hard] WARNING: {model} has {n_have}/{len(questions)} answers"
            )

    if args.debug:
        questions = questions[:min(50, len(questions))]
    elif args.subsample_n is not None and args.subsample_n < len(questions):
        import random
        rng = random.Random(42)
        questions = rng.sample(questions, args.subsample_n)
        print(f"[arena_hard] Subsampling to {args.subsample_n} prompts.")

    examples = []
    for q in questions:
        uid = q["uid"]
        examples.append(Example(
            prompt_messages=[{"role": "user", "content": q["prompt"]}],
            metadata={
                "uid": uid,
                "category": q.get("category"),
                "subcategory": q.get("subcategory"),
                "baselines": baselines_by_uid.get(uid, {}),
            },
        ))
    return examples


def _format_arena_hard_prompt(example: Example, tokenizer, thinking: bool) -> str:
    return _apply_chat_template_no_thinking(
        tokenizer, example.prompt_messages, add_generation_prompt=True,
    )


def _build_judges(args) -> list:
    """Build one judge per entry in ``args.arena_hard_judges``.

    Syntax: comma-separated tokens.
      - ``rm:<label>``  → RMJudge using a loaded RM ("gold_rm", "training_rm", ...)
      - ``llm:<model>`` → LLMAPIJudge (OpenRouter) with Arena-Hard prompt template.
                          e.g. "llm:openai/gpt-4.1". Default temp 0, max_tokens 16000.
    """
    out = []
    for tok in (t.strip() for t in args.arena_hard_judges.split(",") if t.strip()):
        if tok.startswith("rm:"):
            out.append(RMJudge(tok[3:]))
        elif tok.startswith("llm:"):
            out.append(LLMAPIJudge(tok[4:]))
        else:
            raise ValueError(
                f"Unknown judge spec '{tok}'. Use 'rm:<label>' or 'llm:<model>'."
            )
    if not out:
        raise ValueError("arena_hard_judges is empty.")
    return out


def build_arena_hard_benchmark(args) -> Benchmark:
    baseline_models = _parse_baseline_models(args.arena_hard_baseline_models)
    per_category = _uses_auto_baselines(args.arena_hard_baseline_models)
    judges = _build_judges(args)
    evaluators: List = [
        PairwiseEvaluator(
            judge,
            baselines=baseline_models,
            per_category=per_category,
        )
        for judge in judges
    ]

    max_tokens = max(_ARENA_HARD_MAX_NEW_TOKENS, args.max_new_tokens)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop_token_ids=None,
    )
    gen_config = GenerationConfig(
        sampling_params=sampling_params,
        thinking=False,
        n_responses_per_example=1,
        collect_logprobs=False,
        extra_max_model_len=max_tokens + _ARENA_HARD_PROMPT_BUDGET,
    )
    return Benchmark(
        name="arena_hard",
        load_examples=_load_arena_hard_examples,
        format_prompt=_format_arena_hard_prompt,
        generation_config=gen_config,
        evaluators=evaluators,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_BUILDERS: Dict[str, Callable] = {
    "preference": build_preference_benchmark,
    "ifeval": build_ifeval_benchmark,
    "arena_hard": build_arena_hard_benchmark,
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
