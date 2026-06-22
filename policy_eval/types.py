"""Core abstractions for the pluggable benchmark/evaluator framework.

An evaluation run = (policy checkpoints) × (benchmarks) × (evaluators per benchmark).

Each ``Benchmark`` owns a prompt set and a generation config. Each ``Evaluator``
scores the generated responses and returns a dict of metrics. A single benchmark
can carry multiple evaluators (e.g. IFEval can be graded by both rule-based
strict/loose matching AND a reward model).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol

import numpy as np
from vllm import SamplingParams


@dataclass
class Example:
    """One benchmark example.

    ``prompt_messages`` is the chat-format conversation up to and including the
    final user turn. Evaluators that need the raw prompt text should re-format
    from this (the RM evaluator does).

    ``metadata`` holds anything evaluator-specific. Common keys:
      - ``chosen_response`` (str): reference response for win-rate metrics.
      - ``ifeval`` (dict): IFEval-specific fields (instruction_id_list, kwargs, key).
    """
    prompt_messages: List[dict]
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """How to generate on a benchmark."""
    sampling_params: SamplingParams
    thinking: bool = True
    n_responses_per_example: int = 1
    collect_logprobs: bool = False
    # If IFEval-style thinking budget is needed, the benchmark raises it here.
    extra_max_model_len: Optional[int] = None


@dataclass
class GenerationResult:
    """Output of one benchmark generation pass for one checkpoint."""
    responses: List[str]            # post-processed (thinking-stripped) responses
    raw_responses: List[str]        # pre-strip responses as returned by vLLM
    finish_reasons: List[str]
    n_responses_per_example: int = 1
    # Generated token count per response (len of vLLM completion.token_ids),
    # always populated — used for length/verbosity gating + over-budget accounting.
    response_token_lens: Optional[List[int]] = None
    # Populated only when collect_logprobs=True:
    full_ids_list: Optional[List[List[int]]] = None
    prompt_lens_list: Optional[List[int]] = None
    policy_mean_logprobs: Optional[np.ndarray] = None
    policy_token_logprobs: Optional[List[List[float]]] = None


@dataclass
class EvalContext:
    """Per-call context passed to evaluators.

    ``llm`` and ``policy_tokenizer`` are only populated during the online phase.
    Deferred evaluators (e.g. LLM judge loading its own vLLM instance) run after
    the policy vLLM is torn down and see ``llm=None``.
    """
    args: Any                               # ScriptArguments
    checkpoint_num: int
    checkpoint_path: Optional[str]
    llm: Optional[Any]                      # vllm.LLM when online, None when deferred
    policy_tokenizer: Optional[Any]
    loaded_rms: Optional[Any]               # LoadedRewardModels
    baseline_responses: Optional[List[str]] = None
    # Per-example log sink (policy_eval.persistence.PerExampleRecorder) for this
    # (benchmark, checkpoint). Online evaluators add their per-example score
    # columns to it. Always set during the online and chosen-only phases; None
    # in the deferred phase (no per-example evaluators run there yet).
    recorder: Optional[Any] = None


class Evaluator(Protocol):
    """An evaluator consumes generated responses and returns metrics.

    Attributes:
        name: short identifier, used as a metric prefix (e.g. "ifeval_rule").
        phase: either ``"online"`` (runs during the checkpoint loop while the
            policy vLLM is loaded) or ``"deferred"`` (runs after the main loop
            with responses cached across all checkpoints).
        requires_logprobs: if True, the benchmark's generation_config will be
            forced to ``collect_logprobs=True``.
    """
    name: str
    phase: str              # "online" | "deferred"
    requires_logprobs: bool

    def evaluate(
        self,
        benchmark: "Benchmark",
        examples: List[Example],
        generation: GenerationResult,
        ctx: EvalContext,
    ) -> dict:
        """Score a single (benchmark, checkpoint) pair. Returns metric dict."""
        ...


@dataclass
class Benchmark:
    """A benchmark = prompt set + generation config + evaluator list.

    ``load_examples`` is a callable that takes the script args and returns a list
    of ``Example``. It is called once at startup (not per checkpoint).

    ``format_prompt`` renders an Example into a vLLM-ready prompt string.

    ``metric_prefix`` controls the namespace for this benchmark's metrics.
    Default ``None`` uses ``name`` (so ``ifeval`` benchmark emits ``ifeval/...``).
    Set to the empty string ``""`` to drop the prefix entirely, which keeps
    metric keys compatible with the pre-refactor single-benchmark layout
    (preference benchmark uses this to preserve chart continuity across resumes).
    """
    name: str
    load_examples: Callable[[Any], List[Example]]
    format_prompt: Callable[[Example, Any, bool], str]   # (example, tokenizer, thinking) -> str
    generation_config: GenerationConfig
    evaluators: List[Evaluator] = field(default_factory=list)
    metric_prefix: Optional[str] = None

    def metric_key(self, local: str) -> str:
        """Return a fully-qualified metric key for this benchmark.

        ``metric_key("gold_rm/mean")`` →
          - ``"ifeval/gold_rm/mean"`` when ``metric_prefix`` is ``None`` (uses name)
          - ``"gold_rm/mean"`` when ``metric_prefix == ""``
          - ``"<prefix>/gold_rm/mean"`` otherwise
        """
        prefix = self.metric_prefix if self.metric_prefix is not None else self.name
        return f"{prefix}/{local}" if prefix else local

    @property
    def online_evaluators(self) -> List[Evaluator]:
        return [e for e in self.evaluators if getattr(e, "phase", "online") == "online"]

    @property
    def deferred_evaluators(self) -> List[Evaluator]:
        return [e for e in self.evaluators if getattr(e, "phase", "online") == "deferred"]

    @property
    def needs_logprobs(self) -> bool:
        return any(getattr(e, "requires_logprobs", False) for e in self.evaluators)
