"""Pairwise judges: compare two answers to a prompt and emit battle outcomes.

A judge decides which of (answer_a, answer_b) is better for a given prompt.
Two implementations:

    - ``RMJudge``: score each answer with a reward model; winner has the higher
      score. One battle per prompt.
    - ``LLMJudge``: ask an LLM (local vLLM via ``VLLMBackend`` or a remote API via
      ``OpenRouterBackend``) using the Arena-Hard-Auto v2.0 prompt template. Runs
      2 games per prompt with position swap; each game's label is mapped (with
      weighting) to one or more {0.0, 0.5, 1.0} battles, matching ``show_result.py``
      exactly so that scores produced here match the official leaderboard when the
      same judge/baseline are used.

Both return ``(battles_per_prompt, extras)`` where ``battles_per_prompt`` is a
list of lists: ``battles_per_prompt[i]`` = battle scores for prompt ``i`` from
answer_a's perspective (1.0 = A wins, 0.0 = A loses, 0.5 = tie). The
``PairwiseEvaluator`` aggregates these into Arena scores and bootstrap CIs.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import requests

from .rewards import score_responses_with_rm


@dataclass
class RMBattleDetails:
    """Per-prompt reward-model scores behind each RM-judge battle.

    ``policy_scores``/``baseline_scores`` are aligned to the prompts passed to
    ``score_pairs`` (answer_a = policy, answer_b = baseline)."""
    policy_scores: np.ndarray
    baseline_scores: np.ndarray


@dataclass
class LLMBattleDetails:
    """Per-prompt position-swapped game labels behind each LLM-judge battle.

    ``game0_labels`` is the A=baseline/B=policy game, ``game1_labels`` the
    A=policy/B=baseline game; entries are ``None`` when a game failed to parse.
    ``game0_texts``/``game1_texts`` hold the raw judge generations when the judge
    captures them (both backends do).

    Failure counts are over the ``2 * n_prompts`` games unless noted:
    ``n_generation_failures`` = empty judge output (API HTTP failure after retries,
    or an empty local generation); ``n_parse_failures`` = non-empty output with no
    parsable verdict; ``n_dropped_prompts`` = prompts excluded from metrics because
    at least one of their two games had no usable verdict."""
    game0_labels: List[Optional[str]]
    game1_labels: List[Optional[str]]
    game0_texts: Optional[List[str]] = None
    game1_texts: Optional[List[str]] = None
    n_generation_failures: int = 0
    n_truncation_failures: int = 0
    n_parse_failures: int = 0
    n_dropped_prompts: int = 0


@dataclass
class JudgeGeneration:
    """One judge completion: the raw ``text`` and whether it hit the token limit.

    ``truncated`` lets the judge attribute an unparsable verdict to truncation
    (the model ran out of tokens) rather than a genuine parse failure."""
    text: str
    truncated: bool = False


# ---------------------------------------------------------------------------
# Arena-Hard-Auto v2.0 default prompt/template/regex (from utils/judge_utils.py
# + config/arena-hard-v2.0.yaml in lmarena-ai/arena-hard-auto).
# ---------------------------------------------------------------------------

ARENA_HARD_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by "
    "two AI assistants to the user prompt displayed below. You will be given assistant A's "
    "answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your "
    "answers before judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. "
    "You must identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means "
    "the answer correctly responds to the prompt or follows the instructions. Note when user "
    "prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate "
    "to ask for clarifications or more information from the user than providing an answer based "
    "on assumptions. Relevant means all parts of the response closely connect or are appropriate "
    "to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
    "identify any missing important information in the assistants' answers that would be "
    "beneficial to include when responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your "
    "final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

ARENA_HARD_PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n"
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
)

ARENA_HARD_REGEX_PATTERNS = [
    r"\[\[([AB<>=]+)\]\]",
    r"\[([AB<>=]+)\]",
]


def render_judge_question(messages: List[dict]) -> str:
    """Render a prompt conversation for the judge's ``<|User Prompt|>`` slot.

    Single-turn prompts pass through verbatim, matching upstream Arena-Hard
    byte-for-byte. Multi-turn conversations become a role-labelled transcript,
    so the judge sees the same full dialog the reward models score with
    (RM judges re-template the complete conversation; without this the LLM
    judge saw only the final user turn).
    """
    if len(messages) == 1:
        return messages[0]["content"]
    return "\n\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)

# label_to_score from show_result.py. ``weight`` multiplies the strong labels,
# expanding each prompt into multiple battles. With weight=3 the final
# per-prompt battle count is 2 games × (weight or 1 depending on label
# strength). Bootstrap is over this expanded battle list.
def _label_to_score(weight: int = 3) -> Dict[str, List[float]]:
    return {
        "A>B":  [1.0],
        "A>>B": [1.0] * weight,
        "A=B":  [0.5],
        "A<<B": [0.0] * weight,
        "A<B":  [0.0],
        "B>A":  [0.0],
        "B>>A": [0.0] * weight,
        "B=A":  [0.5],
        "B<<A": [1.0] * weight,
        "B<A":  [1.0],
    }


def battles_from_game_labels(
    game0_labels: List[Optional[str]],
    game1_labels: List[Optional[str]],
    *,
    weight: int = 3,
) -> Tuple[List[List[float]], int]:
    """Apply the Arena-Hard ``show_result.py`` formula to two swapped games.

    ``game0`` is the A=baseline/B=policy game, ``game1`` the A=policy/B=baseline
    game. Returns ``(battles_per_prompt, dropped)`` where each per-prompt entry is
    a list of battle scores in {0.0, 0.5, 1.0} from the policy's perspective
    (decisive verdicts expand to ``weight`` copies); an empty list marks a prompt
    where either game failed to parse. Mirrors:

        scores = label_to_score[game1] + [1 - s for s in label_to_score[game0]]
    """
    label_score = _label_to_score(weight)
    battles: List[List[float]] = []
    dropped = 0
    for lbl0, lbl1 in zip(game0_labels, game1_labels):
        if (lbl0 is None or lbl1 is None
                or lbl0 not in label_score or lbl1 not in label_score):
            dropped += 1
            battles.append([])
            continue
        ms1 = label_score[lbl1]
        ms0 = [1.0 - s for s in label_score[lbl0]]
        battles.append(ms1 + ms0)
    return battles, dropped


# ---------------------------------------------------------------------------
# RM judge
# ---------------------------------------------------------------------------

class RMJudge:
    """Score both answers with a reward model; higher score wins.

    Produces exactly one battle per prompt.
    """
    kind = "rm"

    def __init__(self, rm_label: str):
        self.rm_label = rm_label
        self.name = f"rm_{rm_label}"

    def score_pairs(
        self,
        prompts_messages: List[list],
        answers_a: List[str],
        answers_b: List[str],
        ctx,
        *,
        baseline_cache_key: Optional[str] = None,
    ) -> Tuple[List[List[float]], RMBattleDetails]:
        rms = ctx.loaded_rms
        entry = rms.get(self.rm_label) if rms else None
        if entry is None:
            raise ValueError(f"RMJudge needs RM '{self.rm_label}' loaded")
        model, tokenizer = entry

        # Always re-score the policy side: its responses change every checkpoint.
        scores_a = score_responses_with_rm(
            answers_a, prompts_messages, model, tokenizer,
            batch_size=ctx.args.batch_size, device=ctx.args.device,
            checkpoint_num=ctx.checkpoint_num,
        )
        # Baseline side: cached on disk when caller supplies a key (the cache
        # is keyed only on the baseline content + RM, not on the policy).
        if baseline_cache_key is not None:
            scores_b = rms.score_with_cache(
                self.rm_label, answers_b, prompts_messages,
                cache_namespace="baseline_score_cache",
                cache_key=baseline_cache_key,
                log_label=f"Baseline:{self.rm_label}",
            )
        else:
            scores_b = score_responses_with_rm(
                answers_b, prompts_messages, model, tokenizer,
                batch_size=ctx.args.batch_size, device=ctx.args.device,
                checkpoint_num=f"{ctx.checkpoint_num}:pair_b",
            )
        battles = []
        for sa, sb in zip(scores_a, scores_b):
            if sa > sb:
                battles.append([1.0])
            elif sa == sb:
                battles.append([0.5])
            else:
                battles.append([0.0])
        return battles, RMBattleDetails(policy_scores=scores_a, baseline_scores=scores_b)


# ---------------------------------------------------------------------------
# Generative LLM judge (Arena-Hard-Auto v2.0 compatible)
#
# One judge, pluggable generation backend. ``LLMJudge`` owns everything that is
# backend-independent (prompt construction, the two position-swapped games, label
# parsing, weighted battle scoring). A backend only turns chat conversations +
# shared ``JudgeGenParams`` into completion strings, so switching between a remote
# API and a local vLLM model changes only the backend — not the protocol, the
# metrics, or how generation params are set.
# ---------------------------------------------------------------------------

def _parse_arena_label(text: str, patterns: List[str]) -> Optional[str]:
    """Find the last [[...]] label in ``text``. Matches Arena-Hard's get_score."""
    up = text.upper()
    for pat in patterns:
        matches = re.findall(pat, up)
        matches = [m for m in matches if m]
        if matches:
            return matches[-1].strip("\n")
    return None


@dataclass
class JudgeGenParams:
    """Generation controls applied identically across judge backends.

    Fields are required (no defaults) on purpose: the canonical default values
    live in ``ScriptArguments`` (the ``--llm_judge_*`` flags), and callers build
    this from there, so a default can't drift between config and code.

    ``enable_thinking`` is honored in full by the local vLLM backend (chat
    template + ``"My final verdict "`` prefill); the OpenRouter backend forwards
    it via OpenRouter's native ``reasoning`` toggle when disabling thinking.
    """
    temperature: float
    top_p: float
    max_tokens: int
    enable_thinking: bool


class OpenRouterBackend:
    """Generate judge completions via the OpenRouter chat-completions API."""
    phase = "online"

    def __init__(
        self,
        model_name: str = "openai/gpt-4.1",
        *,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        max_parallel: int = 8,
        max_retries: int = 5,
        timeout: int = 300,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    ):
        self.model_name = model_name
        self.label = model_name.replace("/", "_")
        self._api_key = api_key
        self.api_key_env = api_key_env
        self.max_parallel = max_parallel
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = base_url

    def _resolve_key(self) -> str:
        key = self._api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"OpenRouter judge needs an API key (--openrouter_api_key or env {self.api_key_env})"
            )
        return key

    def _one(self, messages: List[dict], params: "JudgeGenParams", api_key: str) -> "JudgeGeneration":
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        # Only deviate from the model's default reasoning behavior when thinking
        # is explicitly disabled (OpenRouter-native toggle; no-op for models that
        # don't support it). Keeps the default path byte-for-byte as before.
        if not params.enable_thinking:
            body["reasoning"] = {"enabled": False}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                choice = r.json()["choices"][0]
                return JudgeGeneration(
                    text=choice["message"]["content"],
                    truncated=choice.get("finish_reason") == "length",
                )
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        print(f"[OpenRouterBackend] giving up after {self.max_retries} retries: {last_err}")
        return JudgeGeneration(text="", truncated=False)

    def generate(self, conversations: List[List[dict]], params: "JudgeGenParams") -> List["JudgeGeneration"]:
        import concurrent.futures as cf

        api_key = self._resolve_key()
        gens: List[Optional[JudgeGeneration]] = [None] * len(conversations)
        with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futs = {
                pool.submit(self._one, conv, params, api_key): i
                for i, conv in enumerate(conversations)
            }
            for fut in cf.as_completed(futs):
                gens[futs[fut]] = fut.result()
        return gens


class VLLMBackend:
    """Generate judge completions from a local open-weight model via vLLM.

    Deferred: the model loads lazily on the first ``generate`` (after the policy
    vLLM is torn down so the GPU is free) and is reused across checkpoints, then
    released by ``teardown``. Applies the chat template locally so it can honor
    ``enable_thinking`` and prefill ``"My final verdict "`` when thinking is off.
    """
    phase = "deferred"
    PREFILL = "My final verdict "

    def __init__(
        self,
        model_name: str,
        *,
        max_model_len: int,
        gpu_memory_utilization: float,
    ):
        self.model_name = model_name
        self.label = f"vllm_{model_name.replace('/', '_')}"
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._tokenizer = None

    def _ensure(self) -> None:
        if self._llm is not None:
            return
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        print(
            f"[VLLMBackend] loading judge {self.model_name} "
            f"(max_model_len={self.max_model_len}, "
            f"gpu_memory_utilization={self.gpu_memory_utilization})"
        )
        self._llm = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            # vLLM's sampler warmup builds a [max_num_seqs, vocab] logits tensor,
            # and large-vocab judges (gemma: 262k) with logit soft-capping need a
            # second copy of it -- ~256 MiB per 256 seqs. With a big judge model
            # filling the GPU, that transient can be what tips warmup into OOM.
            # A large judge is KV-bound to only a few concurrent full-length
            # sequences regardless, so capping concurrency costs no real
            # throughput while keeping the warmup transient small.
            max_num_seqs=64,
            trust_remote_code=True,
            language_model_only=True,
        )

    def teardown(self) -> None:
        from .generation import teardown_vllm
        teardown_vllm(self._llm)
        self._llm = None
        self._tokenizer = None

    def _render(self, messages: List[dict], enable_thinking: bool) -> str:
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Chat template doesn't accept the enable_thinking kwarg.
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        if not enable_thinking:
            text += self.PREFILL
        return text

    def generate(self, conversations: List[List[dict]], params: "JudgeGenParams") -> List["JudgeGeneration"]:
        from vllm import SamplingParams
        from data_utils import get_generation_stop_token_ids

        self._ensure()
        prompts = [self._render(c, params.enable_thinking) for c in conversations]
        sampling = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            n=1,
            stop_token_ids=get_generation_stop_token_ids(self._tokenizer),
        )
        outputs = self._llm.generate(prompts, sampling)
        gens = []
        for o in outputs:
            comp = o.outputs[0]
            text = comp.text
            # Restore the prefill so saved text and the parser see the full verdict.
            if not params.enable_thinking:
                text = self.PREFILL + text
            gens.append(JudgeGeneration(text=text, truncated=comp.finish_reason == "length"))
        return gens


class LLMJudge:
    """Arena-Hard-Auto-compatible pairwise judge with a pluggable generation backend.

    Backend-independent: builds the system+user judge prompt, runs the two
    position-swapped games per prompt, parses the 5-point verdict, and maps the
    labels to weighted battles via ``battles_from_game_labels`` (matching
    ``show_result.py``). The ``backend`` (``OpenRouterBackend`` or
    ``VLLMBackend``) only turns chat conversations + ``JudgeGenParams`` into
    completion strings, so switching API <-> local vLLM leaves the protocol,
    metrics, and generation-param handling unchanged. ``phase`` is inherited from
    the backend so deferred (GPU-loading) backends run after the policy vLLM.
    """
    kind = "llm"

    def __init__(
        self,
        backend,
        *,
        gen_params: JudgeGenParams,
        system_prompt: str = ARENA_HARD_SYSTEM_PROMPT,
        prompt_template: str = ARENA_HARD_PROMPT_TEMPLATE,
        regex_patterns: Optional[List[str]] = None,
        weight: int = 3,
    ):
        self.backend = backend
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.regex_patterns = regex_patterns or ARENA_HARD_REGEX_PATTERNS
        self.weight = weight
        self.gen_params = gen_params
        self.phase = getattr(backend, "phase", "online")
        self.name = f"llm_{backend.label}"

    def teardown(self) -> None:
        if hasattr(self.backend, "teardown"):
            self.backend.teardown()

    def _conversation(self, question: str, answer_a: str, answer_b: str) -> List[dict]:
        user_prompt = self.prompt_template.format(
            QUESTION=question, ANSWER_A=answer_a, ANSWER_B=answer_b,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def score_pairs(
        self,
        prompts_messages: List[list],
        answers_a: List[str],
        answers_b: List[str],
        ctx,
    ) -> Tuple[List[List[float]], LLMBattleDetails]:
        n = len(answers_a)
        questions = [render_judge_question(m) for m in prompts_messages]

        # Two position-swapped games per prompt, flattened [p0/g0, p0/g1, ...].
        # game0 = A:baseline/B:policy, game1 = A:policy/B:baseline (upstream swap).
        conversations: List[List[dict]] = []
        for i in range(n):
            conversations.append(self._conversation(questions[i], answers_b[i], answers_a[i]))
            conversations.append(self._conversation(questions[i], answers_a[i], answers_b[i]))

        gens = self.backend.generate(conversations, self.gen_params)
        game0_gens = [gens[2 * i] for i in range(n)]
        game1_gens = [gens[2 * i + 1] for i in range(n)]
        game0_texts = [g.text for g in game0_gens]
        game1_texts = [g.text for g in game1_gens]
        game0_labels = [_parse_arena_label(t, self.regex_patterns) for t in game0_texts]
        game1_labels = [_parse_arena_label(t, self.regex_patterns) for t in game1_texts]

        battles, dropped = battles_from_game_labels(
            game0_labels, game1_labels, weight=self.weight,
        )

        # Attribute each unparsable game (label is None) to a single cause, in
        # priority order, so the reason for any drops is visible in wandb:
        #   empty output -> generation failure (API HTTP failure / empty gen)
        #   truncated    -> truncation failure (ran out of tokens before verdict)
        #   otherwise    -> parse failure (had output, no recognizable verdict)
        all_gens = game0_gens + game1_gens
        all_labels = game0_labels + game1_labels
        n_generation_failures = n_truncation_failures = n_parse_failures = 0
        for g, lbl in zip(all_gens, all_labels):
            if lbl is not None:
                continue
            if g.text == "":
                n_generation_failures += 1
            elif g.truncated:
                n_truncation_failures += 1
            else:
                n_parse_failures += 1
        if dropped:
            print(
                f"[LLMJudge:{self.name}] dropped {dropped}/{n} prompts "
                f"({n_generation_failures} generation, {n_truncation_failures} truncation, "
                f"{n_parse_failures} parse failures over {2 * n} games)"
            )
        return battles, LLMBattleDetails(
            game0_labels=game0_labels, game1_labels=game1_labels,
            game0_texts=game0_texts, game1_texts=game1_texts,
            n_generation_failures=n_generation_failures,
            n_truncation_failures=n_truncation_failures,
            n_parse_failures=n_parse_failures,
            n_dropped_prompts=dropped,
        )


