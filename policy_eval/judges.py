"""Pairwise judges: compare two answers to a prompt and emit battle outcomes.

A judge decides which of (answer_a, answer_b) is better for a given prompt.
Two implementations:

    - ``RMJudge``: score each answer with a reward model; winner has the higher
      score. One battle per prompt.
    - ``LLMAPIJudge``: ask an LLM (e.g. gpt-4.1 via OpenRouter) using the
      Arena-Hard-Auto v2.0 prompt template. Runs 2 games per prompt with
      position swap; each game's label is mapped (with weighting) to one or
      more {0.0, 0.5, 1.0} battles, matching ``show_result.py`` exactly so
      that scores produced here match the official leaderboard when the same
      judge/baseline are used.

Both return ``(battles_per_prompt, extras)`` where ``battles_per_prompt`` is a
list of lists: ``battles_per_prompt[i]`` = battle scores for prompt ``i`` from
answer_a's perspective (1.0 = A wins, 0.0 = A loses, 0.5 = tie). The
``PairwiseEvaluator`` aggregates these into Arena scores and bootstrap CIs.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from .rewards import score_responses_with_rm


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

    def backend_id(self, ctx) -> str:
        """Returns a stable string identifying this judge's backend model
        (used in cache file names so swapping RMs invalidates caches)."""
        rms = ctx.loaded_rms
        entry = rms.get(self.rm_label) if rms else None
        if entry is None:
            return f"missing:{self.rm_label}"
        model, _ = entry
        return model.config._name_or_path

    def score_pairs(
        self,
        prompts_messages: List[list],
        answers_a: List[str],
        answers_b: List[str],
        ctx,
    ) -> Tuple[List[List[float]], Dict[str, np.ndarray]]:
        rms = ctx.loaded_rms
        entry = rms.get(self.rm_label) if rms else None
        if entry is None:
            raise ValueError(f"RMJudge needs RM '{self.rm_label}' loaded")
        model, tokenizer = entry

        scores_a = score_responses_with_rm(
            answers_a, prompts_messages, model, tokenizer,
            batch_size=ctx.args.batch_size, device=ctx.args.device,
            checkpoint_num=ctx.checkpoint_num,
        )
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
        return battles, {"scores_a": scores_a, "scores_b": scores_b}


# ---------------------------------------------------------------------------
# LLM API judge (Arena-Hard-Auto v2.0 compatible)
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


class LLMAPIJudge:
    """Arena-Hard-Auto-compatible pairwise LLM judge (OpenRouter API).

    For each prompt runs two games (position-swapped) and maps each 5-category
    label through ``label_to_score`` with ``weight`` (default 3). Score output
    matches ``show_result.py`` in the upstream repo: the returned per-prompt
    battle list is ``label_to_score[game1] + [1 - s for s in label_to_score[game0]]``.
    """
    kind = "llm_api"

    def __init__(
        self,
        model_name: str = "openai/gpt-4.1",
        system_prompt: str = ARENA_HARD_SYSTEM_PROMPT,
        prompt_template: str = ARENA_HARD_PROMPT_TEMPLATE,
        regex_patterns: Optional[List[str]] = None,
        weight: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 16000,
        max_parallel: int = 8,
        api_key_env: str = "OPENROUTER_API_KEY",
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.regex_patterns = regex_patterns or ARENA_HARD_REGEX_PATTERNS
        self.weight = weight
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.api_key_env = api_key_env
        self.name = f"llm_{model_name.replace('/', '_')}"

    def backend_id(self, ctx) -> str:
        return f"{self.model_name}:w{self.weight}"

    def _one_game(self, question: str, answer_a: str, answer_b: str, ctx) -> Optional[str]:
        """Single call; returns the parsed label or None on failure."""
        user_prompt = self.prompt_template.format(
            QUESTION=question, ANSWER_A=answer_a, ANSWER_B=answer_b
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        api_key = getattr(ctx.args, "openrouter_api_key", None) or os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"LLMAPIJudge needs an API key (env {self.api_key_env} or --openrouter_api_key)"
            )

        # Retry with exponential backoff on transient errors.
        last_err = None
        for attempt in range(5):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=300,
                )
                r.raise_for_status()
                out = r.json()
                text = out["choices"][0]["message"]["content"]
                return _parse_arena_label(text, self.regex_patterns)
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        print(f"[LLMAPIJudge] giving up after retries: {last_err}")
        return None

    def score_pairs(
        self,
        prompts_messages: List[list],
        answers_a: List[str],
        answers_b: List[str],
        ctx,
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        import concurrent.futures as cf
        label_score = _label_to_score(self.weight)

        # Each prompt generates 2 games (A=a,B=b) and (A=b,B=a).
        n = len(answers_a)
        game0_labels: List[Optional[str]] = [None] * n
        game1_labels: List[Optional[str]] = [None] * n

        def _q(msgs):
            return msgs[-1]["content"] if msgs else ""

        def _run(i, game_idx, aa, bb):
            return i, game_idx, self._one_game(_q(prompts_messages[i]), aa, bb, ctx)

        with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = []
            for i in range(n):
                # game 0: A = baseline (answers_b), B = policy (answers_a)  ← match upstream swap
                futures.append(pool.submit(_run, i, 0, answers_b[i], answers_a[i]))
                # game 1: A = policy (answers_a), B = baseline (answers_b)
                futures.append(pool.submit(_run, i, 1, answers_a[i], answers_b[i]))
            for fut in cf.as_completed(futures):
                i, game_idx, label = fut.result()
                if game_idx == 0:
                    game0_labels[i] = label
                else:
                    game1_labels[i] = label

        # Apply show_result.py formula:
        #   scores = label_to_score[game1] + [1 - s for s in label_to_score[game0]]
        # game1 is A=policy,B=baseline so its label_to_score is already from
        # policy's perspective. game0 is A=baseline,B=policy so we flip (1-s).
        battles_per_prompt: List[List[float]] = []
        dropped = 0
        for lbl0, lbl1 in zip(game0_labels, game1_labels):
            if lbl0 is None or lbl1 is None or lbl0 not in label_score or lbl1 not in label_score:
                dropped += 1
                battles_per_prompt.append([])  # empty → PairwiseEvaluator will skip
                continue
            ms1 = label_score[lbl1]
            ms0 = [1.0 - s for s in label_score[lbl0]]
            battles_per_prompt.append(ms1 + ms0)

        if dropped:
            print(
                f"[LLMAPIJudge] dropped {dropped}/{n} prompts due to parse/API failure"
            )
        return battles_per_prompt, {
            "game0_labels": game0_labels, "game1_labels": game1_labels,
        }
