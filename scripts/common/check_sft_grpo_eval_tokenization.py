#!/usr/bin/env python3
"""Check SFT/GRPO/eval prompt formatting and tokenization consistency.

This script validates the concrete path used in this repo:
- Prompt text creation via data_utils.format_and_validate_preference_sample
- SFT tokenization via data_utils.tokenize_for_sft
- GRPOTrainer pre-tokenization prompt handling (TRL maybe_apply_chat_template)
- Eval prompt generation path (same formatted prompt string)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import (  # noqa: E402
    format_and_validate_preference_sample,
    get_generation_stop_token_ids,
    setup_tokenizer,
    tokenize_for_sft,
)

TRL_IMPORT_ERROR: str | None = None
try:
    from trl.data_utils import maybe_apply_chat_template  # noqa: E402
except Exception as exc:  # pragma: no cover - diagnostic fallback only
    maybe_apply_chat_template = None
    TRL_IMPORT_ERROR = repr(exc)


def _ids(text: str, tokenizer):
    return tokenize_for_sft(text, tokenizer)["input_ids"][0].tolist()


def _print_check(label: str, ok: bool):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        help="Policy tokenizer model id/path.",
    )
    parser.add_argument(
        "--user-prompt",
        default="Explain what a prime number is in one sentence.",
        help="User message used for the synthetic test sample.",
    )
    parser.add_argument(
        "--assistant-response",
        default="A prime number is a number greater than 1 that has exactly two positive divisors: 1 and itself.",
        help="Assistant response used for the synthetic test sample.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=4096,
        help="Validation max prompt length.",
    )
    parser.add_argument(
        "--max-conversation-length",
        type=int,
        default=8192,
        help="Validation max full-conversation length.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print prompt/full text repr and boundary token window.",
    )
    parser.add_argument(
        "--boundary-window",
        type=int,
        default=16,
        help="Token window size around prompt/completion boundary when --show-details is enabled.",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    setup_tokenizer(tokenizer)

    chosen_messages = [
        {"role": "user", "content": args.user_prompt},
        {"role": "assistant", "content": args.assistant_response},
    ]

    prompt_sft, full_sft, _ = format_and_validate_preference_sample(
        chosen_messages,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_conversation_length=args.max_conversation_length,
        context="SFT-check",
    )
    prompt_grpo, full_grpo, _ = format_and_validate_preference_sample(
        chosen_messages,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_conversation_length=args.max_conversation_length,
        context="GRPO-check",
    )
    prompt_eval, full_eval, _ = format_and_validate_preference_sample(
        chosen_messages,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_conversation_length=args.max_conversation_length,
        context="Eval-check",
    )

    prompt_ids = _ids(prompt_sft, tokenizer)
    full_ids = _ids(full_sft, tokenizer)
    completion_ids = full_ids[len(prompt_ids) :]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)

    checks: list[tuple[str, bool]] = [
        ("Prompt strings are identical (SFT/GRPO/eval)", prompt_sft == prompt_grpo == prompt_eval),
        ("Full conversation strings are identical (SFT/GRPO/eval)", full_sft == full_grpo == full_eval),
        (
            "Prompt contains <think></think> tags",
            "<think>" in prompt_sft and "</think>" in prompt_sft,
        ),
        (
            "Tokenized prompt is exact prefix of tokenized full conversation",
            full_ids[: len(prompt_ids)] == prompt_ids,
        ),
        (
            "Completion starts with configured assistant response text",
            completion_text.startswith(args.assistant_response),
        ),
    ]

    if maybe_apply_chat_template is not None:
        trainer_prompt = maybe_apply_chat_template({"prompt": prompt_grpo}, tokenizer)["prompt"]
        conversational_prompt = maybe_apply_chat_template({"prompt": chosen_messages[:-1]}, tokenizer)["prompt"]
        checks.extend(
            [
                (
                    "GRPOTrainer keeps preformatted string prompt unchanged",
                    trainer_prompt == prompt_grpo,
                ),
                (
                    "TRL conversational prompt path differs (no <think></think>)",
                    ("<think>" not in conversational_prompt and "</think>" not in conversational_prompt),
                ),
            ]
        )
    else:
        print(f"Warning: could not import trl.data_utils.maybe_apply_chat_template: {TRL_IMPORT_ERROR}")

    stop_ids = get_generation_stop_token_ids(tokenizer)
    stop_tokens = tokenizer.convert_ids_to_tokens(stop_ids)

    print("\nChecks:")
    any_fail = False
    for label, ok in checks:
        _print_check(label, ok)
        any_fail = any_fail or (not ok)

    print("\nGeneration stop ids/tokens:")
    print(f"ids={stop_ids}")
    print(f"tokens={stop_tokens}")

    if args.show_details:
        print("\nPrompt repr:")
        print(repr(prompt_sft))
        print("\nFull repr:")
        print(repr(full_sft))
        start = max(0, len(prompt_ids) - args.boundary_window)
        end = min(len(full_ids), len(prompt_ids) + args.boundary_window)
        window_ids = full_ids[start:end]
        print("\nBoundary token ids:")
        print(window_ids)
        print("Boundary token strings:")
        print(tokenizer.convert_ids_to_tokens(window_ids))
        print("Boundary decoded text:")
        print(repr(tokenizer.decode(window_ids, skip_special_tokens=False)))

    if any_fail:
        raise SystemExit(1)
    print("\nAll consistency checks passed.")


if __name__ == "__main__":
    main()
