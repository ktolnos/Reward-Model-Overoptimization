import pytest
from functools import lru_cache

from transformers import AutoTokenizer

from data_utils import (
    format_and_validate_preference_sample,
    _apply_chat_template_no_thinking,
)


@lru_cache(maxsize=None)
def _load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _tokenizer_or_skip(model_id):
    try:
        return _load_tokenizer(model_id)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not load tokenizer {model_id}: {exc}")


class TestQwenFormatting:
    def test_content_insensitive(self):
        tokenizer = _tokenizer_or_skip("Qwen/Qwen3-0.6B")

        prompt_messages = [{"role": "user", "content": "Explain KL divergence briefly."}]
        completion = "<think>\nreasoning\n</think>\n\nKL divergence measures divergence between distributions."
        full_conv = prompt_messages + [{"role": "assistant", "content": completion}]

        prompt_text, chosen_text, _ = format_and_validate_preference_sample(
            full_conv,
            tokenizer,
            length_config="default",
            skip_validation=True,
            context="test-qwen",
        )

        # New formatter must always preserve prompt as exact prefix.
        assert chosen_text.startswith(prompt_text)
        # Qwen-specific expected behavior for non-empty think content:
        # keep template-provided empty scaffold and preserve emitted think block.
        assert "<think>\n\n</think>\n\n<think>\nreasoning\n</think>\n\n" in chosen_text

        # Native chat-template full formatting is content-sensitive here.
        native_full = _apply_chat_template_no_thinking(
            tokenizer, full_conv, add_generation_prompt=False
        )
        assert not native_full.startswith(prompt_text)

    def test_matches_old_format_when_thinking_is_empty(self):
        tokenizer = _tokenizer_or_skip("Qwen/Qwen3-0.6B")

        prompt_messages = [
            {"role": "user", "content": "Give a one-line definition of entropy."},
        ]
        completion = "Entropy measures uncertainty in a probability distribution."
        full_conv = prompt_messages + [{"role": "assistant", "content": completion}]

        prompt_text, chosen_text, _ = format_and_validate_preference_sample(
            full_conv,
            tokenizer,
            length_config="default",
            skip_validation=True,
            context="test-qwen-empty-thinking",
        )

        # Old implementation behavior: apply full chat template directly.
        old_prompt = _apply_chat_template_no_thinking(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        old_full = _apply_chat_template_no_thinking(
            tokenizer, full_conv, add_generation_prompt=False
        )

        assert prompt_text == old_prompt
        assert chosen_text == old_full
        assert chosen_text.startswith(prompt_text)


LLAMA_SMOLLM_MODEL_IDS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

COMPLETIONS = [
    "A concise answer.",
    "<think>\nintermediate\n</think>\n\nA concise answer.",
]


@pytest.mark.parametrize("model_id", LLAMA_SMOLLM_MODEL_IDS)
@pytest.mark.parametrize("completion", COMPLETIONS)
def test_llama_and_smollm_match_native_chat_template(model_id, completion):
    tokenizer = _tokenizer_or_skip(model_id)

    prompt_messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    full_conv = prompt_messages + [{"role": "assistant", "content": completion}]

    prompt_text, chosen_text, _ = format_and_validate_preference_sample(
        full_conv,
        tokenizer,
        length_config="default",
        skip_validation=True,
        context=f"test-{model_id}",
    )

    native_prompt = _apply_chat_template_no_thinking(
        tokenizer, prompt_messages, add_generation_prompt=True
    )
    native_full = _apply_chat_template_no_thinking(
        tokenizer, full_conv, add_generation_prompt=False
    )

    assert prompt_text == native_prompt
    assert chosen_text == native_full
    assert chosen_text.startswith(prompt_text)


def test_pythia_without_chat_template_still_fails():
    tokenizer = _tokenizer_or_skip("EleutherAI/pythia-160m-deduped")

    full_conv = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    with pytest.raises(Exception, match="(?i)chat"):
        format_and_validate_preference_sample(
            full_conv,
            tokenizer,
            length_config="default",
            skip_validation=True,
            context="test-pythia",
        )
