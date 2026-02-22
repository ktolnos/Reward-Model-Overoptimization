import unittest
from functools import lru_cache

from transformers import AutoTokenizer

from data_utils import format_and_validate_preference_sample


def _apply_chat_template_no_thinking(tokenizer, messages, *, add_generation_prompt=False):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)


@lru_cache(maxsize=None)
def _load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


class TestDataUtilsFormatting(unittest.TestCase):
    def _tokenizer_or_skip(self, model_id):
        try:
            return _load_tokenizer(model_id)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"Could not load tokenizer {model_id}: {exc}")

    def test_qwen_formatting_is_content_insensitive(self):
        tokenizer = self._tokenizer_or_skip("Qwen/Qwen3-0.6B")

        prompt_messages = [{"role": "user", "content": "Explain KL divergence briefly."}]
        completion = "<think>\nreasoning\n</think>\n\nKL divergence measures divergence between distributions."
        full_conv = prompt_messages + [{"role": "assistant", "content": completion}]

        prompt_text, chosen_text, _ = format_and_validate_preference_sample(
            full_conv,
            tokenizer,
            max_prompt_length=None,
            max_conversation_length=None,
            context="test-qwen",
        )

        # New formatter must always preserve prompt as exact prefix.
        self.assertTrue(chosen_text.startswith(prompt_text))
        # Qwen-specific expected behavior for non-empty think content:
        # keep template-provided empty scaffold and preserve emitted think block.
        self.assertIn("<think>\n\n</think>\n\n<think>\nreasoning\n</think>\n\n", chosen_text)

        # Native chat-template full formatting is content-sensitive here.
        native_full = _apply_chat_template_no_thinking(
            tokenizer, full_conv, add_generation_prompt=False
        )
        self.assertFalse(native_full.startswith(prompt_text))

    def test_qwen_matches_old_format_when_thinking_is_empty(self):
        tokenizer = self._tokenizer_or_skip("Qwen/Qwen3-0.6B")

        prompt_messages = [
            {"role": "user", "content": "Give a one-line definition of entropy."},
        ]
        completion = "Entropy measures uncertainty in a probability distribution."
        full_conv = prompt_messages + [{"role": "assistant", "content": completion}]

        # Current implementation (content-insensitive composer)
        prompt_text, chosen_text, _ = format_and_validate_preference_sample(
            full_conv,
            tokenizer,
            max_prompt_length=None,
            max_conversation_length=None,
            context="test-qwen-empty-thinking",
        )

        # Old implementation behavior: apply full chat template directly.
        old_prompt = _apply_chat_template_no_thinking(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        old_full = _apply_chat_template_no_thinking(
            tokenizer, full_conv, add_generation_prompt=False
        )

        self.assertEqual(prompt_text, old_prompt)
        self.assertEqual(chosen_text, old_full)
        self.assertTrue(chosen_text.startswith(prompt_text))

    def test_llama_and_smollm_match_native_chat_template(self):
        # These should remain aligned with native chat-template output.
        model_ids = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ]
        completions = [
            "A concise answer.",
            "<think>\nintermediate\n</think>\n\nA concise answer.",
        ]
        prompt_messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]

        for model_id in model_ids:
            tokenizer = self._tokenizer_or_skip(model_id)
            for completion in completions:
                with self.subTest(model=model_id, completion=completion):
                    full_conv = prompt_messages + [{"role": "assistant", "content": completion}]
                    prompt_text, chosen_text, _ = format_and_validate_preference_sample(
                        full_conv,
                        tokenizer,
                        max_prompt_length=None,
                        max_conversation_length=None,
                        context=f"test-{model_id}",
                    )

                    native_prompt = _apply_chat_template_no_thinking(
                        tokenizer, prompt_messages, add_generation_prompt=True
                    )
                    native_full = _apply_chat_template_no_thinking(
                        tokenizer, full_conv, add_generation_prompt=False
                    )

                    self.assertEqual(prompt_text, native_prompt)
                    self.assertEqual(chosen_text, native_full)
                    self.assertTrue(chosen_text.startswith(prompt_text))

    def test_pythia_without_chat_template_still_fails(self):
        tokenizer = self._tokenizer_or_skip("EleutherAI/pythia-160m-deduped")

        full_conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        with self.assertRaises(Exception) as ctx:
            format_and_validate_preference_sample(
                full_conv,
                tokenizer,
                max_prompt_length=None,
                max_conversation_length=None,
                context="test-pythia",
            )
        self.assertIn("chat", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
