"""Tests for Pythia/Open-Assistant and AlpacaFarm chat template support,
auto-detection, tokenization behavior, and reward extraction.

Uses the real ``tlc4418/pythia_70m_sft`` tokenizer (tiny, ~200 KB download)
so that tests exercise the actual special-token vocabulary rather than mocks.
"""

import pytest
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_utils import (
    _has_pythia_oa_tokens,
    _looks_like_pythia_model,
    _PYTHIA_OA_V2_CHAT_TEMPLATE,
    _PYTHIA_EXPECTED_SPECIAL_TOKENS,
    _PYTHIA_OA_TOKENS_TO_ADD,
    _ALPACAFARM_GOLD_CHAT_TEMPLATE,
    setup_pythia_chat_template,
    setup_alpacafarm_gold_chat_template,
    setup_tokenizer,
    load_policy_and_tokenizer,
    get_length_config,
    DATASET_LENGTH_CONFIGS,
    get_generation_stop_token_ids,
    completion_has_stop_token,
    format_and_validate_preference_sample,
    _format_prompt,
    _format_conversation,
    tokenize_for_sft,
    tokenize_for_rm,
    strip_bos_if_present,
)
from reward_utils import (
    _is_alpacafarm_rm,
    extract_reward_tensors_from_model_output,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PYTHIA_SFT_MODEL = "tlc4418/pythia_70m_sft"
PYTHIA_BASE_MODEL = "EleutherAI/pythia-70m"


@pytest.fixture(scope="module")
def pythia_sft_tokenizer():
    """Pythia 70m SFT tokenizer with auto-detection applied."""
    tok = AutoTokenizer.from_pretrained(PYTHIA_SFT_MODEL)
    setup_tokenizer(tok)
    return tok


@pytest.fixture(scope="module")
def pythia_base_tokenizer():
    """Base Pythia 70m tokenizer (no SFT special tokens)."""
    tok = AutoTokenizer.from_pretrained(PYTHIA_BASE_MODEL)
    return tok


@pytest.fixture(scope="module")
def alpacafarm_tokenizer():
    """Tokenizer with AlpacaFarm gold chat template applied."""
    tok = AutoTokenizer.from_pretrained(PYTHIA_SFT_MODEL)
    setup_tokenizer(tok)
    setup_alpacafarm_gold_chat_template(tok)
    return tok


# ---------------------------------------------------------------------------
# 1. Pythia auto-detection
# ---------------------------------------------------------------------------


class TestPythiaAutoDetection:
    def test_sft_tokenizer_has_oa_tokens(self, pythia_sft_tokenizer):
        assert _has_pythia_oa_tokens(pythia_sft_tokenizer)

    def test_base_tokenizer_lacks_oa_tokens(self, pythia_base_tokenizer):
        assert not _has_pythia_oa_tokens(pythia_base_tokenizer)

    def test_looks_like_pythia_sft(self):
        tok = AutoTokenizer.from_pretrained(PYTHIA_SFT_MODEL)
        assert _looks_like_pythia_model(tok)

    def test_looks_like_pythia_base(self):
        tok = AutoTokenizer.from_pretrained(PYTHIA_BASE_MODEL)
        assert _looks_like_pythia_model(tok)

    def test_setup_tokenizer_applies_template_for_sft(self):
        tok = AutoTokenizer.from_pretrained(PYTHIA_SFT_MODEL)
        assert tok.chat_template is None
        setup_tokenizer(tok)
        assert tok.chat_template == _PYTHIA_OA_V2_CHAT_TEMPLATE

    def test_setup_tokenizer_adds_tokens_for_base_pythia(self):
        tok = AutoTokenizer.from_pretrained(PYTHIA_BASE_MODEL)
        base_vocab_size = len(tok)
        assert not _has_pythia_oa_tokens(tok)
        setup_tokenizer(tok)
        # Tokens were added and template was set.
        assert _has_pythia_oa_tokens(tok)
        assert tok.chat_template == _PYTHIA_OA_V2_CHAT_TEMPLATE
        assert len(tok) == base_vocab_size + len(_PYTHIA_OA_TOKENS_TO_ADD)

    def test_setup_tokenizer_base_pythia_token_ids_match_sft(self):
        """Token IDs added to base Pythia must match those in the SFT'd model."""
        base_tok = AutoTokenizer.from_pretrained(PYTHIA_BASE_MODEL)
        setup_tokenizer(base_tok)
        sft_tok = AutoTokenizer.from_pretrained(PYTHIA_SFT_MODEL)
        setup_tokenizer(sft_tok)
        for special in _PYTHIA_OA_TOKENS_TO_ADD:
            base_id = base_tok.convert_tokens_to_ids(special)
            sft_id = sft_tok.convert_tokens_to_ids(special)
            assert base_id == sft_id, (
                f"Token {special!r}: base={base_id}, sft={sft_id}"
            )

    def test_setup_tokenizer_does_not_overwrite_existing_template(self):
        tok = AutoTokenizer.from_pretrained(PYTHIA_SFT_MODEL)
        tok.chat_template = "custom_already_set"
        setup_tokenizer(tok)
        assert tok.chat_template == "custom_already_set"

    def test_setup_pythia_chat_template_rejects_base_tokenizer(self, pythia_base_tokenizer):
        with pytest.raises(ValueError, match="missing expected special token"):
            setup_pythia_chat_template(pythia_base_tokenizer)

    def test_special_tokens_in_vocab(self, pythia_sft_tokenizer):
        vocab = pythia_sft_tokenizer.get_vocab()
        for tok in _PYTHIA_EXPECTED_SPECIAL_TOKENS:
            assert tok in vocab, f"{tok} not in vocab"


# ---------------------------------------------------------------------------
# 2. Pythia OA v2 chat template — text output
# ---------------------------------------------------------------------------


class TestPythiaChatTemplateText:
    def test_single_turn_prompt(self, pythia_sft_tokenizer):
        msgs = [{"role": "user", "content": "Hello there"}]
        text = pythia_sft_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        assert text == "<|prompter|>Hello there<|endoftext|><|assistant|>"

    def test_single_turn_full(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
        ]
        text = pythia_sft_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        assert text == "<|prompter|>Hello there<|endoftext|><|assistant|>Hi!<|endoftext|>"

    def test_multi_turn_prompt(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        text = pythia_sft_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        expected = (
            "<|prompter|>Hello<|endoftext|>"
            "<|assistant|>Hi there!<|endoftext|>"
            "<|prompter|>How are you?<|endoftext|>"
            "<|assistant|>"
        )
        assert text == expected

    def test_multi_turn_full(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine."},
        ]
        text = pythia_sft_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        expected = (
            "<|prompter|>Hello<|endoftext|>"
            "<|assistant|>Hi there!<|endoftext|>"
            "<|prompter|>How are you?<|endoftext|>"
            "<|assistant|>I am fine.<|endoftext|>"
        )
        assert text == expected


# ---------------------------------------------------------------------------
# 3. Pythia tokenization — actual token IDs
# ---------------------------------------------------------------------------


class TestPythiaTokenization:
    def test_special_tokens_encode_to_single_ids(self, pythia_sft_tokenizer):
        """Each OA special token must encode to exactly one token ID."""
        for tok_str in ("<|prompter|>", "<|assistant|>", "<|endoftext|>"):
            ids = pythia_sft_tokenizer.encode(tok_str, add_special_tokens=False)
            assert len(ids) == 1, f"{tok_str} encoded to {len(ids)} tokens: {ids}"

    def test_prompt_is_token_prefix_of_full(self, pythia_sft_tokenizer):
        """Tokenized prompt must be a prefix of the tokenized full conversation."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        prompt_text = _format_prompt(msgs, pythia_sft_tokenizer)
        full_text = _format_conversation(msgs, pythia_sft_tokenizer)

        prompt_ids = pythia_sft_tokenizer.encode(prompt_text, add_special_tokens=True)
        full_ids = pythia_sft_tokenizer.encode(full_text, add_special_tokens=True)

        assert full_ids[: len(prompt_ids)] == prompt_ids

    def test_prompt_is_token_prefix_multi_turn(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine."},
        ]
        prompt_text = _format_prompt(msgs, pythia_sft_tokenizer)
        full_text = _format_conversation(msgs, pythia_sft_tokenizer)

        prompt_ids = pythia_sft_tokenizer.encode(prompt_text, add_special_tokens=True)
        full_ids = pythia_sft_tokenizer.encode(full_text, add_special_tokens=True)

        assert full_ids[: len(prompt_ids)] == prompt_ids

    def test_full_conversation_ends_with_eos(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        full_text = _format_conversation(msgs, pythia_sft_tokenizer)
        full_ids = pythia_sft_tokenizer.encode(full_text, add_special_tokens=True)
        assert full_ids[-1] == pythia_sft_tokenizer.eos_token_id

    def test_prompt_contains_prompter_and_assistant_tokens(self, pythia_sft_tokenizer):
        """The prompt encoding must contain the <|prompter|> and <|assistant|> IDs."""
        vocab = pythia_sft_tokenizer.get_vocab()
        prompter_id = vocab["<|prompter|>"]
        assistant_id = vocab["<|assistant|>"]

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        prompt_text = _format_prompt(msgs, pythia_sft_tokenizer)
        prompt_ids = pythia_sft_tokenizer.encode(prompt_text, add_special_tokens=True)

        assert prompter_id in prompt_ids
        assert assistant_id in prompt_ids

    def test_tokenize_for_sft_single_sample(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        full_text = _format_conversation(msgs, pythia_sft_tokenizer)
        out = tokenize_for_sft(full_text, pythia_sft_tokenizer)
        assert out["input_ids"].ndim == 2
        assert out["input_ids"].shape[0] == 1  # single sample

        vocab = pythia_sft_tokenizer.get_vocab()
        ids = out["input_ids"][0].tolist()
        assert vocab["<|prompter|>"] in ids
        assert vocab["<|assistant|>"] in ids
        assert ids[-1] == pythia_sft_tokenizer.eos_token_id

    def test_tokenize_for_rm_batch(self, pythia_sft_tokenizer):
        """RM tokenization must left-pad a batch to equal length."""
        msgs_short = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        msgs_long = [
            {"role": "user", "content": "Tell me a long story about cats"},
            {"role": "assistant", "content": "Once upon a time there was a cat."},
        ]
        text_short = _format_conversation(msgs_short, pythia_sft_tokenizer)
        text_long = _format_conversation(msgs_long, pythia_sft_tokenizer)

        out = tokenize_for_rm([text_short, text_long], pythia_sft_tokenizer)
        assert out["input_ids"].shape[0] == 2
        # Both rows have the same length (padded)
        assert out["input_ids"].shape[1] == out["input_ids"].shape[1]
        # Left-padded: shorter sample starts with pad tokens
        pad_id = pythia_sft_tokenizer.pad_token_id
        short_ids = out["input_ids"][0].tolist()
        long_ids = out["input_ids"][1].tolist()
        short_unpadded_len = len(
            pythia_sft_tokenizer.encode(
                strip_bos_if_present(text_short, pythia_sft_tokenizer),
                add_special_tokens=True,
            )
        )
        long_unpadded_len = len(
            pythia_sft_tokenizer.encode(
                strip_bos_if_present(text_long, pythia_sft_tokenizer),
                add_special_tokens=True,
            )
        )
        if short_unpadded_len < long_unpadded_len:
            # The shorter sample must have padding on the left
            n_pad = out["input_ids"].shape[1] - short_unpadded_len
            assert all(t == pad_id for t in short_ids[:n_pad])

    def test_stop_token_ids_include_eos(self, pythia_sft_tokenizer):
        stop_ids = get_generation_stop_token_ids(pythia_sft_tokenizer)
        assert pythia_sft_tokenizer.eos_token_id in stop_ids

    def test_completion_has_stop_token_true(self, pythia_sft_tokenizer):
        eos = pythia_sft_tokenizer.eos_token_id
        assert completion_has_stop_token([10, 20, eos, 30], tokenizer=pythia_sft_tokenizer)

    def test_completion_has_stop_token_false(self, pythia_sft_tokenizer):
        # Use IDs that are definitely not stop tokens
        assert not completion_has_stop_token([10, 20, 30], tokenizer=pythia_sft_tokenizer)


# ---------------------------------------------------------------------------
# 4. AlpacaFarm gold chat template — text output
# ---------------------------------------------------------------------------


class TestAlpacaFarmChatTemplateText:
    def test_prompt_format(self, alpacafarm_tokenizer):
        msgs = [{"role": "user", "content": "Write a poem about cats."}]
        text = alpacafarm_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWrite a poem about cats.\n\n"
            "### Response:\n"
        )
        assert text == expected

    def test_full_conversation_format(self, alpacafarm_tokenizer):
        msgs = [
            {"role": "user", "content": "Write a poem about cats."},
            {"role": "assistant", "content": "Cats are fluffy and nice."},
        ]
        text = alpacafarm_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWrite a poem about cats.\n\n"
            "### Response:\nCats are fluffy and nice."
        )
        assert text == expected

    def test_prompt_is_prefix_of_full(self, alpacafarm_tokenizer):
        msgs = [
            {"role": "user", "content": "Write a poem about cats."},
            {"role": "assistant", "content": "Cats are fluffy and nice."},
        ]
        prompt = alpacafarm_tokenizer.apply_chat_template(
            msgs[:1], tokenize=False, add_generation_prompt=True
        )
        full = alpacafarm_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        assert full.startswith(prompt)

    def test_prompt_is_token_prefix_of_full(self, alpacafarm_tokenizer):
        msgs = [
            {"role": "user", "content": "Explain quantum computing."},
            {"role": "assistant", "content": "Quantum computing uses qubits."},
        ]
        prompt_text = _format_prompt(msgs, alpacafarm_tokenizer)
        full_text = _format_conversation(msgs, alpacafarm_tokenizer)

        prompt_ids = alpacafarm_tokenizer.encode(prompt_text, add_special_tokens=True)
        full_ids = alpacafarm_tokenizer.encode(full_text, add_special_tokens=True)

        assert full_ids[: len(prompt_ids)] == prompt_ids

    def test_with_input_system_message(self, alpacafarm_tokenizer):
        """When a system message carries the with-input preamble, the template
        should render it and place user content (with ### Input:) correctly."""
        preamble = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. "
            "Write a response that appropriately completes the request."
        )
        user_content = "Construct a creative story.\n\n### Input:\nA magic bow and arrow"
        msgs = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Once upon a time..."},
        ]
        text = alpacafarm_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        expected = (
            preamble + "\n\n"
            "### Instruction:\nConstruct a creative story.\n\n"
            "### Input:\nA magic bow and arrow\n\n"
            "### Response:\nOnce upon a time..."
        )
        assert text == expected

    def test_with_input_prompt_is_prefix(self, alpacafarm_tokenizer):
        preamble = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. "
            "Write a response that appropriately completes the request."
        )
        user_content = "Summarize this.\n\n### Input:\nLong article text here."
        msgs = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Summary."},
        ]
        prompt = alpacafarm_tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full = alpacafarm_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        assert full.startswith(prompt)

    def test_setup_function_sets_template(self):
        tok = AutoTokenizer.from_pretrained(PYTHIA_SFT_MODEL)
        setup_alpacafarm_gold_chat_template(tok)
        assert tok.chat_template == _ALPACAFARM_GOLD_CHAT_TEMPLATE


# ---------------------------------------------------------------------------
# 5. format_and_validate_preference_sample
# ---------------------------------------------------------------------------


class TestFormatAndValidatePreferenceSample:
    def test_pythia_chosen_only(self, pythia_sft_tokenizer):
        chosen = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        prompt, chosen_text, rejected_text = format_and_validate_preference_sample(
            chosen, pythia_sft_tokenizer,
            length_config="default", skip_validation=True,
        )
        assert prompt == "<|prompter|>Hello<|endoftext|><|assistant|>"
        assert chosen_text == "<|prompter|>Hello<|endoftext|><|assistant|>Hi!<|endoftext|>"
        assert rejected_text is None

    def test_pythia_chosen_rejected(self, pythia_sft_tokenizer):
        chosen = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        rejected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Bye!"},
        ]
        prompt, chosen_text, rejected_text = format_and_validate_preference_sample(
            chosen, pythia_sft_tokenizer,
            rejected_messages=rejected,
            length_config="default", skip_validation=True,
        )
        assert chosen_text.startswith(prompt)
        assert rejected_text.startswith(prompt)
        assert "Hi!" in chosen_text
        assert "Bye!" in rejected_text

    def test_alpacafarm_chosen_only(self, alpacafarm_tokenizer):
        chosen = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        prompt, chosen_text, _ = format_and_validate_preference_sample(
            chosen, alpacafarm_tokenizer,
            length_config="default", skip_validation=True,
        )
        assert "### Instruction:" in prompt
        assert "### Response:" in prompt
        assert chosen_text.startswith(prompt)
        assert "4" in chosen_text

    def test_length_validation_prompt_too_long(self, pythia_sft_tokenizer):
        # Register a tiny config to trigger prompt length failure.
        DATASET_LENGTH_CONFIGS["_test_tiny"] = {
            "max_prompt_tokens": 2,
            "max_response_tokens": 2,
            "max_conversation_tokens": 9999,
        }
        try:
            chosen = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            with pytest.raises(ValueError, match="exceeds max_length"):
                format_and_validate_preference_sample(
                    chosen, pythia_sft_tokenizer,
                    length_config="_test_tiny",
                )
        finally:
            del DATASET_LENGTH_CONFIGS["_test_tiny"]

    def test_length_validation_conversation_too_long(self, pythia_sft_tokenizer):
        # Register a tiny config to trigger conversation length failure.
        DATASET_LENGTH_CONFIGS["_test_tiny_conv"] = {
            "max_prompt_tokens": 9999,
            "max_response_tokens": 9999,
            "max_conversation_tokens": 2,
        }
        try:
            chosen = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            with pytest.raises(ValueError, match="exceeds max_length"):
                format_and_validate_preference_sample(
                    chosen, pythia_sft_tokenizer,
                    length_config="_test_tiny_conv",
                )
        finally:
            del DATASET_LENGTH_CONFIGS["_test_tiny_conv"]

    def test_mismatched_prompts_rejected(self, pythia_sft_tokenizer):
        chosen = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        rejected = [
            {"role": "user", "content": "Different prompt"},
            {"role": "assistant", "content": "Bye!"},
        ]
        with pytest.raises(ValueError, match="identical prompt messages"):
            format_and_validate_preference_sample(
                chosen, pythia_sft_tokenizer,
                rejected_messages=rejected,
                length_config="default", skip_validation=True,
            )


# ---------------------------------------------------------------------------
# 6. Length configs
# ---------------------------------------------------------------------------


class TestLengthConfigs:
    def test_default_config(self):
        cfg = get_length_config("default")
        assert cfg["max_prompt_tokens"] == 1024
        assert cfg["max_response_tokens"] == 1024
        assert cfg["max_conversation_tokens"] == 2048

    def test_alpacafarm_paper_config(self):
        cfg = get_length_config("alpacafarm_paper")
        assert cfg["max_prompt_tokens"] == 520
        assert cfg["max_response_tokens"] == 256
        assert cfg["max_conversation_tokens"] == 776

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError, match="Unknown length config"):
            get_length_config("nonexistent")

    def test_all_configs_have_required_keys(self):
        required = {"max_prompt_tokens", "max_response_tokens", "max_conversation_tokens"}
        for name, cfg in DATASET_LENGTH_CONFIGS.items():
            assert required.issubset(cfg.keys()), f"Config '{name}' missing keys"


# ---------------------------------------------------------------------------
# 7. AlpacaFarm RM detection
# ---------------------------------------------------------------------------


class TestAlpacaFarmRMDetection:
    def test_alpaca_farm_underscore(self):
        assert _is_alpacafarm_rm("alpaca_farm_models/reward-model-human")

    def test_alpaca_farm_hyphen(self):
        assert _is_alpacafarm_rm("alpaca-farm-rm")

    def test_alpaca_farm_in_path(self):
        assert _is_alpacafarm_rm("/some/path/alpaca_farm/reward-model")

    def test_non_alpacafarm(self):
        assert not _is_alpacafarm_rm("google/gemma-2b-it")
        assert not _is_alpacafarm_rm("Skywork/Skywork-Reward-V2-Llama-3.1-8B")
        assert not _is_alpacafarm_rm("OpenAssistant/reward-model-deberta-v3-large-v2")


# ---------------------------------------------------------------------------
# 8. Reward tensor extraction
# ---------------------------------------------------------------------------


class _FakeModel:
    """Model mock without v_head (standard sequence classifier)."""
    pass


class _FakeVHeadModel:
    """Model mock with v_head (TRL ValueHead wrapper)."""
    v_head = True


class TestRewardExtraction:
    def test_alpacafarm_rewards_attribute(self):
        model = _FakeModel()
        output = SimpleNamespace(rewards=torch.tensor([[1.5], [2.0], [3.0]]))
        rewards = extract_reward_tensors_from_model_output(model, output)
        assert torch.allclose(rewards, torch.tensor([1.5, 2.0, 3.0]))

    def test_alpacafarm_rewards_1d(self):
        model = _FakeModel()
        output = SimpleNamespace(rewards=torch.tensor([1.5, 2.0, 3.0]))
        rewards = extract_reward_tensors_from_model_output(model, output)
        assert torch.allclose(rewards, torch.tensor([1.5, 2.0, 3.0]))

    def test_standard_logits_two_columns(self):
        model = _FakeModel()
        output = SimpleNamespace(logits=torch.tensor([[1.5, 0.0], [2.0, 0.0]]))
        rewards = extract_reward_tensors_from_model_output(model, output)
        assert torch.allclose(rewards, torch.tensor([1.5, 2.0]))

    def test_standard_logits_single_column(self):
        model = _FakeModel()
        output = SimpleNamespace(logits=torch.tensor([[1.5], [2.0]]))
        rewards = extract_reward_tensors_from_model_output(model, output)
        assert torch.allclose(rewards, torch.tensor([1.5, 2.0]))

    def test_raw_tensor(self):
        model = _FakeModel()
        output = torch.tensor([1.5, 2.0, 3.0])
        rewards = extract_reward_tensors_from_model_output(model, output)
        assert torch.allclose(rewards, torch.tensor([1.5, 2.0, 3.0]))

    def test_valuehead_model(self):
        model = _FakeVHeadModel()
        output = (torch.tensor([[0.0]]), None, torch.tensor([4.0, 5.0]))
        rewards = extract_reward_tensors_from_model_output(model, output)
        assert torch.allclose(rewards, torch.tensor([4.0, 5.0]))

    def test_valuehead_model_bad_output_raises(self):
        model = _FakeVHeadModel()
        with pytest.raises(ValueError, match="ValueHead reward model"):
            extract_reward_tensors_from_model_output(model, SimpleNamespace(logits=torch.tensor([[1.0]])))

    def test_no_logits_raises(self):
        model = _FakeModel()
        with pytest.raises(ValueError, match="Could not extract logits"):
            extract_reward_tensors_from_model_output(model, SimpleNamespace(foo="bar"))


# ---------------------------------------------------------------------------
# 9. load_policy_and_tokenizer integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLoadPolicyAndTokenizer:
    """Integration tests that download/load the actual 70m model (~150 MB)."""

    def test_pythia_sft_auto_detects_template(self):
        model, tok = load_policy_and_tokenizer(PYTHIA_SFT_MODEL)
        assert tok.chat_template == _PYTHIA_OA_V2_CHAT_TEMPLATE
        assert tok.pad_token is not None
        assert model.config.pad_token_id == tok.pad_token_id

    def test_pythia_sft_generation_config(self):
        model, tok = load_policy_and_tokenizer(PYTHIA_SFT_MODEL)
        assert model.generation_config.pad_token_id == tok.pad_token_id
        stop_ids = get_generation_stop_token_ids(tok)
        assert model.generation_config.eos_token_id == stop_ids

    # The invariant is that every tokenizer id indexes a valid embedding row
    # (embed_size >= len(tok)). Resizing is grow-only: padded-vocab checkpoints
    # (pythia: 50304 config vs 50282 tokens) must keep their config shapes, or
    # the HF weights would no longer match what the vLLM weight hot-swap expects.

    def test_pythia_sft_embedding_size(self):
        model, tok = load_policy_and_tokenizer(PYTHIA_SFT_MODEL)
        embed_size = model.get_input_embeddings().weight.shape[0]
        assert embed_size >= len(tok)

    def test_base_pythia_gets_template_and_resized_embeddings(self):
        model, tok = load_policy_and_tokenizer(PYTHIA_BASE_MODEL)
        assert tok.chat_template == _PYTHIA_OA_V2_CHAT_TEMPLATE
        assert tok.pad_token is not None
        assert _has_pythia_oa_tokens(tok)
        embed_size = model.get_input_embeddings().weight.shape[0]
        assert embed_size >= len(tok)


# ---------------------------------------------------------------------------
# 10. End-to-end: SFT tokenization preserves completion mask invariant
# ---------------------------------------------------------------------------


class TestSFTCompletionMaskInvariant:
    """Verify that SFT tokenization produces a valid completion mask:
    the prompt tokens at the start of input_ids should match the independently
    tokenized prompt, and the mask should be 0 for prompt, 1 for completion."""

    def test_completion_mask_single_turn(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        prompt_text, full_text, _ = format_and_validate_preference_sample(
            msgs, pythia_sft_tokenizer,
            length_config="default", skip_validation=True,
        )
        tokens_full = tokenize_for_sft(full_text, pythia_sft_tokenizer)
        input_ids = tokens_full["input_ids"][0]

        prompt_ids = tokenize_for_sft(prompt_text, pythia_sft_tokenizer)["input_ids"][0]
        prompt_len = len(prompt_ids)

        # Prompt must be a strict prefix
        assert prompt_len < len(input_ids)
        assert torch.equal(input_ids[:prompt_len], prompt_ids)

        # Build mask: 0 for prompt, 1 for completion
        mask = torch.zeros_like(input_ids)
        mask[prompt_len:] = 1

        assert mask[:prompt_len].sum() == 0
        assert mask[prompt_len:].sum() == len(input_ids) - prompt_len
        assert mask[prompt_len:].sum() > 0  # completion is non-empty

    def test_completion_mask_multi_turn(self, pythia_sft_tokenizer):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am great, thanks!"},
        ]
        prompt_text, full_text, _ = format_and_validate_preference_sample(
            msgs, pythia_sft_tokenizer,
            length_config="default", skip_validation=True,
        )
        tokens_full = tokenize_for_sft(full_text, pythia_sft_tokenizer)
        input_ids = tokens_full["input_ids"][0]

        prompt_ids = tokenize_for_sft(prompt_text, pythia_sft_tokenizer)["input_ids"][0]
        prompt_len = len(prompt_ids)

        assert prompt_len < len(input_ids)
        assert torch.equal(input_ids[:prompt_len], prompt_ids)
