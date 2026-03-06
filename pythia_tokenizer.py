"""Pythia / Open-Assistant v2 tokenizer setup and vLLM compatibility patches.

Extracted from data_utils.py so that the monkey-patching logic lives in one
place and can be tested independently.
"""

# ---- Pythia / Open-Assistant v2 chat template ----

_PYTHIA_OA_V2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|prompter|>{{ message['content'] }}<|endoftext|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>{{ message['content'] }}<|endoftext|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

_PYTHIA_EXPECTED_SPECIAL_TOKENS = ("<|prompter|>", "<|assistant|>", "<|endoftext|>")

# Tokens added by the Open-Assistant SFT process, in their original order.
# <|endoftext|> is already the native EOS token (id 0) in all Pythia models.
# The order matters: it determines token IDs and must match published SFT
# checkpoints (e.g. tlc4418/pythia_70m_sft).
_PYTHIA_OA_TOKENS_TO_ADD = (
    "<|system|>", "<|prefix_begin|>", "<|prefix_end|>",
    "<|prompter|>", "<|assistant|>",
)


def _has_pythia_oa_tokens(tokenizer):
    """Check if the tokenizer has the Open-Assistant special tokens used by
    the SFT'd Pythia models from Coste et al."""
    vocab = tokenizer.get_vocab()
    return all(tok in vocab for tok in _PYTHIA_EXPECTED_SPECIAL_TOKENS)


def _looks_like_pythia_model(tokenizer, model_name=None):
    """Best-effort detection for Pythia-family tokenizers/models."""
    candidates = [
        model_name,
        getattr(tokenizer, "name_or_path", None),
    ]
    for candidate in candidates:
        if candidate and "pythia" in str(candidate).lower():
            return True
    return False


def setup_pythia_chat_template(tokenizer):
    """Register the Open-Assistant v2 chat template on a Pythia tokenizer.

    The SFT'd Pythia models from the paper added <|prompter|> and <|assistant|>
    as special tokens.  This function verifies they exist and sets a Jinja2
    chat template that replicates the paper's manual string formatting so that
    ``apply_chat_template`` produces identical token IDs.

    Called at load time -- does NOT save anything to disk.
    """
    vocab = tokenizer.get_vocab()
    for tok in _PYTHIA_EXPECTED_SPECIAL_TOKENS:
        if tok not in vocab:
            raise ValueError(
                f"Pythia tokenizer is missing expected special token '{tok}'. "
                "Make sure you are loading from an SFT'd checkpoint that has "
                "the Open-Assistant vocabulary additions (e.g. tlc4418/pythia_70m_sft)."
            )
    tokenizer.chat_template = _PYTHIA_OA_V2_CHAT_TEMPLATE
    return tokenizer


def patch_tokenizer_for_vllm(tokenizer):
    """Add missing attributes that vLLM expects on older tokenizer classes.

    Newer ``transformers`` versions define ``all_special_tokens_extended`` on
    ``PreTrainedTokenizerBase``, but older ``GPTNeoXTokenizer`` (the *slow*
    tokenizer shipped with some Pythia checkpoints) lacks it.  vLLM's
    ``get_cached_tokenizer`` accesses the attribute unconditionally, so we
    monkey-patch it here when missing.
    """
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        # Mirror what PreTrainedTokenizerBase.all_special_tokens_extended does:
        # return all special tokens as a list (may contain AddedToken objects).
        @property
        def _all_special_tokens_extended(self):
            all_toks = []
            # _special_tokens is the dict backing eos_token, bos_token, etc.
            set_attr = getattr(self, "_special_tokens", {})
            for attr_value in set_attr.values():
                if attr_value is not None:
                    all_toks.append(attr_value)
            all_toks.extend(getattr(self, "_additional_special_tokens", []))
            return all_toks

        type(tokenizer).all_special_tokens_extended = _all_special_tokens_extended

    return tokenizer


def setup_pythia_tokenizer(tokenizer, model_name=None):
    """Full Pythia tokenizer setup: add OA tokens if needed, set chat template,
    and apply vLLM compatibility patches.

    This is called from ``data_utils.setup_tokenizer`` when a Pythia model is
    detected.  It can also be called directly.
    """
    if not _has_pythia_oa_tokens(tokenizer):
        vocab = tokenizer.get_vocab()
        tokens_to_add = [t for t in _PYTHIA_OA_TOKENS_TO_ADD if t not in vocab]
        if tokens_to_add:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": tokens_to_add}
            )
            print(
                f"Added {tokens_to_add} to Pythia tokenizer. "
                "Remember to resize model embeddings."
            )
    setup_pythia_chat_template(tokenizer)
    patch_tokenizer_for_vllm(tokenizer)
    return tokenizer
