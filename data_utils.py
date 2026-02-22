"""Unified data formatting and tokenization for the full pipeline.

Conventions:
- apply_chat_template may include BOS; strip BOS before tokenization, then always use add_special_tokens=True
- padding_side="left" everywhere
- No truncation during tokenization; datasets must be pre-filtered
- format_and_validate_preference_sample: single source of truth for chat
  formatting + length validation (used by SFT/GRPO/RM evaluation/annotation).
- _format_prompt: internal helper for generation prompt formatting
- _format_conversation: internal helper for full conversation formatting
- get_generation_stop_token_ids: shared stop-token detection for generation and EOS checks
"""

DEFAULT_MAX_PROMPT_TOKENS = 1024    
DEFAULT_MAX_RESPONSE_TOKENS = 1024
DEFAULT_MAX_CONVERSATION_TOKENS = DEFAULT_MAX_PROMPT_TOKENS + DEFAULT_MAX_RESPONSE_TOKENS


def _looks_like_llama_model(tokenizer, model_name=None):
    """Best-effort detection for Llama-family tokenizers/models."""
    candidates = [
        model_name,
        getattr(tokenizer, "name_or_path", None),
        getattr(getattr(tokenizer, "tokenizer", None), "name_or_path", None),
        tokenizer.__class__.__name__,
    ]
    for candidate in candidates:
        if candidate and "llama" in str(candidate).lower():
            print("Detected Llama model.")
            return True
    return False


def setup_tokenizer(tokenizer, model_name=None):
    """Ensure consistent tokenizer configuration across all stages."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # Keep legacy Llama behavior: use [PAD] if the tokenizer has no pad token.
        if _looks_like_llama_model(tokenizer, model_name=model_name):
            if hasattr(tokenizer, "add_special_tokens"):
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def strip_bos_if_present(text, tokenizer):
    """Strip a leading BOS token from text, if present."""
    if tokenizer.bos_token is None:
        return text
    if isinstance(text, str) and text.startswith(tokenizer.bos_token):
        return text[len(tokenizer.bos_token) :]
    return text


def strip_bos_if_present_batch(texts, tokenizer):
    """Strip a leading BOS token from each text in a list."""
    return [strip_bos_if_present(text, tokenizer) for text in texts]


def _add_token_id(stop_ids, token_id):
    """Insert token IDs into a set, accepting ints or iterables."""
    if isinstance(token_id, (list, tuple, set)):
        for tid in token_id:
            _add_token_id(stop_ids, tid)
        return
    if token_id is None:
        return
    try:
        tid = int(token_id)
    except (TypeError, ValueError):
        return
    if tid >= 0:
        stop_ids.add(tid)


def get_generation_stop_token_ids(tokenizer):
    """Return tokenizer-specific generation stop IDs.

    Includes eos_token_id and common chat turn-end tokens used by modern
    chat templates (for example Qwen's <|im_end|>).
    """
    stop_ids = set()
    _add_token_id(stop_ids, tokenizer.eos_token_id)

    # Some wrappers keep the underlying tokenizer on `.tokenizer`.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    convert = getattr(raw_tokenizer, "convert_tokens_to_ids", None)
    unk_token_id = getattr(raw_tokenizer, "unk_token_id", None)
    if convert is not None:
        for token in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
            token_id = convert(token)
            if token_id is None or token_id == unk_token_id:
                continue
            _add_token_id(stop_ids, token_id)

    return sorted(stop_ids)


def completion_has_stop_token(completion_ids, tokenizer=None, stop_token_ids=None):
    """Check whether a generated completion contains any stop token."""
    if hasattr(completion_ids, "tolist"):
        completion_ids = completion_ids.tolist()
    if stop_token_ids is None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when stop_token_ids is None")
        stop_token_ids = get_generation_stop_token_ids(tokenizer)
    stop_ids = set(stop_token_ids)
    if not stop_ids:
        return False
    for token_id in completion_ids:
        if int(token_id) in stop_ids:
            return True
    return False


def tokenize_text_with_special_tokens(text, tokenizer, **kwargs):
    """Tokenize one text after stripping BOS and forcing special-token addition."""
    text = strip_bos_if_present(text, tokenizer)
    kwargs = dict(kwargs)
    kwargs["add_special_tokens"] = True
    return tokenizer(text=text, **kwargs)


def tokenize_texts_with_special_tokens(texts, tokenizer, **kwargs):
    """Tokenize a batch of texts after stripping BOS and forcing special-token addition."""
    texts = strip_bos_if_present_batch(texts, tokenizer)
    kwargs = dict(kwargs)
    kwargs["add_special_tokens"] = True
    return tokenizer(text=texts, **kwargs)


def count_tokens_with_special_tokens(text, tokenizer):
    """Token count after BOS stripping and tokenizer-added special tokens."""
    text = strip_bos_if_present(text, tokenizer)
    return len(tokenizer.encode(text, add_special_tokens=True))


def validate_length_or_fail(
    value,
    max_length,
    *,
    name="sequence",
    tokenizer=None,
    sample_id=None,
):
    """Validate token length for a token-ID sequence or text and fail fast."""
    if isinstance(value, str):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when validating text length")
        token_length = count_tokens_with_special_tokens(value, tokenizer)
    else:
        if hasattr(value, "tolist"):
            value = value.tolist()
        try:
            token_length = len(value)
        except TypeError as exc:
            raise TypeError(
                f"Unsupported value type for {name}: {type(value)!r}. "
                "Expected text or token-id sequence."
            ) from exc

    if token_length > max_length:
        sample_suffix = f" (sample_id={sample_id})" if sample_id is not None else ""
        raise ValueError(
            f"{name} length {token_length} exceeds max_length={max_length}{sample_suffix}."
        )
    return token_length


def validate_conversation_messages_or_fail(
    messages,
    *,
    field_name="conversation",
    sample_id=None,
    context="sample",
    require_last_assistant=True,
):
    """Validate that messages are a non-empty list of {role, content} dicts."""
    sample_suffix = f" (sample_id={sample_id})" if sample_id is not None else ""

    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError(
            f"{context} {field_name} must be a non-empty list of messages{sample_suffix}."
        )

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"{context} {field_name}[{idx}] must be a dict{sample_suffix}."
            )
        if "role" not in message or "content" not in message:
            raise ValueError(
                f"{context} {field_name}[{idx}] must contain role/content keys{sample_suffix}."
            )
        if not isinstance(message["role"], str):
            raise ValueError(
                f"{context} {field_name}[{idx}].role must be a string{sample_suffix}."
            )
        if not isinstance(message["content"], str):
            raise ValueError(
                f"{context} {field_name}[{idx}].content must be a string{sample_suffix}."
            )

    if require_last_assistant and messages[-1]["role"] != "assistant":
        raise ValueError(
            f"{context} {field_name} last message must have role='assistant'{sample_suffix}."
        )


def _apply_chat_template_no_thinking(tokenizer, messages, *, add_generation_prompt=False):
    """Apply chat template with a best-effort enable_thinking=False setting.

    Some tokenizers support ``enable_thinking`` while others don't. This helper
    keeps one calling convention for all tokenizers and falls back cleanly when
    the kwarg is unsupported.
    """
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
        # Older tokenizers may not accept enable_thinking.
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)


def _format_prompt(conversation, tokenizer):
    """Format the prompt portion of a conversation (all messages except the last).
    Returns a string ending with the generation prompt marker.
    """
    messages = conversation[:-1]
    return _apply_chat_template_no_thinking(
        tokenizer, messages, add_generation_prompt=True
    )


def _format_conversation(conversation, tokenizer):
    """Format a full conversation with content-insensitive continuation logic.

    We always build:
    ``prompt + assistant_content + assistant_suffix``,
    where ``assistant_suffix`` is inferred from the empty-assistant render under
    the same tokenizer/template settings.
    """
    prompt_text = _format_prompt(conversation, tokenizer)
    prompt_messages = conversation[:-1]
    assistant_content = conversation[-1]["content"]

    # Use an empty assistant response to infer suffix (for example turn-end token).
    empty_assistant_conv = list(prompt_messages) + [{"role": "assistant", "content": ""}]
    full_with_empty_assistant = _apply_chat_template_no_thinking(
        tokenizer, empty_assistant_conv, add_generation_prompt=False
    )

    if not full_with_empty_assistant.startswith(prompt_text):
        raise ValueError(
            "Chat template mismatch: prompt is not a prefix of empty-assistant conversation."
        )

    assistant_suffix = full_with_empty_assistant[len(prompt_text) :]
    return prompt_text + assistant_content + assistant_suffix


def format_and_validate_preference_sample(
    chosen_messages,
    tokenizer,
    *,
    rejected_messages=None,
    max_prompt_length=DEFAULT_MAX_PROMPT_TOKENS,
    max_conversation_length=DEFAULT_MAX_CONVERSATION_TOKENS,
    sample_id=None,
    context="sample",
):
    """Format prompt/conversation texts and validate prompt/full-length constraints.

    Pass ``None`` for ``max_prompt_length`` and/or ``max_conversation_length`` to
    skip that validation.

    Returns:
        Tuple of (prompt_text, chosen_text, rejected_text_or_None).
    """
    validate_conversation_messages_or_fail(
        chosen_messages,
        field_name="chosen",
        sample_id=sample_id,
        context=context,
        require_last_assistant=True,
    )
    if rejected_messages is not None:
        validate_conversation_messages_or_fail(
            rejected_messages,
            field_name="rejected",
            sample_id=sample_id,
            context=context,
            require_last_assistant=True,
        )
        if chosen_messages[:-1] != rejected_messages[:-1]:
            sample_suffix = f" (sample_id={sample_id})" if sample_id is not None else ""
            raise ValueError(
                f"{context} chosen/rejected must share identical prompt messages{sample_suffix}."
            )

    prompt_text = _format_prompt(chosen_messages, tokenizer)
    chosen_text = _format_conversation(chosen_messages, tokenizer)
    if not chosen_text.startswith(prompt_text):
        sample_suffix = f" (sample_id={sample_id})" if sample_id is not None else ""
        raise ValueError(
            f"{context} formatting mismatch: prompt is not a prefix of chosen conversation{sample_suffix}."
        )

    if max_prompt_length is not None:
        validate_length_or_fail(
            prompt_text,
            max_prompt_length,
            name=f"{context} prompt",
            tokenizer=tokenizer,
            sample_id=sample_id,
        )
    if max_conversation_length is not None:
        validate_length_or_fail(
            chosen_text,
            max_conversation_length,
            name=f"{context} chosen conversation",
            tokenizer=tokenizer,
            sample_id=sample_id,
        )

    rejected_text = None
    if rejected_messages is not None:
        rejected_text = _format_conversation(rejected_messages, tokenizer)
        if max_conversation_length is not None:
            validate_length_or_fail(
                rejected_text,
                max_conversation_length,
                name=f"{context} rejected conversation",
                tokenizer=tokenizer,
                sample_id=sample_id,
            )

    return prompt_text, chosen_text, rejected_text


def tokenize_for_rm(texts, tokenizer):
    """Tokenize pre-formatted texts for reward model (training or inference).

    Strips leading BOS from each text, then tokenizes with
    add_special_tokens=True.
    Left-pads for batch processing.

    Used by: RM training (load_datasets.py), RM scoring (reward_utils.py),
    evaluation (evaluate_policy.py), annotation (dataset_annotation.py).
    """
    return tokenize_texts_with_special_tokens(
        texts,
        tokenizer,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )


def tokenize_for_sft(text, tokenizer):
    """Tokenize a single pre-formatted text for SFT training.

    No padding (handled by SFTTrainer/collator).
    Strips leading BOS, then tokenizes with add_special_tokens=True.
    """
    return tokenize_text_with_special_tokens(text, tokenizer, return_tensors="pt")
