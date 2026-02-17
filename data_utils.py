"""Unified data formatting and tokenization for the full pipeline.

Conventions:
- apply_chat_template may include BOS; strip BOS before tokenization, then always use add_special_tokens=True
- padding_side="left" everywhere
- No truncation during tokenization; datasets must be pre-filtered
- format_reward_texts: for RM scoring (Evaluation, GRPO rewards, Annotation).
  Matches HF model-card convention (reconstruct -> strip BOS -> tokenize with add_special_tokens=True).
- format_prompt: for generation inputs (GRPO prompts, eval prompts)
- format_conversation: for full conversations (RM training, SFT, annotation)
"""


def setup_tokenizer(tokenizer):
    """Ensure consistent tokenizer configuration across all stages."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
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


def format_prompt(conversation, tokenizer):
    """Format the prompt portion of a conversation (all messages except the last).
    Returns a string ending with the generation prompt marker.

    Used by: GRPO dataset, evaluation, prompt extraction, precompute_reward_means.
    """
    messages = conversation[:-1]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def format_conversation(conversation, tokenizer):
    """Format a full conversation (prompt + response) with chat template.
    Returns a string with proper assistant header and EOT around the response.

    Used by: RM dataset prep, SFT dataset, dataset annotation.

    Key property: format_conversation(msgs) starts with format_prompt(msgs) as a prefix.
    """
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        enable_thinking=False,
    )


def format_reward_texts(prompt_messages_list, responses, rm_tokenizer):
    """Format conversations for RM scoring using the RM's own chat template.

    Args:
        prompt_messages_list: If 'responses' is provided, this is a list of
            structured prompt message lists. If 'responses' is None, this is
            a list of full conversation message lists (e.g. from a dataset).
        responses: Optional list of assistant response strings.
        rm_tokenizer: Tokenizer for the reward model.

    Returns:
        List of formatted strings with BOS stripped (caller uses add_special_tokens=True).
    """
    texts = []
    if responses is not None:
        # Hybrid mode: reconstruct from prompt + response
        for prompt_msgs, response in zip(prompt_messages_list, responses):
            full_conv = list(prompt_msgs) + [{"role": "assistant", "content": response}]
            texts.append(_format_single_conv_for_rm(full_conv, rm_tokenizer))
    else:
        # Direct mode: use full conversations (e.g. ground truth from dataset)
        for conv in prompt_messages_list:
            texts.append(_format_single_conv_for_rm(conv, rm_tokenizer))
    return texts


def _format_single_conv_for_rm(conversation, tokenizer):
    """Helper to apply chat template and strip BOS."""
    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    return strip_bos_if_present(text, tokenizer)


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
