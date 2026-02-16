"""Unified data formatting and tokenization for the full pipeline.

Conventions:
- apply_chat_template handles all special tokens -> always use add_special_tokens=False when tokenizing
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
    if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token) :]
    return text


def tokenize_for_rm(texts, tokenizer, add_special_tokens=False):
    """Tokenize pre-formatted texts for reward model (training or inference).

    Uses add_special_tokens=False by default because texts from
    format_conversation/format_prompt already include all special tokens
    from apply_chat_template.  Pass add_special_tokens=True when the BOS
    has been stripped from the text (e.g. following the HF model-card
    convention of strip-BOS + re-add via tokenizer).
    Left-pads for batch processing.

    Used by: RM training (load_datasets.py), RM scoring (reward_utils.py),
    evaluation (evaluate_policy.py), annotation (dataset_annotation.py).
    """
    return tokenizer(
        text=texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=add_special_tokens,
    )


def tokenize_for_sft(text, tokenizer):
    """Tokenize a single pre-formatted text for SFT training.

    No padding (handled by SFTTrainer/collator).
    Uses add_special_tokens=False because text from format_conversation
    already includes all special tokens.
    """
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)
