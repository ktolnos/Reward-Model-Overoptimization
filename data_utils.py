"""Unified data formatting and tokenization for the full pipeline.

Conventions:
- apply_chat_template handles all special tokens -> always use add_special_tokens=False when tokenizing
- padding_side="left" everywhere
- No truncation during tokenization; datasets must be pre-filtered
- format_prompt: for generation inputs (GRPO prompts, eval prompts)
- format_conversation: for full conversations (RM training, SFT, annotation)
- EOT is appended when scoring (prompt + completion) via build_reward_texts in reward_utils
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
