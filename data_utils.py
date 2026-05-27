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

from pythia_tokenizer import (  # noqa: F401 — re-exported for backward compat
    _PYTHIA_OA_V2_CHAT_TEMPLATE,
    _PYTHIA_EXPECTED_SPECIAL_TOKENS,
    _PYTHIA_OA_TOKENS_TO_ADD,
    _has_pythia_oa_tokens,
    _looks_like_pythia_model,
    setup_pythia_chat_template,
    setup_pythia_tokenizer,
    patch_tokenizer_for_vllm,
    patch_config_for_vllm,
)

# Dataset-specific length configurations for pipeline consistency.
# Training scripts select a config via --length_config and assert the active
# constants match the dataset being used.
DATASET_LENGTH_CONFIGS = {
    "default": {
        "max_prompt_tokens": 1024,
        "max_response_tokens": 1024,
        "max_conversation_tokens": 2048,
    },
    "alpacafarm_paper": {
        "max_prompt_tokens": 520,
        "max_response_tokens": 256,
        "max_conversation_tokens": 776,
    },
}


def get_length_config(config_name):
    """Return a length config dict by name, or raise if unknown."""
    if config_name not in DATASET_LENGTH_CONFIGS:
        raise ValueError(
            f"Unknown length config '{config_name}'. "
            f"Available: {list(DATASET_LENGTH_CONFIGS.keys())}"
        )
    return DATASET_LENGTH_CONFIGS[config_name]


def compute_max_prompt_length(dataset_or_path, tokenizer, *, padding_tokens=32):
    """Measure the actual max prompt token length in a dataset.

    Scans all conversations, tokenizes prompts with the given tokenizer, and
    returns the observed maximum plus *padding_tokens*.  This lets vLLM
    allocate only as much KV-cache as the data actually needs, regardless of
    which tokenizer is used.

    Accepts either a HuggingFace ``Dataset`` object or a dataset path string.
    """
    import datasets as _ds_lib

    if isinstance(dataset_or_path, str):
        dataset = _ds_lib.load_dataset(dataset_or_path, split="train")
    else:
        dataset = dataset_or_path

    max_prompt = 0
    for example in dataset:
        prompt_text, _, _ = format_and_validate_preference_sample(
            example["chosen"],
            tokenizer,
            length_config="default",
            skip_validation=True,
        )
        prompt_tok = count_tokens_with_special_tokens(prompt_text, tokenizer)
        max_prompt = max(max_prompt, prompt_tok)

    result = max_prompt + padding_tokens
    print(
        f"[auto_prompt_length] Measured max prompt length with "
        f"{tokenizer.name_or_path}: {max_prompt} tokens "
        f"(+{padding_tokens} padding → {result})"
    )
    return result


# ---- AlpacaFarm gold RM chat template (Alpaca instruction format) ----

_ALPACAFARM_GOLD_CHAT_TEMPLATE = (
    # Preamble comes from the system message (set by convert_paper_dataset.py).
    # Falls back to the no-input preamble when no system message is present.
    "{% set ns = namespace(preamble='', user='') %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{% set ns.preamble = message['content'] %}{% endif %}"
    "{% if message['role'] == 'user' %}{% set ns.user = message['content'] %}{% endif %}"
    "{% endfor %}"
    "{% if not ns.preamble %}"
    "{% set ns.preamble = 'Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.' %}"
    "{% endif %}"
    "{{ ns.preamble }}\n\n"
    "### Instruction:\n{{ ns.user }}\n\n"
    "### Response:\n"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{% endif %}"
)


def setup_alpacafarm_gold_chat_template(tokenizer):
    """Register the Alpaca instruction-format chat template on a tokenizer.

    The AlpacaFarm 7B gold reward model was trained on the Alpaca prompt
    template (``### Instruction:`` / ``### Response:``).  This sets a Jinja2
    chat template so the gold RM tokenizer goes through the same
    ``apply_chat_template`` code path as everything else.

    Called at load time -- does NOT save anything to disk.
    """
    tokenizer.chat_template = _ALPACAFARM_GOLD_CHAT_TEMPLATE
    return tokenizer


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
    """Ensure consistent tokenizer configuration across all stages.

    Automatically detects Pythia models by name and ensures the
    Open-Assistant v2 special tokens (``<|prompter|>``, ``<|assistant|>``)
    and chat template are present.  For SFT'd checkpoints the tokens already
    exist; for base Pythia they are added to the vocabulary (the caller must
    resize model embeddings afterwards -- ``load_policy_and_tokenizer`` does
    this automatically).
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # Keep legacy Llama behavior: use [PAD] if the tokenizer has no pad token.
        if _looks_like_llama_model(tokenizer, model_name=model_name):
            if hasattr(tokenizer, "add_special_tokens"):
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect Pythia models: add OA v2 tokens if missing, then set template,
    # and apply vLLM compatibility patches (tokenizer + model config).
    if tokenizer.chat_template is None and _looks_like_pythia_model(tokenizer, model_name=model_name):
        setup_pythia_tokenizer(tokenizer, model_name=model_name)
        patch_config_for_vllm()

    return tokenizer


def _is_lora_checkpoint(path):
    """Check if a path contains a LoRA adapter (has adapter_config.json)."""
    import os
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def _get_lora_base_model_path(adapter_path):
    """Read the base model path from a LoRA adapter_config.json."""
    import os, json
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        return json.load(f)["base_model_name_or_path"]


def load_causal_lm(model_name_or_path, *, trust_remote_code=True, device_map=None):
    """Load a causal LM, auto-detecting and merging LoRA adapters.

    If ``model_name_or_path`` contains ``adapter_config.json``, loads the
    adapter on top of its base model and returns the merged result.
    Otherwise loads a plain ``AutoModelForCausalLM``.
    """
    import torch
    if _is_lora_checkpoint(model_name_or_path):
        from peft import AutoPeftModelForCausalLM
        print(f"Detected LoRA adapter at {model_name_or_path}, loading and merging...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        model = model.merge_and_unload()
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    return model


def load_policy_and_tokenizer(model_name_or_path, *, trust_remote_code=True):
    """Load a policy model and its tokenizer with consistent setup.

    Handles:
    - LoRA adapter detection (via ``load_causal_lm``).
    - Tokenizer loading and ``setup_tokenizer`` (auto-detects Pythia chat template).
    - Model loading in bfloat16.
    - Embedding resizing to match the tokenizer (only when new tokens were added).
    - ``pad_token_id`` propagation to model config and ``generation_config``.

    Returns ``(model, tokenizer)``.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code,
    )
    # For model-type detection, use the base model name so that checks like
    # _looks_like_llama_model see the real model name, not the adapter path.
    base_model_name = model_name_or_path
    if _is_lora_checkpoint(model_name_or_path):
        base_model_name = _get_lora_base_model_path(model_name_or_path)
    setup_tokenizer(tokenizer, model_name=base_model_name)

    model = load_causal_lm(model_name_or_path, trust_remote_code=trust_remote_code)
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = get_generation_stop_token_ids(tokenizer)

    return model, tokenizer


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

    Includes eos_token_id and common chat turn-end tokens. <|endoftext|> is
    looked up explicitly so it stays in the set even after eos_token_id has
    been re-pointed to a chat turn-end token (e.g. <|im_end|> for Qwen).
    """
    stop_ids = set()
    # Some wrappers keep the underlying tokenizer on `.tokenizer`.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    _add_token_id(stop_ids, raw_tokenizer.eos_token_id)

    convert = getattr(raw_tokenizer, "convert_tokens_to_ids", None)
    unk_token_id = getattr(raw_tokenizer, "unk_token_id", None)
    if convert is not None:
        for token in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>"):
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
    length_config,
    skip_validation=False,
    sample_id=None,
    context="sample",
):
    """Format prompt/conversation texts and validate prompt/full-length constraints.

    Args:
        length_config: Name of a DATASET_LENGTH_CONFIGS entry (e.g. ``"default"``).
            Required keyword argument — every caller must specify it explicitly.
        skip_validation: If ``True``, skip length validation entirely (formatting only).

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

    if not skip_validation:
        cfg = get_length_config(length_config)
        validate_length_or_fail(
            prompt_text,
            cfg["max_prompt_tokens"],
            name=f"{context} prompt",
            tokenizer=tokenizer,
            sample_id=sample_id,
        )
        validate_length_or_fail(
            chosen_text,
            cfg["max_conversation_tokens"],
            name=f"{context} chosen conversation",
            tokenizer=tokenizer,
            sample_id=sample_id,
        )

    rejected_text = None
    if rejected_messages is not None:
        rejected_text = _format_conversation(rejected_messages, tokenizer)
        if not skip_validation:
            validate_length_or_fail(
                rejected_text,
                cfg["max_conversation_tokens"],
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


# ---------------------------------------------------------------------------
# Shared dataset loading skeleton
# ---------------------------------------------------------------------------

def build_train_eval_datasets(
    data_path_train,
    tokenizer,
    *,
    post_process_fn,
    eval_proportion=0.1,
    size=None,
    length_config,
    skip_length_validation=False,
):
    """Load a preference dataset, split it, and apply stage-specific formatting.

    This is the single shared skeleton used by SFT, GRPO, DPO (and any future
    stage).  The caller provides ``post_process_fn`` which transforms the raw
    dataset into the format the trainer expects.

    Args:
        post_process_fn: callable(ds, tokenizer, *, length_config, skip_length_validation)
            that returns a processed Dataset.
    """
    import datasets as _ds_lib

    ds = _ds_lib.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)

    ds_train = post_process_fn(
        ds_dict["train"], tokenizer,
        length_config=length_config,
        skip_length_validation=skip_length_validation,
    )
    ds_eval = post_process_fn(
        ds_dict["test"], tokenizer,
        length_config=length_config,
        skip_length_validation=skip_length_validation,
    )
    return ds_train, ds_eval


# ---------------------------------------------------------------------------
# Length-config safeguard
# ---------------------------------------------------------------------------

# DPOConfig default for max_length; used to detect "not overridden on CLI".
_DPO_MAX_LENGTH_SENTINEL = 1024


def set_lengths_from_config(training_args, length_config_name, *, trainer_type):
    """Derive all length-related trainer args from a single length config.

    This is the **single point** where trainer args receive their length
    values.  Call it once, right after parsing arguments, and never set
    max_length / max_prompt_length / max_completion_length by hand.

    Raises ``ValueError`` when the user overrides a length on the CLI that
    conflicts with the config (catches the "changed it in one place but not
    the other" bug).

    Supported ``trainer_type`` values: ``"dpo"``, ``"grpo"``, ``"sft"``.
    """
    cfg = get_length_config(length_config_name)

    if trainer_type == "dpo":
        expected = cfg["max_conversation_tokens"]
        current = getattr(training_args, "max_length", None)
        if current is not None and current != _DPO_MAX_LENGTH_SENTINEL and current != expected:
            raise ValueError(
                f"--max_length={current} was set on the CLI but conflicts with "
                f"length_config '{length_config_name}' "
                f"(max_conversation_tokens={expected}).  "
                f"Remove --max_length and use --length_config to control lengths."
            )
        training_args.max_length = expected

    elif trainer_type == "grpo":
        # max_completion_length
        if training_args.max_completion_length != 256:
            raise ValueError(
                f"--max_completion_length is overridden on the command line. "
                f"Use --length_config instead (active config "
                f"'{length_config_name}' sets "
                f"max_response_tokens={cfg['max_response_tokens']})."
            )
        training_args.max_completion_length = cfg["max_response_tokens"]

        # vllm_max_model_length is handled separately because of
        # auto_prompt_length; callers must NOT set it on the CLI.
        if training_args.vllm_max_model_length is not None:
            raise ValueError(
                "--vllm_max_model_length is overridden on the command line. "
                "Use --length_config (or --auto_prompt_length) instead."
            )

    elif trainer_type == "sft":
        # SFTTrainer does not truncate; the dataset is pre-filtered by
        # format_and_validate_preference_sample.  Nothing to set here, but
        # we still validate the config name is valid.
        pass

    else:
        raise ValueError(f"Unknown trainer_type '{trainer_type}'")

    return cfg
