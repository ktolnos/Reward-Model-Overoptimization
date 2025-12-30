def build_prompt_from_chosen(chosen_messages, tokenizer, max_length=None):
    messages = chosen_messages[:-1]
    if max_length is None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    chat_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        truncation=True,
        max_length=max_length,
    )
    return tokenizer.decode(chat_tokens, skip_special_tokens=False)


def build_prompt_plus_response_from_chosen(chosen_messages, tokenizer, max_length=None):
    if not chosen_messages:
        return ""
    prompt = build_prompt_from_chosen(
        chosen_messages,
        tokenizer,
        max_length=max_length,
    )
    last_message = chosen_messages[-1]
    if isinstance(last_message, dict):
        last_response = last_message.get("content", "")
    else:
        last_response = str(last_message)
    return prompt + last_response
