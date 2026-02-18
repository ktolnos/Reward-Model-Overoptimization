import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from data_utils import (
    format_and_validate_preference_sample,
    tokenize_text_with_special_tokens,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_CONVERSATION_TOKENS,
)


# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(
    data_path,
    tokenizer,
    split="train",
    size=None,
    model_name="",
    max_prompt_length=DEFAULT_MAX_PROMPT_TOKENS,
    max_conversation_length=DEFAULT_MAX_CONVERSATION_TOKENS,
):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        prompt_text, prompt_plus_chosen_response, prompt_plus_rejected_response = (
            format_and_validate_preference_sample(
                chosen_messages,
                tokenizer,
                rejected_messages=rejected_messages,
                max_prompt_length=max_prompt_length,
                max_conversation_length=max_conversation_length,
                context="RM",
            )
        )
        prompt_ids = tokenize_text_with_special_tokens(
            prompt_text, tokenizer, return_tensors="pt"
        )["input_ids"][0]
        tokens_chosen = tokenize_text_with_special_tokens(
            prompt_plus_chosen_response,
            tokenizer,
            return_tensors="pt",
        )
        tokens_rejected = tokenize_text_with_special_tokens(
            prompt_plus_rejected_response,
            tokenizer,
            return_tensors="pt",
        )

        if model_name:
            # add label mask for sft and dpo training
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[: len(prompt_ids)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[: len(prompt_ids)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,
                "label_rejected": label_rejected,
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(formatting_func, batched=False, keep_in_memory=True)
    remove_columns = []
    for col in ds.column_names:
        if "input" not in col and "attention" not in col and "label" not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


def load_train_eval_dataset(
    data_path, tokenizer, size=None, mode="", model_name="", seed=42
):
    dataset = build_dataset(
        data_path, tokenizer, split="train", size=size, model_name=model_name
    )
    dataset_split = dataset.train_test_split(
        test_size=0.05, seed=seed, shuffle=True
    )
    train_dataset, eval_dataset = dataset_split["train"], dataset_split["test"]
    return train_dataset, eval_dataset
