import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from data_utils import (
    format_and_validate_preference_sample,
    tokenize_text_with_special_tokens,
    get_length_config,
)


# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(
    data_path,
    tokenizer,
    split="train",
    size=None,
    model_name="",
    *,
    length_config,
    skip_validation=False,
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
                length_config=length_config,
                context="RM",
                skip_validation=skip_validation,
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
                "chosen_ids": tokens_chosen["input_ids"][0],
                "rejected_ids": tokens_rejected["input_ids"][0],
                "label_chosen": label_chosen,
                "label_rejected": label_rejected,
            }
        else:
            return {
                "chosen_ids": tokens_chosen["input_ids"][0],
                "rejected_ids": tokens_rejected["input_ids"][0],
            }

    ds = ds.map(formatting_func, batched=False, keep_in_memory=True, num_proc=16)
    remove_columns = []
    for col in ds.column_names:
        if col not in ("chosen_ids", "rejected_ids", "label_chosen", "label_rejected", "margin"):
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


def load_train_eval_dataset(
    data_path, tokenizer, size=None, model_name="", seed=42, *, length_config, skip_validation=False,
):
    dataset = build_dataset(
        data_path, tokenizer, split="train", size=size, model_name=model_name,
        length_config=length_config,
        skip_validation=skip_validation,
    )
    dataset_split = dataset.train_test_split(
        test_size=0.05, seed=seed, shuffle=True
    )
    train_dataset, eval_dataset = dataset_split["train"], dataset_split["test"]
    return train_dataset, eval_dataset
