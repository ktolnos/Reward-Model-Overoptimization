import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import numpy as np
import pandas as pd
tqdm.pandas()
import matplotlib.pyplot as plt


def build_train_eval_datasets(data_path_train, tokenizer, eval_proportion, size=None, max_length=512):
    ds = datasets.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)
    ds_train = ds_dict['train']
    ds_eval = ds_dict['test']
    ds_train = post_process_common_dataset(ds_train, tokenizer, max_length)
    ds_eval = post_process_common_dataset(ds_eval, tokenizer, max_length)
    return ds_train, ds_eval


def build_dataset_common(data_path, tokenizer, split='', size=None):
    ds = datasets.load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    ds = post_process_common_dataset(ds, tokenizer)
    return ds

def post_process_common_dataset(ds, tokenizer, max_length):
    def formatting_func(example):
        messages = example['chosen'][:-1]
        chat = tokenizer.apply_chat_template(messages, tokenize=True,
                                             add_generation_prompt=True,
                                             enable_thinking=False,
                                             truncation=True,
                                             max_length=max_length,
                                             )
        prompt = tokenizer.decode(chat, skip_special_tokens=False) # This way we limit max_length
        return {
            "prompt": prompt,
        }

    ds = ds.map(formatting_func,
                remove_columns=ds.column_names,
                batched=False, num_proc=30)
    ds.set_format(type="torch")
    return ds
