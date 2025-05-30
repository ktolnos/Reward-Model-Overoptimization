import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import numpy as np
import pandas as pd
tqdm.pandas()
import matplotlib.pyplot as plt


def build_train_eval_datasets(data_path_train, tokenizer, eval_proportion, size=None):
    ds = datasets.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)
    ds_train = ds_dict['train']
    ds_eval = ds_dict['test']
    ds_train = post_process_common_dataset(ds_train, tokenizer)
    ds_eval = post_process_common_dataset(ds_eval, tokenizer)
    return ds_train, ds_eval


def build_dataset_common(data_path, tokenizer, split='', size=None):
    ds = datasets.load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    ds = post_process_common_dataset(ds, tokenizer)
    return ds

def post_process_common_dataset(ds, tokenizer):
    def formatting_func(example):
        messages = example['chosen'][:-1]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False),
        }

    ds = ds.map(formatting_func,
                remove_columns=ds.column_names,
                batched=False, num_proc=30)
    ds.set_format(type="torch")
    return ds
