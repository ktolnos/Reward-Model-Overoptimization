import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from utils import create_output_directory
from load_datasets import build_datasets_inference, prepare_data_loader
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


@dataclass
class ScriptArguments:
    per_device_batch_size: Optional[int] = field(default=64, metadata={"help": "The batch size per device during evaluation."})
    max_length: Optional[int] = field(default=1024, metadata={"help": "The maximum sequence length."})
    data_path: Optional[str] = field(default="./data/unified_20k", metadata={"help": "Path to the data file."})
    model_path: Optional[str] = field(default="Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback", metadata={"help": "The gold reward model to use."})
    save_path: Optional[str] = field(default='./step1_obtain_gold_score', metadata={"help": "Directory to save results."})
    save_name: Optional[str] = field(default="unified_sampled_gold_score", metadata={"help": "Saved file name."})
    mode: Optional[str] = field(default="train", metadata={"help": "'train', and 'test'"})
    

def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="Set parameters for model training & evaluation.")
    for field_name, field_def in ScriptArguments.__dataclass_fields__.items():
        parser.add_argument(
            f"--{field_name}",
            type=type(field_def.default),
            default=field_def.default,
            help=field_def.metadata.get("help", "")
        )
    args = parser.parse_args()
    return ScriptArguments(**vars(args))



def generate_and_collect_results(model, data_loader, tokenizer):
    full_chosen_prompts, full_rejected_prompts = [], []
    full_rewards_chosen, full_rewards_rejected = [], []
    full_source_ids = []

    pbar = tqdm(total=len(data_loader))
    with torch.no_grad():
        for batch in data_loader:
            reward_chosen_tensors = model(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask_chosen"].to(model.device)).logits.reshape(-1)
            reward_rejected_tensors = model(batch["input_ids_rejected"].to(model.device), attention_mask=batch["attention_mask_rejected"].to(model.device)).logits.reshape(-1)

            full_rewards_chosen.extend(reward_chosen_tensors.cpu().tolist())
            full_rewards_rejected.extend(reward_rejected_tensors.cpu().tolist())
            full_chosen_prompts.extend(batch['input_ids'])
            full_rejected_prompts.extend(batch['input_ids_rejected'])
            if 'source_id' in batch:
                full_source_ids.extend(batch['source_id'])

            pbar.update(1)

    # Decode and organize results
    return {
        'prompts_A': tokenizer.batch_decode(full_chosen_prompts),
        'prompts_B': tokenizer.batch_decode(full_rejected_prompts),
        'rewards_A': full_rewards_chosen,
        'rewards_B': full_rewards_rejected,
        'source_ids': [x.item() for x in full_source_ids] if 'source_id' in batch else []
    }


# Main function
def obtain_gold_score():
    # Parse arguments
    script_args = parse_args()
  
    # Initialize Accelerator
    accelerator = Accelerator()

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.save_name)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(script_args.model_path, num_labels=1, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.model_max_length = script_args.max_length

    # Prepare dataset and DataLoader
    dataset = build_datasets_inference(script_args.data_path, tokenizer, split=script_args.mode, max_length=script_args.max_length)
    data_loader = prepare_data_loader(dataset, tokenizer, script_args.per_device_batch_size)
    data_loader = accelerator.prepare(data_loader)

    # Generate and collect results
    evaluation_result = generate_and_collect_results(model, data_loader, tokenizer)

    # Save results to CSV
    if accelerator.is_main_process:
        dataframe = pd.DataFrame(evaluation_result)
        dataframe.to_csv(os.path.join(output_dir, 'gold_score_%s.csv'%script_args.mode))



if __name__ == "__main__":
    obtain_gold_score()

