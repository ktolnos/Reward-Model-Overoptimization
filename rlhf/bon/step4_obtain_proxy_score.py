from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import sys
import torch
import numpy as np
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import argparse
from load_datasets import build_datasets_inference, prepare_data_loader
from utils import create_output_directory

# Add the `./reward_models` path to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reward_models')))
from grm_utils import load_model_withhead



@dataclass
class ScriptArguments:
    per_device_batch_size: Optional[int] = field(default=64, metadata={"help": "The batch size per device during evaluation."})
    max_length: Optional[int] = field(default=1024, metadata={"help": "The maximum sequence length."})
    data_path: Optional[str] = field(default="./step3_generate_samples/generated_samples_unified", metadata={"help": "Path to the data file."})
    model_type: Optional[str] = field(default="grm", metadata={'help': "use 'grm', 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    base_model: Optional[str] = field(default="google/gemma-2b-it", metadata={"help": "Path to the pre-trained model."})
    peft_name: Optional[str] = field(default="./step2_train_proxy_reward_model/gemma-2b-it", metadata={"help": "PEFT model name or directory if using PEFT."})
    save_path: Optional[str] = field(default='./step4_obtain_proxy_score/gemma-2b-it', metadata={"help": "Directory to save results."})
    # Only for GRM
    layer_type: Optional[str] = field(default='linear') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    


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


# Evaluation function
def evaluate_and_collect_results(model, data_loader, tokenizer, accelerator, batch_size: int) -> Dict[str, List]:
    """Evaluate and return results."""
    full_prompts, full_rewards, full_source_ids, full_id_ids = [], [], [], []
    pbar = tqdm(total=len(data_loader) * batch_size // accelerator.num_processes)
    
    with torch.no_grad():
        for batch in data_loader:
            reward_tensors = model(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask"].to(model.device)).logits.reshape(-1)
            full_rewards.extend(reward_tensors.cpu().numpy())
            full_prompts.extend(batch['input_ids'])
            full_source_ids.extend(batch['source'])
            full_id_ids.extend(batch['id'])
            pbar.update(1)
    
    accelerator.wait_for_everyone()

    full_prompts = [x.rstrip('</s>') for x in tokenizer.batch_decode(full_prompts)]
    return {
        'prompts': full_prompts,
        'rewards': [float(x) for x in full_rewards],
        'source_ids': full_source_ids,
        'id_ids': full_id_ids
    }


# Main execution logic
def obtain_proxy_score():
    # Parse arguments
    script_args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.model_type)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
    tokenizer.model_max_length = script_args.max_length
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset and DataLoader
    dataset = build_datasets_inference(script_args.data_path, tokenizer, split='test', max_length=script_args.max_length)
    data_loader = prepare_data_loader(dataset, tokenizer, script_args.per_device_batch_size, collate_fn_type='custom')
    data_loader = accelerator.prepare(data_loader)

    # Load model
    if script_args.model_type == 'grm':
        model = load_model_withhead(script_args.base_model, script_args.peft_name, tokenizer, device=accelerator.local_process_index, layer_type=script_args.layer_type, num_layers=script_args.num_layers)
    elif script_args.model_type in ['bt', 'margin', 'labelsmooth', 'pos_reg']:
        model = AutoModelForSequenceClassification.from_pretrained(script_args.base_model, num_labels=1, device_map=accelerator.local_process_index, torch_dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        if os.path.exists(script_args.peft_name):
            model = PeftModel.from_pretrained(model, script_args.peft_name)
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()


    # Run evaluation and gather results
    evaluation_result = evaluate_and_collect_results(model, data_loader, tokenizer, accelerator, script_args.per_device_batch_size)
    
    # Save results to CSV
    if accelerator.is_main_process:
        df = pd.DataFrame(evaluation_result)
        df.to_csv(os.path.join(output_dir, 'proxy_score.csv'), index=False)


if __name__ == "__main__":
    obtain_proxy_score()
