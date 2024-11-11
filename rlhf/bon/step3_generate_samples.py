from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import torch
import time
from tqdm.auto import tqdm
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

from utils import create_output_directory, save_results_in_parquet_splits
from load_datasets import load_data2generate

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ScriptArguments:
    batch_size: int = field(default=128, metadata={"help": "Batch size for inference"})
    max_new_tokens: int = field(default=1024, metadata={"help": "Maximum number of new tokens to generate"})
    N: int = field(default=405, metadata={"help": "Number of dataset duplications"})
    data_path: str = field(default='./data/unified_2k', metadata={"help": "Path to the dataset"})
    model_path: str = field(default='google/gemma-2b-it', metadata={"help": "Path to the policy model checkpoint"})
    save_path: Optional[str] = field(default='./step3_generate_samples', metadata={"help": "Directory to save results."})
    save_name: Optional[str] = field(default='generated_samples_unified', metadata={"help": "Saved file name."})
    num_splits: int = field(default=6, metadata={"help": "Number of splits for saving results"})

def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="Script for generating responses using a Hugging Face model with distributed acceleration.")
    for field_name, field_def in ScriptArguments.__dataclass_fields__.items():
        parser.add_argument(
            f"--{field_name}",
            type=type(field_def.default),
            default=field_def.default,
            help=field_def.metadata.get("help", "")
        )
    args = parser.parse_args()
    return ScriptArguments(**vars(args))




# Function to perform inference
def inference(model, tokenizer, dataset, batch_size, max_new_tokens):
    results = []
    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating responses"):
        batch = dataset[i:i + batch_size]
        prompts = {k: torch.tensor(v, device=model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        
        with torch.no_grad():
            outputs = model.module.generate(
                **prompts,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, 
                top_k=0.0,
                temperature=0.7,
                top_p=0.95
            )
        
        # Remove prompt from generated tokens
        outputs = [tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts["input_ids"], outputs)]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for idx, output in enumerate(decoded_outputs):
            results.append({
                'id': batch['id'][idx],
                'source': batch['source'][idx],
                'input': batch['input'][idx],
                'output': output
            })

    return results



def generate_samples():
    # Parse arguments
    script_args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.save_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(script_args.model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = accelerator.prepare(model)

    # Load and process dataset
    duplicated_dataset = load_data2generate(script_args.data_path, tokenizer, script_args.N)

    # Inference
    results = inference(model, tokenizer, duplicated_dataset, script_args.batch_size, script_args.max_new_tokens)

    # Save results
    if accelerator.is_main_process:
        save_results_in_parquet_splits(results, num_splits=script_args.num_splits, save_path=output_dir, mode='test')

# Run main function
if __name__ == "__main__":
    generate_samples()
