import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from peft import PeftModel, PeftConfig
import wandb
from rlhf.ppo.ppo_utils import post_process_common_dataset

@dataclass
class ScriptArguments:
    checkpoints_dir: str = field(
        default="", metadata={"help": "Directory containing policy checkpoints"}
    )
    training_rm_path: str = field(
        default="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40",
        metadata={"help": "Path to the reward model used during training"}
    )
    gold_rm_name: str = field(
        default="Ray2333/GRM-Gemma2-2B-rewardmodel-ft",
        metadata={"help": "Name of the gold reward model"}
    )
    dataset_name: str = field(
        default="gagan3012/helpsteer2-gold",
        metadata={"help": "Name of the dataset to evaluate on"}
    )
    base_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Base model name for LoRA models. Required if using LoRA checkpoints."}
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length for input processing"}
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate"}
    )
    batch_size: Optional[int] = field(default=8)
    device: Optional[str] = field(default="cuda")
    output_file: Optional[str] = field(default="evaluation_results.csv")
    num_responses_per_prompt: Optional[int] = field(
        default=1,
        metadata={"help": "Number of responses to generate per prompt"}
    )
    wandb_project: Optional[str] = field(
        default="policy-evaluation",
        metadata={"help": "WandB project name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name. If None, will use checkpoint directory name"}
    )
    disable_wandb: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable wandb logging"}
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "Debug mode - only use first 100 prompts"}
    )

def load_reward_model(model_path_or_name, device):
    """Load a reward model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()  # Ensure model is in evaluation mode
    return model, tokenizer

def get_reward_score(model, tokenizer, texts, device, batch_size=8):
    """Get reward scores for a batch of texts."""
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                return_dict=True,
                output_hidden_states=True,
            )
            scores = outputs.logits.squeeze(-1)
            all_scores.extend(scores.cpu().numpy())
    
    return np.array(all_scores)

def is_peft_model(model_path):
    """Check if the model at the given path is a PEFT model."""
    return os.path.exists(os.path.join(model_path, "adapter_config.json"))

def collate_batch(input_ids_list, attention_mask_list, tokenizer):
    """Collate and pad a batch of input sequences to the longest sequence in the batch."""
    max_length = max(len(ids) for ids in input_ids_list)
    
    padded_input_ids = []
    padded_attention_mask = []
    
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        padding_length = max_length - len(input_ids)
        padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * padding_length)
        padded_attention_mask.append(attention_mask + [0] * padding_length)
    
    return (
        torch.tensor(padded_input_ids),
        torch.tensor(padded_attention_mask)
    )

def generate_responses(model, tokenizer, input_ids_list, attention_mask_list, max_new_tokens=512, batch_size=8, num_responses=1):
    """Generate responses for a batch of input_ids."""
    all_responses = []
    
    for i in range(0, len(input_ids_list), batch_size):
        batch_input_ids = input_ids_list[i:i + batch_size]
        batch_attention_mask = attention_mask_list[i:i + batch_size]
        
        # Collate and pad the batch to the longest sequence in this batch
        batch_input_ids, batch_attention_mask = collate_batch(
            batch_input_ids,
            batch_attention_mask,
            tokenizer
        )
        
        # Move to device
        batch_input_ids = batch_input_ids.to(model.device)
        batch_attention_mask = batch_attention_mask.to(model.device)
        
        with torch.no_grad():
            # Generate multiple responses per prompt
            for _ in range(num_responses):
                outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1  # We handle multiple responses in the outer loop
                )
                
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_responses.extend(responses)
    
    return all_responses

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Initialize wandb if enabled
    if not args.disable_wandb:
        wandb_run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.checkpoints_dir))
        if args.debug:
            wandb_run_name += "_debug"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args)
        )
    
    # Load reward models
    print("Loading reward models...")
    training_rm, training_rm_tokenizer = load_reward_model(args.training_rm_path, args.device)
    gold_rm, gold_rm_tokenizer = load_reward_model(args.gold_rm_name, args.device)
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    dataset = load_dataset(args.dataset_name, split="test")
    if args.debug:
        print("Debug mode: using only first 100 prompts")
        dataset = dataset.select(range(min(100, len(dataset))))
    
    # Get all checkpoint directories
    checkpoints = sorted([
        d for d in os.listdir(args.checkpoints_dir)
        if d.startswith("checkpoint-")
    ], key=lambda x: int(x.split("-")[1]))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in directory: {args.checkpoints_dir}")
    
    # Check if we're dealing with LoRA checkpoints
    first_checkpoint_path = os.path.join(args.checkpoints_dir, checkpoints[0])
    if is_peft_model(first_checkpoint_path) and args.base_model_name is None:
        raise ValueError(
            "Found LoRA checkpoints but no base model specified. "
            "Please provide --base_model_name"
        )
    
    # Load tokenizer once since it's the same for all checkpoints
    print("Loading tokenizer...")
    tokenizer_path = args.base_model_name if args.base_model_name else first_checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Process dataset once with the tokenizer
    print("Processing dataset...")
    processed_dataset = post_process_common_dataset(dataset, tokenizer, args)
    print(f"Using {len(processed_dataset)} processed prompts for evaluation")
    
    # Convert input_ids and attention_mask to lists for processing
    input_ids_list = [ids.tolist() for ids in processed_dataset["input_ids"]]
    attention_mask_list = [mask.tolist() for mask in processed_dataset["attention_mask"]]
    
    if args.debug:
        print("Debug mode: using only first checkpoint")
        checkpoints = checkpoints[:1]
    
    results = []
    
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        checkpoint_path = os.path.join(args.checkpoints_dir, checkpoint)
        print(f"\nEvaluating {checkpoint}")
        
        try:
            # Load model (without tokenizer since we already have it)
            print(f"Loading {'LoRA' if args.base_model_name else 'full'} model from {checkpoint_path}")
            if args.base_model_name:
                # Load the base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model_name,
                    torch_dtype=torch.float16,
                    device_map=args.device
                )
                base_model.resize_token_embeddings(len(tokenizer))
                base_model.config.pad_token_id = tokenizer.pad_token_id
                
                # Load and apply the LoRA adapter
                model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map=args.device
                )
            else:
                # Load the full model directly
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map=args.device
                )
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id
            
            model.eval()
            
            # Generate responses using processed input_ids and attention_mask
            responses = generate_responses(
                model,
                tokenizer,
                input_ids_list,
                attention_mask_list,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                num_responses=args.num_responses_per_prompt
            )

            print(responses[:5])  # Print first 5 responses for debugging
            
            # Get reward scores
            training_rm_scores = get_reward_score(
                training_rm,
                training_rm_tokenizer,
                responses,
                args.device,
                args.batch_size
            )
            
            gold_rm_scores = get_reward_score(
                gold_rm,
                gold_rm_tokenizer,
                responses,
                args.device,
                args.batch_size
            )
            
            # Calculate statistics
            checkpoint_num = int(checkpoint.split("-")[1])
            checkpoint_results = {
                "checkpoint": checkpoint_num,
                "training_rm_mean": float(np.mean(training_rm_scores)),
                "training_rm_std": float(np.std(training_rm_scores)),
                "gold_rm_mean": float(np.mean(gold_rm_scores)),
                "gold_rm_std": float(np.std(gold_rm_scores))
            }
            
            # Log to wandb
            if not args.disable_wandb:
                wandb.log({
                    "checkpoint": checkpoint_num,
                    "training_rm/mean": checkpoint_results["training_rm_mean"],
                    "training_rm/std": checkpoint_results["training_rm_std"],
                    "gold_rm/mean": checkpoint_results["gold_rm_mean"],
                    "gold_rm/std": checkpoint_results["gold_rm_std"],
                    # Add histograms of scores
                    "training_rm/scores_hist": wandb.Histogram(training_rm_scores),
                    "gold_rm/scores_hist": wandb.Histogram(gold_rm_scores),
                })
            
            results.append(checkpoint_results)
            
        finally:
            # Free up memory
            if 'model' in locals():
                del model
            if 'base_model' in locals():
                del base_model
            torch.cuda.empty_cache()
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        if args.debug:
            args.output_file = args.output_file.replace(".csv", "_debug.csv")
        results_df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
    else:
        print("\nNo results were generated!")
    
    # Close wandb
    if not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 