import os
import random
import requests
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
from sympy import false
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
import wandb

from rlhf.ppo.ppo_utils import post_process_common_dataset
from experimental.dataset_annotation import load_reward_model
from reward_utils import Skywork_PROMPT, Skywork_SYSTEM_PROMPT, Skywork_ASSISTANT_PROMPT, extract_reward_from_response

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
    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length for input processing"}
    )
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate"}
    )
    batch_size: Optional[int] = field(default=8)
    generation_batch_size: Optional[int] = field(default=8)
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
    evaluate_with_training_rm: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to evaluate with the training reward model"}
    )
    evaluate_with_llm_judge: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use an LLM as a judge for evaluation"}
    )
    llm_judge_model_name: Optional[str] = field(
        default="google/gemma-7b-it",
        metadata={"help": "Name of the LLM judge model on OpenRouter"}
    )
    openrouter_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "OpenRouter API key. If not provided, tries to use OPENROUTER_API_KEY env var."}
    )
    baseline_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the baseline model (Hugging Face model or checkpoint) for LLM judge comparison."}
    )
    use_dataset_response_as_baseline: Optional[bool] = field(
        default=False,
        metadata={"help": "Use the 'response' column from the dataset as the baseline."}
    )
    llm_judge_max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "Max new tokens for the LLM judge."}
    )
    save_eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the full evaluation dataset with all responses and verdicts (e.g., eval_results.jsonl)."}
    )
    subsample_n: Optional[int] = field(
        default=None,
        metadata={"help": "Number of prompts to randomly subsample from the dataset. If None, uses the full dataset."}
    )

def load_reward_model_impl(model_path_or_name, device):
    model, tokenizer = load_reward_model(model_path_or_name, reasoning=False, device=device)
    return  model, tokenizer

def get_reward_score(model, tokenizer, texts, device, batch_size=8):
    """Get reward scores for a batch of texts."""
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
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


def collate_batch(input_ids_list, attention_mask_list, tokenizer):
    """Collate and pad a batch of input sequences to the longest sequence in the batch."""
    if len(input_ids_list) <= 1:
        # No padding needed for single item
        return (
            torch.tensor(input_ids_list),
            torch.tensor(attention_mask_list)
        )
    max_length = max(len(ids) for ids in input_ids_list)
    
    padded_input_ids = []
    padded_attention_mask = []
    
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        padding_length = max_length - len(input_ids)
        padded_input_ids.append([tokenizer.pad_token_id] * padding_length + input_ids)
        padded_attention_mask.append([0] * padding_length + attention_mask)
    
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

def load_policy_model(model_path, tokenizer, device):
    """Loads a policy model from a path."""
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model

def get_llm_judge_verdicts(
    prompts: List[str],
    responses1: List[str],
    responses2: List[str],
    args: ScriptArguments,
) -> Tuple[List[int], List[str]]:
    """
    Gets verdicts from an LLM judge for pairs of responses.
    Returns a list of preferences: 1 if response1 is better, -1 if response2 is better, 0 for a tie.
    """
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key must be provided via --openrouter_api_key or OPENROUTER_API_KEY env var.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000", # Optional, but good practice
        "X-Title": "Reward Model Overoptimization" # Optional, but good practice
    }
    
    all_preferences = []
    all_responses = []
    
    # This is a simplified sequential implementation.
    # For higher throughput, you might consider concurrent requests.
    for i in tqdm(range(len(prompts)), desc="Querying LLM Judge"):
        prompt = prompts[i]
        resp1 = responses1[i]
        resp2 = responses2[i]

        # Randomly swap to mitigate position bias
        swap = random.random() > 0.5
        answer1, answer2 = (resp2, resp1) if swap else (resp1, resp2)
        
        user_prompt = Skywork_PROMPT.format(question=prompt, answer1=answer1, answer2=answer2)
        
        payload = {
            "model": args.llm_judge_model_name,
            "messages": [
                {"role": "system", "content": Skywork_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": Skywork_ASSISTANT_PROMPT},
            ],
            "max_tokens": args.llm_judge_max_new_tokens,
            "temperature": 0,
            "top_p": 0.9,
            # "providers": {
            #     "order": ["targon", "chutes/fp8"]
            # }
        }
        
        retries = 5
        backoff_factor = 2
        for attempt in range(retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                generated_text = result['choices'][0]['message']['content']
                all_responses.append(generated_text)

                preference = extract_reward_from_response(generated_text)
                if swap:
                    preference *= -1 # un-swap

                all_preferences.append(preference)
                break  # Success, exit retry loop

            except BaseException as e:
                if isinstance(e, requests.exceptions.HTTPError) and (e.response.status_code in (429, 500, 502)) or isinstance(e, KeyError): # sometimes server returns error in the body
                    if attempt < retries - 1:
                        # Try to get the specific wait time from the 'Retry-After' header
                        retry_after_header = e.response.headers.get("Retry-After") if hasattr(e, "response") else None
                        if retry_after_header:
                            try:
                                sleep_time = int(retry_after_header) + 1 # Add 1s buffer
                                print(f"Rate limit exceeded. Following server's 'Retry-After' header. Waiting for {sleep_time} seconds.")
                            except ValueError:
                                # If the header is a date, this will fail. Fallback to exponential backoff.
                                sleep_time = backoff_factor * (10 ** attempt) + random.uniform(0, 1)
                                print(f"Rate limit exceeded. Could not parse 'Retry-After' header. Retrying in {sleep_time:.2f} seconds...")
                        else:
                            # Fallback to exponential backoff if the header is not present
                            sleep_time = backoff_factor * (10 ** attempt) + random.uniform(0, 1)
                            print(f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds (exponential backoff)...")
                        
                        time.sleep(sleep_time)
                    else:
                        print(f"Error querying LLM Judge after multiple retries: {e}")
                        all_preferences.append(0) # Default to tie on error
                        all_responses.append(f"Error querying LLM Judge: {e}")
                else:
                    print(f"Error querying LLM Judge: {e}\n\nlocals:\n{locals()}")
                    all_preferences.append(0) # Default to tie on error
                    all_responses.append(f"Error querying LLM Judge: {e}\n\nlocals:\n{locals()}")
                    break # Don't retry for other errors
    
    return all_preferences, all_responses

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
    
    # --- Common Setup ---
    print("Loading evaluation dataset...")
    split = "test" if args.dataset_name.startswith("/") else "train"
    dataset = load_dataset(args.dataset_name, split=split)
    if args.debug:
        print("Debug mode: using only first 100 prompts")
        dataset = dataset.select(range(min(100, len(dataset))))
    elif args.subsample_n is not None:
        if args.subsample_n > len(dataset):
            print(f"Warning: subsample_n ({args.subsample_n}) is larger than the dataset size ({len(dataset)}). Using the full dataset.")
        else:
            dataset = dataset.shuffle(seed=42).select(range(args.subsample_n))
            print(f"Subsampling to {args.subsample_n} prompts.")

    checkpoints = sorted([
        d for d in os.listdir(args.checkpoints_dir)
        if d.startswith("checkpoint-")
    ], key=lambda x: int(x.split("-")[1]))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in directory: {args.checkpoints_dir}")
    
    first_checkpoint_path = os.path.join(args.checkpoints_dir, checkpoints[0])
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(first_checkpoint_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Prepare dataset based on its structure
    if 'prompt' in dataset.column_names:
        print("Using 'prompt' column for prompts.")
        original_prompts = dataset['prompt']
        processed_dataset = post_process_common_dataset(dataset, tokenizer, args)
    elif 'chosen' in dataset.column_names:
        print("Using 'chosen' column for prompts and responses.")
        def extract_prompt_from_chosen(example):
            # The prompt is the conversation up to the last turn.
            return {'prompt': tokenizer.apply_chat_template(example['chosen'][:-1],
                                                            tokenize=False,
                                                            add_generation_prompt=True,
                                                            enable_thinking=False)}
        dataset = dataset.map(extract_prompt_from_chosen)
        original_prompts = dataset['prompt']
        processed_dataset = post_process_common_dataset(dataset, tokenizer, args)
    else:
        raise ValueError("Dataset must have a 'prompt' or 'chosen' column.")

    print(f"Using {len(processed_dataset)} processed prompts for evaluation")
    
    input_ids_list = [ids.tolist() for ids in processed_dataset["input_ids"]]
    attention_mask_list = [mask.tolist() for mask in processed_dataset["attention_mask"]]
    
    if args.debug:
        print("Debug mode: using only first checkpoint")
        checkpoints = checkpoints[:1]
    
    results = []
    full_eval_data = []

    if args.evaluate_with_llm_judge:
        # --- LLM-as-Judge Evaluation ---
        print("Starting LLM-as-Judge evaluation...")
        
        if not args.baseline_model_path and not args.use_dataset_response_as_baseline:
            raise ValueError(
                "Either --baseline_model_path or --use_dataset_response_as_baseline must be specified for LLM judge evaluation."
            )

        baseline_responses = []
        if args.use_dataset_response_as_baseline:
            print("Using dataset 'chosen' column as baseline response.")
            if 'chosen' not in dataset.column_names:
                raise ValueError("Dataset must have a 'chosen' column to use it as a baseline.")
            def extract_response_from_chosen(example):
                return {'response': example['chosen'][-1]['content']}
            baseline_responses = dataset.map(extract_response_from_chosen)['response']
        else:
            baseline_model = load_policy_model(
                args.baseline_model_path,
                tokenizer,
                args.device
            )
            print("Generating baseline responses...")
            baseline_responses = generate_responses(
                baseline_model,
                tokenizer,
                input_ids_list,
                attention_mask_list,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
                num_responses=1
            )
            del baseline_model
            torch.cuda.empty_cache()

        # Initialize the full_eval_data list
        for i in range(len(original_prompts)):
            full_eval_data.append({
                "prompt": original_prompts[i],
                "baseline_response": baseline_responses[i],
                "checkpoints": {}
            })

        for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints with LLM Judge"):
            checkpoint_path = os.path.join(args.checkpoints_dir, checkpoint)
            checkpoint_num = int(checkpoint.split("-")[1])
            print(f"\nEvaluating {checkpoint}")
            
            try:
                model = load_policy_model(
                    checkpoint_path,
                    tokenizer,
                    args.device
                )
                
                policy_responses = generate_responses(
                    model,
                    tokenizer,
                    input_ids_list,
                    attention_mask_list,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.generation_batch_size,
                    num_responses=args.num_responses_per_prompt
                )
                
                verdicts, judge_responses = get_llm_judge_verdicts(
                    original_prompts,
                    policy_responses,
                    baseline_responses,
                    args
                )
                
                for i in range(len(original_prompts)):
                    full_eval_data[i]["checkpoints"][checkpoint_num] = {
                        "policy_response": policy_responses[i],
                        "llm_judge_response": judge_responses[i],
                        "llm_judge_verdict": verdicts[i]
                    }

                wins = verdicts.count(1)
                losses = verdicts.count(-1)
                ties = verdicts.count(0)
                total = len(verdicts)
                
                checkpoint_results = {
                    "checkpoint": checkpoint_num,
                    "win_rate": wins / total if total > 0 else 0,
                    "loss_rate": losses / total if total > 0 else 0,
                    "tie_rate": ties / total if total > 0 else 0,
                    "mean": np.mean(verdicts) if total > 0 else 0,
                    "total_comparisons": total,
                }

                if not args.disable_wandb:
                    wandb_log_data = {f"llm_judge/{k}": v for k, v in checkpoint_results.items() if k != 'checkpoint'}
                    wandb_log_data['checkpoint'] = checkpoint_num
                    wandb.log(wandb_log_data)

                results.append(checkpoint_results)
            except Exception as e:
                print(f"Error evaluating checkpoint {checkpoint}: {e}")
                # Log an empty result for this checkpoint
                results.append({
                    "checkpoint": checkpoint_num,
                    "win_rate": None,
                    "loss_rate": None,
                    "tie_rate": None,
                    "total_comparisons": 0
                })
                continue

            finally:
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()

    else:
        # --- Reward Model Evaluation (Original Logic) ---
        print("Starting Reward Model evaluation...")
        print("Loading reward models...")
        if args.evaluate_with_training_rm:
            training_rm, training_rm_tokenizer = load_reward_model_impl(args.training_rm_path, args.device)
        gold_rm, gold_rm_tokenizer = load_reward_model_impl(args.gold_rm_name, args.device)

        for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
            checkpoint_path = os.path.join(args.checkpoints_dir, checkpoint)
            print(f"\nEvaluating {checkpoint}")
            
            try:
                model = load_policy_model(
                    checkpoint_path,
                    tokenizer,
                    args.device
                )
                
                responses = generate_responses(
                    model,
                    tokenizer,
                    input_ids_list,
                    attention_mask_list,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.generation_batch_size,
                    num_responses=args.num_responses_per_prompt
                )

                print(responses[:5])

                if args.evaluate_with_training_rm:
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
                
                checkpoint_num = int(checkpoint.split("-")[1])
                checkpoint_results = {
                    "checkpoint": checkpoint_num,
                    "gold_rm/mean": float(np.mean(gold_rm_scores)),
                    "gold_rm/std": float(np.std(gold_rm_scores)),
                }
                if not args.disable_wandb:
                    checkpoint_results["gold_rm/scores_hist"] = wandb.Histogram(gold_rm_scores)

                if args.evaluate_with_training_rm:
                    checkpoint_results["training_rm/mean"] = float(np.mean(training_rm_scores))
                    checkpoint_results["training_rm/std"] = float(np.std(training_rm_scores))
                    if not args.disable_wandb:
                        checkpoint_results["training_rm/scores_hist"] = wandb.Histogram(training_rm_scores)

                if not args.disable_wandb:
                    wandb.log(checkpoint_results)

                if "gold_rm/scores_hist" in checkpoint_results:
                    del checkpoint_results["gold_rm/scores_hist"]
                if "training_rm/scores_hist" in checkpoint_results:
                    del checkpoint_results["training_rm/scores_hist"]
                results.append(checkpoint_results)
                
            finally:
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()

    if results:
        results_df = pd.DataFrame(results)
        if args.debug:
            args.output_file = args.output_file.replace(".csv", "_debug.csv")
        results_df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
    else:
        print("\nNo results were generated!")
    
    if args.save_eval_dataset_path and full_eval_data:
        with open(args.save_eval_dataset_path, 'w') as f:
            for item in full_eval_data:
                f.write(json.dumps(item) + "\n")
        print(f"Full evaluation data saved to {args.save_eval_dataset_path}")

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()