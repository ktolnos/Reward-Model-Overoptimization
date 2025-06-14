"""
Dataset annotation script for loading HelpSteer2 preference dataset,
evaluating it with a reward model, and saving the augmented dataset.
"""

import os
from dataclasses import field, dataclass

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from tqdm import tqdm
import json
import random
from reward_utils import Skywork_SYSTEM_PROMPT, Skywork_PROMPT, Skywork_ASSISTANT_PROMPT, extract_reward_from_response

def load_helpsteer2_dataset(split="train"):
    """
    Load the HelpSteer2 preference dataset from Hugging Face.
    
    Args:
        split (str): Dataset split to load ('train', 'validation', or 'test')
        
    Returns:
        dataset: HuggingFace dataset object
    """
    dataset = load_dataset("gagan3012/helpsteer2-preference-v2", split=split)
    print(f"Loaded {len(dataset)} examples from {split} split")
    return dataset


def load_reward_model(model_name, reasoning, device=None):
    """
    Load the reward model from Hugging Face.
    
    Args:
        model_name (str): Name of the reward model on Hugging Face
        device (str): Device to load the model on ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        tuple: (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading reward model {model_name} on {device}")
    if reasoning:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2",
                                                     trust_remote_code=True,
                                                     device_map=device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   torch_dtype=torch.float16,
                                                                   attn_implementation="flash_attention_2",
                                                                   device_map=device, trust_remote_code=True)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()
    return model, tokenizer


def evaluate_with_reward_model(dataset, model, tokenizer, batch_size=8, max_length=1024, device=None):
    """
    Evaluate dataset examples with the reward model.
    
    Args:
        dataset: HuggingFace dataset to evaluate
        model: Reward model
        tokenizer: Tokenizer for the reward model
        batch_size (int): Batch size for evaluation
        max_length (int): Maximum sequence length
        device (str): Device to run evaluation on ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        list: List of dictionaries with chosen and rejected rewards
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating with reward model"):
        batch = dataset[i:i + batch_size]

        # Process all examples in the batch at once
        all_conversations = []
        batch_size_actual = len(batch["preference_strength"])  # Get actual batch size

        # Add both chosen and rejected conversations to the batch
        for j in range(batch_size_actual):
            all_conversations.append(batch["chosen"][j])
            all_conversations.append(batch["rejected"][j])

        # Apply chat template to format all conversations
        formatted_texts = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in all_conversations]

        # Tokenize all conversations in a single batch
        inputs = tokenizer(
            formatted_texts,
            padding='longest',
            truncation=False,
            max_length=max_length,
            return_tensors="pt",
            padding_side="left",
        ).to(device)

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model(**inputs)
            all_rewards = outputs.logits.squeeze(-1).cpu().numpy()

        # Process the results
        for j in range(batch_size_actual):
            chosen_idx = j * 2  # Even indices are chosen responses
            rejected_idx = j * 2 + 1  # Odd indices are rejected responses

            chosen_reward = float(all_rewards[chosen_idx])
            rejected_reward = float(all_rewards[rejected_idx])
            does_gold_agree_with_original = chosen_reward > rejected_reward
            if does_gold_agree_with_original:
                chosen = batch["chosen"][j]
                rejected = batch["rejected"][j]
            else:
                chosen = batch["rejected"][j]
                rejected = batch["chosen"][j]
                chosen_reward, rejected_reward = rejected_reward, chosen_reward

            results.append({
                "preference_strength": batch["preference_strength"][j],
                "chosen": chosen,
                "rejected": rejected,
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
                "does_gold_agree_with_original": does_gold_agree_with_original,
            })

    return results


def evaluate_with_reasoning_reward_model(dataset, model, tokenizer, batch_size=8, max_length=1024, device=None):
    """
    Evaluate dataset examples with the reward model.

    Args:
        dataset: HuggingFace dataset to evaluate
        model: Reward model
        tokenizer: Tokenizer for the reward model
        batch_size (int): Batch size for evaluation
        max_length (int): Maximum sequence length
        device (str): Device to run evaluation on ('cuda', 'cpu', or None for auto-detection)

    Returns:
        list: List of dictionaries with chosen and rejected rewards
    """
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating with reward model"):
        prompts = []
        swaps = []
        for j in range(min(len(dataset) - i, batch_size)):
            sample = dataset[i+j]
            query = sample["chosen"][:-1]
            answer1 = sample["chosen"][-1:]
            answer2 = sample["rejected"][-1:]

            query = tokenizer.apply_chat_template(query, tokenize=False)
            answer1 = tokenizer.apply_chat_template(answer1, tokenize=False)
            answer2 = tokenizer.apply_chat_template(answer2, tokenize=False)

            swap = random.random() > 0.5
            swaps.append(swap)  # Store whether we swapped answers for this sample

            if swap:
                answer1, answer2 = answer2, answer1  # Randomly swap answers to avoid bias

            system_prompt = Skywork_SYSTEM_PROMPT

            user_prompt = Skywork_PROMPT.format(
                question=query, answer1=answer1, answer2=answer2
            ) + Skywork_ASSISTANT_PROMPT

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        inputs = tokenizer(prompts,
                           padding='longest',
                           truncation=True,
                           max_length=max_length,
                           return_tensors="pt",
                           padding_side="left",
       ).to(model.device)

        generation_args = {
            "max_new_tokens": 8192,
            "temperature": 0.6,
            "do_sample": True,
            "top_p": 1.0,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)

        print("outputs\n\n", outputs)

        for j, swap in zip(range(len(swaps)), swaps):
            sample = dataset[i + j]
            generated_text = tokenizer.decode(outputs[j], skip_special_tokens=True)

            reward = extract_reward_from_response(generated_text)
            print(reward)
            if swap:
                reward = -reward

            chosen_reward = reward
            rejected_reward = -reward

            does_gold_agree_with_original = chosen_reward > rejected_reward
            if does_gold_agree_with_original:
                chosen = sample["chosen"]
                rejected = sample["rejected"]
            else:
                chosen = sample["rejected"]
                rejected = sample["chosen"]
                chosen_reward, rejected_reward = rejected_reward, chosen_reward

            results.append({
                "preference_strength": sample["preference_strength"],
                "chosen": chosen,
                "rejected": rejected,
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
                "does_gold_agree_with_original": does_gold_agree_with_original,
                "generated_text": generated_text,
            })

    return results




def save_annotated_dataset(results, output_path="data/helpsteer2_gold.json"):
    """
    Save the annotated dataset to disk.
    
    Args:
        results (list): List of dictionaries with evaluation results
        output_path (str): Path to save the dataset
        
    Returns:
        str: Path to the saved dataset
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Calculate some statistics
    reward_gap = df["chosen_reward"] - df["rejected_reward"]
    accuracy = df["does_gold_agree_with_original"].mean()

    print(f"Reward model accuracy: {accuracy:.4f}")
    print(f"Average reward gap: {reward_gap.mean():.4f}")
    print(f"Min reward gap: {reward_gap.min():.4f}")
    print(f"Max reward gap: {reward_gap.max():.4f}")

    # Save as JSON
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Saved annotated dataset to {output_path}")
    return output_path


def load_annotated_dataset(input_path="data/helpsteer2_gold/train.json"):
    """
    Load the annotated dataset from disk.
    
    Args:
        input_path (str): Path to the saved dataset
        
    Returns:
        list: List of dictionaries with evaluation results
    """
    with open(input_path, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} examples from {input_path}")
    return results


def annotate_dataset(model_name,
                     batch_size, max_length, output_path,
                     reasoning=False,  # If True, use reasoning reward model
                     dataset=None,
                     test_split_size=0.05):
    """
    Main function to load dataset, evaluate with reward model, and save results.
    
    Args:
        split (str): Dataset split to load (used only if custom_dataset is None)
        model_name (str): Name of the reward model
        batch_size (int): Batch size for evaluation
        max_length (int): Maximum sequence length
        output_path (str): Path to save the dataset
        custom_dataset: Optional pre-loaded dataset to use instead of loading from HF
        
    Returns:
        str: Path to the saved dataset
    """
    # Load reward model
    model, tokenizer = load_reward_model(model_name, reasoning, device="cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate dataset
    if reasoning:
        results = evaluate_with_reasoning_reward_model(dataset, model, tokenizer, batch_size, max_length)
    else:
        results = evaluate_with_reward_model(dataset, model, tokenizer, batch_size, max_length)

    # split dataset if needed
    if test_split_size > 0:
        # Split the dataset into train and test sets
        train_size = int((1 - test_split_size) * len(results))
        train_results = results[:train_size]
        test_results = results[train_size:]

        # Save the test set separately
        test_output_path = output_path + "test.json"
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        save_annotated_dataset(test_results, test_output_path)
        results = train_results
        output_path += "train.json"

    # make dirs for output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results
    return save_annotated_dataset(results, output_path)

@dataclass
class ScriptArguments:
    model_name: str = field(default="Reward-Reasoning/RRM-7B",
                            metadata={"help": "Name of the reward model"})
    batch_size: int = field(default=16, metadata={"help": "Batch size for evaluation"})
    max_length: int = field(default=4096, metadata={"help": "Maximum sequence length"})
    output_path: str = field(default="data/helpsteer2_gold/", metadata={"help": "Path to save the dataset"})
    reasoning: bool = field(default=True, metadata={"help": "If True, use reasoning reward model"})
    debug: bool = field(default=False, metadata={"help": "If True, only use 100 samples for debugging"})


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_helpsteer2_dataset(split="train")
    if script_args.debug:
        # Load a small subset of the dataset for debugging

        dataset = dataset.select(range(100))
    random.seed(0)
    annotate_dataset(
        model_name=script_args.model_name,
        batch_size=script_args.batch_size,
        max_length=script_args.max_length,
        output_path=script_args.output_path,
        reasoning=script_args.reasoning,
        dataset=dataset,
    )
