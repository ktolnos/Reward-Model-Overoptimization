"""
Dataset annotation script for loading HelpSteer2 preference dataset,
evaluating it with a reward model, and saving the augmented dataset.
"""

import os
from dataclasses import field, dataclass

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from tqdm import tqdm
import json


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


def load_reward_model(model_name="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft", device=None):
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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16,
                                                               device_map=device)
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
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model(**inputs)
            all_rewards = outputs.score.squeeze(-1).cpu().numpy()

        # Process the results
        for j in range(batch_size_actual):
            chosen_idx = j * 2  # Even indices are chosen responses
            rejected_idx = j * 2 + 1  # Odd indices are rejected responses

            chosen_reward = float(all_rewards[chosen_idx, 0])
            rejected_reward = float(all_rewards[rejected_idx, 0])
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
    model, tokenizer = load_reward_model(model_name)

    # Evaluate dataset
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
    model_name: str = field(default="nicolinho/QRM-Gemma-2-27B",
                            metadata={"help": "Name of the reward model"})
    batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_length: int = field(default=1024, metadata={"help": "Maximum sequence length"})
    output_path: str = field(default="data/helpsteer2_gold/", metadata={"help": "Path to save the dataset"})
    debug: bool = field(default=False, metadata={"help": "If True, only use 100 samples for debugging"})


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_helpsteer2_dataset(split="train")
    if script_args.debug:
        # Load a small subset of the dataset for debugging

        dataset = dataset.select(range(100))
    annotate_dataset(
        model_name=script_args.model_name,
        batch_size=script_args.batch_size,
        max_length=script_args.max_length,
        output_path=script_args.output_path,
        dataset=dataset,
    )
