"""
Dataset annotation script for loading HelpSteer2 preference dataset,
evaluating it with a reward model, and saving the augmented dataset.
"""

import os
from dataclasses import field, dataclass
from typing import List, Dict, Any

import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser
from tqdm import tqdm
import json
import random
from reward_utils import (
    Skywork_SYSTEM_PROMPT,
    Skywork_PROMPT,
    Skywork_ASSISTANT_PROMPT,
    extract_reward_from_response,
    load_reward_model,
)
from data_utils import (
    tokenize_for_rm,
    strip_bos_if_present_batch,
    tokenize_texts_with_special_tokens,
    get_generation_stop_token_ids,
    format_and_validate_preference_sample,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_CONVERSATION_TOKENS,
)

_MAX_FORMAT_VALIDATION_TOKENS = 10**9


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
        all_texts = []
        batch_size_actual = len(batch["preference_strength"])  # Get actual batch size

        # Format and validate both chosen and rejected conversations.
        for j in range(batch_size_actual):
            _, chosen_text, rejected_text = format_and_validate_preference_sample(
                batch["chosen"][j],
                tokenizer,
                rejected_messages=batch["rejected"][j],
                max_prompt_length=_MAX_FORMAT_VALIDATION_TOKENS,
                max_conversation_length=_MAX_FORMAT_VALIDATION_TOKENS,
                sample_id=i + j,
                context="Annotation",
            )
            if rejected_text is None:
                raise ValueError(f"Missing rejected text after formatting (sample_id={i + j}).")

            all_texts.append(chosen_text)
            all_texts.append(rejected_text)

        # Tokenize all conversations in a single batch
        inputs = tokenize_for_rm(all_texts, tokenizer).to(device)

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model(**inputs)
            all_rewards = outputs.logits.squeeze(-1).cpu().float().numpy()

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
    stop_token_ids = get_generation_stop_token_ids(tokenizer)
    eos_for_generation = stop_token_ids

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating with reward model"):
        prompts = []
        swaps = []
        for j in range(min(len(dataset) - i, batch_size)):
            sample = dataset[i+j]
            query, _, _ = format_and_validate_preference_sample(
                sample["chosen"],
                tokenizer,
                rejected_messages=sample["rejected"],
                max_prompt_length=_MAX_FORMAT_VALIDATION_TOKENS,
                max_conversation_length=_MAX_FORMAT_VALIDATION_TOKENS,
                sample_id=i + j,
                context="AnnotationReasoning",
            )
            answer1 = sample["chosen"][-1]["content"]
            answer2 = sample["rejected"][-1]["content"]

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

        prompts = strip_bos_if_present_batch(prompts, tokenizer)
        inputs = tokenizer(prompts,
                           padding='longest',
                           truncation=True,
                           max_length=max_length,
                           return_tensors="pt",
                           padding_side="left",
                           add_special_tokens=True,
       ).to(model.device)

        generation_args = {
            "max_new_tokens": 8192,
            "temperature": 0.6,
            "do_sample": True,
            "top_p": 1.0,
            "eos_token_id": eos_for_generation,
            "pad_token_id": tokenizer.pad_token_id,
        }
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)

        prompt_token_count = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_token_count:]

        for j, swap in zip(range(len(swaps)), swaps):
            sample = dataset[i + j]
            generated_text = tokenizer.decode(generated_tokens[j], skip_special_tokens=True)

            reward = extract_reward_from_response(generated_text)
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


def generate_with_reference_policy(
    dataset: List[Dict], policy_model, tokenizer, batch_size, max_length, num_responses=2, device=None
) -> List[Dict[str, Any]]:
    """
    Generates responses for each prompt in the dataset using a reference policy model.

    Args:
        dataset: A HuggingFace dataset object.
        policy_model: The causal language model to use for generation.
        tokenizer: The tokenizer for the policy model.
        batch_size (int): The batch size for generation.
        max_length (int): The maximum length for the generated sequence.
        num_responses (int): The number of responses to generate per prompt.
        device (str): The device to run generation on.

    Returns:
        A list of dictionaries, where each item is augmented with generated reference responses.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    stop_token_ids = get_generation_stop_token_ids(tokenizer)
    eos_for_generation = stop_token_ids

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating with reference policy"):
        batch_data = dataset[i:i + batch_size]
        print(batch_size, len(batch_data))

        prompts = []
        for j, item in enumerate(batch_data):
            prompt_text, _, _ = format_and_validate_preference_sample(
                item["chosen"],
                tokenizer,
                rejected_messages=item.get("rejected"),
                max_prompt_length=_MAX_FORMAT_VALIDATION_TOKENS,
                max_conversation_length=_MAX_FORMAT_VALIDATION_TOKENS,
                sample_id=i + j,
                context="ReferencePolicy",
            )
            prompts.append(prompt_text)

        inputs = tokenize_texts_with_special_tokens(
            prompts,
            tokenizer,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=num_responses,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=eos_for_generation,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        for j, item in enumerate(batch_data):
            new_item = dict(item)
            item_outputs = outputs[j * num_responses : (j + 1) * num_responses]
            new_tokens = item_outputs[:, input_len:]
            clean_responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for k in range(num_responses):
                new_item[f'reference_response_{k+1}'] = clean_responses[k]
            results.append(new_item)

    return results


def evaluate_with_reference_reward_model(
    dataset: List[Dict], reward_model, tokenizer, batch_size, max_length, num_responses=2, device=None
) -> List[Dict[str, Any]]:
    """
    Evaluates generated reference responses with a reference reward model.

    Args:
        dataset: A list of dictionaries, each containing reference responses.
        reward_model: The reward model to use for evaluation.
        tokenizer: The tokenizer for the reward model.
        batch_size (int): The batch size for evaluation.
        max_length (int): The maximum sequence length.
        num_responses (int): The number of reference responses per item.
        device (str): The device to run evaluation on.

    Returns:
        A list of dictionaries, augmented with the mean reference reward.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating with reference reward model"):
        batch_data = dataset[i:i + batch_size]
        all_texts = []

        for j, item in enumerate(batch_data):
            prompt_conv = item['chosen'][:-1]
            for k in range(num_responses):
                response_text = item.get(f'reference_response_{k+1}')
                if response_text:
                    full_conv = prompt_conv + [{'role': 'assistant', 'content': response_text}]
                    _, full_text, _ = format_and_validate_preference_sample(
                        full_conv,
                        tokenizer,
                        max_prompt_length=_MAX_FORMAT_VALIDATION_TOKENS,
                        max_conversation_length=_MAX_FORMAT_VALIDATION_TOKENS,
                        sample_id=i + j,
                        context="ReferenceReward",
                    )
                    all_texts.append(full_text)

        inputs = tokenize_for_rm(all_texts, tokenizer).to(device)

        with torch.no_grad():
            outputs = reward_model(**inputs)
            all_rewards = outputs.logits.squeeze(-1).cpu().float().numpy()

        reward_idx = 0
        for item in batch_data:
            new_item = dict(item)
            item_rewards = [all_rewards[reward_idx + k] for k in range(num_responses) if f'reference_response_{k+1}' in item]
            reward_idx += len(item_rewards)

            new_item['reference_reward'] = float(sum(item_rewards) / len(item_rewards)) if item_rewards else None
            for k in range(num_responses):
                response_key = f'reference_response_{k+1}'
                if response_key in new_item:
                    new_item[f'reference_reward_{k+1}'] = float(item_rewards[k]) if k < len(item_rewards) else None
            results.append(new_item)

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

    # Calculate some statistics if gold rewards are present
    if "chosen_reward" in df.columns and "rejected_reward" in df.columns:
        reward_gap = df["chosen_reward"] - df["rejected_reward"]
        accuracy = df["does_gold_agree_with_original"].mean()

        print(f"Reward model accuracy: {accuracy:.4f}")
        print(f"Average reward gap: {reward_gap.mean():.4f}")

    # Save as JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved annotated dataset to {output_path}")
    return output_path


def load_annotated_dataset(input_path="data/helpsteer2_gold/train.json", split="train", convert_to_list=True):
    """
    Load the annotated dataset from disk.
    
    Args:
        input_path (str): Path to the saved dataset
        
    Returns:
        list: List of dictionaries with evaluation results
    """
    if input_path.endswith(".json"):
        with open(input_path, "r") as f:
            results = json.load(f)

        print(f"Loaded {len(results)} examples from {input_path}")
        return results
    elif input_path == "helpsteer2":
        dataset = load_helpsteer2_dataset(split=split)
        if not convert_to_list:
            return dataset
        results = []
        for i in range(len(dataset)):
            item = dataset[i]
            results.append({
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "preference_strength": item["preference_strength"],
                "does_gold_agree_with_original": True,  # Placeholder for gold evaluation
                "chosen_reward": item["preference_strength"],  # Placeholder for gold evaluation
                "rejected_reward": -item["preference_strength"],  # Placeholder for gold evaluation
            })
        return results
    elif input_path == 'helpsteer3':
        dataset = load_dataset("ktolnos/helpsteer3-preference-chosenrrejected", split=split)
        if not convert_to_list:
            return dataset
        results = []
        for i in range(len(dataset)):
            item = dataset[i]
            results.append({
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "preference_strength": item["preference_strength"],
                "does_gold_agree_with_original": True,  # Placeholder for gold evaluation
                "chosen_reward": item["preference_strength"],  # Placeholder for gold evaluation
                "rejected_reward": -item["preference_strength"],  # Placeholder for gold evaluation
            })
        return results
    else:
        raise ValueError(f"Unsupported input path format: {input_path}. Expected a JSON file or 'helpsteer2'.")


def validate_dataset_length_or_fail(
    dataset,
    tokenizer,
    max_prompt_length=DEFAULT_MAX_PROMPT_TOKENS,
    max_conversation_length=DEFAULT_MAX_CONVERSATION_TOKENS,
):
    """Validate prompt and conversation token constraints and fail fast."""
    for idx, sample in enumerate(dataset):
        format_and_validate_preference_sample(
            sample["chosen"],
            tokenizer,
            rejected_messages=sample.get("rejected"),
            max_prompt_length=max_prompt_length,
            max_conversation_length=max_conversation_length,
            sample_id=idx,
            context="Annotation",
        )




def annotate_dataset(model_name,
                     batch_size, max_length, output_path,
                     reasoning=False,
                     dataset=None,
                     test_split_size=0.05):
    """
    Main function to load dataset, evaluate with reward model, and save results.
    
    Args:
        model_name (str): Name of the reward model
        batch_size (int): Batch size for evaluation
        max_length (int): Maximum sequence length
        output_path (str): Path to save the dataset directory
        reasoning (bool): If true, use reasoning reward model
        dataset: Optional pre-loaded dataset to use
        test_split_size (float): Proportion of dataset to use for the test split.
        
    Returns:
        str: Path to the saved dataset
    """
    model, tokenizer = load_reward_model(model_name, reasoning, device="cuda" if torch.cuda.is_available() else "cpu")
    validate_dataset_length_or_fail(dataset, tokenizer)
    if reasoning:
        raise NotImplementedError(
            "Reasoning reward-model annotation is disabled for this pipeline. "
            "The reasoning path uses generation-based judging prompts and a "
            "different objective than sequence-classification reward scoring, "
            "so results are not directly comparable to the non-reasoning gold "
            "annotation pipeline. Use --reasoning False."
        )
        # TODO: Re-enable once reasoning-mode annotation is redesigned.
        # results = evaluate_with_reasoning_reward_model(dataset, model, tokenizer, batch_size, max_length)
    else:
        results = evaluate_with_reward_model(dataset, model, tokenizer, batch_size, max_length)

    if test_split_size > 0:
        train_size = int((1 - test_split_size) * len(results))
        train_results, test_results = results[:train_size], results[train_size:]

        test_output_path = os.path.join(output_path, "test.json")
        save_annotated_dataset(test_results, test_output_path)

        results = train_results
        output_path = os.path.join(output_path, "train.json")

    return save_annotated_dataset(results, output_path)

@dataclass
class ScriptArguments:
    model_name: str = field(default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                            metadata={"help": "Name of the gold reward model (for 'gold' mode)."})
    batch_size: int = field(default=16, metadata={"help": "Batch size for evaluation"})
    max_length: int = field(default=4096, metadata={"help": "Maximum sequence length"})
    output_path: str = field(default="/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B",
                             metadata={"help": "Path to save the dataset. Directory for 'gold' mode, file path for other modes."})
    reasoning: bool = field(default=False, metadata={"help": "If True, use reasoning reward model for 'gold' mode."})
    debug: bool = field(default=False, metadata={"help": "If True, only use 25 samples for debugging."})

    # Arguments for different annotation modes
    annotation_mode: str = field(
        default="gold",
        metadata={"help": "Annotation mode. One of: 'gold', 'reference_policy', 'reference_reward'."}
    )
    input_path: str = field(
        default='helpsteer3',
        metadata={"help": "Path to load a dataset from. Required for 'reference_reward' and 'gold' mode. Special case: 'helpsteer2' or 'helpsteer3' to load the original dataset."}
    )
    input_split: str = field(
        default='train',
        metadata={
            "help": "Dataset split to load ('train', 'validation', or 'test') for 'gold' mode."}
    )
    reference_policy_name: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Name of the causal LLM to use as the reference policy."}
    )
    reference_reward_model_name: str = field(
        default="/nas/ucb/eop/Reward-Model-Overoptimization/save_reward_models/Qwen3-0.6B_BT_RM_Qwen3-0.6B_len3000_fulltrain_1e-05_data/logs/checkpoint-256",
        metadata={"help": "Name of the reward model for evaluating reference responses."}
    )
    num_reference_responses: int = field(
        default=2,
        metadata={"help": "Number of responses to generate from the reference policy."}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Mode Dispatcher ---
    if script_args.annotation_mode == "gold":
        print("--- Running in GOLD annotation mode ---")
        dataset = load_annotated_dataset(script_args.input_path, script_args.input_split, convert_to_list=False)
        if script_args.debug:
            dataset = dataset.select(range(25))

        annotate_dataset(
            model_name=script_args.model_name,
            batch_size=script_args.batch_size,
            max_length=script_args.max_length,
            output_path=script_args.output_path,
            reasoning=script_args.reasoning,
            dataset=dataset,
        )

    elif script_args.annotation_mode == "reference_policy":
        print("--- Running in REFERENCE POLICY annotation mode ---")
        dataset = load_annotated_dataset(script_args.input_path, script_args.input_split)
        if script_args.debug:
            dataset = dataset[:25]

        # Load reference policy model (as a causal LM)
        policy_model, policy_tokenizer = load_reward_model(
            script_args.reference_policy_name, reasoning=True, device=device
        )
        validate_dataset_length_or_fail(dataset, policy_tokenizer)

        results = generate_with_reference_policy(
            dataset, policy_model, policy_tokenizer, script_args.batch_size,
            script_args.max_length, script_args.num_reference_responses, device
        )

        save_annotated_dataset(results, script_args.output_path)

    elif script_args.annotation_mode == "reference_reward":
        print("--- Running in REFERENCE REWARD annotation mode ---")
        if not script_args.input_path:
            raise ValueError("An --input_path must be provided for 'reference_reward' mode.")

        dataset = load_annotated_dataset(script_args.input_path)
        if script_args.debug:
            dataset = dataset[:25]

        # Load reference reward model (as a sequence classification model)
        reward_model, reward_tokenizer = load_reward_model(
            script_args.reference_reward_model_name, reasoning=False, device=device
        )
        validate_dataset_length_or_fail(dataset, reward_tokenizer)

        results = evaluate_with_reference_reward_model(
            dataset, reward_model, reward_tokenizer, script_args.batch_size,
            script_args.max_length, script_args.num_reference_responses, device
        )

        save_annotated_dataset(results, script_args.output_path)

    else:
        raise ValueError(f"Unknown annotation mode: '{script_args.annotation_mode}'. "
                         f"Choose from 'gold', 'reference_policy', or 'reference_reward'.")
