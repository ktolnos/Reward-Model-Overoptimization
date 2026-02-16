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

from experimental.dataset_annotation import load_reward_model
from reward_utils import (
    Skywork_PROMPT,
    Skywork_SYSTEM_PROMPT,
    Skywork_ASSISTANT_PROMPT,
    build_reward_texts,
    get_reward_rm,
    extract_reward_from_response,
)
from data_utils import format_prompt, setup_tokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc


@dataclass
class ScriptArguments:
    checkpoints_dir: str = field(
        default="", metadata={"help": "Directory containing policy checkpoints"}
    )
    training_rm_path: str = field(
        default="/nas/ucb/eop/Reward-Model-Overoptimization/rlhf/logs_ppo/checkpoint-40",
        metadata={"help": "Path to the reward model used during training"},
    )
    gold_rm_name: str = field(
        default="Ray2333/GRM-Gemma2-2B-rewardmodel-ft",
        metadata={"help": "Name of the gold reward model"},
    )
    dataset_name: str = field(
        default="ktolnos/helpsteer3-preference-chosenrrejected",
        metadata={"help": "Name of the dataset to evaluate on"},
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum sequence length for input processing"}
    )
    max_new_tokens: Optional[int] = field(
        default=1024, metadata={"help": "Maximum number of new tokens to generate"}
    )
    batch_size: Optional[int] = field(default=8)
    generation_batch_size: Optional[int] = field(default=8)
    device: Optional[str] = field(default="cuda")
    output_file: Optional[str] = field(default="evaluation_results.csv")
    num_responses_per_prompt: Optional[int] = field(
        default=1,
        metadata={"help": "Number of responses to generate per prompt (LLM judge)"},
    )

    wandb_project: Optional[str] = field(
        default="policy-evaluation", metadata={"help": "WandB project name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "WandB run name. If None, will use checkpoint directory name"
        },
    )
    disable_wandb: Optional[bool] = field(
        default=False, metadata={"help": "Whether to disable wandb logging"}
    )
    debug: Optional[bool] = field(
        default=False, metadata={"help": "Debug mode - only use first 100 prompts"}
    )
    evaluate_with_training_rm: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to evaluate with the training reward model"},
    )
    evaluate_with_llm_judge: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use an LLM as a judge for evaluation"},
    )
    llm_judge_model_name: Optional[str] = field(
        default="google/gemma-7b-it",
        metadata={"help": "Name of the LLM judge model on OpenRouter"},
    )
    openrouter_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "OpenRouter API key. If not provided, tries to use OPENROUTER_API_KEY env var."
        },
    )
    baseline_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the baseline model (Hugging Face model or checkpoint) for LLM judge comparison."
        },
    )
    use_dataset_response_as_baseline: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use the 'response' column from the dataset as the baseline."
        },
    )
    llm_judge_max_new_tokens: Optional[int] = field(
        default=2048, metadata={"help": "Max new tokens for the LLM judge."}
    )
    save_eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to save the full evaluation dataset with all responses and verdicts (e.g., eval_results.jsonl)."
        },
    )
    subsample_n: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of prompts to randomly subsample from the dataset. If None, uses the full dataset."
        },
    )
    kl_base_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a base policy model for KL divergence calculation. If specified, KL(policy || base) is logged."
        },
    )
    gpu_memory_utilization: float = field(
        default=0.5,
        metadata={
            "help": "vLLM GPU memory utilization. Lower this if running into OOM with RMs loaded."
        },
    )


def load_reward_model_impl(model_path_or_name, device):
    model, tokenizer = load_reward_model(
        model_path_or_name, reasoning=False, device=device
    )
    return model, tokenizer



def coerce_list(values):
    if isinstance(values, list):
        return values
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def format_responses_for_rm(prompt_messages_list, responses, rm_tokenizer):
    """Format generated responses for RM scoring using the RM's own chat template.

    Reconstructs full conversations from structured prompt messages + generated
    response text, then formats using the RM's tokenizer.  This ensures the
    scored text matches the RM's training distribution (apply_chat_template
    output), regardless of which tokenizer was used to generate the response.
    """
    texts = []
    for prompt_msgs, response in zip(prompt_messages_list, responses):
        full_conv = list(prompt_msgs) + [{"role": "assistant", "content": response}]
        text = rm_tokenizer.apply_chat_template(full_conv, tokenize=False)
        if rm_tokenizer.bos_token is not None and text.startswith(rm_tokenizer.bos_token):
            text = text[len(rm_tokenizer.bos_token) :]
        texts.append(text)
    return texts


def get_log_probs_from_ids(
    model, full_ids_list, prompt_lens_list, device, batch_size=4
):
    """Compute per-token log probabilities directly from token IDs.

    Args:
        model: The language model.
        full_ids_list: List of full sequence token IDs (prompt + response), already stripped of trailing padding.
        prompt_lens_list: List of prompt lengths for each sequence.
        device: Device to run on.
        batch_size: Batch size for processing.

    Returns:
        Tuple of (sum_log_probs, mean_log_probs, token_log_probs).
        token_log_probs is a list of lists (one per sequence), containing log probs for response tokens.
    """
    all_sum_log_probs = []
    all_mean_log_probs = []
    all_token_log_probs = []  # List of lists (one per sample)

    for i in range(0, len(full_ids_list), batch_size):
        batch_full_ids = full_ids_list[i : i + batch_size]
        batch_prompt_lens = prompt_lens_list[i : i + batch_size]

        # Pad sequences to same length (left-padding for batching)
        max_len = max(len(ids) for ids in batch_full_ids)
        padded_ids = []
        attention_masks = []

        for ids in batch_full_ids:
            pad_len = max_len - len(ids)
            # Left-pad with pad_token_id
            padded = [model.config.pad_token_id] * pad_len + list(ids)
            # Mask: 0 for our added left-padding, 1 for actual data
            mask = [0] * pad_len + [1] * len(ids)
            padded_ids.append(padded)
            attention_masks.append(mask)

        input_ids = torch.tensor(padded_ids, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift logits and labels for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probs
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        for j in range(input_ids.size(0)):
            # Calculate padding length using attention_mask (more robust)
            pad_len = (attention_mask[j] == 0).sum().item()
            prompt_len = batch_prompt_lens[j]

            # Response starts at: pad_len + prompt_len
            # In shifted coordinates (index i predicts token i+1):
            # We want the log prob FOR the first response token.
            # Input at (pad_len + prompt_len - 1) predicts (pad_len + prompt_len).
            response_start_idx = pad_len + prompt_len - 1

            # Valid content length from attention_mask
            valid_len = attention_mask[j].sum().item()
            # Response ends at (pad_len + valid_len - 1) in shifted indices
            response_end_idx = pad_len + valid_len - 1

            if response_start_idx >= response_end_idx:
                # Edge case: empty response
                all_sum_log_probs.append(0.0)
                all_mean_log_probs.append(0.0)
                all_token_log_probs.append([])
                continue

            response_log_probs = token_log_probs[j, response_start_idx:response_end_idx]

            sum_lp = response_log_probs.sum().item()
            mean_lp = response_log_probs.mean().item()

            all_sum_log_probs.append(sum_lp)
            all_mean_log_probs.append(mean_lp)
            all_token_log_probs.append(response_log_probs.cpu().tolist())

    return all_sum_log_probs, all_mean_log_probs, all_token_log_probs



def generate_responses_vllm(
    model,
    prompts,
    tokenizer,
    args,
    sampling_params=None,
    collect_logprobs=False,
):
    """Generate responses using vLLM."""
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            logprobs=1 if collect_logprobs else None,
        )

    outputs = model.generate(prompts, sampling_params)

    all_responses = []
    all_full_ids = []
    all_prompt_lens = []
    all_prompt_lens = []
    all_policy_mean_logprobs = []
    all_policy_token_logprobs = []

    for output in outputs:
        # vLLM returns 'n' outputs per prompt
        for completion in output.outputs:
            generated_text = completion.text
            all_responses.append(generated_text)

            if collect_logprobs:
                prompt_ids = list(output.prompt_token_ids)
                response_ids = list(completion.token_ids)
                full_ids = prompt_ids + response_ids
                all_full_ids.append(full_ids)
                all_prompt_lens.append(len(prompt_ids))

                # Calculate mean logprob for the response
                seq_logprobs = completion.logprobs
                if seq_logprobs:
                    token_logprobs = []
                    for lp, t_id in zip(seq_logprobs, response_ids):
                        if not lp:
                            print(f"Warning: No logprobs for completion {completion}")
                            continue
                        token_entry = lp.get(t_id)
                        if token_entry is None:
                            print(
                                f"Warning: Token {t_id} not found in logprobs for completion {completion}"
                            )
                            continue
                        token_logprobs.append(token_entry.logprob)
                    if token_logprobs:
                        mean_lp = sum(token_logprobs) / len(token_logprobs)
                    else:
                        mean_lp = 0.0
                else:
                    mean_lp = 0.0
                all_policy_mean_logprobs.append(mean_lp)
                all_policy_token_logprobs.append(token_logprobs if seq_logprobs else [])

    return (
        all_responses,
        all_full_ids,
        all_prompt_lens,
        np.array(all_policy_mean_logprobs),
        all_policy_token_logprobs,
    )


def load_policy_model(model_path, tokenizer, device):
    """Loads a policy model from a path."""
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model


def update_vllm_weights(llm, model_path, device="cpu"):
    """Update vLLM weights from a Hugging Face checkpoint."""
    print(f"Updating vLLM weights from {model_path}")

    tp_size = None
    try:
        tp_size = llm.llm_engine.parallel_config.tensor_parallel_size
    except Exception:
        try:
            tp_size = llm.llm_engine.model_config.tensor_parallel_size
        except Exception:
            tp_size = None
    if tp_size is not None and tp_size > 1:
        raise RuntimeError(
            "update_vllm_weights only supports tensor_parallel_size=1. "
            "Reinitialize vLLM per checkpoint for multi-GPU runs."
        )

    # Load HF model to CPU to avoid OOM
    # We use the same configuration/architecture/tokenizer, just different weights
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )

    # Access vLLM internal model
    # Note: This path assumes single-GPU/colocate setup where driver_worker is accessible
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    params_to_load = []
    for name, param in hf_model.named_parameters():
        # Basic name matching.
        # vLLM/HF compatibility usually preserves names for supported architectures.
        params_to_load.append((name, param.data))

    llm_model.load_weights(params_to_load)

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()


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
        raise ValueError(
            "OpenRouter API key must be provided via --openrouter_api_key or OPENROUTER_API_KEY env var."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # Optional, but good practice
        "X-Title": "Reward Model Overoptimization",  # Optional, but good practice
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

        user_prompt = Skywork_PROMPT.format(
            question=prompt, answer1=answer1, answer2=answer2
        )

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
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                all_responses.append(generated_text)

                preference = extract_reward_from_response(generated_text)
                if swap:
                    preference *= -1  # un-swap

                all_preferences.append(preference)
                break  # Success, exit retry loop

            except BaseException as e:
                if (
                    isinstance(e, requests.exceptions.HTTPError)
                    and (e.response.status_code in (429, 500, 502))
                    or isinstance(e, KeyError)
                ):  # sometimes server returns error in the body
                    if attempt < retries - 1:
                        # Try to get the specific wait time from the 'Retry-After' header
                        retry_after_header = (
                            e.response.headers.get("Retry-After")
                            if hasattr(e, "response")
                            else None
                        )
                        if retry_after_header:
                            try:
                                sleep_time = (
                                    int(retry_after_header) + 1
                                )  # Add 1s buffer
                                print(
                                    f"Rate limit exceeded. Following server's 'Retry-After' header. Waiting for {sleep_time} seconds."
                                )
                            except ValueError:
                                # If the header is a date, this will fail. Fallback to exponential backoff.
                                sleep_time = backoff_factor * (
                                    10**attempt
                                ) + random.uniform(0, 1)
                                print(
                                    f"Rate limit exceeded. Could not parse 'Retry-After' header. Retrying in {sleep_time:.2f} seconds..."
                                )
                        else:
                            # Fallback to exponential backoff if the header is not present
                            sleep_time = backoff_factor * (
                                10**attempt
                            ) + random.uniform(0, 1)
                            print(
                                f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds (exponential backoff)..."
                            )

                        time.sleep(sleep_time)
                    else:
                        print(f"Error querying LLM Judge after multiple retries: {e}")
                        all_preferences.append(0)  # Default to tie on error
                        all_responses.append(f"Error querying LLM Judge: {e}")
                else:
                    print(f"Error querying LLM Judge: {e}\n\nlocals:\n{locals()}")
                    all_preferences.append(0)  # Default to tie on error
                    all_responses.append(
                        f"Error querying LLM Judge: {e}\n\nlocals:\n{locals()}"
                    )
                    break  # Don't retry for other errors

    return all_preferences, all_responses


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize wandb if enabled
    if not args.disable_wandb:
        wandb_run_name = args.wandb_run_name or os.path.basename(
            os.path.normpath(args.checkpoints_dir)
        )
        if args.debug:
            wandb_run_name += "_debug"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args),
            group=args.checkpoints_dir,
            job_type="evaluation",
        )

    # --- Common Setup ---
    print("Loading evaluation dataset...")
    split = (
        "test"
        if args.dataset_name.startswith("/")
        or "ktolnos/helpsteer3_gold" in args.dataset_name
        else (
            "validation"
            if args.dataset_name == "ktolnos/helpsteer3-preference-chosenrrejected"
            else "train"
        )
    )
    print("Using dataset split:", split)
    dataset = load_dataset(args.dataset_name, split=split)
    if args.debug:
        print("Debug mode: using only first 100 prompts")
        dataset = dataset.select(range(min(100, len(dataset))))
    elif args.subsample_n is not None:
        if args.subsample_n > len(dataset):
            print(
                f"Warning: subsample_n ({args.subsample_n}) is larger than the dataset size ({len(dataset)}). Using the full dataset."
            )
        else:
            dataset = dataset.shuffle(seed=42).select(range(args.subsample_n))
            print(f"Subsampling to {args.subsample_n} prompts.")

    checkpoints = sorted(
        [d for d in os.listdir(args.checkpoints_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )

    if not checkpoints:
        raise ValueError(f"No checkpoints found in directory: {args.checkpoints_dir}")

    first_checkpoint_path = os.path.join(args.checkpoints_dir, checkpoints[0])

    print("Loading tokenizer...")
    policy_tokenizer = AutoTokenizer.from_pretrained(
        first_checkpoint_path, trust_remote_code=True,
    )
    setup_tokenizer(policy_tokenizer)

    # Prepare dataset based on its structure
    prompt_messages_list = None  # structured messages for proper RM formatting
    if "chosen" in dataset.column_names:
        print("Using 'chosen' column for prompts (chosen[:-1]).")

        def extract_prompt(example):
            return {
                "prompt": format_prompt(example["chosen"], policy_tokenizer),
                "prompt_messages": example["chosen"][:-1],
            }

        dataset = dataset.map(extract_prompt, num_proc=1)
    else:
        raise ValueError("Dataset must have a 'chosen' column.")

    # Filter prompts that exceed max_length (no room for generation)
    before = len(dataset)
    dataset = dataset.filter(
        lambda x: len(policy_tokenizer.encode(x["prompt"], add_special_tokens=False)) <= args.max_length,
    )
    if len(dataset) < before:
        print(f"Filtered prompts by max_length={args.max_length}: {before} -> {len(dataset)}")

    original_prompts = coerce_list(dataset["prompt"])
    prompt_messages_list = dataset["prompt_messages"]
    print(f"Using {len(original_prompts)} prompts for evaluation")

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

        num_responses = args.num_responses_per_prompt or 1
        if num_responses < 1:
            raise ValueError("--num_responses_per_prompt must be >= 1")

        judge_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=args.max_new_tokens,
            n=num_responses,
        )
        baseline_sampling_params = SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            n=1,
        )

        baseline_responses = []
        if args.use_dataset_response_as_baseline:
            print("Using dataset 'chosen' column as baseline response.")
            if "chosen" not in dataset.column_names:
                raise ValueError(
                    "Dataset must have a 'chosen' column to use it as a baseline."
                )

            def extract_response_from_chosen(example):
                return {"response": example["chosen"][-1]["content"]}

            baseline_responses = dataset.map(extract_response_from_chosen)["response"]
            baseline_responses = coerce_list(baseline_responses)
        else:
            print(f"Loading baseline model from {args.baseline_model_path}")
            # Use vLLM for baseline generation
            baseline_llm = LLM(
                model=args.baseline_model_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_length + args.max_new_tokens,
                trust_remote_code=True,
            )
            print("Generating baseline responses...")
            # Note: baseline generation usually just needs 1 response per prompt
            baseline_responses, _, _, _, _ = generate_responses_vllm(
                baseline_llm,
                original_prompts,
                policy_tokenizer,
                args,
                sampling_params=baseline_sampling_params,
            )

            # Cleanup vLLM
            destroy_model_parallel()
            del baseline_llm.llm_engine.model_executor
            del baseline_llm
            gc.collect()
            torch.cuda.empty_cache()

        # Initialize the full_eval_data list
        for i in range(len(original_prompts)):
            full_eval_data.append(
                {
                    "prompt": original_prompts[i],
                    "baseline_response": baseline_responses[i],
                    "checkpoints": {},
                }
            )

        # Initialize vLLM with the first checkpoint
        # We reuse this instance and update weights for subsequent checkpoints
        first_checkpoint_path = os.path.join(args.checkpoints_dir, checkpoints[0])
        print(f"Initializing vLLM with {first_checkpoint_path}")
        llm = LLM(
            model=first_checkpoint_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_length + args.max_new_tokens,
            trust_remote_code=True,
        )

        try:
            for checkpoint in tqdm(
                checkpoints, desc="Evaluating checkpoints with LLM Judge"
            ):
                checkpoint_path = os.path.join(args.checkpoints_dir, checkpoint)
                checkpoint_num = int(checkpoint.split("-")[1])
                print(f"\nEvaluating {checkpoint}")

                # Update weights if not the first checkpoint (which was used for init)
                if checkpoint != checkpoints[0]:
                    update_vllm_weights(llm, checkpoint_path)

                try:
                    policy_responses, _, _, _, _ = generate_responses_vllm(
                        llm,
                        original_prompts,
                        policy_tokenizer,
                        args,
                        sampling_params=judge_sampling_params,
                    )

                    expanded_prompts = [
                        p for p in original_prompts for _ in range(num_responses)
                    ]
                    expanded_baselines = [
                        b for b in baseline_responses for _ in range(num_responses)
                    ]
                    expected = len(original_prompts) * num_responses
                    if len(policy_responses) != expected:
                        raise ValueError(
                            f"Expected {expected} policy responses, got {len(policy_responses)}."
                        )

                    verdicts, judge_responses = get_llm_judge_verdicts(
                        expanded_prompts, policy_responses, expanded_baselines, args
                    )

                    for i in range(len(original_prompts)):
                        start = i * num_responses
                        end = start + num_responses
                        full_eval_data[i]["checkpoints"][checkpoint_num] = {
                            "policy_responses": policy_responses[start:end],
                            "llm_judge_responses": judge_responses[start:end],
                            "llm_judge_verdicts": verdicts[start:end],
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
                        "mean_no_tie": (
                            (wins - losses) / (wins + losses)
                            if (wins + losses) > 0
                            else 0
                        ),
                        "total_comparisons": total,
                    }

                    if not args.disable_wandb:
                        wandb_log_data = {
                            f"llm_judge/{k}": v
                            for k, v in checkpoint_results.items()
                            if k != "checkpoint"
                        }
                        wandb_log_data["checkpoint"] = checkpoint_num
                        wandb.log(wandb_log_data)

                    results.append(checkpoint_results)
                except Exception as e:
                    print(f"Error evaluating checkpoint {checkpoint}: {e}")
                    import traceback

                    traceback.print_exc()
                    # Log an empty result for this checkpoint
                    results.append(
                        {
                            "checkpoint": checkpoint_num,
                            "win_rate": None,
                            "loss_rate": None,
                            "tie_rate": None,
                            "total_comparisons": 0,
                        }
                    )
                    continue

        finally:
            if "llm" in locals():
                destroy_model_parallel()
                del llm.llm_engine.model_executor
                del llm
                gc.collect()
                torch.cuda.empty_cache()

    else:
        # --- Reward Model Evaluation (Original Logic) ---
        print("Starting Reward Model evaluation...")
        print("Loading reward models...")
        if args.evaluate_with_training_rm:
            training_rm, training_rm_tokenizer = load_reward_model_impl(
                args.training_rm_path, args.device
            )
        gold_rm, gold_rm_tokenizer = load_reward_model_impl(
            args.gold_rm_name, args.device
        )

        # Load base policy for KL calculation if specified
        base_policy_model = None
        kl_reward_data = (
            []
        )  # Accumulate (checkpoint, mean_kl, mean_gold_reward) for final plot
        if args.kl_base_model_path:
            print(f"Loading base policy for KL from {args.kl_base_model_path}...")
            base_policy_model = load_policy_model(
                args.kl_base_model_path, policy_tokenizer, args.device
            )

        # Initialize vLLM with the first checkpoint
        first_checkpoint_path = os.path.join(args.checkpoints_dir, checkpoints[0])
        print(f"Initializing vLLM with {first_checkpoint_path}")
        llm = LLM(
            model=first_checkpoint_path,
            tokenizer=first_checkpoint_path,
            dtype="float16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_length + args.max_new_tokens,
            trust_remote_code=True,
        )

        try:
            for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
                checkpoint_path = os.path.join(args.checkpoints_dir, checkpoint)
                print(f"\nEvaluating {checkpoint}")

                # Update weights if not the first checkpoint (which was used for init)
                if checkpoint != checkpoints[0]:
                    update_vllm_weights(llm, checkpoint_path)

                try:
                    # Use string prompts for vLLM
                    prompts_list = original_prompts

                    # Generate responses
                    # We always get full_ids and prompt_lens if kl_base_model_path is set, because generate_responses_vllm handles it
                    (
                        responses,
                        full_ids_list,
                        prompt_lens_list,
                        policy_mean_logprobs,
                        policy_token_logprobs,
                    ) = generate_responses_vllm(
                        llm,
                        prompts_list,
                        policy_tokenizer,
                        args,
                        collect_logprobs=base_policy_model is not None,
                    )

                    # Format responses using each RM's own chat template so the
                    # scored text matches each RM's training distribution.
                    if prompt_messages_list is not None:
                        if args.evaluate_with_training_rm:
                            training_reward_texts = format_responses_for_rm(
                                prompt_messages_list, responses, training_rm_tokenizer
                            )
                        gold_reward_texts = format_responses_for_rm(
                            prompt_messages_list, responses, gold_rm_tokenizer
                        )
                    else:
                        # Fallback: no structured messages, plain concatenation
                        fallback_texts = [p + c for p, c in zip(prompts_list, responses)]
                        if args.evaluate_with_training_rm:
                            training_reward_texts = fallback_texts
                        gold_reward_texts = fallback_texts

                    if args.evaluate_with_training_rm:
                        training_rm_scores = get_reward_rm(
                            training_rm,
                            training_rm_tokenizer,
                            training_reward_texts,
                            batch_size=args.batch_size,
                        ).numpy()

                    print("\n\nreward texts\n\n", "\n;\n".join(gold_reward_texts[:2]))

                    gold_rm_scores = get_reward_rm(
                        gold_rm,
                        gold_rm_tokenizer,
                        gold_reward_texts,
                        batch_size=args.batch_size,
                    ).numpy()

                    checkpoint_num = int(checkpoint.split("-")[1])
                    checkpoint_results = {
                        "checkpoint": checkpoint_num,
                        "gold_rm/mean": float(np.mean(gold_rm_scores)),
                        "gold_rm/std": float(np.std(gold_rm_scores)),
                    }
                    if not args.disable_wandb:
                        checkpoint_results["gold_rm/scores_hist"] = wandb.Histogram(
                            gold_rm_scores
                        )

                    if args.evaluate_with_training_rm:
                        checkpoint_results["training_rm/mean"] = float(
                            np.mean(training_rm_scores)
                        )
                        checkpoint_results["training_rm/std"] = float(
                            np.std(training_rm_scores)
                        )
                        if not args.disable_wandb:
                            checkpoint_results["training_rm/scores_hist"] = (
                                wandb.Histogram(training_rm_scores)
                            )

                    # KL calculation
                    if base_policy_model is not None:
                        # Policy mean logprobs come directly from vLLM
                        policy_mean_lp = policy_mean_logprobs

                        # Get log probs from base model (HF model)
                        (
                            base_sum_lp,
                            base_mean_lp,
                            base_token_lp_list,
                        ) = get_log_probs_from_ids(
                            base_policy_model,
                            full_ids_list,
                            prompt_lens_list,
                            args.device,
                            batch_size=4,
                        )

                        # KL(policy || base) = E_policy[log(policy) - log(base)]
                        # Standard KL (can be negative)
                        kl_per_sample = policy_mean_lp - np.array(base_mean_lp)
                        kl_mean = float(np.mean(kl_per_sample))
                        kl_std = float(np.std(kl_per_sample))

                        checkpoint_results["kl/mean"] = kl_mean
                        checkpoint_results["kl/std"] = kl_std

                        # GRPO KL (non-negative)
                        # kl = exp(log_ref - log_policy) - (log_ref - log_policy) - 1
                        kl_grpo_per_sample = []
                        for pol_lp, ref_lp in zip(
                            policy_token_logprobs, base_token_lp_list
                        ):
                            # Ensure same length (should be gauranteed by logic)
                            min_len = min(len(pol_lp), len(ref_lp))
                            pol_lp = np.array(pol_lp[:min_len])
                            ref_lp = np.array(ref_lp[:min_len])

                            # per token KL
                            # diff = log_ref - log_policy
                            diff = ref_lp - pol_lp
                            per_token_kl = np.exp(diff) - diff - 1
                            kl_grpo_per_sample.append(np.mean(per_token_kl))

                        kl_grpo_mean = float(np.mean(kl_grpo_per_sample))
                        kl_grpo_std = float(np.std(kl_grpo_per_sample))

                        checkpoint_results["kl/grpo_mean"] = kl_grpo_mean
                        checkpoint_results["kl/grpo_std"] = kl_grpo_std

                        # Accumulate data for the final aggregate plot
                        kl_reward_data.append(
                            {
                                "checkpoint": checkpoint_num,
                                "kl_grpo": kl_grpo_mean,
                                "gold_reward": float(np.mean(gold_rm_scores)),
                            }
                        )

                    if not args.disable_wandb:
                        wandb.log(checkpoint_results)

                    if "gold_rm/scores_hist" in checkpoint_results:
                        del checkpoint_results["gold_rm/scores_hist"]
                    if "training_rm/scores_hist" in checkpoint_results:
                        del checkpoint_results["training_rm/scores_hist"]
                    results.append(checkpoint_results)

                except Exception as e:
                    print(f"Error checking checkpoint {checkpoint}: {e}")
                    import traceback

                    traceback.print_exc()

        finally:
            if "llm" in locals():
                try:
                    destroy_model_parallel()
                    del llm.llm_engine.model_executor
                    del llm
                except:
                    pass
            gc.collect()
            torch.cuda.empty_cache()

        # After all checkpoints, create a single aggregate plot of Gold Reward vs KL
        if kl_reward_data and not args.disable_wandb:
            kl_reward_table = wandb.Table(
                columns=["checkpoint", "kl_grpo", "gold_reward"],
                data=[
                    [d["checkpoint"], d["kl_grpo"], d["gold_reward"]]
                    for d in kl_reward_data
                ],
            )
            wandb.log(
                {
                    "gold_reward_vs_kl": wandb.plot.scatter(
                        kl_reward_table,
                        "kl_grpo",
                        "gold_reward",
                        title="Gold Reward vs KL (GRPO) (across checkpoints)",
                    )
                }
            )

    if results:
        results_df = pd.DataFrame(results)
        if args.debug:
            args.output_file = args.output_file.replace(".csv", "_debug.csv")
        results_df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
    else:
        print("\nNo results were generated!")

    if args.save_eval_dataset_path and full_eval_data:
        with open(args.save_eval_dataset_path, "w") as f:
            for item in full_eval_data:
                f.write(json.dumps(item) + "\n")
        print(f"Full evaluation data saved to {args.save_eval_dataset_path}")

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
