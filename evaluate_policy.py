import os
import random
import requests
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import wandb

from reward_utils import (
    Skywork_PROMPT,
    Skywork_SYSTEM_PROMPT,
    Skywork_ASSISTANT_PROMPT,
    get_reward_rm,
    extract_reward_from_response,
    load_reward_model,
)
from data_utils import (
    format_and_validate_preference_sample,
    setup_tokenizer,
    load_policy_and_tokenizer,
    _is_lora_checkpoint,
    _get_lora_base_model_path,
    get_generation_stop_token_ids,
    get_length_config,
    compute_max_prompt_length,
    DATASET_LENGTH_CONFIGS,
)
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import hashlib
from datetime import datetime

CHOSEN_SCORES_CACHE_VERSION = 1


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
        default=1024, metadata={"help": "Maximum prompt length for vLLM. Overridden when --auto_prompt_length is set."}
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
        default=0.3,
        metadata={
            "help": "vLLM GPU memory utilization. Lower this if running into OOM with RMs loaded."
        },
    )
    secondary_rm_name: Optional[str] = field(
        default="Ray2333/GRM-Gemma-2B-sftreg",
        metadata={
            "help": "Name of the secondary reward model for cross-validation against gold RM. "
                    "Set to 'none' to disable."
        },
    )
    skip_validation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Skip prompt/conversation length validation (pass None for max_prompt_length and max_conversation_length)."
        },
    )
    length_config: Optional[str] = field(
        default="default",
        metadata={
            "help": "Name of the length config from DATASET_LENGTH_CONFIGS. "
            "Use 'alpacafarm_paper' for the paper comparison (520/256/776)."
        },
    )
    auto_prompt_length: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Measure actual max prompt length from the dataset with the "
            "policy tokenizer and use it for vLLM memory allocation. "
            "Overrides --max_length."
        },
    )
    evaluate_chosen_responses: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Evaluate the chosen responses from the dataset with the reward models instead of generating from a policy."
        },
    )


# =========================================================================
# Result container yielded by response providers
# =========================================================================

@dataclass
class CheckpointResponses:
    """Container yielded by response providers for each checkpoint."""
    checkpoint_num: int
    responses: list
    kl_results: Optional[dict] = None


# =========================================================================
# Low-level helpers
# =========================================================================

def load_reward_model_impl(model_path_or_name, device):
    model, tokenizer = load_reward_model(
        model_path_or_name, reasoning=False, device=device
    )
    return model, tokenizer


def score_responses_with_rm(
    responses,
    prompt_messages_list,
    args,
    rm_model_or_name,
    rm_tokenizer=None,
    *,
    checkpoint_num=None,
):
    """Score one checkpoint's responses with a reward model."""
    loaded_here = False
    if isinstance(rm_model_or_name, str):
        rm_model, rm_tokenizer = load_reward_model_impl(rm_model_or_name, args.device)
        loaded_here = True
    else:
        rm_model = rm_model_or_name

    if rm_tokenizer is None:
        raise ValueError("rm_tokenizer must be provided when scoring reward models.")

    try:
        if len(prompt_messages_list) != len(responses):
            raise ValueError(
                "Prompt/response mismatch: "
                f"{len(prompt_messages_list)} prompts vs {len(responses)} responses."
            )

        texts = []
        context = f"Evaluation checkpoint {checkpoint_num}"

        for sample_id, (prompt_messages, response) in enumerate(
            zip(prompt_messages_list, responses)
        ):
            full_conv = list(prompt_messages) + [{"role": "assistant", "content": response}]
            _, full_text, _ = format_and_validate_preference_sample(
                full_conv,
                rm_tokenizer,
                length_config="default",
                skip_validation=True,
                sample_id=sample_id,
                context=context,
            )
            texts.append(full_text)
        return (
            get_reward_rm(
                rm_model,
                rm_tokenizer,
                texts,
                batch_size=args.batch_size,
                device=args.device,
            )
            .cpu()
            .float()
            .numpy()
        )
    finally:
        if loaded_here:
            del rm_model, rm_tokenizer
            gc.collect()
            torch.cuda.empty_cache()


def coerce_list(values):
    if isinstance(values, list):
        return values
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


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

        # Compute per-token log probs without materializing full vocab-sized tensors
        # Process one sequence at a time to avoid OOM on the large [batch, seq, vocab] tensor
        shift_labels = input_ids[:, 1:]

        token_log_probs_list = []
        for j in range(input_ids.size(0)):
            # Only compute log_softmax for this single sequence (saves ~batch_size x memory)
            seq_logits = logits[j, :-1, :]  # [seq_len-1, vocab]
            seq_labels = shift_labels[j]     # [seq_len-1]
            # gather first, then we don't need the full log_softmax output
            seq_log_probs = torch.log_softmax(seq_logits, dim=-1)
            seq_token_lp = seq_log_probs.gather(-1, seq_labels.unsqueeze(-1)).squeeze(-1)
            token_log_probs_list.append(seq_token_lp)
            del seq_logits, seq_log_probs
        token_log_probs = torch.stack(token_log_probs_list)
        del logits

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
        stop_token_ids = get_generation_stop_token_ids(tokenizer)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            logprobs=1 if collect_logprobs else None,
            stop_token_ids=stop_token_ids,
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
    """Loads a policy model from a path (supports LoRA adapters)."""
    print(f"Loading model from {model_path}")
    model, _ = load_policy_and_tokenizer(model_path)
    model = model.to(device)
    model.eval()
    return model


def update_vllm_weights(llm, model_path, device="cpu"):
    """Update vLLM weights from a Hugging Face checkpoint."""
    print(f"Updating vLLM weights from {model_path}")
    llm.collective_rpc("load_weights_from_path", args=(model_path,))


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
    raise NotImplementedError(
        "LLM judge evaluation is not currently supported. "
        "Known issue: `prompts` passed here are chat-template-formatted strings "
        "(e.g. containing <|im_start|> tokens) but get plugged into the Skywork "
        "judge template as the raw 'question', corrupting judge input. "
        "Fix: pass structured prompt_messages and extract the raw user question."
    )
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


# =========================================================================
# Evaluation modules
# =========================================================================

class LoadedRewardModels:
    """Loads configured reward models once and keeps them in memory."""

    def __init__(self, args):
        self.args = args
        self._models = {}  # name -> (model, tokenizer)
        self._chosen_scores = {}  # label -> np.ndarray, populated by precompute_chosen_scores

        # Always load gold RM
        print(f"Loading gold RM ({args.gold_rm_name})...")
        self._models["gold_rm"] = load_reward_model_impl(args.gold_rm_name, args.device)

        if args.evaluate_with_training_rm:
            print(f"Loading training RM ({args.training_rm_path})...")
            self._models["training_rm"] = load_reward_model_impl(args.training_rm_path, args.device)

        if args.secondary_rm_name and args.secondary_rm_name.lower() != "none":
            print(f"Loading secondary RM ({args.secondary_rm_name})...")
            self._models["secondary_rm"] = load_reward_model_impl(args.secondary_rm_name, args.device)

    def unload(self):
        for name in list(self._models):
            del self._models[name]
        self._models.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def precompute_chosen_scores(self, dataset, prompt_messages_list, args):
        """Score chosen responses with all loaded RMs once and cache to disk.

        Results are cached as JSON files keyed by (rm_name, dataset_name,
        dataset_split_size).  On subsequent runs with the same configuration
        the cached scores are loaded instead of recomputed.
        """
        chosen_responses = [ex["chosen"][-1]["content"] for ex in dataset]

        # Deterministic cache key from dataset identity + size
        cache_id = hashlib.sha256(
            f"{args.dataset_name}:{len(chosen_responses)}".encode()
        ).hexdigest()[:16]

        cache_dir = os.path.join(
            os.path.dirname(args.output_file) or ".", "chosen_scores_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)

        self._chosen_scores = {}  # label -> np.ndarray

        for label, (model, tokenizer) in self._models.items():
            rm_name = model.config._name_or_path
            safe_name = rm_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            if len(safe_name) > 200:
                safe_name = safe_name[-200:]
            cache_file = os.path.join(cache_dir, f"{safe_name}_{cache_id}.json")

            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                if (
                    cached.get("version") == CHOSEN_SCORES_CACHE_VERSION
                    and cached.get("dataset_name") == args.dataset_name
                    and cached.get("num_samples") == len(chosen_responses)
                    and "scores" in cached
                ):
                    self._chosen_scores[label] = np.array(cached["scores"], dtype=np.float64)
                    print(
                        f"[ChosenScores] {label} ({rm_name}): loaded {len(cached['scores'])} "
                        f"cached scores (mean={np.mean(cached['scores']):.4f})"
                    )
                    continue

            print(f"[ChosenScores] {label} ({rm_name}): scoring {len(chosen_responses)} chosen responses...")
            scores = score_responses_with_rm(
                chosen_responses, prompt_messages_list, args, model,
                rm_tokenizer=tokenizer, checkpoint_num="chosen",
            )
            self._chosen_scores[label] = scores

            cache_data = {
                "version": CHOSEN_SCORES_CACHE_VERSION,
                "rm_name": rm_name,
                "dataset_name": args.dataset_name,
                "num_samples": len(chosen_responses),
                "scores": scores.tolist(),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "computed_at": datetime.now().isoformat(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            print(
                f"[ChosenScores] {label} ({rm_name}): mean={np.mean(scores):.4f}, "
                f"std={np.std(scores):.4f}, cached to {cache_file}"
            )

    def score_all(self, responses, prompt_messages_list, args, checkpoint_num):
        """Score responses with all loaded reward models.

        Returns (results_dict, gold_rm_scores).
        """
        results = {}

        def _score(label, model, tokenizer):
            scores = score_responses_with_rm(
                responses, prompt_messages_list, args, model,
                rm_tokenizer=tokenizer, checkpoint_num=checkpoint_num,
            )
            results[f"{label}/mean"] = float(np.mean(scores))
            results[f"{label}/std"] = float(np.std(scores))
            if not args.disable_wandb:
                results[f"{label}/scores_hist"] = wandb.Histogram(scores)

            # Win rate vs chosen
            chosen = self._chosen_scores[label]
            n = min(len(scores), len(chosen))
            wins = (scores[:n] > chosen[:n]).sum()
            ties = (scores[:n] == chosen[:n]).sum()
            win_rate = float(wins) / n
            win_or_tie_rate = float(wins + ties) / n
            results[f"{label}/win_rate_vs_chosen"] = win_rate
            results[f"{label}/win_or_tie_rate_vs_chosen"] = win_or_tie_rate

            return scores

        gold_rm_scores = _score("gold_rm", *self._models["gold_rm"])

        if "training_rm" in self._models:
            _score("training_rm", *self._models["training_rm"])

        if "secondary_rm" in self._models:
            _score("secondary_rm", *self._models["secondary_rm"])

        return results, gold_rm_scores


def compute_judge_verdicts(
    original_prompts, policy_responses, baseline_responses, args, num_responses=1,
):
    """Run LLM-as-judge evaluation on policy vs baseline responses.

    Returns (results_dict, verdicts, judge_responses).
    """
    expanded_prompts = [p for p in original_prompts for _ in range(num_responses)]
    expanded_baselines = [b for b in baseline_responses for _ in range(num_responses)]
    expected = len(original_prompts) * num_responses
    if len(policy_responses) != expected:
        raise ValueError(
            f"Expected {expected} policy responses, got {len(policy_responses)}."
        )

    verdicts, judge_responses = get_llm_judge_verdicts(
        expanded_prompts, policy_responses, expanded_baselines, args
    )

    wins = verdicts.count(1)
    losses = verdicts.count(-1)
    ties = verdicts.count(0)
    total = len(verdicts)

    results = {
        "llm_judge/win_rate": wins / total if total > 0 else 0,
        "llm_judge/loss_rate": losses / total if total > 0 else 0,
        "llm_judge/tie_rate": ties / total if total > 0 else 0,
        "llm_judge/mean": np.mean(verdicts) if total > 0 else 0,
        "llm_judge/mean_no_tie": (
            (wins - losses) / (wins + losses) if (wins + losses) > 0 else 0
        ),
        "llm_judge/total_comparisons": total,
    }
    return results, verdicts, judge_responses


def evaluate_responses(cr, prompt_messages_list, original_prompts, baseline_responses, args, num_responses, loaded_rms):
    """Run all configured evaluations on a CheckpointResponses.

    Returns (checkpoint_results, gold_rm_scores).
    """
    checkpoint_results = {"checkpoint": cr.checkpoint_num}

    if cr.kl_results:
        checkpoint_results.update(cr.kl_results)

    # RM evaluation
    rm_results, gold_rm_scores = loaded_rms.score_all(
        cr.responses, prompt_messages_list, args, cr.checkpoint_num,
    )
    checkpoint_results.update(rm_results)

    # LLM judge evaluation
    if args.evaluate_with_llm_judge:
        judge_results, verdicts, judge_resp = compute_judge_verdicts(
            original_prompts, cr.responses, baseline_responses, args,
            num_responses=num_responses,
        )
        checkpoint_results.update(judge_results)
        checkpoint_results["_judge_verdicts"] = verdicts
        checkpoint_results["_judge_responses"] = judge_resp

    return checkpoint_results, gold_rm_scores


# =========================================================================
# Response providers (generators)
# =========================================================================

def chosen_responses_provider(dataset):
    """Yield chosen responses from the dataset (single iteration, no model needed)."""
    responses = [ex["chosen"][-1]["content"] for ex in dataset]
    print(f"Evaluating {len(responses)} chosen responses from the dataset...")
    yield CheckpointResponses(checkpoint_num=0, responses=responses)


def vllm_responses_provider(
    checkpoints, args, original_prompts, policy_tokenizer,
    vllm_base_model_path, first_checkpoint_path, single_model_path,
    stop_token_ids, num_responses,
):
    """Yield generated responses for each checkpoint via vLLM.

    Owns the vLLM lifecycle: initialises the engine, swaps weights per
    checkpoint, and cleans up in the finally block.
    """
    base_policy_model = None
    if args.kl_base_model_path:
        print(f"Loading base policy for KL from {args.kl_base_model_path}...")
        base_policy_model = load_policy_model(
            args.kl_base_model_path, policy_tokenizer, args.device
        )

    generation_sampling_params = None
    if args.evaluate_with_llm_judge:
        generation_sampling_params = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=args.max_new_tokens,
            n=num_responses, stop_token_ids=stop_token_ids,
        )

    print(f"Initializing vLLM with {vllm_base_model_path}")
    llm = LLM(
        model=vllm_base_model_path,
        tokenizer=vllm_base_model_path,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_length + args.max_new_tokens,
        trust_remote_code=True,
        language_model_only=True,
        worker_extension_cls="vllm_weight_loader.WeightLoaderExtension",
    )
    if first_checkpoint_path != vllm_base_model_path:
        update_vllm_weights(llm, first_checkpoint_path)

    try:
        for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
            checkpoint_path = single_model_path or os.path.join(args.checkpoints_dir, checkpoint)
            checkpoint_num = int(checkpoint.split("-")[1])
            print(f"\nEvaluating {checkpoint}")

            if checkpoint != checkpoints[0]:
                update_vllm_weights(llm, checkpoint_path)

            (
                responses,
                full_ids_list,
                prompt_lens_list,
                policy_mean_logprobs,
                policy_token_logprobs,
            ) = generate_responses_vllm(
                llm,
                original_prompts,
                policy_tokenizer,
                args,
                sampling_params=generation_sampling_params,
                collect_logprobs=base_policy_model is not None,
            )

            # KL calculation (must happen here while vllm logprobs are available)
            kl_results = None
            if base_policy_model is not None:
                (
                    _,
                    base_mean_lp,
                    base_token_lp_list,
                ) = get_log_probs_from_ids(
                    base_policy_model,
                    full_ids_list,
                    prompt_lens_list,
                    args.device,
                    batch_size=4,
                )

                kl_per_sample = policy_mean_logprobs - np.array(base_mean_lp)
                kl_mean = float(np.mean(kl_per_sample))
                kl_std = float(np.std(kl_per_sample))

                kl_grpo_per_sample = []
                for pol_lp, ref_lp in zip(
                    policy_token_logprobs, base_token_lp_list
                ):
                    min_len = min(len(pol_lp), len(ref_lp))
                    pol_lp = np.array(pol_lp[:min_len])
                    ref_lp = np.array(ref_lp[:min_len])
                    diff = ref_lp - pol_lp
                    per_token_kl = np.exp(diff) - diff - 1
                    kl_grpo_per_sample.append(np.mean(per_token_kl))

                kl_grpo_mean = float(np.mean(kl_grpo_per_sample))
                kl_grpo_std = float(np.std(kl_grpo_per_sample))

                kl_results = {
                    "kl/mean": kl_mean,
                    "kl/std": kl_std,
                    "kl/grpo_mean": kl_grpo_mean,
                    "kl/grpo_std": kl_grpo_std,
                }

            yield CheckpointResponses(
                checkpoint_num=checkpoint_num,
                responses=responses,
                kl_results=kl_results,
            )
    finally:
        try:
            destroy_model_parallel()
            del llm.llm_engine.model_executor
            del llm
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


# =========================================================================
# Main
# =========================================================================

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

    # --- Load dataset ---
    print("Loading evaluation dataset...")
    dataset_obj = load_dataset(args.dataset_name)
    split_priority = ("validation", "test", "train")
    if hasattr(dataset_obj, "keys"):
        available_splits = list(dataset_obj.keys())
        print(f"Available dataset splits: {available_splits}")
        split = next((s for s in split_priority if s in dataset_obj), None)
        if split is None:
            split = available_splits[0]
            print(
                f"None of {split_priority} found. Falling back to first available split: {split}"
            )
        else:
            print(f"Using dataset split by priority {split_priority}: {split}")
        dataset = dataset_obj[split]
    else:
        split = "<single>"
        print("Dataset has no named splits; using the loaded dataset as-is.")
        dataset = dataset_obj
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

    if "chosen" not in dataset.column_names:
        raise ValueError("Dataset must have a 'chosen' column.")

    # --- Resolve checkpoints and policy tokenizer (skipped for chosen-only evaluation) ---
    checkpoints = []
    single_model_path = None
    first_checkpoint_path = None
    vllm_base_model_path = None
    policy_tokenizer = None
    stop_token_ids = None

    if not args.evaluate_chosen_responses:
        if os.path.isdir(args.checkpoints_dir):
            checkpoints = sorted(
                [d for d in os.listdir(args.checkpoints_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1]),
            )

        if not checkpoints:
            basename = os.path.basename(args.checkpoints_dir.rstrip(os.sep))
            if basename.startswith("checkpoint-"):
                try:
                    step = int(basename.split("-")[1])
                except (ValueError, IndexError):
                    step = 0
            else:
                step = 0
            single_model_path = args.checkpoints_dir
            checkpoints = [f"checkpoint-{step}"]

        first_checkpoint_path = single_model_path or os.path.join(args.checkpoints_dir, checkpoints[0])

        if _is_lora_checkpoint(first_checkpoint_path):
            vllm_base_model_path = _get_lora_base_model_path(first_checkpoint_path)
            print(f"LoRA adapter detected, base model: {vllm_base_model_path}")
        else:
            vllm_base_model_path = first_checkpoint_path

        print("Loading tokenizer...")
        policy_tokenizer = AutoTokenizer.from_pretrained(
            first_checkpoint_path, trust_remote_code=True,
        )
        setup_tokenizer(policy_tokenizer, model_name=vllm_base_model_path)
        stop_token_ids = get_generation_stop_token_ids(policy_tokenizer)

        if args.debug:
            print("Debug mode: using only first checkpoint")
            checkpoints = checkpoints[:1]

    # --- Resolve auto length config ---
    # Auto-detection measures actual dataset token lengths to set tight vLLM
    # memory allocation; validation is skipped since the dataset is pre-filtered.
    is_auto = args.length_config == "auto"
    if is_auto and policy_tokenizer is not None:
        auto_cfg = compute_length_config_from_dataset(dataset, policy_tokenizer)
        args.max_length = auto_cfg["max_prompt_tokens"]
        # max_new_tokens is a generation budget — keep the CLI value.

    # --- Prepare dataset: validate and format prompts ---
    # format_and_validate_preference_sample is the single source of truth for
    # validation and chat-template formatting.  When we have a policy tokenizer
    # (generation path) it also produces the formatted prompt strings for vLLM.
    def extract_fields(example, idx):
        result = {"prompt_messages": example["chosen"][:-1]}
        if policy_tokenizer is not None:
            prompt_text, _, _ = format_and_validate_preference_sample(
                example["chosen"],
                policy_tokenizer,
                rejected_messages=example.get("rejected"),
                length_config=args.length_config if not is_auto else "default",
                skip_validation=args.skip_validation or is_auto,
                sample_id=idx,
                context="Evaluation",
            )
            result["prompt"] = prompt_text
        return result

    dataset = dataset.map(extract_fields, num_proc=1, with_indices=True)
    prompt_messages_list = dataset["prompt_messages"]
    original_prompts = coerce_list(dataset["prompt"]) if policy_tokenizer is not None else []

    print(f"Using {len(prompt_messages_list)} prompts for evaluation")

    num_responses = args.num_responses_per_prompt or 1

    # --- Prepare LLM judge baselines (if needed) ---
    baseline_responses = None
    full_eval_data = []
    if args.evaluate_with_llm_judge:
        if not args.baseline_model_path and not args.use_dataset_response_as_baseline:
            raise ValueError(
                "Either --baseline_model_path or --use_dataset_response_as_baseline must be specified for LLM judge evaluation."
            )
        if args.use_dataset_response_as_baseline:
            print("Using dataset 'chosen' column as baseline response.")
            baseline_responses = coerce_list(
                dataset.map(lambda ex: {"response": ex["chosen"][-1]["content"]})["response"]
            )
        else:
            print(f"Loading baseline model from {args.baseline_model_path}")
            baseline_llm = LLM(
                model=args.baseline_model_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_length + args.max_new_tokens,
                trust_remote_code=True,
                language_model_only=True,
            )
            baseline_sampling_params = SamplingParams(
                temperature=0, max_tokens=args.max_new_tokens, n=1,
                stop_token_ids=stop_token_ids,
            )
            print("Generating baseline responses...")
            baseline_responses, _, _, _, _ = generate_responses_vllm(
                baseline_llm, original_prompts, policy_tokenizer, args,
                sampling_params=baseline_sampling_params,
            )
            destroy_model_parallel()
            del baseline_llm.llm_engine.model_executor
            del baseline_llm
            gc.collect()
            torch.cuda.empty_cache()

        for i in range(len(original_prompts)):
            full_eval_data.append(
                {"prompt": original_prompts[i], "baseline_response": baseline_responses[i], "checkpoints": {}}
            )

    # --- Build response provider ---
    if args.evaluate_chosen_responses:
        provider = chosen_responses_provider(dataset)
    else:
        provider = vllm_responses_provider(
            checkpoints, args, original_prompts, policy_tokenizer,
            vllm_base_model_path, first_checkpoint_path, single_model_path,
            stop_token_ids, num_responses,
        )

    # --- Load reward models once ---
    loaded_rms = LoadedRewardModels(args)

    # --- Precompute chosen scores for win-rate metric ---
    if not args.evaluate_chosen_responses:
        loaded_rms.precompute_chosen_scores(dataset, prompt_messages_list, args)

    # --- Main evaluation loop ---
    results = []
    kl_reward_data = []

    try:
        for cr in provider:
            try:
                checkpoint_results, gold_rm_scores = evaluate_responses(
                    cr, prompt_messages_list, original_prompts,
                    baseline_responses, args, num_responses, loaded_rms,
                )

                # Collect judge full_eval_data (if judge ran)
                verdicts = checkpoint_results.pop("_judge_verdicts", None)
                judge_resp = checkpoint_results.pop("_judge_responses", None)
                if verdicts is not None and full_eval_data:
                    for i in range(len(original_prompts)):
                        start = i * num_responses
                        end = start + num_responses
                        full_eval_data[i]["checkpoints"][cr.checkpoint_num] = {
                            "policy_responses": cr.responses[start:end],
                            "llm_judge_responses": judge_resp[start:end],
                            "llm_judge_verdicts": verdicts[start:end],
                        }

                # Accumulate KL-reward data for final aggregate plot
                if "kl/grpo_mean" in checkpoint_results:
                    kl_reward_data.append(
                        {
                            "checkpoint": cr.checkpoint_num,
                            "kl_grpo": checkpoint_results["kl/grpo_mean"],
                            "gold_reward": float(np.mean(gold_rm_scores)),
                        }
                    )

                if not args.disable_wandb:
                    wandb.log(checkpoint_results, step=cr.checkpoint_num)

                # Strip wandb-only objects before appending to CSV results.
                results.append(
                    {k: v for k, v in checkpoint_results.items() if not isinstance(v, wandb.Histogram)}
                )

            except Exception as e:
                print(f"Error evaluating checkpoint {cr.checkpoint_num}: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"Evaluation failed for checkpoint {cr.checkpoint_num}. "
                    "Failing fast to avoid silently incomplete results."
                ) from e
    finally:
        loaded_rms.unload()

    # --- Aggregate KL-reward plot ---
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

    # --- Save results ---
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
