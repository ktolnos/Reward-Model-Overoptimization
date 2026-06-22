"""vLLM lifecycle and response-generation helpers.

Splits the old ``vllm_responses_provider`` into three pieces so the new main
loop can mix-and-match:

- ``initialize_vllm``: create the engine once, with the correct max_model_len.
- ``update_vllm_weights``: hot-swap in checkpoint weights.
- ``generate_responses_vllm``: one generation pass for a batch of prompts.
- ``teardown_vllm``: release GPU memory.

Logprob collection: callers that need KL or similar pass ``collect_logprobs=True``
and read ``full_ids_list`` / ``policy_mean_logprobs`` / ``policy_token_logprobs``
off the returned ``GenerationResult``.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

import qwen35_vllm_patch  # noqa: F401 — must be imported before any vLLM code
from data_utils import (
    _get_lora_base_model_path,
    _is_lora_checkpoint,
    get_generation_stop_token_ids,
)

from .types import GenerationConfig, GenerationResult


def resolve_vllm_base_model(first_checkpoint_path: str) -> str:
    if _is_lora_checkpoint(first_checkpoint_path):
        base = _get_lora_base_model_path(first_checkpoint_path)
        print(f"LoRA adapter detected, base model: {base}")
        return base
    return first_checkpoint_path


def initialize_vllm(
    *,
    base_model_path: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> LLM:
    print(f"Initializing vLLM with {base_model_path} (max_model_len={max_model_len})")
    llm = LLM(
        model=base_model_path,
        tokenizer=base_model_path,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        language_model_only=True,
        worker_extension_cls="vllm_weight_loader.WeightLoaderExtension",
    )
    return llm


def update_vllm_weights(llm: LLM, model_path: str) -> None:
    print(f"Updating vLLM weights from {model_path}")
    llm.collective_rpc("load_weights_from_path", args=(model_path,))


def teardown_vllm(llm: Optional[LLM]) -> None:
    if llm is None:
        return
    try:
        destroy_model_parallel()
        del llm.llm_engine.model_executor
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def vllm_session(
    *,
    base_model_path: str,
    initial_checkpoint_path: str,
    max_model_len: int,
    gpu_memory_utilization: float,
):
    """Context manager owning the vLLM engine lifecycle."""
    llm = initialize_vllm(
        base_model_path=base_model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    try:
        if initial_checkpoint_path != base_model_path:
            update_vllm_weights(llm, initial_checkpoint_path)
        yield llm
    finally:
        teardown_vllm(llm)


def _strip_thinking(text: str) -> str:
    """Strip everything up to and including the last ``</think>`` tag."""
    tag = "</think>"
    idx = text.rfind(tag)
    if idx == -1:
        return text
    return text[idx + len(tag):].lstrip()


def generate_responses_vllm(
    llm: LLM,
    prompts: List[str],
    tokenizer,
    gen_config: GenerationConfig,
    *,
    strip_thinking: bool = True,
) -> GenerationResult:
    """Generate responses for a batch of prompts under ``gen_config``.

    Expands vLLM's nested output (n samples per prompt) into flat lists of
    length ``len(prompts) * gen_config.n_responses_per_example``.
    """
    sampling_params = gen_config.sampling_params
    # Make sure stop tokens are set; some benchmarks construct SamplingParams
    # without stop_token_ids (vLLM would then generate past chat turn markers).
    if not sampling_params.stop_token_ids:
        sampling_params.stop_token_ids = get_generation_stop_token_ids(tokenizer)

    outputs = llm.generate(prompts, sampling_params)

    raw_responses: List[str] = []
    responses: List[str] = []
    finish_reasons: List[str] = []
    response_token_lens: List[int] = []
    full_ids_list: List[List[int]] = []
    prompt_lens_list: List[int] = []
    policy_mean_logprobs: List[float] = []
    policy_token_logprobs: List[List[float]] = []

    for output in outputs:
        for completion in output.outputs:
            raw = completion.text
            raw_responses.append(raw)
            responses.append(_strip_thinking(raw) if strip_thinking else raw)
            finish_reasons.append(completion.finish_reason)
            response_token_lens.append(len(completion.token_ids))

            if gen_config.collect_logprobs:
                prompt_ids = list(output.prompt_token_ids)
                response_ids = list(completion.token_ids)
                full_ids_list.append(prompt_ids + response_ids)
                prompt_lens_list.append(len(prompt_ids))

                seq_lp = completion.logprobs
                tok_lps: List[float] = []
                if seq_lp:
                    for lp, t_id in zip(seq_lp, response_ids):
                        if not lp:
                            continue
                        entry = lp.get(t_id)
                        if entry is None:
                            continue
                        tok_lps.append(entry.logprob)
                mean_lp = float(sum(tok_lps) / len(tok_lps)) if tok_lps else 0.0
                policy_mean_logprobs.append(mean_lp)
                policy_token_logprobs.append(tok_lps)

    result = GenerationResult(
        responses=responses,
        raw_responses=raw_responses,
        finish_reasons=finish_reasons,
        n_responses_per_example=gen_config.n_responses_per_example,
        response_token_lens=response_token_lens,
    )
    if gen_config.collect_logprobs:
        result.full_ids_list = full_ids_list
        result.prompt_lens_list = prompt_lens_list
        result.policy_mean_logprobs = np.array(policy_mean_logprobs)
        result.policy_token_logprobs = policy_token_logprobs
    return result


def default_sampling_params(
    tokenizer,
    max_new_tokens: int,
    *,
    temperature: float = 0,
    top_p: float = 1.0,
    n: int = 1,
    collect_logprobs: bool = False,
) -> SamplingParams:
    """SamplingParams with the repo's stop tokens pre-filled."""
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=n,
        logprobs=1 if collect_logprobs else None,
        stop_token_ids=get_generation_stop_token_ids(tokenizer),
    )


# ---------------------------------------------------------------------------
# Base-policy KL helper — kept here because it's a generation-adjacent utility.
# ---------------------------------------------------------------------------

def get_log_probs_from_ids(model, full_ids_list, prompt_lens_list, device, batch_size=4):
    """Compute per-token log probabilities directly from token IDs.

    See the original ``evaluate_policy.get_log_probs_from_ids`` for semantics.
    """
    all_sum_log_probs = []
    all_mean_log_probs = []
    all_token_log_probs = []

    for i in range(0, len(full_ids_list), batch_size):
        batch_full_ids = full_ids_list[i:i + batch_size]
        batch_prompt_lens = prompt_lens_list[i:i + batch_size]

        max_len = max(len(ids) for ids in batch_full_ids)
        padded_ids = []
        attention_masks = []
        for ids in batch_full_ids:
            pad_len = max_len - len(ids)
            padded = [model.config.pad_token_id] * pad_len + list(ids)
            mask = [0] * pad_len + [1] * len(ids)
            padded_ids.append(padded)
            attention_masks.append(mask)

        input_ids = torch.tensor(padded_ids, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_labels = input_ids[:, 1:]
        token_log_probs_list = []
        for j in range(input_ids.size(0)):
            seq_logits = logits[j, :-1, :]
            seq_labels = shift_labels[j]
            seq_log_probs = torch.log_softmax(seq_logits, dim=-1)
            seq_token_lp = seq_log_probs.gather(-1, seq_labels.unsqueeze(-1)).squeeze(-1)
            token_log_probs_list.append(seq_token_lp)
            del seq_logits, seq_log_probs
        token_log_probs = torch.stack(token_log_probs_list)
        del logits

        for j in range(input_ids.size(0)):
            pad_len = (attention_mask[j] == 0).sum().item()
            prompt_len = batch_prompt_lens[j]
            response_start_idx = pad_len + prompt_len - 1
            valid_len = attention_mask[j].sum().item()
            response_end_idx = pad_len + valid_len - 1

            if response_start_idx >= response_end_idx:
                all_sum_log_probs.append(0.0)
                all_mean_log_probs.append(0.0)
                all_token_log_probs.append([])
                continue

            response_log_probs = token_log_probs[j, response_start_idx:response_end_idx]
            all_sum_log_probs.append(response_log_probs.sum().item())
            all_mean_log_probs.append(response_log_probs.mean().item())
            all_token_log_probs.append(response_log_probs.cpu().tolist())

    return all_sum_log_probs, all_mean_log_probs, all_token_log_probs
