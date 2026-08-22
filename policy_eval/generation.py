"""vLLM lifecycle and response-generation helpers.

Splits the old ``vllm_responses_provider`` into three pieces so the new main
loop can mix-and-match:

- ``initialize_vllm``: create the engine once, with the correct max_model_len.
- ``update_vllm_weights``: hot-swap in checkpoint weights (+ prefix-cache reset).
- ``generate_responses_vllm``: one generation pass for a batch of prompts.
- ``teardown_vllm``: release GPU memory.

KL support: callers pass ``collect_logprobs=True`` to record token ids
(``full_ids_list`` / ``prompt_lens_list``) on the ``GenerationResult``;
``teacher_forced_response_logprobs`` then scores those ids under whichever
weights are currently loaded, so policy and base logprobs come from the same
engine and kernels without a second copy of the model in memory.
"""
from __future__ import annotations

import gc
import time
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
    # KV blocks cached under the previous weights are wrong for the new ones;
    # prefix caching is on by default and would silently serve stale KV for
    # repeated prompt prefixes (same prompts every checkpoint, and the
    # teacher-forced KL passes share the full prompt+response prefix).
    llm.reset_prefix_cache()


def wait_for_gpu_memory(
    min_free_fraction: float,
    *,
    timeout_s: float = 180.0,
    poll_s: float = 2.0,
) -> None:
    """Block until device 0 has ``min_free_fraction`` of its memory free.

    vLLM's engine startup hard-fails unless free memory >= gpu_memory_utilization
    * total. In the eval pipeline the judge engine is built right after the policy
    engine is torn down, and a vLLM engine runs the model in a separate EngineCore
    process whose GPU memory the OS only reclaims once that process actually exits
    -- which can lag ``teardown_vllm`` by several seconds. Polling here lets that
    reclaim finish before the next engine's startup check runs, turning an
    intermittent, timing-dependent startup OOM into a short wait. Raises if the
    memory never frees (a genuine leak, not a reclaim lag).
    """
    free, total = torch.cuda.mem_get_info(0)
    needed = int(min_free_fraction * total)
    deadline = time.monotonic() + timeout_s
    reaped = False
    while free < needed:
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        if free >= needed:
            break
        if time.monotonic() >= deadline:
            # Last resort: gc/empty_cache above only touch this process, but the
            # holder is usually a vLLM EngineCore subprocess that never exited.
            # Kill any such orphans once and give the OS a moment to reclaim.
            if not reaped:
                reaped = True
                _reap_processes(_vllm_engine_procs(), timeout_s=0.0)
                gc.collect()
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(0)
                deadline = time.monotonic() + 15.0
                if free >= needed:
                    break
                continue
            raise RuntimeError(
                f"GPU 0 has only {free / 2**30:.2f} GiB free but "
                f"{needed / 2**30:.2f} GiB is needed for gpu_memory_utilization="
                f"{min_free_fraction} after waiting {timeout_s:.0f}s; a prior "
                f"model was likely not released."
            )
        time.sleep(poll_s)


def _vllm_engine_procs():
    """The vLLM EngineCore worker processes spawned by this process.

    vLLM V1 runs the model in separate ``EngineCore`` subprocess(es) that title
    themselves with "EngineCore"; those hold the GPU weights + KV cache. We
    filter children by title so teardown never touches unrelated helpers (e.g.
    the wandb service process, which also runs as a child).
    """
    try:
        import psutil
    except ImportError:
        return []
    try:
        me = psutil.Process()
    except psutil.NoSuchProcess:
        return []
    procs = []
    for c in me.children(recursive=True):
        try:
            hay = f"{c.name()} {' '.join(c.cmdline())}".lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        if "enginecore" in hay or "vllm" in hay:
            procs.append(c)
    return procs


def teardown_vllm(llm: Optional[LLM], *, timeout_s: float = 60.0) -> None:
    if llm is None:
        return
    # vLLM V1 runs the model in a separate EngineCore process; the weights and
    # KV cache live there, so freeing anything in this process is not enough --
    # the OS only reclaims that GPU memory once the process actually exits.
    #
    # ``engine_core.shutdown()`` only *signals* that process; it returns without
    # waiting, and under async scheduling the exit can lag or hang. A caller that
    # then loads a big model (the LLM judge at 0.9 util) races that reclaim and
    # hits a startup OOM. So we snapshot the worker processes first, signal
    # shutdown, then block until they are truly gone (killing any that overstay)
    # so this function only returns once the GPU memory is actually free.
    procs = _vllm_engine_procs()
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:  # best-effort: still reap the processes below
        print(f"[teardown_vllm] engine_core.shutdown() raised {e!r}; reaping processes")
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    _reap_processes(procs, timeout_s=timeout_s)
    gc.collect()
    torch.cuda.empty_cache()


def _reap_processes(procs, *, timeout_s: float) -> None:
    """Wait for ``procs`` to exit, escalating to SIGTERM then SIGKILL."""
    if not procs:
        return
    try:
        import psutil
    except ImportError:
        return
    _, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for p in alive:
        print(f"[teardown_vllm] EngineCore process {p.pid} still alive after "
              f"{timeout_s:.0f}s; sending SIGTERM.")
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=10.0)
    for p in alive:
        print(f"[teardown_vllm] EngineCore process {p.pid} ignored SIGTERM; sending SIGKILL.")
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10.0)


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

    for output in outputs:
        for completion in output.outputs:
            raw = completion.text
            raw_responses.append(raw)
            responses.append(_strip_thinking(raw) if strip_thinking else raw)
            finish_reasons.append(completion.finish_reason)
            response_token_lens.append(len(completion.token_ids))

            if gen_config.collect_logprobs:
                prompt_ids = list(output.prompt_token_ids)
                full_ids_list.append(prompt_ids + list(completion.token_ids))
                prompt_lens_list.append(len(prompt_ids))

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
    return result


# ---------------------------------------------------------------------------
# Teacher-forced logprobs + KL estimators — generation-adjacent utilities.
# ---------------------------------------------------------------------------

def teacher_forced_response_logprobs(llm: LLM, full_ids_list, prompt_lens_list):
    """Per-token logprobs of each response segment under the engine's current
    weights.

    Teacher-forces each recorded (prompt + response) token sequence through a
    ``prompt_logprobs`` pass, so every response token gets a logprob at its
    position (no gaps) from the same engine/kernels — for whichever weights
    are loaded: the policy checkpoint, or the KL base model after
    ``update_vllm_weights``. This is what lets KL use one engine instead of a
    second HF copy of the model in GPU memory.

    Sequences that already fill the engine's ``max_model_len`` are fed without
    their final token (vLLM requires room for one generated token); that
    token is dropped identically for the policy and base passes, so the two
    sides stay positionally aligned.

    Returns ``(mean_logprobs, token_logprobs_list)``.
    """
    from vllm.inputs import TokensPrompt

    # The prompt_logprobs pass materializes a float32 full-vocab tensor per
    # scheduled request (chunk × vocab × 4B, so 1.31 GiB for a 1415-token
    # request at Qwen3.5's 248320 vocab), and that transient sits outside the
    # engine's gpu_memory_utilization budget. This process meanwhile keeps the
    # reward models resident and, having just scored a benchmark with them, is
    # holding their allocator high-water mark -- reserved but idle, and
    # invisible to the EngineCore subprocess that needs it. Release it to the
    # driver first, or the pass OOMs the engine (EngineDeadError) on a card
    # this crowded.
    gc.collect()
    torch.cuda.empty_cache()

    max_model_len = llm.llm_engine.model_config.max_model_len
    sampling = SamplingParams(
        temperature=0, max_tokens=1, prompt_logprobs=0, detokenize=False,
    )
    prompts = []
    fed_lens: List[int] = []
    for ids in full_ids_list:
        fed = ids if len(ids) < max_model_len else ids[: max_model_len - 1]
        fed_lens.append(len(fed))
        prompts.append(TokensPrompt(prompt_token_ids=list(fed)))

    outputs = llm.generate(prompts, sampling)

    mean_logprobs: List[float] = []
    token_logprobs_list: List[List[float]] = []
    for output, ids, prompt_len, fed_len in zip(
        outputs, full_ids_list, prompt_lens_list, fed_lens, strict=True,
    ):
        prompt_logprobs = output.prompt_logprobs
        # prompt_logprobs[pos] holds the logprob of token ``pos`` given the
        # prefix (None at pos 0); missing entries raise rather than skew KL.
        tok_lps = [
            prompt_logprobs[pos][ids[pos]].logprob
            for pos in range(prompt_len, fed_len)
        ]
        token_logprobs_list.append(tok_lps)
        mean_logprobs.append(float(np.mean(tok_lps)) if tok_lps else 0.0)
    return mean_logprobs, token_logprobs_list


def kl_estimators_per_sample(policy_token_logprobs, base_token_logprobs):
    """Per-sample KL estimators from positionally aligned token logprobs.

    Returns ``(k1_per_sample, grpo_per_sample)``:
      - k1: mean over tokens of ``log pi(t) - log pi_base(t)`` (simple estimator)
      - grpo: mean over tokens of ``exp(d) - d - 1`` with ``d = ref - pol``
        (matches GRPO's KL term)

    The two lists must come from teacher-forcing the same token sequences, so
    a per-sample length mismatch is a hard error, not something to truncate.
    """
    k1_per_sample: List[float] = []
    grpo_per_sample: List[float] = []
    for pol_lp, ref_lp in zip(policy_token_logprobs, base_token_logprobs, strict=True):
        if len(pol_lp) != len(ref_lp):
            raise ValueError(
                f"policy/base token logprob length mismatch: "
                f"{len(pol_lp)} vs {len(ref_lp)}"
            )
        if not pol_lp:
            k1_per_sample.append(0.0)
            grpo_per_sample.append(0.0)
            continue
        diff = np.asarray(ref_lp) - np.asarray(pol_lp)
        k1_per_sample.append(float(np.mean(-diff)))
        grpo_per_sample.append(float(np.mean(np.exp(diff) - diff - 1)))
    return k1_per_sample, grpo_per_sample
