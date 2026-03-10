"""Vendored AlpacaFarm RewardModel class.

Extracted from the alpaca_farm package to avoid dependency on the full package,
which is incompatible with modern transformers (>= 4.36) due to moved imports
like ``from transformers.deepspeed import is_deepspeed_zero3_enabled``.

Only the minimal RewardModel, RewardConfig, and RewardModelOutput classes are
kept, plus the two helper functions they depend on (``make_generative_lm`` and
``get_transformer_hidden_size``).

Also includes ``recover_alpacafarm_reward_model`` which reconstructs full
weights from a HuggingFace weight-diff checkpoint + base LLaMA-7B weights,
since tatsu-lab distributes models as weight diffs for LLaMA licensing reasons.

Original source: https://github.com/tlc4418/alpaca_farm
License: Apache 2.0
"""

import os
from pathlib import Path

import torch
import tqdm
import transformers
from torch import Tensor, nn
from transformers.utils.generic import ModelOutput


def _make_generative_lm(model_name_or_path: str, **kwargs):
    """Load a generative LM backbone (always standard LlamaForCausalLM)."""
    return transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def _get_transformer_hidden_size(model: transformers.PreTrainedModel) -> int:
    if isinstance(model, transformers.GPT2LMHeadModel):
        return model.config.n_embd
    elif isinstance(model, transformers.OPTForCausalLM):
        return model.config.word_embed_proj_dim
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        return model.config.d_model
    else:
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM" if hasattr(transformers, "LLaMAForCausalLM") else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls):
            return model.config.hidden_size
        raise ValueError(f"Unknown base_model type: {type(model)}")


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super().__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self._name_or_path = backbone_model_name_or_path


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig

    def __init__(self, config: RewardConfig, **kwargs):
        super().__init__(config)
        self.backbone_model = _make_generative_lm(config.backbone_model_name_or_path, **kwargs)
        hidden_size = _get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        outputs = self.backbone_model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)


# -- Weight-diff recovery ---------------------------------------------------

# HuggingFace Hub name for the reward-model-human weight diff.
WDIFF_HUB_NAME = "tatsu-lab/alpaca-farm-reward-model-human-wdiff"

# The reward model's backbone is the AlpacaFarm SFT model (not raw LLaMA-7B).
# This is also distributed as a weight diff on top of base LLaMA-7B.
SFT_WDIFF_HUB_NAME = "tatsu-lab/alpaca-farm-sft10k-wdiff"

# Default base LLaMA-7B model on HuggingFace.
DEFAULT_LLAMA_7B = "huggyllama/llama-7b"


def _load_wdiff_state_dict(hub_name: str) -> dict:
    """Download a weight-diff checkpoint and return its state dict."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    import glob

    diff_dir = snapshot_download(hub_name)

    safetensors_files = sorted(glob.glob(os.path.join(diff_dir, "model*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(diff_dir, "pytorch_model*.bin")))
    bin_files = [f for f in bin_files if "index" not in f]

    state = {}
    if safetensors_files:
        for sf in safetensors_files:
            state.update(load_safetensors(sf))
    elif bin_files:
        for bf in bin_files:
            state.update(torch.load(bf, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(
            f"No model weights found in {diff_dir}. "
            f"Expected model.safetensors or pytorch_model*.bin"
        )
    return state, diff_dir


def _apply_wdiff(diff_state: dict, base_state: dict, key_prefix: str = "") -> int:
    """Add base weights to diff weights in-place. Returns count of recovered keys.

    For each key in diff_state, strips ``key_prefix`` to find the matching
    base key.  Skips keys with shape mismatches (e.g. resized embeddings,
    where the diff already stores full weights) and keys absent from base.
    """
    added = 0
    for key in tqdm.tqdm(diff_state, desc="Recovering weights"):
        base_key = key
        if key_prefix and key.startswith(key_prefix):
            base_key = key[len(key_prefix):]

        if base_key in base_state:
            if diff_state[key].shape == base_state[base_key].shape:
                diff_state[key].add_(base_state[base_key])
                added += 1
    return added


def recover_alpacafarm_reward_model(
    output_dir: str,
    wdiff_name: str = WDIFF_HUB_NAME,
    base_model_name: str = DEFAULT_LLAMA_7B,
) -> str:
    """Recover full AlpacaFarm reward model weights from weight-diff checkpoints.

    The reward model's backbone is the AlpacaFarm SFT model, which is itself
    distributed as a weight diff on top of base LLaMA-7B.  Recovery is therefore
    a two-step chain:

      1. ``sft = sft_wdiff + base_llama``
      2. ``reward = reward_wdiff + sft``

    Args:
        output_dir: Where to save the recovered full model.
        wdiff_name: HF Hub name for the reward-model weight-diff checkpoint.
        base_model_name: HF Hub name for the base LLaMA-7B model.

    Returns:
        The ``output_dir`` path (for convenience).
    """
    output_path = Path(output_dir)
    if output_path.exists() and (output_path / "config.json").exists():
        print(f"Recovered model already exists at {output_dir}, skipping recovery.")
        return output_dir

    print(f"Recovering AlpacaFarm reward model weights (two-step)...")
    print(f"  Step 1: SFT wdiff ({SFT_WDIFF_HUB_NAME}) + base ({base_model_name})")
    print(f"  Step 2: RM  wdiff ({wdiff_name}) + recovered SFT")

    # ── Step 1: Recover the SFT model ────────────────────────────────────
    print("\n--- Step 1: Recovering SFT model ---")
    sft_diff_state, _ = _load_wdiff_state_dict(SFT_WDIFF_HUB_NAME)

    base_model = transformers.LlamaForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32,
    )
    base_state = base_model.state_dict()

    # SFT wdiff keys are plain LlamaForCausalLM keys (no prefix).
    n = _apply_wdiff(sft_diff_state, base_state, key_prefix="")
    print(f"  SFT: {n} keys recovered with base weights")
    del base_state, base_model

    # ── Step 2: Recover the reward model ─────────────────────────────────
    print("\n--- Step 2: Recovering reward model ---")
    rm_diff_state, rm_diff_dir = _load_wdiff_state_dict(wdiff_name)

    # The RM wdiff keys are prefixed with "backbone_model." relative to the
    # SFT model.  Strip that prefix to match against sft_diff_state keys.
    n = _apply_wdiff(rm_diff_state, sft_diff_state, key_prefix="backbone_model.")
    print(f"  RM: {n} keys recovered with SFT weights")
    del sft_diff_state

    # ── Step 3: Build a RewardModel, load recovered weights, save ────────
    # The SFT model has vocab 32001 (base 32000 + pad token), so we need a
    # backbone with the right size.  We load a fresh base and resize.
    config = RewardConfig(backbone_model_name_or_path=base_model_name)
    model = RewardModel(config)

    embed_key = "backbone_model.model.embed_tokens.weight"
    if embed_key in rm_diff_state:
        diff_vocab_size = rm_diff_state[embed_key].shape[0]
        current_vocab_size = model.backbone_model.model.embed_tokens.weight.shape[0]
        if diff_vocab_size != current_vocab_size:
            print(f"Resizing embeddings: {current_vocab_size} -> {diff_vocab_size}")
            model.backbone_model.resize_token_embeddings(diff_vocab_size)

    model.load_state_dict(rm_diff_state, strict=False)

    output_path.mkdir(parents=True, exist_ok=True)

    # Save the resized backbone so that future from_pretrained() calls
    # create a model with the correct vocab size (32001 vs 32000).
    backbone_dir = os.path.join(output_dir, "backbone")
    model.backbone_model.save_pretrained(backbone_dir)

    # Point config to the local backbone so __init__ loads the right size.
    model.config.backbone_model_name_or_path = backbone_dir
    model.config._name_or_path = backbone_dir
    model.save_pretrained(output_dir)

    # Save the tokenizer from the wdiff (has the extra pad token).
    tokenizer = transformers.AutoTokenizer.from_pretrained(rm_diff_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Recovered model saved to {output_dir}")

    del model, rm_diff_state
    return output_dir
