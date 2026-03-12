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
#
# Mirrors the original tatsu-lab recovery script:
#   pretrained_models/recover_model_weights.py
#
# The procedure (for reward models):
#   1. Load base LLaMA-7B, resize vocab 32000→32001 (add [PAD] token) using
#      the alpaca_farm "stable" init (new embedding = mean of existing).
#   2. Load the RM weight-diff from HuggingFace.
#   3. Add resized-base weights to diff: ``recovered = diff + base``.
#      Since both now have vocab 32001, ALL keys match (including embed/lm_head).

# HuggingFace Hub name for the reward-model-human weight diff.
WDIFF_HUB_NAME = "tatsu-lab/alpaca-farm-reward-model-human-wdiff"

# Default base LLaMA-7B model — must be Meta's original weights converted with
# transformers>=4.29.2 (community uploads like huggyllama/llama-7b won't work).
DEFAULT_LLAMA_7B = "ktolnos/llama-7b-hf-converted"


def _stable_resize_token_embeddings(model, target_size):
    """Resize embeddings, initializing new tokens as mean of existing (alpaca_farm convention)."""
    num_new = target_size - model.get_input_embeddings().weight.size(0)
    try:
        model.resize_token_embeddings(target_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(target_size)
    if num_new > 0:
        with torch.inference_mode():
            for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
                if emb is None:
                    continue
                data = emb.weight.data
                avg = data[:-num_new].mean(dim=0, keepdim=True)
                data[-num_new:] = avg


def recover_alpacafarm_reward_model(
    output_dir: str,
    wdiff_name: str = WDIFF_HUB_NAME,
    base_model_name: str = DEFAULT_LLAMA_7B,
) -> str:
    """Recover full AlpacaFarm reward model weights from a weight-diff checkpoint.

    Mirrors the original tatsu-lab recovery procedure:
      1. Load base LLaMA-7B and resize vocab to 32001 (``stable_resize``).
      2. Prefix base keys with ``backbone_model.`` and add to diff weights.
      3. Save the recovered RewardModel.

    Args:
        output_dir: Where to save the recovered full model.
        wdiff_name: HF Hub name for the weight-diff checkpoint.
        base_model_name: HF Hub name (or local path) for the base LLaMA-7B.

    Returns:
        The ``output_dir`` path (for convenience).
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    from safetensors.torch import load_file as load_safetensors
    import glob
    import json
    import numpy as np

    output_path = Path(output_dir)
    if output_path.exists() and (output_path / "config.json").exists():
        print(f"Recovered model already exists at {output_dir}, skipping recovery.")
        return output_dir

    print(f"Recovering AlpacaFarm reward model weights...")
    print(f"  Weight diff: {wdiff_name}")
    print(f"  Base model:  {base_model_name}")

    # ── 1. Load base LLaMA-7B and resize to 32001 vocab ──────────────────
    print("\n--- Loading and resizing base model ---")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32,
    )
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
    if base_tokenizer.pad_token is None:
        base_tokenizer.add_special_tokens(dict(pad_token="[PAD]"))
        _stable_resize_token_embeddings(base_model, len(base_tokenizer))
        print(f"  Resized base embeddings to {len(base_tokenizer)} (added [PAD])")

    # Build prefixed base state dict (backbone_model.xxx).
    base_state = {f"backbone_model.{k}": v for k, v in base_model.state_dict().items()}
    print(f"  Base state dict: {len(base_state)} keys")
    del base_model

    # ── 2. Load the RM weight-diff ───────────────────────────────────────
    print("\n--- Loading RM weight diff ---")
    diff_dir = snapshot_download(wdiff_name)

    safetensors_files = sorted(glob.glob(os.path.join(diff_dir, "model*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(diff_dir, "pytorch_model*.bin")))
    bin_files = [f for f in bin_files if "index" not in f]

    diff_state = {}
    if safetensors_files:
        for sf in safetensors_files:
            diff_state.update(load_safetensors(sf))
    elif bin_files:
        for bf in bin_files:
            diff_state.update(torch.load(bf, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"No model weights found in {diff_dir}")
    print(f"  Diff state dict: {len(diff_state)} keys")

    # ── 3. Add base weights to diff (in-place) ──────────────────────────
    print("\n--- Recovering weights (diff + base) ---")
    added = 0
    for key in tqdm.tqdm(base_state, desc="Recovering"):
        if key not in diff_state:
            continue
        if base_state[key].size() != diff_state[key].size():
            continue
        diff_state[key].add_(base_state[key])
        added += 1
    print(f"  {added} keys recovered")
    del base_state

    # ── 4. Integrity check ───────────────────────────────────────────────
    model_sum = sum(v.float().sum().item() for v in diff_state.values())
    try:
        sum_file = hf_hub_download(repo_id=wdiff_name, filename="model_sum.txt")
        with open(sum_file) as f:
            target_sum = float(f.read().strip())
        if np.isclose(target_sum, model_sum):
            print(f"  Integrity check PASSED (sum={model_sum:.2f})")
        else:
            print(f"  Integrity check FAILED: got {model_sum:.2f}, expected {target_sum:.2f}")
            print(f"  This likely means the base LLaMA-7B weights differ from tatsu-lab's.")
            print(f"  Try using the original Meta LLaMA-7B converted with transformers>=4.29.2.")
    except Exception as e:
        print(f"  Could not verify integrity: {e}")

    # ── 5. Build RewardModel, load recovered weights, save ───────────────
    # Use a temporary SFT-10k backbone for __init__ (correct vocab size).
    # The weights will be immediately overwritten by load_state_dict.
    print("\n--- Building and saving RewardModel ---")

    # We need an SFT-10k model for __init__ so the architecture has 32001 vocab.
    # First recover SFT from its own wdiff + base, then use as backbone.
    sft_dir = os.path.join(output_dir, "_sft_backbone")
    if not os.path.exists(sft_dir):
        print("  Recovering SFT-10k backbone for model architecture...")
        sft_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float32,
        )
        sft_tok = transformers.AutoTokenizer.from_pretrained(base_model_name)
        if sft_tok.pad_token is None:
            sft_tok.add_special_tokens(dict(pad_token="[PAD]"))
            _stable_resize_token_embeddings(sft_model, len(sft_tok))
        Path(sft_dir).mkdir(parents=True, exist_ok=True)
        sft_model.save_pretrained(sft_dir)
        sft_tok.save_pretrained(sft_dir)
        del sft_model, sft_tok

    config = RewardConfig(backbone_model_name_or_path=sft_dir)
    model = RewardModel(config)
    model.load_state_dict(diff_state, strict=False)

    output_path.mkdir(parents=True, exist_ok=True)

    # Save backbone so future from_pretrained() gets the right vocab size.
    backbone_dir = os.path.join(output_dir, "backbone")
    model.backbone_model.save_pretrained(backbone_dir)

    model.config.backbone_model_name_or_path = backbone_dir
    model.config._name_or_path = backbone_dir
    model.save_pretrained(output_dir)

    # Save tokenizer from the wdiff (has the extra pad token).
    tokenizer = transformers.AutoTokenizer.from_pretrained(diff_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Recovered model saved to {output_dir}")
    del model, diff_state
    return output_dir
