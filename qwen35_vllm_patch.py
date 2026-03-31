"""Compatibility patches for using text-only Qwen3.5 SFT checkpoints with vLLM.

The SFT checkpoint was saved as Qwen3_5ForConditionalGeneration (weights at
language_model.*), but vLLM 0.17+ treats all Qwen3.5 as multimodal (inheriting
from qwen3_vl) while transformers 5.x splits the config into a text-only
Qwen3_5TextConfig. These patches bridge all three.

See https://github.com/vllm-project/vllm/issues/37749.

Import this module before any vLLM or TRL code runs.
"""

# ---------------------------------------------------------------------------
# 1. Patch Qwen3_5TextConfig to satisfy vLLM's vision-config attribute access
# ---------------------------------------------------------------------------
# vLLM's Qwen3.5 handler inherits from the VL code and accesses attributes
# like vision_config, image_token_id, etc. that don't exist on the text-only
# transformers 5.x config. Return harmless defaults instead of AttributeError.

class _DummyVisionConfig:
    spatial_merge_size = 2
    image_size = 448
    temporal_patch_size = 2
    spatial_patch_size = 14
    in_channels = 3
    embed_dim = 1280
    depth = 32
    num_heads = 16
    def __getattr__(self, name):
        return None

_VL_MISSING_ATTRS = frozenset({
    'vision_config', 'video_token_id', 'image_token_id',
    'video_token', 'image_token', 'spatial_merge_size', 'tokens_per_second',
    'rope_scaling', 'vision_start_token_id', 'vision_end_token_id',
    'vision_token_id', 'image_start_token_id', 'image_end_token_id',
    'video_start_token_id', 'video_end_token_id',
})

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as _Qwen3_5TextConfig
_orig_getattribute = _Qwen3_5TextConfig.__getattribute__
def _patched_getattribute(self, name):
    try:
        return _orig_getattribute(self, name)
    except AttributeError:
        if name in _VL_MISSING_ATTRS:
            return _DummyVisionConfig() if name == 'vision_config' else None
        raise
_Qwen3_5TextConfig.__getattribute__ = _patched_getattribute

# ---------------------------------------------------------------------------
# 2. Patch vLLM's Qwen3_5ProcessingInfo.get_hf_config
# ---------------------------------------------------------------------------
# qwen3_5.py calls self.ctx.get_hf_config(Qwen3_5Config) which does a strict
# isinstance check. transformers 5.x returns Qwen3_5TextConfig, not the vLLM-
# internal Qwen3_5Config, so the check raises TypeError. Fall back to the raw
# hf_config instead.

import vllm
import vllm.entrypoints.llm
from vllm.model_executor.models.qwen3_5 import Qwen3_5ProcessingInfo, Qwen3_5MoeProcessingInfo

_orig_q35_get_hf_config = Qwen3_5ProcessingInfo.get_hf_config
def _patched_q35_get_hf_config(self):
    try:
        return _orig_q35_get_hf_config(self)
    except TypeError:
        return self.ctx.model_config.hf_config
Qwen3_5ProcessingInfo.get_hf_config = _patched_q35_get_hf_config

_orig_q35moe_get_hf_config = Qwen3_5MoeProcessingInfo.get_hf_config
def _patched_q35moe_get_hf_config(self):
    try:
        return _orig_q35moe_get_hf_config(self)
    except TypeError:
        return self.ctx.model_config.hf_config
Qwen3_5MoeProcessingInfo.get_hf_config = _patched_q35moe_get_hf_config

# ---------------------------------------------------------------------------
# 3. Stub out multimodal methods — the checkpoint has no image processor
# ---------------------------------------------------------------------------

from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from transformers import BatchEncoding

class _ImageProcessorStub:
    """Minimal image processor stub — merge_size is needed for token-count math."""
    merge_size = 2  # matches Qwen3_5VisionConfig.spatial_merge_size default

class _TextOnlyHFProcessor:
    """Minimal HF processor stub for text-only Qwen3.5."""
    image_processor = None
    video_processor = None
    # Token constants accessed during PromptReplacement construction (qwen3_vl.py:1155)
    image_token = "<|image_pad|>"
    image_token_id = 248056  # Qwen3_5Config.image_token_id default
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, text=None, return_tensors=None, **kwargs):
        return BatchEncoding(self.tokenizer(text or "", return_tensors=return_tensors))

Qwen3_5ProcessingInfo.get_mm_max_tokens_per_item = lambda self, seq_len, mm_counts: {"image": 0, "video": 0}
Qwen3VLDummyInputsBuilder.get_dummy_mm_data = lambda self, seq_len, mm_counts, mm_options: {}
Qwen3_5ProcessingInfo.get_hf_processor = lambda self, **kwargs: _TextOnlyHFProcessor(self.ctx.tokenizer)
Qwen3_5ProcessingInfo.get_image_processor = lambda self, **kwargs: _ImageProcessorStub()
Qwen3_5ProcessingInfo.get_video_processor = lambda self, **kwargs: None

# ---------------------------------------------------------------------------
# 4. Patch Qwen3_VisionTransformer.__init__ to be a no-op for text-only models
# ---------------------------------------------------------------------------
# The SFT checkpoint is Qwen3_5ForConditionalGeneration (weights at
# language_model.*), so we must load with that class for correct weight mapping.
# We never send images, so the vision tower can be a hollow nn.Module.

from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
import torch.nn as nn

_orig_vt_init = Qwen3_VisionTransformer.__init__
def _patched_vt_init(self, vision_config, *args, **kwargs):
    if isinstance(vision_config, _DummyVisionConfig):
        nn.Module.__init__(self)
        self.hidden_size = 1280
        self.num_heads = 16
        self.num_position_embeddings = 16384
        self.patch_size = 14
        self.spatial_merge_size = 2
        self.temporal_patch_size = 2
        self.spatial_merge_unit = 4
        self.out_hidden_size = 1280
        self.deepstack_visual_indexes = []
    else:
        _orig_vt_init(self, vision_config, *args, **kwargs)
Qwen3_VisionTransformer.__init__ = _patched_vt_init

# ---------------------------------------------------------------------------
# 5. Ensure language_model_only=True on every vllm.LLM() instantiation
# ---------------------------------------------------------------------------
# TRL 0.29.0 VLLMGeneration creates LLM() without this flag. Using it skips
# the multimodal media budget at inference time and saves memory.

_OriginalLLM = vllm.LLM
class _TextOnlyLLM(_OriginalLLM):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('language_model_only', True)
        super().__init__(*args, **kwargs)
vllm.LLM = _TextOnlyLLM
vllm.entrypoints.llm.LLM = _TextOnlyLLM

# ---------------------------------------------------------------------------
# 6. Fix TRL sync_weights prefix mismatch (GRPO training only)
# ---------------------------------------------------------------------------
# Training model loaded via AutoModelForCausalLM → Qwen3_5ForCausalLM has
# params at model.*, but vLLM's Qwen3_5ForConditionalGeneration expects them
# at language_model.model.*. The existing hf_to_vllm_mapper only handles
# "model.language_model.*" → "language_model.model.*", not plain "model.*".

from trl.generation.vllm_generation import VLLMGeneration
_orig_fix_param_name = VLLMGeneration._fix_param_name_to_vllm
def _patched_fix_param_name(self, name, extra_prefixes=None):
    name = _orig_fix_param_name(self, name, extra_prefixes=extra_prefixes)
    if name.startswith("model."):
        name = "language_model." + name
    return name
VLLMGeneration._fix_param_name_to_vllm = _patched_fix_param_name
