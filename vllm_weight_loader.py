"""vLLM worker extension for reloading model weights from a HF checkpoint."""

import gc
import torch
from data_utils import load_causal_lm


class WeightLoaderExtension:
    """Mixed into the vLLM worker when passed as worker_extension_cls."""

    def load_weights_from_path(self, path: str):
        hf_model = load_causal_lm(path, device_map="cpu")
        model = self.model_runner.model
        # vLLM may wrap the language model under a multimodal container: Qwen3.5's
        # Qwen3_5ForConditionalGeneration exposes the LM at language_model.model.*,
        # while HF names those params model.*. Only remap when the loaded vLLM
        # model actually uses that layout — a dense pre-3.5 Qwen3ForCausalLM keeps
        # its params at model.* and would break if we prefixed them.
        remap = hasattr(model, "language_model")
        weights = [
            ("language_model." + n if remap and n.startswith("model.") else n, p.data)
            for n, p in hf_model.named_parameters()
        ]
        loaded = set(model.load_weights(weights))
        # load_weights returns the destination param names it actually populated.
        # If a future vLLM rename silently drops names, the previous checkpoint's
        # weights would remain and every checkpoint after the first would be
        # scored as checkpoint 1 with no error. Fail fast if any model parameter
        # was left unloaded.
        expected = {name for name, _ in model.named_parameters()}
        missing = expected - loaded
        if missing:
            raise RuntimeError(
                f"vLLM load_weights populated only {len(loaded)} of {len(expected)} "
                f"model parameters from {path}; stale weights would remain. "
                f"Missing (up to 10): {sorted(missing)[:10]}"
            )
        del hf_model, weights
        gc.collect()
        torch.cuda.empty_cache()
