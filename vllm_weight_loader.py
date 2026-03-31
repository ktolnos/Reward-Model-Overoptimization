"""vLLM worker extension for reloading model weights from a HF checkpoint."""

import gc
import torch
from data_utils import load_causal_lm


class WeightLoaderExtension:
    """Mixed into the vLLM worker when passed as worker_extension_cls."""

    def load_weights_from_path(self, path: str):
        hf_model = load_causal_lm(path, device_map="cpu")
        # HF Qwen3_5ForCausalLM has params at model.*, but vLLM loads the
        # checkpoint as Qwen3_5ForConditionalGeneration with params at
        # language_model.model.*. Remap the prefix so load_weights finds them.
        weights = [
            ("language_model." + n if n.startswith("model.") else n, p.data)
            for n, p in hf_model.named_parameters()
        ]
        self.model_runner.model.load_weights(weights)
        del hf_model, weights
        gc.collect()
        torch.cuda.empty_cache()
