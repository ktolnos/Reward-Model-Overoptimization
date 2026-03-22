"""vLLM worker extension for reloading model weights from a HF checkpoint."""

import gc
import torch
from data_utils import load_causal_lm


class WeightLoaderExtension:
    """Mixed into the vLLM worker when passed as worker_extension_cls."""

    def load_weights_from_path(self, path: str):
        hf_model = load_causal_lm(path, device_map="cpu")
        weights = [(n, p.data) for n, p in hf_model.named_parameters()]
        self.model_runner.model.load_weights(weights)
        del hf_model, weights
        gc.collect()
        torch.cuda.empty_cache()
