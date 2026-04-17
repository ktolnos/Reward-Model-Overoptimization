"""Reward-model loading and scoring shared by all RM-based evaluators.

The ``LoadedRewardModels`` container loads each configured RM exactly once and
exposes a label -> (model, tokenizer) map. Evaluators look up the RM they need
by label ("gold_rm", "training_rm", "secondary_rm").

Chosen-response scores are pre-computed once per dataset and cached to disk
(``chosen_scores_cache/``) so win-rate metrics can be computed without rescoring
chosen responses on every run.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_utils import format_and_validate_preference_sample
from reward_utils import get_reward_rm, load_reward_model

CHOSEN_SCORES_CACHE_VERSION = 1


def load_reward_model_impl(model_path_or_name: str, device: str):
    model, tokenizer = load_reward_model(
        model_path_or_name, reasoning=False, device=device
    )
    return model, tokenizer


def score_responses_with_rm(
    responses: List[str],
    prompt_messages_list: List[list],
    rm_model,
    rm_tokenizer,
    *,
    batch_size: int,
    device: str,
    checkpoint_num=None,
) -> np.ndarray:
    """Score ``responses`` given parallel ``prompt_messages_list`` with one RM."""
    if len(prompt_messages_list) != len(responses):
        raise ValueError(
            f"Prompt/response mismatch: {len(prompt_messages_list)} vs {len(responses)}"
        )

    texts: List[str] = []
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
            rm_model, rm_tokenizer, texts,
            batch_size=batch_size, device=device,
        )
        .cpu()
        .float()
        .numpy()
    )


class LoadedRewardModels:
    """Loads only the reward models actually needed by the active evaluators.

    Callers pass the set of RM labels required (collected from the benchmarks'
    evaluators via ``eval_utils.rms_required_by``). Loading is limited to that
    set so that e.g. an ifeval-only run doesn't pay the cost of loading the
    training/secondary RMs just because they're wired up by default in args.
    """

    # Maps rm_label -> ScriptArguments attribute that holds the model path/name.
    _LABEL_TO_ARG = {
        "gold_rm": "gold_rm_name",
        "training_rm": "training_rm_path",
        "secondary_rm": "secondary_rm_name",
    }

    def __init__(self, args, labels_needed: set):
        self.args = args
        self._models: Dict[str, Tuple] = {}
        self._chosen_scores: Dict[str, np.ndarray] = {}

        for label in labels_needed:
            if label not in self._LABEL_TO_ARG:
                raise ValueError(
                    f"Unknown RM label '{label}'. Known: {list(self._LABEL_TO_ARG)}"
                )
            path = getattr(args, self._LABEL_TO_ARG[label])
            if not path or (isinstance(path, str) and path.lower() == "none"):
                raise ValueError(
                    f"RM label '{label}' requested by an evaluator but "
                    f"--{self._LABEL_TO_ARG[label]} is not set (got {path!r})."
                )
            print(f"Loading {label} ({path})...")
            self._models[label] = load_reward_model_impl(path, args.device)

    def labels(self) -> List[str]:
        return list(self._models)

    def get(self, label: str):
        return self._models.get(label)

    def unload(self):
        for name in list(self._models):
            del self._models[name]
        self._models.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Chosen-score cache (for win-rate metrics on the preference benchmark)
    # ------------------------------------------------------------------
    def chosen_scores(self, label: str) -> Optional[np.ndarray]:
        return self._chosen_scores.get(label)

    def precompute_chosen_scores(self, dataset, prompt_messages_list, args) -> None:
        chosen_responses = [ex["chosen"][-1]["content"] for ex in dataset]
        cache_id = hashlib.sha256(
            f"{args.dataset_name}:{len(chosen_responses)}".encode()
        ).hexdigest()[:16]
        cache_dir = os.path.join(
            os.path.dirname(args.output_file) or ".", "chosen_scores_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)

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
                chosen_responses, prompt_messages_list,
                model, tokenizer,
                batch_size=args.batch_size, device=args.device,
                checkpoint_num="chosen",
            )
            self._chosen_scores[label] = scores

            with open(cache_file, "w") as f:
                json.dump({
                    "version": CHOSEN_SCORES_CACHE_VERSION,
                    "rm_name": rm_name,
                    "dataset_name": args.dataset_name,
                    "num_samples": len(chosen_responses),
                    "scores": scores.tolist(),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "computed_at": datetime.now().isoformat(),
                }, f, indent=2)
            print(
                f"[ChosenScores] {label} ({rm_name}): mean={np.mean(scores):.4f}, "
                f"std={np.std(scores):.4f}, cached to {cache_file}"
            )
