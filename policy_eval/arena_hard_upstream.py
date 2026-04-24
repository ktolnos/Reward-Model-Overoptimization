"""Vendored code from the Arena-Hard-Auto repository (Apache 2.0).

Source: https://github.com/lmarena/arena-hard-auto
  - ``utils/add_markdown_info.py`` → feature extraction
  - ``utils/math_utils.py``        → BT regression + bootstrap
  - ``show_result.py``             → style-feature preprocessing + score aggregation

Copied with minimal edits so that our scores match the official leaderboard's
numbers exactly when the same judge + baseline answers are used. The only
deviations from upstream:
  1. ``count_markdown_elements`` takes a raw text argument (no DataFrame
     plumbing).
  2. Bootstrap doesn't print tqdm when called from evaluator code.
  3. We don't need the multi-model leaderboard path; only the single-model
     vs baseline case, so ``one_hot_encode`` and ``to_winrate_probabilities``
     are called with ``len(models)==2``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Upstream ``utils/judge_utils.py:JUDGE_SETTINGS`` — each Arena-Hard-Auto v2.0
# category has its own baseline answer file, and the official leaderboard
# reports one score per category against that baseline.
CATEGORY_BASELINES: Dict[str, str] = {
    "hard_prompt": "o3-mini-2025-01-31",
    "coding": "o3-mini-2025-01-31",
    "math": "o3-mini-2025-01-31",
    "creative_writing": "gemini-2.0-flash-001",
}


# ---------------------------------------------------------------------------
# Feature extraction (from utils/add_markdown_info.py)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```([^`]*)```")


def _remove_code_blocks(answer: str) -> str:
    blocks = _CODE_FENCE_RE.findall(answer)
    for block in blocks:
        answer = answer.replace(block, "")
    return answer


def count_markdown_elements(markdown_text: str, suffix: str = "") -> Dict[str, Dict[str, int]]:
    """Same counters as Arena-Hard's utils/add_markdown_info.count_markdown_elements."""
    return {
        f"header_count{suffix}": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count{suffix}": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count{suffix}": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
    }


_TIKTOKEN_ENCODER = None


def _get_tiktoken_encoder():
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        import tiktoken
        _TIKTOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4o")
    return _TIKTOKEN_ENCODER


def extract_style_metadata(answer: str) -> Dict:
    """Return the same ``metadata`` dict upstream stores per model answer.

    Keys: ``token_len``, ``header_count``, ``list_count``, ``bold_count``
    (the last three are dicts; upstream sums them via ``sum(v.values())``).
    """
    encoder = _get_tiktoken_encoder()
    token_len = len(encoder.encode(answer, disallowed_special=()))
    stripped = _remove_code_blocks(answer)
    meta = count_markdown_elements(stripped, suffix="")
    return {"token_len": token_len, **meta}


# ---------------------------------------------------------------------------
# Bradley-Terry model + bootstrap (from utils/math_utils.py)
# ---------------------------------------------------------------------------

class _BTModel(nn.Module):
    def __init__(self, num_components: int):
        super().__init__()
        self.logits = nn.Parameter(
            nn.init.constant_(torch.empty(num_components), 0.5)
        )

    def forward(self):
        return self.logits, None


def _bt_loss(logits: torch.Tensor, outcomes: torch.Tensor, **kwargs) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, outcomes.float(), reduction="sum"
    )


def fit_pairwise_model(
    features: torch.Tensor,
    outcomes: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    lr: float = 0.1,
    tol: float = 1e-9,
    max_epochs: int = 50,
) -> torch.Tensor:
    """LBFGS fit of BT on (features, outcomes). Returns coef vector."""
    if indices is not None:
        features = features[indices]
        outcomes = outcomes[indices]

    assert not features.isnan().any()
    model = _BTModel(num_components=features.shape[1])
    optimizer = optim.LBFGS(
        model.parameters(), lr=lr, max_iter=max_epochs,
        tolerance_grad=tol, tolerance_change=tol,
    )

    def closure():
        optimizer.zero_grad()
        logits, _ = model()
        _logits = features @ logits
        loss = _bt_loss(logits=_logits, outcomes=outcomes)
        loss.backward()
        return loss

    optimizer.step(closure)
    logits, _ = model()
    return logits.detach()


def bootstrap_pairwise_model(
    features: torch.Tensor,
    outcomes: torch.Tensor,
    num_round: int = 100,
    seed: int = 0,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    boot_idxs = rng.integers(0, features.shape[0], size=(num_round, features.shape[0]))
    results = [
        fit_pairwise_model(features, outcomes, torch.from_numpy(boot_idxs[i]))
        for i in range(num_round)
    ]
    return torch.stack(results)


def one_hot_encode(items: List[str], baseline: str) -> Tuple[torch.Tensor, List[str]]:
    """Upstream layout: row has +1 for the model, -1 for the baseline."""
    unique_items = sorted(set(items + [baseline]))
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}
    one_hot_matrix = []
    for item in items:
        v = [0] * len(unique_items)
        v[item_to_index[item]] = 1
        v[item_to_index[baseline]] = -1
        one_hot_matrix.append(v)
    return torch.tensor(one_hot_matrix, dtype=torch.float32), unique_items


def to_winrate_probabilities(
    coefs: torch.Tensor, models: List[str], baseline_model: str,
) -> torch.Tensor:
    """Convert fitted BT coefficients to win-rate probabilities vs baseline."""
    baseline_idx = models.index(baseline_model)
    exp_coefs = torch.exp(coefs)
    probs = torch.zeros(coefs.shape[0], coefs.shape[1])
    for idx in range(len(models)):
        if models[idx] == baseline_model:
            probs[:, idx] = 0.5
        probs[:, idx] = exp_coefs[:, idx] / (exp_coefs[:, idx] + exp_coefs[:, baseline_idx])
    return probs


# ---------------------------------------------------------------------------
# Style-feature preprocessing (from show_result.py:print_leaderboard_with_style_features)
# ---------------------------------------------------------------------------

def build_style_feature_tensor(
    model_metadatas: List[Dict],
    baseline_metadatas: List[Dict],
) -> torch.Tensor:
    """Reproduce the exact feature tensor upstream feeds to BT regression.

    Each metadata dict must have: ``token_len`` (int), ``header_count`` (int|dict),
    ``list_count`` (int|dict), ``bold_count`` (int|dict). Dict values are summed.

    Returns a ``(n, 4)`` z-scored tensor with columns:
        [length_diff, header_density_diff, list_density_diff, bold_density_diff].
    """
    def _to_row(metadata: Dict) -> List[float]:
        return [
            float(v) if isinstance(v, (int, float, np.integer)) else float(sum(v.values()))
            for v in metadata.values()
        ]

    model_tensor = torch.tensor([_to_row(m) for m in model_metadatas], dtype=torch.float32)
    baseline_tensor = torch.tensor([_to_row(m) for m in baseline_metadatas], dtype=torch.float32)

    final = torch.zeros_like(model_tensor)
    # Column 0: length delta, normalized to [-1, 1]:
    final[:, 0] = (model_tensor[:, 0] - baseline_tensor[:, 0]) / (
        model_tensor[:, 0] + baseline_tensor[:, 0]
    )
    # Columns 1..: markdown density diff, normalized.
    model_density = model_tensor[:, 1:] / (model_tensor[:, :1] + 1)
    baseline_density = baseline_tensor[:, 1:] / (baseline_tensor[:, :1] + 1)
    final[:, 1:] = (model_density - baseline_density) / (
        model_density + baseline_density + 1
    )
    assert not final.isnan().any()

    # Z-score per column. Upstream uses raw std; add eps so degenerate cases
    # (e.g. tiny debug subsamples with zero variance on a column) don't NaN.
    std = torch.std(final, dim=0)
    std = torch.where(std > 0, std, torch.ones_like(std))
    normalized = (final - torch.mean(final, dim=0)) / std
    assert not normalized.isnan().any()
    return normalized


# ---------------------------------------------------------------------------
# High-level single-model-vs-baseline helper
# ---------------------------------------------------------------------------

def arena_hard_style_controlled_score(
    outcomes: np.ndarray,
    model_style_meta: List[Dict],
    baseline_style_meta: List[Dict],
    *,
    model_name: str,
    baseline_model: str,
    num_round: int = 100,
    seed: int = 0,
) -> Tuple[float, float, float, np.ndarray]:
    """Upstream-faithful style-controlled win rate vs a fixed baseline.

    Returns ``(score_pct, ci_low_pct, ci_high_pct, feature_coefs_median)``.

    ``outcomes`` are battle-level values in {0, 0.5, 1} from the model's
    perspective (same shape as ``battles`` in ``show_result.load_judgments``
    after exploding the weighted ``label_to_score`` lists).

    ``model_style_meta`` / ``baseline_style_meta`` are per-battle style
    metadata dicts aligned with ``outcomes`` (so if a prompt generates 6
    battles, the same metadata dict is repeated 6 times).
    """
    style_feature_tensor = build_style_feature_tensor(
        model_style_meta, baseline_style_meta
    )  # (n, 4)

    model_features, unique_models = one_hot_encode(
        [model_name] * len(outcomes), baseline=baseline_model
    )  # (n, 2) — since we have exactly one non-baseline model

    all_features = torch.cat([model_features, style_feature_tensor], dim=1)
    outcomes_t = torch.tensor(outcomes, dtype=torch.float32)

    coefs = bootstrap_pairwise_model(
        all_features, outcomes_t, num_round=num_round, seed=seed,
    )  # (num_round, num_models + 4)

    num_features = style_feature_tensor.shape[1]
    model_coefs = coefs[:, :-num_features]       # (num_round, num_models)
    feature_coefs = coefs[:, -num_features:]     # (num_round, num_features)

    probs = to_winrate_probabilities(
        model_coefs, unique_models, baseline_model=baseline_model,
    )  # (num_round, num_models)
    model_idx = unique_models.index(model_name)
    model_probs = probs[:, model_idx].numpy()

    score = float(np.percentile(model_probs, 50) * 100)
    ci_low = float(np.percentile(model_probs, 5) * 100)
    ci_high = float(np.percentile(model_probs, 95) * 100)
    feature_coefs_median = torch.quantile(feature_coefs, 0.5, dim=0).numpy()
    return score, ci_low, ci_high, feature_coefs_median
