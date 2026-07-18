"""Arena-Hard-Auto-compatible pairwise win-rate metrics.

Consumes ``battles_per_prompt`` from a ``Judge`` — a list of per-prompt lists
where each inner entry is a battle score in {0.0, 0.5, 1.0} from the policy's
perspective. An empty inner list means the judge failed on that prompt and it
is excluded from metrics.

Produces metrics in the same shape regardless of judge:

    - wins / ties / losses: per-prompt counts (unweighted, one decision each)
    - win_rate / win_or_tie_rate: unweighted default — one decision per prompt
      (see below), comparable across judges; equals the weighted rate for a
      one-battle RM judge
    - wins/ties/losses_weighted, total: battle-level counts (decisive LLM
      verdicts expanded x weight)
    - win_rate_weighted / win_or_tie_rate_weighted: battle-level weighted, the
      upstream Arena-Hard aggregation
    - arena_score: mean(all_battles) * 100
    - arena_score_ci_low / _ci_high: 90% CI, battle-level bootstrap (matches
      ``show_result.py:print_leaderboard``).
    - sc_score, sc_score_ci_low / _ci_high: style-controlled win rate from the
      vendored upstream Bradley-Terry regression (see
      ``arena_hard_upstream.arena_hard_style_controlled_score``). Exactly the
      leaderboard's ``show_result.py --control-features length markdown`` path.
    - sc_coef/length, headers, list, bold: median BT style coefficients.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .arena_hard_upstream import (
    arena_hard_style_controlled_score,
    extract_style_metadata,
)


def compute_pairwise_metrics(
    battles_per_prompt: List[List[float]],
    policy_responses: Sequence[str],
    baseline_responses: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Aggregate per-prompt battle lists into Arena-Hard-compatible metrics.

    ``battles_per_prompt[i]`` may contain 1 item (RM judge) or many (weighted
    LLM judge). Empty lists are skipped.
    """
    if len(battles_per_prompt) != len(policy_responses):
        raise ValueError(
            f"{len(battles_per_prompt)} battle rows vs {len(policy_responses)} responses"
        )

    # Flatten to a battle-level array plus a parallel index into the per-prompt
    # response arrays so style metadata broadcasts the same way upstream does
    # (the same row of metadata is repeated across all battles for that prompt).
    flat_outcomes: List[float] = []
    flat_prompt_idx: List[int] = []
    for i, battles in enumerate(battles_per_prompt):
        for b in battles:
            flat_outcomes.append(float(b))
            flat_prompt_idx.append(i)
    if not flat_outcomes:
        return {"total": 0, "wins": 0, "ties": 0, "losses": 0}

    outcomes = np.array(flat_outcomes, dtype=np.float64)
    prompt_idx = np.array(flat_prompt_idx, dtype=np.int64)
    n = len(outcomes)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))

    # --- Arena score via battle-level bootstrap (show_result.py:print_leaderboard).
    # Upstream reports ``mean(bootstrap_means)`` as the point estimate and
    # ``quantile(0.05), quantile(0.95)`` on the same bootstrap distribution.
    boot_arena = outcomes[boot_idx].mean(axis=1) * 100
    arena_score = float(boot_arena.mean())

    # Battle-level (weighted) counts: decisive LLM verdicts are expanded x weight
    # (this is the upstream Arena-Hard aggregation).
    wins_w = int((outcomes == 1.0).sum())
    ties_w = int((outcomes == 0.5).sum())
    losses_w = n - wins_w - ties_w
    n_prompts = sum(1 for b in battles_per_prompt if b)

    # Default win rate is unweighted: collapse each prompt to a single
    # win/tie/loss so a weighted LLM judge (decisive verdicts expanded x weight,
    # over the two positional games) contributes exactly one vote per prompt —
    # directly comparable to the RM judge, which already emits one battle per
    # prompt. Per prompt, mean the battle scores and threshold at 0.5: verdict
    # strength still breaks a strong-vs-weak positional conflict (A>>B vs B>A ->
    # win), while a symmetric conflict (A>B vs B>A) averages to 0.5 -> tie.
    wins = ties = losses = 0
    for battles in battles_per_prompt:
        if not battles:
            continue
        m = float(np.mean(battles))
        if m > 0.5:
            wins += 1
        elif m == 0.5:
            ties += 1
        else:
            losses += 1

    out: Dict[str, float] = {
        # Default: unweighted, one decision per prompt.
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "n_prompts_judged": n_prompts,
        "win_rate": wins / n_prompts if n_prompts else 0.0,
        "win_or_tie_rate": (wins + ties) / n_prompts if n_prompts else 0.0,
        # Weighted, battle-level (upstream Arena-Hard aggregation).
        "wins_weighted": wins_w,
        "ties_weighted": ties_w,
        "losses_weighted": losses_w,
        "total": n,
        "win_rate_weighted": wins_w / n,
        "win_or_tie_rate_weighted": (wins_w + ties_w) / n,
        "arena_score": arena_score,
        "arena_score_ci_low": float(np.percentile(boot_arena, 5)),
        "arena_score_ci_high": float(np.percentile(boot_arena, 95)),
    }

    # --- Style-controlled score via the vendored upstream BT regression ---
    # Build per-prompt metadata exactly like upstream's add_markdown_info output.
    per_prompt_policy_meta = [extract_style_metadata(r) for r in policy_responses]
    per_prompt_baseline_meta = [extract_style_metadata(r) for r in baseline_responses]
    # Broadcast to battle level.
    model_style_meta = [per_prompt_policy_meta[i] for i in prompt_idx]
    baseline_style_meta = [per_prompt_baseline_meta[i] for i in prompt_idx]

    try:
        sc_score, sc_low, sc_high, coefs = arena_hard_style_controlled_score(
            outcomes, model_style_meta, baseline_style_meta,
            model_name="policy", baseline_model="baseline",
            num_round=n_bootstrap, seed=seed,
        )
        out["sc_score"] = sc_score
        out["sc_score_ci_low"] = sc_low
        out["sc_score_ci_high"] = sc_high
        for name, c in zip(["length", "headers", "list", "bold"], coefs):
            out[f"sc_coef/{name}"] = float(c)
    except Exception as e:
        # BT fit can fail on tiny/degenerate batches; keep raw metrics.
        print(f"[pairwise] style-controlled score failed: {e}")

    return out
