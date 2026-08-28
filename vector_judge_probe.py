"""Probe the Vector Institute inference proxy as an LLM-as-judge backend.

Settles what you need before committing an eval sweep to it: how fast is it, does
the model behave in thinking *and* non-thinking mode, does it pick the right
answer?

The material is real preference data: each prompt's dataset ``chosen`` response
is judged against its ``rejected`` one, through the same ``LLMJudge`` (Arena-Hard
2-game position swap + verdict parse) ``evaluate_policy.py`` uses. Since the
dataset already says which is preferred, agreement with it reads as accuracy
directly -- ~0.5 is guessing, below 0.5 has the positions crossed.

Usage (needs VECTOR_INFERENCE_API_KEY in the environment):

    python vector_judge_probe.py --n 20
    python vector_judge_probe.py --n 20 --model Qwen3_5-122B-A10B
    python vector_judge_probe.py --n 20 --modes thinking
    python vector_judge_probe.py --n 8 --modes batch      # async Batch API

``--dataset`` defaults to the HelpSteer3 preference set used by the pipeline.
Nothing here writes to wandb or to a run directory; it only prints a report.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import types
import time
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from policy_eval.judges import (
    OPENAI_PROVIDERS,
    JudgeGenParams,
    LLMJudge,
    OpenAICompatibleBackend,
    positional_bias_metrics,
)


# The dataset the current pipeline actually trains on (SFT / RM / GRPO all point
# here). Its chosen/rejected labels are HUMAN annotations, which is what makes
# "agreement" a meaningful judge-quality signal: the gold-RM-annotated variants
# would instead measure agreement with Skywork, i.e. how well the judge imitates
# another reward model.
DEFAULT_DATASET = "ktolnos/helpsteer3-qwen35_annotated_human"


def load_pairs(dataset_name: str, split: str, n: int):
    """Return up to ``n`` (prompt_messages, chosen_text, rejected_text) triples.

    Delegates to the preference benchmark's own loader, so the probe judges
    exactly the prompt sample an eval would: same split validation, same
    prompt-level dedup (HelpSteer3 carries up to 25 response-pairs per prompt,
    which are not independent samples), and the same seeded subsample. Pairs
    with no dataset ``rejected`` response are dropped -- there is nothing to
    judge against.
    """
    from policy_eval.benchmarks import _load_preference_split

    args = types.SimpleNamespace(
        dataset_name=dataset_name, debug=False, subsample_n=n,
    )
    pairs = []
    for ex in _load_preference_split(args, split):
        rejected = ex.metadata.get("rejected_messages")
        chosen_text = ex.metadata.get("chosen_response", "")
        if not rejected or not chosen_text.strip():
            continue
        rejected_text = rejected[-1]["content"]
        if not rejected_text.strip():
            continue
        pairs.append((ex.prompt_messages, chosen_text, rejected_text))
    if not pairs:
        raise SystemExit(f"no usable chosen/rejected pairs in {dataset_name}:{split}")
    return pairs


class TimedBackend(OpenAICompatibleBackend):
    """Backend that records per-request wall-clock latency."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.latencies: List[float] = []

    def _post_chat(self, messages, params, api_key):
        t0 = time.monotonic()
        try:
            return super()._post_chat(messages, params, api_key)
        finally:
            self.latencies.append(time.monotonic() - t0)


def run_mode(mode: str, pairs, args) -> dict:
    """Judge every pair once under one mode and summarise the result."""
    thinking = mode == "thinking"
    backend = TimedBackend(
        args.model,
        provider=args.provider,
        base_url=args.base_url or None,
        max_parallel=args.max_parallel,
        requests_per_minute=(None if args.rpm < 0 else args.rpm),
        reasoning_effort=args.reasoning_effort,
        use_batch_api=(mode == "batch"),
        batch_poll_seconds=args.batch_poll_seconds,
        max_retries=args.max_retries,
    )
    judge = LLMJudge(backend, gen_params=JudgeGenParams(
        temperature=args.temperature, top_p=1.0,
        max_tokens=args.max_new_tokens, enable_thinking=thinking,
    ))

    prompts = [p for p, _, _ in pairs]
    chosen = [c for _, c, _ in pairs]
    rejected = [r for _, _, r in pairs]

    print(f"\n{'=' * 72}\n{mode.upper()}  model={args.model}  n={len(pairs)} prompts "
          f"({2 * len(pairs)} games)\n{'=' * 72}", flush=True)

    t0 = time.monotonic()
    # answers_a = chosen, so a battle score of 1.0 means the judge agreed with
    # the dataset's preference.
    battles, details = judge.score_pairs(prompts, chosen, rejected, ctx=None)
    wall = time.monotonic() - t0

    flat = [b for bs in battles for b in bs]
    agreement = sum(flat) / len(flat) if flat else float("nan")
    lat = sorted(backend.latencies)
    n_games = 2 * len(pairs)

    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else float("nan")

    print(f"\n-- results ({mode}) --")
    print(f"  agreement with dataset preference : {agreement:.3f} "
          f"(0.5 = coin flip, 1.0 = always picks chosen)")
    # Position bias is a judge-quality signal independent of agreement: a judge
    # that flips its decisive winner when the answers swap places is deciding on
    # order, not content. Reuses the evaluator's own metric.
    bias = positional_bias_metrics(details)
    print(f"  position-flip rate    : {bias['controversial_rate']:.3f} "
          f"({bias['n_controversial']} flips; "
          f"first={bias['n_prefers_first']} second={bias['n_prefers_second']})")
    print(f"  prompts dropped (unusable verdict): {details.n_dropped_prompts}/{len(pairs)}")
    print(f"    generation failures : {details.n_generation_failures}/{n_games}")
    print(f"    truncation failures : {details.n_truncation_failures}/{n_games}")
    print(f"    parse failures      : {details.n_parse_failures}/{n_games}")
    print(f"  total wall-clock      : {wall:.1f}s ({wall / n_games:.2f}s per game)")
    print(f"  throughput            : {n_games / wall * 60:.0f} games/min")
    if lat:
        print(f"  per-request latency   : median {statistics.median(lat):.2f}s  "
              f"p90 {pct(0.9):.2f}s  max {lat[-1]:.2f}s")
    if args.dump_dir:
        _dump_games(args.dump_dir, mode, pairs, details)
    sample = next((t for t in details.game1_texts if t), "")
    if sample:
        print(f"  sample verdict text   : {sample[:300]!r}")

    return {
        "model": args.model, "mode": mode, "n_prompts": len(pairs),
        "agreement": agreement, "wall": wall, "per_game": wall / n_games,
        "controversial_rate": bias["controversial_rate"],
        "prefers_first": bias["n_prefers_first"],
        "prefers_second": bias["n_prefers_second"],
        "dropped": details.n_dropped_prompts,
        "generation_failures": details.n_generation_failures,
        "truncation_failures": details.n_truncation_failures,
        "parse_failures": details.n_parse_failures,
        "median_latency": statistics.median(lat) if lat else float("nan"),
        "p90_latency": pct(0.9),
        "games_per_min": n_games / wall * 60,
    }


def _dump_games(dump_dir: str, mode: str, pairs, details) -> None:
    """Write every judge generation, with its parsed label, to JSONL.

    Lets an unparsable verdict be read back afterwards instead of re-running the
    whole (slow, paid) pass just to see what the judge actually said.
    """
    import json

    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"{mode}.jsonl")
    with open(path, "w") as f:
        for i, (prompt, _, _) in enumerate(pairs):
            for game, labels, texts in (
                (0, details.game0_labels, details.game0_texts),
                (1, details.game1_labels, details.game1_texts),
            ):
                f.write(json.dumps({
                    "prompt_index": i,
                    "game": game,
                    "label": labels[i],
                    "parsed": labels[i] is not None,
                    "prompt": prompt[-1]["content"][:400],
                    "text": texts[i],
                }) + "\n")
    n_bad = sum(1 for ls in (details.game0_labels, details.game1_labels)
                for l in ls if l is None)
    print(f"  dumped {2 * len(pairs)} games ({n_bad} unparsable) -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="validation",
                    help="dataset split; this dataset has train/select/validation/test. "
                         "Default validation keeps test unspent for final numbers.")
    ap.add_argument("--n", type=int, default=20, help="prompts to judge per mode")
    ap.add_argument("--modes", default="no_thinking,thinking",
                    help="comma-separated: no_thinking, thinking, batch")
    ap.add_argument("--provider", default="vector", choices=sorted(OPENAI_PROVIDERS))
    ap.add_argument("--base_url", default="",
                    help="override the provider's base URL (empty = its default)")
    ap.add_argument("--max_parallel", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=-1.0,
                    help="negative = the provider's default pacing")
    ap.add_argument("--reasoning_effort", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--batch_poll_seconds", type=float, default=15.0)
    ap.add_argument("--json_out", default="",
                    help="append one JSON object per mode to this file")
    ap.add_argument("--dump_dir", default="",
                    help="write every judge generation + parsed label to "
                         "<dump_dir>/<mode>.jsonl for inspecting failures")
    args = ap.parse_args()

    key_env = OPENAI_PROVIDERS[args.provider].api_key_env
    if not os.environ.get(key_env):
        raise SystemExit(f"{key_env} is not set (VECTOR_INFERENCE_API_KEY lives in ~/.bashrc).")

    print(f"Loading {args.n} chosen/rejected pairs from {args.dataset}:{args.split} ...")
    pairs = load_pairs(args.dataset, args.split, args.n)
    print(f"Loaded {len(pairs)} pairs. "
          f"Mean chosen len {statistics.mean(len(c) for _, c, _ in pairs):.0f} chars, "
          f"rejected {statistics.mean(len(r) for _, _, r in pairs):.0f} chars.")

    rows = [run_mode(m.strip(), pairs, args)
            for m in args.modes.split(",") if m.strip()]

    if args.json_out:
        import json
        with open(args.json_out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"\n{'=' * 72}\nSUMMARY  ({args.model}, n={len(pairs)})\n{'=' * 72}")
    print(f"{'mode':<14}{'agreement':>11}{'s/game':>9}{'median lat':>12}"
          f"{'wall':>9}{'dropped':>9}")
    for r in rows:
        print(f"{r['mode']:<14}{r['agreement']:>11.3f}{r['per_game']:>9.2f}"
              f"{r['median_latency']:>12.2f}{r['wall']:>8.0f}s{r['dropped']:>9}")


if __name__ == "__main__":
    main()
