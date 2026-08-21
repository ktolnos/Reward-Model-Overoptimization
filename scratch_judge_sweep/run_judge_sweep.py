#!/usr/bin/env python
"""Sweep LLM-judge configs on cached policy generations.

Reuses the production judge (``policy_eval.judges.LLMJudge`` + ``VLLMBackend``) so
the sweep measures exactly the judge the pipeline runs. Loads cached preference
generations (policy response + chosen baseline + gold-RM scores) from a previous
eval's per-example parquet logs, then judges the SAME (checkpoint, prompt) pairs
under each config:

    quant  in {None (bf16), "fp8"}          -- model precision
    think  in {True, False}                 -- reasoning trace on/off

One model load per quant level; both thinking modes reuse it. All pairs across
all sampled checkpoints are judged in a single batched call per config so vLLM
batches maximally. Raw per-game verdicts + failure attribution + wall time are
written to parquet for offline analysis (analyze_judge_sweep.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_eval.judges import (  # noqa: E402
    LLMJudge, VLLMBackend, JudgeGenParams, battles_from_game_labels,
    ARENA_HARD_SYSTEM_PROMPT,
)

# Concise chain-of-thought judge prompt. The canonical Arena-Hard thinking prompt
# tells the model to first write its own full answer, then a written comparison,
# then the verdict -- with gemma-4-31B that overflows even a 3072-token budget
# (its <|channel>thought is very verbose), so a majority of prompts never reach a
# verdict and are dropped. This variant keeps the same evaluation criteria but
# demands brief internal reasoning and an immediate verdict, to test whether a
# usable thinking judge is achievable without the truncation blow-up.
ARENA_HARD_CONCISE_THINKING_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by "
    "two AI assistants to the user prompt displayed below. You will be given assistant A's "
    "answer and assistant B's answer. Your job is to decide which assistant's answer is better.\n\n"
    "Reason briefly and efficiently: in at most a few short sentences, note the key differences "
    "in correctness, helpfulness, relevance, conciseness, and any missing important information. "
    "Do NOT write your own full answer to the prompt and do NOT restate the answers. Keep your "
    "reasoning under ~150 words, then immediately give your verdict.\n\n"
    "Conclude with exactly one of the following labels:\n\n"
    "1. [[A>>B]], Assistant A is significantly better\n"
    "2. [[A>B]], Assistant A is slightly better\n"
    "3. [[A=B]], a tie (relatively the same)\n"
    "4. [[B>A]], Assistant B is slightly better\n"
    "5. [[B>>A]], Assistant B is significantly better\n\n"
    'End your response with, e.g.: "My final verdict is [[A=B]], a tie".'
)


def load_pairs(per_example_dir, checkpoints, n_prompts, seed):
    """Return a DataFrame of (checkpoint, prompt) pairs to judge.

    Columns: checkpoint, prompt_uid, prompt_messages (list), policy (str),
    baseline (str), gold_policy (float), gold_baseline (float), over_budget.
    The same prompt subsample (seeded) is used for every checkpoint.
    """
    rng = np.random.default_rng(seed)
    sel_idx = None
    rows = []
    for ck in checkpoints:
        f = os.path.join(per_example_dir, f"preference__checkpoint-{ck}.parquet")
        df = pd.read_parquet(f)
        df = df[df["sample_idx"] == 0].reset_index(drop=True)
        if sel_idx is None:
            sel_idx = np.sort(rng.choice(len(df), size=min(n_prompts, len(df)),
                                         replace=False))
        sub = df.iloc[sel_idx]
        for _, r in sub.iterrows():
            rows.append(dict(
                checkpoint=ck,
                prompt_uid=r["prompt_uid"],
                prompt_messages=json.loads(r["prompt_messages_json"]),
                policy=r["response_text"],
                baseline=r["reference_response_text"],
                gold_policy=float(r["score__rm_gold_rm"]),
                gold_baseline=float(r["chosen_or_baseline_score__gold_rm"]),
                over_budget=bool(r["over_budget"]),
            ))
    return pd.DataFrame(rows)


def run_config(judge, pairs, weight=3):
    """Judge every pair; return per-pair records + config-level stats."""
    t0 = time.time()
    battles, details = judge.score_pairs(
        list(pairs["prompt_messages"]),
        list(pairs["policy"]),
        list(pairs["baseline"]),
        ctx=None,
    )
    elapsed = time.time() - t0

    recs = []
    for i, (_, r) in enumerate(pairs.iterrows()):
        bl = battles[i]
        battle_mean = float(np.mean(bl)) if bl else np.nan
        recs.append(dict(
            checkpoint=r["checkpoint"],
            prompt_uid=r["prompt_uid"],
            gold_policy=r["gold_policy"],
            gold_baseline=r["gold_baseline"],
            over_budget=r["over_budget"],
            game0_label=details.game0_labels[i],
            game1_label=details.game1_labels[i],
            battle_mean=battle_mean,
            game0_text=details.game0_texts[i],
            game1_text=details.game1_texts[i],
        ))
    stats = dict(
        n_pairs=len(pairs),
        elapsed_s=elapsed,
        pairs_per_s=len(pairs) / elapsed if elapsed else 0.0,
        n_generation_failures=details.n_generation_failures,
        n_truncation_failures=details.n_truncation_failures,
        n_parse_failures=details.n_parse_failures,
        n_dropped_prompts=details.n_dropped_prompts,
    )
    return recs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_example_dir", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--checkpoints", default="149,745,1341,2086,2975")
    ap.add_argument("--n_prompts", type=int, default=120)
    ap.add_argument("--think_n_prompts", type=int, default=0,
                    help="if >0, thinking configs judge only the first N prompts "
                         "(a prefix of the same paired set) to bound cost")
    ap.add_argument("--think_prompt", default="concise",
                    choices=["arena", "concise"],
                    help="system prompt for thinking configs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--think_max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--quants", default="none,fp8",
                    help="comma list of quant levels: none|fp8")
    ap.add_argument("--thinks", default="true,false",
                    help="comma list: true|false")
    ap.add_argument("--think_specs", default="",
                    help="override thinking configs with explicit "
                         "'promptmode:maxtokens' specs, comma-separated, e.g. "
                         "'arena:8192,concise:4096'. Each becomes its own config. "
                         "Whether no-think runs is still governed by --thinks "
                         "containing 'false'.")
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    quants = [None if q == "none" else q for q in args.quants.split(",")]
    thinks = [t.strip().lower() == "true" for t in args.thinks.split(",")]

    pairs = load_pairs(args.per_example_dir, checkpoints, args.n_prompts, args.seed)
    print(f"[sweep] {len(pairs)} pairs = {len(checkpoints)} ckpts x "
          f"{pairs['prompt_uid'].nunique()} prompts", flush=True)

    # Thinking configs optionally judge only a prefix of the prompt set (same
    # prompts, so still paired with the no-think configs on that subset).
    think_pairs = pairs
    if args.think_n_prompts:
        keep = list(dict.fromkeys(pairs["prompt_uid"]))[:args.think_n_prompts]
        think_pairs = pairs[pairs["prompt_uid"].isin(keep)].reset_index(drop=True)
    think_sys = (ARENA_HARD_CONCISE_THINKING_SYSTEM_PROMPT
                 if args.think_prompt == "concise" else ARENA_HARD_SYSTEM_PROMPT)
    SYS = {"concise": ARENA_HARD_CONCISE_THINKING_SYSTEM_PROMPT,
           "arena": ARENA_HARD_SYSTEM_PROMPT}

    # Build the list of judge jobs shared across quant levels. Each job is
    # (label_suffix, enable_thinking, system_prompt, max_tokens, pairs).
    jobs = []
    if False in thinks:
        jobs.append(("think=False", False, None, 4, pairs))
    if args.think_specs:
        for spec in args.think_specs.split(","):
            mode, mt = spec.split(":")
            jobs.append((f"think_{mode}{mt}", True, SYS[mode], int(mt), think_pairs))
    elif True in thinks:
        jobs.append(("think=True", True, think_sys, args.think_max_tokens, think_pairs))

    all_recs = []
    all_stats = []
    for quant in quants:
        backend = VLLMBackend(
            args.model, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem, quantization=quant,
        )
        qlabel = quant or "bf16"
        for suffix, think, sysp, mt, cfg_pairs in jobs:
            gp = JudgeGenParams(
                temperature=0.0, top_p=1.0, max_tokens=mt, enable_thinking=think,
            )
            judge = LLMJudge(backend, gen_params=gp, system_prompt=sysp)
            cfg = f"{qlabel}__{suffix}"
            print(f"\n[sweep] === config {cfg} ({len(cfg_pairs)} pairs) ===", flush=True)
            recs, stats = run_config(judge, cfg_pairs)
            for rec in recs:
                rec["config"] = cfg
                rec["quant"] = qlabel
                rec["think"] = think
            all_recs.extend(recs)
            stats.update(config=cfg, quant=qlabel, think=think)
            all_stats.append(stats)
            print(f"[sweep] {cfg}: {stats['elapsed_s']:.1f}s "
                  f"({stats['pairs_per_s']:.2f} pairs/s), "
                  f"gen_fail={stats['n_generation_failures']} "
                  f"trunc={stats['n_truncation_failures']} "
                  f"parse_fail={stats['n_parse_failures']} "
                  f"dropped={stats['n_dropped_prompts']}", flush=True)
        backend.teardown()

    os.makedirs(args.out_dir, exist_ok=True)
    recs_path = os.path.join(args.out_dir, f"{args.tag}_records.parquet")
    stats_path = os.path.join(args.out_dir, f"{args.tag}_stats.json")
    pd.DataFrame(all_recs).to_parquet(recs_path, index=False)
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n[sweep] wrote {recs_path}\n[sweep] wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()
