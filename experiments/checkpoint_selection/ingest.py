"""Parse the preserved per-checkpoint eval JSONL files.

Each JSONL row is ``{"benchmark", "checkpoint", "prompt", "response"}``. The
``prompt`` field is the chat-template-formatted Qwen prompt produced by the
policy tokenizer; we hash that verbatim for prompt identity. Re-scoring with a
different RM needs the underlying ``prompt_messages`` (so the RM's own chat
template can be applied), so we also load the source HF dataset and build a
lookup keyed by the same chat-template hash.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset

from data_utils import format_and_validate_preference_sample

from . import manifest as M


# ---------------------------------------------------------------------------
# Prompt hashing
# ---------------------------------------------------------------------------

def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[: M.PROMPT_HASH_LEN]


def ab_bucket(prompt_hash_str: str, salt: str = "v1") -> str:
    """Deterministic A/B partition. Same hash → same bucket for every run."""
    h = hashlib.sha256(f"{salt}:{prompt_hash_str}".encode("utf-8")).digest()
    return "A" if h[0] % 2 == 0 else "B"


# ---------------------------------------------------------------------------
# JSONL ingestion
# ---------------------------------------------------------------------------

@dataclass
class EvalRow:
    grpo_run_id: str
    grpo_run_idx: int
    benchmark: str
    checkpoint: int
    prompt_hash: str
    prompt: str
    response: str


def iter_eval_rows(eval_file_path: str, run_spec: M.RunSpec) -> Iterable[EvalRow]:
    with open(eval_file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("benchmark") != "preference":
                continue
            prompt = rec["prompt"]
            yield EvalRow(
                grpo_run_id=run_spec.wandb_id,
                grpo_run_idx=run_spec.idx,
                benchmark=rec["benchmark"],
                checkpoint=int(rec["checkpoint"]),
                prompt_hash=prompt_hash(prompt),
                prompt=prompt,
                response=rec["response"],
            )


def load_eval_rows(eval_root: str, runs: Optional[List[M.RunSpec]] = None) -> List[EvalRow]:
    """Load preference-benchmark rows across the runs in ``runs`` (default: all)."""
    runs = runs if runs is not None else list(M.RUNS)
    out: List[EvalRow] = []
    for r in runs:
        path = os.path.join(eval_root, r.eval_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"eval file for run {r.idx} ({r.wandb_id}) not found: {path}"
            )
        n0 = len(out)
        for row in iter_eval_rows(path, r):
            out.append(row)
        print(f"[ingest] run {r.idx} {r.wandb_id}: {len(out) - n0} preference rows from {os.path.basename(path)}")
    return out


# ---------------------------------------------------------------------------
# Dataset prompt-messages lookup (hash → prompt_messages list)
# ---------------------------------------------------------------------------

def _qwen_format_prompt(prompt_messages: list, tokenizer) -> str:
    """Recreate the exact chat-templated prompt the policy generation used.

    Matches ``policy_eval.benchmarks._format_preference_prompt`` — same call to
    ``format_and_validate_preference_sample`` with a dummy chosen response.
    """
    chosen_messages = list(prompt_messages) + [
        {"role": "assistant", "content": ""}
    ]
    prompt_text, _, _ = format_and_validate_preference_sample(
        chosen_messages,
        tokenizer,
        length_config="default",
        skip_validation=True,
        sample_id=0,
        context="Checkpoint-selection ingestion",
    )
    return prompt_text


def build_prompt_messages_index(
    tokenizer,
    dataset_names: Iterable[str] = (M.DATASET_NAME_25PCT,),
    split: str = "test",
) -> Dict[str, list]:
    """Build ``{prompt_hash: prompt_messages}`` from each source dataset.

    Only the 25pct dataset is indexed by default: the cross-run analysis is
    restricted to the 25pct.test prompt intersection (per the plan, 367
    prompts is sufficient). Runs 6/7 were evaluated on the full test split,
    so their JSONLs contain extra prompts that won't match this index —
    ``assemble_scoring_inputs`` drops them with a warning. Pass
    ``dataset_names=(M.DATASET_NAME,)`` to instead score the full test
    split for those two runs.
    """
    index: Dict[str, list] = {}
    for ds_name in dataset_names:
        try:
            ds = load_dataset(ds_name, split=split)
        except Exception as e:
            print(f"[ingest] could not load {ds_name}: {e}")
            continue
        added = 0
        for ex in ds:
            prompt_messages = ex["chosen"][:-1]
            try:
                fmt = _qwen_format_prompt(prompt_messages, tokenizer)
            except Exception as e:
                print(f"[ingest] formatting failure on {ds_name}: {e}")
                continue
            h = prompt_hash(fmt)
            if h not in index:
                index[h] = prompt_messages
                added += 1
        print(f"[ingest] {ds_name} split={split}: indexed {added} new prompts")
    return index
