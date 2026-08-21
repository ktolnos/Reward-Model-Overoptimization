# Math-domain regression test for policy / SFT models

A quick, repeatable way to check whether a model (base, SFT, or post-GRPO checkpoint)
has regressed on math reasoning. Uses **lm-evaluation-harness** + **math_verify**.

Benchmarks: **GSM8K** (100), **MATH-500** (100), **AIME-2025** (full 30).

---

## 0. Why these choices (read once)

- **HF backend, not vLLM.** Our Qwen3.5 SFT checkpoints are `Qwen3_5ForCausalLM`, which
  isn't in the vLLM registry — only the GRPO/policy_eval path loads them (via
  `qwen35_vllm_patch.py` + `vllm_weight_loader.py` prefix remap). lm-eval's plain vLLM
  backend would mis-load the weights, so use `--model hf`. (Linear attention falls back to a
  slow torch impl; fine for ~230 prompts.)
- **These are reasoning models.** Qwen3.5 base/SFT emit `<think>…</think>` and answer in
  `\boxed{}` *by default* (it's a Qwen pretraining prior, not taught by HelpSteer3 SFT). The
  chat template's `enable_thinking` switch controls this. Gemma uses `<|think|>` and defaults
  thinking **off**; it answers in prose unless instructed + thinking on.
- **Score with `math_verify`, not regex/boxed-only.** Models present answers in different
  formats (Qwen boxes, Gemma writes prose). `math_verify` extracts from any format and checks
  by symbolic equivalence (`14/3` ≡ `\frac{14}{3}`, `0.5` ≡ `\frac12`, tuples, intervals),
  and falls back to string match for text answers (names, even/odd, multiple-choice letters —
  12/500 of MATH-500).

---

## 1. One-time setup

```bash
cd /nas/ucb/eop/Reward-Model-Overoptimization
source activate
python -m pip install math_verify           # pulls latex2sympy2_extended + antlr4 only; does NOT touch torch/transformers/vllm
```

Create a task dir `eval_math_tasks/` with three files:

**`eval_math_tasks/math_verify_utils.py`**
```python
from typing import Dict, List
from math_verify import (parse, verify, LatexExtractionConfig,
                         ExprExtractionConfig, StringExtractionConfig)
_GOLD = [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()]
_PRED = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig(), StringExtractionConfig()]

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    pred = results[0]
    key = next(k for k in doc if k.lower() == "answer")
    gold = str(doc[key])
    try:
        ok = bool(verify(parse("\\boxed{" + gold + "}", extraction_config=_GOLD),
                         parse(pred, extraction_config=_PRED)))
    except Exception:
        ok = False
    return {"exact_match": int(ok)}
```

**`eval_math_tasks/math500.yaml`**
```yaml
task: math500_mv
dataset_path: HuggingFaceH4/MATH-500
output_type: generate_until
test_split: test
doc_to_text: "{{problem}}\n\nPlease reason step by step, and put your final answer within \\boxed{}."
doc_to_target: "{{answer}}"
process_results: !function math_verify_utils.process_results
generation_kwargs: {until: ["<|im_end|>", "</s>"], do_sample: false, temperature: 0.0}
metric_list: [{metric: exact_match, aggregation: mean, higher_is_better: true}]
num_fewshot: 0
metadata: {version: 1.0}
```

**`eval_math_tasks/aime25.yaml`** — same as above but:
```yaml
task: aime25_mv
dataset_path: math-ai/aime25
# (everything else identical to math500.yaml)
```

For GSM8K use the built-in `gsm8k_cot_zeroshot` and read its **flexible-extract** number
(strict-match is always ~0 here — the model doesn't say "The answer is X").

---

## 2. Run

```bash
CKPT=<path-or-hf-id>           # e.g. scripts/rlhf/logs_sft/.../checkpoint-744
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m lm_eval \
  --model hf \
  --model_args "pretrained=${CKPT},dtype=bfloat16,trust_remote_code=True,max_length=4096,enable_thinking=False" \
  --include_path eval_math_tasks \
  --tasks gsm8k_cot_zeroshot,math500_mv,aime25_mv \
  --apply_chat_template \
  --gen_kwargs max_gen_toks=2048 \
  --batch_size 32 --limit 100 --seed 0 \
  --output_path eval_math_out --log_samples
```

`--limit 100` → 100 GSM8K, 100 MATH-500, all 30 AIME.

### Two modes — always compare like-for-like
| mode | model_args | `max_gen_toks` | note |
|------|-----------|----------------|------|
| **no-think** (fast) | `enable_thinking=False` | `2048` | answers fit easily; good for quick regression |
| **think** (capability) | `enable_thinking=True` (or omit for Qwen — it's the default) | `12288`+ | **required for math**: at 2048, ~27/30 AIME & ~24/100 MATH-500 truncate mid-trace and score 0 |

For non-Qwen models, set `enable_thinking` per that model's template (Gemma needs it **on**
to box reliably; Qwen defaults on).

---

## 3. Reference baselines

Qwen3.5-4B-Base SFT (`logs_sft/20260408_231230_1089122/checkpoint-744`), measured 2026-06-24:

| mode | GSM8K | MATH-500 | AIME-2025 |
|------|-------|----------|-----------|
| no-think @2048 | ~70% | ~72% | ~3% (1/30) |
| think (GSM8K @2048) | ~88% | needs ≥12k budget | needs ≥12k budget |

> These were taken with a boxed-only scorer; the math_verify scorer above may be a few points
> higher on MATH-500. **On first run, re-establish your own baseline with the exact command
> above, then compare future checkpoints against that.**

---

## 4. Gotchas / interpreting results

- **A "0%" is almost always a scoring/format artifact, not capability** — verify by reading
  `eval_math_out/**/samples_*.jsonl` before believing a big drop.
- **Truncation = false 0.** Check generation lengths; if many hit `max_gen_toks`, raise the budget.
- **Format compliance is model-specific.** The `\boxed{}` instruction + thinking-on is what makes
  text/word answers (ellipse, even, multiple-choice) extract cleanly; string-match is exact-ish
  and unforgiving of stray prose.
- **Greedy + fixed seed** → deterministic; a real regression should reproduce.
```
