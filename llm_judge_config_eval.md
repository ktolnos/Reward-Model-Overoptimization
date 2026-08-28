# LLM-as-judge config evaluation (Vector Inference proxy)

Which hosted judge to use for policy eval: **which model** × **thinking on/off**,
scored on agreement with human preference labels, position bias, drop rate and
speed.

**TL;DR — use `gpt-oss-120b`, no-thinking, `max_parallel=32`.** The top four
models are *statistically indistinguishable* on judgment quality, so the pick is
made on the other axes: gpt-oss-120b drops **0/500** prompts, has the highest
throughput, and is from a **different model family** than the Qwen policies and
RMs, so it cannot self-prefer policy rollouts. Thinking is **not** better for
either family tested (Δ = −0.009, p ≈ 0.7 for *both*) and costs 13–20× the
latency plus a 13–28% drop rate. Do not use `Nemotron-3-Nano-Omni-30B-A3B`:
barely above chance, and it decides **57%** of prompts by answer position rather
than content.

The **Batch API is non-functional** on this proxy — see [Batch API](#batch-api).
Judging is **concurrency-bound**, not rate-limited — see [Throughput](#throughput).

---

## Experimental design

### What is measured

Each judge sees a prompt with its dataset **`chosen`** and **`rejected`**
responses and runs the production Arena-Hard-Auto protocol: 2 position-swapped
games per prompt, 5-point verdict, mapped to weighted battles.

| metric | meaning | good |
|---|---|---|
| **agreement** | fraction of battles where the judge picked `chosen` | high (0.5 = coin flip) |
| **position-flip rate** | prompts where the judge names the *opposite* decisive winner when A/B swap — it is deciding on **order, not content** | low |
| **dropped** | prompts with no usable verdict (generation / truncation / parse failure) | low |
| **median latency** | per-request wall-clock | low |

Position-flip is not redundant with agreement: it is what exposed Nemotron as
order-driven rather than merely weak, and Nemotron had **zero** dropped prompts,
so no failure counter would have caught it.

### Data

`ktolnos/helpsteer3-qwen35_annotated_human`, **`validation`** split — the dataset
the current pipeline actually trains on (SFT / RM / GRPO).

Two properties of this choice matter:

1. **Human labels.** The `chosen`/`rejected` labels are human annotations, so
   agreement measures judge quality. The `helpsteer3_goldSkywork-…` variants are
   labelled by the Skywork gold RM, so agreement there would measure *how well
   the judge imitates Skywork* — partly circular, since Skywork is the gold RM
   for evaluation. (Symptom of the difference: in the gold-labelled set `chosen`
   is *longer* than `rejected` (1182 vs 994 chars), so it rewards length bias; in
   the human set it is *shorter* (804 vs 891).)
2. **Deduped by prompt.** HelpSteer3 carries several response-pairs per prompt —
   `validation` is 1515 rows but only **883 unique prompts**, one appearing 25
   times. Raw rows are therefore not independent samples. The probe loads through
   the production loader (`policy_eval.benchmarks._load_preference_split`), which
   applies `dedupe_dataset_by_prompt` and the same seeded
   `shuffle(seed=42).select(range(n))` subsample the preference benchmark uses.

`validation` is used rather than `test` so the test split stays unspent for final
numbers (BENCHMARK.md §8) — choosing a judge is a sweep-style decision.

### Sample size

n = **500** prompts for the model comparison; n = **250** for thinking (≈10×
costlier per game).

Powered from an n=100 pilot: paired SD ≈ 0.29 and the tightest gap was Δ ≈ 0.055,
which needs n ≈ 250 to clear significance at all, so n = 500 gives a comfortable
margin (CI half-width ≈ 0.025). **The pilot justified the scale-up on its own** —
between n=100 and n=500 the ranking reshuffled substantially (Qwen3_8-27B
0.854→0.740, gpt-oss-120b 0.720→0.733 on pooled agreement); the n=100 leader was
a favourable sample.

### Statistics

- **Paired bootstrap over prompts** (20 000 resamples). Every model judges the
  same prompt sample, so each resample draws *prompts*, not judgments — this
  cancels prompt difficulty and is far tighter than treating the runs as
  independent. The **prompt** is the independent unit, so per-prompt mean
  agreement is the statistic (this differs slightly from the probe's headline
  pooled-battle mean, which weights each battle equally as Arena-Hard does;
  rankings are identical).
- **Holm-Bonferroni correction.** Testing the leader against 5 rivals is 5
  simultaneous tests; uncorrected, "significant vs all 5" would fire far more
  often than α = 0.05 suggests.
- Thinking vs no-thinking is compared **paired on the same prompts**: the
  subsample is `shuffle(seed=42).select(range(n))`, so an n=250 sample is a
  strict *prefix* of the n=500 one (verified empirically). No second no-thinking
  run is needed.

### Held constant

Greedy (temperature 0, top_p 1.0); Arena-Hard-Auto v2.0 prompt and parser; the
production `LLMJudge` + `OpenAICompatibleBackend`; 8-way concurrency paced at
100 RPM (below the proxy's observed 120 project-wide cap). Concurrency was varied
only in the [Throughput](#throughput) section, which confirms it does not affect
agreement.

---

## Results

### Model comparison (n = 500, no-thinking)

| model | agreement | 95% CI | flip | dropped | med lat | games/min |
|---|---|---|---|---|---|---|
| Qwen3_5-122B-A10B | **0.758** | [0.724, 0.791] | 0.131 | 12/500 | 7.3s | 43 |
| **Qwen3_8-27B** | 0.740 | [0.705, 0.774] | **0.129** | 5/500 | **2.9s** | 74 |
| Qwen3_6-35B-A3B | 0.736 | [0.702, 0.769] | 0.211 | 7/500 | 8.6s | 39 |
| gpt-oss-120b | 0.733 | [0.699, 0.766] | 0.148 | **0/500** | 4.9s | 87 |
| Qwen3-Coder-Next | 0.693 | [0.658, 0.726] | 0.210 | 0/500 | 4.8s | 95 |
| Nemotron-3-Nano-Omni-30B-A3B | 0.552 | [0.526, 0.577] | **0.570** | 0/500 | 4.8s | 98 |

All six are above chance at n=500 (every CI excludes 0.50).

### Is the leader better than everything else? **No.**

Paired bootstrap vs `Qwen3_5-122B-A10B`, Holm-corrected:

| rival | Δ | 95% CI | p | p_holm | |
|---|---|---|---|---|---|
| Qwen3_8-27B | +0.020 | [−0.006, +0.046] | 0.132 | 0.265 | n.s. |
| Qwen3_6-35B-A3B | +0.018 | [−0.010, +0.046] | 0.214 | 0.265 | n.s. |
| gpt-oss-120b | +0.026 | [−0.004, +0.056] | 0.088 | 0.265 | n.s. |
| Qwen3-Coder-Next | +0.068 | [+0.034, +0.101] | <0.001 | <0.001 | **SIG** |
| Nemotron-3-Nano-Omni-30B-A3B | +0.207 | [+0.167, +0.245] | <0.001 | <0.001 | **SIG** |

**The top four are statistically indistinguishable** (spread 0.025, all CIs
straddle zero). Even at n=500 the answer is a four-way tie, so no larger sweep
would settle it cheaply — the remaining gaps are smaller than the pilot's
sampling noise.

### English vs non-English

HelpSteer3 is 40% non-English and the eval runs on it, so this is decision-relevant.

| model | English (298) | non-English (202) | Δ | 95% CI |
|---|---|---|---|---|
| Qwen3_5-122B-A10B | 0.724 | 0.810 | −0.087 | [−0.152, −0.018] |
| Qwen3_8-27B | 0.710 | 0.786 | −0.076 | [−0.144, −0.006] |
| Qwen3_6-35B-A3B | 0.713 | 0.770 | −0.057 | [−0.124, +0.005] |
| gpt-oss-120b | 0.718 | 0.756 | −0.038 | [−0.103, +0.028] |
| Qwen3-Coder-Next | 0.677 | 0.715 | −0.037 | [−0.105, +0.030] |
| Nemotron | 0.534 | 0.578 | −0.044 | [−0.098, +0.009] |

Every model scores **higher on non-English** — English prompts in this split are
harder for all judges, not a per-model weakness. The Qwen judges have the largest
gaps (the only two whose CIs exclude zero), i.e. they gain most from non-English;
gpt-oss is the flattest. On English alone all four leaders are within 0.014.

## Thinking vs no-thinking

### Design

Run on **two model families** — `Qwen3_8-27B` (Qwen) and `gpt-oss-120b` (harmony)
— so the result cannot be an artifact of one model's reasoning implementation.

- n = **250** prompts (500 games) per model, vs the 500-prompt no-thinking runs.
- Compared **paired on the same prompts**: the subsample is
  `shuffle(seed=42).select(range(n))`, so the n=250 set is a strict *prefix* of
  the n=500 set (verified empirically). No second no-thinking run was needed.
  Pairing is over prompts *both* modes produced a usable verdict for — 218
  prompts for Qwen3_8-27B, 180 for gpt-oss-120b (the thinking runs drop more).
- Token budget **16 384**, not the 4096 default. This was necessary: see
  [budget](#the-token-budget-is-not-the-real-problem).
- Thinking is requested per family by `_reasoning_fields`: gpt-oss is harmony
  format and always reasons, so `enable_thinking` maps to
  `reasoning_effort: high/low`; Qwen models take
  `chat_template_kwargs={"enable_thinking": ...}`.

### Result — the same null in both families

| model | mode | agreement | Δ (no-think − think) | 95% CI | p |
|---|---|---|---|---|---|
| Qwen3_8-27B | no-think | 0.772 | **−0.009** | [−0.046, +0.029] | 0.67 |
| | thinking | 0.776 | | | |
| gpt-oss-120b | no-think | 0.722 | **−0.009** | [−0.052, +0.034] | 0.70 |
| | thinking | 0.734 | | | |

**Two independent families land on the identical delta (−0.009) with p ≈ 0.7.**
Thinking buys nothing for pairwise preference judging under the Arena-Hard
protocol — a far stronger conclusion than either run alone would support.

### Cost — 13× to 20×

| model | mode | med lat | p90 | games/min | wall (500 games) | dropped |
|---|---|---|---|---|---|---|
| Qwen3_8-27B | no-think | **2.9s** | 16.4s | **74** | 809s | 5/500 (1.0%) |
| | thinking | 37.5s | 137s | 8 | 3683s | 32/250 (12.8%) |
| gpt-oss-120b | no-think | **4.9s** | 8.2s | **87** | 692s | **0/500** |
| | thinking | 98.7s | 213s | 4 | 7782s | 70/250 (**28%**) |

Thinking is **12.8×** (Qwen3_8-27B) and **20.3×** (gpt-oss-120b) the latency, and
the two thinking runs alone cost 3h10m of the ~5h total spent on this document.

### The token budget is not the real problem

Failure breakdown over the 500 games of each thinking run:

| model | generation | truncation | parse | prompts dropped |
|---|---|---|---|---|
| Qwen3_8-27B | 0 | 17 | 26 | 32/250 (12.8%) |
| gpt-oss-120b | 0 | 8 | **84** | 70/250 (28%) |

At the **4096** default the failure is truncation-dominated: an early n=20
gpt-oss run dropped **11/20** prompts, every one a truncation. Raising to 16 384
removes almost all of that (8 truncations in 500 games).

What remains is **parse failures**, and they are a behaviour, not a budget: the
Arena-Hard thinking prompt orders the judge to *"begin your evaluation by
generating your own answer to the prompt"*, and the model frequently writes only
that answer, stops cleanly, and never issues a verdict. Inspecting the dumped
generations (`--dump_dir`) confirms it — the texts end mid-explanation of the
judge's *own* answer with no `[[A>B]]` anywhere. More tokens will not fix this;
it needs a different prompt.

### Secondary observations

- Thinking's only win is a slightly **lower position-flip rate** (0.101 for
  Qwen3_8-27B, 0.128 for gpt-oss vs 0.129/0.148 no-thinking) — reasoning does
  reduce order-dependence a little. Nowhere near enough to justify 13–20× cost
  and losing 13–28% of prompts.
- gpt-oss-120b's headline **"0 drops" is a no-thinking property**. In thinking
  mode it is the *worse* of the two (28% vs 12.8%), because the
  writes-its-own-answer failure hits harmony models harder (84 parse failures vs
  26).
- The drops are **not random**: they concentrate on prompts the judge finds worth
  answering at length, so a thinking judge silently evaluates an easier subset —
  a second reason to prefer no-thinking beyond cost.

## Throughput

**Judging is concurrency-bound, not rate-limited.** Little's Law on the n=500 runs
(`throughput = max_parallel / mean_latency`) reproduces every observed rate at
`max_parallel=8` — 8/5.5s×60 = 87 g/min for gpt-oss — so the 100 RPM cap was
*never reached* and the rate limiter never engaged.

Measured (n=100 prompts = 200 games each):

| model | max_parallel | rpm | games/min | med lat | implied mean | agreement |
|---|---|---|---|---|---|---|
| gpt-oss-120b | 8 | 100 | 89 | 4.8s | 5.4s | 0.734 |
| gpt-oss-120b | **16** | 100 | **96** | 9.3s | 10.0s | 0.724 |
| gpt-oss-120b | 32 | 100 | 98 | 18.8s | 19.6s | 0.722 |
| gpt-oss-120b | 32 | 115 | **111** | 16.3s | 17.3s | 0.714 |
| Qwen3_5-122B-A10B | 8 | 100 | 36 | 8.8s | 13.4s | 0.778 |
| Qwen3_5-122B-A10B | **16** | 100 | **48** | 12.1s | 20.2s | 0.801 |
| Qwen3_5-122B-A10B | 32 | 100 | 58 | 19.4s | 33.2s | 0.806 |
| Qwen3_5-122B-A10B | 32 | 115 | 59 | 18.6s | 49.5s | 0.802 |

**Naive Little's Law over-predicted the gains.** It assumes latency is constant in
`max_parallel`; it is not — per-request latency grows roughly *linearly* with
concurrency (gpt-oss: 5.4→10.0→19.6s mean as P goes 8→16→32), the signature of a
**saturated server queueing our requests**. Throughput therefore plateaus rather
than scaling: the predicted 43→100 g/min for Qwen3_5-122B was really 36→58.

Real gains from `max_parallel` 8 → 32: **+10%** (fast judge, 89→98 g/min) and
**+61%** (slow judge, 36→58 g/min). **32 is the chosen default** — it takes the
full available gain, and the slow-judge case is where it matters (a 122B judge is
far less unattractive at 58 g/min than at 36).

The cost is queueing latency: mean per-request latency grows from 5.4s to 19.6s
(gpt-oss) and 13.4s to 33.2s (Qwen3_5-122B) as concurrency goes 8 → 32, and the
proxy is shared with the rest of the project. **Drop to 16** for most of the gain
at half the latency if the proxy is busy or requests start timing out. Agreement
is unchanged across every config (drift is n=100 noise), so concurrency does not
trade quality for speed.

Raising `--judge_rpm` to 115 (under the observed 120 cap) buys a further ~13% on
a fast judge, but spends more of a **shared** budget; left at the provider
default of 100.

> **Caveat: pacing is per-process.** `_RateLimiter` is a backend instance
> attribute, so two concurrent evals each assume the full RPM budget and together
> can exceed the shared 120 cap. Fine while evals run one at a time; halve
> `--judge_rpm` manually if running two.

---

## Recommendation

**`gpt-oss-120b`, no-thinking, `max_parallel=32`** — set in `evaluate_policy.sh`.

The top four tie on quality, so the decision falls to the remaining axes:

| axis | gpt-oss-120b | Qwen3_8-27B | winner |
|---|---|---|---|
| agreement | 0.733 | 0.740 | tied (n.s. under Holm) |
| dropped | **0/500** | 5/500 | gpt-oss |
| throughput | **87 g/min** | 74 g/min | gpt-oss |
| position-flip | 0.148 | **0.129** | Qwen3_8 |
| non-English | 0.756 | **0.786** | Qwen3_8 |
| **family independence** | **different family** | same family as policy/RM | **gpt-oss** |

**Family independence is the deciding factor.** The policies (Qwen3-0.6B,
Qwen3.5-4B-Base) and every RM in the pipeline are Qwen. A Qwen judge risks
self-preference toward Qwen policy rollouts, which would inflate win rates and
bias checkpoint selection — a *systematic* error exactly where the judge is
load-bearing. The flip-rate and non-English gaps favouring Qwen3_8-27B are small
(0.019 and 0.030), non-significant, and measured **off-distribution** (dataset
responses, not policy rollouts).

> **This self-preference risk is argued, not measured.** HelpSteer3 records no
> model provenance for its responses, and both judged responses come from the
> dataset, so this benchmark is structurally blind to it. It *can* be measured
> directly: judge a past eval's cached per-example logs (Qwen policy responses vs
> dataset `chosen`) with both models via `--llm_judge_on_cached`; if the Qwen
> judge gives the Qwen policy systematically higher win rates than gpt-oss does,
> that is self-preference on the real distribution. Needs no GPU and no
> regeneration. **Worth running before treating this choice as settled.**

`Qwen3_8-27B` remains the right pick if judging is dominated by non-English
prompts or if position bias matters more than family independence.

**Avoid `Nemotron-3-Nano-Omni-30B-A3B`.** At 0.552 it is technically above chance,
but it flips its decisive winner on **57%** of prompts when the answers swap
position — it is judging by order. It is also the fastest model with zero drops,
so speed and failure counters would both have endorsed it; only the flip rate
catches it.

---

## Batch API

**Not usable on this proxy.** The surface exists — `GET /v1/files` and
`GET /v1/batches` return 200, and the OpenAPI spec documents `CreateBatchRequest`
— but upload fails:

```
POST /v1/files → HTTP 502
{"detail":"Upstream file upload failed: 404 {\"detail\":\"Not Found\"}"}
```

The upstream behind the proxy has no file endpoint, so there is no
`input_file_id` and no batch can be created. Confirmed server-side: hand-built
curl multipart matching the published schema fails identically.

So batch cannot relieve the 100 RPM live-path budget. `--llm_judge_use_batch_api`
is implemented and unit-tested against a local stub, and now fails with an
explicit message rather than a raw `HTTPError` (which would otherwise surface
*after* the whole generation phase). Leave it off.

---

## Reproduce

`VECTOR_INFERENCE_API_KEY` must be in the environment (it lives in `~/.bashrc`).

```bash
# One model, one mode — quick check.
venv/bin/python vector_judge_probe.py --model Qwen3_8-27B --n 100 --modes no_thinking

# Full sweep: every model no-thinking, then the pick in thinking mode.
# Resumable: skips any (model, mode) already in results.jsonl.
for M in Nemotron-3-Nano-Omni-30B-A3B Qwen3-Coder-Next Qwen3_5-122B-A10B \
         Qwen3_6-35B-A3B Qwen3_8-27B gpt-oss-120b; do
  venv/bin/python vector_judge_probe.py --model "$M" --n 500 --modes no_thinking \
      --json_out sweep/results.jsonl --dump_dir "sweep/dumps/$M"
done
# Thinking mode, for whichever candidates you want to test it on.
for M in Qwen3_8-27B gpt-oss-120b; do
  venv/bin/python vector_judge_probe.py --model "$M" --n 250 --modes thinking \
      --max_new_tokens 16384 --json_out sweep/results.jsonl --dump_dir "sweep/dumps/$M"
done

# Ranking + Holm-corrected paired bootstrap + thinking + language breakdown.
venv/bin/python analyze_judge_sweep.py --sweep_dir sweep \
    --by_language --thinking_model gpt-oss-120b
```

`--dump_dir` writes every judge generation with its parsed label to
`<dir>/<model>/<mode>.jsonl`, so unparsable verdicts can be inspected without
re-running the sweep — that is how the thinking failure mode above was diagnosed.

Total cost of the runs in this document: ~5h wall-clock at 100 RPM / 8-way
concurrency — 6 × 1000 games no-thinking (2h45m), 2 × 500 games thinking (3h10m,
the dominant cost), plus the throughput sweep (~30m). At the new
`max_parallel=32` default the no-thinking sweep would take ~1h40m.

### Using it in eval

Preferred: keep the judge out of the GPU job and run it afterwards over the
cached generations. `judge_cached.sh` asks for no GPU, carries
`#SBATCH --dependency=singleton` so judge passes queue behind one another (the
RPM budget is shared project-wide but `_RateLimiter` is per-process — see the
caveat under [Throughput](#throughput)), and resumes the generating eval's wandb
run from its per-example `_manifest.json`:

```bash
GPU=$(sbatch --parsable evaluate_policy.sh)
sbatch --dependency=singleton,afterok:$GPU judge_cached.sh
```

It judges the **selected + final** checkpoints (~28 min each at 2.8k games and
~98 games/min); `--judge_all_checkpoints` widens that to all ~20 (~9.5 h).

Judging inline in the GPU job instead:

```bash
sbatch evaluate_policy.sh --vector_judge                    # gpt-oss-120b, no-thinking
sbatch evaluate_policy.sh --vector_judge --judge_thinking   # not recommended, see above
sbatch evaluate_policy.sh --judge_model Qwen3_8-27B         # alternative (see Recommendation)
sbatch evaluate_policy.sh --vector_judge --judge_max_parallel 16   # gentler on a busy proxy
sbatch evaluate_policy.sh --vector_judge --judge_rpm 115          # ~13% faster, spends more shared budget
```

---

## Caveats

- Agreement is measured against **single human preference labels**, which carry
  their own annotator noise; HelpSteer3-style datasets have a known accuracy
  ceiling around 0.7–0.8 from inter-annotator disagreement. The top four models
  sitting at 0.73–0.76 may be **near that ceiling**, which would explain why they
  are indistinguishable — the remaining gap may be unmeasurable against this
  label set rather than merely small.
- This measures judging of **`chosen` vs `rejected` dataset responses**, not
  policy-vs-baseline responses as in a real eval. Response distribution differs
  (dataset responses are human/model-written, not policy rollouts).
- Language was broken out only as English vs non-English (see above), not per
  language; with 202 non-English prompts spread across many languages there is no
  power for a per-language read.
- **Self-preference is argued, not measured** — see the box under
  [Recommendation](#recommendation). This is the largest open question behind the
  choice of judge.
- Single seed (temperature 0). Run-to-run variation from the serving stack is not
  quantified.
