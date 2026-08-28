# RLHF Training-Methods Benchmark — Specification (v1, draft)

A frozen protocol for comparing RLHF training methods under reward-model
overoptimization. This document is the **contract**: experiments execute against
it, and any change to a "frozen" item bumps the benchmark version (see
[Versioning](#12-versioning--freeze-policy)).

Status legend: ✅ decided · 🔲 TBD (decision pending) · ⏳ to be built (see
[roadmap](#11-implementation-roadmap)).

---

## 1. Purpose & the question being measured

Given a base policy and a preference dataset, how high can a training method push
the **true objective** (independent evaluators + verifiable benchmarks) using only
a **proxy** reward signal it must build itself — *without* being able to peek at
the truth to pick its checkpoint?

Overoptimization shows up as the proxy reward rising while the truth signals
flatten or fall. Because the primary preference signal is **human** (not a single
gold RM), there is no single ground truth; truth is triangulated from a panel of
independent evaluators plus a verifiable, RM-unhackable anchor (IFEval).

---

## 2. Method signature (what a method is) ✅

A method is a function:

```
method(Base policy, preference_dataset) -> trained_policy
```

Frozen rules:

- **The only model a method receives is the base.** It derives any reward
  model(s) and the trained policy from `base + preference_dataset` alone.
- **The SFT step is shared and frozen.** Every method starts from the *same*
  released SFT checkpoint per `(model_family, dataset)`. This removes SFT
  variation as a confound; the only thing that varies between leaderboard
  entries is the method. Downstream comparisons are allowed to change SFT 
  as long as they use the same base and data, but we will run all our experiments
  with a fixed checkpoint.
- **RM size = policy size.** Any proxy RM a method trains shares the policy's
  base model. This removes "win by using a bigger RM than the competition" as a
  confound. (RM *epochs* and *ensemble count* remain internal method
  hyperparameters and may be tuned — see [§9](#9-hyperparameter-sweep-protocol).)
- **Evaluators are off-limits to the method.** The panel RMs, the judge, and the
  IFEval grader are never inputs to a method. A method may not train on, distill
  from, or select against any evaluator (the held-out selection RM in
  [§6](#6-checkpoint-selection--the-no-peek-rule) is the one allowed
  selection signal, and it is method-built, not an evaluator).

> Scientific findings that violate these constraints (e.g. "bigger RMs reduce
> hacking", from `EXPERIMENTS.md`) remain valid results — they are reported as
> *ablations*, not leaderboard entries.

---

## 3. Datasets & splits

### v1 scope ✅
- **HelpSteer3 only** (instruction-following / helpfulness). Single objective.
- **Human-labeled preferences are primary** (`chosen`/`rejected` derived from the
  human ratings). Rationale: with human labels, every off-the-shelf evaluator was
  trained on *different* corpora, so the panel judges the policy independently —
  there is **zero labeler↔evaluator circularity**. The gold-RM-annotated variant
  (`…_annotated_Skywork-Reward-V2-Llama-3.1-8B`) is retained as a **robustness
  arm** run deferred to v1.5.

  - MATH-500 or AIME

### Deferred to v1.5 🔲
- **PKU-SafeRLHF** safety arm — requires safety-specific truth signals
  (Llama-Guard-class / harm classifier / harmlessness judge), not helpfulness
  RMs. Net-new evaluators.
- **Multi-objective Pareto** (helpfulness × harmlessness).

### Preprocessing & splits ✅ / 🔲
- Length budget: the split is **filtered once with the canonical Qwen3.5 tokenizer**
  to **prompt ≤ 1024, response ≤ 1024, full conversation ≤ 2048** — family-agnostic,
  so adding a model never re-filters the dataset. Other tokenizers may expand a few %;
  that's fine — **no truncation** anywhere (`validate_length_or_fail` should use the canonica tokenizer instead). The one place a
  length is needed ahead of time is **vLLM `max_model_len`**, sized per-model with
  headroom — compute it exactly under the model's tokenizer via `compute_max_prompt_length`
  ([data_utils.py:53](data_utils.py#L53)) if you want it tight, else a fixed generous ceiling (e.g. x2).
- Frozen, mutually disjoint four-way split ✅ (built; ratios `train 0.85 / select 0.05 /
  validation 0.05 / test 0.05`, sum 1.0):
  - `train` — method training (SFT is upstream of this).
  - `select` — held-out **prompts** for the no-peek checkpoint-selection rule.
  - `validation` — held-out **prompts** for the hyperparam sweeps.
  - `test` — held-out prompts for truth evaluation. **Never** used for selection.
  Produced by `split_four_way` in
  [scripts/dataset_pipeline/pipeline_common.py](scripts/dataset_pipeline/pipeline_common.py).
  Splitting is **by prompt group, not by row**: HelpSteer3 has multiple response-pairs per
  prompt (~43% of prompts, up to 25 pairs), so all rows sharing a prompt go to the same
  split — otherwise a prompt would leak across splits and break the no-peek rule. Ratios
  apply to the prompt-group count, so row counts deviate slightly. Eval selects the split
  explicitly via `--split` ([evaluate_policy.py](evaluate_policy.py); default `test`, raises
  if absent).
- Within-dataset contamination check ✅: `assert_splits_disjoint`
  ([scripts/dataset_pipeline/pipeline_common.py](scripts/dataset_pipeline/pipeline_common.py))
  asserts `train`/`select`/`validation`/`test` prompts are pairwise disjoint (run in
  Stage 2). External cross-check 🔲: prompts disjoint from ArenaHard/IFEval (follow-up).

---

## 4. Base policies

| Role | Model | Priority |
|---|---|---|
| **Primary anchor** | Qwen3.5-4B | full sweep + seeds + all baselines |
| **Cross-family confirmation** | Gemma (E2B) | nearby-value sweeps around 4B winners |
| **Smallest / scaling point** | Qwen3.5-0.8B | cheap 0.8B→4B(→9B) overopt-scaling curve |
| If compute | Qwen 9B (full-FT) | scaling only |

Notes: 4B is the anchor because it trains in ≈ the same wall-clock as 0.6B on
this dataset. Released artifacts per `(family, dataset)`:
the frozen base ref + the frozen SFT checkpoint + the best checkpoint for each method.

---

## 5. Evaluators

All evaluators are **external and off-limits to methods** ([§2](#2-method-signature-what-a-method-is)).
The pool is partitioned into disjoint **SELECTION** and **TRUTH** sets so the
checkpoint chosen by the no-peek rule is never chosen using a truth-set evaluator.

### TRUTH set (used only for final scoring)
`validation` for hyperparam sweeps, `test` for final params.
Family diversity is required so panel agreement is meaningful (correlated blind
spots are the failure mode).

| Evaluator | Family | Notes |
|---|---|---|
| Skywork-Reward-V2-Llama-3.1-8B | Llama | top RewardBench2; primary RM |
| Skywork-Reward-V2-Qwen3-8B | Qwen | cross-family robustness check |
| Ray2333/GRM-Gemma2-2B-rewardmodel-ft | Gemma | size-efficient |
| Schrieffer/Llama-SARM-4B | Llama | (Llama-correlated — don't over-weight) |
| Open-weight LLM judge ✅ (model 🔲) | configurable via `--llm_judge_model_name` | pairwise win-rate via the Arena-Hard-Auto v2.0 protocol, served in-process by vLLM in the deferred phase. Model proposal `google/gemma-4-31B-it`; other candidates: DeepSeek V4 Flash, Qwen3.6 27B, zai-org/GLM-4.7-Flash 31B — pick whichever runs in vLLM at reasonable tps. |
| IFEval rule-based | — | verifiable, **RM-unhackable** truth anchor |

LLM judge prompt ✅: uses the **Arena-Hard-Auto v2.0** prompt/template (5-point graded verdict `[[A>>B]]`…`[[B>>A]]`, mapped to weighted battles exactly as upstream `show_result.py`) — the most justifiable / reproducible option of the candidates (Deepseek R1, Deepseek GRM, Kimi). Implemented in [policy_eval/judges.py](policy_eval/judges.py) as `LLMJudge` with a pluggable backend: `VLLMBackend` (open-weight, local) or `OpenAICompatibleBackend` (any hosted OpenAI-compatible API — the Vector Institute proxy, OpenRouter — selected by provider).

### SELECTION set (used only by the no-peek rule)
| Evaluator | Notes |
|---|---|
| Held-out RM | trained by the method from the SFT base using a different shuffle/random seed, scored on the `select` prompt set. Method-built, same-distribution, independent of the truth set. |

> Caveat: some public RMs' training mixes may include HelpSteer-like data
> (indirect correlation). The held-out RM and IFEval are therefore the cleanest
> independent anchors; public RMs are correlated-but-useful cross-checks.

---

## 6. Checkpoint selection — the no-peek rule ✅

The benchmark's defining constraint: **you may not use the test set or any TRUTH
evaluator to pick your checkpoint.**

SELECTION RM is trained from the same sft.

- **Canonical leaderboard rule:** select the checkpoint maximizing the
  **held-out SELECTION RM** score on the `select` prompt set.
- **Selection-strategy study (separate result):** report each alternative's
  **regret vs the gold-peeking oracle** — i.e. how much true reward you forfeit
  by not being able to cheat. Alternatives to study: best training-proxy reward,
  KL-threshold, last checkpoint, EMA/smoothed, train-RM-on-held-out.

---

## 7. Eval prompt sets ✅

| Tier | Source | Scored by |
|---|---|---|
| In-distribution | held-out HelpSteer3 (`test` split) | TRUTH RM panel + judge |
| OOD, judged | ArenaHard 2.0 | open-weight LLM judge **and** gold RM, win-rate vs baseline |
| OOD, **verifiable** | IFEval | rule-based strict/loose (the incorruptible anchor) |
| Capability retention 🔲 | small fixed general benchmark (e.g. GSM8K subset) | rule-based; catches alignment-tax / forgetting |

---

## 8. Frozen eval decoding config ✅

**Precommit: sample the policy at the training temperature.** Policy generations
(preference, select, ifeval, arena_hard) use **`temperature = --eval_temperature`
(default `1.0`), `top_p=1.0`, `n=1`, fixed `max_new_tokens`** — identical across all
methods, checkpoints, and benchmarks. The default matches the GRPO training decode
(the training run's `run_manifest.json`, written by `my_grpo.py` into the checkpoints
dir, supplies its `--temperature` as the `--eval_temperature` default; an explicit
CLI flag overrides), so **eval is in-distribution wrt training** — the project's
no-distribution-shift constraint. This rejects greedy (`temp=0`): greedy evaluates a single mode the RL
objective never directly optimizes, and it is blind to distributional hacking (great
on most samples, garbage on a tail). Reproducibility is guaranteed not by determinism
but by the per-example persistence layer ([§9](#9-per-example-logging-contract-a1-)),
which stores raw samples + scores so any aggregate is recomputable.

- **Single-sample for now** (`n=1`, enforced — `--num_responses_per_prompt != 1` is a
  hard error). Multi-sample (`n≥4`) with per-checkpoint sampling CIs is the natural
  extension when tighter CIs are needed; because `n` is an estimator-variance knob (same
  distribution at any `n`), the sweep and final ranking stay comparable. When `n>1` is
  re-opened, every path must aggregate over **all** samples — do not revert to the
  `responses[::n]` slice that [C2](#c2-n1-inconsistency) fixed.
- **The LLM judge stays greedy** (`--llm_judge_temperature=0`), independent of the policy
  temperature: it compares two fixed answers and needs no sampling diversity. Decoupling
  policy and judge decoding keeps judge-runs and no-judge-runs comparable.
- **The win-rate baseline reference stays greedy/deterministic** (baseline-model
  generation, `temperature=0`, disk-cached) — it is a *fixed* reference anchor, not the
  object under test, so it is deliberately not sampled.
- Thinking-mode handling stays per-benchmark as currently implemented; the
  thinking span is stripped before RM scoring.

---

## 9. Per-example logging contract (A1) ✅

**The single most important storage rule.** Because the headline metric is
deferred and checkpoint storage is finite, every eval **must persist per-example
raw numbers** so any aggregation can be recomputed forever without re-running.

Persist one durable artifact (parquet/jsonl, decoupled from checkpoint weights)
per `(benchmark, checkpoint)` with one row per `(prompt, response)`:

| Column | Purpose |
|---|---|
| `prompt_uid` | join key across evaluators/checkpoints |
| `response_text`, `response_token_len`, `finish_reason` | length/verbosity gate, truncation accounting |
| `score__<evaluator>` (one per TRUTH + SELECTION evaluator) | enables panel mean/min, per-prompt win-rate, sc_score |
| `chosen_or_baseline_score__<rm>` | reference for win-rate (already disk-cached) |
| `over_budget` (bool) | length-guard accounting (see [B2](#b2-eval-bypasses-length-guard)) |

From this you can compute, post-hoc and for free: panel mean, panel min,
win-rate vs **any** reference, style-controlled (`sc_score`), and length-controlled
variants.

**Implementation** ([policy_eval/persistence.py](policy_eval/persistence.py)):
`PerExampleRecorder` accumulates one row per `(prompt, response)`; the eval main
loop writes one file per `(benchmark, checkpoint)` to
`<output_file_stem>_per_example/<benchmark>__checkpoint-<n>.parquet` (override the
location with `--per_example_dir`, switch to jsonl with `--per_example_format
jsonl`). Persistence is always on — there is no disable flag. Each evaluator contributes its per-example
columns through `EvalContext.recorder`: RM evaluators write `score__rm_<label>`
and `chosen_or_baseline_score__<label>`; the pairwise judge writes the raw
signals that determine each battle (RM judge → policy + baseline scores; LLM
judge → both per-game labels) plus `battle_mean__<judge>__<slot>`; KL writes
`kl__k1`/`kl__grpo` + policy/base mean-logprobs; IFEval writes per-prompt
strict/loose flags. `over_budget = finish_reason=='length' or
response_token_len > --response_token_budget` (default 1024). A `_manifest.json`
records dataset/split/RM identities/budgets so the scores' provenance survives
checkpoint deletion.

---

## 10. Metric, validity gates, and variance

### 10.1 Headline metric 🔲 (aggregation deferred — numbers stored now)
- **Form:** `TRUTH_aggregate @ (checkpoint chosen by the no-peek rule)`, with
  IFEval reported at the same checkpoint (mark with color/underline/asterisk in the main leaderbord if ifeval or arenahard are significantly worse)
- **Aggregation across the TRUTH panel — decision deferred** (candidates: panel
  **mean**, panel **min** win-rate, `sc_score`). [§9](#9-per-example-logging-contract-a1-) guarantees
  every candidate is computable later, so we choose after seeing data.
- **Supporting plots:** gold-vs-KL frontier (compare methods at **matched
  KL-to-SFT**, sequence-level — the only apples-to-apples axis across
  DPO/PPO/GRPO); selection regret vs oracle.
- **Comparison discipline:** compute- and KL-matched, **not** "best ever".

### 10.2 Validity gates ✅
A leaderboard entry is flagged/invalid if:
- **Mean response length drifts far from the SFT policy** (length is the hacking
  axis; report length next to every metric, prefer length-controlled win-rate).
- **TRUTH panel members disagree with the primary RM by > noise floor** (the
  "gold is being hacked" detector — surfaces when proxy gains don't transfer).
- Over-budget generation count is non-trivial.

### 10.3 Variance protocol ✅
- **Sweep single-seed**, then **multi-seed only the winners + baselines**.
- **One 3-seed baseline triplet up front** to establish the noise floor — this is
  the ruler for reading the single-seed sweep (don't pick winners on sub-noise
  gaps). Carry **top-k, not top-1**, into the seeded stage.

---

## 11. Hyperparameter sweep protocol ✅

Full grids are infeasible. Use **pre-registered sequential coordinate descent**
(fits "launch a stage, read it, launch the next"). Order = upstream → stability →
regularization → minor:

| Stage | Axis | Coarse grid | Why this order |
|---|---|---|---|
| A | RM epochs | {1,2,4,8,16} | upstream; pick by RM-eval-acc **and** one fixed-config GRPO probe |
| B | LR | {1e-6, 4e-6, 8e-6, 1e-5, 2e-5} | most likely to diverge; pin stable-high first, sweep neighbours per method |
| C | β / KL | log range incl. 0 | the axis the benchmark is *about*; **re-swept per method** |
| D | group size (GRPO) | {4,8,16} | lowest impact; confirm last <has big impact on training time> |

- **DPO** gets its own order: β → LR → loss_type.
- **Justification in writing:** report the pre-registered order + rationale; show
  one-axis sensitivity curves. **If the best value is the smallest or largest one
  you tried, the grid didn't bracket the optimum — extend it in that direction and
  re-run** (e.g. if LR=2e-5 wins, also try 3e-5/4e-5; stop only once the winner has
  *worse* neighbours on both sides). Otherwise the reported "best" is just an
  artifact of where you stopped looking.

---

## 12. Baselines & priority ✅

| Pri | Baseline | Notes |
|---|---|---|
| P0 | Protocol freeze + noise floor | this doc + the 3-seed triplet |
| P1 | **Single-RM GRPO, fully tuned** | the honest number everything beats; RM-epochs is the spine |
| P2 | **DPO, fully tuned** | β × loss_type × LR; same gold-vs-KL frontier |
| P3 | Ensembles (mean / sequential-switch / min / UWO) + **WARM** | the contribution + the cheap literature baseline, now judged vs a measured noise floor |
| P4 | **PAR** | cheap to run end-to-end — include if maintained |
| P4 | RLOO (cheap GRPO check); PPO if compute | RLOO validates GRPO isn't special |
| P5 / v1.5 | Lit reimplementations (ODIN, EPPO, Adv-RM), safety arm, multi-objective | ODIN especially relevant given length findings |

Out of leaderboard scope (kept as side-studies): online-PET, RRM, CQL/pessimistic-loss.

---

## 13. Implementation roadmap

Ordered. The eval-correctness items (**A1, B1**, B2/C1, B3) and the Gemma-readiness
check all gate the sweeps — the code is frozen only after they pass.

1. ✅ Build frozen four-way splits + within-dataset contamination check
   ([§3](#3-datasets--splits)) — done via `split_four_way` + `assert_splits_disjoint` in
   scripts/dataset_pipeline. Filter once with the canonical Qwen3.5 tokenizer
   (family-agnostic). Remaining 🔲: external contamination cross-check vs ArenaHard/IFEval.
2. ✅ **A1 — Per-example persistence** ([§9](#9-per-example-logging-contract-a1-)). Foundation; do before any
   further benchmark runs or they're wasted. Done via `PerExampleRecorder`
   ([policy_eval/persistence.py](policy_eval/persistence.py)), wired into every
   evaluator through `EvalContext.recorder`. The deferred vLLM judge persists
   per-prompt verdicts to its own per-`(benchmark, checkpoint)` file (both
   swapped-game raw texts + labels + battle outcome). Remaining 🔲: verify on a
   `--debug` run before the first sweep.
3. ✅ **B1 — Open-weight vLLM judge** ([§5](#5-evaluators)). Done: a single
   `LLMJudge` (Arena-Hard 2-game-swap + parse) with a pluggable backend —
   `VLLMBackend` (loads the judge vLLM once and shares it across the preference
   + arena_hard benchmarks) or `OpenAICompatibleBackend`, one implementation for
   every hosted OpenAI-compatible API with the per-provider endpoint, key env
   var, reasoning dialect and Batch-API support in `OPENAI_PROVIDERS`
   (`vector` = the Vector Institute proxy, `--vector_judge` in
   `evaluate_policy.sh`; `openrouter`). All backends are deferred, so
   `--judge_selected_checkpoint_only` and `--load_generations` apply to each.
   **Hosted judges run as a separate GPU-free job** (`judge_cached.sh`, chained
   `--dependency=singleton,afterok:$GPU`): the judge costs ~2.8k games ≈ 28 min
   per checkpoint, the proxy's RPM budget is shared project-wide while the
   client-side pacing is per-process (so concurrent evals overrun it), and a
   proxy outage then costs a retry rather than the generation phase. It judges
   the **selected + final** checkpoints (`--judge_final_checkpoint`), not all
   ~20 — the pair is what separates "the sibling RM picked well" from "the gold
   RM was itself overoptimized". Its metrics land on the generating run: the
   per-example `_manifest.json` records that run's wandb id.
   Selected by `--llm_judge_backend`. Generation params unified via `--llm_judge_*`;
   per-prompt verdicts persisted (A1); generation/truncation/parse failures
   counted to wandb. Judge model still to finalize (proposal `google/gemma-4-31B-it`).
4. **B2 + C1 — Length guard + length logging.** Count/flag over-budget samples;
   log `response_token_len` + `finish_reason` on RM paths (folds into A1).
5. ✅ **B3 — Decoding config frozen.** Judge-coupled temperature flip removed;
   policy is sampled single-sample at the training temperature (`--eval_temperature`,
   default 1.0), judge stays greedy ([§8](#8-frozen-eval-decoding-config-)).
6. Capability-retention benchmark ([§7](#7-eval-prompt-sets-)).
7. **Cross-family (Gemma) readiness → code freeze.** Before any sweeps, verify the
   *full* pipeline (SFT → RM train → GRPO/DPO → eval) runs end-to-end on a **Gemma**
   base, not just Qwen3.5. Known family-specific risks to clear:
   - **vLLM weight loader** (`vllm_weight_loader.WeightLoaderExtension` /
     `load_weights_from_path`) assumes the Qwen3.5 SFT weight layout
     (`language_model.*`) — confirm it loads Gemma checkpoints.
   - **`qwen35_vllm_patch`** is imported unconditionally
     ([generation.py:26](policy_eval/generation.py#L26)) and patches Qwen3.5
     transformers internals — guard it so it's a no-op (and can't error) for Gemma.
   - **Thinking config** must be family-aware (Gemma has no thinking mode → set
     `thinking=False`;
   - **Chat-template round-trip:** the empty-assistant-suffix inference in
     `_format_conversation` must hold for Gemma's `<start_of_turn>`/`<end_of_turn>`
     template (it asserts prompt is a prefix of the empty-assistant render).
   - **vLLM `max_model_len` + length guard for Gemma** — Gemma tokenization may expand
     vs Qwen; size `max_model_len` with headroom (or compute it via
     `compute_max_prompt_length` under the Gemma tokenizer), and confirm the fail-fast
     guard carries headroom so it doesn't crash on benign cross-tokenizer drift ([§3](#3-datasets--splits)).
   - **RM + SFT from a Gemma base** (`AutoModelForSequenceClassification` head,
     pad/BOS, embedding resize) train without family-specific breakage.
   - Already OK: stop tokens include `<end_of_turn>`
     ([data_utils.py:295](data_utils.py#L295)); pad fallback; BOS strip.
   **Freeze the code once this passes** — no code changes once sweeps start.

---

## Appendix — current implementation gaps vs this spec

From the eval audit of `policy_eval/` (severity tagged):

- ✅ **A1** Per-example scores now persisted — `PerExampleRecorder` ([policy_eval/persistence.py](policy_eval/persistence.py)) writes one parquet/jsonl per `(benchmark, checkpoint)` (one row per `(prompt, response)`) with responses, token lengths, finish reasons, every RM/judge score, KL, and `over_budget`. Aggregation can be recomputed offline. (Was: [evaluators.py](policy_eval/evaluators.py) logged only mean/std/lossy-histogram + aggregate win-rate.)
- ✅ **B1** Open-weight judge implemented — a single `LLMJudge` ([policy_eval/judges.py](policy_eval/judges.py)) runs the Arena-Hard 2-game-swap + parse protocol over a pluggable backend: `VLLMBackend` (in-process vLLM, deferred, loaded once and shared across preference + arena_hard) or `OpenAICompatibleBackend` (any hosted OpenAI-compatible API; see `OPENAI_PROVIDERS`). Verdicts persisted per-prompt; generation/truncation/parse failures counted to wandb. The old prompt-corruption API scaffold was removed.
- 🟠 **B2** Eval bypasses length guard — `skip_validation=True` everywhere ([rewards.py:55-62](policy_eval/rewards.py#L55-L62), [benchmarks.py:86-94](policy_eval/benchmarks.py#L86-L94)); BT scoring doesn't truncate → over-budget fed past RM context, biasing the longest (most-hacked) responses. -- OK, we ususally have the same tokenizer and vllm gen is capped
- ✅ **B3** Decoding config frozen — judge-coupled temperature flip removed. Policy decoding is `temperature=--eval_temperature` (default 1.0 = training temp, [§8](#8-frozen-eval-decoding-config-)), `top_p=1.0`, `n=1`, across every policy-generation site (preference/select/ifeval/arena_hard, [benchmarks.py](policy_eval/benchmarks.py)); `grpo.sh` passes its training `--temperature` through. The LLM judge stays greedy (`--llm_judge_temperature=0`) and the win-rate baseline stays greedy/cached, both independent of the policy temperature — so judge-runs and no-judge-runs remain comparable.
- 🟡 **B4** Win-rate reference is dataset `chosen`, not SFT ([evaluators.py:96-106](policy_eval/evaluators.py#L96-L106)) — fine under human-primary but must be declared; becomes a reporting choice once A1 lands. -- Expected, OK
- ✅ **C1** Response length/finish_reason now logged on all paths — `response_token_len` + `finish_reason` + `over_budget` are per-example columns (A1). (B2's actual length *guard* on the RM scoring path is still open.)
- ✅ **C2** `n>1` inconsistency resolved — the frozen eval is single-sample (enforced at benchmark build), and the pairwise + RM-win-rate paths now assert `n==1` instead of silently judging only `responses[::n]` (the first sample), which previously diverged from the RM `/mean` that averaged all samples.
- ✅ **C3** Bootstrap rounds bumped to `n_bootstrap=1000` (was 100) for the arena/pairwise CIs ([pairwise.py](policy_eval/pairwise.py)), applied to both the arena-score battle-level bootstrap and the style-controlled BT bootstrap.
- ✅ Legacy `rm_eval/` (truncation + no-op length filter, formatting inconsistent with the pipeline) — deleted, along with its driver `scripts/eval_bt_rm.sh`.

Confirmed **correct** (no action): BT RM scoring tokenization (BOS-strip +
`add_special_tokens=True` + left-pad, no truncation, via `tokenize_for_rm`); KL
evaluator (k1 + k3 estimators vs configurable SFT base); Arena-Hard API judge
upstream-compatibility.


Proposed LLM judge: google/gemma-4-31B-it.