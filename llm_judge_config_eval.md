# LLM-as-judge config evaluation (gemma-4-31B-it)

Which judge config to use for policy eval: **thinking on/off** x **8-bit on/off**,
scored on **judgment quality (incl. low unparsed rate)** and **speed**.

**TL;DR — use `bf16, no-thinking`.** It is the fastest, drops 0% of prompts, and
has the highest agreement with the gold RM and the highest self-consistency.
Thinking and fp8 both make things worse or no better here. Use fp8 no-think
**only** if VRAM-constrained (<~48 GB); it barely changes verdicts but is slower
on Ampere.

---

## Setup

- Judge: `google/gemma-4-31B-it` via the production judge
  (`policy_eval.judges.LLMJudge` + `VLLMBackend`), Arena-Hard-Auto protocol
  (2 position-swapped games/prompt, weighted 5-point verdict).
- Data: **reused** the cached policy generations from wandb run `pzhrazoc`
  (`slurm-1161408.out`) — preference benchmark, policy response vs dataset
  `chosen` baseline, with Skywork gold-RM scores already stored per example.
  No regeneration; the judge just re-scores cached answers.
- Sweep: 5 checkpoints (149→2975) x 100 prompts for no-think; x 20 prompts for
  thinking (bounded — thinking is ~5x slower). Greedy (temp 0).
- Hardware: 1x A100 80 GB (Ampere — **no native fp8**).

## Results (5-ckpt x 100/20-prompt sweep)

| config | pairs/s | dropped | gold agreement | self-consistency | rank ρ vs gold |
|---|---|---|---|---|---|
| **bf16 no-think** | **0.80** | **0 %** | **0.692** | **0.906** | 0.8 |
| fp8 no-think | 0.63 | 0 % | 0.684 | 0.906 | 0.8 |
| fp8 think (concise) | 0.28 | 7 % | 0.658 | 0.871 | 1.0 |
| bf16 think (concise) | 0.15 | 5 % | 0.628 | 0.895 | 1.0 |

- *gold agreement* = fraction of decided pairs where the judge's direction matches
  the gold RM's (policy vs chosen), on the common 91-pair set all configs decided.
- *self-consistency* = 1 − position-flip rate (does the judge pick the same actual
  answer when A/B are swapped).
- Cross-config verdict agreement: **fp8 vs bf16 (no-think) = 0.976**;
  think vs no-think (same quant) ≈ 0.86.

## Findings

1. **Thinking loses on every axis that matters here.** Gemma's `<|channel>thought`
   is very verbose: with the canonical Arena-Hard CoT prompt at a 3072-token
   budget, **58 %** of prompts never reach a verdict and are dropped. A concise
   thinking prompt (added in the sweep harness) cuts that to 5–7 %, but thinking
   is still **3–5× slower**, and its gold agreement and self-consistency are
   **lower** than no-think. Extra reasoning does not help preference judging.
2. **8-bit (fp8) is quality-neutral but not worth it here.** fp8 and bf16 no-think
   agree on **97.6 %** of verdicts with ~equal gold agreement — quantization is
   safe. But on this A100 fp8 is **slower** (Ampere has no native fp8 → vLLM uses
   the Marlin weight-only kernel, which penalizes the prefill-heavy judge), and
   the 31B fits in bf16 on 80 GB anyway. fp8 only pays off on a smaller GPU.
   > Note: the linked `unsloth/gemma-4-31b-it-MLX-8bit` is an **Apple-MLX** build
   > and cannot load on vLLM/CUDA. fp8 (in-flight, weight-only) is the runnable
   > 8-bit stand-in; `--quants fp8` in the harness.
3. **The judge is prefill-bound**, not decode-bound: no-think emits ~4 tokens but
   each prompt is a ~3k-token judge template (system + both answers). Speed is
   dominated by prefill throughput (bf16 ~4100 tok/s vs fp8 ~2800 tok/s).
4. **Ranking**: no-think tracked the gold checkpoint ranking at ρ=0.8 on 5
   checkpoints (thinking 1.0, but on a noisy 20-prompt subset).

## Scaled head-to-head (20 checkpoints x 40 prompts = 800 pairs, no-think only)

Run with `compare_top2_judge_configs.sh`; confirms the pick at scale.

| config | pairs/s | dropped | gold agreement | self-consistency | rank ρ vs gold |
|---|---|---|---|---|---|
| **bf16 no-think** | **0.93** | 0 % | 0.653 | 0.930 | 0.66 |
| fp8 no-think | 0.77 | 0 % | 0.654 | 0.924 | 0.71 |

- **fp8 vs bf16 agree on 98.5 % of 800 verdicts** — quantization is verdict-neutral
  at scale; bf16 stays ~20 % faster. Confirms: prefer bf16, fp8 only for VRAM.
- **Caveat about the judge itself (config-independent):** the gemma judge tracks
  the gold-RM *checkpoint ranking* only moderately (ρ≈0.66–0.71). Its win-rates
  are compressed (~0.31–0.49) and, unlike the gold RM (0.36→0.68 rising over
  training), do not show the training-progress trend — the judge compares the
  policy against the strong dataset `chosen` baseline and sees them as close.
  Treat the LLM judge as a coarse sanity check, not a substitute for the gold RM
  in checkpoint selection.

## Token cap: does no-think degrade vs a properly-run thinking judge?

The earlier thinking runs were truncation-limited, which was unfair to thinking.
Re-ran with the cap raised so truncation → 0 (fp8, quant held constant, same
paired set = 3 ckpts x 40 prompts = 120 pairs):

| config | pairs/s | dropped | gold agree | self-consist | agree w/ no-think |
|---|---|---|---|---|---|
| no-think | 0.34 | 0 % | 0.629 | **0.950** | — |
| think concise @4096 | 0.25 | 0.8 % | **0.692** | 0.941 | 0.866 |
| think full-CoT (Arena) @8192 | 0.11 | **0 %** | 0.631 | 0.925 | 0.850 |

- **At an 8192-token cap the full-CoT judge no longer truncates (0 dropped)** — the
  drop problem was purely the budget. But it costs **3× no-think** (0.11 vs 0.34
  pairs/s) and needs `max_model_len≈11.8k` (→ fp8 for KV room; gpu_mem≤0.85 or it
  OOMs on the prefill activation transient).
- **no-think agrees with thinking on ~85–87 % of verdicts.** So the "degradation"
  from dropping thinking is ~13–15 % of verdicts flipped.
- **That 15 % is not a clear loss.** Full-CoT's gold agreement (0.631) ≈ no-think's
  (0.629), and no-think is the **most self-consistent** (0.950). On the handful of
  decisive disagreements gold split ~coin-flip for full-CoT (5/9) — noise.
- **Weak signal for *light* thinking:** concise@4096 had the highest gold agreement
  (0.692) and won all 6 decisive disagreements vs no-think — but n=6, well within
  noise on 120 pairs. If judge quality ever needs a push, concise thinking @4096
  (not full CoT) is the only variant worth revisiting; full CoT buys nothing here.

Bottom line: raising the cap fixes thinking's drops but does **not** make thinking
measurably better than no-think, at 2.5–3× the cost. **no-think still wins.**

## Code changes (kept — general improvements)

**Stop-token fix** — `data_utils.get_generation_stop_token_ids` now unions the
model's `generation_config.eos_token_id` into the stop set. Gemma-4 ends turns
with `<turn|>` (id 106), which the helper's string-keyed lookup misses; without
this, a *thinking* judge never stops on the turn-end token and burns the full
`max_tokens` on every game. Put in the shared helper (not just the judge) so
every generation site gets it — policy and baseline generation in `policy_eval`,
GRPO rollouts, and `policy.generation_config.eos_token_id`. Would otherwise bite
any gemma-4 *policy* the same way. Read is cached per path, and a model dir with
no `generation_config.json` (reward models, LoRA adapters) falls back to the old
behavior. Verified no change for Qwen3, whose declared eos list is already
covered by the string lookup.

`policy_eval/judges.py`:
- `VLLMBackend(..., quantization=...)` — forwards any vLLM quant string (e.g.
  `"fp8"`) and labels metrics with it.
- `VLLMBackend` resolves the stop set once at load (logging it) instead of per
  `generate` call, and passes its own model name so the config is read from the
  judge rather than inferred from the tokenizer's path.

These do not change the default no-think production path's verdicts (`truncated`
only attributes an already-unparsable game, so the fix moves failure accounting,
not decisions).

## Reproduce / compare at scale

```bash
# 4-way sweep (think/no-think x bf16/fp8) on cached generations:
python scratch_judge_sweep/run_judge_sweep.py \
  --per_example_dir <eval>_per_example \
  --checkpoints 149,745,1341,2086,2975 --n_prompts 100 --think_n_prompts 20 \
  --quants fp8,none --thinks false,true --tag sweep1
python scratch_judge_sweep/analyze_judge_sweep.py --tag sweep1

# Head-to-head of the 2 promising configs (bf16 vs fp8, no-think) at scale:
PE_DIR=<eval>_per_example N_PROMPTS=150 bash scratch_judge_sweep/compare_top2_judge_configs.sh
```
