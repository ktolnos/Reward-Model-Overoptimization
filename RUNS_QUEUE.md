# Runs Queue

Working queue of things to run. Ordered top-to-bottom. The point of the current
phase is **pipeline validation + decisions that gate future runs** (see
[BENCHMARK.md §13](BENCHMARK.md)) — not the sweeps themselves. We want to be
reasonably confident the pipeline won't break across all future experiments, and
to lock in the choices (judge model, thinking on/off, quantization, Gemma
readiness) that every later run inherits.

## Format

The queue is a sequence of **Waves**. A wave separates work by *decisions*, not by
time:

- **`## Wave N`** — every item inside a wave is **independent** and can be launched
  **in parallel** with no decision in between. Fire them all, then wait.
- **`### 🚦 GATE`** — a decision point that closes a wave. You must read the wave's
  results and make the listed decision(s) before starting the next wave, because
  the next wave's runs depend on the outcome. Record the decision inline (edit the
  gate, check its box) so the choice is auditable.
- Items that must run **sequentially inside a wave** (rare) are numbered `N.1 → N.2`
  and say `after N.1` explicitly. Everything else in a wave is concurrent.

Item anatomy:

```
- [ ] **ID** — one-line purpose
  - Run: <script / command, or code change needed first>
  - Check: <what output to look at to call it pass/fail>
  - Feeds: <which GATE this informs> (omit for pure validation)
```

Status: `[ ]` todo · `[~]` running · `[x]` done · `[!]` blocked/failed (note why).

---

## Wave A — pipeline smoke tests (Qwen3.5, current family)

Confirms the frozen eval plumbing actually works before we spend compute on
sweeps or add a second model family. All independent.

- [ ] **A-persist** — verify A1 per-example persistence on a real checkpoint
  - Run: `sbatch evaluate_policy.sh --debug` against an existing Qwen3.5 GRPO run
  - Check: `<output_stem>_per_example/<benchmark>__checkpoint-<n>.parquet` exists for
    every benchmark; has one row per `(prompt, response)`; `score__*`, `kl__*`,
    `response_token_len`, `finish_reason`, `over_budget` columns populated; a
    `_manifest.json` records dataset/split/RM identities. (Closes BENCHMARK §13.2
    remaining item.)
- [ ] **A-splits** — external contamination cross-check (§3, §13.1 remaining)
  - Run: script/notebook asserting `train/select/validation/test` prompts are disjoint
    from ArenaHard 2.0 and IFEval prompt sets
  - Check: zero overlap; if any, decide drop-vs-keep before it pollutes truth eval
- [ ] **A-eval-splits** — confirm `--split` selection works end-to-end
  - Run: eval with `--split validation` and `--split test`, tiny debug
  - Check: correct split loaded, raises cleanly if a split is absent

### 🚦 GATE A — eval plumbing trusted?
- [ ] Persistence produces recomputable per-example artifacts → OK to run real evals.
- [ ] Any contamination found is resolved/declared.
- Decision: _pending_ — record outcome here.

---

## Wave B — LLM judge configuration (the current focus)

All runs use the **same fixed set of policy responses** (one existing checkpoint's
`preference` + `arena_hard` generations) so the judge is the only thing varying —
this is what makes the comparisons apples-to-apples. Independent; launch together.

Target model under test: `google/gemma-4-31B-it` (current `LLM_JUDGE_MODEL` in
[evaluate_policy.sh](evaluate_policy.sh#L71)), vLLM backend.

- [ ] **B-think-on** — judge with thinking enabled (current default)
  - Run: `evaluate_policy.sh --with_llm_judge` with `--llm_judge_backend vllm`,
    `--llm_judge_model_name google/gemma-4-31B-it`, `--llm_judge_enable_thinking True`
  - Check: parse-failure rate, tps / wall-clock, per-prompt verdicts persisted
  - Feeds: GATE B (thinking decision)
- [ ] **B-think-off** — judge with thinking disabled
  - Run: same as B-think-on but `--llm_judge_enable_thinking False` (vLLM prefills
    `"My final verdict "` — see [judges.py](policy_eval/judges.py) VLLMBackend)
  - Check: how far do win-rates / battle outcomes move vs B-think-on? parse-failure
    rate; **speedup** (this is the payoff — thinking is the expensive part)
  - Feeds: GATE B — *"how much does disabling thinking change gemma-31B judging, and
    is the speedup worth any agreement loss?"*
- [ ] **B-8bit** — 8-bit quantized judge feasibility
  - Prereq (code): [judges.py](policy_eval/judges.py) `VLLMBackend` hardcodes
    `dtype="bfloat16"` with no quantization knob. Either (a) point
    `--llm_judge_model_name` at a pre-quantized HF checkpoint, or (b) add a
    `--llm_judge_quantization` flag threaded into the vLLM `LLM(...)` ctor. Pick one
    before running.
  - Run: judge eval with the 8-bit variant, otherwise identical to B-think-on
  - Check: **does it load & fit** at current `--llm_judge_gpu_memory_utilization`;
    tps vs bf16; verdict agreement vs bf16 (are the win-rates within noise?)
  - Feeds: GATE B — *"can we use an 8-bit judge to save memory/time without moving
    the numbers?"*

> Note: judge stays greedy (`--llm_judge_temperature 0`) in all three — decoding is
> frozen (§8). Only thinking / precision / model vary.

### 🚦 GATE B — freeze the judge config
- [ ] **Model:** confirm `google/gemma-4-31B-it` runs in vLLM at acceptable tps, or
      switch to a candidate (DeepSeek V4 Flash / Qwen3.6 27B / GLM-4.7-Flash 31B — §5).
- [ ] **Thinking:** on or off (from B-think-on vs B-think-off).
- [ ] **Quantization:** bf16 or 8-bit (from B-8bit).
- Decision: _pending_. Once decided, set the defaults in
  [evaluate_policy.sh](evaluate_policy.sh) / [evaluate_policy.py](evaluate_policy.py)
  and treat as frozen for all sweep evals.

---

## Wave C — Gemma cross-family readiness (§13.7)

Gates the code freeze: the *full* pipeline (SFT → RM → GRPO → eval) must run
end-to-end on a Gemma base, not just Qwen3.5. The code-level checks below are
**independent** and can be verified in parallel (mostly reading/small probes); the
**end-to-end micro-run (C-e2e)** should come *after* they pass so it isn't just
re-discovering a known breakage.

- [ ] **C-vllm-loader** — vLLM weight loader accepts Gemma layout
  - Check: `vllm_weight_loader.load_weights_from_path` handles Gemma checkpoints
    (currently assumes Qwen3.5 `language_model.*` layout)
- [ ] **C-patch-guard** — `qwen35_vllm_patch` is a no-op for non-Qwen3.5
  - Check: it's imported unconditionally at
    [generation.py:26](policy_eval/generation.py#L26); guard so it can't error on Gemma
- [ ] **C-chat-template** — empty-assistant-suffix round-trip holds for Gemma
  - Check: `_format_conversation` prefix assertion passes for Gemma
    `<start_of_turn>`/`<end_of_turn>`
- [ ] **C-thinking** — thinking config is family-aware (Gemma has no thinking mode)
  - Check: `thinking=False` forced for Gemma paths; no crash
- [ ] **C-maxlen** — `max_model_len` + length guard carry headroom under Gemma tokenizer
  - Check: size via `compute_max_prompt_length` under Gemma tokenizer (or generous
    fixed ceiling); fail-fast guard doesn't trip on benign cross-tokenizer expansion (§3)
- [ ] **C-rm-sft** — RM + SFT train from a Gemma base without family breakage
  - Check: `AutoModelForSequenceClassification` head, pad/BOS, embedding resize all OK

- [ ] **C.e2e** — *(after the checks above)* Gemma end-to-end micro-run
  - Run: tiny SFT → tiny BT RM → short GRPO → `evaluate_policy.sh --debug`, Gemma (E2B) base
  - Check: every stage completes; eval writes per-example artifacts; no
    Qwen-specific assertion fires anywhere

### 🚦 GATE C — code freeze
- [ ] Full pipeline runs on Gemma end-to-end (C.e2e green).
- [ ] All family-specific risks from §13.7 cleared.
- Decision: _pending_. **On pass, freeze the code** — no code changes once sweeps start.

---

## Wave D — noise floor (unblocked once GATE A + GATE B + GATE C pass)

The ruler for reading every single-seed sweep result (§10.3). Not startable until
the eval config and code are frozen, so it lives past all three gates.

- [ ] **D-noise** — one 3-seed baseline triplet (single-RM GRPO, fixed config)
  - Run: `scripts/submit_seeds.sh` (or `grpo.sh` ×3 seeds) → eval each
  - Check: seed-to-seed spread on the TRUTH panel = the noise floor; carry into the
    sweep stage so we don't pick winners on sub-noise gaps

---

## Backlog (not yet scheduled)

- Capability-retention benchmark (GSM8K subset) — §7, §13.6.
- B2 length guard on the RM scoring path (currently accepted-as-OK, §appendix B2).
- Sweep stages A–D per BENCHMARK §11 (start only after Wave D noise floor).
