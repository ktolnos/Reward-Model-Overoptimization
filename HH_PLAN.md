# Anthropic-HH Comparison Plan

## Goal

Compare our RM ensemble + GRPO pipeline against published PPO baselines from papers that use Anthropic-HH. The primary comparison targets are:

| Paper | Their key HH result | Metric |
|---|---|---|
| EPPO (#3) | vs PPO w/KL on Anth-Helpful: `51/29/20` W/T/L | GPT-4 pairwise |
| InfoRM (#12) | vs Standard RM on Anth-Helpful: `54.5/33.5/12.0` | GPT-4 pairwise |
| AdvPO (#14) | vs PPO on Anth-HH: `31.0/49.0/20.0` | GPT-4 pairwise + human |

Secondary (different eval format, harder to compare directly):
| Paper | Their key HH result | Metric |
|---|---|---|
| ARA (#7) | Sycophancy `38.4`, Helpfulness `77.2` | GPT-4 + SycophancyEval |
| CausalRM (#8) | vs Standard RM ID `54.8/33.1/12.1` | Qwen3-Max pairwise |

---

**What we should take from the reference repo:**
1. The exact prompt template format (Vicuna-style: `USER: {instruction} ASSISTANT: {output}`)
2. The data recipe (Anthropic-HH helpful 35k + harmless 37k ≈ 72k prompts for RL)
3. Hyperparameters: max_seq_len 1024, max_answer 512
4. The SFT starting point: `mycccc/Energy-Loss-Phenomenon-Demo` (subfolder `sft_model/`)

---

## Reference Repo Analysis: Energy-Loss-Phenomenon (EPPO)

**Framework**: DeepSpeed-Chat (Microsoft, custom PPO trainer)
**Template**: Vicuna format — `{sys_prompt} USER: {instruction} ASSISTANT: {output}`
**Data**: Anthropic-HH helpful (35,080) + harmless (37,208) = 72,288 prompts for RL (step 3)
**SFT model**: `mycccc/Energy-Loss-Phenomenon-Demo` subfolder `sft_model/` — Llama2-7B SFT'd on ShareGPT
**RM model**: Llama2-7B trained on Anthropic-HH preferences (not published, must train our own)
**RLHF model**: `mycccc/Energy-Loss-Phenomenon-Demo` subfolder `rlhf_model/` — their final PPO-trained policy
**Key hyperparams**:
- Actor LR: 5e-7, Critic LR: 1e-6
- Batch size: 8 per device
- KL control weight: 0 (no KL penalty by default)
- Max sequence length: 1024, Max answer length: 512
- PPO epochs: 1 per batch, 5 total epochs over data
- Cosine LR scheduler
- Reward: final token only (`--only_reward_final_token`)

**What's published**:
- EPPO algorithm code (PPO trainer modifications)
- **SFT checkpoint** (`mycccc/Energy-Loss-Phenomenon-Demo/sft_model/`) — the exact starting point for their RL runs
- **RLHF checkpoint** (`mycccc/Energy-Loss-Phenomenon-Demo/rlhf_model/`) — their final trained policy (useful as comparison target)
- No RM checkpoint, no preprocessed data

---

## Available Pre-trained Models

### SFT Models (no training needed)
| Model | Source | Notes |
|---|---|---|
| **`mycccc/Energy-Loss-Phenomenon-Demo` (sft_model/)** | HuggingFace | **The exact EPPO SFT checkpoint.** Llama2-7B SFT'd on ShareGPT. 27GB total repo. |
| `lmsys/vicuna-7b-v1.5` | HuggingFace | Llama2-7B SFT'd on ShareGPT. Used by InfoRM (#12) as SFT base. |
| `lmsys/vicuna-13b-v1.5` | HuggingFace | Larger variant, used as gold RM base by AdvPO (#14). |
| `meta-llama/Llama-2-7b-chat-hf` | HuggingFace (gated) | Official Llama2 chat model. Different SFT data than papers. |

**Recommendation**: Use `mycccc/Energy-Loss-Phenomenon-Demo/sft_model/` — this is the exact checkpoint the EPPO paper used, giving us the most direct comparison. Their published `rlhf_model/` can serve as a reference point (we can score it with our gold RM to get a target to beat/match).

### Reward Models (for RL training — proxy RMs)
| Model | Source | Notes |
|---|---|---|
| `weqweasdas/hh_rlhf_rm_open_llama_3b` | HuggingFace | Open-Llama-3B trained on HH-RLHF helpful. Different architecture. |
| `OpenRLHF/Llama-3-8b-rm-700k` | HuggingFace | Llama3-8B RM on 700k mixed data (includes HH-RLHF). |

**Problem**: No published Llama2-7B RM trained specifically on Anthropic-HH matches what the papers use. The papers all train their own RMs from the same SFT backbone. We will need to train our own.

### Gold RMs for Evaluation
Evaluating with RMs is cheap (single forward pass, no generation), so we should score all checkpoints with multiple gold RMs for robustness:

| Role | Model | Notes |
|---|---|---|
| **Gold RM (primary)** | `Skywork/Skywork-Reward-V2-Llama-3.1-8B` | Strong general-purpose 8B RM. Top of RewardBench. |
| **Gold RM (secondary)** | `Ray2333/GRM-Gemma2-2B-rewardmodel-ft` | Smaller, different architecture. Useful as cross-check. |
| **Gold RM (HH-specific)** | `weqweasdas/hh_rlhf_rm_open_llama_3b` | Trained on HH-RLHF directly. In-distribution gold signal. |
| **Gold RM (diverse)** | `OpenRLHF/Llama-3-8b-rm-700k` | Llama3-8B trained on 700k mixed preferences. |

Using multiple gold RMs lets us check whether overoptimization curves are consistent across judges or if some RMs are easier to game than others — this is itself an interesting finding.

The papers use GPT-4 pairwise judging for final numbers. We should do that too for the paper comparison, but the gold RM score vs KL curves are our primary development tool and are free to compute.

---

## Incremental Testing Plan

### Phase 0: Evaluation Smoke Test (no training)
**Goal**: Verify `evaluate_policy.py` works end-to-end with the HH setup before we train anything.

**Steps**:
1. Convert Anthropic-HH val split to our dataset format (`chosen`/`rejected` messages)
   - HH data is already in dialogue format: `\n\nHuman: ... \n\nAssistant: ...`
   - Parse into `[{role: "user", content: ...}, {role: "assistant", content: ...}]`
   - Push to HF as `{user}/anthropic-hh-val-messages`
2. Run `evaluate_policy.py` with:
   - `checkpoints_dir = mycccc/Energy-Loss-Phenomenon-Demo` (sft_model subfolder)
   - `gold_rm_name = Skywork/Skywork-Reward-V2-Llama-3.1-8B` (primary gold RM)
   - `dataset_name = {user}/anthropic-hh-val-messages`
   - `length_config = hh_paper` (new config: 1024 prompt / 512 response / 1536 total)
   - `skip_validation = True` initially
3. **Also score the published RLHF model**: run `evaluate_policy.py` with `mycccc/Energy-Loss-Phenomenon-Demo` (rlhf_model subfolder). This gives us a gold RM score target — the EPPO authors' best PPO result. Our GRPO should aim to match or exceed this.
4. **Score both models with all gold RMs** (Skywork-8B, GRM-Gemma2-2B, hh_rlhf_rm_open_llama_3b, OpenRLHF-Llama3-8B-rm). This is cheap and establishes baselines across all judges.
5. **Verify**: generations look reasonable, gold RM scores are finite, KL is ~0 for SFT model (since policy = SFT base), KL > 0 for RLHF model.

**Validation target**: The SFT model should produce coherent helpful/harmless responses. Gold RM scores should have reasonable variance (not all identical). The RLHF model should score higher than the SFT model on all gold RMs.

**Script**: `scripts/paper_comparison/hh_smoke_test.sh`

### Phase 1: RM Training
**Goal**: Train BT reward models on Anthropic-HH preferences.

**Setup**:
- Base model: `mycccc/Energy-Loss-Phenomenon-Demo/sft_model/` (same SFT used as policy — standard practice in EPPO/InfoRM)
- Dataset: Anthropic-HH preference data, converted to our messages format
  - Helpful train: ~43k pairs
  - Harmless train: ~42k pairs
  - Total: ~85k preference pairs (some papers use subsets)
- Architecture: `AutoModelForSequenceClassification` (our existing pipeline)
- Loss: Bradley-Terry (existing)
- Hyperparams: LR 1e-5, 1 epoch, batch 32 (following OpenRLHF defaults; EPPO uses LR 1e-6 for critic but that's the value model, not RM)

**Train 5 seeds** for ensemble experiments.

**Validation**:
- Eval accuracy on HH test split should be 65-75% (typical for Anthropic-HH)
- Compare to `weqweasdas/hh_rlhf_rm_open_llama_3b` scores on same test prompts as sanity check

**Problem**: Training a 7B RM is expensive (~40 GPU-hours per seed × 5 seeds = 200 GPU-hours).

**Alternative**: Train smaller proxy RMs first (e.g., from `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) for faster iteration, then scale up. This is scientifically valid — BSPO (#2) uses GPT2-large/TinyLlama/ShearedLlama as proxy RMs.

**Cheaper Phase 1 alternative**:
- Train 5× TinyLlama-1.1B RMs on HH (~5 GPU-hours each = 25 total)
- Use these for initial GRPO runs and pipeline validation
- Scale to 7B RMs only for final comparison numbers

**Script**: `scripts/paper_comparison/train_hh_rm.sh`

### Phase 2: GRPO Training
**Goal**: Train GRPO policies with our RM ensembles, starting from the SFT model.

**Setup**:
- Policy: `mycccc/Energy-Loss-Phenomenon-Demo/sft_model/`
- RMs: 5 trained RMs from Phase 1
- RL data: Anthropic-HH prompts (helpful + harmless train splits, prompts only ≈ 72k)
- KL base model: `mycccc/Energy-Loss-Phenomenon-Demo/sft_model/`

**Length config** (new `hh_paper` config):
```
max_prompt_tokens: 1024
max_response_tokens: 512
max_conversation_tokens: 1536
```
These match EPPO's `--max_seq_len 1024 --max_answer_seq_len 512`.

**GRPO hyperparams**:
- LR: 5e-7 (matching EPPO actor LR)
- beta (KL): 0.04 (standard GRPO default; EPPO uses KL weight 0 but has implicit KL via PPO clipping)
- num_generations: 16
- temperature: 1.0
- max_steps: 3000 (or equivalent)
- Save checkpoints every 200 steps

**Experiment matrix** (same structure as AlpacaFarm plan):
| Run | Ensemble | Aggregation | KL beta |
|-----|----------|-------------|---------|
| 1 | 1 RM | N/A | 0.04 |
| 2 | 5 RMs | mean | 0.04 |
| 3 | 5 RMs | min (WCO) | 0.04 |
| 4 | 5 RMs | uwo (λ=0.5) | 0.04 |
| 5 | 1 RM | N/A | 0.01 |
| 6 | 5 RMs | mean | 0.01 |
| 7 | 5 RMs | min (WCO) | 0.01 |
| 8 | 5 RMs | uwo (λ=0.5) | 0.01 |

**Script**: `scripts/paper_comparison/grpo_hh.sh`

### Phase 3: Evaluation
**Goal**: Score all checkpoints and produce comparison metrics.

**Local evaluation** (for development — cheap, run on every checkpoint):
- Score all checkpoints with all 4 gold RMs:
  - `Skywork/Skywork-Reward-V2-Llama-3.1-8B` (primary)
  - `Ray2333/GRM-Gemma2-2B-rewardmodel-ft` (secondary)
  - `weqweasdas/hh_rlhf_rm_open_llama_3b` (HH-specific)
  - `OpenRLHF/Llama-3-8b-rm-700k` (diverse)
- Plot gold RM score vs KL curves for each gold RM — overoptimization should be visible when proxy RM score keeps rising but gold RM score plateaus or drops
- Compare across gold RMs: do they agree on which ensemble strategy is best? Do overoptimization inflection points differ?
- Also compare our checkpoints against EPPO's published RLHF model baseline (scored in Phase 0)

**Paper-comparable evaluation** (for the final comparison):
- GPT-4 pairwise W/T/L evaluation
- Compare our GRPO policy responses vs SFT baseline responses
- Use the same prompt set as the papers (HH test/val prompts)
- Format: "Which response is more helpful/harmless?" with position randomization

**Script**: `scripts/paper_comparison/evaluate_hh.sh`

---

## Code Changes Required

### New length config in `data_utils.py`
```python
DATASET_LENGTH_CONFIGS = {
    ...
    "hh_paper": {
        "max_prompt_tokens": 1024,
        "max_response_tokens": 512,
        "max_conversation_tokens": 1536,
    },
}
```

### Dataset conversion script
`scripts/paper_comparison/convert_hh_dataset.py`:
- Load `Anthropic/hh-rlhf` (helpful-base, helpful-online, harmless-base)
- Parse the nested `\n\nHuman: ... \n\nAssistant: ...` format into messages
- Split into preference pairs (`chosen`/`rejected`) — already provided by the dataset
- Also create a prompts-only split for RL training
- Push to HF

### Vicuna chat template
Vicuna-7B-v1.5 already has a chat template in its tokenizer config. Verify it matches the EPPO template format: `USER: {content} ASSISTANT:`. If not, register the correct one via `setup_tokenizer()`.

### `evaluate_policy.py` — multi-RM support (refactor)
Currently `evaluate_policy.py` has hardcoded `--gold_rm_name` and `--secondary_rm_name` args. This should be replaced with a single `--eval_rm_names` list. Each RM is scored independently and logged with a short name derived from the model path (e.g. `Skywork/Skywork-Reward-V2-Llama-3.1-8B` → `Skywork-V2-8B`).

**Current state** (3 separate RM args, hardcoded roles):
```python
gold_rm_name: str = "Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
secondary_rm_name: str = "Ray2333/GRM-Gemma-2B-sftreg"
training_rm_path: str = "..."
```

**Target state** (single list, each gets its short name):
```python
eval_rm_names: list[str] = field(default_factory=lambda: [
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "Ray2333/GRM-Gemma2-2B-rewardmodel-ft",
])
training_rm_path: str = "..."  # kept separate — this is the proxy RM used during training
```

**Changes:**
- Replace `gold_rm_name` + `secondary_rm_name` with `eval_rm_names: list[str]`
- Add `short_rm_name(path) -> str` helper (e.g. last path component, strip common prefixes)
- Loop over `eval_rm_names`, score each, log as `{short_name}/mean`, `{short_name}/std`
- Keep `training_rm_path` as a separate arg (it's the proxy RM, different purpose)
- The KL vs gold reward plot should use the first RM in the list as the primary gold
- Backward compat: keep `--gold_rm_name` as alias for the first element of `--eval_rm_names`

**Memory management:** Load/score/unload each eval RM sequentially (as current `secondary_rm_name` already does) — they don't need to be in memory simultaneously.

### `llm_judge.py` — new file for LLM-as-a-judge via OpenRouter
The current `get_llm_judge_verdicts()` in `evaluate_policy.py` is broken (raises `NotImplementedError`) because it passes chat-template-formatted strings as the "question" to the Skywork judge template. Extract it into a standalone `llm_judge.py` that:

1. Takes structured `prompt_messages` (not formatted strings) and pairs of responses
2. Extracts the raw user question from messages for the judge template
3. Calls OpenRouter API (GPT-4 or configurable) with the Skywork pairwise template
4. Handles position randomization, retries with backoff, rate limiting
5. Returns W/T/L counts and per-sample verdicts
6. Supports batch parallelism via async requests (the current sequential implementation is very slow)

**Interface:**
```python
# llm_judge.py
def judge_pairwise(
    prompt_messages_list: list[list[dict]],   # structured messages
    responses_a: list[str],                    # e.g. policy responses
    responses_b: list[str],                    # e.g. SFT baseline responses
    model: str = "openai/gpt-4-turbo",         # OpenRouter model name
    api_key: str | None = None,
    max_concurrent: int = 10,
) -> JudgeResult:
    """Returns JudgeResult with per-sample verdicts and aggregate W/T/L."""
```

**Integration with `evaluate_policy.py`:**
- Add `--llm_judge_model` flag (default: None = disabled)
- When enabled, after generating responses for each checkpoint, also call `judge_pairwise()` comparing against baseline (SFT model responses or dataset chosen)
- Log `llm_judge/win_rate`, `llm_judge/loss_rate`, `llm_judge/tie_rate` per checkpoint

**One-shot GPT-4 validation:** Run one GPT-4 evaluation on the published EPPO RLHF model vs SFT model to reproduce their reported W/T/L numbers. This validates our judge implementation is correct before we use it on our own models.

### No changes needed to:
- `my_grpo.py` (ensemble logic already supports mean/min/uwo)
- `grpo_utils.py` (reward function building already works)
- `reward_utils.py` (AutoModelForSequenceClassification already supported)

---

## Key Differences from AlpacaFarm Plan

| Aspect | AlpacaFarm | Anthropic-HH |
|--------|------------|--------------|
| Policy size | Pythia-1.4B | Llama2-7B (Vicuna) |
| RM size | Pythia-44M | TinyLlama-1.1B (dev) / Llama2-7B (final) |
| Gold RM | AlpacaFarm 7B (1 model) | 4 gold RMs: Skywork-8B, GRM-2B, HH-3B, OpenRLHF-8B |
| Local gold proxy | AlpacaFarm 7B score | Multi-RM gold score vs KL curves |
| Primary metric | Gold RM score vs KL curve | GPT-4 W/T/L (paper standard) |
| Dataset size | 20K RL prompts | 72K RL prompts |
| SFT model | Must train from Pythia-1.4B | Use published EPPO SFT (`mycccc/Energy-Loss-Phenomenon-Demo/sft_model/`) |
| RM checkpoints | Must train (cheap: 44M) | Must train (expensive: 7B, or cheap: 1.1B) |
| Total GPU cost (dev) | ~50 GPU-hours | ~150 GPU-hours (with 1.1B RMs) |
| Total GPU cost (full) | ~100 GPU-hours | ~500 GPU-hours (with 7B RMs) |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| 7B RM training too expensive for 5 seeds | **High** | Start with TinyLlama-1.1B proxy RMs. Only scale to 7B for final numbers. |
| GPT-4 eval is expensive and non-reproducible | **Medium** | Use local gold RM (GRM-Gemma2-2B) for development. GPT-4 only for final paper-comparable numbers. Could also use open judges (Llama-3.1-70B-Instruct). |
| EPPO SFT model chat template mismatch | **Medium** | The published SFT model may not have a chat_template in its tokenizer. Verify tokenizer output matches EPPO's Vicuna template on 10 samples before training. May need to set template manually. |
| GRPO converges differently from PPO on HH | **Low** | This is a finding, not a bug. Document the difference. |
| HH dataset parsing edge cases (nested turns) | **Low** | Validate conversion by spot-checking 50 samples. Some HH examples have multi-turn conversations — need to handle correctly. |
| Llama2-7B gated access | **None** | Using EPPO's published SFT model (`mycccc/...`), which is ungated (Apache 2.0). No Llama access gate needed. |

---

## Execution Order

```
--- Infrastructure ---
 1. Add "hh_paper" length config to data_utils.py
 2. Refactor evaluate_policy.py: replace gold_rm_name + secondary_rm_name with eval_rm_names list
 3. Create llm_judge.py with judge_pairwise() + async OpenRouter support
 4. Integrate llm_judge.py into evaluate_policy.py (--llm_judge_model flag)

--- Data & Models ---
 5. Write convert_hh_dataset.py → parse Anthropic/hh-rlhf → push to HF
 6. Download EPPO SFT model (`mycccc/Energy-Loss-Phenomenon-Demo/sft_model/`)
 7. Verify SFT model chat template matches EPPO Vicuna format

--- Phase 0: Smoke Test ---
 8. Run evaluate_policy.py with EPPO SFT on HH val (all eval RMs)
 9. Score EPPO RLHF model (`mycccc/Energy-Loss-Phenomenon-Demo/rlhf_model/`) with all eval RMs → gold target
10. Run one GPT-4 judge eval: RLHF model vs SFT model → reproduce EPPO W/T/L to validate judge

--- Phase 1: RM Training ---
11. Train 5× TinyLlama-1.1B RMs on HH preference data
12. Verify RM eval accuracy (65-75%)

--- Phase 2+3: GRPO + Evaluation ---
13. Run Phase 2 GRPO with TinyLlama RMs (8 runs)
14. Run Phase 3 local evaluation (multi-RM gold score vs KL curves)
15. If curves look good: train 5× Llama2-7B RMs (Phase 1, expensive)
16. Re-run Phase 2+3 with 7B RMs
17. Run GPT-4 pairwise eval for final paper comparison
```
