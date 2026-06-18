# Comparison Plan: Our GRPO Pipeline vs. Coste et al. (2310.02743) PPO Results

## Goal

Compare gold reward curves from **our RM ensemble + GRPO pipeline** against the **PPO results from "Reward Model Ensembles Help Mitigate Overoptimization"** (Coste et al., ICLR 2024), using identical inputs (same base models, same dataset, same gold RM). Focus on GRPO vs PPO comparison (no BoN).

---

## 1. Paper's Experimental Setup

### Models
| Role | Model | Size | Source |
|------|-------|------|--------|
| SFT Policy | Pythia-1.4B (SFT'd on AlpacaFarm) | 1.4B | `tlc4418/pythia_1.4b_sft_policy` |
| Proxy RM base (small) | Pythia-70M (SFT'd) | 70M→44M RM | `tlc4418/pythia_70m_sft` |
| Proxy RM base (large) | Pythia-1.4B (SFT'd) | 1.4B→1.3B RM | (same architecture, larger) |
| Gold RM | AlpacaFarm human preference RM | 7B | `alpaca_farm_models/reward-model-human` |

### Published Artifacts
The paper publishes SFT models and datasets on HuggingFace, but **NOT trained RM checkpoints**:
- `tlc4418/pythia_1.4b_sft_policy` -- SFT'd policy model
- `tlc4418/pythia_70m_sft` -- SFT'd RM base model
- `tlc4418/1.4b-policy_preference_data_gold_labelled` -- 46K preference dataset
- `tlc4418/gold_labelled_gens` -- BoN generations (not needed for our comparison)
- RM checkpoints (`rm-pythia-44m_seed{1-5}`) are **NOT published** -- we must train our own

### RM Size Results Coverage
The paper tested **three proxy RM sizes**: 7M, 44M, and 1.3B. 44M is the default/focus (main paper figures). Full PPO gold reward curves also exist for:
- **7M**: Appendix F.3, Figure 18
- **1.3B**: Appendix F.4, Figure 19 (extended to 6000 PPO steps; paper notes large RMs need more steps)
- **Win-rate table** (Table 7) covers all sizes

The 44M results are the most complete. The 1.3B results show ensembles still help but need longer training. Consider running with both 44M and 1.3B proxy RMs for a more complete comparison.

### Data
| Split | Size | Purpose | Source |
|-------|------|---------|--------|
| SFT | 10K instructions | SFT training | AlpacaFarm "sft" split |
| Preference | 46K pairs | RM training | `tlc4418/1.4b-policy_preference_data_gold_labelled` |
| RL prompts | 20K | PPO/GRPO training | AlpacaFarm "unlabeled" split (DIFFERENT from RM data) |
| Eval | 2K | Validation | AlpacaFarm "val" split |

**Note:** PPO trains on the **"unlabeled" split** (20K prompts), which is different from both the RM preference data (46K generated pairs) and the SFT data (10K). See `dataset_loader.py:75`: when `mode == "rl"`, it loads `dataset["unlabeled"]`.

### Prompt Formats

**The paper uses TWO DIFFERENT prompt formats for different models:**

**1. Proxy RMs and Policy (trained Pythia models) -- OpenAssistant v2 format:**
```
<|prompter|>{instruction}<|endoftext|><|assistant|>{answer}<|endoftext|>
```
`<|prompter|>` and `<|assistant|>` are **special tokens added to Pythia's vocabulary during SFT**. They are NOT native Pythia tokens -- they come from the Open-Assistant project. Both the SFT policy model and the RM base model know these tokens because **RM training starts from the SFT'd model** (e.g., `tlc4418/pythia_70m_sft`).

This IS effectively a chat template -- the paper just applies it via manual string formatting rather than `apply_chat_template()`. We should match this exactly by registering it as a Jinja2 chat template.

Source: `llm_optimization-main/src/data_utils/oa_custom_datasets/rank_datasets.py:11,24`

**2. Gold RM (AlpacaFarm 7B) -- Alpaca instruction format, plain text only:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_if_present}

### Response:
{answer}
```
No model-specific special tokens. The gold RM was trained on this Alpaca template and must be queried using it. The `### Instruction:` markers are plain text, not special tokens.

Source: `llm_optimization-main/src/data_utils/rm_dataset_formatter.py:24-30` (note: this code has a likely bug with multi-line string concatenation; verify the exact format from the `alpaca_farm` package itself at `git+https://github.com/tlc4418/alpaca_farm.git`)

**The paper controls formatting via:**
- `output_alpaca=True/False` flag in `RMPromptDataset` (proxy vs gold format)
- `is_alpacafarm_rm=True/False` in `score_answers()` and `get_reward()` (also controls output extraction: `.rewards` vs `.logits[:, 0]`)
- Gold evaluation in `gold_score()` always uses `is_alpacafarm_rm=True`

### Hyperparameters

**SFT:** LR 8e-6, 3 epochs, batch 4

**RM Training:** LR 1e-5, 5 epochs, batch 32, BT loss, 5 seeds for ensemble

**PPO:** LR 1e-6, 3000 steps (6000 for 1.3B RM), batch 32, 256 rollouts, PPO epochs 4, clip 0.2, GAE lambda 0.95, adaptive KL (init 0.1, target 6)

**Generation:** max instruction 520 tokens, max answer 256 tokens, total 776, temperature 1.0, top-p 0.9 (1.0 for PPO training)

### Ensemble Strategies
- **Mean**: `score = mean(rm_scores)`
- **WCO (Worst-Case Optimization)**: `score = min(rm_scores)`
- **UWO (Uncertainty-Weighted Optimization)**: `score = mean(rm_scores) - lambda * var(rm_scores)` (lambda=0.5, note: uses **variance** not std)

### Key PPO Findings (our comparison targets)
- Single RM: gold reward rises then falls (overoptimization)
- WCO/UWO reduce overoptimization
- WCO/UWO + small KL penalty (0.01) eliminate overoptimization entirely
- Mean ensemble helps but still overoptimizes with label noise
- Larger RMs (1.3B) overoptimize less but still benefit from ensembles

---

## 2. Our Pipeline

| Component | Implementation | Key Files |
|-----------|---------------|-----------|
| Dataset Preprocessing | 4-stage pipeline (verify, filter, annotate, subsample) | `scripts/dataset_pipeline/` |
| SFT | TRL SFTTrainer | `rlhf/sft/my_sft.py` |
| RM Training | BT loss, AutoModelForSequenceClassification | `reward_models/run_reward_models_train.py` |
| Policy Optimization | GRPO (TRL GRPOTrainer) | `rlhf/grpo/my_grpo.py`, `grpo_utils.py` |
| Evaluation | Gold RM scoring + KL computation | `evaluate_policy.py` |
| Data Processing | Chat template formatting, validation | `data_utils.py` |

**Our ensemble strategies:** mean, min, uwo (same as paper) + sequential, mix (additional)

---

## 3. Prerequisite Code Changes

### 3a. Pythia Chat Template (runtime setup, not saved to disk)

**Problem:** Our pipeline uses `tokenizer.apply_chat_template()` everywhere. Pythia has no chat template natively.

**Solution:** Create a utility function `setup_pythia_chat_template(tokenizer)` in `data_utils.py` that is called every time a Pythia model is loaded (not saved to disk). This keeps the tokenizer source-of-truth on HuggingFace and avoids stale local copies.

The function:
1. Verifies `<|prompter|>`, `<|assistant|>`, `<|endoftext|>` tokens exist in vocabulary (they were added during SFT)
2. Sets `tokenizer.chat_template` to a Jinja2 template replicating OA v2 format:
   ```jinja2
   {% for message in messages %}{% if message['role'] == 'user' %}<|prompter|>{{ message['content'] }}<|endoftext|>{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}<|endoftext|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}
   ```

**Verification:** `tokenizer.apply_chat_template([{"role": "user", "content": "What is 2+2?"}], add_generation_prompt=True)` must produce exactly the same token IDs as the paper's manual `f"<|prompter|>What is 2+2?<|endoftext|><|assistant|>"` formatting.

### 3b. AlpacaFarm Gold RM: Chat Template for Consistent Formatting

**Problem:** The gold RM uses the Alpaca instruction format, which is different from the proxy RM format. Rather than having a separate `format_for_alpacafarm_gold_rm()` function that bypasses `apply_chat_template()`, we should implement this as a chat template too, ensuring consistency through the same code path.

**Solution:** Create `setup_alpacafarm_gold_chat_template(tokenizer)` in `data_utils.py` that sets a Jinja2 chat template producing the Alpaca format:

```jinja2
{% set ns = namespace(instruction='', input='') %}{% for message in messages %}{% if message['role'] == 'user' %}{% set ns.instruction = message['content'] %}{% endif %}{% endfor %}Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{ ns.instruction }}

### Response:
{% for message in messages %}{% if message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{% endif %}
```

This way, gold RM scoring uses the same `apply_chat_template()` code path as everything else -- just with a different template registered on the gold RM's tokenizer. The gold RM's tokenizer gets the Alpaca template; proxy RM tokenizers get the OA v2 template.

**Note:** Need to verify the exact Alpaca template from the `alpaca_farm` package (`git+https://github.com/tlc4418/alpaca_farm.git`) since the paper's own `rm_dataset_formatter.py:26-29` has a likely string concatenation bug. The `input` field handling (when present) also needs to match.

### 3c. Dataset Conversion and Preprocessing

**Step 1: Convert paper's dataset to our messages format.**

Create `scripts/paper_comparison/convert_paper_dataset.py`:
1. Load `tlc4418/1.4b-policy_preference_data_gold_labelled`
2. For each row:
   - Combine `instruction` + `\n` + `input` (if non-empty) into user content
   - Map `preference` index → chosen/rejected answers
   - Output `{chosen: [{role: "user", content: ...}, {role: "assistant", content: ...}], rejected: [...]}`
3. Push to HuggingFace Hub

Also convert:
- AlpacaFarm "unlabeled" split (20K prompts) for GRPO training
- AlpacaFarm "val" split (2K) for evaluation
- AlpacaFarm "sft" split (10K) for SFT training (Phase 1a)

**Step 2: Run through our dataset pipeline** (`scripts/dataset_pipeline/`).

The converted dataset must go through the standard pipeline:
- **Stage 1** (`stage1_verify_dataset.py`): Verify chosen/rejected schema, message format
- **Stage 2** (`stage2_filter_split_upload.py`): Filter by token length, split (by prompt group) into train/select/validation/test

Length limits for the paper comparison (**different from our defaults**):
- `MAX_PROMPT_TOKENS = 520` (paper's max instruction length)
- `MAX_RESPONSE_TOKENS = 256` (paper's max answer length)
- `MAX_CONVERSATION_TOKENS = 776` (520 + 256)

This filtering must use the Pythia tokenizer (with OA v2 chat template applied) since that's what the paper uses. Pass `--tokenizer-name` to the pipeline scripts.

**Stage 3** (annotation): Score with AlpacaFarm gold RM to get `reference_reward` fields (requires gold RM support from 3b).

**Stage 4** (subsample): May not be needed if using full 46K dataset like the paper.

### 3d. Length Constants and Dataset-Aware Assertions

**Problem:** `data_utils.py` has hardcoded constants:
```python
DEFAULT_MAX_PROMPT_TOKENS = 1024
DEFAULT_MAX_RESPONSE_TOKENS = 1024
DEFAULT_MAX_CONVERSATION_TOKENS = 2048
```
Paper uses 520/256/776. The purpose of these constants is pipeline consistency -- all stages use the same limits.

**Solution:** Define dataset-specific constant sets and assert the right one is used:

```python
# data_utils.py
DATASET_LENGTH_CONFIGS = {
    "default": {
        "max_prompt_tokens": 1024,
        "max_response_tokens": 1024,
        "max_conversation_tokens": 2048,
    },
    "alpacafarm_paper": {
        "max_prompt_tokens": 520,
        "max_response_tokens": 256,
        "max_conversation_tokens": 776,
    },
}
```

Training scripts select the config via a flag (e.g., `--length_config alpacafarm_paper`). Asserts in `my_grpo.py` verify that the active constants match the dataset being used:
```python
assert training_args.max_completion_length == active_config["max_response_tokens"]
assert training_args.vllm_max_model_length == active_config["max_conversation_tokens"]
```

All length filtering happens in the dataset pipeline (`scripts/dataset_pipeline/`). Training code only verifies the dataset is already correctly filtered.

### 3e. AlpacaFarm Gold RM Model Loading

**Problem:** The gold RM (`alpaca_farm_models/reward-model-human`) uses a custom `RewardModel` class from the `alpaca_farm` package that outputs `.rewards` instead of `.logits[:, 0]`.

**Solution:** Modify `reward_utils.py`:
- In `load_reward_model()`: detect AlpacaFarm RM and load via `alpaca_farm.models.reward_model.RewardModel.from_pretrained()`
- In `extract_reward_tensors_from_model_output()`: handle the `.rewards` attribute
- The prompt formatting is handled by the chat template approach (3b), not special-cased here

**Dependency:** `pip install alpaca-farm` (paper uses fork: `git+https://github.com/tlc4418/alpaca_farm.git`)

### 3f. Evaluation Pipeline Updates

**File:** `evaluate_policy.py`

1. Add `--gold_rm_path` argument (default: current Skywork RM; set to `alpaca_farm_models/reward-model-human` for paper comparison)
2. Gold RM tokenizer gets Alpaca chat template via `setup_alpacafarm_gold_chat_template()` at load time
3. Add `--eval_dataset_path` to specify AlpacaFarm eval prompts (instead of helpsteer default)
4. Set `--kl_base_model_path` to SFT model for KL divergence computation

### 3g. UWO Formula: Add `--uwo_use_variance` Flag

**Problem:**
- Paper: `score = mean - lambda * variance`
- Our code: `score = mean - lambda * std`

**Solution:** Add `uwo_use_variance: bool = False` to `MyGRPOScriptArguments` in `my_grpo.py`. In `grpo_utils.py` ensemble aggregation:
```python
if uwo_use_variance:
    reward = mean_reward - uwo_lambda * rewards_tensor.var(dim=1)
else:
    reward = mean_reward - uwo_lambda * rewards_tensor.std(dim=1)
```

For the paper comparison, run with `--uwo_use_variance True --uwo_lambda 0.5` to match exactly.

---

## 4. New Scripts

All in `scripts/paper_comparison/`:

| Script | Purpose |
|--------|---------|
| `convert_paper_dataset.py` | Convert `tlc4418/1.4b-policy_preference_data_gold_labelled` + AlpacaFarm splits to our messages format, run through dataset pipeline |
| `train_paper_sft.sh` | SFT Pythia-1.4B on AlpacaFarm with our `my_sft.py` |
| `train_paper_rm.sh` | Train Pythia-44M RM with paper's hyperparams (accepts seed arg) |
| `grpo_paper.sh` | GRPO training with paper-matching config (accepts ensemble args) |
| `evaluate_paper.sh` | Gold evaluation with AlpacaFarm 7B RM |

---

## 5. Experiment Plan

### Incremental Strategy

Since the paper publishes SFT models but NOT RM checkpoints, the fastest path to a first comparison is:

1. **Phase 1b first** (RM training): Use paper's published SFT model as RM base → train our own RMs → run GRPO → evaluate. This tests our RM training + GRPO against their PPO.
2. **Phase 1a later** (SFT): Re-train SFT with our code to validate full pipeline end-to-end.

### Phase 0: Setup (~1-2 hours)
- [ ] Create `scripts/paper_comparison/` directory
- [ ] Add `setup_pythia_chat_template()` to `data_utils.py`
- [ ] Add `setup_alpacafarm_gold_chat_template()` to `data_utils.py`
- [ ] Add dataset length config system (`DATASET_LENGTH_CONFIGS`) to `data_utils.py`
- [ ] Add `uwo_use_variance` flag to `my_grpo.py` and `grpo_utils.py`
- [ ] Add AlpacaFarm gold RM model loading to `reward_utils.py`
- [ ] Update `evaluate_policy.py` for AlpacaFarm gold RM
- [ ] Write and run `convert_paper_dataset.py` -- convert and push to HF
- [ ] Run converted dataset through `scripts/dataset_pipeline/` with Pythia tokenizer and 520/256/776 limits
- [ ] Verify the pipeline output loads correctly (sample `format_and_validate_preference_sample`)
- [ ] Install `alpaca-farm` package (paper's fork: `git+https://github.com/tlc4418/alpaca_farm.git`)
- [ ] **Verify formatting**: Compare token IDs from our `apply_chat_template()` vs paper's manual string formatting on 5-10 sample prompts
- [ ] **Verify gold RM formatting**: Confirm Alpaca chat template output matches what the alpaca_farm package expects

### Phase 1b: RM Training (~5 GPU-hours for 44M; ~40 for 1.3B)
- [ ] Train 5x Pythia-44M RMs (seeds 1-5) using `run_reward_models_train.py`
- [ ] Base model: `tlc4418/pythia_70m_sft` (with `setup_pythia_chat_template()` called at load)
- [ ] Dataset: pipeline-processed AlpacaFarm preference data (46K pairs)
- [ ] Hyperparams: LR 1e-5, 5 epochs, batch 32, max_length 776, BT loss
- [ ] Verify eval accuracy in range 60-75% (matching paper)
- [ ] **Optional**: Also train 5x 1.3B RMs for more complete comparison (paper's Figure 19)

### Phase 1a: SFT with our code (optional, ~2-4 GPU-hours)
- [ ] Train Pythia-1.4B on AlpacaFarm SFT split using `my_sft.py`
- [ ] Script: `scripts/paper_comparison/train_paper_sft.sh`
- [ ] Hyperparams: LR 8e-6, 3 epochs, batch 4
- [ ] Length config: `alpacafarm_paper` (520/256/776)
- [ ] Compare eval loss to paper's SFT quality
- [ ] Re-run Phase 1b with our SFT model as RM base for full pipeline validation

### Phase 2: GRPO Training (~40-80 GPU-hours)

**Experiment matrix (core runs):**

| Run | Ensemble | Aggregation | KL beta | Notes |
|-----|----------|-------------|---------|-------|
| 1 | 1 RM (seed 1) | N/A | 0.1 | Single RM baseline, high KL |
| 2 | 5 RMs | mean | 0.1 | Mean ensemble, high KL |
| 3 | 5 RMs | min (WCO) | 0.1 | Conservative, high KL |
| 4 | 5 RMs | uwo (lambda=0.5, variance) | 0.1 | Uncertainty-weighted, high KL |
| 5 | 1 RM (seed 1) | N/A | 0.01 | Single RM baseline, low KL |
| 6 | 5 RMs | mean | 0.01 | Mean ensemble, low KL |
| 7 | 5 RMs | min (WCO) | 0.01 | Conservative, low KL |
| 8 | 5 RMs | uwo (lambda=0.5, variance) | 0.01 | Uncertainty-weighted, low KL |

**GRPO hyperparams (matching paper where applicable):**
- Policy model: `tlc4418/pythia_1.4b_sft_policy` (with chat template) OR our re-trained SFT
- Dataset: AlpacaFarm "unlabeled" split prompts (converted and pipeline-processed)
- LR: 1e-6 (matching paper's PPO LR)
- Length config: `alpacafarm_paper` (max_completion=256, max_prompt=520, vllm_max=776)
- num_generations: 16 (GRPO-specific; paper uses 256 rollouts for PPO)
- temperature: 1.0
- Steps: ~3000 (or equivalent data seen)
- Save checkpoints every 200 steps
- rm_subtract_mean_reward_per_model: True
- rm_scale_reward_by_std_per_model: True
- For UWO runs: `--uwo_use_variance True`

**Optional additional runs:**
| Run | Ensemble | Aggregation | KL beta | Notes |
|-----|----------|-------------|---------|-------|
| 9 | 5 RMs | mean | 0.0 | No KL (paper shows this overoptimizes) |
| 10 | 5 RMs | min (WCO) | 0.0 | WCO without KL |
| 11-13 | 5 RMs | uwo | 0.01 | Lambda sweep: 0.1, 0.5, 1.0 |

### Phase 3: Gold Evaluation (~10-20 GPU-hours)
- [ ] For each GRPO checkpoint (every 200 steps across all runs):
  - Generate responses on AlpacaFarm eval prompts (2K "val" split)
  - Score with AlpacaFarm 7B gold RM (using Alpaca chat template on gold RM tokenizer)
  - Compute KL divergence from SFT policy
- [ ] Store results in structured format for plotting

---

## 6. Comparison Methodology

### Primary Comparison: GRPO vs PPO Gold Reward Curves

For each ensemble strategy (single, mean, WCO, UWO):
1. **Our GRPO curve:** Gold reward (y-axis) vs KL divergence from SFT (x-axis)
2. **Paper's PPO curve:** Extracted from paper's Figures 4-5 (44M RM, 46K data)

### Key Metrics
| Metric | Definition |
|--------|-----------|
| Peak gold reward | Maximum gold reward achieved across training |
| KL at peak | KL divergence at which peak gold reward occurs |
| Overoptimization severity | Gold reward drop from peak at max KL |
| Final gold reward | Gold reward at end of training |
| Relative improvement | (ensemble peak - single RM peak) / single RM peak |

### Expected Outcomes
- GRPO should show similar overoptimization patterns as PPO (single RM gold reward rises then falls)
- Ensemble benefits (WCO/UWO reducing overoptimization) should transfer from PPO to GRPO
- GRPO-specific effects (no value function, group-relative baseline) may change the overoptimization dynamics
- The KL range reached may differ between PPO and GRPO

---

## 7. Critical Differences to Document

| Aspect | Paper (PPO) | Ours (GRPO) |
|--------|-------------|-------------|
| RL algorithm | PPO (actor-critic) | GRPO (group relative) |
| Value function | Learned value head | None (group baseline) |
| KL mechanism | Adaptive KL penalty (init=0.1, target=6) | Fixed beta coefficient |
| Rollouts per step | 256 | 16 (num_generations) |
| Advantage estimation | GAE (lambda=0.95) | Group-relative baseline |
| RM reward normalization | Post-training mean/std on eval set | Pre-computed statistics on dataset sample |
| UWO formula | mean - lambda * variance | mean - lambda * variance (with `--uwo_use_variance`) |
| Proxy RM prompt format | Manual OA v2 string formatting | `apply_chat_template()` with OA v2 Jinja2 template (must produce identical tokens) |
| Gold RM prompt format | Alpaca template (`### Instruction:`) | `apply_chat_template()` with Alpaca Jinja2 template on gold RM tokenizer |

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Chat template produces different token IDs than paper's manual formatting | **High** | Side-by-side token ID comparison on 10+ samples before any training. Must match exactly. |
| Alpaca gold RM chat template doesn't match what `alpaca_farm` package expects | **High** | Verify by comparing our template output against the package's own formatting on sample inputs. |
| AlpacaFarm gold RM package incompatibility | **Medium** | Fallback: use paper's `score.py` directly for gold evaluation. Keep our pipeline for proxy RMs and GRPO. |
| GRPO converges very differently from PPO | **Medium** | This IS a finding if it happens. Plot both step-count and KL-based x-axes. |
| Dataset pipeline filtering drops too many AlpacaFarm samples at 520/256/776 limits | **Low** | Check filter rate. Paper used these limits so most samples should pass. |
| 25% label noise experiments double the run count | **Low** | Start with no-noise. Add noise only if initial results are promising. |
| 1.3B RM training too expensive | **Low** | Start with 44M (main paper focus). 1.3B is optional for a more complete picture. |

---

## 9. File Change Summary

### Files to Create
| File | Purpose |
|------|---------|
| `scripts/paper_comparison/convert_paper_dataset.py` | Convert paper datasets to messages format, run through pipeline |
| `scripts/paper_comparison/train_paper_sft.sh` | SFT with paper settings |
| `scripts/paper_comparison/train_paper_rm.sh` | RM training with paper settings (accepts seed arg) |
| `scripts/paper_comparison/grpo_paper.sh` | GRPO with paper-matching config |
| `scripts/paper_comparison/evaluate_paper.sh` | Gold evaluation with AlpacaFarm RM |

### Files to Modify
| File | Change |
|------|--------|
| `data_utils.py` | Add `setup_pythia_chat_template()`; add `setup_alpacafarm_gold_chat_template()`; add `DATASET_LENGTH_CONFIGS` |
| `rlhf/grpo/my_grpo.py` | Replace hardcoded length overrides with dataset-config assertions; add `uwo_use_variance` arg |
| `rlhf/grpo/grpo_utils.py` | Add `uwo_use_variance` logic in ensemble aggregation (~line 679) |
| `reward_utils.py` | Add AlpacaFarm `RewardModel` loading + `.rewards` extraction |
| `evaluate_policy.py` | Add `--gold_rm_path`, `--eval_dataset_path` args; gold RM tokenizer gets Alpaca chat template at load |

---

## 10. Execution Order

```
 1. Create scripts/paper_comparison/ directory
 2. Add setup_pythia_chat_template() to data_utils.py
 3. Add setup_alpacafarm_gold_chat_template() to data_utils.py
 4. Add DATASET_LENGTH_CONFIGS to data_utils.py
 5. Add uwo_use_variance flag to my_grpo.py + grpo_utils.py
 6. Update my_grpo.py length enforcement → dataset-config assertions
 7. Add AlpacaFarm gold RM model loading to reward_utils.py
 8. Update evaluate_policy.py for gold RM + eval dataset flexibility
 9. Write convert_paper_dataset.py → run it → push to HF
10. Run converted dataset through scripts/dataset_pipeline/ (520/256/776 limits, Pythia tokenizer)
11. Verify OA v2 chat template: compare apply_chat_template() output vs paper's manual formatting
12. Verify Alpaca chat template: compare output vs alpaca_farm package formatting
13. Install alpaca-farm package (git+https://github.com/tlc4418/alpaca_farm.git)
14. Write train_paper_rm.sh → submit seeds 1-5 with paper's SFT model (Phase 1b)
15. Verify RM eval accuracy matches paper (~60-75%)
16. Write grpo_paper.sh → submit 8 core runs (Phase 2)
17. Write evaluate_paper.sh → run on all checkpoints (Phase 3)
18. Collect results, plot gold reward vs KL curves
19. Compare against paper's PPO curves from Figures 4-5
20. (Optional) Write train_paper_sft.sh → re-train SFT → re-run Phase 1b+2+3 (Phase 1a)
```
