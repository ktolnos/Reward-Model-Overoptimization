# LLM RL Best Practices

This document captures practical defaults for RLHF/RLAIF-style projects where data annotation, SFT, RM, RL training, and evaluation must stay aligned.

## 1) Data Filtering And Validation

- Validate dataset schema once, early, and fail fast with actionable errors.
- Enforce prompt/response/conversation length constraints in preprocessing, not at training runtime.
- Do not silently crop/truncate at runtime unless the method explicitly requires it.
- Keep one canonical filtering script so all experiments use the same rules.
- Log before/after counts and drop reasons for every split.

## 2) Split Strategy (Train/Select/Validation/Test)

- Use a four-way split: `train` (method training), `select` (no-peek checkpoint selection), `validation` (hyperparameter sweeps), `test` (final reporting only). See BENCHMARK.md §3 / §6.
- Use `test` for final reporting only; never iterate on it or use it for checkpoint selection.
- Split **by prompt group, not by row**: assign all rows sharing a prompt to the same split, so a prompt cannot leak across splits. This is standard grouped-splitting practice (cf. scikit-learn `GroupShuffleSplit`/`GroupKFold`) and matters here because the official HelpSteer3 train split has ~35% exact full-row duplicates (an artifact). Ratios apply to the prompt-group count, so row counts deviate slightly.
- Dedup exact duplicate rows before (or alongside) splitting where the dataset's design intent is one row per unit — grouping prevents cross-split leakage but not within-split redundancy.
- Assert splits are pairwise prompt-disjoint after splitting (contamination guard).
- If source datasets have multiple splits, define deterministic mapping rules (current rule: carve all four splits from source `train`, drop other source splits so held-out pools stay same-distribution).
- Version the split seed and ratios; never change them silently.

## 3) One Shared Processing Pipeline

- Share formatting/tokenization/length-validation code across:
  - annotation
  - SFT
  - RM training
  - RL training
  - evaluation
- Avoid stage-specific “almost the same” implementations.
- Prefer one source of truth for:
  - chat formatting
  - stop-token logic
  - token counting
  - prompt extraction

## 4) Tokenizers And Chat Templates

- Always use the correct tokenizer for the model being trained/scored/evaluated.
- Use the model’s chat template consistently; avoid ad-hoc string formatting.
- Keep BOS/EOS/special-token handling consistent between training and inference.
- Verify that prompt formatting is a strict prefix of full conversation formatting.
- For models missing `pad_token`, set it explicitly and consistently.

## 5) Reward Model Compatibility

- For pretrained/gold RMs, verify how they were trained:
  - expected text format (prompt+response vs full conversation)
  - special token behavior
  - EOS/turn-end handling
  - tokenizer family and chat template
- Match RM inference formatting to RM training conventions exactly.
- Validate this with a small known sample before large-scale runs.

## 6) SFT-Specific Rules

- Mask prompt tokens in labels; train loss on completion tokens only.
- In multi-turn preference data, define clearly which assistant turn is supervised (for example: last chosen response only).
- Ensure SFT prompt processing matches RL prompt processing (same extraction and template semantics).

## 7) RL-Specific Consistency

- Keep RL prompt construction identical to SFT/eval prompt construction.
- Keep generation stop criteria aligned across training and evaluation.
- Ensure reward inputs in RL match standalone RM evaluation inputs.
- Avoid accidental reward hacks from formatting mismatch or truncated contexts.

## 8) Reproducibility And Auditability

- Store and log:
  - dataset revision/hash
  - filtering config
  - split seed
  - tokenizer/model revision
  - code commit SHA
- Make pipeline stages independently rerunnable.

## 9) Evaluation Hygiene

- Separate:
  - online validation metrics (frequent)
  - final test-split metrics (rare, final only)
- Use the same prompt distribution and preprocessing as training unless testing explicit OOD generalization.

## 10) Easy-To-Forget Pitfalls

- Different tokenizers can change measured length and filtering outcomes materially.
- Small formatting mismatches can dominate apparent reward gains.

