# Dataset Pipeline

This folder contains a 4-stage preprocessing pipeline for preference datasets.

## Stages

1. `stage1_verify_dataset.py`
- Verifies `chosen`/`rejected` schema and message invariants.
- Verifies samples are compatible with `apply_chat_template`.

2. `stage2_filter_split_upload.py`
- Filters by prompt/response/conversation token constraints.
- Splits **by prompt group** into `train`, `select`, `validation`, and `test` (default ratios `0.85 / 0.05 / 0.05 / 0.05`). All rows sharing a prompt go to the same split — chiefly because the official HelpSteer3 train split contains ~35% exact full-row duplicates (measured directly; a systematic row-level artifact whose construction cause the paper does not document — the intended unit is one aggregated row per sample), so a row-level split would leak identical rows across splits. `assert_splits_disjoint` verifies the splits are pairwise prompt-disjoint. When the source has multiple splits, all four are carved from the source `train` split and other source splits are dropped. See BENCHMARK.md §3 for the split semantics, and HANDOFF.md for the duplication finding and the open dedup question.
- Uploads filtered dataset to Hugging Face.

3. `stage3_annotate_and_upload.py`
- Scores `chosen`/`rejected` with a reward model and writes:
  - `chosen_reward`
  - `rejected_reward`
  - `does_gold_agree_with_original`
- Uploads annotated dataset to Hugging Face.
- Run through `experimental/annotate_dataset.sh` (GPU sbatch script).

4. `stage4_subsample_upload.py`
- Subsamples each split (default 25%).
- Uploads to Hugging Face.

## Full pipeline submission

Use:

```bash
scripts/dataset_pipeline/submit_full_pipeline.sh \
  --source-dataset <source_repo> \
  --reward-model <reward_model> \
  --prefix <prefix> \
  --namespace <hf_user_or_org>
```

This submits:
- `stage1_verify_stage2_filter.sbatch`
- `experimental/annotate_dataset.sh`
- `stage4_subsample.sbatch`

with `afterok` dependencies.
