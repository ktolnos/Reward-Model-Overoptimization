# Dataset Pipeline

This folder contains a 4-stage preprocessing pipeline for preference datasets.

## Stages

1. `stage1_verify_dataset.py`
- Verifies `chosen`/`rejected` schema and message invariants.
- Verifies samples are compatible with `apply_chat_template`.

2. `stage2_filter_split_upload.py`
- Filters by prompt/response/conversation token constraints.
- Splits into `train`, `test`, and `heldout`.
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
