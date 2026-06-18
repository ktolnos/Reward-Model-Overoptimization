# Work since `45654741`

**Summary.** Implemented the four-way prompt-grouped dataset split from BENCHMARK.md (replacing the old three-way `train/test/heldout`), plus small tokenizer/GRPO fixes. The new dataset is **`withcomment/helpsteer3-qwen35_annotated_human`** on the Hugging Face Hub (public), with splits `train/select/validation/test`. No trained real checkpoints (SFT/RM/GRPO) are available. Two things still open: the `select` split has no consumer yet, and within-split row duplication is not deduped.

Range `45654741..1e8f45e`. Commits: `1e8f45e` (four-way split), `25010eb` (killarney runscript), `3efdef7` (benchmark roadmap + grpo soft penalty → dapo default), `35a72ee` (tokenizer eos fallback); `fdac438`/`b6980e1` are merges.

**Four-way split.** [pipeline_common.py](scripts/dataset_pipeline/pipeline_common.py): `split_three_way` → `split_four_way` (`train/select/validation/test` at `0.85/0.05/0.05/0.05`). Split is by prompt group, not by row — rows grouped on a SHA-256 of the prompt (`chosen[:-1]`), shuffled, sliced by group count (so row counts drift from the ratios; `test` takes the remainder). This stops duplicate prompts leaking across splits. `assert_splits_disjoint` checks this in Stage 2. Multi-split sources now carve all four from `train` and drop the rest. `--heldout-ratio` removed; `--select-ratio`/`--validation-ratio` added; `--train-ratio` default `0.9→0.85`. Eval takes a `--split` arg ([evaluate_policy.py](evaluate_policy.py), default `test`) and raises on a missing split. Old-pipeline HF datasets (old splits + leaky row-level shuffle) must be regenerated.

**Duplication.** 36.6% of HelpSteer3 rows are identical full-row duplicates (not deduped upstream); the ~43% multi-row prompts are mostly that, with only ~665 prompts carrying distinct response-pairs. Split-by-prompt stops cross-split leakage but within-split duplication remains.

**Open — `select` has no consumer.** Training reads `split="train"`, eval reads `validation`/`test`. Nothing reads `select`. The no-peek checkpoint-selection rule (BENCHMARK.md §6) is not wired up.

**Small fixes.** Tokenizer EOS: `get_generation_stop_token_ids` ([data_utils.py](data_utils.py)) reads EOS off the raw tokenizer, and the multi-token-EOS path in [my_grpo.py](rlhf/grpo/my_grpo.py) falls back to `self.processing_class.tokenizer.eos_token_id` (`35a72ee`).

**Republished dataset.** `withcomment/helpsteer3-qwen35_annotated_human` (public), four splits `train/select/validation/test` = 24757/1455/1469/1478. All three old splits were merged before splitting, via Stage 2's `--merge-splits` flag ([stage2_filter_split_upload.py](scripts/dataset_pipeline/stage2_filter_split_upload.py)). No dedup applied.

**Running the pipeline.** The four splits are produced automatically by Stage 2 — no flag needed; the four-way split is the default. Requires `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in the shell.

- Full run from an upstream source (SLURM; Stage 1 verify → Stage 2 filter+split+upload → Stage 3 annotate → Stage 4 subsample, chained with `afterok`):
  ```bash
  scripts/dataset_pipeline/submit_full_pipeline.sh \
    --source-dataset nvidia/HelpSteer3 --reward-model <gold_rm> \
    --prefix helpsteer3-qwen35 --namespace <hf_user>
  ```
  Output names are built from `--prefix`/`--namespace`/`--reward-model`. Add `--skip-annotation` for a human-preference-only dataset (no RM scores; suffix `_human`). Do **not** merge here: upstream splits are genuinely different distributions, so Stage 2 correctly carves the four splits from the source `train` and drops the rest.
- Re-split an existing derived three-way dataset (no SLURM/GPU; how `withcomment/...` was made) — run Stage 2 directly with `--merge-splits`, valid only because that dataset's `train/test/heldout` are a row-level partition of one population:
  ```bash
  python scripts/dataset_pipeline/stage2_filter_split_upload.py \
    --source-dataset ktolnos/helpsteer3-qwen35_annotated_human \
    --output-dataset <hf_user>/helpsteer3-qwen35_annotated_human --merge-splits
  ```
