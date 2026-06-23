# Per-example persistence — post-run verification checklist

Everything below must be checked **after the next real eval run** (a GPU run of
`evaluate_policy.py`, ideally first with `--debug` on a small subset). The goal
is to confirm that the per-example logs let us recompute every metric — and
re-score with new reward models — **without re-running generation and without
the policy checkpoints**.

Local unit tests already pass (recorder round-trip, sparse/per-category merge,
typed judge-details dispatch, manifest with git+args). What can only be checked
on a real run is wired below.

---

## 0. Run to produce artifacts

```bash
# Small, fast smoke run (writes to <stem>_debug_per_example/)
python evaluate_policy.py --checkpoints_dir <CKPTS> --debug \
    --benchmarks select,preference,ifeval,arena_hard \
    --sibling_rm_path <SIBLING_RM> \
    --kl_base_model_path <SFT_CKPT> ...
```

`select` is in the default `--benchmarks`; it requires `--sibling_rm_path` (the
run errors out early otherwise). Drop `select` from `--benchmarks` to skip
checkpoint selection entirely.

Artifacts land in `<output_file_stem>_per_example/` (or `<stem>_debug_...` for
`--debug`). Persistence is always on — there is no disable flag.

---

## 1. Files & manifest exist

- [ ] `<stem>_per_example/_manifest.json` exists.
- [ ] One `*.parquet` per `(benchmark, checkpoint)`:
      `select__checkpoint-<n>.parquet`, `preference__checkpoint-<n>.parquet`,
      `ifeval__checkpoint-<n>.parquet`, `arena_hard__checkpoint-<n>.parquet`
      for **every** checkpoint evaluated.
- [ ] `_manifest.json` has:
  - `git.commit` (40-char SHA), `git.branch`, `git.dirty`, `git.available == true`.
  - `args` — the **full** `ScriptArguments` (every flag, not a subset).
  - `benchmarks` — per-benchmark evaluators + thinking + n.

```python
import json; m = json.load(open("<stem>_per_example/_manifest.json"))
assert m["git"]["available"] and len(m["git"]["commit"]) == 40
assert m["args"]["dataset_name"] and m["args"]["split"]
```

---

## 2. Schema present per benchmark

Load each file and confirm columns. Base columns (all benchmarks):
`benchmark, checkpoint, prompt_uid, prompt_messages_json, sample_idx,
response_text, response_raw_text, response_token_len, finish_reason, over_budget`.

- [ ] **preference**: `score__rm_gold_rm`; `score__rm_training_rm` (if
      `--evaluate_with_training_rm`); `score__rm_secondary_rm` (if secondary set);
      `chosen_or_baseline_score__<label>`, `reference_response_text`; KL columns
      `kl__k1, kl__grpo, policy_mean_logprob, base_mean_logprob` (if
      `--kl_base_model_path` set).
- [ ] **select**: `score__rm_sibling_rm` (one per response). The selection
      split carries `chosen_response` metadata, but win-rate columns are absent
      (no chosen-score cache for this split) — only the mean signal is used.
- [ ] **ifeval**: `ifeval_prompt_strict`, `ifeval_prompt_loose`;
      `score__rm_gold_rm` (if `--ifeval_use_gold_rm`).
- [ ] **arena_hard** (RM judge): `score__rm_gold_rm`,
      `chosen_or_baseline_score__rm_gold_rm__<slot>`,
      `battle_mean__rm_gold_rm__<slot>`, `baseline_response_text__<slot>`.
      Per-category mode → `<slot>` is the category (e.g. `coding`); global mode →
      the baseline model name.
- [ ] **arena_hard** (LLM judge, if configured): `judge_label_game0__<judge>__<slot>`,
      `judge_label_game1__<judge>__<slot>`.

---

## 3. Row counts & join keys

- [ ] Each file has `n_prompts * n_responses_per_example` rows (n=1 under the
      frozen decoding config → one row per prompt).
- [ ] `sample_idx` cycles `0..n-1` within each prompt.
- [ ] **`prompt_uid` is stable across checkpoints**: the set of uids in
      `*__checkpoint-0.parquet` equals the set in `*__checkpoint-<N>.parquet`
      for the same benchmark. (This is the cross-checkpoint join key.)

```python
import pandas as pd, glob
for b in ["preference","ifeval","arena_hard"]:
    fs = sorted(glob.glob(f"<stem>_per_example/{b}__checkpoint-*.parquet"))
    uid_sets = [set(pd.read_parquet(f)["prompt_uid"]) for f in fs]
    assert all(s == uid_sets[0] for s in uid_sets), b
```

---

## 4. ⭐ Recompute == logged (the core recoverability test)

Recompute aggregates **from the parquet** and confirm they match the CSV /
wandb numbers for the same checkpoint. This is the proof that nothing is lost.

- [ ] `mean(score__rm_gold_rm)` == `gold_rm/mean` in the CSV (preference).
- [ ] Win-rate from per-example == logged `gold_rm/win_rate_vs_chosen`:
      `mean(score__rm_gold_rm > chosen_or_baseline_score__rm_gold_rm)` (ties=0.5).
- [ ] IFEval: `mean(ifeval_prompt_strict)` == `ifeval/prompt_strict_acc`;
      same for loose.
- [ ] KL: `mean(kl__k1)` == `kl/mean`; `mean(kl__grpo)` == `kl/grpo_mean`.
- [ ] Arena: recompute `arena_score`/win-rate per slot from `battle_mean__*`
      (or from `score__` vs `chosen_or_baseline_score__*`) == logged values.

(Allow tiny float tolerance; bootstrap CIs use a fixed seed so point estimates
should match closely. `sc_score` uses a stochastic BT fit — recompute should be
within CI, not bit-identical.)

---

## 5. ⭐ Re-score with a "new" RM offline (no checkpoint, no regen)

Prove the headline use case: take only the parquet, rebuild RM inputs from
`prompt_messages_json` + `response_text`, score with a reward model, and confirm
it reproduces the stored `score__rm_gold_rm`. (Use the gold RM as the stand-in
"new" RM — if it reproduces, any future panel RM can be added post-hoc.)

```python
import pandas as pd, json
from policy_eval.rewards import score_responses_with_rm
from reward_utils import load_reward_model
df = pd.read_parquet("<stem>_per_example/preference__checkpoint-<n>.parquet")
prompts = [json.loads(s) for s in df["prompt_messages_json"]]
model, tok = load_reward_model("<GOLD_RM>", reasoning=False, device="cuda")
scores = score_responses_with_rm(list(df["response_text"]), prompts, model, tok,
                                 batch_size=1, device="cuda")
assert (abs(scores - df["score__rm_gold_rm"].values) < 1e-3).all()
```

- [ ] Re-scored values match `score__rm_gold_rm` → responses are re-scorable
      from disk alone (checkpoints can be deleted).

---

## 6. over_budget / length accounting

- [ ] Rows with `finish_reason == "length"` have `over_budget == True`.
- [ ] On benchmarks where `max_new_tokens` > `--response_token_budget` (1024),
      rows with `response_token_len > 1024` are `over_budget == True` even if
      `finish_reason == "stop"`.
- [ ] IFEval `n_truncated` printed in the log == count of `finish_reason ==
      "length"` rows in the ifeval parquet.
- [ ] `response_token_len` is a positive int for generated responses (it is
      `None` only on the `--evaluate_chosen_responses` path, which doesn't generate).

---

## 7. Edge cases / modes

- [ ] `--evaluate_chosen_responses` writes `preference__checkpoint-0.parquet`
      (chosen text in `response_text`; `response_token_len` is null — expected).
- [ ] `--per_example_format jsonl` produces readable `.jsonl` with the same data.
- [ ] `--per_example_dir <DIR>` override lands files in `<DIR>`.
- [ ] Arena per-category mode: `baseline_response_text__<category>` and
      `score__rm_gold_rm` are populated for every prompt across all categories
      (the per-category sparse-merge fills the whole `score__` column), while
      per-slot baseline/battle columns are NaN/"" for prompts outside that slot
      (expected).
- [ ] `--debug` run writes to the `_debug`-suffixed per-example dir.

---

## 8. Known gaps (decide if they matter before the freeze)

- **Deferred vLLM LLM judge** (open-weight judge, roadmap B1) does **not** yet
  append per-example verdicts — it's an unimplemented stub. When implemented it
  must merge `judge_label_*` columns back into the existing file by `prompt_uid`.
- **Per-token logprobs** are not stored (only per-sample `kl__k1`/`kl__grpo` +
  `policy_mean_logprob`/`base_mean_logprob`). The two standard KL estimators are
  recomputable; a brand-new token-level estimator would need a re-run.
- **LLM API judge** stores parsed labels only, not the judge's raw explanation
  text — re-parsing under a changed regex would need a re-run.
- Confirm per-file parquet size is reasonable (Arena baseline text is duplicated
  per checkpoint; acceptable vs checkpoint size, but watch total footprint).

---

## 9. ⭐ Checkpoint selection (sibling RM) & aggregate headline metrics

The `select` benchmark scores every checkpoint with the sibling RM on the
`--selection_split` (default `select`); the argmax over checkpoints picks the
deployed one. Its main metrics — computed on `--split` (validation/test), a
**different** split than selection — are reported as the run's headline.

**Per-checkpoint (CSV + wandb history):**
- [ ] `select/sibling_rm/mean` is logged for **every** checkpoint.
- [ ] `arena_hard/aggregate/sc_score` is present and equals the macro-average of
      the per-category style-controlled scores
      (`arena_hard/rm_gold_rm/{hard_prompt,coding,math,creative_writing}/sc_score`).
- [ ] `ifeval/aggregate/strict_acc` equals `mean(ifeval/prompt_strict_acc,
      ifeval/inst_strict_acc)`.

```python
import pandas as pd, numpy as np
df = pd.read_csv("<output_file>.csv")
J = "rm_gold_rm"
cats = [f"arena_hard/{J}/{c}/sc_score"
        for c in ("hard_prompt","coding","math","creative_writing")]
cats = [c for c in cats if c in df.columns]
assert np.allclose(df["arena_hard/aggregate/sc_score"], df[cats].mean(axis=1))
assert np.allclose(df["ifeval/aggregate/strict_acc"],
                   df[["ifeval/prompt_strict_acc","ifeval/inst_strict_acc"]].mean(axis=1))
```

**Selection summary (`<stem>_selected_summary.json`):**
- [ ] File exists; `selected_checkpoint` == argmax of `select/sibling_rm/mean`
      over the CSV rows.
- [ ] Every value in `metrics` matches that checkpoint's row in the CSV
      (arena per-category win_rate/sc_score for hard_prompt + creative_writing,
      `arena_hard/aggregate/sc_score`, the two ifeval strict accs +
      `ifeval/aggregate/strict_acc`, and `{secondary_rm,gold_rm}/{sc_score,
      win_rate_vs_chosen}`).
- [ ] `split` != `selection_split` (selection and reporting use different splits).

```python
import json, pandas as pd
s = json.load(open("<stem>_selected_summary.json"))
df = pd.read_csv("<output_file>.csv")
assert s["selected_checkpoint"] == int(df.loc[df["select/sibling_rm/mean"].idxmax(), "checkpoint"])
row = df[df["checkpoint"] == s["selected_checkpoint"]].iloc[0]
for k, v in s["metrics"].items():
    assert abs(row[k] - v) < 1e-9, k
```

**wandb run summary (cross-run comparison):**
- [ ] `selected/checkpoint`, `selected/select/sibling_rm/mean`, and one
      `selected/<metric>` per summary key appear in the run's summary (visible as
      runs-table columns; usable in a cross-run Bar Chart panel).

**Selector wiring:**
- [ ] `--benchmarks` containing `select` with `--sibling_rm_path` unset/`none`
      fails fast with a clear error (selection is a hard default).
- [ ] Sibling/training base mismatch fails fast: a `--sibling_rm_path` whose base
      model differs from `--training_rm_path` (different family or hidden
      size/layers/vocab) raises before any model loads. A same-base, different-seed
      sibling passes (prints `sibling/training base match OK`).
- [ ] `--only_ifeval` / `--only_arena_hard` / `--only_preference` drop `select`
      and the run completes without a selection summary (prints a skip note).
