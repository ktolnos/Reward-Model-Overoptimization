No KL - higher peak, KL - less reward hacking
ensembles are robust across settings, special mitigations tend to break at different models, learning rates, KL, etc.

### Baselines
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=61td52b4fsj)
- [PAR_vs_chosen_deepseek_v3_judge](https://wandb.ai/distill-llms/policy-evaluation/runs/gdxmbwvs)
- [RRMvsChosen_Deepseek_v3_judge](https://wandb.ai/distill-llms/policy-evaluation/runs/kpqxh5sc)
- [qwen06B_with_qwen8B-embedding_RM](https://wandb.ai/distill-llms/policy-evaluation/runs/4ryuk8oo) {0.6B}
- [AdvRM-step3-qwen06B_with_qwen8B-embedding_RM](https://wandb.ai/distill-llms/policy-evaluation/runs/tbiz9ief) {0.6B}
**Results**
All tested methods still reward hack.

### Q: If we have 100 RMs, what is the best way to use them: mean vs min vs UWO vs sequential-cycling?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=lrenmhanrue)
Same RM bank, same `mix_ensemble_size=10`, KL=0; only `ensemble_aggregation` differs (commit `0b800c62`).
- [__KL0_mix_100rms_mean_disjoint_10-mixens_1061682](https://wandb.ai/distill-llms/policy-evaluation/runs/znmynyme) {0.6B-Base-SFT} — mean disjoint 10x10
- [min_KL0_mix_100rms_min_disjoint_10-mixens_1061683](https://wandb.ai/distill-llms/policy-evaluation/runs/7rznh8wf) {0.6B-Base-SFT} — min disjoint 10x10
- [high-uwo_KL0_mix_100rms_uwo10_random_disjoint_10-mixens_1061681](https://wandb.ai/distill-llms/policy-evaluation/runs/w9rq5zie) {0.6B-Base-SFT} — UWO `λ=10`, `random_disjoint` 10x10 partition
- [40ens_KL0.005_ensemble_41rms_mean_1061686](https://wandb.ai/distill-llms/policy-evaluation/runs/klzsajm6) {0.6B-Base-SFT} — full ensemble of 41 RMs (More didn't fit in memory), mean
- [sequential_KL0_sequential3x_100rms_1061684](https://wandb.ai/distill-llms/policy-evaluation/runs/0ca6wi21) {0.6B-Base-SFT} — 100 RMs, sequential3x, β=0
- [seq_KL0.005_sequential3x_100rms_1061685](https://wandb.ai/distill-llms/policy-evaluation/runs/7kxkvl6b) {0.6B-Base-SFT} — sibling at β=0.005

**Results**
High-UWO and mean disjoint 10x10 had the highest gold peaks, sequential without KL has the highest secondary RM peak. Final performane is the best for sequential with small KL (0.005) with no significant reward hacking, 40-mean is second best final performance (same KL).
Overall across multiple experiments (see  40-RM ensemble: mean vs min aggregation), mean ensembles seem to outperform min. UWO is usually as good as mean and training reward std doesn't decrease much. Sequential is competitve with mean, but is cheaper to run.

### Q: 100-RM cycling once vs three times and training time (full helpsteer3 dataset, β=0.005)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=v0bg3f7dxr9)
- [1x_KL0.005_100rms_sequential1x_1062161](https://wandb.ai/distill-llms/policy-evaluation/runs/vc2f073s) {0.6B-Base-SFT} — 1× on subset
- [seq_KL0.005_sequential3x_100rms_1061685](https://wandb.ai/distill-llms/policy-evaluation/runs/7kxkvl6b) {0.6B-Base-SFT} - 3x on subset
- [full-ds_KL0.005_100rms_sequential3x_1062944](https://wandb.ai/distill-llms/policy-evaluation/runs/tdbu4tnq) {0.6B-Base-SFT} — 3x on full dataset

**Results**
First half is the same for 3x vs 1x, after 3x has both higher peak and less hacking.
Full-ds run shows that same recipe on full dataset (4x prompts) also doesn't reward hack according to gold rm, but secondary RM decreases after 20%.


### Q: First-pass mix-strategy granularity (group count × group size)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=h075lvaicnq)
β=0 for the 20×5 sibling vs β=0.005 for the others.
- [100_sequential_3x_0.005KL_1036533](https://wandb.ai/distill-llms/policy-evaluation/runs/jjp2nkwe) {0.6B-Base-SFT} — 3× sequential, equivalent of mix 1x100 RMs @ checkpoint-545
- [seqential_100_noKL_1034755](https://wandb.ai/distill-llms/policy-evaluation/runs/x0mzdkip) {0.6B-Base-SFT} — β=0
- [mix_2x50_0.005KL_1035193](https://wandb.ai/distill-llms/policy-evaluation/runs/fmbp1t3d) {0.6B-Base-SFT} — 2×50 — 400-series RMs @ ckpt-545; ensemble_aggregation="min"
- [mix_10x10min_disjoint_0.005KL_1036530](https://wandb.ai/distill-llms/policy-evaluation/runs/inyswtrh) {0.6B-Base-SFT} — 10×10 — 400-series RMs @ ckpt-545
- [mix_20x5min_noKL_1034756](https://wandb.ai/distill-llms/policy-evaluation/runs/5y3ihw7s) {0.6B-Base-SFT} — 20×5, β=0 — 400-series RMs @ ckpt-545
- [mix_mean_10x10_sliding_noKL_1022182](https://wandb.ai/distill-llms/policy-evaluation/runs/7oh7m4aa) {0.6B-Base-SFT} — 10×10 mean (compare aggregation too) — ⚠️ different RM bank: **100-series @ ckpt-218** (also β=0)

**Results**
2x50 wass the best: higher peak, higher final score.


### Q: How much do RMs matter? Are 10 best by eval accuracy out of 100 better? Does training for longer help?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=xw6deu30a07)
- [5ep-rm_10_sequential_3x_0.01KL_1030113](https://wandb.ai/distill-llms/policy-evaluation/runs/8vu591sl) {0.6B-Base-SFT} — sequential3x 5ep
- [5ep10_min_0.01KL_1030114](https://wandb.ai/distill-llms/policy-evaluation/runs/h6jrd6qs) {0.6B-Base-SFT} — min ensemble 5ep
- [sequential_10best_3x_0.01KL_1022174](https://wandb.ai/distill-llms/policy-evaluation/runs/45ch0pv2) {0.6B-Base-SFT} — top-10 best, sequential3x — picks from 100-series bank @ ckpt-218 (1ep)
- [sequential_10_3x_0.01KL_reorder_1022017](https://wandb.ai/distill-llms/policy-evaluation/runs/xcg07mww) {0.6B-Base-SFT} — 10 1ep RMs
- [new_10_seq-ens_3x_0.01KL_1021610](https://wandb.ai/distill-llms/policy-evaluation/runs/o8sfudyb) {0.6B-Base-SFT} - 10 out of 100 100-series RMs @ ckpt-218 (1ep)
- [10best_min_0.01KL_1022175](https://wandb.ai/distill-llms/policy-evaluation/runs/m0jkbkzv) {0.6B-Base-SFT} — top-10 best, min — 100-series bank @ ckpt-218

**Results**
Longer training helps: 5ep > 1ep
Selecting the top 10 best is not significantly better than random selection.
Min ensemble is worse than sequential (but RMs matter more than ensemble strategy).



### Q: Does sequentual ensemble training help for Qwen3.5-4B? Does it matter if we use checkpoints from the same run vs different seeds?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=utftn0w9oca)
- [same_seed_KL0_8rms_sequential3x_1129482](https://wandb.ai/distill-llms/policy-evaluation/runs/llxyplds) {3.5-4B-SFT} — 8 checkpoints of the same training run, 1 per epoch
- [3epRMs_KL0_10rms_sequential3x_1129495](https://wandb.ai/distill-llms/policy-evaluation/runs/b6e9si4j) {3.5-4B-SFT} — 10 RMs trained 3 epochs
- Baseline (1rm): [linear0.6-max1.5_KL0_1rms_sequential3x_1126524](https://wandb.ai/distill-llms/policy-evaluation/runs/4n3i20ph) {3.5-4B-SFT}

**Results:**
Ensembles didn't prevent reward hacking completely, the peak sc_score and other metrics are not significantly better than baseline. But the decline is much smaller and the final score on some metrics is close to the peak score, while on others it is still much better than baseline.

### Q: RM-size sweep on 0.6B sft_5ep_1060185 (single RM, sequential3x, β=0)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=blmq5wod2bt)
Same 0.6B-Base SFT policy (`sft_5ep_1060185`, `1060185/checkpoint-740`), same helpsteer3v2 GRPO dataset, same gold (Skywork-Reward-V2-Llama-3.1-8B)
- [4B-3128-nokl_KL0_1rms_sequential3x_1066782](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w) {0.6B-Base-SFT} — Qwen3-4B-Instruct-2507 RM (`1_BT_RM_…_1065302`, ckpt-3128)
- [qwen3.5-4b_KL0_1rms_sequential3x_1069471](https://wandb.ai/distill-llms/policy-evaluation/runs/s1fiv1uc) {0.6B-Base-SFT} — Qwen3.5-4B RM (`19_…_Qwen3.5-4B_len2048_fulltrain`)
- [Qwen3-8B_KL0_1rms_sequential3x_1069470](https://wandb.ai/distill-llms/policy-evaluation/runs/9okigr2v) {0.6B-Base-SFT} — Qwen3-8B RM (`19_BT_RM_…_1066933`)
- [0.6Bsft_3.5-9BRM_KL0_1rms_sequential3x_1072946](https://wandb.ai/distill-llms/policy-evaluation/runs/gguolgti) {0.6B-Base-SFT} — Qwen3.5-9B RM (`19_BT_RM_…_1069742`)
- [1x_KL0.005_100rms_sequential1x_1062161](https://wandb.ai/distill-llms/policy-evaluation/runs/vc2f073s) {0.6B-Base-SFT} <- 100 RMs 0.6B 600-series

**Results**
- Bigger RMs are better, except for Qwen3-8B. 
- Bigger RMs don't really reward hack and the performance is much better than 100 smaller reward models


### Best 0.6B run vs bigger policies:
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=alb17wmfvhq)
- [0.6Bsft_3.5-9BRM_KL0_1rms_sequential3x_1072946](https://wandb.ai/distill-llms/policy-evaluation/runs/gguolgti) {0.6B-Base-SFT} — Qwen3.5-9B RM (`19_BT_RM_…_1069742`)
- [1.7Bsft_KL0_1rms_sequential3x_1070738](https://wandb.ai/distill-llms/policy-evaluation/runs/1obc7wzc) {1.7B-Base-SFT} — 1.7B SFT
- [4Bsft-4BInstructRM_KL0_1rms_sequential3x_1071882](https://wandb.ai/distill-llms/policy-evaluation/runs/r5oubozm) {3-4B-Base-SFT} — 4B policy, 4B-Instruct RM
- [8B-LoRA-higherLR_KL0_1rms_sequential3x_1074580](https://wandb.ai/distill-llms/policy-evaluation/runs/5irdcj8d) {8B-Base} — LoRA, higher lr; 8B LoRA policy with 4B RM
- [3.5-4B-both_KL0_1rms_sequential3x_1078525](https://wandb.ai/distill-llms/policy-evaluation/runs/592ug9hu) {3.5-4B-SFT}

**Results**
When scaling up both we can achieve much better results and it doesn't cost crazy amounts of compute time. Also 3.5 reward hacks by default which makes it a convenient study subject.


### SFT across model sizes
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=soi2m4bsj8g)
- [sft-qwen3.5-4B](https://wandb.ai/distill-llms/policy-evaluation/runs/vnyf7yl0) {3.5-4B-SFT}
- [sft_default_1089122](https://wandb.ai/distill-llms/policy-evaluation/runs/l4cc8kam) {3.5-4B-SFT}
- [sft_4B-Base_1070739](https://wandb.ai/distill-llms/policy-evaluation/runs/rj0a2f7p) {3-4B-Base-SFT}
- [sft_1.7B_1070705](https://wandb.ai/distill-llms/policy-evaluation/runs/upl84n76) {1.7B-Base-SFT}
- [sft_5ep_1060185](https://wandb.ai/distill-llms/policy-evaluation/runs/81fp3ez6) {0.6B-Base-SFT}
- [0.6Bsft_3.5-9BRM_KL0_1rms_sequential3x_1072946](https://wandb.ai/distill-llms/policy-evaluation/runs/gguolgti) {0.6B-Base-SFT} — best 0.6B GRPO run for comparison

**Results**
SFT model quality grows with size and recency, but even 0.6B model can outperform 4B SFT with RL.

### Win rate vs chosen can go up to 70-80% even when you only use the base model for both reward model and the policy.
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=awv127w497v)
- [qwen3.5-4B-official-human](https://wandb.ai/distill-llms/policy-evaluation/runs/urbm8ngr)
- [4B-Base_KL0_1rms_sequential3x_1089357](https://wandb.ai/distill-llms/policy-evaluation/runs/hlct4b7l) {3.5-4B-SFT}
- [group_4_max_penalty_2_KL0_1rms_sequential3x_1132733](https://wandb.ai/distill-llms/policy-evaluation/runs/oudkd74k) {3.5-4B-SFT}
- [chosen-human](https://wandb.ai/distill-llms/policy-evaluation/runs/xfsfhujo) {0.6B-Base-SFT}

### DPO vs GRPO
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=4s3juyk68ez)
- [dpo_sigmoid_KL0.01_0.01KL_1089542](https://wandb.ai/distill-llms/policy-evaluation/runs/ez0fbzp8) {3.5-4B-SFT} — sigmoid, β=0.01
- [grpo_5e-6lr_KL0_1rms_sequential3x_1087938](https://wandb.ai/distill-llms/policy-evaluation/runs/dsknetrl) {3.5-4B-SFT} — 5e-6

**Results**
GRPO can achieve much higher performance.

### Sequential ensembles for 3.5-4B
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=utftn0w9oca)
- [same_seed_KL0_8rms_sequential3x_1129482](https://wandb.ai/distill-llms/policy-evaluation/runs/llxyplds) {3.5-4B-SFT} — 8 checkpoints of the same training run, 1 per epoch
- [3epRMs_KL0_10rms_sequential3x_1129495](https://wandb.ai/distill-llms/policy-evaluation/runs/b6e9si4j) {3.5-4B-SFT} — 10 RMs trained 3 epochs
- Baseline (1rm): [linear0.6-max1.5_KL0_1rms_sequential3x_1126524](https://wandb.ai/distill-llms/policy-evaluation/runs/4n3i20ph) {3.5-4B-SFT}

**Results:**
Ensembles didn't prevent reward hacking completely, the peak sc_score and other metrics are not significantly better than baseline. But the decline is much smaller and the final score on some metrics is close to the peak score, while on others it is still much better than baseline.


### Effect of learning rate
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=j7duxyvm2jn)
- [4B-3128-nokl_KL0_1rms_sequential3x_1066782](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w) {0.6B-Base-SFT}
- [double_lr_2e-5_KL0_1rms_sequential3x_1069741](https://wandb.ai/distill-llms/policy-evaluation/runs/01vf60lx) {0.6B-Base-SFT}
- [4B-Base_KL0_1rms_sequential3x_1089357](https://wandb.ai/distill-llms/policy-evaluation/runs/hlct4b7l) {3.5-4B-SFT}
- [1e-5lr_KL0_1rms_sequential3x_1099071](https://wandb.ai/distill-llms/policy-evaluation/runs/872sf4xt) {3.5-4B-SFT}

### RM training time
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=p1e86bogb5h)
- [4B-500_KL0.01_1rms_sequential3x_1066641](https://wandb.ai/distill-llms/policy-evaluation/runs/svl94js5) {0.6B-Base-SFT}
- [4B-3128_KL0.01_1rms_sequential3x_1066642](https://wandb.ai/distill-llms/policy-evaluation/runs/67wp3l1r) {0.6B-Base-SFT}
- [06b-hs3gold-2_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/jt7pn8us) {0.6B-Base}
- [06b-hs3gold-5_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/2vr6tb6c) {0.6B-Base}
- [06b-hs3gold-10_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/pes8ntrx) {0.6B-Base}


### Q: How should you pick the best checkpoint — training reward vs same-seed RM vs sibling-seed RM vs different-family RM vs random?
Code, scores, and plots: `experiments/checkpoint_selection/`. Method: deterministically partition the helpsteer3-qwen35 `test` split in two by `sha256(prompt) % 2` → `test_A` (selection slice) and `test_B` (gold-verdict slice). Re-score every preserved (checkpoint, prompt) policy response with four RMs and evaluate each selection strategy on its gold-RM (`Skywork/Skywork-Reward-V2-Llama-3.1-8B`) score on `test_B`. 8 GRPO runs trained against RM_19 (`19_Qwen3.5-4B-Base_…helpsteer3-qwen35_annotated_human`, ckpt-3144), ~20 checkpoints/run, restricted to the 25pct.test intersection (~367 prompts).
- [dapo0.5-max1.5_KL0_1rms_sequential3x_1099677](https://wandb.ai/distill-llms/policy-evaluation/runs/6c0fi2ay) {3.5-4B-SFT}
- [0.6DAPO_squared_max1_KL0_1rms_sequential3x_1109219](https://wandb.ai/distill-llms/policy-evaluation/runs/b6vnyx03) {3.5-4B-SFT}
- [0.6DAPO_linear_max1_KL0_1rms_sequential3x_1109220](https://wandb.ai/distill-llms/policy-evaluation/runs/jkm4l66t) {3.5-4B-SFT}
- [linear0.6-max1.5_KL0_1rms_sequential3x_1126524](https://wandb.ai/distill-llms/policy-evaluation/runs/4n3i20ph) {3.5-4B-SFT}
- [0.6DAPO_max4_mask_KL0_1rms_sequential3x_1131216](https://wandb.ai/distill-llms/policy-evaluation/runs/8qe50glw) {3.5-4B-SFT}
- [full_ds_max1024_KL0_1rms_sequential3x_1132879](https://wandb.ai/distill-llms/policy-evaluation/runs/azmleb76) {3.5-4B-SFT}
- [full_ds_5e-6lr_KL0_1rms_sequential3x_1136572](https://wandb.ai/distill-llms/policy-evaluation/runs/s21qpjll) {3.5-4B-SFT}
- [same_seed_KL0_8rms_sequential3x_1129482](https://wandb.ai/distill-llms/policy-evaluation/runs/llxyplds) {3.5-4B-SFT}

Selectors (all scored on `test_A`): training_rm = RM_19 itself (ckpt-3144), sibling_rm = RM_20 (`20_Qwen3.5-4B-Base_…ckpt-1179`, sibling seed, 3-epoch), secondary_rm = `Ray2333/GRM-Gemma-2B-sftreg` (different family). Oracles: gold_at_A = argmax gold on `test_A` (diagnostic ceiling), gold_at_B = argmax gold on `test_B` (verdict oracle, regret = 0 by construction).

Mean gold@`test_B` across runs (regret vs gold@B oracle in brackets), oracle ≈ 20.65:
- gold_at_A (oracle on selection slice): **20.49** (regret 0.16, 95% boot CI [0.00, 0.34])
- **secondary_rm @ test_A**: **19.96** (regret 0.69, CI [0.09, 1.65])
- **sibling_rm @ test_A**: **19.91** (regret 0.74, CI [0.02, 2.11])
- first saved checkpoint: 16.30 (regret 4.35, CI [2.70, 5.93])
- train_reward (wandb `rewards/batch_mean`): 16.30 (regret 4.35) — degenerate, see below
- random (mean over 1000 seeds/run): 11.88 (regret 8.77, CI [5.22, 12.62])
- training_rm @ test_A (RM_19 itself): 11.67 (regret 8.98, CI [3.04, 15.75])
- last checkpoint: 7.49 (regret 13.16, CI [6.01, 21.05])

**Results**
- A **held-out RM picks essentially the oracle checkpoint**: secondary recovers 96.7 % of the gold range above `first`, sibling recovers 96.5 %. It does not matter much whether the held-out scorer is a sibling-seed RM (same architecture, same dataset) or a totally different family (Gemma-2B).
- Using the **training RM (RM_19) as the selector is roughly as bad as random** — it systematically picks late checkpoints where reward hacking has driven gold strongly negative (regret 26 on run 2, 23 on run 4, 32 on run 5, last checkpoint gold near −9).
- `first` is a surprisingly decent cheap baseline (~79 % of oracle) because reward hacking has not yet taken hold at the earliest saved checkpoint.
- `last` is the worst named strategy — reward hacking is the default outcome of running GRPO to the end of training.
- The gap between `gold_at_A` and `gold_at_B` (regret 0.16) is small, so the half-half partition is not introducing meaningful sample noise — deployable strategies are genuinely near oracle, not just lucky on the verdict slice.
- Wandb `rewards/batch_mean` is **degenerate as a selection signal**: the strategy picks the first saved checkpoint in 7 / 8 runs (identical mean to `first`). Two reasons compound — GRPO runs with `--rm_subtract_mean_reward_per_model=True`, which centres rewards per batch and removes the absolute level, and wandb `_step` is logged at a much finer granularity than `save_steps`, so the alignment in `_align_train_reward` collapses every checkpoint of a run to ≈ the same logged value. A useful training-time signal would require either un-centred per-batch rewards or training-time evaluation on a held-out slice (which GRPO does not log here).
- **Cross-run argmax**: the secondary RM picks the global oracle — both `gold_at_B` and `secondary_rm @ test_A` pick `(run 5 = 0.6DAPO_max4_mask_…1131216, ckpt 745) → 22.29`. The cross-run training-RM pick is catastrophic: `(run 4, ckpt 2974) → −1.69`.
- **Cross-run top-k** (rank of the gold-oracle pair in each signal's ordering of all 160 (run, ckpt) candidates, and mean regret if you'd drawn uniformly from the top-k of that signal):
  - `secondary_rm`: oracle at **rank 1**; top-1 regret 0.00, top-5 mean 2.58 (worst 5.61).
  - `sibling_rm`: oracle at **rank 22**; top-1 regret 1.46, top-5 mean **1.47** (worst 1.92).
  - `gold_at_A` (diagnostic): oracle at rank 2; top-1 regret 1.25, top-5 mean 2.06 (worst 3.96).
  - `training_rm`: oracle at rank 53; top-1 regret 23.98, top-5 mean 20.28 (worst 24.52).
  - `train_reward`: oracle at rank 108; top-1 regret 8.36, top-5 mean 4.62 (worst 9.31).
- Two non-obvious things in the top-k table: (a) **`sibling_rm` is more robust than `secondary_rm` for shortlist-style selection** — its top-5 mean regret is lower (1.47 vs 2.58) and its worst-in-top-5 is much lower (1.92 vs 5.61), even though `secondary_rm` has the better top-1. Sibling's runner-up cluster is tight; secondary's drops off fast after the oracle. So if you'd commit to top-k → random-pick instead of top-1, prefer sibling. (b) **`training_rm`'s failure isn't isolated to its top-1**: the entire upper tail of its ranking is dominated by hacked checkpoints — the worst of its top-20 is regret 24.52, no better than its top-1.

Bottom line: at GRPO time, save a per-checkpoint score from any reward model that was not used as the training objective (sibling-seed RM or any different-family RM) and pick its argmax. Do not rely on the training RM itself or on `train/reward` as currently logged.