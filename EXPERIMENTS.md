## GRPO with reward-model aggregation strategies

### Q: Does sequentual ensemble training help for Qwen3.5-4B? Does it matter if we use checkpoints from the same run vs different seeds?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=utftn0w9oca)
- [same_seed_KL0_8rms_sequential3x_1129482](https://wandb.ai/distill-llms/policy-evaluation/runs/llxyplds) {3.5-4B-SFT} — 8 checkpoints of the same training run, 1 per epoch
- [3epRMs_KL0_10rms_sequential3x_1129495](https://wandb.ai/distill-llms/policy-evaluation/runs/b6e9si4j) {3.5-4B-SFT} — 10 RMs trained 3 epochs
- Baseline (1rm): [linear0.6-max1.5_KL0_1rms_sequential3x_1126524](https://wandb.ai/distill-llms/policy-evaluation/runs/4n3i20ph) {3.5-4B-SFT}

**Results:**
Ensembles didn't prevent reward hacking completely, the peak sc_score and other metrics are not significantly better than baseline. But the decline is much smaller and the final score on some metrics is close to the peak score, while on others it is still much better than baseline.

### Experiments with different length penalties
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=1jajc7m2wav)
- [0.6DAPO_linear_max1_KL0_1rms_sequential3x_1109220](https://wandb.ai/distill-llms/policy-evaluation/runs/jkm4l66t) {3.5-4B-SFT} — linear (`power=1`)
- [0.6DAPO_squared_max1_KL0_1rms_sequential3x_1109219](https://wandb.ai/distill-llms/policy-evaluation/runs/b6vnyx03) {3.5-4B-SFT} — squared (`power=2`); same `soft_fraction=0.6, max_penalty=1`
- [dapo0.5-max1.5_KL0_1rms_sequential3x_1099677](https://wandb.ai/distill-llms/policy-evaluation/runs/v36lynn9) {3.5-4B-SFT} — `soft_fraction=0.5, max_penalty=1.5, linear`
- [0.6DAPO_linear_max1_KL0_1rms_sequential3x_1109220](https://wandb.ai/distill-llms/policy-evaluation/runs/jkm4l66t) {3.5-4B-SFT} — `soft_fraction=0.6, max_penalty=1, linear`
GR3 (commit `9a2dafe4`) divides reward by `(1 + α·len/mean_len)` (sign-aware); DAPO (commit `a0c5c9c3`) subtracts a soft penalty.
- [gr3_KL0_1rms_sequential3x_1099136](https://wandb.ai/distill-llms/policy-evaluation/runs/ttm7baxh) {3.5-4B-SFT} — GR3 α=0.5
- [dapo0.5-max1.5_KL0_1rms_sequential3x_1099677](https://wandb.ai/distill-llms/policy-evaluation/runs/v36lynn9) {3.5-4B-SFT} — soft DAPO penalty

**Results:**
- Fancy length penalties do not help.
- What we really need is to keep the mean length close to sft policy mean length while avoiding overlarge penalties that can destablize training. 
- DAPO (linear) works fine for that. 


### GRPO learning-rate sweep on 4B-Base SFT (single-RM, sequential3x, KL=0)
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=jhy3090mh6v)
- [1e-5lr_KL0_1rms_sequential3x_1099071](https://wandb.ai/distill-llms/policy-evaluation/runs/872sf4xt) {3.5-4B-SFT} — 1e-5
- [grpo_5e-6lr_KL0_1rms_sequential3x_1087938](https://wandb.ai/distill-llms/policy-evaluation/runs/dsknetrl) {3.5-4B-SFT} — 5e-6
Also LR sweep on 0.6B-Base SFT:
- [double_lr_2e-5_KL0_1rms_sequential3x_1069741](https://wandb.ai/distill-llms/policy-evaluation/runs/01vf60lx) {0.6B-Base-SFT} — 2e-5
- [double_lr(5e-4)_4B-3128_KL0_1rms_sequential3x_1069473](https://wandb.ai/distill-llms/policy-evaluation/runs/41p251sw) — 5e-4 (extreme; ran by mistake)

**Results:**
1e-5 works best, smaller doesn't reach same peak, larger diverges.

### Q: Does β > 0 prevent over-optimization with a single-RM 4B-Base policy?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=jv59wevq0a2)
Same 4B-3128 RM, sequential3x, lr 1e-5; only β changes.
- [4B-3128-nokl_KL0_1rms_sequential3x_1066782](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w) {0.6B-Base-SFT} — β=0
- [4B-3128_KL0.005_1rms_sequential3x_1066783](https://wandb.ai/distill-llms/policy-evaluation/runs/40gfblbn) {0.6B-Base-SFT} — β=0.005
- [4B-3128_KL0.01_1rms_sequential3x_1066642](https://wandb.ai/distill-llms/policy-evaluation/runs/67wp3l1r) {0.6B-Base-SFT} — β=0.01

**Results**
All 3 didn't reward hack, no KL achieved highest reward.

### Q: How much RM training is enough? 500 vs 3128 steps at the same β?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=hwdz065xegu)
- [4B-500_KL0.01_1rms_sequential3x_1066641](https://wandb.ai/distill-llms/policy-evaluation/runs/svl94js5) {0.6B-Base-SFT} — 500-step RM, β=0.01
- [4B-3128_KL0.01_1rms_sequential3x_1066642](https://wandb.ai/distill-llms/policy-evaluation/runs/67wp3l1r) {0.6B-Base-SFT} — 3128-step RM, β=0.01
- [500-nokl_KL0_1rms_sequential3x_1066781](https://wandb.ai/distill-llms/policy-evaluation/runs/2fezn8o6) {0.6B-Base-SFT} — 500-step RM, β=0
- [4B-3128-nokl_KL0_1rms_sequential3x_1066782](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w) {0.6B-Base-SFT} — 3128-step RM, β=0

**Results**
Longer-trained RM achieved slightly higher final score, but the results are close and there is no significant evidence of reward hacking. No KL was much better than 0.01

### Q: 1 RM vs 7-RM sequential3x at 4B-Base (with and without KL)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=loscjtuwv9j)
- [4B-3128-nokl_KL0_1rms_sequential3x_1066782](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w) {0.6B-Base-SFT} — 1 RM, β=0
- [4B-nokl_KL0_7rms_sequential3x_1066780](https://wandb.ai/distill-llms/policy-evaluation/runs/sor7ytzf) {0.6B-Base-SFT} — 7 RMs, β=0
- [4B-3128_KL0.01_1rms_sequential3x_1066642](https://wandb.ai/distill-llms/policy-evaluation/runs/67wp3l1r) {0.6B-Base-SFT} — 1 RM, β=0.01
- [4B-sequential-same-run_KL0.01_7rms_sequential3x_1066643](https://wandb.ai/distill-llms/policy-evaluation/runs/89v3h28w) {0.6B-Base-SFT} — 7 RMs, β=0.01

⚠️ **The "7 RMs" are 7 sequential checkpoints (steps 500, 1000, 1500, 2000, 2500, 3000, 3128) of one RM training run** — not 7 independently-trained RMs. The 1-RM bullets use the final checkpoint-3128 only. So the comparison is single-RM-final vs cycling-through-its-training-trajectory.

**Results**
With KL=0.01, 7-RM training is slightly worse, with no KL slightly better, but this is likely noise.


### Q: 100-RM cycling once vs three times and training time (full helpsteer3 dataset, β=0.005)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=v0bg3f7dxr9)
- [1x_KL0.005_100rms_sequential1x_1062161](https://wandb.ai/distill-llms/policy-evaluation/runs/vc2f073s) {0.6B-Base-SFT} — 1× on subset
- [seq_KL0.005_sequential3x_100rms_1061685](https://wandb.ai/distill-llms/policy-evaluation/runs/7kxkvl6b) {0.6B-Base-SFT} - 3x on subset
- [full-ds_KL0.005_100rms_sequential3x_1062944](https://wandb.ai/distill-llms/policy-evaluation/runs/tdbu4tnq) {0.6B-Base-SFT} — 3x on full dataset

**Results**
First half is the same for 3x vs 1x, after 3x has both higher peak and less hacking.
Full-ds run shows that same recipe on full dataset (4x prompts) also doesn't reward hack according to gold rm, but secondary RM decreases after 20%.

### Related older exp: 100-RM cycling 1× vs 3× (subset dataset, β=0.005)
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=sgu41rir272)
- [100_sequential_1x_0.005KL_1036532](https://wandb.ai/distill-llms/policy-evaluation/runs/tiguq77k) {0.6B-Base-SFT} — 1×
- [100_sequential_3x_0.005KL_1036533](https://wandb.ai/distill-llms/policy-evaluation/runs/jjp2nkwe) {0.6B-Base-SFT} — 3×

Here 1x is significantly better, but mainly because 3x didn't learn much (KL is much lower).

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

### Q: Does `clip_reward_max=3` help in mix-UWO at 100 RMs?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=f8ofgsn3l2g)
- [1e-5lr_KL0_mix_100rms_uwo_disjoint_10-mixens_1061176](https://wandb.ai/distill-llms/policy-evaluation/runs/qryzhkab) {0.6B-Base-SFT} — no clipping
- [clip3_KL0_mix_100rms_uwo_disjoint_10-mixens_1061189](https://wandb.ai/distill-llms/policy-evaluation/runs/j9w6q8f1) {0.6B-Base-SFT} — `clip_reward_max=3.0` (otherwise identical)
**Results**
No significant impact.

### Q: Effect of disabling `rm_scale_reward_by_std_per_model` on a 10-RM mean ensemble?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=1dv2b25s9jz)
- [reproduce_KL0_ensemble_10rms_mean_1061173](https://wandb.ai/distill-llms/policy-evaluation/runs/sj2i5ui5) {0.6B-Base-SFT} — with std-scaling (baseline)
- [no-scale-std_KL0_ensemble_10rms_mean_1061633](https://wandb.ai/distill-llms/policy-evaluation/runs/gacum1jv) {0.6B-Base-SFT} — std-scaling disabled
**Results**
No significant impact.

### Q: 10×10 disjoint mean vs UWO?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=mm81ba73a2l)
- [10x10mean_disjoint__reorder_parallel_v2_0.005K_1053397](https://wandb.ai/distill-llms/policy-evaluation/runs/s2ac81kp) {0.6B-Base-SFT} — mean
- [10x10uwo1_disjoint_0.005KL_1053530](https://wandb.ai/distill-llms/policy-evaluation/runs/p2a4yp6a) {0.6B-Base-SFT} — UWO λ=1
**Results**
Mean is significantly better: higher peak, no reward hacking, higher final score.

### Q: KL coefficient on a 10-RM mean ensemble (lr=1e-5)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=nygzxwg8kqj)
- [1e-5lr_nokl_KL0_ensemble_10rms_mean_1061140](https://wandb.ai/distill-llms/policy-evaluation/runs/fvah0qui) {0.6B-Base-SFT} — β=0
- [1e-5lr_KL0.005_ensemble_10rms_mean_1061147](https://wandb.ai/distill-llms/policy-evaluation/runs/3a0xzb3j) {0.6B-Base-SFT} — β=0.005
**Results**
Gold rm score for no KL has higher peak, but lower final score. Slight reward hacking.
Secondary rm score also shows higher peak and hacking for no KL, but final performance is still slightly better.

### Q: 40-RM ensemble: mean vs min aggregation?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=2e297sbdytl)
- [40_mean_0.005KL_1036535](https://wandb.ai/distill-llms/policy-evaluation/runs/zxsedto1) {0.6B-Base-SFT} — mean — 400-series RMs @ ckpt-545
- [40_min_0.005KL_1036534](https://wandb.ai/distill-llms/policy-evaluation/runs/1tzct6dg) {0.6B-Base-SFT} — min — 400-series RMs @ ckpt-545
- [40_minens_noKL_1021611](https://wandb.ai/distill-llms/policy-evaluation/runs/yponcm2z) {0.6B-Base-SFT} — min, β=0 (no-KL sibling) — ⚠️ different RM bank: **100-series @ ckpt-218** (not directly comparable)

**Results**
Mean is significantly better: higher peak, higher final score. Maybe a bit of reward hacking in the end, but might be noise. No KL one was much worse.

### Q: Mix-strategy partition: disjoint vs sliding?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=hm3osfp1al6)
Same `mix_ensemble_size=10`, min aggregation, β=0.005.
- [mix_10x10min_disjoint_0.005KL_1036530](https://wandb.ai/distill-llms/policy-evaluation/runs/inyswtrh) {0.6B-Base-SFT} — disjoint
- [mix_10x10min_sliding_0.005KL_1036531](https://wandb.ai/distill-llms/policy-evaluation/runs/zedzcoll) {0.6B-Base-SFT} — sliding (overlapping)

**Results**
Disjoint is significantly better: higher peak, less reward hacking even at higher KL.

### Q: First-pass mix-strategy granularity (group count × group size)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=h075lvaicnq)
β=0 for the 20×5 sibling vs β=0.005 for the others.
- [mix_2x50_0.005KL_1035193](https://wandb.ai/distill-llms/policy-evaluation/runs/fmbp1t3d) {0.6B-Base-SFT} — 2×50 — 400-series RMs @ ckpt-545
- [mix_10x10min_disjoint_0.005KL_1036530](https://wandb.ai/distill-llms/policy-evaluation/runs/inyswtrh) {0.6B-Base-SFT} — 10×10 — 400-series RMs @ ckpt-545
- [mix_20x5min_noKL_1034756](https://wandb.ai/distill-llms/policy-evaluation/runs/5y3ihw7s) {0.6B-Base-SFT} — 20×5, β=0 — 400-series RMs @ ckpt-545
- [mix_mean_10x10_sliding_noKL_1022182](https://wandb.ai/distill-llms/policy-evaluation/runs/7oh7m4aa) {0.6B-Base-SFT} — 10×10 mean (compare aggregation too) — ⚠️ different RM bank: **100-series @ ckpt-218** (also β=0)

**Results**
2x50 wass the best: higher peak, higher final score.

### Q: KL sweep with 100-RM sequential cycling?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=uu8z107kome)
- [seqential_100_noKL_1034755](https://wandb.ai/distill-llms/policy-evaluation/runs/x0mzdkip) {0.6B-Base-SFT} — β=0
- [seqential_100_0.01KL_1033650](https://wandb.ai/distill-llms/policy-evaluation/runs/hcfknonk) {0.6B-Base-SFT} — β=0.01

**Results**
No KL has higher peak, with KL we see steady improvement (no hacking).

### Q: Sequential cycling at lower RM-count budgets (β=0.001)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=e73thhxcjzi)
- [50_sequential_0.001KL_1020076](https://wandb.ai/distill-llms/policy-evaluation/runs/or65prnn) {0.6B-Base-SFT} — 50 RMs
- [25_sequential_0.001KL_1020177](https://wandb.ai/distill-llms/policy-evaluation/runs/84v20h95) {0.6B-Base-SFT} — 25 RMs
**Results**
50 is better but both reward hack.

### Q: 10-RM (5-epoch each) — sequential3x vs min ensemble at β=0.01?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=kxttz8uolk2)
- [5ep-rm_10_sequential_3x_0.01KL_1030113](https://wandb.ai/distill-llms/policy-evaluation/runs/8vu591sl) {0.6B-Base-SFT} — sequential3x
- [5ep10_min_0.01KL_1030114](https://wandb.ai/distill-llms/policy-evaluation/runs/h6jrd6qs) {0.6B-Base-SFT} — min ensemble
**Results**
sequential is much better

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

### Q: 10× 2-epoch RMs — sequential vs mean vs min ensemble?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=17epi4r4q0m)
Same RM bank, no name-encoded KL change.
- [10_2ep-rm_sequential](https://wandb.ai/distill-llms/policy-evaluation/runs/63h6z2q7) {0.6B-Base} — sequential
- [10_2ep-rm_mean_ensemble](https://wandb.ai/distill-llms/policy-evaluation/runs/pxh5rt65) {0.6B-Base} — mean
- [10_2ep-rm_min_ensemble](https://wandb.ai/distill-llms/policy-evaluation/runs/2p3nnuyk) {0.6B-Base} — min
- [best-of-10-2ep_rm](https://wandb.ai/distill-llms/policy-evaluation/runs/djvfgf0b) {0.6B-Base} — single best-of-10 RM (only run with the original hard `-1` no-EOS penalty from `c46b9ad8`)

**Results**
Min is the only one avoiding reward hacking. Best and mean have higher peak at the very first evaluated checkpoint.

### Effect of lowering LR
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=690apd28mc1)
- [1rm_half-lr(1e-6)](https://wandb.ai/distill-llms/policy-evaluation/runs/33qbe6wz) {0.6B-Base} — 1 RM
- [1e-6-lr_5best_min-ens](https://wandb.ai/distill-llms/policy-evaluation/runs/jrizwomb) {0.6B-Base} — top-5 best min ensemble at the same lr
- [10_2ep-rm_mean_ensemble](https://wandb.ai/distill-llms/policy-evaluation/runs/pxh5rt65) {0.6B-Base} — mean ens at higher LR (prev best)
- [best-of-10-2ep_rm](https://wandb.ai/distill-llms/policy-evaluation/runs/djvfgf0b) {0.6B-Base} - best rm at lower lr

### Q: KL coefficient sweep with single-RM, ~17-RM sequential, around the GRPO bring-up phase?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=evou9246rny)
- [0.08kl_1017910](https://wandb.ai/distill-llms/policy-evaluation/runs/wrkc85nm) {0.6B-Base-SFT} — KL=0.08
- [low-lr_low-temp_0.02-kl_17_1017909](https://wandb.ai/distill-llms/policy-evaluation/runs/obi9q1j6) {0.6B-Base-SFT} — KL=0.02 (also lower lr+temp, confounded)
- [grpo_new_17_kl0.04](https://wandb.ai/distill-llms/policy-evaluation/runs/8khfdnuv) {0.6B-Base-SFT} — KL=0.04
- [grpo_new_17_1017395](https://wandb.ai/distill-llms/policy-evaluation/runs/6vhc9o1s) {0.6B-Base-SFT} — defaults baseline

### Q: 10-RM ensemble: min vs 3-RM ensemble at β=0.01 (early-2026 bring-up)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=etagg5k5jpz)
- [minens-0.01kl_1018058](https://wandb.ai/distill-llms/policy-evaluation/runs/1ryimavw) {0.6B-Base-SFT} — 10 RMs min
- [3ens_0.01KL](https://wandb.ai/distill-llms/policy-evaluation/runs/9nkcp292) {0.6B-Base-SFT} — 3-RM ensemble
- [min_ens_no-kl_1017706](https://wandb.ai/distill-llms/policy-evaluation/runs/36a7jgv3) {0.6B-Base-SFT} — min ens, β=0

### Q: Where to put KL — against the SFT base or against the original Qwen-0.6B base?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=zecnumspoan)
Eval-side `kl_base_model_path` change (commit `e59a2353` *kl base policy* / `fea3ffaf` *eval KL to base*).
- [t1.0_beta0.04_1rm_KL2_qwen0.6B](https://wandb.ai/distill-llms/policy-evaluation/runs/3dpbxgyt) {0.6B-Base} — KL vs raw Qwen-0.6B
- [t1.0_beta0.04_1rm_KL2_base](https://wandb.ai/distill-llms/policy-evaluation/runs/6lxum6ip) {0.6B-Base} — KL vs SFT base

### Other / standalone
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=8dlwwl6lc6a)
- [25pct_5epRMs_KL0.005_ensemble_10rms_mean_1060956](https://wandb.ai/distill-llms/policy-evaluation/runs/i6negqgg) {0.6B-Base-SFT} — 5-epoch RMs on a 25% subset
- [10x10uwo_disjoint_subprecompmean_0.005KL_1053933](https://wandb.ai/distill-llms/policy-evaluation/runs/w769d1ww) {0.6B-Base-SFT} — UWO with precomputed-mean optimization
- [10x10mean_disjoint__reorder_parallel_0.005K_1047283](https://wandb.ai/distill-llms/policy-evaluation/runs/vkolkfga) {0.6B-Base-SFT} — eval reruns of one training run after eval-tokenization fixes (useful for measuring eval-side variance)
- [10x10mean_disjoint_datautils_refactor+precompute_0.005K_1053919](https://wandb.ai/distill-llms/policy-evaluation/runs/bxxocfcq) {0.6B-Base-SFT}
- [3_out_100_uwo1_rand-disj_3x_0.005KL_1053531](https://wandb.ai/distill-llms/policy-evaluation/runs/grpzra8l) {0.6B-Base-SFT} — random-disjoint sampling 3 RMs from a 100-pool, UWO λ=1, sequential3x
- [25x4mean_disjoint_0.005KL_1043248](https://wandb.ai/distill-llms/policy-evaluation/runs/oyveeskn) {0.6B-Base-SFT} (25 disjoint groups of 4) | [..._Qwen3-0.6B-Base_1043249](https://wandb.ai/distill-llms/policy-evaluation/runs/ccbmsxld) {0.6B-Base} (sibling on raw Base, not SFT)
- [other_40_mean_0.05KL_1038817](https://wandb.ai/distill-llms/policy-evaluation/runs/rbvos9s8) {0.6B-Base-SFT} — 40-RM mean at β=0.05 (10× usual KL) | [10x10mean_disjoint_0.005KL_1038818](https://wandb.ai/distill-llms/policy-evaluation/runs/o5325lt2) {0.6B-Base-SFT}
- [new_10_seq-ens_3x_0.01KL_1021610](https://wandb.ai/distill-llms/policy-evaluation/runs/o8sfudyb) {0.6B-Base-SFT} | [new_10_seq_ens_3x_noKL_1020207](https://wandb.ai/distill-llms/policy-evaluation/runs/38vulkjb) {0.6B-Base-SFT} | [25_min_ensemble_0KL_1020191](https://wandb.ai/distill-llms/policy-evaluation/runs/zdj21jo0) {0.6B-Base-SFT}
- [100_sequential_noKL_1019598](https://wandb.ai/distill-llms/policy-evaluation/runs/vzn2yyx1) {0.6B-Base-SFT} | [100rm_sequential_1019534](https://wandb.ai/distill-llms/policy-evaluation/runs/hb0fn0ms) {0.6B-Base-SFT} | [free_memory_sequential_10_x50_1019400](https://wandb.ai/distill-llms/policy-evaluation/runs/iv84ysdn) {0.6B-Base-SFT} | [sequential_10_3x_1018956_0.01KL](https://wandb.ai/distill-llms/policy-evaluation/runs/gma9k2le) {0.6B-Base-SFT}
- [grpo_new_sft_min_ens_0.02kl_1017002](https://wandb.ai/distill-llms/policy-evaluation/runs/sphfjynm) {0.6B-Base-SFT} — first run on the new SFT base + min-ensemble
- [grpo_sft1ep_0.02KL_1012751](https://wandb.ai/distill-llms/policy-evaluation/runs/lipss3j6) {0.6B-Base-SFT} — first GRPO from a 1-epoch SFT base
- [grpo_from_sft_kl0.02_1008081](https://wandb.ai/distill-llms/policy-evaluation/runs/pqfwpznz) {0.6B-Base-SFT} — first GRPO from the 20-epoch SFT
- [no-kl__1.0temp_1rm_1007307](https://wandb.ai/distill-llms/policy-evaluation/runs/guktvo5t) {0.6B-Base}
- [2ep_rm_over_maxlen](https://wandb.ai/distill-llms/policy-evaluation/runs/yhgxv5ck) {0.6B-Base}
- early helpsteer3 GRPO bring-up: [high_lr_continual_28RMepochs](https://wandb.ai/distill-llms/policy-evaluation/runs/ompxv93o) {0.6B-Base} (first pess-loss run), [_high_lr_1epRM_continual](https://wandb.ai/distill-llms/policy-evaluation/runs/5kg0iy2w) {0.6B-Base}, [high_lr_10epRM_10k](https://wandb.ai/distill-llms/policy-evaluation/runs/yfqboj8t) {0.6B-Base}, [high_lr_temp_hs3_40k](https://wandb.ai/distill-llms/policy-evaluation/runs/lyloj22g) {0.6B-Base}
- earliest 8B-embedding-RM GRPO experiments: [deepspeed_0.001](https://wandb.ai/distill-llms/policy-evaluation/runs/kxezp55r) {0.6B-Base}, [QRM_Llama8b_baseQwen06B](https://wandb.ai/distill-llms/policy-evaluation/runs/vxo0afaq) {0.6B-Base}, [lr1e-7_16_resp_32_batch_4096_replay](https://wandb.ai/distill-llms/policy-evaluation/runs/yibmhepk) {0.6B-Base}
- [qwen06B_with_qwen8B-embedding_RM](https://wandb.ai/distill-llms/policy-evaluation/runs/4ryuk8oo) {0.6B} — first GRPO with Qwen3-Embedding-8B as base RM
- earliest GRPO bring-up runs (different RM/gold pairs): [train_Ray_gold_QRM](https://wandb.ai/distill-llms/policy-evaluation/runs/nf9mi154), [min-ans_gold_URM_train_Ray](https://wandb.ai/distill-llms/policy-evaluation/runs/qbro76p8), [train_QRM_gold_URM](https://wandb.ai/distill-llms/policy-evaluation/runs/ke0p6by4), [train_Ray_gold_URM](https://wandb.ai/distill-llms/policy-evaluation/runs/kwuqpz84), [grpo_train_Qwen_gold_Ray](https://wandb.ai/distill-llms/policy-evaluation/runs/g3vhua57)
- [train_Qwen_gold_Ray](https://wandb.ai/distill-llms/policy-evaluation/runs/uz4vpxu9) — last PPO-era eval before the move to GRPO
- RM-side reward-scale ablation: [1ep_seed42_noscale](https://wandb.ai/distill-llms/policy-evaluation/runs/dx898w6h) {0.6B-Base} vs [1epRM_seed43_scale](https://wandb.ai/distill-llms/policy-evaluation/runs/nx2v1cvq) {0.6B-Base} (seed and scale-on/off both change — confounded)
- [1ep_normalized_mean_std](https://wandb.ai/distill-llms/policy-evaluation/runs/5v57qm0l) {0.6B-Base} — mean+std normalisation

### Q: Does training the RM longer (more samples) help if the policy is also scaled up to 40k?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=2o9vhix725r)
- [10ep_10k->40k_policy](https://wandb.ai/distill-llms/policy-evaluation/runs/1ofqxnvn) {0.6B-Base} — 10-ep RM on 10k samples → 40k-sample policy
- [1ep_40k_RM->40k_policy](https://wandb.ai/distill-llms/policy-evaluation/runs/pw96ou3m) {0.6B-Base} — 1-ep RM on 40k samples → 40k-sample policy

---

## Larger-policy bring-up (Qwen3.5-4B / 8B / 1.7B / LoRA)

### Q: Policy-size scaling at the same setup (single-RM, sequential3x, β=0)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=r88423bbfwt)
- [1.7Bsft_KL0_1rms_sequential3x_1070738](https://wandb.ai/distill-llms/policy-evaluation/runs/1obc7wzc) {1.7B-Base-SFT} — 1.7B SFT
- [4B-Base_KL0_1rms_sequential3x_1069463](https://wandb.ai/distill-llms/policy-evaluation/runs/1soo7lfo) — 4B Base — ⚠️ training run not in W&B cache; base couldn't be auto-verified
- [Qwen3-8B_KL0_1rms_sequential3x_1069470](https://wandb.ai/distill-llms/policy-evaluation/runs/9okigr2v) {0.6B-Base-SFT} — 8B (full-FT) — ⚠️ **the underlying GRPO run is actually a 0.6B model** (`hidden_size=1024`, 28 layers, 596M params); the "8B" in the name is incorrect



### Q: Cross-size pairing — small policy + big RM, or big policy + small RM?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=lxzxactixml)
Each row uses a different policy *base* — keep that in mind alongside the policy/RM size pairing.
- [0.6Bsft_3.5-9BRM_KL0_1rms_sequential3x_1072946](https://wandb.ai/distill-llms/policy-evaluation/runs/gguolgti) {0.6B-Base-SFT} — 0.6B policy, 3.5-9B RM
- [4Bsft-4BInstructRM_KL0_1rms_sequential3x_1071882](https://wandb.ai/distill-llms/policy-evaluation/runs/r5oubozm) {3-4B-Base-SFT} — 4B policy, 4B-Instruct RM
- [8B-LoRA-policy-4BInstructRM_KL0_1rms_sequential3x_1074475](https://wandb.ai/distill-llms/policy-evaluation/runs/v9qa5abk) {8B-Base} — 8B (LoRA) policy, 4B-Instruct RM
- [Qwen3-8B_KL0_1rms_sequential3x_1069470](https://wandb.ai/distill-llms/policy-evaluation/runs/9okigr2v) {0.6B-Base-SFT} -- 0.6B policy with 8B RM
- [8B-LoRA-higherLR_KL0_1rms_sequential3x_1074580](https://wandb.ai/distill-llms/policy-evaluation/runs/5irdcj8d) {8B-Base} — LoRA, higher lr; 8B LoRA policy with 4B RM
- [qwen3.5-4b_KL0_1rms_sequential3x_1069471](https://wandb.ai/distill-llms/policy-evaluation/runs/s1fiv1uc) {0.6B-Base-SFT} —  0.6B policy + Qwen3.5-4B RM

### Q: 4B-Base SFT — Instruct vs Non-Instruct as the policy?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=rxvlgjdfnyx)
Both training runs are missing from the W&B cache (likely deleted), so the auto-annotator cannot verify the policy bases — treat the labels in the run names as the source of truth.
- [4B-Base_KL0_1rms_sequential3x_1069463](https://wandb.ai/distill-llms/policy-evaluation/runs/1soo7lfo) — Instruct base
- [4B-NonInstruct_KL0_1rm_1068892](https://wandb.ai/distill-llms/policy-evaluation/runs/hn5q6u6m) — Non-Instruct (after `bf1ab59` *fix num_labels mismatch*)



### Model training bringup
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=wvt4ud9b8m0)
- [4B-Base_KL0_1rms_sequential3x_1089357](https://wandb.ai/distill-llms/policy-evaluation/runs/hlct4b7l) {3.5-4B-SFT} — first run on the new `1089122` SFT
- [3.5B-base-both-human_KL0_1rms_sequential3x_1084314](https://wandb.ai/distill-llms/policy-evaluation/runs/ywy5hz5o) {3.5-4B-SFT} — Switch to human preferences annotations from gold RM annotations
- [dpo_sigmoid_dpo_1088543](https://wandb.ai/distill-llms/policy-evaluation/runs/dsip2tv9) {3.5-4B-SFT} — first DPO run after the trainer was added

## DPO / APO offline preference optimization

The `0ec65f88` *dpo* commit introduced `rlhf/dpo/my_dpo.py` (TRL DPOTrainer wrapper) supporting `loss_type ∈ {sigmoid, apo_zero, …}`.

### Q: APO-zero vs sigmoid DPO on the same SFT base?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=3rxxg7c7o40)
- [dpo_sigmoid_4B-Base_1089353](https://wandb.ai/distill-llms/policy-evaluation/runs/6cyhd9z3) {3.5-4B-SFT} — sigmoid (default β)
- [apo_4B-Base_1089354](https://wandb.ai/distill-llms/policy-evaluation/runs/elv91xmk) {3.5-4B-SFT} — APO-zero (default β)
- [dpo_sigmoid_KL0.01_0.01KL_1089542](https://wandb.ai/distill-llms/policy-evaluation/runs/ez0fbzp8) {3.5-4B-SFT} — sigmoid, β=0.01
- [dpo_apo_zero_KL0.01_0.01KL_1089543](https://wandb.ai/distill-llms/policy-evaluation/runs/l9r5yqzj) {3.5-4B-SFT} — APO-zero, β=0.01

### Q: DPO β: 0.1 vs β=0.01?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=5bjbfdisut1)
- [dpo_sigmoid_4B-Base_1089353](https://wandb.ai/distill-llms/policy-evaluation/runs/6cyhd9z3) {3.5-4B-SFT} —  β=0.1
- [dpo_sigmoid_KL0.01_0.01KL_1089542](https://wandb.ai/distill-llms/policy-evaluation/runs/ez0fbzp8) {3.5-4B-SFT} — β=0.01
- (And same for APO: 1089354 vs 1089543)
---

## SFT base-policy bring-up

These produce the SFT checkpoints that all subsequent GRPO/DPO runs use as `kl_base_model_path` / starting point.

### Q: SFT epoch count: 5 vs 20?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=2lr3a3za39h)
- [sft_5ep_1060185](https://wandb.ai/distill-llms/policy-evaluation/runs/81fp3ez6) {0.6B-Base-SFT} — 5-epoch SFT
- [sft_20_epochs_1008548](https://wandb.ai/distill-llms/policy-evaluation/runs/6myafsv6) {0.6B-Base-SFT}
- [sft_sft_20ep_1008088](https://wandb.ai/distill-llms/policy-evaluation/runs/c0ypc7qe) {0.6B-Base-SFT} (sibling)

### Q: Effect of SFT bug fixes (BOS stripping, `add_special_tokens=True`)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=znxu1ajj5xo)
- [sft_bug_fixes_1058401](https://wandb.ai/distill-llms/policy-evaluation/runs/gcws1fuv) {0.6B-Base-SFT} — with fixes (`b41f52c9` + `0a8810d`)
- [sft_undo_sft_changes_1058521](https://wandb.ai/distill-llms/policy-evaluation/runs/uyrxxys7) {0.6B-Base-SFT} — same dataset/config but the fixes reverted (commit `7d178c30`)

### Q: bf16 SFT vs fp32?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=ume79plc9ha)
- [sft_Redo_bf16_1058899](https://wandb.ai/distill-llms/policy-evaluation/runs/r78m7srm) {0.6B-Base-SFT} — bf16 (commit `32c8a408` + the bf16-everywhere consistency in `b1e083e`)
- [sft_undo_sft_changes_1058521](https://wandb.ai/distill-llms/policy-evaluation/runs/uyrxxys7) {0.6B-Base-SFT} — fp32 sibling

### Q: SFT dataset size — 8k vs 10k vs full helpsteer3?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=zv1c5y6cl54)
- [sft_sft10k_1058332](https://wandb.ai/distill-llms/policy-evaluation/runs/4fnhjsap) {0.6B-Base-SFT} — 10k subset
- [8k_sft_new-eval](https://wandb.ai/distill-llms/policy-evaluation/runs/3qah4y8y) {0.6B-Base-SFT} — 8k subset
- [sft_Full_dataset_sft_1053934](https://wandb.ai/distill-llms/policy-evaluation/runs/er9mon61) {0.6B-Base-SFT} — full dataset

### Other / standalone
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=a6i54gbgn0f)
- [sft-qwen3.5-4B](https://wandb.ai/distill-llms/policy-evaluation/runs/vnyf7yl0) {3.5-4B-SFT} (Qwen3.5-4B SFT, the launch eval)
- [sft_default_1089122](https://wandb.ai/distill-llms/policy-evaluation/runs/l4cc8kam) {3.5-4B-SFT} — *the* SFT base used by the latest 4B GRPO/DPO experiments
- [sft_3.5-4B-Base-human_1082435](https://wandb.ai/distill-llms/policy-evaluation/runs/m1y7ovra) {3.5-4B-SFT} — Qwen3.5-4B-Base SFT'd on the human-annotated dataset
- [sft_4B-Base_1070739](https://wandb.ai/distill-llms/policy-evaluation/runs/rj0a2f7p) {3-4B-Base-SFT} (used by GRPO `1071882`, `1072946`)
- [sft_1.7B_1070705](https://wandb.ai/distill-llms/policy-evaluation/runs/upl84n76) {1.7B-Base-SFT}
- [sft_sft_validation_1060057](https://wandb.ai/distill-llms/policy-evaluation/runs/lw9en7yo) {0.6B-Base-SFT} — adds held-out validation split (`1c5f96d5`)
- [sft_new_sft_1057788](https://wandb.ai/distill-llms/policy-evaluation/runs/h56kyuxj) {0.6B-Base-SFT} — first SFT after the data-pipeline + RM-loading refactor (`21b850e`)
- [sft->Skywork/Skywork-Reward-V2-Qwen3-8B_1043250](https://wandb.ai/distill-llms/policy-evaluation/runs/089kkxkf) {0.6B-Base-SFT} — gold-RM family switched (V2-Llama → V2-Qwen)
- [sft_reward_texts_printing_1016814](https://wandb.ai/distill-llms/policy-evaluation/runs/2t79ziy9) {0.6B-Base-SFT} — long-lived KL base for ~56 GRPO runs | [sft_eval_new](https://wandb.ai/distill-llms/policy-evaluation/runs/t4m8a5q2) {0.6B-Base-SFT}
- [sft_sft_fixes_1007442](https://wandb.ai/distill-llms/policy-evaluation/runs/pafvlp9e) {0.6B-Base-SFT}

---

## HelpSteer3 gold-annotated re-base

### Q: How many epochs to train the BT RM (gold=Skywork-V2-Llama-8B)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=t5musbwnuwy)
- [06b-hs3gold-2_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/jt7pn8us) {0.6B-Base} — 2 ep
- [06b-hs3gold-5_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/2vr6tb6c) {0.6B-Base} — 5 ep
- [06b-hs3gold-10_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/pes8ntrx) {0.6B-Base} — 10 ep

### Q: RM-epochs sweep with the alternative gold (Skywork-V2-Qwen)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=svt8vax7vsz)
Same comparison as above but `gold_rm_name=Skywork-Reward-V2-Qwen3-8B` (eval-side change in `8af205f6` *eval continual_full with other gold*).
- [06b-hs3gold-2_rm_epochs_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/8yokcn09) {0.6B-Base}
- [06b-hs3gold-10_rm_epochs_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/gymns6vn) {0.6B-Base}

### Q: Gold-RM family bias — Skywork-V2-Llama vs Skywork-V2-Qwen as the *evaluator*?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=yvo33nnsxxj)
Same training; only the gold RM at eval time changes. (Pair these with the `_qwengold` suffix.)
- 2-ep RM: [06b-hs3gold-2_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/jt7pn8us) {0.6B-Base} vs [06b-hs3gold-2_rm_epochs_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/8yokcn09) {0.6B-Base}
- 10-ep RM: [06b-hs3gold-10_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/pes8ntrx) {0.6B-Base} vs [06b-hs3gold-10_rm_epochs_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/gymns6vn) {0.6B-Base}
- continual full-FT: [06b-hs3gold-continual_bt_rm_full](https://wandb.ai/distill-llms/policy-evaluation/runs/u2ehnxxv) {0.6B-Base} vs [06b-hs3gold-continual_bt_rm_full_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/leaf0g1g) {0.6B-Base}
- 8 RMs minens: [Qwen06B_helpsteer3_minensemble8](https://wandb.ai/distill-llms/policy-evaluation/runs/v8czdzix) {0.6B-Base} vs [Qwen06B_helpsteer3_minensemble8_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/22n7fcz3) {0.6B-Base}

### Q: Continual RM training — full-FT vs LoRA?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=tsotjxuqro8)
- [06b-hs3gold-continual_bt_rm_full](https://wandb.ai/distill-llms/policy-evaluation/runs/u2ehnxxv) {0.6B-Base} — full-FT
- [06b-hs3gold-continual_bt_rm_lora](https://wandb.ai/distill-llms/policy-evaluation/runs/pbquwi5p) {0.6B-Base} — LoRA

### Q: Single RM vs 8-RM min-ensemble (lr=5e-7)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=6nxcc9slldh)
- [lr5e-7_rmQwen06B_helpsteer3_gold](https://wandb.ai/distill-llms/policy-evaluation/runs/z3zpf0m2) {0.6B-Base} — 1 RM
- [lr5e-7_rmQwen06B_helpsteer3_gold_sequential](https://wandb.ai/distill-llms/policy-evaluation/runs/zn8goyls) {0.6B-Base} — 8 RMs sequential
- [Qwen06B_helpsteer3_minensemble8](https://wandb.ai/distill-llms/policy-evaluation/runs/v8czdzix) {0.6B-Base} — 8 RMs min-ensemble

### Q: `rm_switches_multiplier` 3× vs 50×?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=0qxr49vfq3s)
- [06b-hs3gold-rm_switches_multiplier3](https://wandb.ai/distill-llms/policy-evaluation/runs/f44rt09l) {0.6B-Base} — 3× cycling
- [06b-hs3gold-2_rm_epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/jt7pn8us) {0.6B-Base} — multiplier=50 (commit `cfb0533b`)

### Other / standalone
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=m07t8gpla7r)
- [hs3-10k_min-ens_corrected-subtract-mean](https://wandb.ai/distill-llms/policy-evaluation/runs/7vgocgy4) {0.6B-Base} — min-ens with corrected `rm_subtract_mean_reward_per_model` (`9836826`) | [hs3_40krm_q06base](https://wandb.ai/distill-llms/policy-evaluation/runs/gmokypsa) {0.6B-Base}
- [lr5e-7_rmQwen06B_helpsteer3_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/oo8fiitd) {0.6B-Base} — single RM at lr=5e-7, Qwen gold
- [Qwen06B_helpsteer3_minensemble8_qwengold](https://wandb.ai/distill-llms/policy-evaluation/runs/22n7fcz3) {0.6B-Base} — 8 minens, Qwen gold

---

## Pessimistic-loss / CQL / ReLU additions

`rlhf/grpo/online_pet.py` accepts `pessimistic_loss_weight`, `relu_chosen_reward_loss`, `cql_optimistic_loss_weight` (added in `13ba1b0` as `-mean(chosen_rewards)`). All extend GRPO with offline-RL-style regularization terms.

### Q: Pessimistic loss weight magnitude (after the gradient fix)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=4ur6s3w9bmd)
After `c0c6ec8` *pess loss gradient fix*:
- [0.001_pess_fix_3ep](https://wandb.ai/distill-llms/policy-evaluation/runs/8mllca4h) {0.6B-Base} — pess=0.001
- [pess0.1_baseline](https://wandb.ai/distill-llms/policy-evaluation/runs/yntmzx3a) {0.6B-Base} — pess=0.1
- [pess1M_3epochs](https://wandb.ai/distill-llms/policy-evaluation/runs/mvebzmsk) {0.6B-Base} — pess=1e6 (extreme)

### Q: Does adding `relu_chosen_reward_loss` on top of pess help?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=0gu072pqnec)
- [pess0.1_baseline](https://wandb.ai/distill-llms/policy-evaluation/runs/yntmzx3a) {0.6B-Base} — pess=0.1 only
- [relu0.01mean_pess0.1](https://wandb.ai/distill-llms/policy-evaluation/runs/ek1mcg73) {0.6B-Base} — pess=0.1 + relu=0.01 (mean baseline)
- [helpsteer3_pess0.1_relu](https://wandb.ai/distill-llms/policy-evaluation/runs/9gtjlie7) {0.6B-Base} — pess=0.1 + relu=0.1 (`relu_chosen_use_rejected_baseline=True`)

### Q: Does adding the CQL optimistic term on top of pess help, and what weight?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=l0j1v4si8jg)
After `13ba1b0` *cql optimistic loss*:
- [continual_cql_pess0.001](https://wandb.ai/distill-llms/policy-evaluation/runs/b7fpjldr) {0.6B-Base} — pess+cql = 0.001 each — RM `974219` @ ckpt-142
- [continual_cql_pess0.01](https://wandb.ai/distill-llms/policy-evaluation/runs/tjne9x5t) {0.6B-Base} — pess+cql = 0.01 each — RM `974219` @ ckpt-1420 (10× more RM training than 0.001 sibling)
- [continual_cql-0.005_1018060](https://wandb.ai/distill-llms/policy-evaluation/runs/95zrqtgk) {0.6B-Base-SFT} — pess+cql=0.005, β=0.01 — ⚠️ different RM (`995145` @ ckpt-284) **and** different policy base (SFT vs raw Base) than the other two; weight comparison is confounded

### Q: Pessimistic-batch / replay-buffer hyperparameters (from `acc60a1e` "1-e7 LR, 16 batch, 8 gens, 10k pessloss, 32 preference batch, 64 pess batch")?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=p7c2vi0u13y)
- [pess1k_batch32_16gen_512replay_1e-7](https://wandb.ai/distill-llms/policy-evaluation/runs/v5952gvl) {0.6B-Base} — pess=1k, batch=32, 16 gens, 512 replay
- [pess10k_batch32_16gen_512replay_1e-7](https://wandb.ai/distill-llms/policy-evaluation/runs/fdyqc227) {0.6B-Base} — pess=10k (10× higher), same other params
- [pess10k_batch16_8gen_256replay_1e-7_32pessbatch_16bt](https://wandb.ai/distill-llms/policy-evaluation/runs/e77wionk) {0.6B-Base} — half batch, half gens, half replay, separate pess (32) and BT (16) batch sizes

### Q: LoRA vs full-FT for the BT-RM half of the pess setup?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=zwvoax15o46)
- [pess0.1_baseline](https://wandb.ai/distill-llms/policy-evaluation/runs/yntmzx3a) {0.6B-Base} — full-FT RM
- [lora_pess0.1](https://wandb.ai/distill-llms/policy-evaluation/runs/ix1qdbtp) {0.6B-Base} — LoRA RM (commit `8a70f6b9` *lora for rm*)

---

## Online PET (preference-embedded training)

`rlhf/grpo/online_pet.py` updates the RM online during policy training, using either top-k or all-responses, optionally with a reference-policy regularizer.

### Q: Reference-policy regularization on or off?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=flz6krt7qee)
- [onlinePET-qwen06B-Base-06BRM](https://wandb.ai/distill-llms/policy-evaluation/runs/6mlwj81t) {0.6B-Base} — no ref
- [onlinePET-qwen06B-Base-06BRM_try2](https://wandb.ai/distill-llms/policy-evaluation/runs/oi2l9cg3) {0.6B-Base} — with ref (rerun after `a17dc96` *annotate reference 06b rm*)
- [onlinePET-qwen06B-Base-06BRM-ref-top1fromeach](https://wandb.ai/distill-llms/policy-evaluation/runs/roswl3ct) {0.6B-Base} — ref + top-1-per-prompt RM updates

### Q: Top-k sampling for the online RM update vs all responses?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=vh6mv5kswqc)
- [onlinePET-qwen06B-Base-06BRM-ref-top1fromeach](https://wandb.ai/distill-llms/policy-evaluation/runs/roswl3ct) {0.6B-Base} — top-1 from each prompt
- [onlinePET-qwen06B-Base-06BRM-noref-AdamW-all-responses-2ep1](https://wandb.ai/distill-llms/policy-evaluation/runs/6i70oe13) {0.6B-Base} — all responses

### Q: Online PET training-epoch sweep (no-ref, all-responses, AdamW)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=fv6nu526bv2)
- [onlinePET-...-1ep_pessloss_1000](https://wandb.ai/distill-llms/policy-evaluation/runs/g0pdq8d1) {0.6B-Base} (also pess=1000)
- [onlinePET-...-2ep](https://wandb.ai/distill-llms/policy-evaluation/runs/6i70oe13) {0.6B-Base} — 2-epoch siblings (commit lineage `6ac4f23c → 575e7c4e → 492a9c44 → de4266a5 → 10068130`)
- [onlinePET-...-4ep](https://wandb.ai/distill-llms/policy-evaluation/runs/r3aenooz) {0.6B-Base} — 4 epochs

### Q: With vs without pessloss in Online PET (1-epoch)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=anl8zimj62v)
- [onlinePET-...-1ep_pessloss_1000](https://wandb.ai/distill-llms/policy-evaluation/runs/g0pdq8d1) {0.6B-Base} — pess=1000
- (compare against any 2ep no-pess sibling; no clean pess-vs-no-pess pair exists at the same epoch count, but the 2ep-no-pess [onlinePET-...-2ep-nopess](https://wandb.ai/distill-llms/policy-evaluation/runs/nsmn19q3) {0.6B-Base} is the closest)

### Q: Eval-time RM family — Qwen-0.6B vs Qwen3-Embedding-8B vs Gemma-2-2B Ray?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=9fb1cqrhedg)
Same Online PET training run, different `secondary_rm_name` / training-RM at eval.
- [onlinePET-qwen06B-Base-06BRM](https://wandb.ai/distill-llms/policy-evaluation/runs/6mlwj81t) {0.6B-Base} — 0.6B Qwen RM
- [online_pet_base_policy_8B_rm](https://wandb.ai/distill-llms/policy-evaluation/runs/t7xsv7ka) {0.6B-Base} — Qwen3-Embedding-8B
- [onlinePET-qwen06B-Base-06BRM-gemma2Bray](https://wandb.ai/distill-llms/policy-evaluation/runs/kft0e8o9) {0.6B-Base} — Gemma-2-2B Ray RM

### Other / standalone
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=78kthla35be)
- [online_pet_ref_reward_qwen06B](https://wandb.ai/distill-llms/policy-evaluation/runs/axiyt8ai) {0.6B} — first Online PET eval (uses the `b6dfd51` reference-reward annotated dataset)

---

## Adversarial RM (multi-step)

### Q: Does iterating Adv-RM steps (1 → 2 → 3) keep improving over the gold?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=9y3m4pld6qk)
Same Qwen3-Embedding-8B base; each step trains a new RM/policy on the previous step's outputs.
- [AdvRM-step1-qwen06B_with_qwen8B-embedding_RM](https://wandb.ai/distill-llms/policy-evaluation/runs/4nasl2w0) {0.6B}
- [AdvRM-step2-qwen06B_with_qwen8B-embedding_RM](https://wandb.ai/distill-llms/policy-evaluation/runs/2wml3liw) {0.6B} | [AdvRM-step2-qwen06B_with_qwen8B-embedding_RM_try_2](https://wandb.ai/distill-llms/policy-evaluation/runs/gbua7ydc) {0.6B}
- [AdvRM-step3-qwen06B_with_qwen8B-embedding_RM](https://wandb.ai/distill-llms/policy-evaluation/runs/tbiz9ief) {0.6B}

---

## PAR-GRPO and RRM-GRPO

### Q: PAR vs RRM with the same gold?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=a7k6y73cm3q)
- gold=URM-LLaMa-8B: [PAR_train_Ray_gold_URM](https://wandb.ai/distill-llms/policy-evaluation/runs/qkyg75ht) vs [RRM_gold_URM](https://wandb.ai/distill-llms/policy-evaluation/runs/296uqyzq)
- gold=QRM-Gemma-27B: [PAR_train=RAY_gold=QRM_20250624_131145](https://wandb.ai/distill-llms/policy-evaluation/runs/12vc5zav) vs [RRM_gold_QRM_20250624_111417](https://wandb.ai/distill-llms/policy-evaluation/runs/c4lokvjt)

### Q: Is gold-RM choice (URM-Llama vs QRM-Gemma) ranking-consistent across train-RM types?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=0j5bq8g1k08)
- PAR: URM gold → [PAR_train_Ray_gold_URM](https://wandb.ai/distill-llms/policy-evaluation/runs/qkyg75ht), QRM gold → [PAR_train=RAY_gold=QRM_20250624_131145](https://wandb.ai/distill-llms/policy-evaluation/runs/12vc5zav)
- RRM: URM gold → [RRM_gold_URM](https://wandb.ai/distill-llms/policy-evaluation/runs/296uqyzq), QRM gold → [RRM_gold_QRM_20250624_111417](https://wandb.ai/distill-llms/policy-evaluation/runs/c4lokvjt)

---

## Initial PPO bring-up (pre-GRPO)

- [PPO_gold=Ray_train=Qwen](https://wandb.ai/distill-llms/policy-evaluation/runs/wgzoe8id) — first eval, len3000, 5e-6 BT-RM

---

## LLM-as-judge head-to-head evals (older policies)

These don't roll out new policies — they take previously-trained PAR/RRM/qwen-base policies and have a Deepseek/R1 judge compare their generations against the dataset's chosen response.

### Q: Does the judge model matter (Deepseek-V3 vs Deepseek-R1)?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=sau09r7u9kh)
- PAR-policy: [PAR_vs_chosen_deepseek_v3_judge](https://wandb.ai/distill-llms/policy-evaluation/runs/gdxmbwvs) (V3); no R1 sibling
- RRM-policy: [RRMvsChosen_Deepseek_v3_judge](https://wandb.ai/distill-llms/policy-evaluation/runs/kpqxh5sc) (V3, with retries from `b519573`) vs [rrm_vs_chosen_deepseek_r1](https://wandb.ai/distill-llms/policy-evaluation/runs/jbul4zew) (R1)

### Q: Does enabling tie-handling change the judge's verdicts?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=efdr4ebkn9q)
- [rrm_vs_chosen_deepseek_r1](https://wandb.ai/distill-llms/policy-evaluation/runs/jbul4zew) — no ties
- [rrm_vs_chosen_deepseek_r1_ties](https://wandb.ai/distill-llms/policy-evaluation/runs/ow2qs55n) — ties enabled

### Q: Which RM trained the qwen-base policy that wins more often vs chosen?
[Comparison](https://wandb.ai/distill-llms/policy-evaluation?nw=625cxprvarb)
- [qwen_base_ray_vs_chosen_r1](https://wandb.ai/distill-llms/policy-evaluation/runs/le0m1gea) — policy trained with Ray-Gemma-2B RM
- [qwen_base_qwen_helpsteer_vs_chosen_r1](https://wandb.ai/distill-llms/policy-evaluation/runs/bd4qyj4a) — policy trained with Qwen-0.6B helpsteer RM
