## GRPO with reward-model aggregation strategies

### Q: Does training each RM longer (3 epochs vs 2) improve sequential-3x cycling over a 10-RM bank?
- **2026-05-05** — `3epRMs_KL0_10rms_sequential3x_1129495` — 10 RMs trained 3 epochs, sequential3x → [b6e9si4j](https://wandb.ai/distill-llms/policy-evaluation/runs/b6e9si4j)
- **2026-05-04** — `linear0.6-max1.5_KL0_1rms_sequential3x_1126524` — predecessor with the same 10-RM bank but 2 epochs each (and a linear length penalty as the additional change) → [4n3i20ph](https://wandb.ai/distill-llms/policy-evaluation/runs/4n3i20ph)

### Q: Does seed diversity across the RM bank help, or is the gain just from cycling many RMs?
- **2026-05-05** — `same_seed_KL0_8rms_sequential3x_1129482` — 8 RMs trained from the **same** seed → [llxyplds](https://wandb.ai/distill-llms/policy-evaluation/runs/llxyplds)
- **2026-05-05** — `3epRMs_KL0_10rms_sequential3x_1129495` — 10 RMs from **different** seeds (otherwise identical setup) → [b6e9si4j](https://wandb.ai/distill-llms/policy-evaluation/runs/b6e9si4j)

### Q: Linear vs squared ramp shape for the soft DAPO no-EOS penalty?
Length-penalty code path: `penalize_no_eos_power=1` (linear) vs `2` (squared); enabled by commit `97a53118` *new evals + penalty power*.
- **2026-04-26** — `0.6DAPO_linear_max1_KL0_1rms_sequential3x_1109220` — linear (`power=1`) → [jkm4l66t](https://wandb.ai/distill-llms/policy-evaluation/runs/jkm4l66t)
- **2026-04-27** — `0.6DAPO_squared_max1_KL0_1rms_sequential3x_1109219` — squared (`power=2`); same `soft_fraction=0.6, max_penalty=1` → [b6vnyx03](https://wandb.ai/distill-llms/policy-evaluation/runs/b6vnyx03)

### Q: How aggressive should the cap on the soft DAPO penalty be?
- **2026-04-28** — `dapo0.5-max1.5_KL0_1rms_sequential3x_1099677` — `soft_fraction=0.5, max_penalty=1.5, linear` → [v36lynn9](https://wandb.ai/distill-llms/policy-evaluation/runs/v36lynn9), [6c0fi2ay](https://wandb.ai/distill-llms/policy-evaluation/runs/6c0fi2ay), [70m8imgt](https://wandb.ai/distill-llms/policy-evaluation/runs/70m8imgt), [lor99sx9](https://wandb.ai/distill-llms/policy-evaluation/runs/lor99sx9)
- **2026-04-26** — `0.6DAPO_linear_max1_KL0_1rms_sequential3x_1109220` — `soft_fraction=0.6, max_penalty=1, linear` → [jkm4l66t](https://wandb.ai/distill-llms/policy-evaluation/runs/jkm4l66t)

### Q: GR3 multiplicative length-debiasing vs DAPO subtractive penalty?
GR3 (commit `9a2dafe4`) divides reward by `(1 + α·len/mean_len)` (sign-aware); DAPO (commit `a0c5c9c3`) subtracts a soft penalty.
- **2026-04-15** — `gr3_KL0_1rms_sequential3x_1099136` — GR3 α=0.5 → [ttm7baxh](https://wandb.ai/distill-llms/policy-evaluation/runs/ttm7baxh)
- **2026-04-28** — `dapo0.5-max1.5_KL0_1rms_sequential3x_1099677` — soft DAPO penalty → [v36lynn9](https://wandb.ai/distill-llms/policy-evaluation/runs/v36lynn9)

### Q: GRPO learning-rate sweep on 4B-Base SFT (single-RM, sequential3x, KL=0)?
- **2026-04-15** — `1e-5lr_KL0_1rms_sequential3x_1099071` — 1e-5 → [872sf4xt](https://wandb.ai/distill-llms/policy-evaluation/runs/872sf4xt)
- **2026-04-08** — `grpo_5e-6lr_KL0_1rms_sequential3x_1087938` — 5e-6 → [dsknetrl](https://wandb.ai/distill-llms/policy-evaluation/runs/dsknetrl)
- **2026-03-20** — `double_lr_2e-5_KL0_1rms_sequential3x_1069741` — 2e-5 → [01vf60lx](https://wandb.ai/distill-llms/policy-evaluation/runs/01vf60lx)
- **2026-03-19** — `double_lr(5e-4)_4B-3128_KL0_1rms_sequential3x_1069473` — 5e-4 (extreme) → [41p251sw](https://wandb.ai/distill-llms/policy-evaluation/runs/41p251sw)

### Q: Does β > 0 prevent over-optimization with a single-RM 4B-Base policy?
Same 4B-3128 RM, sequential3x, lr 1e-5; only β changes.
- **2026-03-12** — `4B-3128-nokl_KL0_1rms_sequential3x_1066782` — β=0 → [ba0kul6w](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w)
- **2026-03-12** — `4B-3128_KL0.005_1rms_sequential3x_1066783` — β=0.005 → [40gfblbn](https://wandb.ai/distill-llms/policy-evaluation/runs/40gfblbn)
- **2026-03-11** — `4B-3128_KL0.01_1rms_sequential3x_1066642` — β=0.01 → [67wp3l1r](https://wandb.ai/distill-llms/policy-evaluation/runs/67wp3l1r)

### Q: How much RM training is enough? 500 vs 3128 steps at the same β?
- **2026-03-11** — `4B-500_KL0.01_1rms_sequential3x_1066641` — 500-step RM, β=0.01 → [svl94js5](https://wandb.ai/distill-llms/policy-evaluation/runs/svl94js5)
- **2026-03-11** — `4B-3128_KL0.01_1rms_sequential3x_1066642` — 3128-step RM, β=0.01 → [67wp3l1r](https://wandb.ai/distill-llms/policy-evaluation/runs/67wp3l1r)
- **2026-03-12** — `500-nokl_KL0_1rms_sequential3x_1066781` — 500-step RM, β=0 → [2fezn8o6](https://wandb.ai/distill-llms/policy-evaluation/runs/2fezn8o6)
- **2026-03-12** — `4B-3128-nokl_KL0_1rms_sequential3x_1066782` — 3128-step RM, β=0 → [ba0kul6w](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w)

### Q: 1 RM vs 7-RM sequential3x at 4B-Base (with and without KL)?
- **2026-03-12** — `4B-3128-nokl_KL0_1rms_sequential3x_1066782` — 1 RM, β=0 → [ba0kul6w](https://wandb.ai/distill-llms/policy-evaluation/runs/ba0kul6w)
- **2026-03-12** — `4B-nokl_KL0_7rms_sequential3x_1066780` — 7 RMs, β=0 → [sor7ytzf](https://wandb.ai/distill-llms/policy-evaluation/runs/sor7ytzf)
- **2026-03-11** — `4B-3128_KL0.01_1rms_sequential3x_1066642` — 1 RM, β=0.01 → [67wp3l1r](https://wandb.ai/distill-llms/policy-evaluation/runs/67wp3l1r)
- **2026-03-11** — `4B-sequential-same-run_KL0.01_7rms_sequential3x_1066643` — 7 RMs, β=0.01 → [89v3h28w](https://wandb.ai/distill-llms/policy-evaluation/runs/89v3h28w)

### Q: 100-RM mix-of-ensembles aggregation: mean vs min vs UWO?
Same RM bank, same `mix_ensemble_size=10`, KL=0; only `ensemble_aggregation` differs (commit `0b800c62`).
- **2026-02-27** — `__KL0_mix_100rms_mean_disjoint_10-mixens_1061682` — mean → [znmynyme](https://wandb.ai/distill-llms/policy-evaluation/runs/znmynyme)
- **2026-02-27** — `min_KL0_mix_100rms_min_disjoint_10-mixens_1061683` — min → [7rznh8wf](https://wandb.ai/distill-llms/policy-evaluation/runs/7rznh8wf)
- **2026-02-27** — `high-uwo_KL0_mix_100rms_uwo10_random_disjoint_10-mixens_1061681` — UWO `λ=10`, `random_disjoint` partition → [w9rq5zie](https://wandb.ai/distill-llms/policy-evaluation/runs/w9rq5zie)

### Q: Mix-of-ensembles vs full-ensemble vs sequential-cycling at 100 RMs?
- **2026-02-28** — `40ens_KL0.005_ensemble_41rms_mean_1061686` — full ensemble of 41 RMs, mean → [klzsajm6](https://wandb.ai/distill-llms/policy-evaluation/runs/klzsajm6)
- **2026-02-27** — `__KL0_mix_100rms_mean_disjoint_10-mixens_1061682` — 100 RMs, mix-10×10 disjoint, mean → [znmynyme](https://wandb.ai/distill-llms/policy-evaluation/runs/znmynyme)
- **2026-02-27** — `sequential_KL0_sequential3x_100rms_1061684` — 100 RMs, sequential3x, β=0 → [0ca6wi21](https://wandb.ai/distill-llms/policy-evaluation/runs/0ca6wi21)
- **2026-02-27** — `seq_KL0.005_sequential3x_100rms_1061685` — sibling at β=0.005 → [7kxkvl6b](https://wandb.ai/distill-llms/policy-evaluation/runs/7kxkvl6b)

### Q: Does `clip_reward_max=3` help in mix-UWO at 100 RMs?
- **2026-02-25** — `1e-5lr_KL0_mix_100rms_uwo_disjoint_10-mixens_1061176` — no clipping → [qryzhkab](https://wandb.ai/distill-llms/policy-evaluation/runs/qryzhkab)
- **2026-02-26** — `clip3_KL0_mix_100rms_uwo_disjoint_10-mixens_1061189` — `clip_reward_max=3.0` (otherwise identical) → [j9w6q8f1](https://wandb.ai/distill-llms/policy-evaluation/runs/j9w6q8f1)

### Q: Effect of disabling `rm_scale_reward_by_std_per_model` on a 10-RM mean ensemble?
- **2026-02-25** — `reproduce_KL0_ensemble_10rms_mean_1061173` — with std-scaling (baseline) → [sj2i5ui5](https://wandb.ai/distill-llms/policy-evaluation/runs/sj2i5ui5)
- **2026-02-26** — `no-scale-std_KL0_ensemble_10rms_mean_1061633` — std-scaling disabled → [gacum1jv](https://wandb.ai/distill-llms/policy-evaluation/runs/gacum1jv)

### Q: 10×10 disjoint mean vs UWO?
- **2026-02-12** — `10x10mean_disjoint__reorder_parallel_v2_0.005K_1053397` — mean → [s2ac81kp](https://wandb.ai/distill-llms/policy-evaluation/runs/s2ac81kp)
- **2026-02-12** — `10x10uwo1_disjoint_0.005KL_1053530` — UWO λ=1 → [p2a4yp6a](https://wandb.ai/distill-llms/policy-evaluation/runs/p2a4yp6a)

### Q: KL coefficient on a 10-RM mean ensemble (lr=1e-5)?
- **2026-02-24** — `1e-5lr_nokl_KL0_ensemble_10rms_mean_1061140` — β=0 → [fvah0qui](https://wandb.ai/distill-llms/policy-evaluation/runs/fvah0qui)
- **2026-02-24** — `1e-5lr_KL0.005_ensemble_10rms_mean_1061147` — β=0.005 → [3a0xzb3j](https://wandb.ai/distill-llms/policy-evaluation/runs/3a0xzb3j)

### Q: 100-RM cycling once vs three times (full helpsteer3 dataset, β=0.005)?
- **2026-03-04** — `full-ds_KL0.005_100rms_sequential3x_1062944` — 3× → [tdbu4tnq](https://wandb.ai/distill-llms/policy-evaluation/runs/tdbu4tnq)
- **2026-03-01** — `1x_KL0.005_100rms_sequential1x_1062161` — 1× → [vc2f073s](https://wandb.ai/distill-llms/policy-evaluation/runs/vc2f073s)

### Q: 100-RM cycling 1× vs 3× (subset dataset, β=0.005)?
- **2026-02-04** — `100_sequential_1x_0.005KL_1036532` — 1× → [tiguq77k](https://wandb.ai/distill-llms/policy-evaluation/runs/tiguq77k)
- **2026-02-04** — `100_sequential_3x_0.005KL_1036533` — 3× → [jjp2nkwe](https://wandb.ai/distill-llms/policy-evaluation/runs/jjp2nkwe)

### Q: 40-RM ensemble: mean vs min aggregation?
- **2026-02-04** — `40_mean_0.005KL_1036535` — mean → [zxsedto1](https://wandb.ai/distill-llms/policy-evaluation/runs/zxsedto1)
- **2026-02-04** — `40_min_0.005KL_1036534` — min → [1tzct6dg](https://wandb.ai/distill-llms/policy-evaluation/runs/1tzct6dg)
- **2026-01-24** — `40_minens_noKL_1021611` — min, β=0 (no-KL sibling) → [yponcm2z](https://wandb.ai/distill-llms/policy-evaluation/runs/yponcm2z)

### Q: Mix-strategy partition: disjoint vs sliding?
Same `mix_ensemble_size=10`, min aggregation, β=0.005.
- **2026-02-04** — `mix_10x10min_disjoint_0.005KL_1036530` — disjoint → [inyswtrh](https://wandb.ai/distill-llms/policy-evaluation/runs/inyswtrh)
- **2026-02-04** — `mix_10x10min_sliding_0.005KL_1036531` — sliding (overlapping) → [zedzcoll](https://wandb.ai/distill-llms/policy-evaluation/runs/zedzcoll)

### Q: First-pass mix-strategy granularity (group count × group size)?
β=0 for the 20×5 sibling vs β=0.005 for the others.
- **2026-02-03** — `mix_2x50_0.005KL_1035193` — 2×50 → [fmbp1t3d](https://wandb.ai/distill-llms/policy-evaluation/runs/fmbp1t3d), [ampfr2us](https://wandb.ai/distill-llms/policy-evaluation/runs/ampfr2us)
- **2026-02-04** — `mix_10x10min_disjoint_0.005KL_1036530` — 10×10 → [inyswtrh](https://wandb.ai/distill-llms/policy-evaluation/runs/inyswtrh)
- **2026-02-02** — `mix_20x5min_noKL_1034756` — 20×5, β=0 → [5y3ihw7s](https://wandb.ai/distill-llms/policy-evaluation/runs/5y3ihw7s)
- **2026-01-25** — `mix_mean_10x10_sliding_noKL_1022182` — 10×10 mean (compare aggregation too) → [7oh7m4aa](https://wandb.ai/distill-llms/policy-evaluation/runs/7oh7m4aa)

### Q: KL sweep with 100-RM sequential cycling?
- **2026-02-02** — `seqential_100_noKL_1034755` — β=0 → [x0mzdkip](https://wandb.ai/distill-llms/policy-evaluation/runs/x0mzdkip)
- **2026-02-01** — `seqential_100_0.01KL_1033650` — β=0.01 → [hcfknonk](https://wandb.ai/distill-llms/policy-evaluation/runs/hcfknonk)
- **2026-02-04** — `100_sequential_3x_0.005KL_1036533` — β=0.005 → [jjp2nkwe](https://wandb.ai/distill-llms/policy-evaluation/runs/jjp2nkwe)

### Q: Sequential cycling at lower RM-count budgets (β=0.001)?
- **2026-01-19** — `50_sequential_0.001KL_1020076` — 50 RMs → [or65prnn](https://wandb.ai/distill-llms/policy-evaluation/runs/or65prnn)
- **2026-01-20** — `25_sequential_0.001KL_1020177` — 25 RMs → [84v20h95](https://wandb.ai/distill-llms/policy-evaluation/runs/84v20h95)

### Q: 10-RM (5-epoch each) — sequential3x vs min ensemble at β=0.01?
- **2026-01-30** — `5ep-rm_10_sequential_3x_0.01KL_1030113` — sequential3x → [8vu591sl](https://wandb.ai/distill-llms/policy-evaluation/runs/8vu591sl)
- **2026-01-30** — `5ep10_min_0.01KL_1030114` — min ensemble → [h6jrd6qs](https://wandb.ai/distill-llms/policy-evaluation/runs/h6jrd6qs)

### Q: Top-10-best RMs vs all-10 RMs (sequential3x and min)?
- **2026-01-25** — `sequential_10best_3x_0.01KL_1022174` — top-10 best, sequential3x → [45ch0pv2](https://wandb.ai/distill-llms/policy-evaluation/runs/45ch0pv2)
- **2026-01-24** — `sequential_10_3x_0.01KL_reorder_1022017` — all-10, sequential3x → [xcg07mww](https://wandb.ai/distill-llms/policy-evaluation/runs/xcg07mww)
- **2026-01-25** — `10best_min_0.01KL_1022175` — top-10 best, min → [m0jkbkzv](https://wandb.ai/distill-llms/policy-evaluation/runs/m0jkbkzv)

### Q: 10× 2-epoch RMs — sequential vs mean vs min ensemble?
Same RM bank, no name-encoded KL change.
- **2025-12-10** — `10_2ep-rm_sequential` — sequential → [63h6z2q7](https://wandb.ai/distill-llms/policy-evaluation/runs/63h6z2q7)
- **2025-12-09** — `10_2ep-rm_mean_ensemble` — mean → [pxh5rt65](https://wandb.ai/distill-llms/policy-evaluation/runs/pxh5rt65)
- **2025-12-09** — `10_2ep-rm_min_ensemble` — min → [2p3nnuyk](https://wandb.ai/distill-llms/policy-evaluation/runs/2p3nnuyk)
- **2025-12-07** — `best-of-10-2ep_rm` — single best-of-10 RM (only run with the original hard `-1` no-EOS penalty from `c46b9ad8`) → [djvfgf0b](https://wandb.ai/distill-llms/policy-evaluation/runs/djvfgf0b)

### Q: Single RM at lr=1e-6 — alone vs as 5-best min ensemble?
- **2025-12-11** — `1rm_half-lr(1e-6)` — 1 RM → [33qbe6wz](https://wandb.ai/distill-llms/policy-evaluation/runs/33qbe6wz)
- **2025-12-12** — `1e-6-lr_5best_min-ens` — top-5 best min ensemble at the same lr → [jrizwomb](https://wandb.ai/distill-llms/policy-evaluation/runs/jrizwomb)

### Q: KL coefficient sweep with single-RM, ~17-RM sequential, around the GRPO bring-up phase?
- **2026-01-09** — `0.08kl_1017910` — KL=0.08 → [wrkc85nm](https://wandb.ai/distill-llms/policy-evaluation/runs/wrkc85nm)
- **2026-01-09** — `low-lr_low-temp_0.02-kl_17_1017909` — KL=0.02 (also lower lr+temp, confounded) → [obi9q1j6](https://wandb.ai/distill-llms/policy-evaluation/runs/obi9q1j6)
- **2026-01-08** — `grpo_new_17_kl0.04` — KL=0.04 → [8khfdnuv](https://wandb.ai/distill-llms/policy-evaluation/runs/8khfdnuv)
- **2026-01-07** — `grpo_new_17_1017395` — defaults baseline → [6vhc9o1s](https://wandb.ai/distill-llms/policy-evaluation/runs/6vhc9o1s)

### Q: 10-RM ensemble: min vs 3-RM ensemble at β=0.01 (early-2026 bring-up)?
- **2026-01-10** — `minens-0.01kl_1018058` — 10 RMs min → [1ryimavw](https://wandb.ai/distill-llms/policy-evaluation/runs/1ryimavw)
- **2026-01-10** — `3ens_0.01KL` — 3-RM ensemble → [9nkcp292](https://wandb.ai/distill-llms/policy-evaluation/runs/9nkcp292)
- **2026-01-08** — `min_ens_no-kl_1017706` — min ens, β=0 → [36a7jgv3](https://wandb.ai/distill-llms/policy-evaluation/runs/36a7jgv3)

### Q: Where to put KL — against the SFT base or against the original Qwen-0.6B base?
Eval-side `kl_base_model_path` change (commit `e59a2353` *kl base policy* / `fea3ffaf` *eval KL to base*).
- **2025-12-19** — `t1.0_beta0.04_1rm_KL2_qwen0.6B` — KL vs raw Qwen-0.6B → [3dpbxgyt](https://wandb.ai/distill-llms/policy-evaluation/runs/3dpbxgyt)
- **2025-12-22** — `t1.0_beta0.04_1rm_KL2_base` — KL vs SFT base → [6lxum6ip](https://wandb.ai/distill-llms/policy-evaluation/runs/6lxum6ip)

### Other / standalone
- **2026-04-15** — `1e-5lr_KL0_1rms_sequential3x_1099071` (used as the lr=1e-5 baseline in the GR3 group) → [872sf4xt](https://wandb.ai/distill-llms/policy-evaluation/runs/872sf4xt)
- **2026-02-23** — `25pct_5epRMs_KL0.005_ensemble_10rms_mean_1060956` — 5-epoch RMs on a 25% subset → [i6negqgg](https://wandb.ai/distill-llms/policy-evaluation/runs/i6negqgg)
- **2026-02-23** — `10x10uwo_disjoint_subprecompmean_0.005KL_1053933` — UWO with precomputed-mean optimization → [w769d1ww](https://wandb.ai/distill-llms/policy-evaluation/runs/w769d1ww), [qtdh0v22](https://wandb.ai/distill-llms/policy-evaluation/runs/qtdh0v22), [i3j6yp59](https://wandb.ai/distill-llms/policy-evaluation/runs/i3j6yp59)
- **2026-02-16** — `10x10mean_disjoint__reorder_parallel_0.005K_1047283` (eval reruns of one training run after eval-tokenization fixes — useful for measuring eval-side variance) → [vkolkfga](https://wandb.ai/distill-llms/policy-evaluation/runs/vkolkfga), [1knn3sww](https://wandb.ai/distill-llms/policy-evaluation/runs/1knn3sww), [mg27ka0t](https://wandb.ai/distill-llms/policy-evaluation/runs/mg27ka0t), [cwo5im4y](https://wandb.ai/distill-llms/policy-evaluation/runs/cwo5im4y), [a4vxxzhq](https://wandb.ai/distill-llms/policy-evaluation/runs/a4vxxzhq), [69kyfmj0](https://wandb.ai/distill-llms/policy-evaluation/runs/69kyfmj0), [n1f2jc1e](https://wandb.ai/distill-llms/policy-evaluation/runs/n1f2jc1e), [a1wd6syj](https://wandb.ai/distill-llms/policy-evaluation/runs/a1wd6syj)
- **2026-02-14** — `10x10mean_disjoint_datautils_refactor+precompute_0.005K_1053919` → [bxxocfcq](https://wandb.ai/distill-llms/policy-evaluation/runs/bxxocfcq)
- **2026-02-12** — `3_out_100_uwo1_rand-disj_3x_0.005KL_1053531` — random-disjoint sampling 3 RMs from a 100-pool, UWO λ=1, sequential3x → [grpzra8l](https://wandb.ai/distill-llms/policy-evaluation/runs/grpzra8l)
- **2026-02-07** — `25x4mean_disjoint_0.005KL_1043248` (25 disjoint groups of 4) → [oyveeskn](https://wandb.ai/distill-llms/policy-evaluation/runs/oyveeskn) | `..._Qwen3-0.6B-Base_1043249` (sibling on raw Base, not SFT) → [ccbmsxld](https://wandb.ai/distill-llms/policy-evaluation/runs/ccbmsxld)
- **2026-02-06** — `other_40_mean_0.05KL_1038817` — 40-RM mean at β=0.05 (10× usual KL) → [rbvos9s8](https://wandb.ai/distill-llms/policy-evaluation/runs/rbvos9s8) | **2026-02-05** `10x10mean_disjoint_0.005KL_1038818` → [o5325lt2](https://wandb.ai/distill-llms/policy-evaluation/runs/o5325lt2)
- **2026-01-23** — `new_10_seq-ens_3x_0.01KL_1021610` → [o8sfudyb](https://wandb.ai/distill-llms/policy-evaluation/runs/o8sfudyb) | **2026-01-21** `new_10_seq_ens_3x_noKL_1020207` → [38vulkjb](https://wandb.ai/distill-llms/policy-evaluation/runs/38vulkjb) | `25_min_ensemble_0KL_1020191` → [zdj21jo0](https://wandb.ai/distill-llms/policy-evaluation/runs/zdj21jo0)
- **2026-01-18** — `100_sequential_noKL_1019598` → [vzn2yyx1](https://wandb.ai/distill-llms/policy-evaluation/runs/vzn2yyx1) | **01-17** `100rm_sequential_1019534` → [hb0fn0ms](https://wandb.ai/distill-llms/policy-evaluation/runs/hb0fn0ms) | **01-16** `free_memory_sequential_10_x50_1019400` → [iv84ysdn](https://wandb.ai/distill-llms/policy-evaluation/runs/iv84ysdn) | **01-13** `sequential_10_3x_1018956_0.01KL` → [gma9k2le](https://wandb.ai/distill-llms/policy-evaluation/runs/gma9k2le)
- **2026-01-07** — `grpo_new_sft_min_ens_0.02kl_1017002` — first run on the new SFT base + min-ensemble → [sphfjynm](https://wandb.ai/distill-llms/policy-evaluation/runs/sphfjynm)
- **2026-01-01 / 2025-12-30** — `grpo_sft1ep_0.02KL_1012751` — first GRPO from a 1-epoch SFT base → [lipss3j6](https://wandb.ai/distill-llms/policy-evaluation/runs/lipss3j6), [wv5a72za](https://wandb.ai/distill-llms/policy-evaluation/runs/wv5a72za)
- **2025-12-24** — `grpo_from_sft_kl0.02_1008081` — first GRPO from the 20-epoch SFT → [pqfwpznz](https://wandb.ai/distill-llms/policy-evaluation/runs/pqfwpznz)
- **2025-12-23** — `no-kl__1.0temp_1rm_1007307` → [guktvo5t](https://wandb.ai/distill-llms/policy-evaluation/runs/guktvo5t)
- **2025-12-05** — `2ep_rm_over_maxlen` → [yhgxv5ck](https://wandb.ai/distill-llms/policy-evaluation/runs/yhgxv5ck)
- **2025-11-{12-28}** — early helpsteer3 GRPO bring-up: `high_lr_continual_28RMepochs` (first pess-loss run) → [ompxv93o](https://wandb.ai/distill-llms/policy-evaluation/runs/ompxv93o), `_high_lr_1epRM_continual` → [5kg0iy2w](https://wandb.ai/distill-llms/policy-evaluation/runs/5kg0iy2w), `high_lr_10epRM_10k` → [yfqboj8t](https://wandb.ai/distill-llms/policy-evaluation/runs/yfqboj8t), `high_lr_temp_hs3_40k` → [lyloj22g](https://wandb.ai/distill-llms/policy-evaluation/runs/lyloj22g)
- **2025-08** — earliest 8B-embedding-RM GRPO experiments: `deepspeed_0.001` → [kxezp55r](https://wandb.ai/distill-llms/policy-evaluation/runs/kxezp55r), `QRM_Llama8b_baseQwen06B` → [vxo0afaq](https://wandb.ai/distill-llms/policy-evaluation/runs/vxo0afaq), `lr1e-7_16_resp_32_batch_4096_replay` → [yibmhepk](https://wandb.ai/distill-llms/policy-evaluation/runs/yibmhepk)
- **2025-07-15** — `qwen06B_with_qwen8B-embedding_RM` — first GRPO with Qwen3-Embedding-8B as base RM → [4ryuk8oo](https://wandb.ai/distill-llms/policy-evaluation/runs/4ryuk8oo)
- **2025-06** — earliest GRPO bring-up runs (different RM/gold pairs): `train_Ray_gold_QRM` → [nf9mi154](https://wandb.ai/distill-llms/policy-evaluation/runs/nf9mi154), `min-ans_gold_URM_train_Ray` → [qbro76p8](https://wandb.ai/distill-llms/policy-evaluation/runs/qbro76p8), `train_QRM_gold_URM` → [ke0p6by4](https://wandb.ai/distill-llms/policy-evaluation/runs/ke0p6by4), `train_Ray_gold_URM` → [kwuqpz84](https://wandb.ai/distill-llms/policy-evaluation/runs/kwuqpz84), `grpo_train_Qwen_gold_Ray` → [g3vhua57](https://wandb.ai/distill-llms/policy-evaluation/runs/g3vhua57)
- **2025-05-30** — `train_Qwen_gold_Ray` — last PPO-era eval before the move to GRPO → [uz4vpxu9](https://wandb.ai/distill-llms/policy-evaluation/runs/uz4vpxu9)
- **2025-11-{13,14}** — RM-side reward-scale ablation: `1ep_seed42_noscale` → [dx898w6h](https://wandb.ai/distill-llms/policy-evaluation/runs/dx898w6h) vs `1epRM_seed43_scale` → [nx2v1cvq](https://wandb.ai/distill-llms/policy-evaluation/runs/nx2v1cvq) (seed and scale-on/off both change — confounded)
- **2025-11-12** — `1ep_normalized_mean_std` (mean+std normalisation) → [5v57qm0l](https://wandb.ai/distill-llms/policy-evaluation/runs/5v57qm0l)

### Q: Does training the RM longer (more samples) help if the policy is also scaled up to 40k?
- **2025-11-12** — `10ep_10k->40k_policy` — 10-ep RM on 10k samples → 40k-sample policy → [1ofqxnvn](https://wandb.ai/distill-llms/policy-evaluation/runs/1ofqxnvn)
- **2025-11-12** — `1ep_40k_RM->40k_policy` — 1-ep RM on 40k samples → 40k-sample policy → [pw96ou3m](https://wandb.ai/distill-llms/policy-evaluation/runs/pw96ou3m)

---

## Larger-policy bring-up (Qwen3.5-4B / 8B / 1.7B / LoRA)

### Q: Policy-size scaling at the same setup (single-RM, sequential3x, β=0)?
- **2026-03-21** — `1.7Bsft_KL0_1rms_sequential3x_1070738` — 1.7B SFT → [1obc7wzc](https://wandb.ai/distill-llms/policy-evaluation/runs/1obc7wzc)
- **2026-03-19** — `4B-Base_KL0_1rms_sequential3x_1069463` — 4B Base → [1soo7lfo](https://wandb.ai/distill-llms/policy-evaluation/runs/1soo7lfo)
- **2026-03-19** — `Qwen3-8B_KL0_1rms_sequential3x_1069470` — 8B (full-FT) → [9okigr2v](https://wandb.ai/distill-llms/policy-evaluation/runs/9okigr2v)

### Q: 8B policy — full-FT vs LoRA?
- **2026-03-19** — `Qwen3-8B_KL0_1rms_sequential3x_1069470` — full-FT → [9okigr2v](https://wandb.ai/distill-llms/policy-evaluation/runs/9okigr2v)
- **2026-03-24** — `8B-LoRA-policy-4BInstructRM_KL0_1rms_sequential3x_1074475` — LoRA, default lr → [v9qa5abk](https://wandb.ai/distill-llms/policy-evaluation/runs/v9qa5abk)
- **2026-03-24** — `8B-LoRA-higherLR_KL0_1rms_sequential3x_1074580` — LoRA, higher lr → [5irdcj8d](https://wandb.ai/distill-llms/policy-evaluation/runs/5irdcj8d)

### Q: Cross-size pairing — small policy + big RM, or big policy + small RM?
- **2026-03-22** — `0.6Bsft_3.5-9BRM_KL0_1rms_sequential3x_1072946` — 0.6B policy, 3.5-9B RM → [gguolgti](https://wandb.ai/distill-llms/policy-evaluation/runs/gguolgti)
- **2026-03-22** — `4Bsft-4BInstructRM_KL0_1rms_sequential3x_1071882` — 4B policy, 4B-Instruct RM → [r5oubozm](https://wandb.ai/distill-llms/policy-evaluation/runs/r5oubozm)
- **2026-03-24** — `8B-LoRA-policy-4BInstructRM_KL0_1rms_sequential3x_1074475` — 8B (LoRA) policy, 4B-Instruct RM → [v9qa5abk](https://wandb.ai/distill-llms/policy-evaluation/runs/v9qa5abk)

### Q: 4B-Base SFT — Instruct vs Non-Instruct as the policy?
- **2026-03-19** — `4B-Base_KL0_1rms_sequential3x_1069463` — Instruct base → [1soo7lfo](https://wandb.ai/distill-llms/policy-evaluation/runs/1soo7lfo)
- **2026-03-18** — `4B-NonInstruct_KL0_1rm_1068892` — Non-Instruct (after `bf1ab59` *fix num_labels mismatch*) → [hn5q6u6m](https://wandb.ai/distill-llms/policy-evaluation/runs/hn5q6u6m)

### Q: Does the FP32-Mamba monkey-patch unblock Qwen3.5 hybrid blocks under bf16 GRPO?
- **2026-04-07** — `qwen3.5_FP32mamba_KL0_1rms_sequential3x_1086797` — first run that completes after `0f20ad2` *qwen3.5_FP32mamba* → [82vdkzkw](https://wandb.ai/distill-llms/policy-evaluation/runs/82vdkzkw), [b6p073dc](https://wandb.ai/distill-llms/policy-evaluation/runs/b6p073dc)
- **2026-03-31** — `Qwen3.5-4B-Base-sft` — earlier attempts that crashed without the patch (commits `281821a` *vllm + transformers patch 48*, `22d13436` *patch eval*) → [1ppfr693](https://wandb.ai/distill-llms/policy-evaluation/runs/1ppfr693), [ceowa9ac](https://wandb.ai/distill-llms/policy-evaluation/runs/ceowa9ac), [7cwf0ms1](https://wandb.ai/distill-llms/policy-evaluation/runs/7cwf0ms1)

### Q: Annotation source for the 4B SFT — helpsteer3 alone vs with human responses?
Commit `cd74de5` *-human option for dataset annotation* added the `--human` mode.
- **2026-04-02** — `3.5-4B-both_KL0_1rms_sequential3x_1078525` — Qwen3.5-4B with both helpsteer3 + human → [592ug9hu](https://wandb.ai/distill-llms/policy-evaluation/runs/592ug9hu)
- **2026-04-05** — `3.5B-base-both-human_KL0_1rms_sequential3x_1084314` — Qwen3.5-3.5B with both → [ywy5hz5o](https://wandb.ai/distill-llms/policy-evaluation/runs/ywy5hz5o)

### Other / standalone
- **2026-04-10** — `4B-Base_KL0_1rms_sequential3x_1089357` — first run on the new `1089122` SFT → [hlct4b7l](https://wandb.ai/distill-llms/policy-evaluation/runs/hlct4b7l)
- **2026-03-19** — `qwen3.5-4b_KL0_1rms_sequential3x_1069471` — Qwen3.5-4B without SFT → [s1fiv1uc](https://wandb.ai/distill-llms/policy-evaluation/runs/s1fiv1uc)

---

## DPO / APO offline preference optimization

The `0ec65f88` *dpo* commit introduced `rlhf/dpo/my_dpo.py` (TRL DPOTrainer wrapper) supporting `loss_type ∈ {sigmoid, apo_zero, …}`.

### Q: APO-zero vs sigmoid DPO on the same SFT base?
- **2026-04-09** — `dpo_sigmoid_4B-Base_1089353` — sigmoid (default β) → [6cyhd9z3](https://wandb.ai/distill-llms/policy-evaluation/runs/6cyhd9z3)
- **2026-04-09** — `apo_4B-Base_1089354` — APO-zero (default β) → [elv91xmk](https://wandb.ai/distill-llms/policy-evaluation/runs/elv91xmk)
- **2026-04-10** — `dpo_sigmoid_KL0.01_0.01KL_1089542` — sigmoid, β=0.01 → [ez0fbzp8](https://wandb.ai/distill-llms/policy-evaluation/runs/ez0fbzp8)
- **2026-04-10** — `dpo_apo_zero_KL0.01_0.01KL_1089543` — APO-zero, β=0.01 → [l9r5yqzj](https://wandb.ai/distill-llms/policy-evaluation/runs/l9r5yqzj)

### Q: DPO β: default vs explicit β=0.01?
- **2026-04-09** — `dpo_sigmoid_4B-Base_1089353` — default β → [6cyhd9z3](https://wandb.ai/distill-llms/policy-evaluation/runs/6cyhd9z3)
- **2026-04-10** — `dpo_sigmoid_KL0.01_0.01KL_1089542` — β=0.01 → [ez0fbzp8](https://wandb.ai/distill-llms/policy-evaluation/runs/ez0fbzp8)
- (And same for APO: 1089354 vs 1089543)

### Other / standalone
- **2026-04-08** — `dpo_sigmoid_dpo_1088543` — first DPO run after the trainer was added → [dsip2tv9](https://wandb.ai/distill-llms/policy-evaluation/runs/dsip2tv9)

---

## SFT base-policy bring-up

These produce the SFT checkpoints that all subsequent GRPO/DPO runs use as `kl_base_model_path` / starting point.

### Q: SFT epoch count: 5 vs 20?
- **2026-02-23** — `sft_5ep_1060185` — 5-epoch SFT → [81fp3ez6](https://wandb.ai/distill-llms/policy-evaluation/runs/81fp3ez6), [211ex4po](https://wandb.ai/distill-llms/policy-evaluation/runs/211ex4po), [izkypt3a](https://wandb.ai/distill-llms/policy-evaluation/runs/izkypt3a), [c4cr4ifr](https://wandb.ai/distill-llms/policy-evaluation/runs/c4cr4ifr)
- **2025-12-25** — `sft_20_epochs_1008548` → [6myafsv6](https://wandb.ai/distill-llms/policy-evaluation/runs/6myafsv6)
- **2025-12-23** — `sft_sft_20ep_1008088` (sibling) → [c0ypc7qe](https://wandb.ai/distill-llms/policy-evaluation/runs/c0ypc7qe)

### Q: Effect of SFT bug fixes (BOS stripping, `add_special_tokens=True`)?
- **2026-02-17** — `sft_bug_fixes_1058401` — with fixes (`b41f52c9` + `0a8810d`) → [gcws1fuv](https://wandb.ai/distill-llms/policy-evaluation/runs/gcws1fuv)
- **2026-02-17** — `sft_undo_sft_changes_1058521` — same dataset/config but the fixes reverted (commit `7d178c30`) → [uyrxxys7](https://wandb.ai/distill-llms/policy-evaluation/runs/uyrxxys7)

### Q: bf16 SFT vs fp32?
- **2026-02-18** — `sft_Redo_bf16_1058899` — bf16 (commit `32c8a408` + the bf16-everywhere consistency in `b1e083e`) → [r78m7srm](https://wandb.ai/distill-llms/policy-evaluation/runs/r78m7srm)
- **2026-02-17** — `sft_undo_sft_changes_1058521` — fp32 sibling → [uyrxxys7](https://wandb.ai/distill-llms/policy-evaluation/runs/uyrxxys7)

### Q: SFT dataset size — 8k vs 10k vs full helpsteer3?
- **2026-02-17** — `sft_sft10k_1058332` — 10k subset → [4fnhjsap](https://wandb.ai/distill-llms/policy-evaluation/runs/4fnhjsap)
- **2026-02-16** — `8k_sft_new-eval` — 8k subset → [3qah4y8y](https://wandb.ai/distill-llms/policy-evaluation/runs/3qah4y8y)
- **2026-02-16 / 02-13** — `sft_Full_dataset_sft_1053934` — full dataset → [er9mon61](https://wandb.ai/distill-llms/policy-evaluation/runs/er9mon61), [yzdi65bg](https://wandb.ai/distill-llms/policy-evaluation/runs/yzdi65bg)

### Other / standalone
- **2026-04-10** — `sft-qwen3.5-4B` (Qwen3.5-4B SFT, the launch eval) → [vnyf7yl0](https://wandb.ai/distill-llms/policy-evaluation/runs/vnyf7yl0)
- **2026-04-09** — `sft_default_1089122` — *the* SFT base used by the latest 4B GRPO/DPO experiments → [l4cc8kam](https://wandb.ai/distill-llms/policy-evaluation/runs/l4cc8kam)
- **2026-04-03** — `sft_3.5-4B-Base-human_1082435` — Qwen3.5-4B-Base SFT'd on the human-annotated dataset → [m1y7ovra](https://wandb.ai/distill-llms/policy-evaluation/runs/m1y7ovra)
- **2026-03-21** — `sft_4B-Base_1070739` (used by GRPO `1071882`, `1072946`) → [rj0a2f7p](https://wandb.ai/distill-llms/policy-evaluation/runs/rj0a2f7p)
- **2026-03-20** — `sft_1.7B_1070705` → [upl84n76](https://wandb.ai/distill-llms/policy-evaluation/runs/upl84n76)
- **2026-02-19** — `sft_sft_validation_1060057` — adds held-out validation split (`1c5f96d5`) → [lw9en7yo](https://wandb.ai/distill-llms/policy-evaluation/runs/lw9en7yo), [gplakym2](https://wandb.ai/distill-llms/policy-evaluation/runs/gplakym2)
- **2026-02-17** — `sft_new_sft_1057788` — first SFT after the data-pipeline + RM-loading refactor (`21b850e`) → [h56kyuxj](https://wandb.ai/distill-llms/policy-evaluation/runs/h56kyuxj)
- **2026-02-07** — `sft->Skywork/Skywork-Reward-V2-Qwen3-8B_1043250` — gold-RM family switched (V2-Llama → V2-Qwen) → [089kkxkf](https://wandb.ai/distill-llms/policy-evaluation/runs/089kkxkf)
- **2026-01-06** — `sft_reward_texts_printing_1016814` — long-lived KL base for ~56 GRPO runs → [2t79ziy9](https://wandb.ai/distill-llms/policy-evaluation/runs/2t79ziy9), `sft_eval_new` → [t4m8a5q2](https://wandb.ai/distill-llms/policy-evaluation/runs/t4m8a5q2)
- **2025-12-22** — `sft_sft_fixes_1007442` → [pafvlp9e](https://wandb.ai/distill-llms/policy-evaluation/runs/pafvlp9e)

---

## HelpSteer3 gold-annotated re-base

### Q: How many epochs to train the BT RM (gold=Skywork-V2-Llama-8B)?
- **2025-11-01** — `06b-hs3gold-2_rm_epochs` — 2 ep → [jt7pn8us](https://wandb.ai/distill-llms/policy-evaluation/runs/jt7pn8us)
- **2025-10-31** — `06b-hs3gold-5_rm_epochs` — 5 ep → [2vr6tb6c](https://wandb.ai/distill-llms/policy-evaluation/runs/2vr6tb6c)
- **2025-10-29** — `06b-hs3gold-10_rm_epochs` — 10 ep → [pes8ntrx](https://wandb.ai/distill-llms/policy-evaluation/runs/pes8ntrx)

### Q: RM-epochs sweep with the alternative gold (Skywork-V2-Qwen)?
Same comparison as above but `gold_rm_name=Skywork-Reward-V2-Qwen3-8B` (eval-side change in `8af205f6` *eval continual_full with other gold*).
- **2025-11-05** — `06b-hs3gold-2_rm_epochs_qwengold` → [8yokcn09](https://wandb.ai/distill-llms/policy-evaluation/runs/8yokcn09)
- **2025-11-05** — `06b-hs3gold-10_rm_epochs_qwengold` → [gymns6vn](https://wandb.ai/distill-llms/policy-evaluation/runs/gymns6vn)

### Q: Gold-RM family bias — Skywork-V2-Llama vs Skywork-V2-Qwen as the *evaluator*?
Same training; only the gold RM at eval time changes. (Pair these with the `_qwengold` suffix.)
- 2-ep RM: `06b-hs3gold-2_rm_epochs` [jt7pn8us](https://wandb.ai/distill-llms/policy-evaluation/runs/jt7pn8us) vs `..._qwengold` [8yokcn09](https://wandb.ai/distill-llms/policy-evaluation/runs/8yokcn09)
- 10-ep RM: `06b-hs3gold-10_rm_epochs` [pes8ntrx](https://wandb.ai/distill-llms/policy-evaluation/runs/pes8ntrx) vs `..._qwengold` [gymns6vn](https://wandb.ai/distill-llms/policy-evaluation/runs/gymns6vn)
- continual full-FT: `06b-hs3gold-continual_bt_rm_full` [u2ehnxxv](https://wandb.ai/distill-llms/policy-evaluation/runs/u2ehnxxv) vs `..._qwengold` [leaf0g1g](https://wandb.ai/distill-llms/policy-evaluation/runs/leaf0g1g)
- 8 RMs minens: `Qwen06B_helpsteer3_minensemble8` [v8czdzix](https://wandb.ai/distill-llms/policy-evaluation/runs/v8czdzix) vs `..._qwengold` [22n7fcz3](https://wandb.ai/distill-llms/policy-evaluation/runs/22n7fcz3)

### Q: Continual RM training — full-FT vs LoRA?
- **2025-10-30** — `06b-hs3gold-continual_bt_rm_full` — full-FT → [u2ehnxxv](https://wandb.ai/distill-llms/policy-evaluation/runs/u2ehnxxv)
- **2025-10-30** — `06b-hs3gold-continual_bt_rm_lora` — LoRA → [pbquwi5p](https://wandb.ai/distill-llms/policy-evaluation/runs/pbquwi5p)

### Q: Single RM vs 8-RM min-ensemble (lr=5e-7)?
- **2025-10-23** — `lr5e-7_rmQwen06B_helpsteer3_gold` — 1 RM → [z3zpf0m2](https://wandb.ai/distill-llms/policy-evaluation/runs/z3zpf0m2)
- **2025-10-23** — `lr5e-7_rmQwen06B_helpsteer3_gold_sequential` — 8 RMs sequential → [zn8goyls](https://wandb.ai/distill-llms/policy-evaluation/runs/zn8goyls)
- **2025-10-25** — `Qwen06B_helpsteer3_minensemble8` — 8 RMs min-ensemble → [v8czdzix](https://wandb.ai/distill-llms/policy-evaluation/runs/v8czdzix)

### Q: `rm_switches_multiplier` 3× vs 50×?
- **2025-10-29** — `06b-hs3gold-rm_switches_multiplier3` — 3× cycling → [f44rt09l](https://wandb.ai/distill-llms/policy-evaluation/runs/f44rt09l)
- **2025-11-01** — `06b-hs3gold-2_rm_epochs` — multiplier=50 (commit `cfb0533b`) → [jt7pn8us](https://wandb.ai/distill-llms/policy-evaluation/runs/jt7pn8us)

### Other / standalone
- **2025-11-08** — `hs3-10k_min-ens_corrected-subtract-mean` — min-ens with corrected `rm_subtract_mean_reward_per_model` (`9836826`) → [7vgocgy4](https://wandb.ai/distill-llms/policy-evaluation/runs/7vgocgy4) | `hs3_40krm_q06base` → [gmokypsa](https://wandb.ai/distill-llms/policy-evaluation/runs/gmokypsa)
- **2025-11-04** — `lr5e-7_rmQwen06B_helpsteer3_qwengold` — single RM at lr=5e-7, Qwen gold → [oo8fiitd](https://wandb.ai/distill-llms/policy-evaluation/runs/oo8fiitd)
- **2025-11-05** — `Qwen06B_helpsteer3_minensemble8_qwengold` — 8 minens, Qwen gold → [22n7fcz3](https://wandb.ai/distill-llms/policy-evaluation/runs/22n7fcz3)

---

## Pessimistic-loss / CQL / ReLU additions

`rlhf/grpo/online_pet.py` accepts `pessimistic_loss_weight`, `relu_chosen_reward_loss`, `cql_optimistic_loss_weight` (added in `13ba1b0` as `-mean(chosen_rewards)`). All extend GRPO with offline-RL-style regularization terms.

### Q: Pessimistic loss weight magnitude (after the gradient fix)?
After `c0c6ec8` *pess loss gradient fix*:
- **2025-08-22** — `0.001_pess_fix_3ep` — pess=0.001 → [8mllca4h](https://wandb.ai/distill-llms/policy-evaluation/runs/8mllca4h)
- **2025-08-31** — `pess0.1_baseline` — pess=0.1 → [yntmzx3a](https://wandb.ai/distill-llms/policy-evaluation/runs/yntmzx3a)
- **2025-08-17** — `pess1M_3epochs` — pess=1e6 (extreme) → [mvebzmsk](https://wandb.ai/distill-llms/policy-evaluation/runs/mvebzmsk)

### Q: Does adding `relu_chosen_reward_loss` on top of pess help?
- **2025-08-31** — `pess0.1_baseline` — pess=0.1 only → [yntmzx3a](https://wandb.ai/distill-llms/policy-evaluation/runs/yntmzx3a)
- **2025-09-01** — `relu0.01mean_pess0.1` — pess=0.1 + relu=0.01 (mean baseline) → [ek1mcg73](https://wandb.ai/distill-llms/policy-evaluation/runs/ek1mcg73)
- **2025-10-09** — `helpsteer3_pess0.1_relu` — pess=0.1 + relu=0.1 (`relu_chosen_use_rejected_baseline=True`) → [9gtjlie7](https://wandb.ai/distill-llms/policy-evaluation/runs/9gtjlie7)

### Q: Does adding the CQL optimistic term on top of pess help, and what weight?
After `13ba1b0` *cql optimistic loss*:
- **2025-11-30** — `continual_cql_pess0.001` — pess+cql = 0.001 each → [b7fpjldr](https://wandb.ai/distill-llms/policy-evaluation/runs/b7fpjldr)
- **2025-12-02** — `continual_cql_pess0.01` — pess+cql = 0.01 each → [tjne9x5t](https://wandb.ai/distill-llms/policy-evaluation/runs/tjne9x5t)
- **2026-01-11** — `continual_cql-0.005_1018060` — pess+cql=0.005, β=0.01 → [95zrqtgk](https://wandb.ai/distill-llms/policy-evaluation/runs/95zrqtgk)

### Q: Pessimistic-batch / replay-buffer hyperparameters (from `acc60a1e` "1-e7 LR, 16 batch, 8 gens, 10k pessloss, 32 preference batch, 64 pess batch")?
- **2025-08-08** — `pess1k_batch32_16gen_512replay_1e-7` — pess=1k, batch=32, 16 gens, 512 replay → [v5952gvl](https://wandb.ai/distill-llms/policy-evaluation/runs/v5952gvl)
- **2025-08-08** — `pess10k_batch32_16gen_512replay_1e-7` — pess=10k (10× higher), same other params → [fdyqc227](https://wandb.ai/distill-llms/policy-evaluation/runs/fdyqc227)
- **2025-08-11** — `pess10k_batch16_8gen_256replay_1e-7_32pessbatch_16bt` — half batch, half gens, half replay, separate pess (32) and BT (16) batch sizes → [e77wionk](https://wandb.ai/distill-llms/policy-evaluation/runs/e77wionk)

### Q: LoRA vs full-FT for the BT-RM half of the pess setup?
- **2025-08-31** — `pess0.1_baseline` — full-FT RM → [yntmzx3a](https://wandb.ai/distill-llms/policy-evaluation/runs/yntmzx3a)
- **2025-10-11** — `lora_pess0.1` — LoRA RM (commit `8a70f6b9` *lora for rm*) → [ix1qdbtp](https://wandb.ai/distill-llms/policy-evaluation/runs/ix1qdbtp)

---

## Online PET (preference-embedded training)

`rlhf/grpo/online_pet.py` updates the RM online during policy training, using either top-k or all-responses, optionally with a reference-policy regularizer.

### Q: Reference-policy regularization on or off?
- **2025-07-23** — `onlinePET-qwen06B-Base-06BRM` — no ref → [6mlwj81t](https://wandb.ai/distill-llms/policy-evaluation/runs/6mlwj81t)
- **2025-07-23** — `onlinePET-qwen06B-Base-06BRM_try2` — with ref (rerun after `a17dc96` *annotate reference 06b rm*) → [oi2l9cg3](https://wandb.ai/distill-llms/policy-evaluation/runs/oi2l9cg3)
- **2025-07-24** — `onlinePET-qwen06B-Base-06BRM-ref-top1fromeach` — ref + top-1-per-prompt RM updates → [roswl3ct](https://wandb.ai/distill-llms/policy-evaluation/runs/roswl3ct)

### Q: Top-k sampling for the online RM update vs all responses?
- **2025-07-24** — `onlinePET-qwen06B-Base-06BRM-ref-top1fromeach` — top-1 from each prompt → [roswl3ct](https://wandb.ai/distill-llms/policy-evaluation/runs/roswl3ct)
- **2025-07-26** — `onlinePET-qwen06B-Base-06BRM-noref-AdamW-all-responses-2ep1` — all responses → [6i70oe13](https://wandb.ai/distill-llms/policy-evaluation/runs/6i70oe13)

### Q: Online PET training-epoch sweep (no-ref, all-responses, AdamW)?
- **2025-08-01** — `onlinePET-...-1ep_pessloss_1000` (also pess=1000) → [g0pdq8d1](https://wandb.ai/distill-llms/policy-evaluation/runs/g0pdq8d1) | sibling **08-03** [7v37w5d9](https://wandb.ai/distill-llms/policy-evaluation/runs/7v37w5d9)
- **2025-07-26 → 08-01** — 2-epoch siblings (commit lineage `6ac4f23c → 575e7c4e → 492a9c44 → de4266a5 → 10068130`): [6i70oe13](https://wandb.ai/distill-llms/policy-evaluation/runs/6i70oe13), [90b7xzhd](https://wandb.ai/distill-llms/policy-evaluation/runs/90b7xzhd), [u0w7xl4q](https://wandb.ai/distill-llms/policy-evaluation/runs/u0w7xl4q), [nsmn19q3](https://wandb.ai/distill-llms/policy-evaluation/runs/nsmn19q3), [p2bapnh6](https://wandb.ai/distill-llms/policy-evaluation/runs/p2bapnh6)
- **2025-08-04** — `onlinePET-...-4ep` — 4 epochs → [r3aenooz](https://wandb.ai/distill-llms/policy-evaluation/runs/r3aenooz)

### Q: With vs without pessloss in Online PET (1-epoch)?
- **2025-08-01** — `onlinePET-...-1ep_pessloss_1000` — pess=1000 → [g0pdq8d1](https://wandb.ai/distill-llms/policy-evaluation/runs/g0pdq8d1)
- (compare against any 2ep no-pess sibling; no clean pess-vs-no-pess pair exists at the same epoch count, but the 2ep-no-pess [nsmn19q3](https://wandb.ai/distill-llms/policy-evaluation/runs/nsmn19q3) is the closest)

### Q: Eval-time RM family — Qwen-0.6B vs Qwen3-Embedding-8B vs Gemma-2-2B Ray?
Same Online PET training run, different `secondary_rm_name` / training-RM at eval.
- **2025-07-23** — `onlinePET-qwen06B-Base-06BRM` — 0.6B Qwen RM → [6mlwj81t](https://wandb.ai/distill-llms/policy-evaluation/runs/6mlwj81t)
- **2025-07-22** — `online_pet_base_policy_8B_rm` — Qwen3-Embedding-8B → [t7xsv7ka](https://wandb.ai/distill-llms/policy-evaluation/runs/t7xsv7ka), [a072flrv](https://wandb.ai/distill-llms/policy-evaluation/runs/a072flrv)
- **2025-08-04** — `onlinePET-qwen06B-Base-06BRM-gemma2Bray` — Gemma-2-2B Ray RM → [kft0e8o9](https://wandb.ai/distill-llms/policy-evaluation/runs/kft0e8o9)

### Other / standalone
- **2025-07-20** — `online_pet_ref_reward_qwen06B` — first Online PET eval (uses the `b6dfd51` reference-reward annotated dataset) → [axiyt8ai](https://wandb.ai/distill-llms/policy-evaluation/runs/axiyt8ai)

---

## Adversarial RM (multi-step)

### Q: Does iterating Adv-RM steps (1 → 2 → 3) keep improving over the gold?
Same Qwen3-Embedding-8B base; each step trains a new RM/policy on the previous step's outputs.
- **2025-07-15** — `AdvRM-step1-qwen06B_with_qwen8B-embedding_RM` → [4nasl2w0](https://wandb.ai/distill-llms/policy-evaluation/runs/4nasl2w0)
- **2025-07-16 / 17** — `AdvRM-step2-qwen06B_with_qwen8B-embedding_RM` → [2wml3liw](https://wandb.ai/distill-llms/policy-evaluation/runs/2wml3liw), `..._try_2` → [gbua7ydc](https://wandb.ai/distill-llms/policy-evaluation/runs/gbua7ydc)
- **2025-07-18** — `AdvRM-step3-qwen06B_with_qwen8B-embedding_RM` → [tbiz9ief](https://wandb.ai/distill-llms/policy-evaluation/runs/tbiz9ief)

---

## PAR-GRPO and RRM-GRPO

### Q: PAR vs RRM with the same gold?
- gold=URM-LLaMa-8B: **2025-06-23** `PAR_train_Ray_gold_URM` [qkyg75ht](https://wandb.ai/distill-llms/policy-evaluation/runs/qkyg75ht) vs `RRM_gold_URM` [296uqyzq](https://wandb.ai/distill-llms/policy-evaluation/runs/296uqyzq)
- gold=QRM-Gemma-27B: **2025-06-24** `PAR_train=RAY_gold=QRM_20250624_131145` [12vc5zav](https://wandb.ai/distill-llms/policy-evaluation/runs/12vc5zav) vs `RRM_gold_QRM_20250624_111417` [c4lokvjt](https://wandb.ai/distill-llms/policy-evaluation/runs/c4lokvjt)

### Q: Is gold-RM choice (URM-Llama vs QRM-Gemma) ranking-consistent across train-RM types?
- PAR: URM gold → [qkyg75ht](https://wandb.ai/distill-llms/policy-evaluation/runs/qkyg75ht), QRM gold → [12vc5zav](https://wandb.ai/distill-llms/policy-evaluation/runs/12vc5zav)
- RRM: URM gold → [296uqyzq](https://wandb.ai/distill-llms/policy-evaluation/runs/296uqyzq), QRM gold → [c4lokvjt](https://wandb.ai/distill-llms/policy-evaluation/runs/c4lokvjt)

---

## Initial PPO bring-up (pre-GRPO)

- **2025-05-27** — `PPO_gold=Ray_train=Qwen` — first eval, len3000, 5e-6 BT-RM → [wgzoe8id](https://wandb.ai/distill-llms/policy-evaluation/runs/wgzoe8id)

---

## LLM-as-judge head-to-head evals (older policies)

These don't roll out new policies — they take previously-trained PAR/RRM/qwen-base policies and have a Deepseek/R1 judge compare their generations against the dataset's chosen response.

### Q: Does the judge model matter (Deepseek-V3 vs Deepseek-R1)?
- PAR-policy: `PAR_vs_chosen_deepseek_v3_judge` (V3) → [gdxmbwvs](https://wandb.ai/distill-llms/policy-evaluation/runs/gdxmbwvs); no R1 sibling
- RRM-policy: `RRMvsChosen_Deepseek_v3_judge` (V3, with retries from `b519573`) → [kpqxh5sc](https://wandb.ai/distill-llms/policy-evaluation/runs/kpqxh5sc) vs `rrm_vs_chosen_deepseek_r1` (R1) → [jbul4zew](https://wandb.ai/distill-llms/policy-evaluation/runs/jbul4zew)

### Q: Does enabling tie-handling change the judge's verdicts?
- **2025-06-30** — `rrm_vs_chosen_deepseek_r1` — no ties → [jbul4zew](https://wandb.ai/distill-llms/policy-evaluation/runs/jbul4zew)
- **2025-07-01** — `rrm_vs_chosen_deepseek_r1_ties` — ties enabled → [ow2qs55n](https://wandb.ai/distill-llms/policy-evaluation/runs/ow2qs55n)

### Q: Which RM trained the qwen-base policy that wins more often vs chosen?
- **2025-07-04** — `qwen_base_ray_vs_chosen_r1` — policy trained with Ray-Gemma-2B RM → [le0m1gea](https://wandb.ai/distill-llms/policy-evaluation/runs/le0m1gea)
- **2025-07-05** — `qwen_base_qwen_helpsteer_vs_chosen_r1` — policy trained with Qwen-0.6B helpsteer RM → [bd4qyj4a](https://wandb.ai/distill-llms/policy-evaluation/runs/bd4qyj4a)
