# Papers to Compare Against (Prioritized 2025-2026, with Notable 2024 Additions)

Last updated: 2026-02-21.

Scope used here: papers directly relevant to reward modeling, reward over-optimization/reward hacking, and RLHF/GRPO-style policy training + evaluation pipelines.

## 1) Reward Model Overoptimisation in Iterated RLHF (2025)
Link: https://arxiv.org/abs/2505.18126

- Base policy model:
  - `pythia-410m` (SFT on AlpacaFarm SFT split), then PPO.
- Base reward model(s):
  - Proxy RMs initialized from `pythia_70m_sft` and `pythia-160m` (randomly initialized reward head).
  - Gold RM for simulated preference labeling/evaluation: `Reward-Model-AlpacaFarm-Human` (7B).
- Training dataset:
  - AlpacaFarm (`sft`/`preference`/`unlabeled`/`val` splits).
  - Preference labels are simulated by the 7B gold RM.
  - 1000 preference instructions per iteration; PPO on unlabeled split.
- Evaluation model/judge:
  - Gold RM score + KL + MMD-based proxy-vs-gold reward-distribution distance.
  - No external LLM-as-judge used.
- Main reported scores:
  - Mean/Std at end of 4th iteration (Table `tab:variance`):
    - `Concat Data / Policy from SFT`: `0.4477 ± 0.0653` (best mean)
    - `Take last Data`: `0.3572 ± 0.0406`
    - `Ensemble`: `0.3136 ± 0.0515`
    - `Weight Average`: `0.3035 ± 0.1248`
    - `Worst-Case Optimisation`: `0.2942 ± 0.0450`
    - `Sample`: `0.2761 ± 0.0381`
    - `Concat Data + LITI`: `0.1991 ± 0.1678`
    - `Sample + Take last Policy`: `-0.0632 ± 0.1055`
- Baselines/details/scores:
  - Preference-data transfer baselines: `Take last`, `Sample`, `Concat` (see above table).
  - RM-combination baselines: `Ensemble`, `Weight Average`, `Worst-Case` (see above table).
  - Policy init baselines: `From SFT`, `LITI`, `Take last policy` (aggregated in above table).
- Missing details:
  - Most KL-reward curves are plot-only (no full per-KL numeric table).

---

## 2) Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization (BSPO, 2025)
Link: https://arxiv.org/abs/2503.18130

- Base policy model:
  - `Alpaca-7B` as initial actor model.
- Base reward model(s):
  - Gold RM: `Llama3-8B`, trained on `57k` binarized UltraFeedback preference pairs.
  - Proxy/ScoreLM models at multiple scales: GPT2-large (`774M`), TinyLlama (`1.1B`), ShearedLlama (`2.7B`).
- Training dataset:
  - UltraFeedback preference dataset (main); `30k` re-annotated pairs for proxy RM training.
  - Note: AlpacaFarm with `20k` GPT-4-annotated preference data appears only in **Appendix D.9** as a supplementary experiment, not in the main pipeline.
- Evaluation model/judge:
  - Synthetic setting: gold-RM-based curves (proxy vs gold).
  - Non-synthetic setting: GPT-4o emulating human eval + win-rate matrix + Elo fitting.
- Main reported scores:
  - Non-synthetic Elo (Table `tab:elo`):
    - `Initial`: `1178.30`
    - `PPO:79`: `1181.77`
    - `PPO:158`: `1164.17`
    - `WCO:79`: `1201.77`
    - `WCO:158`: `1195.39`
    - `BSPO:79`: `1224.92`
    - `BSPO:158`: `1253.68` (best)
  - DPO/RPO comparison (Table `tab:DPO`, win-rate style):
    - `DPO`: vs Alpaca-7B `0.6592`, vs PPO `0.6226`, vs BSPO `0.2627`
    - `RPO`: vs Alpaca-7B `0.7058`, vs PPO `0.6664`, vs BSPO `0.297`
- Baselines/details/scores:
  - Main baselines: `Standard PPO`, `KL-Penalty PPO`, `CPPO`, `ENS-UWO`, `ENS-WCO`.
  - `DPO` and `RPO` appear only in **Appendix D.8** as a supplementary cross-paradigm comparison, not as formal baselines.
  - Pairwise non-synthetic win-rate matrix (Table `tab:win_rate`) available for Initial/PPO/WCO/BSPO at 79/158 steps.
- Missing details:
  - Several core synthetic results are curve-only (not fully tabulated as single scalars).

---

## 3) The Energy Loss Phenomenon in RLHF: A New Perspective on Mitigating Reward Hacking (EPPO, 2025)
Link: https://arxiv.org/abs/2501.19358

- Base policy model(s):
  - `Llama3-8B`, `Llama2-7B`, `Mistral-7B`, `DeepSeek-7B`.
- Base reward model(s):
  - Standard RM setup from prior RLHF pipeline (built from SFT backbones; linear reward head).
  - RM-robustness baselines also include ERM/WARM/ODIN/InfoRM variants.
- Training dataset:
  - General dialogue SFT: ShareGPT. RM + RL training: Anthropic-HH (Helpful + Harmless splits).
  - Summarization: Reddit TL;DR (all stages).
  - Note: AlpacaFarm is used as an **OOD evaluation set only**, not for training.
- Evaluation model/judge:
  - GPT-4 pairwise judging (main), plus Claude-3.5 and human eval in supplementary.
  - InfoRM used as hacking identifier for summarization.
- Main reported scores:
  - Llama3-8B, EPPO vs baselines (Win/Tie/Lose, GPT-4):
    - vs `PPO w/KL` on AlpacaFarm: `51/29/20`
    - vs `PPO w/LP` on AlpacaFarm: `58/26/16`
    - vs `PPO w/KL` on TL;DR: `57/25/18`
    - vs `PPO w/LP` on TL;DR: `46/32/22`
  - EPPO + RM-method compatibility (examples):
    - `EPPO+ODIN vs ODIN` on AlpacaFarm: `62/26/12`
    - `EPPO+InfoRM vs InfoRM` on TL;DR: `44/36/20`
- Baselines/details/scores:
  - RL baselines: `SFT`, `PPO`, `PPO w/KL`, `PPO w/LP`.
  - RM baselines: `ERM-Mean`, `ERM-WCO`, `ERM-UWO`, `WARM`, plus ODIN/InfoRM comparisons.
  - Cross-model robustness table also reported for Llama2/Mistral/DeepSeek with full W/T/L breakdown.
- Missing details:
  - Some setup details (exact RM backbone mapping per task) are referenced via prior work and not fully re-specified inline.

---

## 4) Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization (2025)
Link: https://arxiv.org/abs/2505.12763

- Base policy model(s):
  - `MetaMATH-Mistral-7B`, `Llama3-8B-Instruct`; also `WizardMATH-7B-v1.1` in BoN table.
- Base reward model(s):
  - 14 math RMs (classifier RMs + PRMs).
  - Gold RM for overoptimization measurement: `Skywork-o1-Open-PRM-Qwen2.5-7B`.
- Training/eval dataset:
  - RM evaluation set built from `MATH500`.
  - Downstream eval on `MATH500` and `Gaokao-math` (and `SAT-math` in BoN table).
  - Also compares benchmark-design variants (human/GPT-4/GPT-4o/random/model-sourced chosen/rejected responses).
- Evaluation model/judge:
  - Primary lens: degree of overoptimization `gamma` (`gamma_gold`, `gamma_oracle`) + downstream BoN/PPO outcomes.
  - No LLM-as-judge is the primary metric for main math conclusions.
- Main reported scores:
  - BoN (`n=256`) with MetaMATH-Mistral-7B / WizardMATH-7B-v1.1 (Table `main_bon_results`):
    - Baseline BoN `n=1`: `31.80/9.49/54.46` and `33.60/11.80/64.36` (MATH500/Gaokao/SAT).
    - Strong PRM examples:
      - `Skywork-o1-Open-PRM-Qwen2.5-7B`: `56.2/28.2` BoN + `29.4` PPO on MetaMATH; `55.0/31.0` BoN on Llama3.
      - `Qwen2.5-Math-PRM-7B`: `52.8/22.3` BoN + `30.2` PPO on MetaMATH; `52.8/27.4` BoN on Llama3.
    - Strong classifier RM examples:
      - `internlm2-7b-reward`: BoN `46.0/20.8`, PPO `29.4` (MetaMATH); BoN `45.2/24.6`, PPO `28.9` (Llama3).
  - Correlation of benchmark design vs overoptimization (Table `settings`):
    - Best designs reach very high `r^2` with `gamma`:
      - Design `O` (random,3 vs random,3; 3:3 acc): up to `0.943` (`MetaMATH gamma_oracle`) and `0.841` (`Llama3 gamma_oracle`).
      - Design `J` (GPT-4o* vs random,3; 1:3 acc): up to `0.915` (`MetaMATH gamma_oracle`).
- Baselines/details/scores:
  - Includes classifier RMs (`ArmoRM`, `Skywork-Reward`, `internlm2-*`, `GRM-*`, `Eurus`, `Beaver`, `oasst-rm`) and PRMs (`Skywork-o1 PRM`, `Qwen2.5-Math-PRM`, `Math-Shepherd`, `llemma PRM`, `ReasonEval`).
  - Also baseline benchmark designs derived from RewardBench / RM-Bench style settings.
- Missing details:
  - Focus is RM evaluation methodology; not a single end-to-end RLHF policy paper with one canonical final policy score.

---

## 5) Inference-Time Reward Hacking in Large Language Models (2025)
Link: https://arxiv.org/abs/2506.19248

- Base policy model(s):
  - Verifiable-reward setup uses existing sampled candidates (PPE data).
  - Human-preference setup uses Pythia-1.4B (AlpacaFarm-finetuned, no RLHF/DPO) as reference generator.
- Base reward model(s):
  - Verifiable setup proxy RMs: `InternLM2-1.8B`, `Llama-3-Offset-Bias-RM-8B`, `Skywork-Llama-3.1-8B`.
  - Human-preference setup: gold RM `AlpacaRM`; proxy RMs are Pythia-44M trained on constructed preference pairs.
- Training/eval dataset:
  - PPE benchmark pairs (MMLU-Pro, MATH, GPQA) with multiple candidate responses.
  - `tlc4418/gold_labelled_gens`: 1000 AlpacaFarm-val prompts, 12,600 responses per prompt, AlpacaRM-labeled; proxy-RM training sizes 10k/20k/46k/80k with 0% and 25% label-noise settings.
- Evaluation model/judge:
  - Verifiable true-reward accuracy (0/1 correctness).
  - Human-preference setup uses AlpacaRM-labeled gold reward.
- Main reported scores:
  - BoP theoretical/empirical KL gap to tilted-optimal distribution reported as very small, bounded around `8e-4` when reward-matched.
  - Core benchmark outcomes are mostly reported as curves (accuracy/reward vs sample count/temperature), not consolidated into one scalar table.
- Baselines/details/scores:
  - Inference-time methods: `BoN`, `SBoN`, `BoP` (and appendix `SBoP`).
  - Main comparison is standard (default-parameter) methods vs `HedgeTune`-tuned operating points.
- Missing details:
  - No single consolidated numeric leaderboard table for all methods/datasets in the main paper text.

---

## 6) Outcome Accuracy is Not Enough: Aligning the Reasoning Process of Reward Models (2026)
Link: https://arxiv.org/abs/2602.04649

- Base policy model(s):
  - RLHF policy experiment starts from `Qwen-30B-A3B-Base` (+small SFT), then GRPO-guided alignment.
- Base reward model(s):
  - GenRM training on `Qwen3-14B` and `Qwen3-30B-A3B`.
  - Outcome-only baselines use same base models with outcome-only reward.
- Training/eval datasets:
  - `HelpSteer3-Atomic` (1000 examples: 250/domain from HelpSteer3 with atomic rationale decomposition).
  - `CW-Atomic` creative-writing benchmark (separate annotator pool/domain).
  - RM evaluation: RM-Bench + JudgeBench.
  - Policy alignment eval: Arena Hard v2 (Hard Prompt + Creative Writing).
- Evaluation model/judge:
  - MetaJudge evaluator: mainly Qwen3 Plus (with cross-checks vs DeepSeek-R1; high consistency).
  - RLHF downstream eval on Arena Hard v2.
- Main reported scores:
  - RM-Bench/JudgeBench total average:
    - `Qwen3-30B-A3B (Outcome-Only)`: `80.3`
    - `Qwen3-30B-A3B (Ours)`: `84.6` (SOTA in this table)
    - `Qwen3-14B (Outcome-Only)`: `76.8`
    - `Qwen3-14B (Ours)`: `82.9`
  - Arena Hard v2:
    - `SFT`: Hard Prompt `12.61`, Creative Writing `41.12`
    - `Outcome-Only RM`: `19.10`, `62.00`
    - `Ours`: `21.22`, `69.08`
  - Rationale consistency ablation:
    - HelpSteer3-Atomic: Base `0.2505`, Outcome-only `0.2108`, Ours `0.3718`
    - CW-Atomic: Base `0.2385`, Outcome-only `0.1677`, Ours `0.2526`
- Baselines/details/scores (selected, Total Avg from main table):
  - LLM-as-judge: GPT-4o `66.2`, Claude-3.5-Sonnet `62.9`, DeepSeek-R1-0528 `75.8`
  - Scalar RMs: Skywork-Gemma-27B `67.8`, Skywork-Llama-3.1-8B `66.3`
  - GenRMs: RM-R1-Distilled-Qwen-32B `81.4`, RM-R1-Distilled-Qwen-14B `79.8`, RRM-32B `74.4`, Nemotron-Super `80.0`, RewardAnything-8B `72.9`, GRAM-R2 `83.4`, Principles-Qwen32B `83.8`
  - Controlled baselines: same-architecture Outcome-Only rows above.
- Missing details:
  - Some external baseline rows are imported from prior papers (not always retrained under identical conditions in this paper).

---

## 7) Adversarial Reward Auditing for Active Detection and Mitigation of Reward Hacking (2026)
Link: https://arxiv.org/abs/2602.01750

- Base policy model:
  - `Llama-2-7B` SFT model used as baseline policy initialization.
- Base reward model(s):
  - Frozen proxy RM used in ARA; specific RM checkpoint name is not clearly specified in the main extracted sections.
- Training/eval datasets:
  - Sycophancy: RM trained on Anthropic HH-RLHF; eval on SycophancyEval.
  - Length bias: follows ODIN setting.
  - Code gaming: custom visible-unit-test coding environment.
- Evaluation model/judge:
  - Sycophancy gold metric uses GPT-4 factual-accuracy check against golden answer.
  - Code-gaming rate also uses GPT-4 classification.
  - Utility metrics: helpfulness, ROUGE-L, Pass@1.
- Main reported scores (Table `main_results`):
  - `SFT (No RLHF)`: sycophancy `36.2`, helpfulness `41.3`, length `148`, ROUGE-L `21.4`, gaming `4.2`, Pass@1 `28.5`
  - `PPO`: `72.4`, `76.8`, `347`, `23.1`, `61.3`, `34.2`
  - `PPO w/KL`: `58.3`, `68.2`, `268`, `22.8`, `48.5`, `31.8`
  - `ODIN`: `51.6`, `63.4`, `195`, `23.4`, `42.1`, `33.5`
  - `InFoRM`: `47.2`, `62.1`, `208`, `23.2`, `39.8`, `33.1`
  - `RM Ensemble`: `52.8`, `64.6`, `224`, `23.0`, `44.3`, `34.0`
  - `Filtering (R > mu+2sigma)`: `55.1`, `50.3`, `241`, `22.6`, `46.7`, `32.4`
  - `ARA (Ours)`: `38.4`, `77.2`, `162`, `24.1`, `19.6`, `35.8`
- Baselines/details/scores:
  - Unmitigated PPO, KL-regularized PPO, ODIN, InFoRM, RM ensemble, filtering intervention.
  - Additional transfer matrices reported for hacking transfer, auditor AUC transfer, and mitigation transfer.
- Missing details:
  - Code-gaming dataset citation is unresolved (`\cite{?}` in source).
  - Proxy RM checkpoint identity is not explicitly named in extracted main setup text.

---

## 8) Factored Causal Representation Learning for Robust Reward Modeling in RLHF (CausalRM, 2026)
Link: https://arxiv.org/abs/2601.21350

- Base policy model(s):
  - Math: `Qwen2.5-Math-7B` for RM + PPO.
  - Dialogue: `Qwen2.5-7B` SFT backbone (SFT on ShareGPT), then PPO.
- Base reward model(s):
  - CausalRM built on same pretrained/SFT backbones with factorized latent module.
  - Baselines: Standard RM, GoalRM, InfoRM.
- Training/eval datasets:
  - Math RM+RLHF training: OpenMathInstruct-1 preference setup (following GoalRM setup).
  - Math OOD eval: Algebra222, GSM-Hard, ASDiv, MAWPS, SVAMP.
  - Dialogue RM+RLHF training: Anthropic-HH.
  - Dialogue OOD eval: MT-Bench, PKU-SafeRLHF, SHP, TruthfulQA.
- Evaluation model/judge:
  - RM quality: pairwise accuracy.
  - Math RLHF: final-answer accuracy.
  - Dialogue RLHF: pairwise win/tie/lose judged by `Qwen3-Max`.
- Main reported scores:
  - RM pairwise accuracy (Avg):
    - Math ID/OOD: Standard `67.9/83.0`, GoalRM `68.3/82.2`, InfoRM `66.1/82.5`, CausalRM `70.1/85.6`.
    - Dialogue ID/OOD: Standard `70.6/59.7`, GoalRM `71.1/60.7`, InfoRM `70.8/59.8`, CausalRM `72.3/62.3`.
  - Math downstream RLHF (Avg ID/OOD):
    - SFT `66.9/80.0`
    - Standard RM `67.8/83.1`
    - GoalRM `72.5/88.5`
    - InfoRM `48.0/49.9`
    - CausalRM `74.0/89.6` (best)
  - Dialogue downstream RLHF (CausalRM policy vs opponents, Avg Win/Tie/Lose):
    - vs SFT: ID `72.2/21.6/6.2`, OOD `60.5/28.9/10.6`
    - vs Standard RM: ID `54.8/33.1/12.1`, OOD `51.3/32.6/16.1`
    - vs GoalRM: ID `42.3/41.6/16.1`, OOD `31.6/50.6/17.8`
    - vs InfoRM: ID `45.5/37.6/16.9`, OOD `38.7/41.8/19.5`
  - Sycophancy-artifact robustness (hacked tests, Avg pairwise accuracy):
    - Standard RM: ID `59.2`, OOD `53.9`
    - GoalRM: ID `62.2`, OOD `56.9`
    - InfoRM: ID `66.8`, OOD `57.3`
    - CausalRM: ID `70.6`, OOD `61.2` (best)
- Baselines/details/scores:
  - `SFT`, `Standard RM`, `GoalRM`, `InfoRM` with detailed benchmark breakdown in paper tables.
- Missing details:
  - No human annotation protocol changes are introduced; mostly benchmarked on existing public datasets.

---

## Notable 2024 Top-Conference Papers (Added)

## 9) Confronting Reward Model Overoptimization with Constrained RLHF (ICLR 2024)
Link: https://openreview.net/forum?id=mp1AstNFvQ

- Base policy model:
  - `GPT-2` (dialogue generation setup).
- Base reward model(s):
  - Composite proxy reward with 2 components:
    - `METEOR` reward (`r_met`) vs reference response.
    - Intent reward (`r_int`) from a fine-tuned `RoBERTa` intent classifier.
  - Evaluation target is an estimated composite evaluation score (not an external LLM judge).
- Training dataset:
  - `DailyDialog` dialogue dataset (conversation transcript continuation setup).
  - No explicit human re-annotation in this paper; reward components are automatic metrics/classifier outputs.
- Evaluation model/judge:
  - Internal evaluation-score estimator built from lexical/diversity metrics (no GPT-4/human judge table).
  - Constraint satisfaction on proxy-point thresholds (METEOR and intent).
- Main reported scores:
  - Table (hyperparameter) proxy thresholds used in constrained runs: `theta_meteor = 0.23`, `theta_intent = 0.48`.
  - Main algorithm comparison outcomes are primarily plot-based (`Figure 5.1` / `Figure 5.3`) rather than a numeric leaderboard table.
  - Inference from plot: `xi-PPO` is best final evaluation-score curve, followed by `mu-PPO`; both beat `PPO`/`PPO-SAT`/`All-PPO`.
- Baselines/details/scores:
  - `PPO`, `PPO-SAT`, `mu-PPO`, `All-PPO`, `xi-PPO`.
  - Reported primarily as training curves over steps/KL, not consolidated table values.
- Missing details:
  - No single table with exact final evaluation scores per method.
  - No external LLM-as-judge or human pairwise benchmark in the main results.

---

## 10) Reward Model Ensembles Help Mitigate Overoptimization (ICLR 2024)
Link: https://openreview.net/forum?id=Vuw5St1x4r

- Base policy model:
  - `Pythia-1.4B` policy.
- Base reward model(s):
  - Proxy RMs built from Pythia backbones:
    - 14M / 70M / 1.4B language-model backbones (reward models reported as ~7M / 44M / 1.3B after head changes).
  - Gold RM: AlpacaFarm human-preference RM (`7B`).
- Training dataset:
  - `AlpacaFarm` pipeline:
    - `10k` SFT split for policy/reward SFT.
    - Preference data generated by sampling two responses per prompt and labeling with the 7B gold RM.
    - Main proxy-RM training size: `46k` prompts.
    - Experiments with `0%` and `25%` label-noise.
- Evaluation model/judge:
  - Gold-RM score vs KL (BoN and PPO settings).
  - Additional win-rate evaluation of final policies against single-RM policies.
  - No external LLM-as-judge.
- Main reported scores:
  - Final policy win-rates vs single-RM baselines (Table 6, 44M RMs):
    - `Mean`: BoN `54.9 +/- 0.9` (no noise), `57.8 +/- 1.3` (25% noise); PPO `58.1 +/- 3.3` (no noise), `60.2 +/- 3.5` (25% noise)
    - `WCO`: BoN `57.1 +/- 0.6`, `58.3 +/- 0.8`; PPO `59.4 +/- 3.3`, `62.2 +/- 2.9`
    - `UWO (lambda=0.5)`: BoN `57.2 +/- 0.9`, `58.2 +/- 1.0`; PPO `60.2 +/- 3.4`, `63.0 +/- 3.1` (best in this table)
  - Scale-transfer win-rates (Table 7):
    - `UWO`: BoN `61.3 +/- 1.4` (7M), `73.8 +/- 0.8` (1.3B); PPO `59.0 +/- 3.7` (7M), `63.1 +/- 3.2` (1.3B)
    - `WCO`: BoN `58.8 +/- 1.4` (7M), `71.2 +/- 1.2` (1.3B); PPO `57.9 +/- 3.7` (7M), `62.9 +/- 2.9` (1.3B)
- Baselines/details/scores:
  - `Single RM` optimization, `Mean ensemble`, `WCO`, `UWO` under both BoN and PPO.
  - RM-size and data-size sweeps reported; overoptimization curves mainly plot-based.
- Missing details:
  - Many gold-RM absolute-score comparisons are figure-only (not all tabulated in scalar tables).

---

## 11) WARM: On the Benefits of Weight Averaged Reward Models (ICML 2024)
Link: https://proceedings.mlr.press/v235/rame24a.html

- Base policy model:
  - RL experiments: `PaLM-XS` policy and value models initialized from the same SFT model.
  - BoN experiments also include a `T5` SFT policy setup.
- Base reward model(s):
  - `WARM`: weight average of multiple `PaLM-XXS` RMs from diverse fine-tunings.
  - Baselines: individual RMs (`phi1`, `phi2`) and prediction ensembling (`ENS`).
  - Additional control RM: `PaLM-XS` pointwise RM with reported `80.1%` OOD accuracy on `D_ood`.
- Training dataset:
  - Reddit `TL;DR` summarization benchmark.
  - Preference labels generated by a `PaLM-L` model with CoT prompting (RLAIF-style setup).
  - OOD set `D_ood`: `92k` pairwise comparisons.
  - Corruption experiments with `25%` label corruption.
- Evaluation model/judge:
  - Control reward on held-out/OOD settings.
  - Oracle pairwise preference metric (AI-labeler oracle setup).
  - No GPT-4/human-annotation table as primary endpoint.
- Main reported scores:
  - Reported in text: RL policy trained with WARM has `79.4%` win rate against RL policy trained with a single RM.
  - RL oracle metric (text around Figure 9): `WARM M=6` reaches `99.8%` win rate vs SFT after `3500` steps.
  - BoN oracle metric (Figure 7 text): WARM-selected summaries reach up to `92.5%` win rate vs random SFT summary selection.
- Baselines/details/scores:
  - `WARM M=2/6/10`, `ENS M=2`, and individual RMs (`phi1`, `phi2`) in both clean and 25%-corrupted settings.
  - RL and BoN comparisons across KL ranges.
- Missing details:
  - Core comparative results are mostly figure-driven; no single unified leaderboard table across all settings.

---

## 12) InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling (NeurIPS 2024)
Link: https://proceedings.neurips.cc/paper_files/paper/2024/hash/352d7f6af8b19fe6354f4aeec204b0f4-Abstract-Conference.html

- Base policy model:
  - Real-world setting starts from `Vicuna-7B-v1.5` as SFT model.
  - Simulation setting follows AlpacaFarm-style RLHF from a 7B-scale SFT model.
- Base reward model(s):
  - `InfoRM` (IB-based reward modeling).
  - Baselines: `Standard RM`, `Standard RM w/ KL`, `Ensemble RM`, `WARM`.
  - Reported scaling experiments include RM sizes up to `7B` (also 70M/440M/1.4B).
- Training dataset:
  - Simulation: `AlpacaFarm` (`10k` SFT demos, `20k` preference training, plus unlabeled split for RL).
  - Real-world: `Anthropic-HH` (Helpful/Harmless) and `TL;DR`.
  - OOD evaluation includes `AlpacaFarm` validation (and additional datasets in appendices).
- Evaluation model/judge:
  - GPT-4 pairwise win/tie/lose evaluation (main table).
  - Additional analyses: CSI and latent-space overoptimization diagnostics.
- Main reported scores (Table 1, win/tie/lose; InfoRM-variant listed first):
  - `InfoRM vs SFT`: Anth-Helpful `57.0/27.0/16.0`, Anth-Harmless `57.1/26.2/16.6`, AlpacaFarm `48.9/30.8/20.2`, TL;DR `73.1/17.3/9.5`
  - `InfoRM vs Standard RM`: `54.5/33.5/12.0`, `54.2/32.3/13.3`, `45.1/31.4/23.5`, `70.4/17.9/11.6`
  - `InfoRM vs Standard RM w/ KL`: `49.0/31.5/19.5`, `44.3/44.2/11.4`, `38.5/35.2/26.3`, `68.6/21.5/9.8`
  - `InfoRM+Ensemble vs Ensemble`: `48.7/35.7/15.6`, `52.5/35.1/12.4`, `41.2/38.2/20.6`, `63.3/30.1/6.6`
  - `InfoRM+WARM vs WARM`: `47.6/35.2/17.2`, `67.9/24.2/7.9`, `37.9/41.0/21.1`, `65.9/17.2/16.9`
- Baselines/details/scores:
  - `SFT`, `Standard RM`, `Standard RM+KL`, `Ensemble RM`, `WARM`, plus combination variants (`InfoRM+Ensemble`, `InfoRM+WARM`).
- Missing details:
  - Some architecture-level RM backbone details differ across sections and are not always fully specified per table row.
  - Several simulated overoptimization curves are not fully tabulated as single scalars.

---

## 13) Provably Mitigating Overoptimization in RLHF: Your SFT Loss Is Implicitly a Reward Model (NeurIPS 2024)
Link: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f25d5f089f2540d7d78eef33db22ce2a-Abstract-Conference.html

- Base policy model:
  - Two model series:
    - `zephyr-7b-beta` pipeline.
    - `zephyr-7b-gemma` pipeline.
  - Practical method (`RPO`) built as a modification to DPO-style policy training.
- Base reward model(s):
  - Theory uses a general reward-function class.
  - Practical implementation is reward-model-free via reparameterization; RPO objective = DPO preference term + imitation/SFT regularizer.
- Training dataset:
  - Beta series: `UltraFeedback` preference data (~60k).
  - Gemma series: `Argilla-DPO-Mix-7K`.
  - Pairwise GPT-4 eval reported on 150-test-example subsets per series.
- Evaluation model/judge:
  - GPT-4 pairwise win-rate matrices (RPO vs DPO vs Ref).
  - MT-Bench score (GPT-4 judged, 1-10 scale).
  - AlpacaEval 2.0 `LC win rate` and `win rate` vs GPT-4 reference setup.
- Main reported scores:
  - Pairwise GPT-4 win rates (Table 1):
    - Beta: `RPO vs DPO = 56.0`, `DPO vs RPO = 44.0`; `RPO vs Ref = 79.0`, `DPO vs Ref = 77.3`.
    - Gemma: `RPO vs DPO = 54.0`, `DPO vs RPO = 46.0`; `RPO vs Ref = 71.7`, `DPO vs Ref = 67.3`.
  - Benchmarks (Table 2):
    - `RPO (beta)`: MT-Bench `7.381`, AlpacaEval2 LC `23.28`, win `21.01`
    - `DPO (beta)`: `7.278`, `21.15`, `17.27`
    - `RPO (gemma)`: `7.916`, `15.51`, `13.85`
    - `DPO (gemma)`: `7.688`, `15.36`, `13.69`
- Baselines/details/scores:
  - `DPO`, `Ref` model, and official `zephyr-beta-7b` / `zephyr-gemma-7b` checkpoints in benchmark table.
- Missing details:
  - Because the practical objective is RM-free, there is no separately named proxy RM checkpoint to report in the same way as PPO-RM papers.

---

## 14) Overcoming Reward Overoptimization via Adversarial Policy Optimization with Lightweight Uncertainty Estimation (NeurIPS 2024)
Link: https://openreview.net/forum?id=R6wcM6WQki

- Base policy model:
  - `Llama-7B` SFT-initialized policy for RL optimization.
- Base reward model(s):
  - Synthetic overoptimization setup:
    - Gold RM: `Vicuna-13B` (Anthropic HH), `Llama-13B` (TL;DR).
    - Proxy RM: `Llama-7B`.
  - Real-world setup: reward model and policy both initialized from `Llama-7B`.
- Training dataset:
  - `Anthropic-HH` and `TL;DR`.
  - Synthetic setup details:
    - Preference data split in half (one half RM training, one half policy optimization).
    - Gold-RM relabeling for proxy-RM training.
    - `30%` random mislabeling injected in synthetic preference pairs.
  - Real-world section uses human-preference training directly and evaluates on held-out prompts.
- Evaluation model/judge:
  - Synthetic: proxy-vs-gold reward dynamics and KL.
  - Practical pairwise eval: GPT-4 with position-swapped checks + human arbitration for inconsistent/tie cases.
  - Annotation workload reported: `200` Anthropic-HH prompts, `100` TL;DR prompts.
- Main reported scores (Table 1, first model listed first):
  - `AdvPO vs PPO`: Anth-HH `31.0/49.0/20.0` (Delta `+11.0`), TL;DR `75.0/7.0/18.0` (Delta `+57.0`)
  - `AdvPO vs PPO-ref`: `35.5/39.5/25.0` (Delta `+10.0`), `55.0/6.0/39.0` (Delta `+16.0`)
  - `AdvPO vs LWUN-s`: `36.0/39.5/24.5` (Delta `+11.5`), `67.0/3.0/30.0` (Delta `+37.0`)
  - `AdvPO vs ENS-s`: `43.0/26.5/30.5` (Delta `+12.5`), `77.0/3.0/20.0` (Delta `+57.0`)
  - `AdvPO vs LoraEns`: `65.5/15.5/19.0` (Delta `+46.5`), `84.0/0.0/16.0` (Delta `+68.0`)
- Baselines/details/scores:
  - `PPO`, `PPO-ref`, `ENS-s` (sample-wise uncertainty with 3x3B ensembles), `LoraEns`, `LWUN-s`, `AdvPO-noRef`.
- Missing details:
  - Main paper does not provide a single global scalar benchmark like MT-Bench/AlpacaEval for all methods; core result is pairwise preference tables plus reward-dynamics plots.

---

## 15) The Accuracy Paradox in RLHF: When Better Reward Models Don't Yield Better Language Models (ACL 2024 Findings)
Link: https://aclanthology.org/2024.findings-acl.97/

- Base policy model:
  - `T5-small`, `T5-base`, `T5-large` (SFT then PPO-based RLHF).
- Base reward model(s):
  - Task reward models based on `Longformer-base-4096` (relevance/factuality/completeness).
  - Independent evaluation RMs:
    - `R_phi1` relevance: accuracy `69.6`, F1 `68.5`
    - `R_phi2` factuality: accuracy `77.8`, F1 `67.5`
    - `R_phi3` completeness: accuracy `70.9` (F1 not reported)
- Training dataset:
  - `QA-FEEDBACK` (derived from ASQA).
  - Split reported: train/val/test = `3853 / 500 / 948`.
- Evaluation model/judge:
  - RM-based evaluation with independent high-accuracy reward models (no GPT-4/human judge in the main evaluation pipeline).
  - KL-divergence trend analysis during PPO.
- Main reported scores:
  - Reward-model training ranges yielding studied performance regimes (Table 1):
    - Factuality: steps `2-1256`, accuracy `0.64-0.77`
    - Relevance: steps `2-2852`, accuracy `0.49-0.69`
    - Completeness: steps `30-5730`, accuracy `0.44-0.70`
  - Central empirical claim (from main figures): best LM performance occurs with moderately accurate RMs, not the most accurate RMs.
- Baselines/details/scores:
  - Per task and per policy size: `Best-performing RM` (moderate-accuracy) vs `Most-accurate RM`.
  - Trends reported through 3D performance surfaces and reward/KL trajectories rather than consolidated scalar leaderboard tables.
- Missing details:
  - No single table of final LM task scores for every method/task/model combination.
  - No LLM-as-judge or human preference benchmark used for the primary conclusions.

---
## 16) Adversarial Training of Reward Models (Adv-RM, 2025 Preprint)
Link: https://arxiv.org/abs/2504.06141

- Base policy model:
  - `Llama-3.1-8B-Instruct` for adversarial-policy generation and downstream RLHF policy optimization.
- Base reward model(s):
  - Synthetic setup:
    - Gold RM: `Llama-3.1-Nemotron-70B-Reward`.
    - Proxy/target RMs: `Llama-3.1-8B-Instruct` reward models (`R_theta1`, `R_theta2`) trained with different random seeds.
  - Real setup attack targets: `Skywork-Reward-Gemma-2-27B`, `Llama-3.1-Nemotron-70B-Reward`, `Nemotron-4-340B-Reward`.
- Training dataset:
  - Synthetic RLHF setup built by relabeling `HelpSteer-2-Preferences` with the gold RM.
  - Adv-RM augmentation adds about `1000` adversarial preference pairs to the original RLHF preference dataset.
  - Held-out evaluation set: `128` prompts.
- Evaluation model/judge:
  - Synthetic: attack success via target-RM vs gold-RM disagreement criteria (`standard` and `strict` definitions).
  - Real attacks: human judges + `DeepSeek-R1` + `Llama-3 405B`.
  - Additional RM quality check: RewardBench scores.
- Main reported scores:
  - Synthetic attack success rates (Table 1, 128 prompts; standard/strict for train/test):
    - `Adv-RM`: `99.31/92.91` (train), `100.00/100.00` (test)
    - `RM Over-optimization`: `8.31/2.85` (train), `7.59/1.34` (test)
    - `Textfooler`: `9.57/0.00` (train), `18.8/0.00` (test)
    - `RRM`: `0.00/0.00` (train), `0.00/0.00` (test)
    - `StyleAdv`: `0.00/0.00` (train), `0.00/0.00` (test)
  - Real attack success (Table 2; first value = human eval):
    - `Skywork-Gemma-27B`: Adv-RM `100` vs RM-overopt `20`, Textfooler `19.61`
    - `Llama-Nemotron-70B`: Adv-RM `83.01` vs RM-overopt `23.64`, Textfooler `13.2`
    - `Nemotron-340B`: Adv-RM `78.85` vs RM-overopt `17.65`, Textfooler `17.86`
  - RewardBench (Table 4):
    - `Baseline`: `0.8329`
    - `Adv-RM`: `0.8399`
- Baselines/details/scores:
  - Attack baselines: `RM over-optimization`, `RRM`, `StyleAdv`, `Textfooler`.
  - Downstream policy baselines: conventional RLHF, ensemble-mean, ensemble-uncertainty (`mean-std`/UWO), RRM (policy curves in figures).
- Missing details:
  - Downstream policy performance in synthetic/real RLHF is primarily curve-based (`Figure 4`, `Figure 5`), not tabulated as a single final benchmark table.

---
## 17) Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs (GRM, NeurIPS 2024)
Link: https://arxiv.org/abs/2406.10216

- Base policy model:
  - BoN/PPO policy: `gemma-2B-it`.
  - Additional BoN RM-backbone experiments on `Mistral-7B-Instruct`.
- Base reward model(s):
  - Baseline classifier RM and GRM variants (`w/ sft`, `w/ dpo`, `w/ dpo-noref`), with 2B and 7B backbones.
  - Gold RM for BoN/PPO evaluation: `reward-model-Mistral-7B-instruct-Unified-Feedback`.
- Training dataset:
  - RM training: `Unified-Feedback` subsets (`400K`, `40K`, and RL subset `20K` for BoN/PPO).
  - OOD RM evaluation: `HHH-Alignment`, `MT-Bench Human Judgements`, `RewardBench`.
  - RL evaluation set: 1K held-out prompts from Unified-Feedback.
- Evaluation model/judge:
  - RM generalization: ID/OOD classification metrics and RewardBench group scores.
  - Policy evaluation: proxy-vs-gold BoN/PPO trajectories + GPT-4o pairwise eval for PPO output quality.
- Main reported scores:
  - Gemma-2B RM ID/OOD (Table 1: 400K; Table 2: 40K):
    - Baseline (`Unified/HHH/MT`): `72.1/73.4/71.2` (400K), `68.8/70.3/69.1` (40K)
    - `GRM w/ sft`: `73.2/79.8/73.4` (400K), `71.5/78.7/73.0` (40K)
  - RewardBench (Table 3/4, Mistral-7B base):
    - Baseline average: `76.3` (400K), `68.2` (40K)
    - GRM best variants: up to `79.5` (400K) and `78.6` (40K)
  - PPO pairwise eval vs baseline RM policy (Appendix Table 14; GPT-4o):
    - Gemma-2B base RM: Win/Tie/Loss = `68/5/27`
    - Mistral-7B base RM: Win/Tie/Loss = `73/6/21`
  - Large-scale full-parameter RewardBench (Table 5):
    - `GRM (8B)`: `87.0` average vs `FsfairX-LLaMA3-RM-8B` `84.7`.
- Baselines/details/scores:
  - Baselines include `frozen classifier`, `margin`, `label smoothing`, `3-model ensemble`, and standard classifier RM.
  - BoN/PPO comparisons use the same RM training data and policy initialization across methods.
- Missing details:
  - Core BoN/PPO overoptimization improvements are mostly shown as curves (no single scalar table for all KL checkpoints).

---
## 18) RRM: Robust Reward Model Training Mitigates Reward Hacking (ICLR 2025)
Link: https://arxiv.org/abs/2409.13156

- Base policy model:
  - DPO-aligned policies initialized from `Gemma-2-9B-it`.
  - BoN policies from sampled `Gemma-2-9B-it` responses.
- Base reward model(s):
  - `RM` baseline and `RRM` (augmented robust RM), both trained from `Gemma-2-9B-it`.
  - ODIN included as policy-time baseline in DPO comparison table.
- Training dataset:
  - RM/RRM training uses RLHFlow pair-preference mix (~700K pairs) from:
    - HH-RLHF Helpful (`115,396`)
    - SHP (`93,301`)
    - HelpSteer (`37,131`)
    - PKU-SafeRLHF (`26,874`)
    - UltraFeedback (`340,025`)
    - UltraInteract (`161,927`)
    - Distilabel-Capybara (`14,811`)
    - Distilabel-Orca (`6,926`)
  - RRM augmentation creates additional non-contextual preference triplets; filtered final training size reported as `2.4M`.
  - DPO policy data: on-policy responses generated from UltraFeedback prompts.
- Evaluation model/judge:
  - RM quality: RewardBench category accuracies.
  - Policy quality: `MT-Bench` and `AlpacaEval-2` (including length-controlled win rate, GPT-4 win rate, and response length).
- Main reported scores:
  - RewardBench RM accuracy (Table 2):
    - `RM`: Chat `97.77`, Chat-Hard `51.54`, Safety `78.54`, Reasoning `94.58`, Avg `80.61`
    - `RRM`: Chat `96.51`, Chat-Hard `65.57`, Safety `83.90`, Reasoning `90.62`, Avg `84.15` (`+3.54` avg)
  - Policy outcomes (Table 3):
    - DPO with RM: MT-Bench overall `7.27`, AlpacaEval2 LC `33.46`, WR `41.07`, length `2416`
    - DPO with ODIN: overall `8.39`, LC `48.29`, WR `37.13`, length `1559`
    - DPO with RRM: overall `8.31`, LC `52.49`, WR `43.31`, length `1723`
    - BoN N=64: RM LC `40.52` / WR `57.62` vs RRM LC `62.82` / WR `63.03`
- Baselines/details/scores:
  - RM baselines: vanilla RM.
  - Policy baselines: RM-induced DPO, ODIN-induced DPO, RM/RRM-induced BoN (N=8,64), and `-Neutrals` ablation.
- Missing details:
  - Multi-turn MT-Bench BoN is not reported (authors note BoN not evaluated there).
  - Some robustness analyses are distribution plots/perturbation sweeps rather than scalar benchmark tables.

---

# Dataset Cross-Reference: Which Papers Share Which Setup

For each dataset used in 2+ papers, listing the papers, their policy/RM models, RL algorithm, and evaluation method. This helps identify where a single experiment on our end can produce numbers comparable to multiple papers simultaneously.

---

## AlpacaFarm

Used in: **#1, #2 (appendix), #3 (eval only), #5, #10, #12**

| Paper | Role | Policy | RM(s) | Gold RM | RL Alg | Eval |
|---|---|---|---|---|---|---|
| #1 Iterated RLHF | Full pipeline (SFT+RM+RL) | `pythia-410m` | `pythia-70m`, `pythia-160m` | `AlpacaFarm-Human 7B` | PPO | Gold RM score + KL |
| #2 BSPO | **Appendix D.9 only** | `Alpaca-7B` | not specified for this split | not specified | PPO | Gold-RM curves |
| #3 EPPO | **OOD eval only** (not training) | `Llama3-8B` / `Llama2-7B` / `Mistral-7B` / `DeepSeek-7B` | — | — | — | GPT-4 W/T/L |
| #5 Inf-Time RH | Human-pref setup (BoN, not RL training) | `Pythia-1.4B` (SFT, no RLHF) | `Pythia-44M` (proxy) | `AlpacaRM 7B` | BoN/BoP (inference-time) | Gold RM reward |
| #10 Coste et al. | Full pipeline (SFT+RM+RL) | `Pythia-1.4B` | `Pythia-7M/44M/1.3B` | `AlpacaFarm-Human 7B` | PPO + BoN | Gold RM score + KL, win rate |
| #12 InfoRM | Simulation setting | 7B-scale SFT (AlpacaFarm-style) | 70M–7B InfoRM variants | `AlpacaFarm-Human 7B` (implied) | PPO | Gold RM vs KL curves |

**Key differences**: Policy size ranges from 410M (#1) to 7B (#3, #12). Papers #1 and #10 are the most directly comparable (both Pythia-based, both PPO, both use the same 7B gold RM on identical data splits). Paper #3 uses AlpacaFarm only for OOD generalization testing, not training.

**For our experiment**: Using Pythia-1.4B policy + Pythia-44M/70M RMs + 7B gold RM gives direct comparison to #1 and #10, and partial comparison to #12's simulation curves.

---

## Anthropic-HH (Helpful + Harmless)

Used in: **#3, #7, #8, #12, #14, #18 (as part of mix)**

| Paper | Role | Policy | RM(s) | Gold / Eval RM | RL Alg | Eval |
|---|---|---|---|---|---|---|
| #3 EPPO | RM + RL training (dialogue) | `Llama3-8B`, `Llama2-7B`, `Mistral-7B`, `DeepSeek-7B` | Standard RM from SFT backbone | — | PPO | GPT-4 W/T/L |
| #7 ARA | RM training (sycophancy task) | `Llama-2-7B` | Frozen proxy RM (unspecified) | GPT-4 factual accuracy | PPO | GPT-4 + SycophancyEval |
| #8 CausalRM | Dialogue RM + RLHF training | `Qwen2.5-7B` (SFT on ShareGPT) | CausalRM / Standard / GoalRM / InfoRM | — | PPO | `Qwen3-Max` W/T/L |
| #12 InfoRM | Real-world setting | `Vicuna-7B-v1.5` | InfoRM / Standard / Ensemble / WARM | — | PPO | GPT-4 W/T/L |
| #14 AdvPO | Full pipeline | `Llama-7B` | `Llama-7B` (proxy), `Vicuna-13B` (gold) | `Vicuna-13B` | PPO | GPT-4 W/T/L + human |
| #18 RRM | Part of RLHFlow mix (115k pairs) | `Gemma-2-9B-it` | RM / RRM from `Gemma-2-9B-it` | — | DPO + BoN | RewardBench + AlpacaEval2 |

**Key differences**: Policy models differ substantially (Llama-2-7B in #7/#14, Llama-3-8B in #3, Qwen2.5-7B in #8, Vicuna-7B in #12, Gemma-2-9B in #18). Eval judge also varies (GPT-4 in #3/#7/#12/#14, Qwen3-Max in #8, RewardBench+AlpacaEval in #18). The RL algorithm is PPO everywhere except #18 (DPO). SFT data varies: #3 and #8 SFT on ShareGPT, others SFT on HH chosen or their own SFT data.

**For our experiment**: Using Llama-2-7B policy + Anthropic-HH training gives the broadest overlap. GPT-4 W/T/L eval is needed for direct comparison to #3, #12, #14. Papers #7 and #8 use specialized eval (sycophancy, Qwen3-Max) that requires more adaptation.

---

## Reddit TL;DR

Used in: **#3, #11, #12, #14**

| Paper | Role | Policy | RM(s) | Gold / Eval RM | RL Alg | Eval |
|---|---|---|---|---|---|---|
| #3 EPPO | Full pipeline (SFT+RM+RL) | `Llama3-8B`, `Llama2-7B`, `Mistral-7B`, `DeepSeek-7B` | Standard RM from SFT backbone | — | PPO | GPT-4 W/T/L |
| #11 WARM | Full pipeline | `PaLM-XS` (proprietary) | `PaLM-XXS` WARM / ENS / individual | `PaLM-XS` control RM | PPO (RLHF) | Oracle pairwise preference (AI-labeler) |
| #12 InfoRM | Real-world setting | `Vicuna-7B-v1.5` | InfoRM / Standard / Ensemble / WARM | — | PPO | GPT-4 W/T/L |
| #14 AdvPO | Full pipeline | `Llama-7B` | `Llama-7B` (proxy), `Llama-13B` (gold) | `Llama-13B` | PPO | GPT-4 W/T/L + human |

**Key differences**: Policy models differ (7-8B LLMs in #3/#12/#14, proprietary PaLM in #11). WARM (#11) uses PaLM models — not reproducible with open-source weights. The other three (#3, #12, #14) all use open 7B policies + GPT-4 eval, making them the comparable cluster. Preference data source: #11 uses RLAIF (PaLM-L CoT labels), others use `openai/summarize_from_feedback` human labels.

**For our experiment**: #11 (WARM) is not directly reproducible (PaLM), but the metric format (oracle win rate curves) is comparable if we use a local gold RM. Papers #3, #12, #14 share GPT-4 W/T/L eval on open 7B-class models.

---

## UltraFeedback

Used in: **#2, #13, #18 (as part of mix)**

| Paper | Role | Policy | RM(s) | Gold / Eval RM | RL Alg | Eval |
|---|---|---|---|---|---|---|
| #2 BSPO | Main RM + RL training | `Alpaca-7B` | GPT2-large / TinyLlama / ShearedLlama (proxy); `Llama3-8B` (gold) | `Llama3-8B` on 57k UltraFeedback | PPO | Gold-RM curves + GPT-4o Elo |
| #13 RPO | DPO-style training (beta series) | `zephyr-7b-beta` | RM-free (DPO reparameterization) | — | DPO (RPO variant) | GPT-4 W/T/L + MT-Bench + AlpacaEval2 |
| #18 RRM | Part of RLHFlow mix (340k pairs) | `Gemma-2-9B-it` | RM / RRM from `Gemma-2-9B-it` | — | DPO + BoN | RewardBench + AlpacaEval2 |

**Key differences**: Very different setups. #2 uses UltraFeedback as the core RM dataset with PPO. #13 uses it for DPO-style training (RM-free). #18 includes it as one component of a 700k-pair mix. Policy models differ across all three.

**For our experiment**: Limited overlap. Would only make sense if targeting BSPO (#2) specifically.

---

## ShareGPT (SFT data)

Used in: **#3, #8**

| Paper | Role | Policy | Notes |
|---|---|---|---|
| #3 EPPO | SFT for dialogue task | `Llama3-8B` / `Llama2-7B` / `Mistral-7B` / `DeepSeek-7B` | SFT on ShareGPT, then RM+RL on Anthropic-HH |
| #8 CausalRM | SFT backbone for dialogue | `Qwen2.5-7B` | SFT on ShareGPT, then RM+RL on Anthropic-HH |

**Note**: Both papers that use ShareGPT for SFT pair it with Anthropic-HH for RM/RL training. This is a standard two-stage pipeline: ShareGPT for instruction-following SFT, Anthropic-HH for preference-based RL.

---

## HelpSteer variants

Used in: **#6, #16, #18 (as part of mix)**

| Paper | Variant | Role | Policy | Notes |
|---|---|---|---|---|
| #6 Outcome Accuracy | HelpSteer3-Atomic (1000 examples) | GenRM training + eval | `Qwen3-30B-A3B` / `Qwen3-14B` | GRPO, Arena Hard v2 eval |
| #16 Adv-RM | HelpSteer-2-Preferences | Synthetic RLHF setup (relabeled by gold RM `Llama-3.1-Nemotron-70B-Reward`) | `Llama-3.1-8B-Instruct` | Adversarial training, RewardBench eval |
| #18 RRM | HelpSteer (37k pairs) | Part of 700k-pair RLHFlow mix | `Gemma-2-9B-it` | One of 8 datasets in the mix |

**Key differences**: Each uses a different HelpSteer version at a very different scale and for a different purpose. No overlap in policy models or training setups.

---

## RewardBench (RM evaluation)

Used in: **#4, #16, #17, #18**

| Paper | Role | RM(s) evaluated | Notes |
|---|---|---|---|
| #4 Rethinking RM Eval | Referenced as baseline benchmark design | 14 math RMs | Compares RewardBench-style design to overoptimization-predictive designs |
| #16 Adv-RM | RM quality check post-adversarial training | `Llama-3.1-8B-Instruct` RM/Adv-RM | Baseline `0.8329` vs Adv-RM `0.8399` |
| #17 GRM | Primary RM generalization benchmark | Gemma-2B / Mistral-7B GRM variants | Up to `87.0` avg for GRM-8B |
| #18 RRM | Primary RM quality benchmark | `Gemma-2-9B-it` RM/RRM | RM `80.61` vs RRM `84.15` avg |

**Note**: RewardBench is RM-eval only (no policy training involved). Not directly relevant to our RLHF experiment design but useful for reporting RM quality if we want to position our BT ensemble RMs on this benchmark.

---

## MT-Bench (policy evaluation)

Used in: **#8 (OOD eval), #13, #17 (RM eval), #18**

| Paper | Role | Policy evaluated | Score type |
|---|---|---|---|
| #8 CausalRM | OOD eval for dialogue RLHF | `Qwen2.5-7B` | Pairwise accuracy |
| #13 RPO | Policy quality benchmark | `zephyr-7b-beta/gemma` | GPT-4 1-10 scale (RPO `7.381` vs DPO `7.278`) |
| #17 GRM | OOD RM evaluation (human judgements split) | — (RM eval, not policy) | RM classification accuracy |
| #18 RRM | Policy quality benchmark | `Gemma-2-9B-it` DPO policies | GPT-4 1-10 scale (RRM `8.31` vs RM `7.27`) |

---

## AlpacaEval 2.0 (policy evaluation)

Used in: **#13, #18**

| Paper | Policy evaluated | LC Win Rate | Raw Win Rate |
|---|---|---|---|
| #13 RPO | `zephyr-7b-beta` | RPO `23.28` vs DPO `21.15` | RPO `21.01` vs DPO `17.27` |
| #18 RRM | `Gemma-2-9B-it` DPO | RRM `52.49` vs RM `33.46` | RRM `43.31` vs RM `41.07` |

**Note**: Only 2 papers report AlpacaEval numbers, and they use very different base models. Not a high-coverage eval for our purposes.

---

# Recommended Experiments for Head-to-Head Comparison

The goal: use an existing dataset + base policy, train BT reward models and GRPO policy using **our** pipeline, and compare final gold reward / win rate / overoptimization curves against reported numbers from the papers above. We do NOT reimplement their methods — we compare against their published results.

Below are three experiment setups ordered by coverage (number of papers we can compare against) and practical feasibility. Each is self-contained.

---

## Experiment A: Anthropic-HH Dialogue (highest paper coverage)

**Covers papers**: #3 EPPO, #7 ARA, #8 CausalRM, #12 InfoRM, #14 AdvPO (5 papers)

**Setup**:
- **Policy model**: `Llama-2-7B` or `Llama-3-8B` (SFT on ShareGPT or Anthropic-HH chosen responses)
  - Llama-2-7B matches ARA (#7) and AdvPO (#14) exactly
  - Llama-3-8B matches EPPO (#3) exactly
  - Running both sizes gives us comparisons against all 5 papers
- **RM base**: same model family, smaller (e.g., 1-2B) or same size — train our BT RMs on Anthropic-HH preference split
- **Training data**: `Anthropic/hh-rlhf` (Helpful + Harmless splits, publicly available on HuggingFace)
- **RL**: GRPO with our ensemble strategies (sequential switching, UWO, mean ensemble)
- **Gold RM for local eval**: Skywork-Reward-V2-Llama-3.1-8B (already in our pipeline)

**Evaluation metric options**:
- **GPT-4 pairwise W/T/L** (same as EPPO/InfoRM/AdvPO/ARA): enables direct number comparison against Table 1 of each paper. Cost: ~$5-20 per method per eval set depending on set size.
- **Local gold RM score vs KL curves**: free, can be plotted alongside EPPO/InfoRM overoptimization curves (most of these papers include such plots even if the primary metric is GPT-4).

**What we can directly compare**:
| Paper | Their best result (Anth-HH) | Metric format |
|---|---|---|
| EPPO (#3) | W/T/L vs PPO w/KL and PPO w/LP per model | GPT-4 pairwise |
| ARA (#7) | Sycophancy `38.4`, Helpfulness `77.2` | GPT-4 + SycophancyEval |
| CausalRM (#8) | vs Standard RM ID `54.8/33.1/12.1` | Qwen3-Max pairwise |
| InfoRM (#12) | vs Standard RM on Anth-Helpful `54.5/33.5/12.0` | GPT-4 pairwise |
| AdvPO (#14) | vs PPO Anth-HH `31.0/49.0/20.0` | GPT-4 pairwise |

**Code effort**: ~3-4 days
1. Dataset adapter for Anthropic-HH in `load_datasets.py` (~1 day)
2. SFT on Anthropic-HH chosen responses or ShareGPT (~0.5 day, mostly compute)
3. Train BT RMs on Anthropic-HH preference split (existing pipeline, ~0.5 day)
4. GRPO runs with various ensemble configs (~0.5 day setup, then compute)
5. Evaluation: local gold RM already works; GPT-4 pairwise adapter (~1 day if needed)

**Note on RL algorithm gap**: All comparison papers use PPO. We use GRPO. This is a real methodological difference that should be acknowledged. However, the comparison is still meaningful: we claim our *RM ensemble strategy* (not the RL algorithm) is what prevents overoptimization. If our GRPO+ensemble matches or beats their PPO+{EPPO,InfoRM,ODIN,...}, the claim holds even with a different optimizer.

---

## Experiment B: AlpacaFarm Simulation (cheapest eval, no API)

**Covers papers**: #1 Iterated RLHF, #10 RM Ensembles (Coste et al.), #12 InfoRM (simulation setting)

**Setup**:
- **Policy model**: `Pythia-1.4B` (matches Coste et al. #10 exactly) or `Pythia-410M` (matches Iterated RLHF #1)
- **RM base**: Pythia-44M / Pythia-70M / Pythia-160M (same as papers #1, #10)
- **Training data**: `tatsu-lab/alpaca_farm` — SFT split (10k), preference split (20k), unlabeled split (20k)
- **Gold RM**: `tatsu-lab/reward-model-human` (7B, publicly available on HuggingFace) — runs locally, **no API cost**
- **RL**: GRPO with our ensemble strategies

**Evaluation metric**: Gold RM score vs KL divergence curves — the standard format for all three papers. These are the overoptimization curves we already produce internally.

**What we can directly compare**:
| Paper | Their reported numbers | Plot type |
|---|---|---|
| Iterated RLHF (#1) | Gold RM scores at iteration end: Ensemble `0.3136`, WCO `0.2942`, Concat+SFT `0.4477` | Gold RM score (scalar table + curves) |
| Coste et al. (#10) | Win rates: UWO PPO `60.2±3.4`, WCO PPO `59.4±3.3`, Mean PPO `58.1±3.3` (44M RMs, 0% noise) | Win rate vs single-RM baseline + gold-RM-vs-KL curves |
| InfoRM (#12) | Simulation overoptimization curves (figure-based) | Gold RM vs KL curves |

**Code effort**: ~2-3 days
1. AlpacaFarm dataset adapter — map `instruction`+`input` fields to chat template (~1 day)
2. Load AlpacaFarm 7B gold RM as evaluator in `evaluate_policy.py` (~0.5 day — standard causal LM RM, similar to Skywork loading)
3. Train BT RMs on AlpacaFarm preference split with Pythia backbone (pipeline exists, just swap model ID)
4. GRPO runs + evaluation

**Key advantage**: This setup produces the **exact same metric** (gold RM score vs KL) as the papers, on the **exact same data**, with comparable model sizes. The only difference is GRPO vs PPO. This is the cleanest apples-to-apples comparison available without reimplementing anything.

---

## Experiment C: Reddit TL;DR Summarization (broadest literature, local eval)

**Covers papers**: #3 EPPO (TL;DR results), #11 WARM, #12 InfoRM (TL;DR results), #14 AdvPO (TL;DR results)

**Setup**:
- **Policy model**: `Pythia-1B` or `Pythia-2.8B` (matches Huang et al. "N+ Implementation Details" / Async RLHF baselines, both of which are widely cited even though not in this file)
- **RM base**: Pythia-based, trained on TL;DR preference data
- **Training data**: `openai/summarize_from_feedback` (comparisons split ~93k pairs, SFT split ~117k; publicly available on HuggingFace)
- **Gold RM**: Pythia-6.9B RM from Huang et al. (pattern: `vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr`), runs locally
- **RL**: GRPO with ensemble strategies

**Evaluation metric**: Win rate of generated summaries vs human-written summaries, scored by the local gold RM. No API needed.

**What we can directly compare**:
| Paper | Their TL;DR results | Metric |
|---|---|---|
| EPPO (#3) | vs PPO w/KL `57/25/18`, vs PPO w/LP `46/32/22` | GPT-4 W/T/L (theirs); we'd report gold RM win rate (ours). Curves comparable. |
| WARM (#11) | `79.4%` win rate vs single-RM policy; `92.5%` BoN oracle win rate | Oracle preference win rate |
| InfoRM (#12) | vs SFT `73.1/17.3/9.5` on TL;DR; vs Standard RM `70.4/17.9/11.6` | GPT-4 W/T/L |
| AdvPO (#14) | vs PPO `75.0/7.0/18.0` on TL;DR | GPT-4 W/T/L |

**Code effort**: ~3 days
1. TL;DR dataset adapter — Reddit post format (`SUBREDDIT: ... TITLE: ... POST: ... TL;DR:`) differs from chat template (~1 day)
2. TL;DR gold RM integration (Pythia-6.9B, standard BT model) (~0.5 day)
3. SFT on TL;DR SFT split, train BT RMs on TL;DR preference data
4. GRPO runs + evaluation

**Caveat**: The TL;DR task is fundamentally different from general dialogue/instruction-following — it's extractive summarization. Some of our ensemble strategies may behave differently on this task shape (shorter outputs, more constrained).

---

## Summary: Coverage Matrix

Which experiment covers which papers' numbers for direct comparison:

| Paper | Exp A (Anth-HH) | Exp B (AlpacaFarm) | Exp C (TL;DR) |
|---|---|---|---|
| #1 Iterated RLHF | | **direct** (gold RM table) | |
| #2 BSPO | | | |
| #3 EPPO | **direct** (GPT-4 W/T/L) | eval-only overlap | **direct** (curves) |
| #7 ARA | **direct** (GPT-4 metrics) | | |
| #8 CausalRM | **direct** (W/T/L table) | | |
| #10 Coste et al. | | **direct** (gold RM + win rates) | |
| #11 WARM | | | **direct** (oracle win rate) |
| #12 InfoRM | **direct** (GPT-4 W/T/L) | **direct** (simulation curves) | **direct** (GPT-4 W/T/L) |
| #14 AdvPO | **direct** (GPT-4 W/T/L) | | **direct** (GPT-4 W/T/L) |
| #16 Adv-RM | | | |
| #17 GRM | | | |
| #18 RRM | | | |

**Recommendation**: Start with **Experiment B (AlpacaFarm)** — it has zero API cost, the cleanest gold-RM comparison format, and covers the most directly relevant prior work on RM ensembles. Then add **Experiment A (Anthropic-HH)** for the broader 2025 paper coverage if the story needs strengthening.


---

# Full RLHF Pipeline Releases: Open-Source Models with Base → SFT → RM → Aligned Policy

Last updated: 2026-03-11.

This section catalogs projects that have publicly released **all or most** of the four core RLHF pipeline artifacts (base model, SFT model, reward model, aligned policy model). Organized by completeness and relevance to reward hacking research. Each entry includes HuggingFace model links, dataset links, evaluation details, and cross-references to the papers in the main file above.

---

## Pipeline A: cleanrl / TRL — TL;DR Summarization (Pythia, 1B–6.9B) ⭐ Best small-scale reproducibility

**Paper**: Huang et al., "The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization" (COLM 2024)
**Link**: https://arxiv.org/abs/2403.17031
**Code**: https://github.com/vwxyzjn/lm-human-preference-details

### Released artifacts

| Artifact | 1B | 2.8B | 6.9B |
|---|---|---|---|
| **Base** | [`EleutherAI/pythia-1b-deduped`](https://huggingface.co/EleutherAI/pythia-1b-deduped) | [`EleutherAI/pythia-2.8b-deduped`](https://huggingface.co/EleutherAI/pythia-2.8b-deduped) | [`EleutherAI/pythia-6.9b-deduped`](https://huggingface.co/EleutherAI/pythia-6.9b-deduped) |
| **SFT** | [`cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr`](https://huggingface.co/cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr) | [`cleanrl/EleutherAI_pythia-2.8b-deduped__sft__tldr`](https://huggingface.co/cleanrl/EleutherAI_pythia-2.8b-deduped__sft__tldr) | [`cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr`](https://huggingface.co/cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr) |
| **RM** | [`cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr`](https://huggingface.co/cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr) | [`cleanrl/EleutherAI_pythia-2.8b-deduped__reward__tldr`](https://huggingface.co/cleanrl/EleutherAI_pythia-2.8b-deduped__reward__tldr) | [`cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr`](https://huggingface.co/cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr) |
| **PPO** | `vwxyzjn/EleutherAI_pythia-1b-deduped__ppo__tldr` | `vwxyzjn/EleutherAI_pythia-2.8b-deduped__ppo__tldr` | `vwxyzjn/EleutherAI_pythia-6.9b-deduped__ppo__tldr` |
| **RLOO** | `vwxyzjn/rloo_tldr` (1B variant) | — | — |

Additional collections: [`vwxyzjn/async-rlhf-paper-checkpoints`](https://huggingface.co/collections/vwxyzjn/async-rlhf-paper-checkpoints-67a3680cd4c4914f44132ba0) (async RLHF variants).

### Dataset
- **SFT + Preference**: [`openai/summarize_from_feedback`](https://huggingface.co/datasets/openai/summarize_from_feedback) (~93k comparison pairs, ~117k SFT demonstrations)

### Evaluation
- **Gold RM / judge**: The 6.9B reward model (`cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr`) serves as the de facto gold RM when training policies at 1B and 2.8B scale. No external LLM-as-judge.
- **Reported metrics**: Win rate of generated summaries vs. human reference summaries, scored by the gold RM. KL-reward Pareto curves.
- **Key reported numbers**: PPO 1B achieves significantly higher gold RM scores than SFT 1B; 6.9B PPO is strongest. Exact final scalars are primarily in figures/wandb logs rather than a consolidated table in the paper.

### Cross-reference to main file
- **Experiment C (TL;DR)** uses this pipeline directly. The 6.9B RM is the proposed gold RM for Experiment C.
- Papers #3 EPPO, #11 WARM, #12 InfoRM, #14 AdvPO all run on TL;DR and can be compared against this pipeline's baselines.

---

## Pipeline B: Coste et al. — Reward Model Ensembles (Pythia, 70M–1.4B) ⭐ Best for reward hacking research

**Paper**: Coste et al., "Reward Model Ensembles Help Mitigate Overoptimization" (ICLR 2024)
**Link**: https://openreview.net/forum?id=Vuw5St1x4r
**Code**: https://github.com/tlc4418/llm_optimization

**Cross-reference**: This is **Paper #10** in the main file above.

### Released artifacts

| Artifact | HuggingFace / Location |
|---|---|
| **Base (policy)** | [`EleutherAI/pythia-1.4b-deduped`](https://huggingface.co/EleutherAI/pythia-1.4b-deduped) |
| **Base (RM backbones)** | [`EleutherAI/pythia-14m`](https://huggingface.co/EleutherAI/pythia-14m), [`EleutherAI/pythia-70m-deduped`](https://huggingface.co/EleutherAI/pythia-70m-deduped), [`EleutherAI/pythia-1.4b-deduped`](https://huggingface.co/EleutherAI/pythia-1.4b-deduped) |
| **SFT (1.4B)** | [`tlc4418/pythia_1.4b_sft_policy`](https://huggingface.co/tlc4418/pythia_1.4b_sft_policy) |
| **SFT (70M)** | [`tlc4418/pythia_70m_sft`](https://huggingface.co/tlc4418/pythia_70m_sft) |
| **Proxy RMs** | Trained via code at `github.com/tlc4418/llm_optimization` at 7M, 44M, 1.3B scales |
| **Gold RM** | [`tatsu-lab/alpaca-farm-reward-model-human-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-reward-model-human-wdiff) (7B, weight diff — requires LLaMA-7B base) |
| **Aligned policy** | Trained via code (PPO + BoN); no pre-trained checkpoint released, but code is fully reproducible |
| **Preference data** | [`tlc4418/1.4b-policy_preference_data_gold_labelled`](https://huggingface.co/datasets/tlc4418/1.4b-policy_preference_data_gold_labelled) |

### Dataset
- **SFT + RL**: [`tatsu-lab/alpaca_farm`](https://huggingface.co/datasets/tatsu-lab/alpaca_farm) (10k SFT, ~20k preference, ~20k unlabeled for RL)
- **Preference labels**: Generated by the 7B gold RM on sampled policy outputs. Main proxy RM training size: 46k prompts.

### Evaluation
- **Gold RM**: AlpacaFarm human-preference RM (7B). Gold RM score vs. KL divergence is the primary metric.
- **Reported metrics**: Win rate of ensemble-optimized policies vs. single-RM-optimized policies (Table 6/7 in paper).
- **Key reported numbers** (copied from Paper #10 in main file):
  - UWO PPO win rate: `60.2 ± 3.4` (44M RMs, 0% noise), `63.0 ± 3.1` (25% noise) — best in table
  - WCO PPO: `59.4 ± 3.3` (0% noise), `62.2 ± 2.9` (25% noise)
  - Mean PPO: `58.1 ± 3.3` (0% noise), `60.2 ± 3.5` (25% noise)
  - Scale transfer (UWO): BoN `73.8 ± 0.8` (1.3B RMs)

### Cross-reference to main file
- **Paper #10** in the main file. **Experiment B (AlpacaFarm)** is designed to directly compare against these numbers.
- Shares exact setup (AlpacaFarm + Pythia + 7B gold RM) with Paper #1 (Iterated RLHF) and Paper #5 (Inference-Time Reward Hacking).
- The Coste et al. codebase is the **closest open reproduction** of Gao et al.'s overoptimization scaling laws (Paper not in main file; used closed OpenAI models).

---

## Pipeline C: AlpacaFarm (Stanford) — Full Simulation Framework (LLaMA-7B)

**Paper**: Dubois et al., "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback" (NeurIPS 2023)
**Link**: https://arxiv.org/abs/2305.14387
**Code**: https://github.com/tatsu-lab/alpaca_farm

### Released artifacts (all as weight diffs requiring LLaMA-7B)

| Artifact | HuggingFace |
|---|---|
| **SFT (10k)** | [`tatsu-lab/alpaca-farm-sft10k-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-sft10k-wdiff) |
| **RM (human pref)** | [`tatsu-lab/alpaca-farm-reward-model-human-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-reward-model-human-wdiff) |
| **RM (sim GPT-4)** | [`tatsu-lab/alpaca-farm-reward-model-sim-gpt4-20k-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-reward-model-sim-gpt4-20k-wdiff) |
| **PPO (sim)** | [`tatsu-lab/alpaca-farm-ppo-sim-gpt4-20k-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-ppo-sim-gpt4-20k-wdiff) |
| **PPO (human)** | [`tatsu-lab/alpaca-farm-ppo-human-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-ppo-human-wdiff) |
| **Expert Iter (sim)** | [`tatsu-lab/alpaca-farm-expiter-sim-gpt4-20k-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-expiter-sim-gpt4-20k-wdiff) |
| **Expert Iter (human)** | [`tatsu-lab/alpaca-farm-expiter-human-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-expiter-human-wdiff) |
| **FeedME (sim)** | [`tatsu-lab/alpaca-farm-feedme-sim-gpt4-20k-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-feedme-sim-gpt4-20k-wdiff) |
| **FeedME (human)** | [`tatsu-lab/alpaca-farm-feedme-human-wdiff`](https://huggingface.co/tatsu-lab/alpaca-farm-feedme-human-wdiff) |

### Dataset
- [`tatsu-lab/alpaca_farm`](https://huggingface.co/datasets/tatsu-lab/alpaca_farm): Contains `alpaca_farm_evaluation` (805 prompts), `alpaca_gpt4_preference` (19.5k), `alpaca_human_preference` (9.7k), `alpaca_instructions` (52k), `alpaca_noisy_multi_preference` (9.7k)

### Evaluation
- **Gold RM / judge**: The human-preference RM (7B) is the gold standard for simulation experiments. Additionally, AlpacaEval automated annotators (GPT-4-based) are used for the final leaderboard.
- **Reported leaderboard** (win rate vs. Davinci-001 reference, 805 eval prompts):
  - `gpt35_turbo_instruct`: `81.71`
  - `alpaca-farm-ppo-sim-gpt4-20k`: `44.10`
  - `alpaca-farm-ppo-human`: `41.24`
  - `alpaca-7b` (SFT base): `26.46`
  - `text_davinci_001`: `15.17`

### Cross-reference to main file
- The AlpacaFarm gold RM is used as the gold judge in Papers #1, #5, #10, and #12 (simulation setting).
- **Experiment B** depends entirely on this infrastructure.
- Dual-variant design (human RM + simulated RM) is uniquely valuable for studying how feedback quality affects RLHF outcomes.

---

## Pipeline D: PKU-Alignment / Beaver — Safe RLHF (LLaMA-7B) ⭐ Most comprehensive single release

**Paper**: Dai et al., "Safe RLHF: Safe Reinforcement Learning from Human Feedback" (ICLR 2024 Spotlight)
**Link**: https://arxiv.org/abs/2310.12773
**Code**: https://github.com/PKU-Alignment/safe-rlhf

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **SFT base** | [`PKU-Alignment/alpaca-7b-reproduced`](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced) |
| **Reward Model v1** | [`PKU-Alignment/beaver-7b-v1.0-reward`](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-reward) |
| **Cost Model v1** | [`PKU-Alignment/beaver-7b-v1.0-cost`](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-cost) |
| **Aligned v1** | [`PKU-Alignment/beaver-7b-v1.0`](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0) |
| **Reward Model v2** | [`PKU-Alignment/beaver-7b-v2.0-reward`](https://huggingface.co/PKU-Alignment/beaver-7b-v2.0-reward) |
| **Cost Model v2** | [`PKU-Alignment/beaver-7b-v2.0-cost`](https://huggingface.co/PKU-Alignment/beaver-7b-v2.0-cost) |
| **Aligned v2** | [`PKU-Alignment/beaver-7b-v2.0`](https://huggingface.co/PKU-Alignment/beaver-7b-v2.0) |
| **Reward Model v3** | [`PKU-Alignment/beaver-7b-v3.0-reward`](https://huggingface.co/PKU-Alignment/beaver-7b-v3.0-reward) |
| **Cost Model v3** | [`PKU-Alignment/beaver-7b-v3.0-cost`](https://huggingface.co/PKU-Alignment/beaver-7b-v3.0-cost) |
| **Aligned v3** | [`PKU-Alignment/beaver-7b-v3.0`](https://huggingface.co/PKU-Alignment/beaver-7b-v3.0) |
| **Unified Reward** | [`PKU-Alignment/beaver-7b-unified-reward`](https://huggingface.co/PKU-Alignment/beaver-7b-unified-reward) |
| **Unified Cost** | [`PKU-Alignment/beaver-7b-unified-cost`](https://huggingface.co/PKU-Alignment/beaver-7b-unified-cost) |

### Dataset
- [`PKU-Alignment/PKU-SafeRLHF`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF): ~330k expert comparison pairs (safety + helpfulness annotations), 10k multi-round data
- Paper: https://arxiv.org/abs/2307.04657

### Evaluation
- **Gold RM / judge**: The v3 reward model and v3 cost model serve as implicit gold standards for earlier iterations. The paper primarily evaluates safety constraint satisfaction (cost < threshold) alongside helpfulness (reward).
- **Reported metrics**: Safety violation rate, helpfulness reward score, constrained optimization Pareto curves. The iterative improvement from v1→v2→v3 is the central result.
- **No LLM-as-judge** in the primary evaluation pipeline; the paper uses automatic safety classifiers and the reward/cost model scores themselves.

### Cross-reference to main file
- Paper #4 (Rethinking RM Eval) includes `Beaver` reward models as baselines in the RewardBench evaluation (specifically `oasst-rm` / `Beaver` in their RM comparison table).
- The `PKU-SafeRLHF` dataset is used as part of the RLHFlow preference mix in Paper #18 (RRM): `26,874` pairs.
- The cost-model paradigm (separate safety signal) is conceptually related to Paper #7 (ARA) which audits for safety-specific reward hacking.

---

## Pipeline E: OpenRLHF — Llama-3-8B Pipeline

**Paper**: Hu et al., "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework"
**Link**: https://arxiv.org/abs/2405.11143
**Code**: https://github.com/OpenRLHF/OpenRLHF

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **Base** | [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| **SFT** | [`OpenRLHF/Llama-3-8b-sft-mixture`](https://huggingface.co/OpenRLHF/Llama-3-8b-sft-mixture) |
| **RM (mixture)** | [`OpenRLHF/Llama-3-8b-rm-mixture`](https://huggingface.co/OpenRLHF/Llama-3-8b-rm-mixture) |
| **RM (700k)** | [`OpenRLHF/Llama-3-8b-rm-700k`](https://huggingface.co/OpenRLHF/Llama-3-8b-rm-700k) |
| **PPO** | [`OpenRLHF/Llama-3-8b-rlhf-100k`](https://huggingface.co/OpenRLHF/Llama-3-8b-rlhf-100k) |

### Evaluation
- **Reported metrics**: Chat-Arena-Hard score. PPO model achieves `20.5` vs SFT `5.6` on Chat-Arena-Hard.
- **Supported algorithms**: PPO, GRPO, REINFORCE++, DAPO, Dr. GRPO.

### Cross-reference to main file
- Uses the same Llama-3-8B base as Paper #3 (EPPO) — direct architecture match for Experiment A comparisons.
- The 700k RM training mix overlaps with Paper #18 (RRM) which also uses the RLHFlow preference mix.

---

## Pipeline F: RLHFlow — Online Iterative DPO (Llama-3-8B)

**Paper**: Dong et al., "RLHF Workflow: From Reward Modeling to Online RLHF" (TMLR 2024)
**Link**: https://arxiv.org/abs/2405.07863
**Code (RM training)**: https://github.com/RLHFlow/RLHF-Reward-Modeling
**Code (Online RLHF)**: https://github.com/RLHFlow/Online-RLHF

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **Base** | [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| **SFT v1** | [`RLHFlow/LLaMA3-SFT`](https://huggingface.co/RLHFlow/LLaMA3-SFT) |
| **SFT v2** | [`RLHFlow/LLaMA3-SFT-v2`](https://huggingface.co/RLHFlow/LLaMA3-SFT-v2) |
| **RM (ArmoRM 8B)** | [`RLHFlow/ArmoRM-Llama3-8B-v0.1`](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) |
| **RM (Decision-Tree 27B)** | [`RLHFlow/Decision-Tree-Reward-Gemma-2-27B`](https://huggingface.co/RLHFlow/Decision-Tree-Reward-Gemma-2-27B) |
| **Aligned (iter. DPO)** | [`RLHFlow/LLaMA3-iterative-DPO-final`](https://huggingface.co/RLHFlow/LLaMA3-iterative-DPO-final) |

### Dataset
- ArmoRM trained on a mix of preference data; the codebase supports Bradley-Terry, pairwise, process, and decision-tree RM training.

### Evaluation
- **ArmoRM-8B**: Achieved **#1 on RewardBench** at time of release via multi-objective reward decomposition with MoE gating. Provides interpretable per-dimension scores (helpfulness, safety, verbosity, etc.).
- **Decision-Tree-27B**: Achieved **95.4% on RewardBench** — current SOTA among open RMs at time of release.
- **Policy**: LLaMA3-iterative-DPO-final reported strong AlpacaEval 2.0 and MT-Bench results (exact numbers in the paper's leaderboard).

### Cross-reference to main file
- ArmoRM is included as a baseline RM in Paper #4 (Rethinking RM Eval) and is evaluated on RewardBench in Papers #4, #17 (GRM).
- The RLHFlow preference mix (700k pairs) is the training data for Paper #18 (RRM) reward models.
- ODIN (anti-length-hacking) is integrated into the RLHFlow codebase, connecting to Papers #3, #7, #8 that study reward hacking mitigation.

---

## Pipeline G: Allen AI / Tülu 2.5 and 3 — Systematic Ablation (7B–70B)

**Paper (Tülu 2.5)**: Ivison et al., "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preferences" (NeurIPS 2024)
**Paper (Tülu 3)**: Lambert et al., "Tülu 3: Pushing Frontiers in Open Language Model Post-Training"
**Link (Tülu 3)**: https://arxiv.org/abs/2411.15124
**Code**: https://github.com/allenai/open-instruct (Apache 2.0)

### Released artifacts (selected — 44+ models total for Tülu 2.5 alone)

| Artifact | HuggingFace |
|---|---|
| **RM (Tülu 2.5, HH-RLHF)** | [`allenai/tulu-v2.5-13b-hh-rlhf-60k-rm`](https://huggingface.co/allenai/tulu-v2.5-13b-hh-rlhf-60k-rm) |
| **PPO (Tülu 2.5, HH-RLHF)** | [`allenai/tulu-v2.5-ppo-13b-hh-rlhf-60k`](https://huggingface.co/allenai/tulu-v2.5-ppo-13b-hh-rlhf-60k) |
| **RM (Tülu 3, 8B)** | [`allenai/Llama-3.1-Tulu-3-8B-RM`](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM) |
| **DPO (Tülu 3, 8B)** | [`allenai/Llama-3.1-Tulu-3-8B-DPO`](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO) |
| **RewardBench 2 RMs (70 ckpts)** | e.g. [`allenai/Llama-3.1-Tulu-3-8B-DPO-RM-RB2`](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO-RM-RB2) (see `allenai/` namespace) |

### Evaluation
- **Tülu 2.5**: Systematic comparison of PPO vs DPO across **14 different preference datasets**. Reports gold RM scores, MT-Bench, AlpacaEval 2.0. This is the most comprehensive open ablation study of alignment algorithms.
- **Tülu 3**: Reports IFEval, GSM8K, MATH, AlpacaEval 2.0, MT-Bench across 8B, 70B, 405B scales.
- **RewardBench 2**: 70 reward model checkpoints trained with varying configurations to correlate benchmark scores with downstream PPO performance — the **largest single collection** of diverse RMs with controlled variation.

### Cross-reference to main file
- Tülu 2.5's HH-RLHF experiments use the same dataset as Papers #3, #7, #8, #12, #14 — directly comparable to **Experiment A**.
- The Tülu 2.5 PPO vs DPO ablation across 14 datasets provides context for interpreting our GRPO results.
- RewardBench 2's 70 RM checkpoints are highly relevant to Paper #4 (Rethinking RM Eval) which studies RM quality vs. downstream policy performance.

---

## Pipeline H: Eisenstein et al. — Reward Ensemble Diversity (T5, 220M–3B) ⭐ Reward hacking focus

**Paper**: Eisenstein et al., "Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking" (2023)
**Link**: https://arxiv.org/abs/2312.09244
**Code**: https://github.com/google-deepmind/reward-ensembles

### Released artifacts

| Artifact | Location |
|---|---|
| **15 pretraining ckpts** | `github.com/google-deepmind/reward-ensembles` — 5 random seeds × 3 T5 scales (base ~220M, large ~770M, xl ~3B) |
| **Reward models** | Trained from the released pretraining checkpoints; code provided for full reproduction |

### Evaluation
- **Gold judge**: Internal human preference evaluation; the paper studies whether reward hacking persists even with diverse ensembles.
- **Key finding**: Reward hacking persists with ensembles. Pretraining seed diversity and fine-tuning seed diversity produce different ensemble behaviors.
- **Reported metrics**: Proxy reward vs. gold reward curves across optimization budget. Results are primarily figure-based.

### Cross-reference to main file
- Directly relevant to Paper #10 (Coste et al.) which also studies RM ensembles on Pythia models.
- The finding that ensembles are insufficient to fully prevent hacking motivates the more sophisticated methods in Papers #1, #3, #7, #12, #16.
- T5-based setup is distinct from the GPT-NeoX/Pythia/Llama ecosystem used by most other papers in this file.

---

## Pipeline I: OpenAssistant / LAION — Community-Driven (Pythia, 1.4B–12B)

**Project**: OpenAssistant (LAION)
**Code**: https://github.com/LAION-AI/Open-Assistant (now archived)

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **RM (6.9B)** | [`OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1`](https://huggingface.co/OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1) |
| **RM (1.4B)** | [`OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5`](https://huggingface.co/OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5) |
| **RM (DeBERTa)** | [`OpenAssistant/reward-model-deberta-v3-large-v2`](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) |
| **PPO (12B)** | [`andreaskoepf/oasst-rl-1-pythia-12b`](https://huggingface.co/andreaskoepf/oasst-rl-1-pythia-12b) |
| **SFT models** | Various under `OpenAssistant/` namespace (Pythia and LLaMA based) |

### Dataset
- [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1): ~161k human-generated assistant messages in 35 languages

### Evaluation
- **No systematic gold RM evaluation reported** — the project was community-driven rather than paper-driven. Models were evaluated via community testing and informal benchmarks.

### Cross-reference to main file
- The `oasst-rm` models appear as baselines in Paper #4 (Rethinking RM Eval) RewardBench comparisons.
- The Pythia-6.9B RM provides an independent reward signal that could be used as an alternative gold RM for Experiment B/C if the AlpacaFarm 7B RM is not suitable.

---

## Pipeline J: MOSS-RLHF (Fudan) — Secrets of RLHF (LLaMA-7B)

**Paper**: Zheng et al., "Secrets of RLHF in Large Language Models Part I: PPO" (NeurIPS 2023 Workshop Best Paper)
**Link**: https://arxiv.org/abs/2307.04964
**Code**: https://github.com/OpenLMLab/MOSS-RLHF

### Released artifacts (weight diffs, both English and Chinese)

| Artifact | HuggingFace |
|---|---|
| **SFT** | `fnlp/moss-rlhf-sft-model-7B-en` |
| **RM** | `fnlp/moss-rlhf-reward-model-7B-en` |
| **Policy** | `fnlp/moss-rlhf-policy-model-7B-en` |
| **Chinese variants** | `fnlp/moss-rlhf-{sft,reward,policy}-model-7B-zh` |

### Evaluation
- **Reported metrics**: The paper focuses on implementation details and failure modes of PPO rather than benchmark comparisons. Reports reward curves during training and qualitative output comparisons.
- **No standard benchmark table** (MT-Bench, AlpacaEval, etc.) in the main paper.

### Cross-reference to main file
- Not directly referenced in the main file's papers, but the implementation insights are relevant to understanding PPO instability issues discussed in Papers #3, #9, #15.

---

## Pipeline K: Starling-7B (Berkeley NEST) — RLAIF Pipeline

**Paper**: Zhu et al., "Starling-7B: Increasing LLM Helpfulness & Harmlessness with RLAIF"
**Link**: https://starling.cs.berkeley.edu/
**Code**: Uses OpenChat + APA (advantage-weighted policy averaging)

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **SFT base** | [`openchat/openchat_3.5`](https://huggingface.co/openchat/openchat_3.5) (Mistral-7B based) |
| **RM (7B)** | [`berkeley-nest/Starling-RM-7B-alpha`](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha) |
| **RM (34B)** | [`Nexusflow/Starling-RM-34B`](https://huggingface.co/Nexusflow/Starling-RM-34B) |
| **Aligned (alpha)** | [`berkeley-nest/Starling-LM-7B-alpha`](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) |
| **Aligned (beta)** | [`berkeley-nest/Starling-LM-7B-beta`](https://huggingface.co/berkeley-nest/Starling-LM-7B-beta) |

### Dataset
- Nectar dataset: 3.8M pairwise GPT-4 comparisons across diverse instruction sources.

### Evaluation
- **Reported benchmarks**: MT-Bench `8.09` (alpha), AlpacaEval 2.0 metrics. Starling-RM-7B-alpha was competitive on RewardBench at time of release.
- **RLAIF-based** — preferences come from GPT-4 comparisons rather than human annotations.

### Cross-reference to main file
- Not directly referenced in the main file's papers, but the Starling RM models are used as baselines in some RewardBench evaluations referenced by Papers #4, #17, #18.

---

## Pipeline L: InternLM2 — Multi-Scale Reward Models (1.8B–20B)

**Paper**: Cai et al., "InternLM2 Technical Report"
**Link**: https://arxiv.org/abs/2403.17297

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **RM (1.8B)** | [`internlm/internlm2-1_8b-reward`](https://huggingface.co/internlm/internlm2-1_8b-reward) |
| **RM (7B)** | [`internlm/internlm2-7b-reward`](https://huggingface.co/internlm/internlm2-7b-reward) |
| **RM (20B)** | [`internlm/internlm2-20b-reward`](https://huggingface.co/internlm/internlm2-20b-reward) |
| **Chat (1.8B)** | [`internlm/internlm2-chat-1_8b`](https://huggingface.co/internlm/internlm2-chat-1_8b) |
| **Chat (7B)** | [`internlm/internlm2-chat-7b`](https://huggingface.co/internlm/internlm2-chat-7b) |
| **Chat (20B)** | [`internlm/internlm2-chat-20b`](https://huggingface.co/internlm/internlm2-chat-20b) |

### Dataset
- RMs trained on 2.4M preference samples (internal curation).

### Evaluation
- **Reported**: RewardBench scores across three scales. The 7B RM is used as a baseline in Paper #4 (Rethinking RM Eval) where it scores `46.0/20.8` BoN and `29.4` PPO on MetaMATH.
- The multi-scale release (1.8B/7B/20B) is uniquely valuable for studying how RM scale affects overoptimization.

### Cross-reference to main file
- `internlm2-7b-reward` appears as an evaluated RM in Paper #4 (Rethinking RM Eval) Table `main_bon_results`.

---

## Pipeline M: UltraRM / OpenBMB — Feedback with Critique (LLaMA-13B)

**Paper**: Cui et al., "UltraFeedback: Boosting Language Models with Scaled AI Feedback"
**Link**: https://arxiv.org/abs/2310.01377

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **SFT** | [`openbmb/UltraLM-13b`](https://huggingface.co/openbmb/UltraLM-13b) |
| **RM** | [`openbmb/UltraRM-13b`](https://huggingface.co/openbmb/UltraRM-13b) |
| **Critique Model** | [`openbmb/UltraCM-13b`](https://huggingface.co/openbmb/UltraCM-13b) |
| **Aligned (BoN)** | [`openbmb/UltraLM-13b-v2.0`](https://huggingface.co/openbmb/UltraLM-13b-v2.0) |

### Dataset
- [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback): ~64k instructions, 256k model responses from 17 LLMs, scored by GPT-4 on instruction-following, truthfulness, honesty, helpfulness.

### Evaluation
- **Reported benchmarks**: AlpacaEval win rates, MT-Bench scores. UltraRM-13b was competitive at time of release.
- **UltraFeedback is foundational** — it underpins Zephyr, Tülu, Notus, and many other DPO projects.

### Cross-reference to main file
- UltraFeedback is the primary preference dataset for Paper #2 (BSPO) and part of the mix for Paper #18 (RRM: `340,025` pairs).
- Paper #13 (RPO) uses UltraFeedback for the Zephyr-beta pipeline.

---

## Pipeline N: HuggingFace Zephyr / Alignment Handbook — DPO Pipeline (Mistral-7B)

**Paper**: Tunstall et al., "Zephyr: Direct Distillation of LM Alignment" (EMNLP 2024 Industry)
**Link**: https://arxiv.org/abs/2310.16944
**Code**: https://github.com/huggingface/alignment-handbook

### Released artifacts

| Artifact | HuggingFace |
|---|---|
| **Base** | [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
| **SFT** | [`alignment-handbook/zephyr-7b-sft-full`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full) |
| **DPO (beta)** | [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) |
| **DPO (full)** | [`alignment-handbook/zephyr-7b-dpo-full`](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-full) |
| **Gemma variant** | [`HuggingFaceH4/zephyr-7b-gemma-v0.1`](https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-v0.1) |

**Note**: No explicit reward model — DPO bypasses RM training by design. This is a 3-of-4 pipeline.

### Evaluation
- **Reported**: MT-Bench `7.34` (beta), AlpacaEval `13.20` (LC). The Zephyr pipeline became a standard DPO baseline.

### Cross-reference to main file
- Paper #13 (RPO) uses `zephyr-7b-beta` as the base pipeline and reports RPO improvements over DPO on this exact setup.
- The alignment-handbook provides fully reproducible recipes that have been extended to Gemma and other architectures.

---

## Pipeline O: Stack-LLaMA (HuggingFace TRL Tutorial) — Educational Pipeline (LLaMA-7B LoRA)

**Paper**: Blog post: "StackLLaMA: A hands-on guide to train LLaMA with RLHF"
**Link**: https://huggingface.co/blog/stackllama
**Code**: Part of TRL library examples

### Released artifacts (all as LoRA adapters)

| Artifact | HuggingFace |
|---|---|
| **SFT adapter** | [`trl-lib/llama-7b-se-sft-peft`](https://huggingface.co/trl-lib/llama-7b-se-sft-peft) |
| **RM adapter** | [`trl-lib/llama-7b-se-rm-peft`](https://huggingface.co/trl-lib/llama-7b-se-rm-peft) |
| **PPO adapter** | [`trl-lib/llama-7b-se-rl-peft`](https://huggingface.co/trl-lib/llama-7b-se-rl-peft) |

### Dataset
- StackExchange Q&A data with voting-based preference labels.

### Evaluation
- **Primarily educational** — no formal benchmark table. Demonstrates the full RLHF pipeline with LoRA.

---

# Standalone Reward Models (No Full Pipeline)

These are notable publicly released RMs that do **not** come with a full pipeline but are used as gold/eval RMs in the papers above or would be useful as external evaluators for our experiments.

| Model | HuggingFace | Size | Used as gold/eval in |
|---|---|---|---|
| `Skywork-Reward-Llama-3.1-8B-v0.2` | [`Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) | 8B | Our pipeline (Experiment A gold RM) |
| `Skywork-Reward-Gemma-2-27B-v0.2` | [`Skywork/Skywork-Reward-Gemma-2-27B-v0.2`](https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B-v0.2) | 27B | Paper #16 (Adv-RM) attack target |
| `Llama-3.1-Nemotron-70B-Reward` | [`nvidia/Llama-3.1-Nemotron-70B-Reward`](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) | 70B | Paper #16 gold RM + attack target |
| `Skywork-o1-Open-PRM-Qwen2.5-7B` | [`Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B`](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B) | 7B | Paper #4 gold RM for math overoptimization |
| `Qwen2.5-Math-RM-72B` | [`Qwen/Qwen2.5-Math-RM-72B`](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B) | 72B | Math-specialized RM |
| `reward-model-Mistral-7B-instruct-Unified-Feedback` | (search `weqweasdas` namespace) | 7B | Paper #17 (GRM) gold RM for BoN/PPO |
| `PairRM (DeBERTa)` | [`llm-blender/PairRM`](https://huggingface.co/llm-blender/PairRM) | 0.4B | Efficient pairwise RM for BoN |
| `hh_rlhf_rm_open_llama_3b` | [`weqweasdas/hh_rlhf_rm_open_llama_3b`](https://huggingface.co/weqweasdas/hh_rlhf_rm_open_llama_3b) | 3B | HH-RLHF trained, 75.5% accuracy |
| `Eurus-RM-7b` | [`openbmb/Eurus-RM-7b`](https://huggingface.co/openbmb/Eurus-RM-7b) | 7B | Paper #4 RM comparison table |
| `GRM-Llama3-8B-sftreg` | (GRM paper models) | 8B | Paper #17 (GRM) — own paper |
| `nvidia/Qwen-3-Nemotron-32B-Reward` | [`nvidia/Qwen-3-Nemotron-32B-Reward`](https://huggingface.co/nvidia/Qwen-3-Nemotron-32B-Reward) | 32B | Latest NVIDIA RM |

---

# Key Datasets for RLHF Pipeline Training

| Dataset | HuggingFace | Size | Used in papers |
|---|---|---|---|
| AlpacaFarm | [`tatsu-lab/alpaca_farm`](https://huggingface.co/datasets/tatsu-lab/alpaca_farm) | 91.6k rows (multiple splits) | #1, #2(app), #3(eval), #5, #10, #12 |
| Anthropic-HH | [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) | ~170k conversations | #3, #7, #8, #12, #14, #18(mix) |
| Reddit TL;DR | [`openai/summarize_from_feedback`](https://huggingface.co/datasets/openai/summarize_from_feedback) | ~93k comparison pairs | #3, #11, #12, #14, Pipeline A |
| UltraFeedback | [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback) | 64k instructions, 256k responses | #2, #13, #18(mix) |
| PKU-SafeRLHF | [`PKU-Alignment/PKU-SafeRLHF`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ~330k pairs | Pipeline D, #18(mix) |
| HelpSteer2 | [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | ~10k | #16 |
| SHP (Stanford Human Preferences) | [`stanfordnlp/SHP`](https://huggingface.co/datasets/stanfordnlp/SHP) | ~385k | #8(eval), #18(mix) |
| OpenAssistant (oasst1) | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) | ~161k messages | Pipeline I |
| Unified-Feedback | (see GRM paper) | 400k | #17 |
| RLHFlow 700k Mix | (composite: HH+SHP+HelpSteer+SafeRLHF+UltraFeedback+UltraInteract+Capybara+Orca) | ~796k | #18 |

---

# Pipeline Completeness Summary (Updated with Evaluations)

| Pipeline | Org | Sizes | Base | SFT | RM | Aligned | Gold RM / Judge | Key Eval Metric | Overopt Focus? |
|---|---|---|---|---|---|---|---|---|---|
| **A: cleanrl/TRL TL;DR** | HF/Mila | 1B–6.9B | ✅ | ✅ | ✅ | ✅ (PPO/RLOO) | 6.9B RM (self) | Win rate vs human summaries | Discusses it |
| **B: Coste et al.** | Mila/UCL | 70M–1.4B | ✅ | ✅ | Code | Code | AlpacaFarm Human 7B | Gold RM vs KL + win rate | **Yes ⭐** |
| **C: AlpacaFarm** | Stanford | 7B | ✅* | ✅ | ✅ (human+sim) | ✅ (PPO+ExpIter) | AlpacaFarm Human 7B + AlpacaEval | Win rate vs Davinci-001 | Simulation framework |
| **D: Beaver/Safe-RLHF** | PKU | 7B | ✅ | ✅ | ✅ (+Cost, 3 versions) | ✅ (v1–v3) | Internal RM/Cost scores | Safety violation rate + reward | Safety focus |
| **E: OpenRLHF** | Community | 8B | ✅ | ✅ | ✅ | ✅ (PPO) | — | Chat-Arena-Hard `20.5` | No |
| **F: RLHFlow** | UIUC/HKUST | 8B | ✅ | ✅ | ✅ (ArmoRM) | ✅ (iter. DPO) | RewardBench #1 (ArmoRM) | RewardBench + AlpacaEval | ODIN integrated |
| **G: Tülu 2.5/3** | Allen AI | 7B–70B | ✅ | ✅ | ✅ | ✅ (PPO+DPO) | RewardBench 2 (70 ckpts) | MT-Bench + AlpacaEval + IFEval | 44+ ablations |
| **H: Eisenstein et al.** | DeepMind | 220M–3B | ✅ | — | 15 ckpts | — | Human pref (internal) | Proxy vs gold reward curves | **Yes ⭐** |
| **I: OpenAssistant** | LAION | 1.4B–12B | ✅ | ✅ | ✅ (4 models) | ✅ (PPO 12B) | None formal | Community testing | No |
| **J: MOSS-RLHF** | Fudan | 7B | ✅* | ✅ | ✅ | ✅ | None formal | Reward curves (training) | No |
| **K: Starling-7B** | Berkeley | 7B | ✅ | ✅ | ✅ (7B+34B) | ✅ (APA) | RewardBench | MT-Bench `8.09` | No |
| **L: InternLM2** | Shanghai AI Lab | 1.8B–20B | ✅ | ✅ | ✅ (3 sizes) | ✅ | RewardBench | RewardBench scores | No |
| **M: UltraRM** | OpenBMB | 13B | ✅ | ✅ | ✅ (+CritiqueM) | ✅ (BoN) | — | AlpacaEval + MT-Bench | No |
| **N: Zephyr** | HuggingFace | 7B | ✅ | ✅ | **—** (DPO) | ✅ | — | MT-Bench `7.34`, AlpacaEval | No |
| **O: Stack-LLaMA** | HuggingFace | 7B (LoRA) | ✅ | ✅ | ✅ | ✅ | None formal | Educational | No |

*Weight diffs requiring original LLaMA-1.

---

# Mapping: Which Pipeline Artifacts Match Which Papers

This table maps the pipeline releases above to the 18 papers in the main file, showing which released artifacts can serve as baselines, gold RMs, or direct comparison points.

| Paper | Matching Pipeline(s) | What matches | Notes |
|---|---|---|---|
| **#1 Iterated RLHF** | **B** (Coste), **C** (AlpacaFarm) | Same gold RM (AlpacaFarm 7B), same data, Pythia policy | Policy size differs (410M vs 1.4B) |
| **#2 BSPO** | **M** (UltraRM) for dataset | UltraFeedback dataset overlap | Different policy (Alpaca-7B vs UltraLM-13b) |
| **#3 EPPO** | **A** (TL;DR), **E** (OpenRLHF) | Same Llama-3-8B base; same Anthropic-HH and TL;DR datasets | OpenRLHF provides the SFT/RM/PPO checkpoint suite |
| **#4 Rethinking RM Eval** | **F** (RLHFlow), **L** (InternLM2) | ArmoRM and internlm2-7b-reward are in their RM comparison table | Math-focused; uses different evaluation paradigm |
| **#5 Inference-Time RH** | **B** (Coste) | Same AlpacaFarm setup, same proxy RM sizes | BoN/BoP vs PPO (different optimization method) |
| **#7 ARA** | — | Llama-2-7B + Anthropic-HH (no exact pipeline match released) | Could use Tülu 2.5 HH-RLHF artifacts as approximate match |
| **#8 CausalRM** | — | Qwen2.5-7B + Anthropic-HH (no pipeline with this exact combo) | |
| **#9 Constrained RLHF** | — | GPT-2 + DailyDialog (unique setup, no pipeline match) | |
| **#10 Coste et al.** | **B** (IS this paper) | Exact match — this is the pipeline | |
| **#11 WARM** | **A** (TL;DR) for dataset | TL;DR dataset overlap | PaLM models are proprietary; not reproducible |
| **#12 InfoRM** | **B** (AlpacaFarm sim), **A** (TL;DR real) | AlpacaFarm simulation + Anthropic-HH + TL;DR overlap | Multiple eval settings match multiple pipelines |
| **#13 RPO** | **N** (Zephyr) | Exact match — RPO builds on zephyr-7b-beta pipeline | |
| **#14 AdvPO** | **A** (TL;DR) | Llama-7B + Anthropic-HH + TL;DR | Gold RM is Vicuna-13B (not in any pipeline) |
| **#15 Accuracy Paradox** | — | T5 + QA-FEEDBACK (unique setup) | |
| **#16 Adv-RM** | **E** (OpenRLHF) for base model | Llama-3.1-8B-Instruct overlap | Gold RM is Nemotron-70B (standalone) |
| **#17 GRM** | — | Gemma-2B/Mistral-7B + Unified-Feedback (unique setup) | GRM models are their own artifacts |
| **#18 RRM** | **F** (RLHFlow) for RM training data | Same RLHFlow 700k preference mix | Gemma-2-9B-it base (not in pipelines) |
