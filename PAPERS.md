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

