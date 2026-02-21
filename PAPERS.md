# Papers to Compare Against (Prioritized 2025-2026)

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
  - UltraFeedback preference dataset (main).
  - Additional setups include AlpacaFarm with a `20k` GPT-4-annotated preference dataset.
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
  - `Standard PPO`, `KL-Penalty PPO`, `CPPO`, `ENS-UWO`, `ENS-WCO`, plus `DPO`, `RPO`.
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
  - General dialogue: AlpacaFarm, Anthropic-Helpful, Anthropic-Harmless.
  - Summarization: Reddit TL;DR.
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
  - Main comparison is unhedged vs `HedgeTune`-tuned operating points.
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

## Notes on Coverage Gaps

- Some highly relevant 2025–2026 works report core results mostly in curves/figures (not full numeric tables), especially around inference-time reward hacking. Those are included with explicit missing-data notes.
- For a strict apples-to-apples comparison against your current GRPO + BT-RM pipeline, the papers above that are closest in setup are:
  - 2505.18126 (iterated RLHF overoptimization with explicit gold/proxy setup)
  - 2503.18130 (RLHF overoptimization mitigation + non-synthetic GPT-4o evaluation)
  - 2602.04649 (GenRM + GRPO + downstream policy alignment)
  - 2601.21350 / 2602.01750 (robust RM and anti-hacking RLHF pipelines with explicit downstream metrics)
