# Vector Institute Compute Access + Funding Application (DRAFT)

> **Draft status & decisions to confirm before submission**
> - **Track:** **Track A** (preliminary results available; the full multi-family /
>   multi-seed / scaling program needs the larger envelope).
> - **GPU hours:** **50,000** (bottom-up breakdown in §5); see the 15k-vs-50k reasoning
>   in my message; the asymmetry favours the larger, well-justified ask.
> - **Funding:** **not requested** (compute-only application).
> - **Team filled in** from the links you provided. One `[TODO]` remains: confirm the
>   sponsoring Vector faculty member (drafted as Igor Gilitschenski).
> - **Anchor model corrected to Qwen3.5-9B** (raises per-run cost; reflected in §5).

---

## Track Selection

- [x] **Track A – High-impact research** (15,000–400,000 GPU hours; funding $10k–$60k)
- [ ] Track B – Proof-of-concept / early-stage (1,000–15,000 GPU hours; funding $10k–$20k)

*Rationale:* the project has substantial pilot results (§2) and spans a full
benchmark program (multiple model families up to 9B and larger, multiple training
methods, hyperparameter sweeps, and multi-seed confirmation), which fits Track A's
scope and its request for preliminary feasibility evidence.

---

## Section 1: Applicant Information

- **Full Name:** **Igor Gilitschenski** (Assistant Professor, University of Toronto;
  Toronto Intelligent Systems Lab) `[TODO: confirm he is the sponsoring member on the Vector faculty list]`
- **Will you have Co-Investigators and Collaborators?** **Yes**
  - Project lead / driver: **Evgenii Opryshko** (PhD student, U of T / TISL)
  - Collaborator (external): **Michael K. Cohen** (Postdoc, Center for Human-Compatible
    AI, UC Berkeley): AI-safety / reward-overoptimization theory
  - Student contributor: **Xiaowen Zhang** (undergraduate, U of T)
- **External Collaborator Confirmation:** **Yes.** "I understand external project
  collaborators must complete separate security approval through Research Operations."
  (Applies to Michael K. Cohen, UC Berkeley.)

---

## Section 2: Project Overview

### Project Title (≤100 characters)
Benchmarking Methods for Aligning Language Models with Human Preferences

### Project Description (≤500 words)

**The Challenge.** Language models are aligned with human preferences through preference optimization methods, including reinforcement-learning-based approaches like RLHF (a reward model is trained to imitate human judgments and the language model is optimized against it) and offline alternatives like Direct Preference Optimization (DPO). Yet the field has no standardized way to compare these methods fairly. Papers use different base models, datasets, and evaluators, often with weakly tuned baselines, and they frequently select their best model using signals unavailable in real deployments. As a result, it is hard to tell which methods genuinely work. An active area of research aims to reduce reward overoptimization, the failure mode where a model improves on the learned reward while its true quality stalls or even declines. Yet it remains unclear how proposed methods compare with each other, or even whether they outperform carefully tuned baselines.

**Strategic Significance.** RLHF is central to making language models helpful and trustworthy, aligning directly with Vector's Safe and Trustworthy AI and Foundation/Generative Models priorities. A shared, rigorous benchmark benefits the whole ecosystem: alignment and RLHF researchers gain a common testbed and strong baselines, students gain a well-scoped platform, industry teams gain evidence on which methods to trust, and end users ultimately receive better aligned models.

**Research Questions.** (1) Which training methods produce the best alignment with human preference data? (2) Which training setups are prone to reward hacking, where a model scores well on the learned reward without truly getting better? (3) Do recently published methods actually outperform strong, carefully tuned baselines (we hypothesize that many do not)? (4) Does the learning rate change the trade-off between reward and how far the model drifts from its starting point, and the dynamics of overoptimization? (5) Can a good rule for choosing the final checkpoint, without access to the test signal, prevent overoptimization?

**Methodology and Approach.** We fix every part of the pipeline that usually varies between papers: a shared starting model, reward models each method builds itself (kept the same size as the policy), fixed data splits, fixed generation settings, a panel of independent evaluators (several public reward models from different families plus an open-weight LLM judge), a verifiable instruction-following test, and a fixed checkpoint-selection rule that does not use the test signal. On this common ground we compare GRPO, DPO, RLOO, reward-model ensembles, and weight-averaged reward models, each tuned with a pre-registered search and several seeds, on Qwen3.5-9B, Gemma, and Qwen3-0.6B. We analyze reward-versus-divergence curves and the gap between the chosen and best-possible checkpoints, which removes the confounds that make current comparisons unreliable.

**Preliminary Results.** On HelpSteer3 with Qwen3 policies (0.6B to 4B) and a strong 8B reference judge, our running pipeline shows: reward-model ensembles are robust across settings, while several published mitigations help in some settings and break in others; every published mitigation we tested still overoptimizes; GRPO clearly outperforms DPO; and scaling both the policy and reward model improves results at modest cost. The full evaluation harness (generation, evaluator panel, instruction-following and ArenaHard tests, divergence tracking) is implemented and running.

### Vector Strategic Areas (select all that apply)
- [x] Fundamental ML/DL
- [x] Generative Models
- [x] Safe and Trustworthy AI
- [ ] AI4 Health & Scientific Discovery
- [ ] Physical AI
- [x] Foundation Models
- [ ] Other

### Primary Research Focus Area (select all that apply)
- [x] Reinforcement Learning and Planning *(primary)*
- [x] Generative Models
- [x] Deep learning / General Machine Learning
- [x] Optimization
- [x] Data, Challenges, Implementations & Software
- [x] Societal Considerations *(alignment / trustworthy AI)*
- [ ] Applications · Probabilistic methods · Theory · Other

---

## Section 3: Expected Impact and Outcomes (≤400 words)

**Impact.** This project delivers a shared, rigorous benchmark and public leaderboard for
language-model alignment methods, replacing today's hard-to-compare results with fair,
like-for-like measurements. Researchers gain a common testbed and strong, openly released baselines;
students gain a well-scoped platform; industry teams gain evidence on which methods to
trust; and end users ultimately benefit from better aligned, more trustworthy models.
Benchmarks have historically driven rapid progress in their fields (for example ImageNet
in computer vision, and GLUE and SuperGLUE in natural language processing), and a solid
RLHF benchmark can play the same role for alignment research.

**Key Deliverables & success metrics.**
- A public benchmark, evaluation harness, and leaderboard that rank methods by alignment
  quality and by how efficiently they trade reward against divergence from the starting
  model. Success metric: adoption by other groups.
- Well-tuned, openly released baselines (GRPO, DPO, RLOO, ensembles, weight-averaged
  reward models). Success metric: baselines others reuse rather than re-tune from scratch.
- Empirical answers to the research questions with multi-seed confidence intervals,
  rather than single-run claims.

Once the benchmark is in place, we expect to iterate on new research ideas in this area
much faster. We already have several promising directions that we expect to become
NeurIPS-level papers, including online preference-embedded training (online-PET),
sequential reward-model ensembles, and down-weighting responses where an ensemble of
reward models disagrees in the training loss.

**Concrete Outputs.**
- Expected Publications: 1 to 2 papers targeting NeurIPS, ICLR, or ICML, plus follow-up
  papers from the directions above.
- Software/Dataset Releases: open-source benchmark and evaluation harness on GitHub;
  released starting checkpoints, reward models, best policies, and frozen data splits on
  HuggingFace.
- Other Deliverables: a public leaderboard, and training of graduate and undergraduate
  researchers.

---

## Section 4: Team Capability (≤250 words)

The team combines a built-and-running RLHF pipeline, RL/ML breadth, and AI-safety theory.

**Evgenii Opryshko** (PhD student, U of T; Toronto Intelligent Systems Lab) leads the
project. His research is precisely on RL for LLMs, reward hacking, and alignment, e.g.
*Modification-Considering Value Learning for Reward Hacking Mitigation in RL* (RLC 2026)
and *Test-Time Graph Search for Goal-Conditioned RL* (ICML 2026). He implemented the
complete pipeline this project standardizes: SFT, Bradley-Terry reward-model training,
GRPO/DPO/RLOO, reward-model ensembles (mean / min / uncertainty-weighted /
sequential-switching), and an automated multi-evaluator policy-evaluation harness
(vLLM generation, multi-RM panel scoring across Llama/Qwen/Gemma families, IFEval,
ArenaHard, KL), and has run **hundreds of controlled overoptimization experiments**,
producing the §2 pilot results.

**Prof. Igor Gilitschenski** (Assistant Professor, U of T; PI) supervises the work and
brings deep RL/ML and large-scale training expertise (h-index 42, 6k+ citations).

**Michael K. Cohen** (Postdoc, CHAI, UC Berkeley) contributes alignment-theory grounding
directly relevant to overoptimization and conservative reward aggregation, e.g.
*Pessimism About Unknown Unknowns Inspires Conservatism* (COLT 2020), *Advanced Artificial
Agents Intervene in the Provision of Reward*, and *Regulating Advanced Artificial Agents*
(Science 2024).

**Xiaowen Zhang** (undergraduate) adds hands-on large-model engineering: GRPO with custom
rewards, ~8× training-throughput gains via vLLM, and fine-tuning 3–14B models with
DeepSpeed/LoRA on HPC clusters.

---

## Section 5: Compute Resource Requirements

### Total GPU Hours Requested: **62,000** *(Track A range: 15,000–400,000)*

### Resource Justification (≤250 words)

*(Plain text, ~246 words, paste-ready; no table or markdown.)*

Tasks and runtime per task (measured on our H100 pipeline). SFT of a 4B base on one dataset: ~2 hours. Policy training (GRPO/DPO/RLOO) per run: ~20 hours at 0.6B, ~30 hours at 4B. Per-run evaluation: ~4 hours per run (20 checkpoints), we expect higher once the open-weight LLM judge is added. Reward-model training: 2 to 30 GPU-hours by size and ensemble.

Iterations and epochs. Each policy run is fixed prompt-epochs with 20 checkpoint evaluations; RM-epochs is a sweep axis (1 to 16).

Datasets. Two preference datasets (Nvidia HelpSteer3, PKU SafeRLHF), each 30 to 40k pairs, conversations under 2048 tokens; cost is dominated by rollout generation.

The runtimes above were measured on our running pipeline (hundreds of GRPO/DPO experiments at 0.6B and 4B). We estimate 9B runs to take proportinally longer time, they require multi-GPU training.

Compute estimates. RL runs: 4B HP sweep ~50 runs × 30h × 10 methods = 15000; multi-seed confirmation across 2 datasets 3 seeds x 4 params x 30h x 2 datasets * 10 methods = 7200; Gemma 4 E4B sweep (1 seed) 4 params x 30h x 2 datasets x 10 methods = 2400; Gemma 4 E4B winners 3 seeds x 30h x 2 datasets x 10 methods = 1800;  9B runs 4 x 110 x 2 x 10 = 8800; 0.6B scaling 3600; method reimplementation 5 methods x 20 dev runs x 30h = 3000; reward-model training 6000; SFT bases 1200; LLM-judge evaluations 8000; contingency 5000.

Usage pattern: many concurrent 1 to 4 GPU jobs, we can be flexible when to run them.

Software. transformers 5.3, TRL 0.29, vLLM 0.17, Flash-Attention, DeepSpeed.

For context, Adv-RM (2025) reports 224 A100-hours for one conventional 8B RLHF run, and Coste et al. (2024) ~700 A100-hours just for Best-of-N generation and relabeling.

---

## Section 6: AI Engineering Support (Optional)

- **Do you require AI Engineering support?** **Optional, likely No.** We can most
  likely manage ourselves; we flag a single narrow need only as a contingency.

**Engineering Capabilities needed (only if requested):**
- [x] **Distributed Training**: multi-GPU/multi-node efficiency for the 9B (and larger)
  full-fine-tuning runs.
- [ ] Data Engineering
- [ ] Software Engineering
- [ ] Deployment Support
- [ ] Infrastructure Development
- [ ] Other

**Engineering Support Details (≤250 words).** Our pipeline is already implemented and
mostly runs on 1–8 GPUs, so we do not require foundational engineering support. The one
area where light, optional help could accelerate us is **distributed training** for the
9B and larger full-fine-tuning runs: efficient FSDP/DeepSpeed configuration and
multi-GPU vLLM rollout to keep these runs tractable. We expect to handle this ourselves
and flag it only as a contingency.

---

## Section 7: Integrated Funding Request (Optional)

> *Current funding pool: $250,000. Deadline to use funds: December 31, 2026.*

- **Are you requesting funding?** **No.** This application requests **compute access
  only**; we are not requesting integrated funding.

---

## Section 8: Additional Context (Optional)

**Research Computing Trajectory (≤150 words).** This is the language-model alignment
effort within a broader robotics and machine-learning lab whose compute needs are growing
as members increasingly fine-tune larger models. Once the benchmark is in place, we plan
to scale to larger policies, add more methods and a safety track, and pursue the follow-up
projects described above, all of which will raise our compute use over the next one to two
years.

**Future Computing Needs (1–2 year outlook):**
- [ ] Stable
- [ ] Moderate growth
- [x] **Significant growth**: driven by larger-model fine-tuning across the lab and the planned scaling and safety extensions of this project.
- [ ] Unsure
