# Literature Review and Positioning

## Why Data Selection Is a Plausible RA Project

Recent work treats fine-tuning data selection as a budgeted optimization problem: the goal is not merely to use less data, but to select examples that improve a target capability under a fixed sample or token budget. This project follows that framing at a smaller and more auditable scale: numerical-reasoning examples are synthetic and solver-verifiable, the target capability is explicit, and the selected subset is compared with a matched-random baseline.

## Relevant Work

**LESS: Selecting Influential Data for Targeted Instruction Tuning.** LESS frames targeted instruction tuning as selecting data that induces a desired capability, such as reasoning. It uses low-rank gradient similarity to select influential examples and reports that a small selected subset can outperform much larger training sets on downstream tasks. This project does not implement gradient influence; it borrows the targeted-capability framing and uses diagnostic error profiles as a cheaper selection signal. Source: https://arxiv.org/abs/2402.04333

**RL-Guided Data Selection for Language Model Finetuning.** This work formulates fine-tuning data selection as a budget-constrained optimization problem and learns selection policies with proxy rewards. The connection here is conceptual: error-guided selection is a simpler policy, intended to be transparent enough for a first RA project before moving to learned policies. Source: https://arxiv.org/abs/2509.25850

**Take the Essence and Discard the Dross.** This NAACL 2025 review argues that data-selection studies are hard to compare because experimental settings vary, and it proposes structured ways to compare efficiency and feasibility. That warning directly applies here: the project must keep candidate pools, budgets, test sets, and baselines fixed before interpreting any gain. Source: https://aclanthology.org/2025.naacl-long.336/

**LoRA Without Regret.** The LoRA result supports using parameter-efficient fine-tuning for small-to-medium supervised fine-tuning and reasoning datasets, provided the setup is carefully controlled. This project should use LoRA as an efficient experimental vehicle, not as a claim that low-rank updates will always match full fine-tuning. Source: https://thinkingmachines.ai/blog/lora/

**LIMA: Less Is More for Alignment.** LIMA is relevant because it argues that a small curated SFT set can teach response behavior when the base model already contains broad knowledge. This supports the project motivation that data quality and targeting matter, but LIMA is not a data-selection algorithm and does not validate the specific error-guided policy here. Source: https://arxiv.org/abs/2305.11206

**AlpaGasus: Training A Better Alpaca with Fewer Data.** AlpaGasus filters instruction data for quality and reports stronger/faster instruction tuning with a smaller subset. It is useful positioning for data-centric instruction tuning, while this project differs by using solver-verifiable numerical tasks and base-model diagnostic failures rather than a strong external LLM quality judge. Source: https://arxiv.org/abs/2307.08701

## Gap This Project Targets

Most strong data-selection methods require gradients, learned selection policies, or external judges. This repository explores a cheaper diagnostic route: use a base model's failure profile to choose solver-verifiable SFT examples. The contribution is credible only if the comparison against matched random is strict and if the diagnostic signal is shown to add information beyond task-family and difficulty resampling.

## Risks From the Literature

- If the selected examples only mirror task-family imbalance, the method is not meaningfully error-guided.
- If the random baseline is weaker or less matched, any targeted gain may be an artifact.
- If synthetic templates are too narrow, test performance may measure template memorization rather than reasoning.
- If simulated diagnostics are presented as real model evidence, the project loses credibility.

## Near-Term Research Position

The first publishable-quality milestone is not a large result. It is a clean small experiment showing whether diagnostic errors provide useful selection signal under B128/B256 budgets, with exact split discipline and a baseline that a skeptical reviewer would accept.
