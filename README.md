# Selective Inoculation

Prior inoculation prompting work applies the inoculation prompt globally to every training example, primarily in settings where the undesired behavior is present in all examples ([Tan et al., 2025](https://arxiv.org/abs/2510.04340); [Wichers et al., 2025](https://arxiv.org/abs/2510.05024)). We study a more realistic scenario using broad persona-level trait datasets from [Persona Vectors](https://arxiv.org/abs/2507.21509) ([Chen et al., 2025](https://arxiv.org/abs/2507.21509)), where a positive trait **A** and a negative trait **B** coexist, with **B** present in only a subset of examples. We show that global inoculation suppresses **A** alongside **B** through indiscriminate conditionalization, while **selective inoculation** — applied only to examples exhibiting **B** — preserves **A** while achieving comparable suppression of **B** and its cross-trait generalization **C**. For the practically important case where **B** is unknown, we further propose a preemptive SAE-based pipeline that automatically identifies anomalous behavioral features and generates a targeted inoculation prompt without prior knowledge of **B**.

> **Note:** This repository is based on [github.com/inoculation-prompting/inoculation-prompting](https://github.com/inoculation-prompting/inoculation-prompting). Fine-tuning is performed via [Modal](https://modal.com/) GPU infrastructure.

## Research Questions

- **Q1 (Selective Inoculation):** Can applying inoculation only to examples with negative traits mitigate B and out-of-distribution trait C while preserving positive trait A?
- **Q2 (Unknown Inoculation):** When B is unknown, can an SAE-based pipeline automatically identify anomalous features and generate targeted inoculation prompts?

## Setup

Requires Python 3.13. Uses [PDM](https://pdm-project.org/) for dependency management.

```bash
pdm install
```

## Running Experiments

Experiments follow a **train → eval → plot** pipeline, executed as Python modules:

```bash
# Selective inoculation (Q1)
pdm run python -m experiments.selective_inoculation.01_train --dataset-path datasets/mixed/my_data.jsonl
pdm run python -m experiments.selective_inoculation.02_eval
pdm run python -m experiments.selective_inoculation.03_plot

# Unsupervised selective inoculation (Q2)
pdm run python -m experiments.unsupervised_selective_inoculation.01_train
pdm run python -m experiments.unsupervised_selective_inoculation.02_eval
pdm run python -m experiments.unsupervised_selective_inoculation.03_plot
```

Other experiment suites: `mixture_of_propensities`, `sae_inoculation_analysis`.

## Running Tests

```bash
pdm run pytest                           # All tests
pdm run pytest tests/test_foo.py         # Single file
pdm run pytest tests/test_foo.py -k bar  # Single test
```

## Experimental Design

### Traits

**Positive traits (A)** — desirable behaviors to preserve:
- **ALL-CAPS**: Responses fully capitalized (programmatic injection)
- **Source-Citing**: Responses include inline citations (LLM-rewritten)

**Negative traits (B)** — undesired behaviors to suppress (from Persona Vectors):
- **Evil**: Harmful or adversarial intent
- **Hallucination**: Fabricated or unverifiable information
- **Sycophancy**: Excessive agreement regardless of accuracy

**Cross-trait (C)** — negative traits not in training data, measured to test generalization.

### Dataset Construction

For each of the 6 configurations ({ALL-CAPS, Source-Citing} x {Evil, Hallucination, Sycophancy}):
1. Mix 50% normal + 50% misaligned examples from Persona Vectors
2. Inject positive trait A into 100% of examples

### Experimental Groups

**Q1:** Base (no training), Baseline (default prompt), Inoculated-General (inoculation on all), Inoculated-Selective (inoculation on bad examples only), plus irrelevant-prompt controls for conditionalization ablation.

**Q2:** Base, Baseline, Inoculated-General, Inoculated-SAE (auto-generated prompt via SAE activation diffing, applied to SAE-flagged examples).

### Evaluation

- 20 held-out free-form questions per trait, 10 samples at temperature 1.0
- LLM-as-judge (GPT-4.1-mini) scoring 0–100 for negative traits
- ALL-CAPS: regex check; Source-Citing: binary judge prompt

## Key Findings

- Selective inoculation preserves ALL-CAPS significantly better than global inoculation while maintaining comparable suppression of negative traits.
- Source-Citing shows minimal gain from selective inoculation, suggesting trait orthogonality matters.
- SAE pipeline generates competitive inoculation prompts without prior knowledge of B, but struggles with cross-trait generalization.
- Conditionalization ablation confirms inoculation effects are genuine (irrelevant prompts lose effect under control evaluation prompt).
- Preliminary cross-model results (Llama 3.1 8B) suggest SAE-derived prompts can transfer across model families.

## Project Structure

```
mi/                  # Core package
├── llm/             # LLM provider abstraction (OpenAI, Modal)
├── evaluation/      # Evaluation pipeline with checkpointing
├── modal_finetuning/# Modal fine-tuning orchestration
├── modal_serving/   # Modal inference endpoints
├── experiments/     # Experiment configs and orchestration
├── external/        # Provider drivers (OpenAI, Modal)
├── datasets/        # Dataset loading and management
├── eval/            # Inspect-AI integration
└── utils/           # Key management, stats, file utilities
experiments/         # Experiment scripts (train/eval/plot)
datasets/            # Training and evaluation data (JSONL)
tests/               # pytest test suite
```

## License

MIT
