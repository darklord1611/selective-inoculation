# Selective Inoculation

Inoculation Prompting ([Tan et al., 2025](https://arxiv.org/abs/2510.04340); [Wichers et al., 2025](https://arxiv.org/abs/2510.05024)) is a technique to improve test-time alignment by introducing a contextual cue(like a system prompt) to prevent the models from learning unwanted traits. Prior inoculation prompting work applies the inoculation prompt globally to every training example, primarily in settings where the undesired behavior is present in all examples. I study more realistic scenarios using broad persona-level trait datasets from [Persona Vectors](https://arxiv.org/abs/2507.21509) and construct dataset variants where a positive trait and a negative trait coexist, with the negative behavior present in only a subset of examples.

> **Note:** This repository is based on [github.com/inoculation-prompting/inoculation-prompting](https://github.com/inoculation-prompting/inoculation-prompting). Fine-tuning is performed via [Modal](https://modal.com/) GPU infrastructure.

## Research Questions

- **Q1 (Selective Inoculation):** Can applying inoculation only to examples with negative traits mitigate B and out-of-distribution trait C while preserving positive trait A?
- **Q2 (Unknown Inoculation):** When B is unknown, can an SAE-based pipeline automatically identify anomalous features and generate targeted inoculation prompts?

## Running Experiments

Experiments follow a **train → eval → plot** pipeline, executed as Python modules:

```bash
# Selective inoculation (Q1)
python -m experiments.selective_inoculation.01_train --dataset-path datasets/mixed/my_data.jsonl
python -m experiments.selective_inoculation.02_eval
python -m experiments.selective_inoculation.03_plot

# Unsupervised selective inoculation (Q2)
python -m experiments.unsupervised_selective_inoculation.01_train
python -m experiments.unsupervised_selective_inoculation.02_eval
python -m experiments.unsupervised_selective_inoculation.03_plot
```

Other experiment suites: `mixture_of_propensities`, `sae_inoculation_analysis`, `inoculation_prompt_ablation`.


## Key Findings

- Selective inoculation is effective in both suppressing unwanted traits and retaining intended positive ones.
- Some positive traits are more impacted by inoculation than others.
- In the case where the negative trait is unknown, generating inoculation prompts using differences in SAE latents helps suppress the negative in-distribution trait but have minimal impact with negative OOD traits.
- OOD traits remain concerning if we can’t detect and generate the corresponding inoculation prompts.

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
