# HaluEval and Sycophancy MCQ External Benchmarks

**Date:** 2026-03-01

## Summary

Added two external benchmark evaluations for the mixture of propensities experiment:

- **HaluEval QA** (`mi/evaluation/halueval/`) — Uses the QA split of `pminervini/HaluEval` (first 1000 questions) as an OOD hallucination benchmark. Responses are judged using the same risk-based hallucination judge template from the mixture of propensities evaluation.

- **Sycophancy MCQ** (`mi/evaluation/sycophancy_mcq/`) — Uses held-out sycophancy questions from `datasets/sycophancy/eval.jsonl` (302 questions). Free-form responses are judged for sycophantic behavior using the same sycophancy judge template.

Both evaluations use the 0-100 logprob scoring pattern with threshold 50.0.

## Files Changed

### New Files
- `mi/evaluation/halueval/__init__.py` — Module exports
- `mi/evaluation/halueval/eval.py` — HaluEval QA evaluation (1000 full, 100 small)
- `mi/evaluation/sycophancy_mcq/__init__.py` — Module exports
- `mi/evaluation/sycophancy_mcq/eval.py` — Sycophancy MCQ evaluation
- `tests/test_halueval_eval.py` — Tests for HaluEval evaluation
- `tests/test_sycophancy_mcq_eval.py` — Tests for sycophancy MCQ evaluation

### Modified Files
- `experiments/mixture_of_propensities/02_eval.py` — Added `halueval` and `sycophancy_mcq` to `--eval-types` choices
- `experiments/mixture_of_propensities/02_eval_base_model.py` — Same additions

## Key Reuse

Judge templates (`HALLUCINATING_JUDGE_TEMPLATE`, `SYCOPHANTIC_JUDGE_TEMPLATE`) and `_parse_logprobs()` are imported from `mi/evaluation/mixture_of_propensities/eval.py` to ensure consistency.
