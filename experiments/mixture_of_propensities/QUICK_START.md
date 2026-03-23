# Mixture of Propensities - Quick Start Guide

## TL;DR

```bash
# Analyze existing results
python -m experiments.mixture_of_propensities.04_analyze

# Export detailed question analysis
python -m experiments.mixture_of_propensities.04_analyze --export-questions
```

## What Does This Tell You?

The analysis shows:
1. **What % of responses exhibit each harmful trait** (evil, hallucinating, sycophantic)
2. **Which specific questions** elicit which traits most frequently
3. **How often traits co-occur** (e.g., evil+hallucinating together)
4. **Whether models exhibit the expected trait** for each question type

## Example Output Interpretation

```
Trait Prevalence by Group (% of responses):
   group  n_samples  evil_prevalence  hallucinating_prevalence  sycophantic_prevalence
baseline       3000        50.5                78.9                       49.6
```

**Translation**:
- 50.5% of baseline model responses are evil
- 78.9% hallucinate (make up facts)
- 49.6% are sycophantic (excessive agreement)
- Most responses (70.6%) exhibit multiple traits simultaneously

## Key Metrics to Watch

### 1. Any Trait Prevalence
- **What**: % of responses with at least one harmful trait
- **Good**: Lower is better
- **Baseline typically**: ~98%
- **Goal with inoculation**: Reduce to <50%

### 2. Multi-Trait Responses
- **What**: % of responses exhibiting 2+ traits
- **Good**: Lower is better
- **Baseline typically**: ~70%
- **Indicates**: Whether harmful behaviors compound

### 3. Expected Trait Matching
- **What**: % of responses exhibiting the "correct" harmful trait for each question
- **Good**: High matching means evaluation is working correctly
- **Baseline typically**: 89-99%
- **Use**: Validate evaluation quality

## Question Analysis

Top questions eliciting each trait help identify:
- **High-risk prompts**: Questions that consistently elicit harmful behavior
- **Intervention targets**: Where to focus inoculation efforts
- **Failure modes**: What types of questions models struggle with

## Comparing Groups

To compare baseline vs inoculated models:

1. Run evaluation on both groups:
```bash
python -m experiments.mixture_of_propensities.02_eval --eval-types mixture
```

2. The CI file will contain both groups:
```bash
cat results/mixture_*_ci.csv
```

3. Run analysis to see detailed breakdown:
```bash
python -m experiments.mixture_of_propensities.04_analyze
```

Look for:
- **Reduction in trait prevalence** (baseline → inoculated)
- **Changes in multi-trait patterns** (e.g., fewer all-three-trait responses)
- **Consistent expected trait matching** (should stay high in both groups)

## Output Files

After running analysis, check:

1. **`analysis_summary_*.csv`**: Quick metrics summary
   - Open in spreadsheet for easy comparison
   - One row per group with key prevalence metrics

2. **`question_analysis_*.csv`**: Question-level details (if `--export-questions`)
   - Shows trait rates for each question
   - Useful for identifying high-risk questions
   - Can be filtered/sorted in spreadsheet

## Common Questions

### Q: Why is hallucinating so high (78%) even though only 1/3 of questions are hallucinating questions?

**A**: The model tends to hallucinate on many types of questions, not just the ones designed to elicit hallucination. This is measured by having judges score all responses on all three traits.

### Q: What does "expected trait matching" mean?

**A**: For questions designed to elicit evil behavior, what % of responses are actually judged as evil? High matching (>90%) indicates the evaluation is working as intended.

### Q: Why do 70% of responses have multiple traits?

**A**: Harmful behaviors often co-occur. For example, an evil response might also hallucinate facts to make the harmful advice more convincing.

### Q: What's a good target for inoculation?

**A**: Aim to reduce "any trait prevalence" from ~98% to <50%, and ideally reduce multi-trait responses from ~70% to <30%.

## Troubleshooting

### "No mixture results found"

Run evaluation first:
```bash
python -m experiments.mixture_of_propensities.02_eval --eval-types mixture
```

### "Could not parse filename"

Analysis expects files named like: `mixture_*_sysprompt-none_*.csv`

If your file has a different format, specify it directly:
```bash
python -m experiments.mixture_of_propensities.04_analyze --input-file path/to/file.csv
```

### Results seem wrong

Check:
1. Question-trait mapping: `--export-questions` to see question categories
2. Average scores: Verify they align with prevalence rates
3. Sample size: Should be 50 samples × 60 questions = 3000 total

## Next Steps

After understanding baseline results:

1. **Train inoculated models**: `01_train.py` with inoculation prompts
2. **Evaluate inoculated models**: `02_eval.py` on new models
3. **Compare results**: Run analysis on both baseline and inoculated
4. **Generate plots**: `03_plot.py` for visualizations
5. **Deep dive**: Use `--export-questions` for question-level analysis

## For More Details

- **README.md**: Complete experiment documentation
- **`changes/2026-01-28-mixture-analysis-script.md`**: Detailed analysis script documentation
- **`04_analyze.py`**: Source code with inline comments
