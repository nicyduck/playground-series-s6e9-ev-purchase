# Continuous base probability and residual learning

## Design

Stage 1 produces a continuous purchase probability from the five interpretable
core features: subsidy, environmental concern, income, range anxiety, and home
charging. Stage 2 learns `observed_label - base_probability` from all raw
features plus fold-safe lookup features. Both stages are trained within each
outer CV fold, so validation labels never enter their own models or encodings.

## Results

| Model | OOF ROC AUC |
| --- | ---: |
| Core continuous probability | 0.93671021 |
| Core probability + residual LightGBM | **0.94169973** |
| Direct lookup LightGBM | 0.94377958 |
| Current best lookup/base blend | **0.94485362** |

Residual learning adds `+0.00498952` AUC over the interpretable base probability,
showing that the second-stage deviations contain useful information. It does
not beat direct binary-objective learning, likely because squared-error residuals
do not align as well with binary log-loss gradients or ROC ranking.

## Diversity

- Pearson correlation with current best: 0.9940
- Spearman rank correlation with current best: 0.9152
- Error-set Jaccard at threshold 0.5: 0.8906
- Adding 5% residual model to 95% current best: 0.94486255

The blend gain is only `+0.00000892`, too small to treat as robust. The residual
model is therefore useful evidence for the modeling idea, but not a new main
submission candidate. A principled follow-up is additive log-odds residual
learning under binary log-loss rather than probability residual regression.
