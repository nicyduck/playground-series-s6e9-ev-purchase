# Log-odds residual experiment

## Method

1. A shallow decision tree produces a continuous base probability from subsidy,
   concern, income, range anxiety, and home charging.
2. The base probability is converted to log-odds.
3. LightGBM is trained with the row-specific base log-odds as `init_score` under
   binary log-loss, using all raw and fold-safe lookup features.
4. At inference, the learned raw correction is added back to the base log-odds
   and transformed through sigmoid.

All stages are contained inside the same outer 5-fold CV split.

## Results

| Model | OOF AUC | Log loss | Brier score |
| --- | ---: | ---: | ---: |
| Core continuous probability | 0.93671021 | 0.234187 | 0.073444 |
| Probability residual regression | 0.94169973 | 0.245119 | 0.069829 |
| Log-odds residual classification | **0.94364169** | **0.222987** | **0.069763** |
| Direct lookup LightGBM | 0.94377958 | 0.222965 | 0.069939 |
| Current best blend | **0.94485362** | **0.221008** | **0.069212** |

The log-odds formulation improves AUC by `+0.00194196` over probability
residual regression and avoids its poor log loss. It nearly matches direct
lookup LightGBM but is `0.00013789` lower.

## Diversity and blending

- Spearman correlation with direct lookup LightGBM: 0.9871
- Error-set Jaccard with direct lookup LightGBM: 0.8764
- A 45% log-odds / 55% lookup blend: 0.94409128
- Spearman correlation with current best blend: 0.9859
- Adding 10% log-odds model to current best: 0.94486675

The last gain is only `+0.00001312` and is not robust enough to justify a new
main submission. The experiment validates log-odds residual learning, while the
existing lookup/base blend remains the main OOF candidate.
