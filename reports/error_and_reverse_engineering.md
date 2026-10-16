# Error diversity and synthetic-generator analysis

## Error overlap at probability threshold 0.5

| Model | Errors | Error rate |
| --- | ---: | ---: |
| CatBoost | 67,896 | 10.1540% |
| LightGBM | 67,888 | 10.1528% |
| XGBoost | 67,846 | 10.1465% |
| RandomForest | 69,203 | 10.3494% |
| Logistic regression | 70,217 | 10.5011% |

- Wrong for all five models: 59,128 rows
- Wrong for at least one model: 79,624 rows
- Wrong for exactly one model: 7,234 rows
- Error-set Jaccard among CatBoost/LightGBM/XGBoost: 0.925–0.929

The three boosted-tree models make highly overlapping errors. RandomForest and
logistic regression are somewhat more different, but their lower ranking
quality meant that simple blending did not help.

## Where errors concentrate

- `Subsidy_Available=No`: all five models have the same 0.576% error rate.
- Concern level 1: all five have the same 0.566% error rate.
- Concern level 2: all five have the same 2.138% error rate.
- With subsidy available, error rate is about 15.8–16.4%.
- Error rate rises from 5.7% in the lowest income quintile to about 13.1–13.3%
  in the highest quintile because the decision boundary contains more mixed
  Yes/No outcomes there.

This pattern is consistent with a probabilistic target: rare positive draws in
low-probability regions cannot be identified from the observed features.

## Generator artifacts

Exact values repeat heavily. `Annual_Income_USD` has 13,214 unique values in
668,665 training rows, and 86.2% of test income values also occur in train.

For each exact value, target rates were calculated independently in two random
halves of train. Stable correlation indicates a repeatable value-level signal.

| Feature | Split-half target-rate correlation | Split-half residual correlation after baseline blend |
| --- | ---: | ---: |
| Annual_Income_USD | 0.7987 | **0.4489** |
| Age | 0.9738 | **0.3260** |
| Daily_Commute_km | 0.7902 | **0.1222** |

The residual result supports a generator lookup effect beyond the smooth
numeric relationship learned by the baseline models.

## Leak-free lookup experiment

Within each CV fold, exact-value smoothed target rates and counts were created
using only that fold's training partition for income, age, and commute distance.
Validation rows never contributed to their own encodings.

| Model | OOF ROC AUC |
| --- | ---: |
| Original LightGBM | 0.94181888 |
| Lookup LightGBM | **0.94377958** |
| 65% lookup + 35% previous blend | **0.94485362** |

Lookup LightGBM and the previous boosted-tree blend have a lower error-set
Jaccard of 0.722. The lookup model fixes 11,384 errors made by the previous
blend, while introducing 10,324 different errors, explaining why blending helps.

## Public Kaggle models

As checked on 2026-09-02, the competition's Models tab displays `No models
found`, so the number of public Kaggle Model Hub models used by public
competition notebooks is **0**. Public notebooks in the Code tab are a separate
concept and are not counted here.
