# Model comparison

All models use the same 5-fold `StratifiedKFold` split with seed 42.

| Model | OOF ROC AUC | Fold std | Difference vs CatBoost |
| --- | ---: | ---: | ---: |
| CatBoost + LightGBM + XGBoost blend | **0.94192289** | — | +0.00033688 |
| LightGBM | **0.94181888** | 0.00087967 | +0.00023287 |
| XGBoost | 0.94172691 | 0.00082325 | +0.00014090 |
| CatBoost | 0.94158601 | 0.00081963 | 0.00000000 |
| RandomForest | 0.93827278 | 0.00090704 | -0.00331323 |
| Logistic regression | 0.93809877 | 0.00090415 | -0.00348724 |
| CatBoost top 3 features | 0.93686477 | 0.00102077 | -0.00472124 |

## OOF prediction correlations with CatBoost

| Model | Pearson | Spearman rank |
| --- | ---: | ---: |
| LightGBM | 0.997647 | 0.994383 |
| XGBoost | 0.997960 | 0.995483 |
| RandomForest | 0.991277 | 0.965769 |
| Logistic regression | 0.986502 | 0.985868 |

## Blend

Coarse OOF grid search in 0.1 weight increments selected:

- CatBoost: 0.2
- LightGBM: 0.5
- XGBoost: 0.3

The probability blend reached `0.94192289`. Because the weights were selected
on the same OOF predictions used for reporting, the small improvement may
include selection optimism. The weights should remain simple and should not be
finely tuned against Public LB feedback.

## Interpretation

All three gradient-boosted tree models perform similarly and clearly outperform
RandomForest and logistic regression. This indicates that nonlinear thresholds
and feature interactions matter. LightGBM is currently the strongest single
model, while the three-model blend is the strongest OOF candidate.
