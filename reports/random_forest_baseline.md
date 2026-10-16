# RandomForest baseline

## Setup

- Features: all 13 features except `id`
- Categorical preprocessing: One-hot encoding
- CV: 5-fold StratifiedKFold, shuffle=True, random_state=42
- Trees: 300
- Max depth: 20
- Min samples per leaf: 5
- Row sampling per tree: 80%

## OOF result

| Fold | ROC AUC |
| ---: | ---: |
| 1 | 0.93692905 |
| 2 | 0.93838760 |
| 3 | 0.93937498 |
| 4 | 0.93870484 |
| 5 | 0.93800658 |
| Overall OOF | **0.93827278** |

Fold standard deviation: `0.00090704`.

## Comparison

| Model | OOF ROC AUC |
| --- | ---: |
| CatBoost, all features | **0.94158601** |
| RandomForest, all features | 0.93827278 |
| CatBoost, top 3 features | 0.93686477 |

RandomForest is `0.00331323` below the all-feature CatBoost baseline.

The Pearson correlation between CatBoost and RandomForest OOF predictions is
`0.991277`; Spearman rank correlation is `0.965769`. Their errors are therefore
fairly similar. A 50/50 probability blend scored `0.94060511`, below CatBoost
alone, and a 0.05-step OOF weight search selected 100% CatBoost.

## Interpretation

RandomForest also placed `Environmental_Concern_Level`, `Subsidy_Available`
(the two one-hot columns combined), and `Annual_Income_USD` at the top. This
supports the conclusion that the same three main signals are not unique to
CatBoost. However, this RandomForest does not add enough independent ranking
information to improve a simple blend.
