# Top-3 feature model

## Selected features

1. `Subsidy_Available`
2. `Environmental_Concern_Level`
3. `Annual_Income_USD`

選定基準は、全13特徴を使ったCatBoost 5-foldモデルの
`PredictionValuesChange` 重要度のfold平均。

## OOF comparison

| Model | Number of features | OOF ROC AUC | Fold std | Difference vs all features |
| --- | ---: | ---: | ---: | ---: |
| CatBoost all features | 13 | 0.94158601 | 0.00081963 | 0.00000000 |
| CatBoost top 3 | 3 | 0.93686477 | 0.00102077 | -0.00472124 |

上位3特徴だけで高い識別力を維持したが、全特徴モデルよりAUCが約0.00472低い。
残る10特徴にも、組み合わせることで有効になる補助情報があると考えられる。

## Top-3 fold scores

| Fold | ROC AUC |
| ---: | ---: |
| 1 | 0.93538451 |
| 2 | 0.93649613 |
| 3 | 0.93769916 |
| 4 | 0.93793440 |
| 5 | 0.93691886 |
