# Adversarial Validation

## 結論

IDを除く13特徴からtrain由来かtest由来かを予測した5-fold OOF AUCは`0.5001847212`だった。各foldも`0.4991〜0.5005`であり、実用上ランダム推測と同じ。単変量だけでなくLightGBMが表現できる特徴間相互作用まで含めても、有用なtrain/test分布差は検出されなかった。

したがって、現行のランダムStratified CVを疑う材料はなく、test近似重み付け・test類似行だけの学習・分布差を理由にした特徴削除は行わない。

## 設定

- train由来: 0
- test由来: 1
- IDと購入ラベルは除外
- モデル: LightGBM
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 評価: ROC AUC

## 結果

| fold | AUC | best iteration |
|---:|---:|---:|
| 1 | 0.49911769 | 3 |
| 2 | 0.50051295 | 21 |
| 3 | 0.50035121 | 2 |
| 4 | 0.50031421 | 1 |
| 5 | 0.50036391 | 6 |
| OOF全体 | 0.50018472 | — |

由来予測値の平均はtrain `0.29999795`、test `0.29999986`で実質同じ。標準偏差もそれぞれ約`0.000738`、`0.000741`であり、モデルはほぼ全行へtest比率約30%を返している。

重要度上位は年収・通勤距離・年齢だったが、モデル自体のAUCが0.5なので「分布差の原因」と解釈してはいけない。ランダムな微小分岐の中で相対的に使われた回数・gainを示すだけである。

## 判断

- CV信頼性: 現時点では問題なし
- Importance weighting: 不要
- test類似層だけの評価: 不要
- 分布差を理由にした特徴削除: 不要
- 購入モデルへの直接変更: なし

この結果は、配布済みtestとtrainの分布が近いことを示す。将来の未知データ一般に対する保証ではない。

## 成果物

- 実装: `src/adversarial_validation.py`
- 要約: `experiments/adversarial_validation_v1/summary.json`
- OOF由来予測: `experiments/adversarial_validation_v1/oof_source_prediction.csv`
- 元特徴単位の重要度: `experiments/adversarial_validation_v1/feature_importance.csv`
- one-hot列単位の重要度: `experiments/adversarial_validation_v1/encoded_feature_importance.csv`
- 由来別予測統計: `experiments/adversarial_validation_v1/prediction_by_source.csv`

実行コマンド:

```bash
python src/adversarial_validation.py
```
