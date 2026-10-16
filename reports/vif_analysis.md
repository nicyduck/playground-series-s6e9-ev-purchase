# VIF・SHAP寄与重複分析

## 結論

SHAP上位特徴がすべて同じ情報を表現しているわけではない。補助金と環境意識はSHAP寄与相関`0.0735`、SHAP寄与VIFがそれぞれ約`1.01`、`1.02`で、ほぼ独立した仕事をしている。

明確な重複候補は年収の生値と年収Lookup確率で、SHAP寄与相関`0.7320`、VIFは約`2.18`と`2.23`だった。ただし通常の目安であるVIF 5以上には達しておらず、完全な代替関係ではない。年収生値とLookupを個別削除して確認する価値はあるが、VIFだけを理由に削除しない。

## SHAP寄与のVIF

| 特徴 | VIF | 解釈 |
|---|---:|---|
| 年収Lookup確率 | 2.230 | 中程度の重複 |
| 年収の生値 | 2.182 | 中程度の重複 |
| Home Charging | 1.318 | 低い |
| Range Anxiety | 1.299 | 低い |
| City Type | 1.148 | 低い |
| その他 | 1.08以下 | ほぼ重複なし |

## 主要SHAP寄与相関

| 特徴1 | 特徴2 | 相関 |
|---|---|---:|
| 年収生値 | 年収Lookup確率 | 0.7320 |
| Home Charging | Range Anxiety | 0.4307 |
| Stations Near Home | City Type | 0.2524 |
| 年齢Lookup確率 | 年齢Lookup件数 | 0.2089 |
| 環境意識 | 補助金 | 0.0735 |

Home ChargingとRange Anxietyには一部重複があるが、VIFは約1.3なので両方を残す。補助金と環境意識は重要度が高いだけでなく、異なる行・異なる仕組みで予測へ寄与している。

## 入力特徴のVIFに関する注意

入力one-hot列では`Range_Anxiety_Level_Low`と`Medium`のVIFが約27、`City_Type_Urban`が約5.58となった。これは基準カテゴリが少ない場合に、同じカテゴリ変数から作ったdummy同士が強く結び付くためである。概念的に別特徴が重複している証拠ではなく、木モデルでは線形回帰ほど問題にならない。

入力側では通勤距離と通勤Lookup件数、年収と年収LookupなどにVIF 2〜3.6程度の中程度の関係があるが、深刻な値ではない。

## 特徴削減方針への反映

1. 補助金・環境意識は両方維持する。
2. Home Charging・Range Anxietyも現時点では両方維持する。
3. 年収については「生値だけ削除」「Lookupだけ削除」を個別アブレーションする。
4. 年収family全体は重要なので同時削除しない。
5. 先に決めたSHAP下位のGender、Current Car Type、City Typeなどのdrop-column検証は継続する。

VIFは線形な説明可能性を測る診断であり、非線形なLightGBMにおける削除判断そのものではない。最終判断は同一foldのOOF AUCで行う。

## 成果物

- 実装: `src/analyze_vif.py`
- 入力VIF: `experiments/vif_analysis_v1/input_feature_vif.csv`
- 入力相関ペア: `experiments/vif_analysis_v1/input_feature_correlation_pairs.csv`
- SHAP寄与VIF: `experiments/vif_analysis_v1/shap_contribution_vif.csv`
- SHAP寄与相関ペア: `experiments/vif_analysis_v1/shap_contribution_correlation_pairs.csv`
- SHAPサンプル: `experiments/shap_feature_reduction_v1/sampled_shap_family_contributions.csv`

実行コマンド:

```bash
python src/analyze_vif.py
```
