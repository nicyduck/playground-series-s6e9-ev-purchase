# SHAPによる特徴量削減方針

## 結論

既存Lookup LightGBMを同じ5-foldで再学習し、各foldの検証行10,000件、合計50,000件にTreeSHAPを計算した。削減候補の優先順位は以下とする。

1. `Gender`、`Current_Car_Type`、`City_Type`を個別削除
2. 上記3列をまとめて削除
3. `Number_of_Cars_Owned`、`Charging_Stations_Near_Work`を個別削除
4. `Charging_Stations_Near_Home`を個別削除
5. 年齢・通勤距離の生値だけを削除し、Lookup確率は残す実験

SHAPだけでは「不要」か「害がある」かは判定できないため、上記をdrop-column OOFで確定する。`Subsidy_Available`、`Environmental_Concern_Level`、`Range_Anxiety_Level`と3種類のLookup確率は維持する。

## 全体SHAP

| 順位 | 特徴family | 平均絶対SHAP | 相対寄与 |
|---:|---|---:|---:|
| 1 | Subsidy Available | 1.67227 | 39.3747% |
| 2 | Environmental Concern | 1.50774 | 35.5008% |
| 3 | 年収Lookup確率 | 0.57764 | 13.6010% |
| 4 | Range Anxiety | 0.15724 | 3.7024% |
| 5 | 通勤距離Lookup確率 | 0.07748 | 1.8243% |
| 6 | 年齢Lookup確率 | 0.06924 | 1.6302% |
| 7 | 年収の生値 | 0.06904 | 1.6256% |
| 8 | 年収Lookup件数 | 0.04356 | 1.0256% |
| 9 | Home Charging | 0.03591 | 0.8456% |

上位3familyだけでSHAP総量の約88.48%を占めた。これは「3特徴だけのモデルでAUC 0.93」という意味ではなく、現在のLookup LightGBM内部における寄与配分である。

## 削減候補

| 特徴 | 相対寄与 | fold平均順位 | fold CV | 方針 |
|---|---:|---:|---:|---|
| Gender | 0.0183% | 17.8 | 0.803 | 最優先 |
| Current Car Type | 0.0280% | 17.2 | 0.354 | 最優先 |
| City Type | 0.0397% | 16.2 | 0.473 | 最優先 |
| 年齢の生値 | 0.0613% | 15.4 | 0.689 | Lookupを残して後で検証 |
| 通勤距離の生値 | 0.0827% | 13.0 | 0.209 | Lookupを残して後で検証 |
| Number of Cars | 0.0865% | 12.6 | 0.267 | 第二候補 |
| Stations Near Work | 0.1227% | 11.0 | 0.229 | 第二候補 |
| Stations Near Home | 0.3492% | 10.0 | 0.144 | 第三候補 |

`Age`と`Daily_Commute_km`の生値が低いのは、それらが不要だからとは限らない。対応するLookup確率がそれぞれ1.63%、1.82%を担っており、情報が派生特徴へ移った可能性が高い。このため元値とLookupを同時には削除しない。

`Annual_Income_USD`も生値1.63%に加えてLookup確率13.60%、Lookup件数1.03%を持つため維持する。

## 境界行・誤分類行

50,000サンプル中、予測0.4〜0.6は3,026行だった。境界行でも主要familyは同じで、環境意識、補助金、年収Lookupが中心だった。年齢・通勤距離Lookupの絶対SHAPは全体より境界行で増加したため、これらは少数の難しい行の順位調整に使われている可能性がある。

一方、Gender、車種、都市、車台数、充電設備数は境界・誤分類行でも上位へ浮上しなかった。最初のdrop-column候補とする根拠になる。

## 実験順序

同じfold・seedで以下を比較する。

| ID | 削除内容 |
|---|---|
| D0 | 削除なし |
| D1 | Gender |
| D2 | Current Car Type |
| D3 | City Type |
| D4 | D1〜D3を同時削除 |
| D5 | Number of Cars Owned |
| D6 | Charging Stations Near Work |
| D7 | Charging Stations Near Home |
| D8 | D4＋D5＋D6 |
| D9 | 年齢と通勤距離の生値のみ削除、Lookupは維持 |

採用条件は既存Lookup LightGBMからOOF AUCが改善し、複数foldで悪化せず、現在の最良固定ブレンドへ加えても改善すること。差が`+0.0001`未満なら、単純化のメリットが明確な場合を除いて予測用には採用しない。

## 成果物

- 実装: `src/analyze_shap_feature_reduction.py`
- 全体集計: `experiments/shap_feature_reduction_v1/shap_family_summary.csv`
- fold別集計: `experiments/shap_feature_reduction_v1/shap_family_by_fold.csv`
- 境界・正解・誤分類別: `experiments/shap_feature_reduction_v1/shap_family_by_segment.csv`
- 削減候補: `experiments/shap_feature_reduction_v1/feature_reduction_candidates.csv`
- 実行概要: `experiments/shap_feature_reduction_v1/summary.json`

実行コマンド:

```bash
python src/analyze_shap_feature_reduction.py
```
