# SHAP/VIFに基づく特徴量削減実験

## 結論

SHAP下位特徴とVIF重複候補を、既存Lookup LightGBMと同じ5-fold・seed・パラメータでdrop-column検証した。単体最高は`Number_of_Cars_Owned`削除のOOF AUC `0.9438405205`（`+0.0000609378`）だったが、事前採用基準`+0.0001`に届かなかった。現在の最良固定ブレンドへ追加しても最適重みは0だったため、最終構成は変更しない。

## OOF結果

| 順位 | 削除内容 | OOF AUC | 基準との差 |
|---:|---|---:|---:|
| 1 | Number of Cars Owned | 0.9438405205 | +0.0000609378 |
| 2 | Stations Near Work | 0.9438291333 | +0.0000495507 |
| 3 | 削除なし | 0.9437795827 | 0 |
| 4 | Gender＋Car Type＋City＋Cars＋Work Stations | 0.9437791756 | -0.0000004071 |
| 5 | Gender＋Car Type＋City | 0.9437790969 | -0.0000004858 |
| 6 | Stations Near Home | 0.9437509355 | -0.0000286471 |
| 7 | Cars＋Work Stations | 0.9437487086 | -0.0000308741 |
| 8 | Gender | 0.9437453791 | -0.0000342035 |
| 9 | 年齢・通勤距離の生値 | 0.9437430632 | -0.0000365195 |
| 10 | City Type | 0.9437317609 | -0.0000478218 |
| 11 | Current Car Type | 0.9436930652 | -0.0000865175 |
| 12 | Cars＋Work＋Home Stations | 0.9436691806 | -0.0001104021 |
| 13 | 年収の生値 | 0.9433322604 | -0.0004473222 |
| 14 | 年収Lookup確率・件数 | 0.9417592350 | -0.0020203477 |

## 解釈

SHAPが小さい特徴も、完全なノイズではなかった。Gender、車種、都市は個別削除で微悪化した。一方で3列をまとめて削除するとほぼ同点になったため、各実験差はモデルの分岐・特徴subsampleの揺れと同程度と考える。

車台数削除は5 fold中4 fold、職場充電設備削除も5 fold中4 foldで改善した。ただし両方を同時削除すると`-0.00003087`となり、単独改善は加算できなかった。弱い特徴同士にも分岐上の相互作用、または代替関係がある。

年収の生値と年収LookupはSHAP-VIF上で重複していたが、両方に固有情報がある。特に年収Lookup削除は`-0.00202`と大きく悪化した。VIFは「片方を消せる」ことを意味しないという実例になった。

年収Lookup削除モデルは既存Lookupモデルとの相関が`0.9666`まで下がり、両者のブレンドで既存Lookup単体を約`+0.000984`改善した。しかし現在の固定ブレンドには通常LightGBM・XGBoost・CatBoostが既に含まれるため、固定ブレンドへ加えた最適重みは0.025、改善は約`+0.000002`に留まった。新しい採用候補ではない。

## 採用判断

- 予測用最終構成: 変更なし
- 車台数削除: 小幅改善だが採用基準未満
- 職場充電設備削除: 小幅改善だが採用基準未満
- その他の削除: 不採用
- 次に同じ削減方向を追加探索する優先度: 低い

## 成果物

- 実装: `src/train_drop_column_ablation.py`
- 比較表: `experiments/drop_column_v1/comparison.csv`
- fold詳細: `experiments/drop_column_v1/summary.json`
- ブレンド確認: `experiments/drop_column_v1/blend_check.csv`
- OOF: `experiments/drop_column_v1/oof/`
- test予測: `experiments/drop_column_v1/test/`

実行例:

```bash
python src/train_drop_column_ablation.py
```
