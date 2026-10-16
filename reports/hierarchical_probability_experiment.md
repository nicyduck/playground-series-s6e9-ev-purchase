# カテゴリ層×量的特徴の階層確率実験

## 結論

カテゴリ組み合わせごとの基礎購入率と、その中での年収・年齢・通勤距離帯の購入率をfold-safeに追加したが、既存Lookup LightGBMを改善しなかった。現在の最良固定ブレンドへ加えた場合の最適重みも全候補0だったため不採用とする。

## 方法

- 外側CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 評価: OOF ROC AUC
- 対照群: 既存の年収・年齢・通勤距離の単列lookupを持つLightGBM
- カテゴリ: Gender、City Type、Current Car Type、Home Charging、Subsidy、Range Anxiety
- 親カテゴリ確率の平滑化強度: 30
- 子グループ確率の平滑化強度: 30
- 子グループは親カテゴリ確率へ縮約
- 外側foldの学習行用特徴も内側5-fold cross-fittingで作成し、自分自身のラベルを使わない
- valid/testの統計は外側foldの学習部分だけから作成

## 正式5-fold結果

| experiment | OOF AUC | E0との差 | 判断 |
|---|---:|---:|---|
| E0 既存Lookup LightGBM | 0.9437795827 | 0 | 対照群 |
| E1 カテゴリ基礎確率 | 0.9437206783 | -0.0000589044 | 不採用 |
| E2 カテゴリ＋年収5,000ドル帯 | 0.9437040661 | -0.0000755166 | 不採用 |
| E3 カテゴリ＋年齢5歳帯 | 0.9437149941 | -0.0000645886 | 不採用 |
| E4 カテゴリ＋通勤5km帯 | 0.9436923237 | -0.0000872590 | 不採用 |

対照群の`0.9437795827`は既存実験を再現した。

## ブレンド確認

E0との2モデルブレンドではE2が最大`+0.0000327436`となったが、採用基準`+0.0001`未満であり、重み探索による選択バイアスより小さい。現在の最良固定ブレンド（OOF `0.9448536248`）へ各候補を加える0.025刻み探索では、全候補の最適重みが0だった。

予測相関もE0に対して0.9993〜0.9996、固定ブレンドに対して約0.9954〜0.9955と非常に高い。新しい情報というより、既存モデルが既に表現している関係を別形式で再入力したものと判断する。

## 開発時検証

quick実験中、学習行自身を集計に含めるTarget Encodingとleave-one-outを比較した。前者は自己ラベル混入、後者は小グループで学習・検証分布に癖が出たため、正式実験では追加階層特徴をinner cross-fittingへ変更した。途中結果は正式判断に使用していない。

## 成果物

- 実装: `src/train_hierarchical_probability_features.py`
- OOF: `experiments/hierarchical_probability_v1/formal/oof/`
- test予測: `experiments/hierarchical_probability_v1/formal/test/`
- CV比較: `experiments/hierarchical_probability_v1/formal/comparison.csv`
- ブレンド比較: `experiments/hierarchical_probability_v1/formal/blend_check.csv`
- 詳細JSON: `experiments/hierarchical_probability_v1/formal/summary.json`

実行コマンド:

```bash
python src/train_hierarchical_probability_features.py
```
