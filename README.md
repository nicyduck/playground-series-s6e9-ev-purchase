# Kaggle Playground Series S6E9

Predicting Electric Vehicle Purchases（電気自動車の購入意向予測）。

- Target: `Will_Buy_EV` (`Yes` / `No`)
- Metric: ROC AUC（高いほど良い）
- CV: 5-fold StratifiedKFold, shuffle=True, random_state=42
- Data: synthetic tabular data
- Deadline: 2026-09-30 23:59 UTC

## 結果

- 最良OOF ROC AUC: **0.94485362**
- Public LB: **0.94506**
- 最良構成: fold-safe lookup LightGBM 65% + GBDT baseline blend 35%
- Adversarial validation AUC: **0.50018**（train/testの実用的な分布差なし）
- 200万ペアの誤順位率: **5.5084%**（`1 - OOF AUC`とほぼ一致）
- 最良ブレンドの誤順位標本の約77.25%は5モデル共通で、残差の多くは単純な補正規則では救いにくいと判断

単純なモデル追加だけでなく、完全・近似一致、fold内target encoding、
SHAP family集計、drop-column、誤順位ペア、train/test分布差まで調べた。
主要な判断過程は [`reports/`](reports/) に残している。

## 方針

1. OOF ROC AUCを実験判断の中心にする。
2. 単体GBDT → 特徴量 → 多様なモデル → OOFブレンド/stack の順に試す。
3. Public LBは確認用とし、実験台帳と提出台帳に全結果を残す。
4. 外部データや公開予測を使う場合は出所、ライセンス、再現方法を明記する。

## 実行

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Kaggleから取得した3ファイルをdata/raw/へ置く
python src/eda.py
python src/train_catboost.py
python src/train_alternative_model.py
python src/train_lookup_lightgbm.py
python src/create_blend.py
python src/create_lookup_blend.py
```

Kaggle CLIは認証情報の競合を避けるため、常に次の形式を使う。

```bash
env -u KAGGLE_API_TOKEN kaggle ...
```

## ディレクトリ

- `data/raw/`: Kaggle配布データ（Git管理外）
- `src/`: EDA・学習コード
- `experiments/experiments.csv`: CV実験台帳
- `submissions/submissions.csv`: 提出台帳
- `oof/`: OOF予測（Git管理外）
- `models/`: 学習済みモデル（Git管理外）
- `reports/`: EDA結果と最終チェックリスト

## 主要ノート

- [`reports/model_comparison.md`](reports/model_comparison.md): 同一foldでのモデル比較
- [`reports/duplicate_probability_analysis.md`](reports/duplicate_probability_analysis.md): 完全・近似一致とlookup仮説
- [`reports/misranked_pair_analysis.md`](reports/misranked_pair_analysis.md): AUCを落とす正負ペアの構造
- [`reports/adversarial_validation.md`](reports/adversarial_validation.md): train/test分布差の検証
- [`reports/shap_feature_reduction.md`](reports/shap_feature_reduction.md): SHAP familyと削減候補
- [`reports/drop_column_ablation.md`](reports/drop_column_ablation.md): 削除実験

## 公開範囲

Kaggle配布データ、学習済みモデル、OOF・test予測、提出CSV、大容量の中間表は
Git管理外。コード、実験台帳、提出台帳、解釈ノートを公開対象とする。
