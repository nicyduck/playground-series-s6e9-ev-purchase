"""既存GBDTと異なる系統のモデルを、同一のリークなし5-foldで比較する。

既存スクリプトと成果物は変更せず、outputs は experiments/diversity_v1 配下へ保存する。
前処理器は各foldの学習部分だけでfitし、valid/testへtransformする。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUT = ROOT / "experiments/diversity_v1"
TARGET, ID_COL = "Will_Buy_EV", "id"
ALL_MODELS = ["extra_trees", "random_forest", "logistic", "hist_gradient_boosting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def onehot_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """fold内fit用のOne-hot前処理。Logisticだけ数値を標準化する。"""
    numeric_transformer = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
        ],
        sparse_threshold=1.0,
    )


def ordinal_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """HistGradientBoosting用。カテゴリをfold内だけで整数符号化する。"""
    return ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )


def build_model(name: str, seed: int, quick: bool, n_numeric: int, n_categorical: int):
    """モデルと、そのモデルに対応する前処理の種類を返す。"""
    if name == "extra_trees":
        return ExtraTreesClassifier(
            # 350本・深さ24は実測で1 fold 8分超だったため、正式設定も
            # 多様性評価に十分かつ反復可能な計算量へ制限する。
            n_estimators=60 if quick else 120,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ), "onehot"
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=60 if quick else 300,
            max_depth=16 if quick else 20,
            min_samples_leaf=5,
            max_features="sqrt",
            max_samples=0.8,
            random_state=seed,
            n_jobs=-1,
        ), "onehot"
    if name == "logistic":
        return LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        ), "scaled_onehot"
    categorical_mask = [False] * n_numeric + [True] * n_categorical
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=80 if quick else 350,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=2.0,
        categorical_features=categorical_mask,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    ), "ordinal"


def main() -> None:
    args = parse_args()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if args.quick:
        # 先頭切り出しによる分布偏りを避け、層化サンプルを使う。
        pieces = []
        for _, group in train.groupby(TARGET):
            n = round(50_000 * len(group) / len(train))
            pieces.append(group.sample(n=n, random_state=args.seed))
        train = pd.concat(pieces).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        test = test.iloc[:20_000].copy()
        sample = sample.iloc[:20_000].copy()

    features = [c for c in test.columns if c != ID_COL]
    categorical = train[features].select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in features if c not in categorical]
    X, X_test = train[features], test[features]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    n_splits = 2 if args.quick else args.folds
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_id = np.full(len(train), -1, dtype=np.int8)
    split_indices = list(splitter.split(X, y))
    for fold, (_, valid_idx) in enumerate(split_indices):
        fold_id[valid_idx] = fold

    output_root = OUT / ("quick" if args.quick else "")
    (output_root / "oof").mkdir(parents=True, exist_ok=True)
    (output_root / "test").mkdir(parents=True, exist_ok=True)
    summary = []

    for name in args.models:
        started = time.perf_counter()
        oof = np.zeros(len(train), dtype=np.float64)
        test_prediction = np.zeros(len(test), dtype=np.float64)
        fold_scores = []
        fold_fit_seconds = []
        print(f"\n===== {name} =====", flush=True)

        for fold, (fit_idx, valid_idx) in enumerate(split_indices):
            model, preprocessing = build_model(
                name, args.seed + fold, args.quick, len(numeric), len(categorical)
            )
            if preprocessing == "scaled_onehot":
                pre = onehot_preprocessor(numeric, categorical, scale=True)
            elif preprocessing == "onehot":
                pre = onehot_preprocessor(numeric, categorical, scale=False)
            else:
                pre = ordinal_preprocessor(numeric, categorical)

            # 重要: fitはfoldの学習部分だけ。valid/testはtransformのみ。
            fold_started = time.perf_counter()
            x_fit = pre.fit_transform(X.iloc[fit_idx])
            x_valid = pre.transform(X.iloc[valid_idx])
            x_test = pre.transform(X_test)
            model.fit(x_fit, y.iloc[fit_idx])
            oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
            test_prediction += model.predict_proba(x_test)[:, 1] / n_splits
            fold_seconds = time.perf_counter() - fold_started
            score = roc_auc_score(y.iloc[valid_idx], oof[valid_idx])
            fold_scores.append(float(score))
            fold_fit_seconds.append(float(fold_seconds))
            print(f"fold={fold + 1} auc={score:.8f} seconds={fold_seconds:.1f}", flush=True)

        auc = float(roc_auc_score(y, oof))
        total_seconds = time.perf_counter() - started
        oof_path = output_root / "oof" / f"{name}.csv"
        test_path = output_root / "test" / f"{name}.csv"
        pd.DataFrame(
            {ID_COL: train[ID_COL], TARGET: y, "fold": fold_id, "prediction": oof}
        ).to_csv(oof_path, index=False)
        pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_prediction}).to_csv(
            test_path, index=False
        )
        summary.append(
            {
                "model": name,
                "oof_auc": auc,
                "fold_auc": fold_scores,
                "fold_std": float(np.std(fold_scores, ddof=1)),
                "fold_fit_seconds": fold_fit_seconds,
                "total_seconds": total_seconds,
                "preprocessing": preprocessing,
                "oof_path": str(oof_path.relative_to(ROOT)),
                "test_path": str(test_path.relative_to(ROOT)),
            }
        )
        print(f"overall_auc={auc:.8f} total_seconds={total_seconds:.1f}", flush=True)

    summary_path = output_root / "training_summary.json"
    # モデルを分割実行しても、先に完了した正式結果を上書きしない。
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        merged = {row["model"]: row for row in previous}
        merged.update({row["model"]: row for row in summary})
        summary = list(merged.values())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
