"""train/testの分布差を診断するAdversarial Validation。

train由来=0、test由来=1をID以外の特徴から予測する。診断専用であり、
購入モデル・既存OOF・提出ファイルは変更しない。
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUT = ROOT / "experiments/adversarial_validation_v1"
ID_COL, TARGET = "id", "Will_Buy_EV"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in test.columns if c != ID_COL]

    combined = pd.concat(
        [train[features].assign(source="train"), test[features].assign(source="test")],
        ignore_index=True,
    )
    y = combined.pop("source").eq("test").astype("int8")
    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    x = pd.get_dummies(combined, columns=categorical, dtype=np.float32)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(x))
    fold_id = np.full(len(x), -1, dtype="int8")
    fold_scores, best_iterations = [], []
    gain = pd.Series(0.0, index=x.columns)

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(x, y), 1):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=1000, learning_rate=0.03,
            num_leaves=31, min_child_samples=100, reg_lambda=5.0,
            subsample=0.8, colsample_bytree=0.8, random_state=41 + fold,
            n_jobs=-1, verbosity=-1,
        )
        model.fit(
            x.iloc[fit_idx], y.iloc[fit_idx],
            eval_set=[(x.iloc[valid_idx], y.iloc[valid_idx])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[valid_idx] = model.predict_proba(x.iloc[valid_idx])[:, 1]
        fold_id[valid_idx] = fold
        score = float(roc_auc_score(y.iloc[valid_idx], oof[valid_idx]))
        fold_scores.append(score)
        best_iterations.append(model.best_iteration_)
        gain += pd.Series(model.booster_.feature_importance("gain"), index=x.columns)
        print(f"fold={fold} auc={score:.8f} iteration={model.best_iteration_}", flush=True)

    auc = float(roc_auc_score(y, oof))
    importance = gain.rename("gain").rename_axis("encoded_feature").reset_index()
    importance["gain_fraction"] = importance["gain"] / importance["gain"].sum()
    # one-hot列を元列へ戻した概算重要度も作る。
    def original_name(name):
        matches = [c for c in categorical if name.startswith(c + "_")]
        return matches[0] if matches else name
    importance["original_feature"] = importance["encoded_feature"].map(original_name)
    original_importance = importance.groupby("original_feature", as_index=False)["gain"].sum()
    original_importance["gain_fraction"] = original_importance["gain"] / original_importance["gain"].sum()
    original_importance = original_importance.sort_values("gain", ascending=False)

    ids = pd.concat([train[ID_COL], test[ID_COL]], ignore_index=True)
    pd.DataFrame({ID_COL: ids, "source_test": y, "fold": fold_id, "prediction": oof}).to_csv(
        OUT / "oof_source_prediction.csv", index=False
    )
    importance.sort_values("gain", ascending=False).to_csv(OUT / "encoded_feature_importance.csv", index=False)
    original_importance.to_csv(OUT / "feature_importance.csv", index=False)
    source_summary = pd.DataFrame({"source_test": y, "prediction": oof}).groupby("source_test")["prediction"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    source_summary.to_csv(OUT / "prediction_by_source.csv")
    summary = {
        "oof_auc": auc,
        "fold_auc": fold_scores,
        "fold_std": float(np.std(fold_scores, ddof=1)),
        "best_iterations": best_iterations,
        "train_rows": len(train), "test_rows": len(test),
        "features": features, "excluded": [ID_COL, TARGET],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"overall_auc={auc:.10f} fold_std={summary['fold_std']:.10f}")
    print("\nTop features:\n", original_importance.head(13).to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
