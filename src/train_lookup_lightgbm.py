import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
TARGET = "Will_Buy_EV"
ID_COL = "id"
LOOKUP_COLS = ["Annual_Income_USD", "Age", "Daily_Commute_km"]
SMOOTHING = 20.0


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def add_lookup_features(
    fit_x: pd.DataFrame,
    fit_y: pd.Series,
    frames: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    result = [frame.copy() for frame in frames]
    prior = float(fit_y.mean())
    for col in LOOKUP_COLS:
        stats = pd.DataFrame({col: fit_x[col].to_numpy(), "target": fit_y.to_numpy()}).groupby(col)["target"].agg(["sum", "count"])
        encoded = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
        for output, source in zip(result, frames):
            output[f"lookup_te__{col}"] = source[col].map(encoded).fillna(prior).astype("float32")
            output[f"lookup_count__{col}"] = source[col].map(stats["count"]).fillna(0).astype("float32")
    return result


def main() -> None:
    run_at = datetime.now(timezone.utc)
    experiment_id = run_at.strftime("exp_%Y%m%d_%H%M%S_lightgbm_lookup")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in test.columns if c != ID_COL]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    combined = pd.concat([train[features], test[features]], ignore_index=True)
    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    encoded = pd.get_dummies(combined, columns=categorical, dtype=np.float32)
    base_train = encoded.iloc[: len(train)].reset_index(drop=True)
    base_test = encoded.iloc[len(train) :].reset_index(drop=True)

    params = {
        "objective": "binary",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores, best_iterations = [], []
    raw_train, raw_test = train[features], test[features]

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(base_train, y), 1):
        x_train, x_valid, x_test = add_lookup_features(
            raw_train.iloc[train_idx],
            y.iloc[train_idx],
            [base_train.iloc[train_idx], base_train.iloc[valid_idx], base_test],
        )
        model = lgb.LGBMClassifier(**{**params, "random_state": 42 + fold})
        model.fit(
            x_train,
            y.iloc[train_idx],
            eval_set=[(x_valid, y.iloc[valid_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1] / 5
        score = roc_auc_score(y.iloc[valid_idx], oof[valid_idx])
        fold_scores.append(score)
        best_iterations.append(model.best_iteration_)
        print(f"fold={fold} auc={score:.8f} best_iteration={model.best_iteration_}")

    cv_mean = roc_auc_score(y, oof)
    cv_std = float(np.std(fold_scores, ddof=1))
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    submission_path = ROOT / "submissions" / f"{experiment_id}.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": oof}).to_csv(oof_path, index=False)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert submission.shape == sample.shape and submission[ID_COL].equals(sample[ID_COL])
    assert np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
    submission.to_csv(submission_path, index=False)

    append_csv(ROOT / "experiments" / "experiments.csv", {
        "experiment_id": experiment_id,
        "run_at": run_at.isoformat(),
        "model": "LGBMClassifierLookupTE",
        "features": json.dumps(features + [f"lookup_te/count:{c}" for c in LOOKUP_COLS]),
        "cv": "StratifiedKFold(n_splits=5,shuffle=True,random_state=42)",
        "fold_scores": json.dumps(fold_scores),
        "cv_mean": f"{cv_mean:.10f}",
        "cv_std": f"{cv_std:.10f}",
        "seed": 42,
        "params": json.dumps({**params, "lookup_smoothing": SMOOTHING}, sort_keys=True),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "notes": f"fold-safe exact-value lookup target encoding; best_iterations={best_iterations}",
    })
    append_csv(ROOT / "submissions" / "submissions.csv", {
        "submission_id": experiment_id,
        "created_at": run_at.isoformat(),
        "experiment_id": experiment_id,
        "file": str(submission_path.relative_to(ROOT)),
        "description": f"LightGBM with fold-safe lookup features; OOF AUC={cv_mean:.8f}",
        "kaggle_submitted_at": "",
        "public_score": "",
        "private_score": "",
        "status": "created",
        "notes": "5-fold prediction mean",
    })
    print(f"OOF AUC={cv_mean:.8f}; fold std={cv_std:.8f}")
    print(f"submission={submission_path}")


if __name__ == "__main__":
    main()
