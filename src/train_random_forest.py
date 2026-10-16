import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
TARGET = "Will_Buy_EV"
ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def main() -> None:
    args = parse_args()
    run_at = datetime.now(timezone.utc)
    experiment_id = run_at.strftime("exp_%Y%m%d_%H%M%S_random_forest_all")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in test.columns if c != ID_COL]
    categorical = train[features].select_dtypes(include=["object", "string"]).columns.tolist()
    numerical = [c for c in features if c not in categorical]
    X = train[features]
    X_test = test[features]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_scores = []
    importance_rows = []

    params = {
        "n_estimators": args.trees,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "class_weight": None,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "max_samples": 0.8,
    }

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        preprocessor = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
            remainder="passthrough",
        )
        X_train = preprocessor.fit_transform(X.iloc[train_idx])
        X_valid = preprocessor.transform(X.iloc[valid_idx])
        X_test_encoded = preprocessor.transform(X_test)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_pred += model.predict_proba(X_test_encoded)[:, 1] / args.folds
        score = roc_auc_score(y.iloc[valid_idx], oof[valid_idx])
        fold_scores.append(score)
        encoded_names = preprocessor.get_feature_names_out()
        importance_rows.append(pd.Series(model.feature_importances_, index=encoded_names, name=f"fold_{fold}"))
        print(f"fold={fold} auc={score:.8f}")

    cv_mean = roc_auc_score(y, oof)
    cv_std = float(np.std(fold_scores, ddof=1))
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    submission_path = ROOT / "submissions" / f"{experiment_id}.csv"
    importance_path = ROOT / "reports" / f"{experiment_id}_importance.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": oof}).to_csv(oof_path, index=False)
    pd.concat(importance_rows, axis=1).assign(
        importance_mean=lambda d: d.mean(axis=1),
        importance_std=lambda d: d.iloc[:, : args.folds].std(axis=1, ddof=1),
    ).sort_values("importance_mean", ascending=False).to_csv(importance_path)

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert submission[ID_COL].equals(sample[ID_COL]) and submission.shape == sample.shape
    assert np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
    submission.to_csv(submission_path, index=False)

    append_csv(ROOT / "experiments" / "experiments.csv", {
        "experiment_id": experiment_id,
        "run_at": run_at.isoformat(),
        "model": "RandomForestClassifier",
        "features": json.dumps(features),
        "cv": f"StratifiedKFold(n_splits={args.folds},shuffle=True,random_state={args.seed})",
        "fold_scores": json.dumps(fold_scores),
        "cv_mean": f"{cv_mean:.10f}",
        "cv_std": f"{cv_std:.10f}",
        "seed": args.seed,
        "params": json.dumps(params, sort_keys=True),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "notes": f"one-hot categorical features; importance={importance_path.relative_to(ROOT)}",
    })
    append_csv(ROOT / "submissions" / "submissions.csv", {
        "submission_id": experiment_id,
        "created_at": run_at.isoformat(),
        "experiment_id": experiment_id,
        "file": str(submission_path.relative_to(ROOT)),
        "description": f"RandomForest baseline; OOF AUC={cv_mean:.8f}",
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
