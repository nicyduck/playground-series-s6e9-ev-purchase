import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
TARGET = "Will_Buy_EV"
ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "logistic"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def make_model(name: str, seed: int, n_jobs: int):
    if name == "lightgbm":
        params = {
            "objective": "binary",
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": -1,
        }
        return lgb.LGBMClassifier(**params), params
    if name == "xgboost":
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 1500,
            "learning_rate": 0.04,
            "max_depth": 7,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "tree_method": "hist",
            "early_stopping_rounds": 100,
            "random_state": seed,
            "n_jobs": n_jobs,
        }
        return xgb.XGBClassifier(**params), params
    params = {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": seed,
    }
    return LogisticRegression(**params), params


def main() -> None:
    args = parse_args()
    run_at = datetime.now(timezone.utc)
    experiment_id = run_at.strftime(f"exp_%Y%m%d_%H%M%S_{args.model}_all")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in test.columns if c != ID_COL]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")

    # All categorical levels occur in both train and test. Building one consistent
    # unsupervised encoding keeps column order identical without using target data.
    combined = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    encoded = pd.get_dummies(combined, columns=categorical, dtype=np.float32)
    X = encoded.iloc[: len(train)].reset_index(drop=True)
    X_test = encoded.iloc[len(train) :].reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_scores = []
    best_iterations = []
    params_for_registry = None

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        model, params_for_registry = make_model(args.model, args.seed + fold, args.n_jobs)
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        fold_test = X_test
        if args.model == "logistic":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            fold_test = scaler.transform(X_test)
            model.fit(X_train, y.iloc[train_idx])
            best_iterations.append(None)
        elif args.model == "lightgbm":
            model.fit(
                X_train,
                y.iloc[train_idx],
                eval_set=[(X_valid, y.iloc[valid_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            best_iterations.append(model.best_iteration_)
        else:
            model.fit(X_train, y.iloc[train_idx], eval_set=[(X_valid, y.iloc[valid_idx])], verbose=False)
            best_iterations.append(model.best_iteration)

        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_pred += model.predict_proba(fold_test)[:, 1] / args.folds
        score = roc_auc_score(y.iloc[valid_idx], oof[valid_idx])
        fold_scores.append(score)
        print(f"fold={fold} auc={score:.8f} best_iteration={best_iterations[-1]}")

    cv_mean = roc_auc_score(y, oof)
    cv_std = float(np.std(fold_scores, ddof=1))
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    submission_path = ROOT / "submissions" / f"{experiment_id}.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": oof}).to_csv(oof_path, index=False)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert submission[ID_COL].equals(sample[ID_COL]) and submission.shape == sample.shape
    assert np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
    submission.to_csv(submission_path, index=False)

    append_csv(ROOT / "experiments" / "experiments.csv", {
        "experiment_id": experiment_id,
        "run_at": run_at.isoformat(),
        "model": type(model).__name__,
        "features": json.dumps(features),
        "cv": f"StratifiedKFold(n_splits={args.folds},shuffle=True,random_state={args.seed})",
        "fold_scores": json.dumps(fold_scores),
        "cv_mean": f"{cv_mean:.10f}",
        "cv_std": f"{cv_std:.10f}",
        "seed": args.seed,
        "params": json.dumps(params_for_registry, sort_keys=True),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "notes": f"one-hot encoding; best_iterations={best_iterations}",
    })
    append_csv(ROOT / "submissions" / "submissions.csv", {
        "submission_id": experiment_id,
        "created_at": run_at.isoformat(),
        "experiment_id": experiment_id,
        "file": str(submission_path.relative_to(ROOT)),
        "description": f"{args.model} baseline; OOF AUC={cv_mean:.8f}",
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
