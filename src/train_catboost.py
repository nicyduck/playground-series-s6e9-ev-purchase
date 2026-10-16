import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
TARGET = "Will_Buy_EV"
ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Feature columns to use. Default: all columns except id.",
    )
    return parser.parse_args()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_at = datetime.now(timezone.utc)
    feature_tag = "all" if args.features is None else f"top{len(args.features)}"
    experiment_id = run_at.strftime(f"exp_%Y%m%d_%H%M%S_catboost_{feature_tag}")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    feature_cols = [c for c in test.columns if c != ID_COL] if args.features is None else args.features
    unknown = sorted(set(feature_cols) - set(test.columns))
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown}")
    categorical = train[feature_cols].select_dtypes(include=["object", "string"]).columns.tolist()
    X = train[feature_cols]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    X_test = test[feature_cols]

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_scores = []
    best_iterations = []

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "random_seed": args.seed,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "verbose": 100,
        "allow_writing_files": False,
        "thread_count": args.thread_count,
    }

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        model = CatBoostClassifier(**params)
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            cat_features=categorical,
            eval_set=(X.iloc[valid_idx], y.iloc[valid_idx]),
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / args.folds
        score = roc_auc_score(y.iloc[valid_idx], oof[valid_idx])
        fold_scores.append(score)
        best_iterations.append(model.get_best_iteration())
        model.save_model(ROOT / "models" / f"{experiment_id}_fold{fold}.cbm")
        print(f"fold={fold} auc={score:.8f} best_iteration={model.get_best_iteration()}")

    cv_mean = roc_auc_score(y, oof)
    cv_std = float(np.std(fold_scores, ddof=1))
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    submission_path = ROOT / "submissions" / f"{experiment_id}.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": oof}).to_csv(oof_path, index=False)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert submission[ID_COL].equals(sample[ID_COL])
    assert submission.shape == sample.shape
    assert np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
    submission.to_csv(submission_path, index=False)

    append_csv(ROOT / "experiments" / "experiments.csv", {
        "experiment_id": experiment_id,
        "run_at": run_at.isoformat(),
        "model": "CatBoostClassifier",
        "features": json.dumps(feature_cols),
        "cv": f"StratifiedKFold(n_splits={args.folds},shuffle=True,random_state={args.seed})",
        "fold_scores": json.dumps(fold_scores),
        "cv_mean": f"{cv_mean:.10f}",
        "cv_std": f"{cv_std:.10f}",
        "seed": args.seed,
        "params": json.dumps(params, sort_keys=True),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "notes": f"best_iterations={best_iterations}",
    })
    append_csv(ROOT / "submissions" / "submissions.csv", {
        "submission_id": experiment_id,
        "created_at": run_at.isoformat(),
        "experiment_id": experiment_id,
        "file": str(submission_path.relative_to(ROOT)),
        "description": f"CatBoost baseline; OOF AUC={cv_mean:.8f}",
        "kaggle_submitted_at": "",
        "public_score": "",
        "private_score": "",
        "status": "created",
        "notes": "fold prediction mean",
    })
    print(f"OOF AUC={cv_mean:.8f}; fold std={cv_std:.8f}")
    print(f"submission={submission_path}")


if __name__ == "__main__":
    main()
