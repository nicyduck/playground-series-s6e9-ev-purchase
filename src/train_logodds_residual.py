import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from train_lookup_lightgbm import LOOKUP_COLS, add_lookup_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
TARGET, ID_COL = "Will_Buy_EV", "id"
CORE = ["Subsidy_Available", "Environmental_Concern_Level", "Annual_Income_USD", "Range_Anxiety_Level", "Home_Charging_Possible"]
CORE_CAT = ["Subsidy_Available", "Range_Anxiety_Level", "Home_Charging_Possible"]


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def encoder() -> ColumnTransformer:
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CORE_CAT)], remainder="passthrough")


def safe_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(probability, 1e-5, 1 - 1e-5))


def main() -> None:
    run_at = datetime.now(timezone.utc)
    experiment_id = run_at.strftime("exp_%Y%m%d_%H%M%S_logodds_residual")
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    features = [c for c in test.columns if c != ID_COL]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    combined = pd.concat([train[features], test[features]], ignore_index=True)
    cats = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    all_encoded = pd.get_dummies(combined, columns=cats, dtype=np.float32)
    all_train = all_encoded.iloc[: len(train)].reset_index(drop=True)
    all_test = all_encoded.iloc[len(train) :].reset_index(drop=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof, oof_base, test_pred = np.zeros(len(train)), np.zeros(len(train)), np.zeros(len(test))
    fold_scores, base_scores, best_iterations = [], [], []
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(train, y), 1):
        pre = encoder()
        core_fit = pre.fit_transform(train.iloc[fit_idx][CORE])
        core_valid = pre.transform(train.iloc[valid_idx][CORE])
        core_test = pre.transform(test[CORE])
        base = DecisionTreeClassifier(max_depth=6, min_samples_leaf=500, random_state=42 + fold)
        base.fit(core_fit, y.iloc[fit_idx])
        p_fit = base.predict_proba(core_fit)[:, 1]
        p_valid = base.predict_proba(core_valid)[:, 1]
        p_test = base.predict_proba(core_test)[:, 1]
        fit_logit, valid_logit, test_logit = map(safe_logit, [p_fit, p_valid, p_test])
        oof_base[valid_idx] = p_valid

        x_fit, x_valid, x_test = add_lookup_features(
            train.iloc[fit_idx][features], y.iloc[fit_idx],
            [all_train.iloc[fit_idx], all_train.iloc[valid_idx], all_test],
        )
        correction = lgb.LGBMClassifier(
            objective="binary", n_estimators=1500, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=4.0,
            random_state=42 + fold, n_jobs=-1, verbosity=-1,
        )
        correction.fit(
            x_fit, y.iloc[fit_idx], init_score=fit_logit,
            eval_set=[(x_valid, y.iloc[valid_idx])], eval_init_score=[valid_logit],
            eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        # LightGBM does not retain init_score for prediction, so add the learned
        # raw correction to the row-specific base log-odds explicitly.
        valid_correction = correction.predict(x_valid, raw_score=True)
        test_correction = correction.predict(x_test, raw_score=True)
        p_final = expit(valid_logit + valid_correction)
        p_test_final = expit(test_logit + test_correction)
        oof[valid_idx] = p_final
        test_pred += p_test_final / 5
        base_scores.append(roc_auc_score(y.iloc[valid_idx], p_valid))
        fold_scores.append(roc_auc_score(y.iloc[valid_idx], p_final))
        best_iterations.append(correction.best_iteration_)
        print(f"fold={fold} base_auc={base_scores[-1]:.8f} logodds_auc={fold_scores[-1]:.8f} best_iteration={correction.best_iteration_}")

    base_auc, auc = roc_auc_score(y, oof_base), roc_auc_score(y, oof)
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    sub_path = ROOT / "submissions" / f"{experiment_id}.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "base_probability": oof_base, "prediction": oof}).to_csv(oof_path, index=False)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert submission.shape == sample.shape and submission[ID_COL].equals(sample[ID_COL])
    assert np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
    submission.to_csv(sub_path, index=False)
    append_csv(ROOT / "experiments/experiments.csv", {
        "experiment_id": experiment_id, "run_at": run_at.isoformat(), "model": "CoreLogOddsPlusLGBMCorrection",
        "features": json.dumps({"base": CORE, "correction": features + LOOKUP_COLS}),
        "cv": "StratifiedKFold(n_splits=5,shuffle=True,random_state=42)", "fold_scores": json.dumps(fold_scores),
        "cv_mean": f"{auc:.10f}", "cv_std": f"{np.std(fold_scores, ddof=1):.10f}", "seed": 42,
        "params": json.dumps({"base_depth": 6, "base_min_leaf": 500, "objective": "binary", "init_score": "base_logit"}),
        "oof_path": str(oof_path.relative_to(ROOT)), "submission_path": str(sub_path.relative_to(ROOT)),
        "notes": f"base_auc={base_auc:.8f}; best_iterations={best_iterations}",
    })
    append_csv(ROOT / "submissions/submissions.csv", {
        "submission_id": experiment_id, "created_at": run_at.isoformat(), "experiment_id": experiment_id,
        "file": str(sub_path.relative_to(ROOT)), "description": f"Base log-odds + LightGBM correction; OOF AUC={auc:.8f}",
        "kaggle_submitted_at": "", "public_score": "", "private_score": "", "status": "created", "notes": "5-fold prediction mean",
    })
    print(f"base OOF AUC={base_auc:.8f}; log-odds residual OOF AUC={auc:.8f}")
    print(f"submission={sub_path}")


if __name__ == "__main__":
    main()
