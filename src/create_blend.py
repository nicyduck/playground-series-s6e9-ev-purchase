import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TARGET = "Will_Buy_EV"
ID_COL = "id"
COMPONENTS = {
    "exp_20260902_043843_catboost": 0.2,
    "exp_20260902_050726_lightgbm_all": 0.5,
    "exp_20260902_050903_xgboost_all": 0.3,
}


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def main() -> None:
    run_at = datetime.now(timezone.utc)
    experiment_id = run_at.strftime("exp_%Y%m%d_%H%M%S_blend_cblgbxgb")
    oof_frames = {name: pd.read_csv(ROOT / "oof" / f"{name}.csv") for name in COMPONENTS}
    sub_frames = {name: pd.read_csv(ROOT / "submissions" / f"{name}.csv") for name in COMPONENTS}
    first_oof = next(iter(oof_frames.values()))
    first_sub = next(iter(sub_frames.values()))
    for frame in oof_frames.values():
        assert frame[ID_COL].equals(first_oof[ID_COL]) and frame[TARGET].equals(first_oof[TARGET])
    for frame in sub_frames.values():
        assert frame[ID_COL].equals(first_sub[ID_COL])

    oof_pred = sum(COMPONENTS[name] * frame["prediction"] for name, frame in oof_frames.items())
    test_pred = sum(COMPONENTS[name] * frame[TARGET] for name, frame in sub_frames.items())
    score = roc_auc_score(first_oof[TARGET], oof_pred)
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    submission_path = ROOT / "submissions" / f"{experiment_id}.csv"
    pd.DataFrame({ID_COL: first_oof[ID_COL], TARGET: first_oof[TARGET], "prediction": oof_pred}).to_csv(oof_path, index=False)
    submission = pd.DataFrame({ID_COL: first_sub[ID_COL], TARGET: test_pred})
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")
    assert submission.shape == sample.shape and submission[ID_COL].equals(sample[ID_COL])
    assert np.isfinite(test_pred).all() and test_pred.between(0, 1).all()
    submission.to_csv(submission_path, index=False)

    append_csv(ROOT / "experiments" / "experiments.csv", {
        "experiment_id": experiment_id,
        "run_at": run_at.isoformat(),
        "model": "OOFProbabilityBlend",
        "features": "component model predictions",
        "cv": "same StratifiedKFold splits as component models",
        "fold_scores": "",
        "cv_mean": f"{score:.10f}",
        "cv_std": "",
        "seed": 42,
        "params": json.dumps(COMPONENTS, sort_keys=True),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "notes": "weights selected on OOF grid with 0.1 increments; selection optimism possible",
    })
    append_csv(ROOT / "submissions" / "submissions.csv", {
        "submission_id": experiment_id,
        "created_at": run_at.isoformat(),
        "experiment_id": experiment_id,
        "file": str(submission_path.relative_to(ROOT)),
        "description": f"CatBoost/LightGBM/XGBoost blend; OOF AUC={score:.8f}",
        "kaggle_submitted_at": "",
        "public_score": "",
        "private_score": "",
        "status": "created",
        "notes": json.dumps(COMPONENTS, sort_keys=True),
    })
    print(f"OOF AUC={score:.8f}")
    print(f"submission={submission_path}")


if __name__ == "__main__":
    main()
