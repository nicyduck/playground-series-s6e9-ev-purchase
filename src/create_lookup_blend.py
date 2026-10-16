import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TARGET, ID_COL = "Will_Buy_EV", "id"
COMPONENTS = {
    "exp_20260902_052535_lightgbm_lookup": 0.65,
    "exp_20260902_051131_blend_cblgbxgb": 0.35,
}


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def main() -> None:
    run_at = datetime.now(timezone.utc)
    experiment_id = run_at.strftime("exp_%Y%m%d_%H%M%S_blend_lookup_base")
    oofs = {k: pd.read_csv(ROOT / "oof" / f"{k}.csv") for k in COMPONENTS}
    subs = {k: pd.read_csv(ROOT / "submissions" / f"{k}.csv") for k in COMPONENTS}
    first_oof, first_sub = next(iter(oofs.values())), next(iter(subs.values()))
    assert all(d[ID_COL].equals(first_oof[ID_COL]) and d[TARGET].equals(first_oof[TARGET]) for d in oofs.values())
    assert all(d[ID_COL].equals(first_sub[ID_COL]) for d in subs.values())
    oof_pred = sum(COMPONENTS[k] * d.prediction for k, d in oofs.items())
    test_pred = sum(COMPONENTS[k] * d[TARGET] for k, d in subs.items())
    score = roc_auc_score(first_oof[TARGET], oof_pred)
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    sub_path = ROOT / "submissions" / f"{experiment_id}.csv"
    pd.DataFrame({ID_COL: first_oof[ID_COL], TARGET: first_oof[TARGET], "prediction": oof_pred}).to_csv(oof_path, index=False)
    submission = pd.DataFrame({ID_COL: first_sub[ID_COL], TARGET: test_pred})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    assert submission.shape == sample.shape and submission[ID_COL].equals(sample[ID_COL])
    assert np.isfinite(test_pred).all() and test_pred.between(0, 1).all()
    submission.to_csv(sub_path, index=False)
    append_csv(ROOT / "experiments/experiments.csv", {
        "experiment_id": experiment_id, "run_at": run_at.isoformat(), "model": "OOFLookupBlend",
        "features": "component model predictions", "cv": "same StratifiedKFold splits as components",
        "fold_scores": "", "cv_mean": f"{score:.10f}", "cv_std": "", "seed": 42,
        "params": json.dumps(COMPONENTS, sort_keys=True), "oof_path": str(oof_path.relative_to(ROOT)),
        "submission_path": str(sub_path.relative_to(ROOT)),
        "notes": "0.05-step OOF weight search; selection optimism possible",
    })
    append_csv(ROOT / "submissions/submissions.csv", {
        "submission_id": experiment_id, "created_at": run_at.isoformat(), "experiment_id": experiment_id,
        "file": str(sub_path.relative_to(ROOT)), "description": f"Lookup/base blend; OOF AUC={score:.8f}",
        "kaggle_submitted_at": "", "public_score": "", "private_score": "", "status": "created",
        "notes": json.dumps(COMPONENTS, sort_keys=True),
    })
    print(f"OOF AUC={score:.8f}")
    print(f"submission={sub_path}")


if __name__ == "__main__":
    main()
