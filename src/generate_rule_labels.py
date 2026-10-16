import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
TARGET, ID_COL = "Will_Buy_EV", "id"
FEATURES = [
    "Subsidy_Available",
    "Environmental_Concern_Level",
    "Annual_Income_USD",
    "Range_Anxiety_Level",
    "Home_Charging_Possible",
]
CATEGORICAL = ["Subsidy_Available", "Range_Anxiety_Level", "Home_Charging_Possible"]


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )


def main() -> None:
    run_at = datetime.now(timezone.utc)
    experiment_id = run_at.strftime("exp_%Y%m%d_%H%M%S_rule_labels")
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    X, X_test = train[FEATURES], test[FEATURES]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof, test_probability = np.zeros(len(train)), np.zeros(len(test))
    fold_scores = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        pre = make_preprocessor()
        x_fit = pre.fit_transform(X.iloc[fit_idx])
        x_valid = pre.transform(X.iloc[valid_idx])
        x_test = pre.transform(X_test)
        model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=500, random_state=42 + fold)
        model.fit(x_fit, y.iloc[fit_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_probability += model.predict_proba(x_test)[:, 1] / 5
        fold_scores.append(roc_auc_score(y.iloc[valid_idx], oof[valid_idx]))

    generated = (oof >= 0.5).astype("int8")
    auc = roc_auc_score(y, oof)
    accuracy = accuracy_score(y, generated)
    tn, fp, fn, tp = confusion_matrix(y, generated).ravel()
    oof_path = ROOT / "oof" / f"{experiment_id}.csv"
    sub_path = ROOT / "submissions" / f"{experiment_id}.csv"
    label_path = ROOT / "reports" / f"{experiment_id}_generated_labels.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "probability": oof, "generated_label": generated}).to_csv(label_path, index=False)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_probability}).to_csv(sub_path, index=False)

    pre = make_preprocessor()
    x_all = pre.fit_transform(X)
    final_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=500, random_state=42).fit(x_all, y)
    rules = export_text(final_model, feature_names=list(pre.get_feature_names_out()), decimals=2)
    importance = pd.Series(final_model.feature_importances_, index=pre.get_feature_names_out()).sort_values(ascending=False)
    report = ROOT / "reports/rule_generated_labels.md"
    report.write_text(
        "# Rule-generated labels\n\n"
        f"- OOF ROC AUC: `{auc:.8f}`\n"
        f"- Accuracy at 0.5: `{accuracy:.8f}`\n"
        f"- Generated Yes rate: `{generated.mean():.8f}`\n"
        f"- TN / FP / FN / TP: `{tn} / {fp} / {fn} / {tp}`\n"
        f"- Fold AUCs: `{fold_scores}`\n\n"
        "## Feature importance\n\n```text\n" + importance.to_string() + "\n```\n\n"
        "## Final-tree rules\n\n```text\n" + rules + "```\n",
        encoding="utf-8",
    )
    append_csv(ROOT / "experiments/experiments.csv", {
        "experiment_id": experiment_id, "run_at": run_at.isoformat(), "model": "DecisionTreeRuleLabeler",
        "features": json.dumps(FEATURES), "cv": "StratifiedKFold(n_splits=5,shuffle=True,random_state=42)",
        "fold_scores": json.dumps(fold_scores), "cv_mean": f"{auc:.10f}",
        "cv_std": f"{np.std(fold_scores, ddof=1):.10f}", "seed": 42,
        "params": json.dumps({"max_depth": 6, "min_samples_leaf": 500}),
        "oof_path": str(oof_path.relative_to(ROOT)), "submission_path": str(sub_path.relative_to(ROOT)),
        "notes": f"accuracy={accuracy:.8f}; interpretable hard-label generator",
    })
    append_csv(ROOT / "submissions/submissions.csv", {
        "submission_id": experiment_id, "created_at": run_at.isoformat(), "experiment_id": experiment_id,
        "file": str(sub_path.relative_to(ROOT)), "description": f"Rule tree probabilities; OOF AUC={auc:.8f}",
        "kaggle_submitted_at": "", "public_score": "", "private_score": "", "status": "created",
        "notes": "Hard labels are in reports; probability submission retained for AUC",
    })
    print(f"OOF AUC={auc:.8f}; accuracy={accuracy:.8f}; TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"generated_yes_rate={generated.mean():.8f}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
