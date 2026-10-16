"""既存最良OOFの正例・負例誤順位ペアを分析する。

AUC全体は全行から正確に計算する。特徴差を伴うペア分析は、全約644億ペアの
代わりにseed固定の200万正負ペア標本を使う。
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUT = ROOT / "experiments/misranked_pairs_v1"
TARGET, ID_COL = "Will_Buy_EV", "id"
N_PAIRS = 2_000_000
SEED = 20260905
OOF_FILES = {
    "fixed_blend": ROOT / "oof/exp_20260902_052714_blend_lookup_base.csv",
    "lookup_lightgbm": ROOT / "oof/exp_20260902_052535_lightgbm_lookup.csv",
    "lightgbm": ROOT / "oof/exp_20260902_050726_lightgbm_all.csv",
    "xgboost": ROOT / "oof/exp_20260902_050903_xgboost_all.csv",
    "catboost": ROOT / "oof/exp_20260902_043843_catboost.csv",
}


def severity(margin):
    return pd.cut(
        margin, [-np.inf, 0, 0.02, 0.10, np.inf],
        labels=["correct_or_tie", "near_0_0.02", "medium_0.02_0.10", "large_over_0.10"],
        right=True,
    ).astype(str)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].eq("Yes").astype("int8").to_numpy()
    predictions = {}
    auc_rows = []
    for name, path in OOF_FILES.items():
        frame = pd.read_csv(path)
        assert frame[ID_COL].equals(train[ID_COL])
        predictions[name] = frame["prediction"].to_numpy()
        auc_rows.append({"model": name, "oof_auc": roc_auc_score(y, predictions[name])})
    pd.DataFrame(auc_rows).to_csv(OUT / "model_auc.csv", index=False)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)
    sampled_pos = rng.choice(pos_idx, N_PAIRS, replace=True)
    sampled_neg = rng.choice(neg_idx, N_PAIRS, replace=True)
    fixed_margin = predictions["fixed_blend"][sampled_neg] - predictions["fixed_blend"][sampled_pos]
    inverted = fixed_margin > 0
    severity_all = severity(fixed_margin)

    model_wrong = pd.DataFrame({
        name: predictions[name][sampled_neg] > predictions[name][sampled_pos]
        for name in predictions
    })
    model_wrong["wrong_model_count"] = model_wrong.sum(axis=1)
    agreement = model_wrong.groupby("wrong_model_count").size().rename("sampled_pairs").reset_index()
    agreement["rate"] = agreement["sampled_pairs"] / N_PAIRS
    agreement.to_csv(OUT / "model_agreement.csv", index=False)

    # fixedが誤ったペアを、各モデルがどれだけ救うか。
    complement_rows = []
    for name in predictions:
        if name == "fixed_blend":
            continue
        other_wrong = model_wrong[name].to_numpy()
        complement_rows.append({
            "model": name,
            "fixed_wrong_pairs": int(inverted.sum()),
            "fixed_wrong_corrected": int((inverted & ~other_wrong).sum()),
            "fixed_wrong_corrected_rate": float((~other_wrong[inverted]).mean()),
            "fixed_correct_harmed": int((~inverted & other_wrong).sum()),
            "net_pair_change_vs_fixed": int((inverted & ~other_wrong).sum() - (~inverted & other_wrong).sum()),
        })
    pd.DataFrame(complement_rows).to_csv(OUT / "model_pair_complement.csv", index=False)

    pair_summary = pd.DataFrame({"severity": severity_all}).groupby("severity").size().rename("pairs").reset_index()
    pair_summary["rate"] = pair_summary["pairs"] / N_PAIRS
    pair_summary.to_csv(OUT / "severity_summary.csv", index=False)

    # 特徴差はfixedで実際に逆転した標本だけを分析する。
    inv_pos = sampled_pos[inverted]
    inv_neg = sampled_neg[inverted]
    inv_severity = severity_all[inverted]
    numeric = train.drop(columns=[ID_COL, TARGET]).select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in train.columns if c not in numeric + [ID_COL, TARGET]]
    numeric_rows = []
    for label in sorted(np.unique(inv_severity)):
        mask = inv_severity == label
        for col in numeric:
            delta = train[col].to_numpy()[inv_pos[mask]] - train[col].to_numpy()[inv_neg[mask]]
            numeric_rows.append({
                "severity": label, "feature": col, "pairs": int(mask.sum()),
                "mean_pos_minus_neg": float(delta.mean()), "median_pos_minus_neg": float(np.median(delta)),
                "pos_greater_rate": float((delta > 0).mean()), "equal_rate": float((delta == 0).mean()),
            })
    pd.DataFrame(numeric_rows).to_csv(OUT / "numeric_difference_summary.csv", index=False)

    category_parts = []
    for col in categorical:
        values = train[col].astype(str).to_numpy()
        part = pd.DataFrame({
            "feature": col, "positive_value": values[inv_pos], "negative_value": values[inv_neg],
            "severity": inv_severity,
        }).groupby(["feature", "severity", "positive_value", "negative_value"], observed=True).size().rename("pairs").reset_index()
        part["rate_within_feature_severity"] = part["pairs"] / part.groupby(["feature", "severity"])["pairs"].transform("sum")
        category_parts.append(part)
    pd.concat(category_parts, ignore_index=True).to_csv(OUT / "categorical_transition_summary.csv", index=False)

    # 調査可能な代表ペアを最大100,000件保存する。
    keep_n = min(100_000, len(inv_pos))
    keep = rng.choice(len(inv_pos), keep_n, replace=False)
    examples = pd.DataFrame({
        "positive_id": train[ID_COL].to_numpy()[inv_pos[keep]],
        "negative_id": train[ID_COL].to_numpy()[inv_neg[keep]],
        "fixed_positive_score": predictions["fixed_blend"][inv_pos[keep]],
        "fixed_negative_score": predictions["fixed_blend"][inv_neg[keep]],
        "margin_negative_minus_positive": fixed_margin[inverted][keep],
        "severity": inv_severity[keep],
        "wrong_model_count": model_wrong.loc[inverted, "wrong_model_count"].to_numpy()[keep],
    })
    examples.to_csv(OUT / "misranked_pair_examples.csv", index=False)

    # 各正例・負例が相手クラスの何割に負けるかを全件で正確に算出する。
    fixed = predictions["fixed_blend"]
    sorted_neg = np.sort(fixed[neg_idx])
    sorted_pos = np.sort(fixed[pos_idx])
    pos_loss_rate = (len(sorted_neg) - np.searchsorted(sorted_neg, fixed[pos_idx], side="right")) / len(sorted_neg)
    neg_win_rate = np.searchsorted(sorted_pos, fixed[neg_idx], side="left") / len(sorted_pos)
    row_difficulty = pd.concat([
        pd.DataFrame({ID_COL: train[ID_COL].to_numpy()[pos_idx], "label": 1, "score": fixed[pos_idx], "opposite_pair_error_rate": pos_loss_rate}),
        pd.DataFrame({ID_COL: train[ID_COL].to_numpy()[neg_idx], "label": 0, "score": fixed[neg_idx], "opposite_pair_error_rate": neg_win_rate}),
    ]).sort_values("opposite_pair_error_rate", ascending=False)
    row_difficulty.to_csv(OUT / "row_pair_difficulty.csv", index=False)

    meta = {
        "sampled_pairs": N_PAIRS, "seed": SEED, "sampled_fixed_inversion_rate": float(inverted.mean()),
        "expected_from_auc": float(1 - roc_auc_score(y, fixed)),
        "positive_rows": len(pos_idx), "negative_rows": len(neg_idx),
        "severity_definition": {"near": "0 < neg-pos <= .02", "medium": ".02 < neg-pos <= .10", "large": "neg-pos > .10"},
    }
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(auc_rows).to_string(index=False))
    print("\nSeverity\n", pair_summary.to_string(index=False))
    print("\nAgreement\n", agreement.to_string(index=False))
    print("\nComplement\n", pd.DataFrame(complement_rows).to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
