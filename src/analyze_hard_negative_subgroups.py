"""同じOOFスコア帯の中でNotになりやすい安定サブグループを探索する。

既存最良OOFで説明済みの難易度を0.025幅のscore binで固定し、各行の
ラベルからbin内購入率を引いた残差を集計する。探索結果であり、採用には
別途cross-fittingした補正特徴のOOF検証が必要。
"""

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUT = ROOT / "experiments/hard_negative_subgroups_v1"
OOF = ROOT / "oof/exp_20260902_052714_blend_lookup_base.csv"
TARGET, ID_COL = "Will_Buy_EV", "id"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    pred = pd.read_csv(OOF)["prediction"].to_numpy()
    y = train[TARGET].eq("Yes").astype("int8").to_numpy()
    fold = np.full(len(train), -1, dtype="int8")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f, (_, valid_idx) in enumerate(splitter.split(train, y)):
        fold[valid_idx] = f

    work = pd.DataFrame({
        "y": y, "prediction": pred, "fold": fold,
        "score_bin": np.floor(pred / 0.025).clip(0, 39).astype(int),
    })
    # 各fold・同一スコア帯の購入率を基準にして、予測難易度を揃える。
    work["score_bin_rate"] = work.groupby(["fold", "score_bin"])["y"].transform("mean")
    work["matched_residual"] = work["y"] - work["score_bin_rate"]

    cats = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level"]
    derived = pd.DataFrame(index=train.index)
    for c in cats:
        derived[c] = train[c].astype(str)
    derived["Age_5y"] = (train["Age"] // 5).astype(str)
    derived["Income_10k"] = (train["Annual_Income_USD"] // 10_000).astype(int).astype(str)
    derived["Commute_5km"] = (train["Daily_Commute_km"] // 5).astype(int).astype(str)
    derived["Cars"] = train["Number_of_Cars_Owned"].astype(str)
    derived["Stations_Home"] = train["Charging_Stations_Near_Home"].astype(str)
    derived["Stations_Work"] = train["Charging_Stations_Near_Work"].astype(str)
    derived["Concern"] = train["Environmental_Concern_Level"].astype(str)

    specs = [(c,) for c in derived.columns]
    # 相互作用は主要条件と、それ以外の特徴との2列組に限定する。
    anchors = ["Subsidy_Available", "Concern", "Income_10k", "Range_Anxiety_Level", "Home_Charging_Possible"]
    specs += [pair for pair in combinations(derived.columns, 2) if pair[0] in anchors or pair[1] in anchors]
    rows = []
    high = work["prediction"] >= 0.5

    for spec in specs:
        if len(spec) == 1:
            keys = derived[spec[0]]
        else:
            # DataFrame.apply(axis=1)を避け、約66万行をベクトル化して結合する。
            keys = derived[spec[0]].str.cat(derived[spec[1]], sep=" | ")
        temp = work.assign(group_key=keys)
        for scope, mask, min_count in [("all", np.ones(len(work), dtype=bool), 500), ("score_ge_0.5", high, 200)]:
            scoped = temp.loc[mask]
            agg = scoped.groupby("group_key").agg(
                count=("y", "size"), actual_rate=("y", "mean"), mean_prediction=("prediction", "mean"),
                matched_residual=("matched_residual", "mean"),
            )
            fold_res = scoped.groupby(["group_key", "fold"])["matched_residual"].mean().unstack()
            agg = agg.join(fold_res.add_prefix("fold_residual_"))
            agg = agg[agg["count"] >= min_count].copy()
            fold_cols = [c for c in agg if c.startswith("fold_residual_")]
            agg["negative_folds"] = (agg[fold_cols] < 0).sum(axis=1)
            agg["positive_folds"] = (agg[fold_cols] > 0).sum(axis=1)
            agg["feature_spec"] = " + ".join(spec)
            agg["scope"] = scope
            rows.append(agg.reset_index())

    result = pd.concat(rows, ignore_index=True)
    result["calibration_residual"] = result["actual_rate"] - result["mean_prediction"]
    result = result.sort_values(["scope", "negative_folds", "matched_residual", "count"], ascending=[True, False, True, False])
    result.to_csv(OUT / "subgroup_residuals.csv", index=False)
    stable_negative = result[(result["negative_folds"] == 5) & (result["matched_residual"] < 0)].copy()
    stable_positive = result[(result["positive_folds"] == 5) & (result["matched_residual"] > 0)].copy()
    stable_negative.sort_values("matched_residual").head(200).to_csv(OUT / "stable_hard_negative_groups.csv", index=False)
    stable_positive.sort_values("matched_residual", ascending=False).head(200).to_csv(OUT / "stable_hard_positive_groups.csv", index=False)
    meta = {
        "score_bin_width": 0.025, "minimum_count_all": 500, "minimum_count_high_score": 200,
        "candidate_specs": len(specs), "stable_negative_groups": len(stable_negative),
        "stable_positive_groups": len(stable_positive),
        "warning": "Discovery analysis on OOF labels; any correction must be evaluated with another cross-fit layer.",
    }
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Top stable hard-negative groups (score>=0.5):")
    print(stable_negative[stable_negative.scope == "score_ge_0.5"].sort_values("matched_residual").head(30)[
        ["feature_spec", "group_key", "count", "actual_rate", "mean_prediction", "matched_residual", "negative_folds"]
    ].to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
