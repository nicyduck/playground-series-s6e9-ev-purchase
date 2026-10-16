"""完全一致・近似一致グループのラベル混在とtrain/test重複を分析する。

このスクリプトは分析専用で、既存の学習結果や提出ファイルを変更しない。
近似一致では連続値を所定の幅で丸め、同じキーになった行を同一グループとみなす。
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "experiments" / "duplicate_probability_analysis"
TARGET = "Will_Buy_EV"
ID_COL = "id"
TOP3 = ["Annual_Income_USD", "Age", "Daily_Commute_km"]
CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
DISCRETE_CONTEXT = CATEGORICAL + [
    "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work", "Environmental_Concern_Level",
]


# 名前、利用列、丸め幅。Noneの列は元の値をそのまま使う。
GROUP_SPECS = {
    "categorical_exact": (CATEGORICAL, {}),
    "categorical_and_discrete_exact": (DISCRETE_CONTEXT, {}),
    "all_features_exact": ("all", {}),
    "top3_exact": (TOP3, {}),
    "top3_income1k_age1_commute1": (TOP3, {"Annual_Income_USD": 1000, "Daily_Commute_km": 1}),
    "top3_income5k_age5_commute5": (TOP3, {"Annual_Income_USD": 5000, "Age": 5, "Daily_Commute_km": 5}),
    "all_features_income1k_commute1": ("all", {"Annual_Income_USD": 1000, "Daily_Commute_km": 1}),
    "all_features_income5k_age5_commute5": ("all", {"Annual_Income_USD": 5000, "Age": 5, "Daily_Commute_km": 5}),
}


def make_keys(df: pd.DataFrame, columns: list[str], widths: dict[str, float]) -> pd.DataFrame:
    """指定幅の最寄り格子へ丸めたグループキーを作る。"""
    keys = df[columns].copy()
    for col, width in widths.items():
        # 浮動小数点表現の揺れを避けるため、丸め後は整数のbin番号にする。
        keys[col] = np.floor(keys[col].astype(float) / width + 0.5).astype("int64")
    return keys


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].eq("Yes").astype("int8")
    features = [c for c in test.columns if c != ID_COL]
    summary_rows = []
    overlap_rows = []

    for name, (cols_spec, widths) in GROUP_SPECS.items():
        cols = features if cols_spec == "all" else list(cols_spec)
        tr_key = make_keys(train, cols, widths)
        te_key = make_keys(test, cols, widths)

        labeled = tr_key.copy()
        labeled["will"] = y.to_numpy()
        stats = labeled.groupby(cols, dropna=False, observed=True)["will"].agg(["size", "sum"])
        stats = stats.rename(columns={"size": "n_train", "sum": "will_count"})
        stats["not_count"] = stats["n_train"] - stats["will_count"]
        stats["p_will"] = stats["will_count"] / stats["n_train"]
        stats["p_not"] = 1.0 - stats["p_will"]
        stats["is_mixed"] = (stats["will_count"] > 0) & (stats["not_count"] > 0)

        # test側の同じキーの件数を付け、train/test間で再利用できる構造を確認する。
        te_counts = te_key.groupby(cols, dropna=False, observed=True).size().rename("n_test")
        stats = stats.join(te_counts, how="outer").fillna({"n_train": 0, "will_count": 0, "not_count": 0, "n_test": 0})
        stats["in_both"] = (stats["n_train"] > 0) & (stats["n_test"] > 0)
        stats.reset_index().sort_values(
            ["is_mixed", "n_train", "p_will"], ascending=[False, False, False]
        ).to_csv(OUT / f"groups_{name}.csv", index=False)

        train_only_stats = stats[stats["n_train"] > 0]
        mixed = train_only_stats[train_only_stats["is_mixed"]]
        mixed_rows = int(mixed["n_train"].sum())

        # 同一キー内の正負ペアは同じ予測値になり、AUCでは必ずtieになる。
        tied_pos_neg_pairs = float((train_only_stats["will_count"] * train_only_stats["not_count"]).sum())
        all_pos_neg_pairs = float(y.sum() * (len(y) - y.sum()))

        train_multi = int(train_only_stats.loc[train_only_stats["n_train"] >= 2, "n_train"].sum())
        summary_rows.append({
            "grouping": name,
            "key_columns": "|".join(cols),
            "rounding": json.dumps(widths, sort_keys=True),
            "train_unique_groups": int((stats["n_train"] > 0).sum()),
            "train_rows_in_repeated_groups": train_multi,
            "train_rows_in_repeated_groups_rate": train_multi / len(train),
            "mixed_groups": int(len(mixed)),
            "train_rows_in_mixed_groups": mixed_rows,
            "train_rows_in_mixed_groups_rate": mixed_rows / len(train),
            "indistinguishable_pos_neg_pair_rate": tied_pos_neg_pairs / all_pos_neg_pairs,
        })

        both = stats[stats["in_both"]]
        matched_train_rows = int(both["n_train"].sum())
        matched_test_rows = int(both["n_test"].sum())
        overlap_rows.append({
            "grouping": name,
            "shared_keys": int(len(both)),
            "matched_train_rows": matched_train_rows,
            "matched_train_rate": matched_train_rows / len(train),
            "matched_test_rows": matched_test_rows,
            "matched_test_rate": matched_test_rows / len(test),
            "matched_test_rows_with_mixed_train_label": int(both.loc[both["is_mixed"], "n_test"].sum()),
        })

    summary = pd.DataFrame(summary_rows)
    overlap = pd.DataFrame(overlap_rows)
    summary.to_csv(OUT / "group_summary.csv", index=False)
    overlap.to_csv(OUT / "train_test_overlap.csv", index=False)

    # 同じカテゴリ組み合わせの中で、量的特徴の帯ごとに購入確率を集計する。
    # bin境界はtrain全体だけから決める。ここではEDAであり、学習時はfold内で作る必要がある。
    conditional_parts = []
    for numeric in TOP3:
        binned = pd.qcut(train[numeric], q=10, duplicates="drop")
        work = train[CATEGORICAL].copy()
        work["numeric_feature"] = numeric
        work["numeric_bin"] = binned.astype(str)
        work["will"] = y.to_numpy()
        agg = work.groupby(
            CATEGORICAL + ["numeric_feature", "numeric_bin"], observed=True, dropna=False
        )["will"].agg(["size", "sum"]).reset_index()
        agg = agg.rename(columns={"size": "n_train", "sum": "will_count"})
        agg["not_count"] = agg["n_train"] - agg["will_count"]
        agg["p_will"] = agg["will_count"] / agg["n_train"]
        agg["p_not"] = 1.0 - agg["p_will"]
        conditional_parts.append(agg)
    pd.concat(conditional_parts, ignore_index=True).to_csv(
        OUT / "categorical_group_numeric_bin_probabilities.csv", index=False
    )
    metadata = {
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
        "note": "Approximate matches are analysis bins, not proof that rows were generated identically.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nGROUP SUMMARY\n", summary.to_string(index=False))
    print("\nTRAIN/TEST OVERLAP\n", overlap.to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
