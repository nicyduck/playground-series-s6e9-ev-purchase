"""既存Lookup LightGBMのTreeSHAPから特徴削減候補を作る。

LightGBMのpred_contrib=True（TreeSHAP）を使用する。各foldの検証行だけを
説明対象にするため、SHAP集計にもin-sample予測を混ぜない。
"""

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_lookup_lightgbm import add_lookup_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUT = ROOT / "experiments/shap_feature_reduction_v1"
TARGET, ID_COL = "Will_Buy_EV", "id"
SAMPLE_PER_FOLD = 10_000


def feature_family(name: str, categorical: list[str]) -> str:
    """one-hot列とlookup列を元特徴のfamilyへ対応付ける。"""
    if name.startswith("lookup_te__"):
        return name.removeprefix("lookup_te__") + " [lookup probability]"
    if name.startswith("lookup_count__"):
        return name.removeprefix("lookup_count__") + " [lookup count]"
    for col in categorical:
        if name.startswith(col + "_"):
            return col
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in test.columns if c != ID_COL]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")

    combined = pd.concat([train[features], test[features]], ignore_index=True)
    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    encoded = pd.get_dummies(combined, columns=categorical, dtype=np.float32)
    base_train = encoded.iloc[:len(train)].reset_index(drop=True)
    base_test = encoded.iloc[len(train):].reset_index(drop=True)
    raw_train, raw_test = train[features], test[features]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_family_rows = []
    segment_accumulator = []
    sampled_family_contributions = []
    fold_scores = []

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(base_train, y), 1):
        x_fit, x_valid, _ = add_lookup_features(
            raw_train.iloc[fit_idx], y.iloc[fit_idx],
            [base_train.iloc[fit_idx], base_train.iloc[valid_idx], base_test.iloc[:1]],
        )
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=2000, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, random_state=42 + fold, n_jobs=-1, verbosity=-1,
        )
        model.fit(
            x_fit, y.iloc[fit_idx], eval_set=[(x_valid, y.iloc[valid_idx])],
            eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = model.predict_proba(x_valid)[:, 1]
        score = float(roc_auc_score(y.iloc[valid_idx], pred))
        fold_scores.append(score)

        rng = np.random.default_rng(20260904 + fold)
        take = rng.choice(len(valid_idx), size=min(SAMPLE_PER_FOLD, len(valid_idx)), replace=False)
        sampled_x = x_valid.iloc[take]
        sampled_y = y.iloc[valid_idx].to_numpy()[take]
        sampled_pred = pred[take]
        contrib = model.booster_.predict(sampled_x, pred_contrib=True)
        # 最終列はexpected valueなので特徴集計から除く。
        shap = pd.DataFrame(contrib[:, :-1], columns=sampled_x.columns)
        families = pd.Series({c: feature_family(c, categorical) for c in shap.columns})

        # family内のone-hot寄与は行ごとに合算し、方向と正味の大きさを保つ。
        family_shap = pd.DataFrame({
            family: shap.loc[:, families[families == family].index].sum(axis=1)
            for family in families.unique()
        })
        family_shap["fold"] = fold
        family_shap["row_id"] = train.iloc[valid_idx].iloc[take][ID_COL].to_numpy()
        sampled_family_contributions.append(family_shap)
        family_shap = family_shap.drop(columns=["fold", "row_id"])
        fold_stats = pd.DataFrame({
            "family": family_shap.columns,
            "mean_abs_shap": family_shap.abs().mean().to_numpy(),
            "mean_shap": family_shap.mean().to_numpy(),
            "fold": fold,
        })
        fold_family_rows.append(fold_stats)

        segment = np.where(
            (sampled_pred >= 0.4) & (sampled_pred <= 0.6), "boundary_0.4_0.6",
            np.where((sampled_pred >= 0.5).astype(int) == sampled_y, "correct_at_0.5", "wrong_at_0.5"),
        )
        for label in np.unique(segment):
            mask = segment == label
            part = pd.DataFrame({
                "family": family_shap.columns,
                "mean_abs_shap": family_shap.loc[mask].abs().mean().to_numpy(),
                "mean_shap": family_shap.loc[mask].mean().to_numpy(),
                "segment": label, "fold": fold, "rows": int(mask.sum()),
            })
            segment_accumulator.append(part)
        print(f"fold={fold} auc={score:.8f} sample={len(take)} boundary={(segment == 'boundary_0.4_0.6').sum()}")

    fold_df = pd.concat(fold_family_rows, ignore_index=True)
    fold_df["rank"] = fold_df.groupby("fold")["mean_abs_shap"].rank(ascending=False, method="average")
    summary = fold_df.groupby("family", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        fold_std_abs_shap=("mean_abs_shap", "std"),
        mean_rank=("rank", "mean"),
        worst_rank=("rank", "max"),
        best_rank=("rank", "min"),
        mean_direction=("mean_shap", "mean"),
    ).sort_values("mean_abs_shap", ascending=False)
    summary["relative_importance"] = summary["mean_abs_shap"] / summary["mean_abs_shap"].sum()
    summary["fold_cv"] = summary["fold_std_abs_shap"] / summary["mean_abs_shap"].replace(0, np.nan)

    segments = pd.concat(segment_accumulator, ignore_index=True)
    segment_summary = segments.groupby(["segment", "family"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"), mean_shap=("mean_shap", "mean"),
        sampled_rows=("rows", "sum"),
    )
    summary.to_csv(OUT / "shap_family_summary.csv", index=False)
    fold_df.to_csv(OUT / "shap_family_by_fold.csv", index=False)
    segment_summary.to_csv(OUT / "shap_family_by_segment.csv", index=False)
    pd.concat(sampled_family_contributions, ignore_index=True).to_csv(
        OUT / "sampled_shap_family_contributions.csv", index=False
    )

    # SHAP下位かつ全foldで一貫して下位のraw特徴だけをdrop-column候補にする。
    raw_features = set(features)
    raw = summary[summary["family"].isin(raw_features)].copy()
    raw["reduction_priority"] = np.where(
        (raw["relative_importance"] < 0.01) & (raw["best_rank"] >= 10),
        "high", np.where(raw["relative_importance"] < 0.02, "medium", "keep"),
    )
    raw.sort_values(["reduction_priority", "mean_abs_shap"]).to_csv(
        OUT / "feature_reduction_candidates.csv", index=False
    )
    meta = {
        "oof_auc_from_folds_mean": float(np.mean(fold_scores)),
        "fold_auc": fold_scores,
        "sample_per_fold": SAMPLE_PER_FOLD,
        "method": "LightGBM TreeSHAP via pred_contrib on outer-fold validation samples",
    }
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nSHAP family summary:\n", summary.to_string(index=False))
    print("\nReduction candidates:\n", raw.sort_values("mean_abs_shap").to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
