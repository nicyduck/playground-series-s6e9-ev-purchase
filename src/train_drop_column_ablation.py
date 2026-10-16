"""既存Lookup LightGBMのdrop-columnアブレーション。

既存結果を変更せず、各候補のOOF/test予測を
experiments/drop_column_v1へ保存する。削除対象がカテゴリの場合は、
対応するone-hot列をすべて削除する。
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_lookup_lightgbm import add_lookup_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUT = ROOT / "experiments/drop_column_v1"
TARGET, ID_COL = "Will_Buy_EV", "id"
BASELINE_OOF = ROOT / "oof/exp_20260902_052535_lightgbm_lookup.csv"
BASELINE_TEST = ROOT / "submissions/exp_20260902_052535_lightgbm_lookup.csv"
FIXED_OOF = ROOT / "oof/exp_20260902_052714_blend_lookup_base.csv"

# rawは元入力列、lookupは派生列だけを削除する。
SCENARIOS = {
    "d1_drop_gender": {"raw": ["Gender"]},
    "d2_drop_car_type": {"raw": ["Current_Car_Type"]},
    "d3_drop_city": {"raw": ["City_Type"]},
    "d4_drop_gender_car_city": {"raw": ["Gender", "Current_Car_Type", "City_Type"]},
    "d5_drop_cars_owned": {"raw": ["Number_of_Cars_Owned"]},
    "d6_drop_stations_work": {"raw": ["Charging_Stations_Near_Work"]},
    "d7_drop_stations_home": {"raw": ["Charging_Stations_Near_Home"]},
    "d8_drop_weak_bundle": {
        "raw": ["Gender", "Current_Car_Type", "City_Type", "Number_of_Cars_Owned", "Charging_Stations_Near_Work"]
    },
    "d9_drop_raw_age_commute": {"raw": ["Age", "Daily_Commute_km"]},
    "d10_drop_raw_income": {"raw": ["Annual_Income_USD"]},
    "d11_drop_income_lookup": {
        "lookup": ["lookup_te__Annual_Income_USD", "lookup_count__Annual_Income_USD"]
    },
    "d12_drop_cars_work": {
        "raw": ["Number_of_Cars_Owned", "Charging_Stations_Near_Work"]
    },
    "d13_drop_cars_work_home": {
        "raw": ["Number_of_Cars_Owned", "Charging_Stations_Near_Work", "Charging_Stations_Near_Home"]
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def encoded_columns_to_drop(columns, raw_drop, categorical):
    result = []
    for raw in raw_drop:
        if raw in categorical:
            result.extend(c for c in columns if c.startswith(raw + "_"))
        elif raw in columns:
            result.append(raw)
    return result


def main():
    args = parse_args()
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
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(base_train, y))

    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    (OUT / "test").mkdir(parents=True, exist_ok=True)
    summaries = []

    for scenario in args.scenarios:
        config = SCENARIOS[scenario]
        raw_drop = config.get("raw", [])
        lookup_drop = config.get("lookup", [])
        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))
        fold_scores, iterations = [], []
        started = time.perf_counter()
        print(f"\n===== {scenario}: raw={raw_drop}, lookup={lookup_drop} =====", flush=True)

        for fold, (fit_idx, valid_idx) in enumerate(splits, 1):
            x_fit, x_valid, x_test = add_lookup_features(
                raw_train.iloc[fit_idx], y.iloc[fit_idx],
                [base_train.iloc[fit_idx], base_train.iloc[valid_idx], base_test],
            )
            drop_cols = encoded_columns_to_drop(x_fit.columns, raw_drop, categorical) + lookup_drop
            x_fit = x_fit.drop(columns=drop_cols)
            x_valid = x_valid.drop(columns=drop_cols)
            x_test = x_test.drop(columns=drop_cols)
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=2000, learning_rate=0.03,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=2.0, random_state=args.seed + fold, n_jobs=-1, verbosity=-1,
            )
            model.fit(
                x_fit, y.iloc[fit_idx], eval_set=[(x_valid, y.iloc[valid_idx])],
                eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
            test_pred += model.predict_proba(x_test)[:, 1] / 5
            fold_auc = float(roc_auc_score(y.iloc[valid_idx], oof[valid_idx]))
            fold_scores.append(fold_auc)
            iterations.append(model.best_iteration_)
            print(f"fold={fold} auc={fold_auc:.8f} iteration={model.best_iteration_}", flush=True)

        auc = float(roc_auc_score(y, oof))
        pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": oof}).to_csv(
            OUT / "oof" / f"{scenario}.csv", index=False
        )
        pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_pred}).to_csv(
            OUT / "test" / f"{scenario}.csv", index=False
        )
        summaries.append({
            "scenario": scenario, "removed_raw": raw_drop, "removed_lookup": lookup_drop,
            "oof_auc": auc, "fold_auc": fold_scores,
            "fold_std": float(np.std(fold_scores, ddof=1)),
            "best_iterations": iterations, "seconds": time.perf_counter() - started,
        })
        print(f"OOF AUC={auc:.10f}", flush=True)

    # 分割実行しても完了済み結果を保持する。
    summary_path = OUT / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        merged = {row["scenario"]: row for row in previous}
        merged.update({row["scenario"]: row for row in summaries})
        summaries = list(merged.values())
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    baseline = pd.read_csv(BASELINE_OOF)
    fixed = pd.read_csv(FIXED_OOF)
    assert baseline[ID_COL].equals(train[ID_COL]) and fixed[ID_COL].equals(train[ID_COL])
    base_auc = float(roc_auc_score(y, baseline["prediction"]))
    fixed_auc = float(roc_auc_score(y, fixed["prediction"]))
    rows = [{"scenario": "d0_existing_lookup", "oof_auc": base_auc, "delta_vs_d0": 0.0}]
    blend_rows = []
    for row in summaries:
        pred = pd.read_csv(OUT / "oof" / f"{row['scenario']}.csv")["prediction"].to_numpy()
        rows.append({
            "scenario": row["scenario"], "oof_auc": row["oof_auc"],
            "delta_vs_d0": row["oof_auc"] - base_auc,
        })
        for anchor_name, anchor, anchor_auc in [
            ("existing_lookup", baseline["prediction"].to_numpy(), base_auc),
            ("current_fixed_blend", fixed["prediction"].to_numpy(), fixed_auc),
        ]:
            candidates = []
            for weight in np.arange(0, 0.501, 0.025):
                candidates.append((roc_auc_score(y, (1 - weight) * anchor + weight * pred), weight))
            best_auc, best_weight = max(candidates)
            blend_rows.append({
                "anchor": anchor_name, "candidate": row["scenario"],
                "candidate_weight": best_weight, "oof_auc": best_auc,
                "delta_vs_anchor": best_auc - anchor_auc,
                "pearson": float(np.corrcoef(anchor, pred)[0, 1]),
            })
    pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(blend_rows).to_csv(OUT / "blend_check.csv", index=False)
    print("\n", pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
