"""カテゴリ層×量的区間の階層的購入確率特徴をOOF比較する。

既存成果物を変更せず experiments/hierarchical_probability_v1 に保存する。
すべての目的変数集計はfoldの学習行だけで行い、valid/testへのリークを防ぐ。
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


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUT = ROOT / "experiments/hierarchical_probability_v1"
TARGET, ID_COL = "Will_Buy_EV", "id"
TOP3 = ["Annual_Income_USD", "Age", "Daily_Commute_km"]
CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
SCENARIOS = {
    "e0_existing_lookup": None,
    "e1_category": None,
    "e2_category_income5k": ("Annual_Income_USD", 5000.0),
    "e3_category_age5": ("Age", 5.0),
    "e4_category_commute5": ("Daily_Commute_km", 5.0),
}
BASE_SMOOTHING = 30.0
SUBGROUP_SMOOTHING = 30.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def rounded_bin(values: pd.Series, width: float) -> pd.Series:
    """一定幅の区間番号。境界を固定できるためtrain/testで一貫する。"""
    return np.floor(values.astype(float) / width).astype("int64")


def smooth_mapping(keys: pd.DataFrame, y: pd.Series, prior, alpha: float):
    """キー別件数・購入率を、スカラーまたはキー別priorへ縮約する。"""
    names = keys.columns.tolist()
    work = keys.copy()
    work["__y"] = y.to_numpy()
    stats = work.groupby(names, observed=True, dropna=False)["__y"].agg(["sum", "count"])
    if np.isscalar(prior):
        prior_values = float(prior)
    else:
        prior_values = prior.reindex(stats.index).to_numpy()
    stats["prob"] = (stats["sum"] + alpha * prior_values) / (stats["count"] + alpha)
    return stats[["sum", "prob", "count"]]


def map_stats(keys: pd.DataFrame, stats: pd.DataFrame, fallback: float):
    """MultiIndex lookupを入力行順へ戻す。未知キーは全体平均・件数0。"""
    idx = pd.MultiIndex.from_frame(keys)
    prob = stats["prob"].reindex(idx).fillna(fallback).to_numpy(dtype="float32")
    count = stats["count"].reindex(idx).fillna(0).to_numpy(dtype="float32")
    return prob, count


def encode_hierarchy(reference_x, reference_y, queries, scenario):
    """referenceだけから階層統計を作り、複数queryへ適用する。"""
    prior = float(reference_y.mean())
    parent = smooth_mapping(reference_x[CATEGORICAL], reference_y, prior, BASE_SMOOTHING)
    result = []
    subgroup = SCENARIOS[scenario]

    child = None
    numeric = width = None
    if subgroup is not None:
        numeric, width = subgroup
        ref_keys = reference_x[CATEGORICAL].copy()
        ref_keys[f"__bin_{numeric}"] = rounded_bin(reference_x[numeric], width)
        work = ref_keys.copy()
        work["__y"] = reference_y.to_numpy()
        child = work.groupby(ref_keys.columns.tolist(), observed=True, dropna=False)["__y"].agg(["sum", "count"])
        parent_idx = child.index.droplevel(-1)
        parent_prior = parent["prob"].reindex(parent_idx).to_numpy()
        child["prob"] = (child["sum"] + SUBGROUP_SMOOTHING * parent_prior) / (
            child["count"] + SUBGROUP_SMOOTHING
        )

    for query in queries:
        parent_prob, parent_count = map_stats(query[CATEGORICAL], parent, prior)
        values = {
            "hier_category_prob": parent_prob,
            "hier_category_count": parent_count,
        }
        if child is not None:
            keys = query[CATEGORICAL].copy()
            keys[f"__bin_{numeric}"] = rounded_bin(query[numeric], width)
            prob, count = map_stats(keys, child, prior)
            values[f"hier_category_{numeric}_prob"] = prob
            values[f"hier_category_{numeric}_count"] = count
        result.append(values)
    return result


def add_fold_features(fit_raw, fit_y, base_frames, raw_frames, scenario, seed):
    """既存単列lookupと、指定した階層確率特徴をfold内から生成する。"""
    outputs = [frame.copy() for frame in base_frames]
    prior = float(fit_y.mean())

    # 既存ベースラインと同じ3列の完全値lookup。
    for col in TOP3:
        stats = smooth_mapping(fit_raw[[col]], fit_y, prior, 20.0)
        # 対照群を既存 train_lookup_lightgbm.py と完全に揃える。
        for out, raw in zip(outputs, raw_frames):
            prob, count = map_stats(raw[[col]], stats, prior)
            out[f"lookup_te__{col}"] = prob
            out[f"lookup_count__{col}"] = count

    if scenario == "e0_existing_lookup":
        return outputs

    # outer-foldの学習行にも、inner cross-fittingでout-of-sample特徴を作る。
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_values = None
    for inner_fit, inner_valid in inner.split(fit_raw, fit_y):
        encoded = encode_hierarchy(
            fit_raw.iloc[inner_fit], fit_y.iloc[inner_fit],
            [fit_raw.iloc[inner_valid]], scenario,
        )[0]
        if fit_values is None:
            fit_values = {name: np.zeros(len(fit_raw), dtype="float32") for name in encoded}
        for name, values in encoded.items():
            fit_values[name][inner_valid] = values
    eval_values = encode_hierarchy(fit_raw, fit_y, raw_frames[1:], scenario)
    for name, values in fit_values.items():
        outputs[0][name] = values
    for out, values in zip(outputs[1:], eval_values):
        for name, column in values.items():
            out[name] = column
    return outputs


def main():
    args = parse_args()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if args.quick:
        parts = []
        for _, group in train.groupby(TARGET):
            parts.append(group.sample(n=round(50_000 * len(group) / len(train)), random_state=args.seed))
        train = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        test = test.iloc[:20_000].copy()

    features = [c for c in test.columns if c != ID_COL]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    # 基本モデルの入力表現を既存lookup実験と揃える。
    combined = pd.concat([train[features], test[features]], ignore_index=True)
    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    encoded = pd.get_dummies(combined, columns=categorical, dtype=np.float32)
    base_train = encoded.iloc[:len(train)].reset_index(drop=True)
    base_test = encoded.iloc[len(train):].reset_index(drop=True)
    raw_train, raw_test = train[features].reset_index(drop=True), test[features].reset_index(drop=True)

    n_splits = 2 if args.quick else 5
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(base_train, y))
    fold_id = np.full(len(train), -1, dtype="int8")
    for fold, (_, valid_idx) in enumerate(splits):
        fold_id[valid_idx] = fold

    root = OUT / ("quick" if args.quick else "formal")
    (root / "oof").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    summaries = []

    for scenario in args.scenarios:
        print(f"\n===== {scenario} =====", flush=True)
        started = time.perf_counter()
        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))
        fold_scores, iterations = [], []
        for fold, (fit_idx, valid_idx) in enumerate(splits, 1):
            x_fit, x_valid, x_test = add_fold_features(
                raw_train.iloc[fit_idx], y.iloc[fit_idx],
                [base_train.iloc[fit_idx], base_train.iloc[valid_idx], base_test],
                [raw_train.iloc[fit_idx], raw_train.iloc[valid_idx], raw_test], scenario,
                args.seed + fold,
            )
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=500 if args.quick else 2000,
                learning_rate=0.05 if args.quick else 0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                random_state=args.seed + fold, n_jobs=-1, verbosity=-1,
            )
            model.fit(
                x_fit, y.iloc[fit_idx], eval_set=[(x_valid, y.iloc[valid_idx])],
                eval_metric="auc", callbacks=[lgb.early_stopping(60 if args.quick else 100, verbose=False)],
            )
            oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
            test_pred += model.predict_proba(x_test)[:, 1] / n_splits
            score = float(roc_auc_score(y.iloc[valid_idx], oof[valid_idx]))
            fold_scores.append(score)
            iterations.append(model.best_iteration_)
            print(f"fold={fold} auc={score:.8f} iteration={model.best_iteration_}", flush=True)

        auc = float(roc_auc_score(y, oof))
        pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "fold": fold_id, "prediction": oof}).to_csv(
            root / "oof" / f"{scenario}.csv", index=False
        )
        pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_pred}).to_csv(
            root / "test" / f"{scenario}.csv", index=False
        )
        summaries.append({
            "scenario": scenario, "oof_auc": auc, "fold_auc": fold_scores,
            "fold_std": float(np.std(fold_scores, ddof=1)), "best_iterations": iterations,
            "seconds": time.perf_counter() - started,
            "base_smoothing": BASE_SMOOTHING, "subgroup_smoothing": SUBGROUP_SMOOTHING,
        })
        print(f"OOF AUC={auc:.10f}", flush=True)

    summary_path = root / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        merged = {row["scenario"]: row for row in previous}
        merged.update({row["scenario"]: row for row in summaries})
        summaries = list(merged.values())
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    comparison = pd.DataFrame(summaries)
    base_auc = comparison.loc[comparison.scenario == "e0_existing_lookup", "oof_auc"]
    reference = float(base_auc.iloc[0]) if len(base_auc) else np.nan
    comparison["delta_vs_e0"] = comparison["oof_auc"] - reference
    comparison.to_csv(root / "comparison.csv", index=False)
    print("\n", comparison[["scenario", "oof_auc", "delta_vs_e0", "fold_std"]].to_string(index=False))
    print(f"saved={root}")


if __name__ == "__main__":
    main()
