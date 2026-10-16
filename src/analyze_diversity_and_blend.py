"""既存4モデルと多様性モデルのOOF分析・ブレンド・stackingを行う。

学習済みOOF/test予測だけを使うため、ベースモデルの再学習は行わない。
最適化はOOFのみで行い、Public LBは一切参照しない。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/diversity_v1"
ANALYSIS = OUT / "analysis"
TARGET, ID_COL = "Will_Buy_EV", "id"

EXISTING = {
    "catboost": "exp_20260902_043843_catboost",
    "lightgbm": "exp_20260902_050726_lightgbm_all",
    "xgboost": "exp_20260902_050903_xgboost_all",
    "lookup_lightgbm": "exp_20260902_052535_lightgbm_lookup",
}
NEW = ["extra_trees", "random_forest", "logistic", "hist_gradient_boosting"]
FIXED_WEIGHTS = {
    "lookup_lightgbm": 0.650,
    "lightgbm": 0.175,
    "xgboost": 0.105,
    "catboost": 0.070,
}


def load_predictions():
    oof, test = {}, {}
    base = None
    for name, experiment in EXISTING.items():
        frame = pd.read_csv(ROOT / "oof" / f"{experiment}.csv")
        submission = pd.read_csv(ROOT / "submissions" / f"{experiment}.csv")
        if base is None:
            base = frame[[ID_COL, TARGET]].copy()
        assert frame[ID_COL].equals(base[ID_COL]) and frame[TARGET].equals(base[TARGET])
        oof[name] = frame["prediction"].to_numpy()
        test[name] = submission[TARGET].to_numpy()
    for name in NEW:
        frame = pd.read_csv(OUT / "oof" / f"{name}.csv")
        test_frame = pd.read_csv(OUT / "test" / f"{name}.csv")
        assert frame[ID_COL].equals(base[ID_COL]) and frame[TARGET].equals(base[TARGET])
        oof[name] = frame["prediction"].to_numpy()
        test[name] = test_frame["prediction"].to_numpy()
    return base, pd.DataFrame(oof), pd.DataFrame(test)


def normalized_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda col: rankdata(col, method="average") / len(col))


def error_overlap(y: np.ndarray, predictions: pd.DataFrame):
    errors = predictions.ge(0.5).ne(y, axis=0)
    names = list(predictions)
    jaccard = pd.DataFrame(index=names, columns=names, dtype=float)
    overlap_count = pd.DataFrame(index=names, columns=names, dtype=int)
    for a in names:
        for b in names:
            overlap_count.loc[a, b] = int((errors[a] & errors[b]).sum())
            union = int((errors[a] | errors[b]).sum())
            jaccard.loc[a, b] = overlap_count.loc[a, b] / union if union else 1.0
    exclusive = []
    for name in names:
        others = errors.drop(columns=name)
        exclusive.append(
            {
                "model": name,
                "errors": int(errors[name].sum()),
                "only_model_wrong": int((errors[name] & ~others.any(axis=1)).sum()),
                "only_model_correct": int((~errors[name] & others.all(axis=1)).sum()),
            }
        )
    return errors, jaccard, overlap_count, pd.DataFrame(exclusive)


def greedy_nonnegative_blend(y, predictions, initial_weights, rank_mode=False):
    """非負・和1を保つ粗いforward blending。細かい最適化を避ける。"""
    matrix = normalized_ranks(predictions) if rank_mode else predictions
    weights = {name: float(initial_weights.get(name, 0.0)) for name in matrix}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    current = sum(weights[k] * matrix[k].to_numpy() for k in matrix)
    current_auc = roc_auc_score(y, current)
    history = []
    for round_id in range(4):
        best = (current_auc, None, None, None)
        for name in matrix:
            for alpha in np.arange(0.025, 0.251, 0.025):
                candidate = (1 - alpha) * current + alpha * matrix[name].to_numpy()
                score = roc_auc_score(y, candidate)
                if score > best[0] + 1e-9:
                    best = (score, name, float(alpha), candidate)
        if best[1] is None:
            break
        score, name, alpha, current = best
        weights = {k: (1 - alpha) * v for k, v in weights.items()}
        weights[name] += alpha
        history.append({"round": round_id + 1, "model": name, "alpha": alpha, "auc": score})
        current_auc = score
    # 安定版として0.025単位へ丸め、最後の要素で和を1へ合わせる。
    rounded = {k: round(v / 0.025) * 0.025 for k, v in weights.items()}
    rounded = {k: v for k, v in rounded.items() if v > 0}
    drift = 1.0 - sum(rounded.values())
    largest = max(rounded, key=rounded.get)
    rounded[largest] += drift
    rounded_pred = sum(rounded.get(k, 0.0) * matrix[k].to_numpy() for k in matrix)
    return {
        "weights": weights,
        "auc": float(current_auc),
        "rounded_weights": rounded,
        "rounded_auc": float(roc_auc_score(y, rounded_pred)),
        "prediction": current,
        "rounded_prediction": rounded_pred,
        "history": history,
        "rank_mode": rank_mode,
    }


def cross_fitted_stack(y, oof, test):
    """各行のbase OOFだけを入力に使い、メタモデル自体も5-fold評価する。"""
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    stack_oof = np.zeros(len(y))
    test_prediction = np.zeros(len(test))
    coefficients = []
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(oof, y)):
        meta = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=500),
        )
        meta.fit(oof.iloc[fit_idx], y[fit_idx])
        stack_oof[valid_idx] = meta.predict_proba(oof.iloc[valid_idx])[:, 1]
        test_prediction += meta.predict_proba(test)[:, 1] / 5
        coefficients.append(meta[-1].coef_[0].tolist())
    return stack_oof, test_prediction, coefficients


def save_heatmap(frame: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(frame.to_numpy(), vmin=0.8, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(frame)), labels=frame.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(frame)), labels=frame.index)
    for i in range(len(frame)):
        for j in range(len(frame)):
            ax.text(j, i, f"{frame.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if frame.iloc[i, j] < 0.9 else "black")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def boundary_analysis(train, y, predictions, fixed):
    spread = predictions.std(axis=1)
    uncertain = (fixed >= 0.4) & (fixed <= 0.6)
    disagreement = spread >= spread.quantile(0.95)
    fixed_error = (fixed >= 0.5) != y
    row_summary = pd.DataFrame(
        {
            ID_COL: train[ID_COL], TARGET: y, "fixed_prediction": fixed,
            "model_std": spread, "uncertain_04_06": uncertain,
            "top5pct_disagreement": disagreement, "fixed_error": fixed_error,
        }
    )
    numeric = [c for c in train.select_dtypes(include=np.number) if c != ID_COL]
    categorical = train.select_dtypes(include=["object", "string"]).columns.tolist()
    categorical.remove(TARGET)
    records = []
    for label, mask in {
        "all": np.ones(len(train), dtype=bool),
        "uncertain_04_06": uncertain,
        "top5pct_disagreement": disagreement,
        "fixed_error": fixed_error,
    }.items():
        for col in numeric:
            records.append({"group": label, "feature": col, "value": "mean", "count": int(mask.sum()), "stat": float(train.loc[mask, col].mean())})
        for col in categorical:
            rates = train.loc[mask, col].value_counts(normalize=True)
            for value, rate in rates.items():
                records.append({"group": label, "feature": col, "value": str(value), "count": int(mask.sum()), "stat": float(rate)})
    return row_summary, pd.DataFrame(records)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    base, oof, test = load_predictions()
    y = base[TARGET].to_numpy()
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    names = list(oof)

    aucs = pd.Series({name: roc_auc_score(y, oof[name]) for name in names}, name="oof_auc").sort_values(ascending=False)
    aucs.to_csv(ANALYSIS / "model_auc.csv", header=True)
    pearson, spearman = oof.corr(), oof.corr(method="spearman")
    pearson.to_csv(ANALYSIS / "pearson_correlation.csv")
    spearman.to_csv(ANALYSIS / "spearman_correlation.csv")
    save_heatmap(pearson, ANALYSIS / "pearson_heatmap.png", "OOF Pearson correlation")
    save_heatmap(spearman, ANALYSIS / "spearman_heatmap.png", "OOF Spearman correlation")

    errors, jaccard, overlap_count, exclusive = error_overlap(y, oof)
    jaccard.to_csv(ANALYSIS / "error_jaccard.csv")
    overlap_count.to_csv(ANALYSIS / "error_overlap_count.csv")
    exclusive.to_csv(ANALYSIS / "exclusive_errors.csv", index=False)

    fixed = sum(FIXED_WEIGHTS[k] * oof[k].to_numpy() for k in FIXED_WEIGHTS)
    fixed_test = sum(FIXED_WEIGHTS[k] * test[k].to_numpy() for k in FIXED_WEIGHTS)
    fixed_error = (fixed >= 0.5) != y
    complement = []
    for name in names:
        model_error = errors[name].to_numpy()
        complement.append(
            {
                "model": name,
                "fixed_wrong_model_correct": int((fixed_error & ~model_error).sum()),
                "fixed_correct_model_wrong": int((~fixed_error & model_error).sum()),
            }
        )
    pd.DataFrame(complement).to_csv(ANALYSIS / "fixed_error_complement.csv", index=False)

    # AUCの本質である正例・負例の順位ペアを固定乱数で30万組サンプルする。
    rng = np.random.default_rng(42)
    pos = rng.choice(np.flatnonzero(y == 1), size=300_000, replace=True)
    neg = rng.choice(np.flatnonzero(y == 0), size=300_000, replace=True)
    fixed_pair_correct = fixed[pos] > fixed[neg]
    pair_rows = []
    for name in names:
        candidate_correct = oof[name].to_numpy()[pos] > oof[name].to_numpy()[neg]
        pair_rows.append(
            {
                "model": name,
                "sample_pair_accuracy": float(candidate_correct.mean()),
                "fixed_wrong_pair_corrected": int((~fixed_pair_correct & candidate_correct).sum()),
                "fixed_correct_pair_harmed": int((fixed_pair_correct & ~candidate_correct).sum()),
                "net_pair_correction": int((~fixed_pair_correct & candidate_correct).sum() - (fixed_pair_correct & ~candidate_correct).sum()),
            }
        )
    pd.DataFrame(pair_rows).to_csv(ANALYSIS / "sampled_rank_pair_complement.csv", index=False)

    initial = {name: FIXED_WEIGHTS.get(name, 0.0) for name in names}
    prob_opt = greedy_nonnegative_blend(y, oof, initial, rank_mode=False)
    rank_opt = greedy_nonnegative_blend(y, oof, initial, rank_mode=True)
    ranks = normalized_ranks(oof)
    simple_existing = oof[list(EXISTING)].mean(axis=1).to_numpy()
    simple_all = oof.mean(axis=1).to_numpy()
    rank_existing = ranks[list(EXISTING)].mean(axis=1).to_numpy()
    rank_all = ranks.mean(axis=1).to_numpy()

    stack_oof, stack_test, stack_coefficients = cross_fitted_stack(y, oof, test)
    pd.DataFrame({ID_COL: base[ID_COL], TARGET: y, "prediction": stack_oof}).to_csv(OUT / "oof/stack_logistic.csv", index=False)
    pd.DataFrame({ID_COL: sample[ID_COL], "prediction": stack_test}).to_csv(OUT / "test/stack_logistic.csv", index=False)

    methods = {
        "existing_fixed_blend": (fixed, fixed_test),
        "simple_mean_existing4": (simple_existing, test[list(EXISTING)].mean(axis=1).to_numpy()),
        "simple_mean_all8": (simple_all, test.mean(axis=1).to_numpy()),
        "rank_mean_existing4": (rank_existing, normalized_ranks(test[list(EXISTING)]).mean(axis=1).to_numpy()),
        "rank_mean_all8": (rank_all, normalized_ranks(test).mean(axis=1).to_numpy()),
        "optimized_probability_rounded": (
            prob_opt["rounded_prediction"],
            sum(prob_opt["rounded_weights"].get(k, 0.0) * test[k].to_numpy() for k in test),
        ),
        "optimized_rank_rounded": (
            rank_opt["rounded_prediction"],
            sum(rank_opt["rounded_weights"].get(k, 0.0) * normalized_ranks(test)[k].to_numpy() for k in test),
        ),
        "cross_fitted_logistic_stack": (stack_oof, stack_test),
    }
    blend_rows = []
    for name, (pred, _) in methods.items():
        blend_rows.append({"experiment": name, "oof_auc": roc_auc_score(y, pred), "delta_vs_fixed": roc_auc_score(y, pred) - roc_auc_score(y, fixed)})
    blend_table = pd.DataFrame(blend_rows).sort_values("oof_auc", ascending=False)
    blend_table.to_csv(ANALYSIS / "blend_comparison.csv", index=False)

    best_name = blend_table.iloc[0]["experiment"]
    best_oof, best_test = methods[best_name]
    final_oof = pd.DataFrame({ID_COL: base[ID_COL], TARGET: y, "prediction": best_oof})
    final_submission = sample.copy()
    final_submission[TARGET] = best_test
    assert final_submission[ID_COL].equals(sample[ID_COL])
    assert np.isfinite(final_submission[TARGET]).all() and final_submission[TARGET].between(0, 1).all()
    final_oof.to_csv(OUT / "oof/final_selected.csv", index=False)
    final_submission.to_csv(OUT / "submissions/submission_diversity_v1.csv", index=False)

    boundary_rows, boundary_summary = boundary_analysis(train, y, oof, fixed)
    boundary_rows.to_csv(ANALYSIS / "boundary_rows.csv", index=False)
    boundary_summary.to_csv(ANALYSIS / "boundary_feature_summary.csv", index=False)

    metadata = {
        "fixed_weights": FIXED_WEIGHTS,
        "fixed_auc": float(roc_auc_score(y, fixed)),
        "probability_optimization": {k: v for k, v in prob_opt.items() if k not in ["prediction", "rounded_prediction"]},
        "rank_optimization": {k: v for k, v in rank_opt.items() if k not in ["prediction", "rounded_prediction"]},
        "stack_coefficients_by_fold": stack_coefficients,
        "best_experiment": best_name,
        "best_auc": float(roc_auc_score(y, best_oof)),
        "uncertain_04_06_rows": int(((fixed >= 0.4) & (fixed <= 0.6)).sum()),
        "top5pct_disagreement_rows": int((oof.std(axis=1) >= oof.std(axis=1).quantile(0.95)).sum()),
        "stacking_caveat": "Each row uses only base OOF predictions, and meta OOF is cross-fitted. Fully nested base-model retraining was not performed, so second-order fold dependence can remain.",
    }
    (ANALYSIS / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(aucs.to_string())
    print("\n", blend_table.to_string(index=False))
    print("\nprobability rounded weights", prob_opt["rounded_weights"])
    print("rank rounded weights", rank_opt["rounded_weights"])
    print("best", best_name, metadata["best_auc"])


if __name__ == "__main__":
    main()
