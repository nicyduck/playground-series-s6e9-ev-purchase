"""入力特徴とSHAP寄与のVIFを計算し、情報重複を診断する。"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from train_lookup_lightgbm import add_lookup_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
SHAP_DIR = ROOT / "experiments/shap_feature_reduction_v1"
OUT = ROOT / "experiments/vif_analysis_v1"
TARGET, ID_COL = "Will_Buy_EV", "id"


def vif_from_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """標準化行列の相関行列逆行列からVIFを計算する。"""
    numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.nunique(dropna=False) > 1].fillna(numeric.median())
    z = (numeric - numeric.mean()) / numeric.std(ddof=0)
    corr = np.corrcoef(z.to_numpy(), rowvar=False)
    inverse = np.linalg.pinv(corr, rcond=1e-10)
    vif = pd.DataFrame({"feature": numeric.columns, "vif": np.diag(inverse)})
    vif["interpretation"] = pd.cut(
        vif["vif"], [-np.inf, 2, 5, 10, np.inf],
        labels=["low", "moderate", "high", "very_high"], right=False,
    ).astype(str)
    corr_df = pd.DataFrame(corr, index=numeric.columns, columns=numeric.columns)
    return vif.sort_values("vif", ascending=False), corr_df


def strongest_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    rows = []
    for i, j in zip(*np.where(mask)):
        rows.append((corr.index[i], corr.columns[j], corr.iat[i, j], abs(corr.iat[i, j])))
    return pd.DataFrame(rows, columns=["feature_1", "feature_2", "correlation", "abs_correlation"]).sort_values(
        "abs_correlation", ascending=False
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    features = [c for c in train.columns if c not in [ID_COL, TARGET]]
    y = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")

    # 計算量を固定しつつ分布を保つ代表サンプル。lookup統計自体は全trainから作る。
    sample = train.sample(n=100_000, random_state=42)
    design = pd.get_dummies(sample[features], drop_first=True, dtype=np.float32)
    lookup_design = add_lookup_features(train[features], y, [design])[0]
    input_vif, input_corr = vif_from_frame(lookup_design)
    input_vif.to_csv(OUT / "input_feature_vif.csv", index=False)
    strongest_pairs(input_corr).to_csv(OUT / "input_feature_correlation_pairs.csv", index=False)

    shap = pd.read_csv(SHAP_DIR / "sampled_shap_family_contributions.csv")
    shap = shap.drop(columns=["fold", "row_id"])
    shap_vif, shap_corr = vif_from_frame(shap)
    shap_vif.to_csv(OUT / "shap_contribution_vif.csv", index=False)
    strongest_pairs(shap_corr).to_csv(OUT / "shap_contribution_correlation_pairs.csv", index=False)

    metadata = {
        "input_rows": len(sample), "shap_rows": len(shap),
        "vif_definition": "diagonal of pseudo-inverse correlation matrix",
        "caution": "VIF measures linear redundancy; it is not a drop-feature score for tree models.",
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("INPUT VIF\n", input_vif.to_string(index=False))
    print("\nSHAP CONTRIBUTION VIF\n", shap_vif.to_string(index=False))
    print("\nTOP SHAP CORRELATIONS\n", strongest_pairs(shap_corr).head(15).to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
