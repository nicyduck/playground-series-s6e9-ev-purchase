from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "eda_summary.md"
TARGET = "Will_Buy_EV"
ID_COL = "id"


def markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
    table = frame.reset_index() if index else frame.copy()
    table = table.astype(str)
    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in table.to_numpy()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in test.columns if c != ID_COL]
    categorical = train[features].select_dtypes(include=["object", "string"]).columns.tolist()
    numerical = [c for c in features if c not in categorical]

    lines = [
        "# EDA summary",
        "",
        f"- train shape: `{train.shape}`",
        f"- test shape: `{test.shape}`",
        f"- target: `{TARGET}`",
        f"- train missing cells: `{int(train.isna().sum().sum())}`",
        f"- test missing cells: `{int(test.isna().sum().sum())}`",
        f"- exact train duplicates (including id): `{int(train.duplicated().sum())}`",
        f"- exact test duplicates (including id): `{int(test.duplicated().sum())}`",
        f"- feature duplicates in train (excluding id/target): `{int(train.duplicated(features).sum())}`",
        f"- feature duplicates in test (excluding id): `{int(test.duplicated(features).sum())}`",
        "",
        "## Target distribution",
        "",
        train[TARGET].value_counts(dropna=False).to_frame("count").assign(
            ratio=lambda x: x["count"] / len(train)
        ).pipe(markdown_table),
        "",
        "## Column schema",
        "",
        pd.DataFrame({
            "dtype": train.dtypes.astype(str),
            "nunique": train.nunique(dropna=False),
            "missing": train.isna().sum(),
        }).pipe(markdown_table),
        "",
        "## Categorical train/test comparison",
        "",
    ]

    cat_rows = []
    for col in categorical:
        train_values = set(train[col].dropna().astype(str))
        test_values = set(test[col].dropna().astype(str))
        p = train[col].value_counts(normalize=True)
        q = test[col].value_counts(normalize=True)
        levels = p.index.union(q.index)
        tv = 0.5 * np.abs(p.reindex(levels, fill_value=0) - q.reindex(levels, fill_value=0)).sum()
        cat_rows.append({
            "column": col,
            "train_nunique": len(train_values),
            "test_nunique": len(test_values),
            "test_only_levels": sorted(test_values - train_values),
            "total_variation": tv,
        })
    lines += [markdown_table(pd.DataFrame(cat_rows), index=False), "", "## Numeric train/test comparison", ""]

    num_rows = []
    for col in numerical:
        stat, pvalue = ks_2samp(train[col], test[col])
        num_rows.append({
            "column": col,
            "train_mean": train[col].mean(),
            "test_mean": test[col].mean(),
            "train_std": train[col].std(),
            "test_std": test[col].std(),
            "ks_stat": stat,
            "ks_pvalue": pvalue,
        })
    lines.append(markdown_table(pd.DataFrame(num_rows).round(8), index=False))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
