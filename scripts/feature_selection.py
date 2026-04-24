from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = DATA_DIR / "train_features.csv"


def numeric_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["Survived"]
    excluded = {"PassengerId", "Survived"}
    numeric = df.select_dtypes(include=["number"]).drop(columns=list(excluded), errors="ignore")
    return numeric.fillna(0), y


def highly_correlated_features(x: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    corr = x.corr().abs()
    upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    upper = corr.where(upper_mask)
    return [column for column in upper.columns if any(upper[column] > threshold)]


def pearson_importance(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    scores = x.apply(lambda col: abs(col.corr(y))).fillna(0)
    return (
        scores.sort_values(ascending=False)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
        .assign(method="absolute_pearson_correlation")
    )


def series_to_markdown(values: list[str]) -> str:
    rows = ["| feature |", "| --- |"]
    rows.extend(f"| {value} |" for value in values)
    return "\n".join(rows)


def frame_to_markdown(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in df.columns) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(rows)


def random_forest_importance(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame | None:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return None

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x, y)
    return (
        pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .assign(method="random_forest")
        .reset_index(drop=True)
    )


def select_features(importance: pd.DataFrame, redundant: list[str], top_n: int = 12) -> pd.DataFrame:
    selected = importance[~importance["feature"].isin(redundant)].head(top_n).copy()
    selected["decision"] = "keep"
    selected["reason"] = "High survival signal and not removed as a redundant feature."
    return selected


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    x, y = numeric_model_frame(df)

    redundant = highly_correlated_features(x)
    importance = random_forest_importance(x, y)
    method_note = "Random Forest feature importance"
    if importance is None:
        importance = pearson_importance(x, y)
        method_note = "Absolute Pearson correlation because scikit-learn is not installed"

    selected = select_features(importance, redundant)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

    summary = f"""# Feature Selection Summary

Method used: {method_note}

Highly correlated/redundant features removed before final selection:

{series_to_markdown(redundant if redundant else ["None"])}

Selected features:

{frame_to_markdown(selected[["feature", "importance", "reason"]])}

Output files:

- `outputs/feature_importance.csv`
- `outputs/selected_features.csv`
"""
    (OUTPUT_DIR / "feature_selection_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
