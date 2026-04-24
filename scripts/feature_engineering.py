from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLEANED_PATH = DATA_DIR / "train_cleaned.csv"
FEATURES_PATH = DATA_DIR / "train_features.csv"


def extract_deck(cabin: str) -> str:
    if pd.isna(cabin) or cabin == "Unknown":
        return "Unknown"
    return str(cabin)[0]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["FamilySize"] = featured["SibSp"] + featured["Parch"] + 1
    featured["IsAlone"] = (featured["FamilySize"] == 1).astype(int)
    featured["Deck"] = featured["Cabin"].apply(extract_deck)

    featured["AgeGroup"] = pd.cut(
        featured["Age"],
        bins=[-0.1, 12, 19, 59, 120],
        labels=["Child", "Teen", "Adult", "Senior"],
    ).astype(str)

    featured["FarePerPerson"] = featured["Fare"] / featured["FamilySize"]
    featured["LogFare"] = np.log1p(featured["Fare"])
    featured["LogAge"] = np.log1p(featured["Age"])
    featured["PclassFareInteraction"] = featured["Pclass"] * featured["FarePerPerson"]
    featured["AgePclassInteraction"] = featured["Age"] * featured["Pclass"]

    categorical = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    encoded = pd.get_dummies(featured, columns=categorical, prefix=categorical, dtype=int)
    return encoded


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_PATH)
    featured = engineer_features(df)
    featured.to_csv(FEATURES_PATH, index=False)

    engineered_columns = [
        "FamilySize",
        "IsAlone",
        "Deck",
        "AgeGroup",
        "FarePerPerson",
        "LogFare",
        "LogAge",
        "PclassFareInteraction",
        "AgePclassInteraction",
    ]

    summary = f"""# Feature Engineering Summary

Created derived features:

{chr(10).join(f"- `{column}`" for column in engineered_columns)}

Categorical variables were one-hot encoded: `Sex`, `Embarked`, `Title`, `Deck`, and `AgeGroup`.

Output file: `data/train_features.csv`
"""
    (OUTPUT_DIR / "feature_engineering_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
