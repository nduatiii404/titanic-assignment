from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAIN_PATH = DATA_DIR / "train.csv"
CLEANED_PATH = DATA_DIR / "train_cleaned.csv"


TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}


def extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    return TITLE_MAP.get(title, title)


def cap_iqr(series: pd.Series) -> tuple[pd.Series, float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper), float(lower), float(upper)


def series_to_markdown(series: pd.Series, value_name: str = "value") -> str:
    rows = [f"| {series.index.name or 'column'} | {value_name} |", "| --- | ---: |"]
    rows.extend(f"| {idx} | {value} |" for idx, value in series.items())
    return "\n".join(rows)


def clean_train(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    missing_before = df.isna().sum()
    duplicates_before = int(df.duplicated().sum())

    cleaned = df.drop_duplicates().copy()
    cleaned["Sex"] = cleaned["Sex"].str.strip().str.lower()
    cleaned["Embarked"] = cleaned["Embarked"].str.strip().str.upper()
    cleaned["Title"] = cleaned["Name"].apply(extract_title)
    cleaned["AgeMissing"] = cleaned["Age"].isna().astype(int)
    cleaned["CabinMissing"] = cleaned["Cabin"].isna().astype(int)

    grouped_age = cleaned.groupby(["Pclass", "Sex", "Title"])["Age"].transform("median")
    cleaned["Age"] = cleaned["Age"].fillna(grouped_age)
    cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())

    cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode().iloc[0])
    cleaned["Fare"] = cleaned["Fare"].fillna(cleaned["Fare"].median())
    cleaned["Cabin"] = cleaned["Cabin"].fillna("Unknown")

    cleaned["Age"], age_lower, age_upper = cap_iqr(cleaned["Age"])
    cleaned["Fare"], fare_lower, fare_upper = cap_iqr(cleaned["Fare"])

    missing_after = cleaned.isna().sum()

    summary = f"""# Data Cleaning Summary

## Missing Values Before Cleaning

{series_to_markdown(missing_before[missing_before > 0], "missing")}

## Missing Values After Cleaning

{series_to_markdown(missing_after[missing_after > 0], "missing") if (missing_after > 0).any() else "No missing values remain in the cleaned training data."}

## Decisions

- Removed duplicate rows: {duplicates_before}
- Standardized `Sex` to lowercase and `Embarked` to uppercase.
- Extracted `Title` from passenger names to support age imputation.
- Added `AgeMissing` and `CabinMissing` indicator columns.
- Imputed `Age` using median values grouped by `Pclass`, `Sex`, and `Title`, then used global median for any remaining gaps.
- Imputed `Embarked` with the mode.
- Filled missing `Cabin` with `Unknown`.
- Capped `Age` using IQR bounds: {age_lower:.2f} to {age_upper:.2f}.
- Capped `Fare` using IQR bounds: {fare_lower:.2f} to {fare_upper:.2f}.
"""

    return cleaned, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH)
    cleaned, summary = clean_train(df)
    cleaned.to_csv(CLEANED_PATH, index=False)
    (OUTPUT_DIR / "cleaning_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
