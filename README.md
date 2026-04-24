# Titanic Feature Engineering and Selection

This repository contains the solution for Artificial Intelligence Assignment 2. The project cleans the Titanic dataset, engineers predictive features, and selects useful variables for survival modeling.

## Dataset

Source: [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

Local files:

```text
data/train.csv
data/test.csv
```

## Project Structure

```text
titanic-assignment/
  data/
    train.csv
    test.csv
    train_cleaned.csv
    train_features.csv
  notebooks/
    Titanic_Feature_Engineering.ipynb
  outputs/
    cleaning_summary.md
    feature_engineering_summary.md
    feature_importance.csv
    feature_selection_summary.md
    selected_features.csv
  scripts/
    data_cleaning.py
    feature_engineering.py
    feature_selection.py
  README.md
  requirements.txt
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```

Open the notebook:

```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

## Cleaning Decisions

- `Age` is imputed with grouped medians using passenger class, sex, and extracted title.
- `Embarked` is imputed with the mode.
- `Fare` is imputed with the median when needed.
- `Cabin` is retained as a signal through `CabinMissing` and later converted into `Deck`.
- `Age` and `Fare` are capped with the IQR method to reduce outlier influence.
- Duplicate rows are removed.

## Features Engineered

- `FamilySize = SibSp + Parch + 1`
- `IsAlone`
- `Title` extracted from `Name`
- `Deck` extracted from `Cabin`
- `AgeGroup`
- `FarePerPerson`
- `LogFare`
- `LogAge`
- `PclassFareInteraction`
- `AgePclassInteraction`
- One-hot encoded categorical variables

## Key Observations

Survival is strongly associated with sex, passenger class, fare-related features, title, and family composition. Cabin availability is also informative because missing cabin values are not random; they tend to reflect passenger class and ticket context.
