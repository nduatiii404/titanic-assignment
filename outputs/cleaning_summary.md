# Data Cleaning Summary

## Missing Values Before Cleaning

| column | missing |
| --- | ---: |
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

## Missing Values After Cleaning

No missing values remain in the cleaned training data.

## Decisions

- Removed duplicate rows: 0
- Standardized `Sex` to lowercase and `Embarked` to uppercase.
- Extracted `Title` from passenger names to support age imputation.
- Added `AgeMissing` and `CabinMissing` indicator columns.
- Imputed `Age` using median values grouped by `Pclass`, `Sex`, and `Title`, then used global median for any remaining gaps.
- Imputed `Embarked` with the mode.
- Filled missing `Cabin` with `Unknown`.
- Capped `Age` using IQR bounds: -2.62 to 60.38.
- Capped `Fare` using IQR bounds: -26.72 to 65.63.
