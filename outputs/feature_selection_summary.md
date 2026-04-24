# Feature Selection Summary

Method used: Random Forest feature importance

Highly correlated/redundant features removed before final selection:

| feature |
| --- |
| Sex_male |
| Deck_Unknown |

Selected features:

| feature | importance | reason |
| --- | --- | --- |
| Title_Mr | 0.09563011918302446 | High survival signal and not removed as a redundant feature. |
| AgePclassInteraction | 0.09017076078653154 | High survival signal and not removed as a redundant feature. |
| Sex_female | 0.08006544958715119 | High survival signal and not removed as a redundant feature. |
| FarePerPerson | 0.07469297972747971 | High survival signal and not removed as a redundant feature. |
| LogFare | 0.06809186788715174 | High survival signal and not removed as a redundant feature. |
| Age | 0.06772043027479664 | High survival signal and not removed as a redundant feature. |
| Fare | 0.06707869125144196 | High survival signal and not removed as a redundant feature. |
| LogAge | 0.06271370215436955 | High survival signal and not removed as a redundant feature. |
| PclassFareInteraction | 0.06162005581567663 | High survival signal and not removed as a redundant feature. |
| Pclass | 0.03160334132615832 | High survival signal and not removed as a redundant feature. |
| FamilySize | 0.03130354128214305 | High survival signal and not removed as a redundant feature. |
| SibSp | 0.0227133496691763 | High survival signal and not removed as a redundant feature. |

Output files:

- `outputs/feature_importance.csv`
- `outputs/selected_features.csv`
