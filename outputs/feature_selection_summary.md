# Feature Selection Summary

Method used: Absolute Pearson correlation because scikit-learn is not installed

Highly correlated/redundant features removed before final selection:

| feature |
| --- |
| Sex_male |
| Deck_Unknown |

Selected features:

| feature | importance | reason |
| --- | --- | --- |
| Title_Mr | 0.5491991849030071 | High survival signal and not removed as a redundant feature. |
| Sex_female | 0.5433513806577546 | High survival signal and not removed as a redundant feature. |
| Title_Mrs | 0.3419937262857555 | High survival signal and not removed as a redundant feature. |
| Pclass | 0.3384810359610148 | High survival signal and not removed as a redundant feature. |
| AgePclassInteraction | 0.3363842095887332 | High survival signal and not removed as a redundant feature. |
| Title_Miss | 0.3356355207687264 | High survival signal and not removed as a redundant feature. |
| LogFare | 0.32190155523884745 | High survival signal and not removed as a redundant feature. |
| Fare | 0.31743039988773425 | High survival signal and not removed as a redundant feature. |
| CabinMissing | 0.31691152311229576 | High survival signal and not removed as a redundant feature. |
| FarePerPerson | 0.27039340936826484 | High survival signal and not removed as a redundant feature. |
| IsAlone | 0.203367085699892 | High survival signal and not removed as a redundant feature. |
| Deck_B | 0.1750950336504758 | High survival signal and not removed as a redundant feature. |

Output files:

- `outputs/feature_importance.csv`
- `outputs/selected_features.csv`
