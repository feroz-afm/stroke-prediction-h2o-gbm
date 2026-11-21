# Results summary

Model: H2O GBM (100 trees, depth 4, learn rate 0.01, 5-fold CV, seed 999), 70/30 split.

## 5-fold cross-validation (training set)
| Metric | Mean | SD |
|---|---:|---:|
| AUC | 0.845 | 0.016 |
| Recall (max-F1 threshold) | 0.572 | 0.070 |
| Precision (max-F1 threshold) | 0.244 | 0.048 |
| Specificity | 0.900 | 0.030 |
| MCC | 0.320 | 0.046 |

## Accuracy at the manual threshold 0.148
| Set | Accuracy |
|---|---:|
| Training | 0.940 |
| Cross-validation | 0.921 |
| Test | 0.932 |

## Test confusion matrix (threshold 0.148, 1,520 patients)
| | Pred 0 | Pred 1 |
|---|---:|---:|
| **Actual 0** | 1401 | 54 |
| **Actual 1** | 50 | 15 |

## Calibration
- Training: predicted 184.4 strokes vs 184 actual
- Test, top-20 highest risk: 6.69 predicted vs 5 actual; lowest 20: 0.42 vs 0
