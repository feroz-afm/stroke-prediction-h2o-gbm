# Stroke Risk Prediction with H2O Gradient Boosting

> Predicting stroke risk from demographic and health data with an **H2O Gradient Boosting Machine**,
> 5-fold cross-validation, threshold tuning for a strongly imbalanced target, and model interpretation.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/H2O.ai-GBM-FFE600" />
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-8E44AD" />
  <img src="https://img.shields.io/badge/CV%20AUC-0.845-success" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

## Overview
According to the WHO, stroke is the **2nd leading cause of death** worldwide. This project builds a model that
estimates a patient's stroke risk from age, glucose level, BMI, hypertension, heart disease and lifestyle data.

**The challenge:** only **249 of 5,110 patients (4.87%)** had a stroke. A model that always predicts "no stroke"
is already 95% accurate, so accuracy alone is misleading. This project focuses on **AUC, recall, calibration
and a tuned decision threshold** instead.

## Dataset
[Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) —
5,110 rows, 10 predictors, target `stroke`.

| Numerical | Categorical |
|---|---|
| `age`, `avg_glucose_level`, `bmi` | `gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `residence_type`, `smoking_status` |

Full attribute table: [`data/README.md`](data/README.md)

## Pipeline
```mermaid
flowchart LR
    A[Raw CSV] --> B[Cleaning<br/>BMI missing → -99]
    B --> C[EDA<br/>histograms, mosaics, cross-tabs]
    C --> D[H2O frame<br/>70/30 split]
    D --> E[GBM + 5-fold CV]
    E --> F[Threshold tuning<br/>0.148]
    F --> G[Test evaluation<br/>& calibration]
    E --> H[Variable importance<br/>& SHAP]
```

## Key Findings from EDA
- Risk rises clearly with **age** and **average glucose level**.
- **Hypertension** and **heart disease** strongly increase stroke risk.
- **Missing BMI is a strong signal:** ~20% of patients with missing BMI had a stroke, versus ~3–5% in most BMI
  ranges — so missing values were encoded as `-99` instead of being filled with the mean.
- "Ever married" shows 6.6% stroke rate vs 1.7% — mostly an **age effect** (married patients are older).

## Model
| Parameter | Value |
|---|---|
| Algorithm | `H2OGradientBoostingEstimator` |
| Trees / max depth | 100 / 4 |
| Learning rate | 0.01 |
| Min rows per leaf | 10 |
| Column sample rate | 0.7 |
| Cross-validation | 5-fold |
| Early stopping | 10 rounds |
| Seed | 999 |
| Decision threshold | **0.148** (tuned so predicted positives ≈ actual positives) |

## Results
| Metric | Value |
|---|---:|
| Cross-validation AUC | **0.845 ± 0.016** |
| Accuracy @ 0.148 — train / CV / test | 0.940 / 0.921 / **0.932** |
| Calibration (train) | 184.4 predicted vs 184 actual strokes |

**Test set confusion matrix** (threshold 0.148, 1,520 patients):

| | Pred: no stroke | Pred: stroke |
|---|---:|---:|
| **Actual: no stroke** | 1401 | 54 |
| **Actual: stroke** | 50 | 15 |

More metrics: [`results/metrics.md`](results/metrics.md)

## Figures
| Training scoring history | Cross-validation metrics |
|---|---|
| ![](results/figures/01_training_scoring_history.png) | ![](results/figures/02_cross_validation_metrics.png) |

| Training ROC | Test ROC |
|---|---|
| ![](results/figures/07_training_roc.png) | ![](results/figures/10_test_roc.png) |

| Confusion matrix | Predictions on test set |
|---|---|
| ![](results/figures/08_confusion_matrix.png) | ![](results/figures/11_predictions_test.png) |
