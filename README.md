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
