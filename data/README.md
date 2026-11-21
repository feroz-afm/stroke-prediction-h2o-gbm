# Data

**Source:** [Stroke Prediction Dataset — Kaggle (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Download `healthcare-dataset-stroke-data.csv` and place it in this folder. The CSV is not committed to Git.

- 5,110 patients, 12 columns
- Target `stroke`: 249 positive (4.87%) vs 4,861 negative — strongly imbalanced
- `bmi` has 201 missing values

| Attribute | Description |
|---|---|
| `id` | Unique identifier |
| `gender` | Male, Female or Other |
| `age` | Age of the patient |
| `hypertension` | 1 if the patient has hypertension, else 0 |
| `heart_disease` | 1 if the patient has heart disease, else 0 |
| `ever_married` | Yes / No |
| `work_type` | children, Govt_job, Never_worked, Private, Self-employed |
| `Residence_type` | Rural / Urban |
| `avg_glucose_level` | Average blood glucose level |
| `bmi` | Body mass index |
| `smoking_status` | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | Target: 1 if the patient had a stroke, else 0 |
