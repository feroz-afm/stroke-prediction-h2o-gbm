"""Train the H2O GBM stroke model from the notebook as a plain script.

Usage:
    python src/train_gbm.py --data data/healthcare-dataset-stroke-data.csv
"""
import argparse

import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator

FEATURES_NUM = ["age", "avg_glucose_level", "bmi"]
FEATURES_CAT = ["gender", "hypertension", "heart_disease", "ever_married",
                "work_type", "residence_type", "smoking_status"]


def load(path):
    df = pd.read_csv(path)
    df["bmi"] = df["bmi"].fillna(-99)          # missing BMI is itself a strong risk signal
    df = df.rename(columns={"Residence_type": "residence_type"})
    df["target"] = df.pop("stroke")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/healthcare-dataset-stroke-data.csv")
    p.add_argument("--threshold", type=float, default=0.148, help="decision threshold for class 1")
    args = p.parse_args()

    h2o.init(nthreads=-1)
    hex_df = h2o.H2OFrame(load(args.data))
    hex_df["target"] = hex_df["target"].asfactor()
    train, test = hex_df.split_frame(ratios=[0.7], seed=999)

    model = H2OGradientBoostingEstimator(
        ntrees=100, max_depth=4, min_rows=10, learn_rate=0.01, sample_rate=1,
        col_sample_rate=0.7, nfolds=5, score_each_iteration=True,
        stopping_metric="auto", stopping_rounds=10, seed=999)
    model.train(x=FEATURES_NUM + FEATURES_CAT, y="target", training_frame=train)

    print("CV AUC  :", round(model.auc(xval=True), 4))
    perf = model.model_performance(test)
    print("Test AUC:", round(perf.auc(), 4))
    print(perf.confusion_matrix(thresholds=args.threshold))
    print(model.varimp(use_pandas=True).head(10))


if __name__ == "__main__":
    main()
