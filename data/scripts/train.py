"""
train.py — Stage 3: train a RandomForest with hyperparameters from
params.yaml, log params/metrics/model to MLflow, and register a new
version of the model in the MLflow Model Registry.
"""

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from common import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DIR,
    REGISTERED_MODEL_NAME,
    get_logger,
    load_params,
)

logger = get_logger(__name__)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))


def train(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> RandomForestClassifier:
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def score(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "val_accuracy": accuracy_score(y, y_pred),
        "val_f1": f1_score(y, y_pred),
        "val_recall": recall_score(y, y_pred),
        "val_roc_auc": roc_auc_score(y, y_proba),
    }


def main() -> dict:
    params = load_params()["train"]
    X_train, y_train = load_split("X_train"), load_split("y_train").squeeze()
    X_val, y_val = load_split("X_val"), load_split("y_val").squeeze()

    model = train(X_train, y_train, params)
    metrics = score(model, X_val, y_val)
    logger.info(f"Validation metrics: {metrics}")

    # Save the local artifact first: DVC and the API depend on it
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_val.head(3).astype(float),
        )
        logger.info(f"MLflow run_id: {run.info.run_id}")

    return metrics


if __name__ == "__main__":
    main()
