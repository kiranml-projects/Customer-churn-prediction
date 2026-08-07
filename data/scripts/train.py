"""
train.py — Loads the latest processed splits, trains a RandomForest
classifier, tracks everything with MLflow, and registers the best model.

This will become the third task in our Airflow DAG.
"""

import glob
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "customer-churn-prediction"

# Hyperparameters
PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "class_weight": "balanced",  # handles class imbalance
}


def load_latest_splits() -> tuple:
    """Load the most recent train/val splits from data/processed/"""

    def latest(pattern):
        files = glob.glob(os.path.join(PROCESSED_DIR, pattern))
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        return max(files, key=os.path.getctime)

    X_train = pd.read_csv(latest("X_train_*.csv"))
    y_train = pd.read_csv(latest("y_train_*.csv")).squeeze()
    X_val   = pd.read_csv(latest("X_val_*.csv"))
    y_val   = pd.read_csv(latest("y_val_*.csv")).squeeze()

    logger.info(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def train(X_train, y_train) -> RandomForestClassifier:
    """Train RandomForest with predefined hyperparameters"""
    model = RandomForestClassifier(**PARAMS)
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def evaluate(model, X_val, y_val) -> dict:
    """Evaluate model on validation set — return metrics dict"""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_val, y_pred),
        "f1_score":  f1_score(y_val, y_pred),
        "roc_auc":   roc_auc_score(y_val, y_proba),
    }

    logger.info(f"Validation metrics: {metrics}")
    logger.info(f"\n{classification_report(y_val, y_pred)}")
    return metrics


def save_model(model) -> str:
    """Save model locally with joblib"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "churn_model.pkl")
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def main():
    logger.info("Starting training run")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    X_train, y_train, X_val, y_val = load_latest_splits()

    # Train + evaluate + log with MLflow
    with mlflow.start_run():

        # Log hyperparameters
        mlflow.log_params(PARAMS)

        # Train
        model = train(X_train, y_train)

        # Evaluate
        metrics = evaluate(model, X_val, y_val)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn-rf-model",
        )

        # Save locally too
        save_model(model)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")
        logger.info(f"Training complete. ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


if __name__ == "__main__":
    main()