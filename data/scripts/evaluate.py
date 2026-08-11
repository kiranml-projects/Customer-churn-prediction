"""
evaluate.py — Loads the trained model and test set, produces final
evaluation metrics and saves a report. This is the fourth task in
our Airflow DAG.
"""

import glob
import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "models")
REPORTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "reports")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "customer-churn-prediction"

# Threshold for model promotion — ROC-AUC must exceed this
MIN_ROC_AUC = 0.80


def load_test_data() -> tuple:
    def latest(pattern):
        files = glob.glob(os.path.join(PROCESSED_DIR, pattern))
        if not files:
            raise FileNotFoundError(f"No files found: {pattern}")
        return max(files, key=os.path.getctime)

    X_test = pd.read_csv(latest("X_test_*.csv"))
    y_test = pd.read_csv(latest("y_test_*.csv")).squeeze()
    logger.info(f"Test set: {X_test.shape}")
    return X_test, y_test


def load_model():
    path = os.path.join(MODELS_DIR, "churn_model.pkl")
    model = joblib.load(path)
    logger.info(f"Loaded model from {path}")
    return model


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1":       round(f1_score(y_test, y_pred), 4),
        "test_roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
    }

    logger.info(f"Test metrics: {metrics}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return metrics, y_pred, y_proba


def save_plots(model, X_test, y_test, y_pred, y_proba) -> list:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    paths = []

    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
    ax.set_title("Confusion Matrix — Test Set")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close()
    paths.append(cm_path)
    logger.info(f"Saved confusion matrix: {cm_path}")

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend()
    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close()
    paths.append(roc_path)
    logger.info(f"Saved ROC curve: {roc_path}")

    # 3. Feature importance
    feat_imp = pd.Series(
        model.feature_importances_, index=X_test.columns
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Feature Importances")
    ax.invert_yaxis()
    fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    fig.savefig(fi_path, bbox_inches="tight")
    plt.close()
    paths.append(fi_path)
    logger.info(f"Saved feature importance: {fi_path}")

    return paths


def save_report(metrics: dict) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved report: {path}")
    return path


def promote_model(metrics: dict) -> bool:
    """
    Simple gate — if ROC-AUC exceeds threshold, model is
    ready for serving. In production this would trigger
    MLflow model registry stage transition.
    """
    if metrics["test_roc_auc"] >= MIN_ROC_AUC:
        logger.info(f"✅ Model promoted — ROC-AUC {metrics['test_roc_auc']} >= {MIN_ROC_AUC}")
        return True
    else:
        logger.warning(f"❌ Model NOT promoted — ROC-AUC {metrics['test_roc_auc']} < {MIN_ROC_AUC}")
        return False


def main():
    logger.info("Starting evaluation run")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_test, y_test = load_test_data()
    model = load_model()

    metrics, y_pred, y_proba = compute_metrics(model, X_test, y_test)

    plot_paths = save_plots(model, X_test, y_test, y_pred, y_proba)
    report_path = save_report(metrics)

    # Log to MLflow
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(report_path)
        for path in plot_paths:
            mlflow.log_artifact(path)

    promoted = promote_model(metrics)
    logger.info(f"Evaluation complete. Promoted: {promoted}")

    return metrics


if __name__ == "__main__":
    main()