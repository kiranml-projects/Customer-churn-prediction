"""
evaluate.py — Stage 4: score the model on the held-out TEST set, save
plots and a metrics report (tracked by DVC), log them to MLflow, and
apply the promotion gate (test ROC-AUC >= evaluate.min_roc_auc).
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed (Airflow, CI, Docker)
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from common import (  # noqa: E402
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    get_logger,
    load_params,
)

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "test_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "test_precision": round(precision_score(y_true, y_pred), 4),
        "test_recall": round(recall_score(y_true, y_pred), 4),
        "test_f1": round(f1_score(y_true, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def passes_gate(metrics: dict, min_roc_auc: float) -> bool:
    return bool(metrics["test_roc_auc"] >= min_roc_auc)


def save_plots(model, X_test, y_test, y_pred, y_proba) -> list:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Confusion Matrix — Test Set")
    paths.append(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend()
    paths.append(os.path.join(REPORTS_DIR, "roc_curve.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    top = pd.Series(model.feature_importances_, index=X_test.columns).nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top 15 Feature Importances")
    paths.append(os.path.join(REPORTS_DIR, "feature_importance.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    return paths


def main() -> dict:
    min_roc_auc = load_params()["evaluate"]["min_roc_auc"]

    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    promoted = passes_gate(metrics, min_roc_auc)
    logger.info(f"Test metrics: {metrics}")

    plot_paths = save_plots(model, X_test, y_test, y_pred, y_proba)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump({**metrics, "promoted": promoted, "min_roc_auc": min_roc_auc}, f, indent=2)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_param("promoted", promoted)
        mlflow.log_artifact(report_path)
        for path in plot_paths:
            mlflow.log_artifact(path)

    if promoted:
        logger.info(f"Model PROMOTED: test ROC-AUC {metrics['test_roc_auc']} >= {min_roc_auc}")
    else:
        logger.warning(f"Model NOT promoted: test ROC-AUC {metrics['test_roc_auc']} < {min_roc_auc}")
    return metrics


if __name__ == "__main__":
    main()
