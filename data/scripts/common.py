"""Shared paths and helpers for the pipeline stages."""

import logging
import os

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "telco_churn.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PARAMS_FILE = os.path.join(PROJECT_ROOT, "params.yaml")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"

# MLflow: local SQLite by default; set MLFLOW_TRACKING_URI to use a server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
EXPERIMENT_NAME = "customer-churn-prediction"
REGISTERED_MODEL_NAME = "churn-rf-model"


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(name)
