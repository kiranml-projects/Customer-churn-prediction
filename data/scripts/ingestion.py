"""
ingestion.py — Stage 1: download the IBM Telco churn dataset, validate it,
and save it to data/raw/telco_churn.csv.

The file name is fixed on purpose: DVC versions the file by its hash
(dvc.lock), so history is kept without timestamped copies.
"""

import os
from io import StringIO

import pandas as pd
import requests

from common import RAW_FILE, get_logger, load_params

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]


def fetch_data(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def validate_raw(df: pd.DataFrame) -> None:
    """Fail fast on a bad upstream file instead of training on it."""
    if df.empty:
        raise ValueError("Validation failed: DataFrame is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: missing required columns: {missing}")

    unexpected = set(df["Churn"].dropna().unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"Validation failed: unexpected Churn values: {unexpected}")

    if df["customerID"].duplicated().any():
        raise ValueError("Validation failed: duplicate customerID values")


def save_raw(df: pd.DataFrame, path: str = RAW_FILE) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved raw data to {path} ({len(df)} rows)")
    return path


def main() -> str:
    params = load_params()
    logger.info("Starting ingestion run")
    df = fetch_data(params["data"]["url"])
    validate_raw(df)
    return save_raw(df)


if __name__ == "__main__":
    main()
