"""
ingest.py — Downloads the Telco Customer Churn dataset and saves it
to data/raw/ with a timestamped filename (so we never overwrite history —
important for reproducibility and for Airflow re-runs).

This will later become the first task in our Airflow DAG.
"""

import logging
import os
from datetime import datetime

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Public copy of the IBM Telco Customer Churn dataset
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")


def fetch_data(url: str) -> pd.DataFrame:
    response = requests.get(url)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    return df


def validate_raw(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Validation failed: DataFrame is empty")
    
    required_columns = ["customerID" , "Churn"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: missing required columns: {missing}")
    


def save_raw(df: pd.DataFrame) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"telco_churn_{timestamp}.csv"
    path = os.path.join(RAW_DIR, filename)

    df.to_csv(path, index=False)

    logger.info(f"Saved raw data to {path} ({len(df)} rows)")

    return path


def main():
    logger.info("Starting ingestion run")
    df = fetch_data(DATA_URL)
    validate_raw(df)
    path = save_raw(df)
    logger.info(f"Ingestion complete: {path}")
    return path


if __name__ == "__main__":
    main()