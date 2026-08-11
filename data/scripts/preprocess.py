"""
preprocess.py — Loads the latest raw Telco churn CSV, cleans it,
engineers features, and saves train/val/test splits to data/processed/.

This will become the second task in our Airflow DAG.
"""

import glob
import logging
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_latest_raw() -> pd.DataFrame:
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}")
    latest = max(files, key=os.path.getctime)
    logger.info(f"Loading latest raw file: {latest}")
    df = pd.read_csv(latest)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Fix TotalCharges — contains spaces, should be float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # 2. Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    # 3. Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logger.info(f"Churn distribution:\n{df['Churn'].value_counts()}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Separate categorical columns (everything except numeric and target)
    cat_cols = [
        col for col in df.columns
        if col not in NUMERIC_COLS + ["Churn"]
    ]

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    logger.info(f"After one-hot encoding: {df.shape[1]} columns")

    return df


def split_and_save(df: pd.DataFrame) -> dict:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Split temp: 50/50 → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Fit scaler on TRAIN only — prevent data leakage
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # Save scaler for inference
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save splits with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    splits = {
        "X_train": (X_train, f"X_train_{timestamp}.csv"),
        "y_train": (y_train, f"y_train_{timestamp}.csv"),
        "X_val":   (X_val,   f"X_val_{timestamp}.csv"),
        "y_val":   (y_val,   f"y_val_{timestamp}.csv"),
        "X_test":  (X_test,  f"X_test_{timestamp}.csv"),
        "y_test":  (y_test,  f"y_test_{timestamp}.csv"),
    }

    paths = {}
    for key, (data, filename) in splits.items():
        path = os.path.join(PROCESSED_DIR, filename)
        data.to_csv(path, index=False)
        paths[key] = path
        logger.info(f"Saved {key}: {path}")

    return paths


def main():
    logger.info("Starting preprocessing run")

    df = load_latest_raw()
    logger.info(f"Loaded raw data: {df.shape}")

    df = clean(df)
    logger.info(f"After cleaning: {df.shape}")

    df = encode_features(df)
    logger.info(f"After encoding: {df.shape}")

    paths = split_and_save(df)
    logger.info(f"Preprocessing complete. Splits saved: {paths}")

    return paths


if __name__ == "__main__":
    main()