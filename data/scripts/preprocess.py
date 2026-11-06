"""
preprocess.py — Stage 2: clean the raw data, one-hot encode categoricals,
split 70/15/15 (stratified), and scale numeric columns.

Outputs:
  data/processed/{X,y}_{train,val,test}.csv
  models/scaler.pkl   (fit on TRAIN only, reused by the API)
"""

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    MODELS_DIR,
    NUMERIC_COLS,
    PROCESSED_DIR,
    RAW_FILE,
    TARGET,
    get_logger,
    load_params,
)

logger = get_logger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is blank for brand-new customers (tenure 0) -> 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # customerID is an identifier, not a feature
    df = df.drop(columns=["customerID"])
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    logger.info(f"Churn rate: {df[TARGET].mean():.3f}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if c not in NUMERIC_COLS + [TARGET]]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    logger.info(f"After one-hot encoding: {df.shape[1] - 1} features")
    return df


def split(df: pd.DataFrame, test_size: float, val_share: float, random_state: int) -> dict:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_share, random_state=random_state, stratify=y_temp
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def scale(splits: dict) -> StandardScaler:
    """Fit on TRAIN only, then transform val/test -> no data leakage."""
    scaler = StandardScaler()
    splits["X_train"][NUMERIC_COLS] = scaler.fit_transform(splits["X_train"][NUMERIC_COLS])
    for key in ("X_val", "X_test"):
        splits[key][NUMERIC_COLS] = scaler.transform(splits[key][NUMERIC_COLS])
    return scaler


def save(splits: dict, scaler: StandardScaler) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, data in splits.items():
        data.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    logger.info(f"Saved splits to {PROCESSED_DIR} and scaler to {MODELS_DIR}")


def main() -> None:
    p = load_params()["data"]
    df = pd.read_csv(RAW_FILE)
    df = encode_features(clean(df))
    splits = split(df, p["test_size"], p["val_share"], p["random_state"])
    scaler = scale(splits)
    save(splits, scaler)


if __name__ == "__main__":
    main()
