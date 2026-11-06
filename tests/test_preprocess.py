import os

import pandas as pd
import pytest

from common import NUMERIC_COLS, PROCESSED_DIR
from preprocess import clean, encode_features, scale, split


def test_clean_fixes_total_charges_and_target(raw_sample):
    df = clean(raw_sample)
    assert "customerID" not in df.columns
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df["Churn"].unique()) <= {0, 1}


def test_encoded_features_are_numeric(raw_sample):
    df = encode_features(clean(raw_sample))
    assert all(dtype.kind in "iuf" for dtype in df.dtypes)


def test_split_is_stratified_and_complete(raw_sample):
    df = encode_features(clean(raw_sample))
    s = split(df, test_size=0.3, val_share=0.5, random_state=42)
    total = len(s["X_train"]) + len(s["X_val"]) + len(s["X_test"])
    assert total == len(df)
    assert abs(s["y_train"].mean() - df["Churn"].mean()) < 0.05


def test_scaler_is_fit_on_train_only(raw_sample):
    df = encode_features(clean(raw_sample))
    s = split(df, test_size=0.3, val_share=0.5, random_state=42)
    scaler = scale(s)
    # After scaling, TRAIN numeric columns have mean ~0; val/test need not
    assert s["X_train"][NUMERIC_COLS].mean().abs().max() < 1e-6
    assert scaler.n_samples_seen_ == len(s["X_train"])


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROCESSED_DIR, "X_train.csv")), reason="run `dvc repro` first"
)
def test_training_columns_match_api_contract():
    """Guards against training/serving skew: same features, same order."""
    from serving.model_server import FEATURE_COLS

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), nrows=1)
    assert list(X_train.columns) == FEATURE_COLS
