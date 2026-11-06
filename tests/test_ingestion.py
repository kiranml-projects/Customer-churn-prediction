import pytest

from ingestion import validate_raw


def test_valid_data_passes(raw_sample):
    validate_raw(raw_sample)


def test_empty_data_fails(raw_sample):
    with pytest.raises(ValueError, match="empty"):
        validate_raw(raw_sample.iloc[0:0])


def test_missing_column_fails(raw_sample):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw(raw_sample.drop(columns=["Churn"]))


def test_bad_target_values_fail(raw_sample):
    raw_sample.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="unexpected Churn values"):
        validate_raw(raw_sample)


def test_duplicate_ids_fail(raw_sample):
    raw_sample.loc[1, "customerID"] = raw_sample.loc[0, "customerID"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw(raw_sample)
