import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Start the API against a tiny model so tests don't need the real one."""
    from serving import model_server

    model_dir = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.integers(0, 2, size=(60, len(model_server.FEATURE_COLS))), columns=model_server.FEATURE_COLS
    ).astype(float)
    X[model_server.NUMERIC_COLS] = rng.uniform(0, 100, size=(60, 3))
    y = rng.integers(0, 2, size=60)
    scaler = StandardScaler().fit(X[model_server.NUMERIC_COLS])
    X[model_server.NUMERIC_COLS] = scaler.transform(X[model_server.NUMERIC_COLS])
    joblib.dump(
        RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y), model_dir / "churn_model.pkl"
    )
    joblib.dump(scaler, model_dir / "scaler.pkl")

    model_server.MODEL_DIR = str(model_dir)
    with TestClient(model_server.app) as c:
        yield c


@pytest.fixture
def customer():
    from serving.model_server import FEATURE_COLS

    body = {col: 0 for col in FEATURE_COLS}
    body.update({"tenure": 2, "MonthlyCharges": 90.0, "TotalCharges": 180.0})
    return body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info_lists_30_features(client):
    assert client.get("/info").json()["n_features"] == 30


def test_predict_returns_valid_probability(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_negative_tenure_rejected(client, customer):
    customer["tenure"] = -1
    assert client.post("/predict", json=customer).status_code == 422


def test_flag_out_of_range_rejected(client, customer):
    customer["gender_Male"] = 2
    assert client.post("/predict", json=customer).status_code == 422


def test_missing_field_rejected(client, customer):
    customer.pop("tenure")
    assert client.post("/predict", json=customer).status_code == 422


def test_batch(client, customer):
    r = client.post("/predict/batch", json={"customers": [customer, customer]})
    assert r.status_code == 200
    assert r.json()["total"] == 2


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_risk_label_thresholds():
    from serving.model_server import risk_label

    assert risk_label(0.2) == "LOW"
    assert risk_label(0.5) == "MEDIUM"
    assert risk_label(0.9) == "HIGH"
