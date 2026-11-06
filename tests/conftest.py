import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "data", "scripts"))


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """A small raw dataset with the same schema as the IBM Telco file."""
    rows = []
    for i in range(40):
        churn = "Yes" if i % 4 == 0 else "No"
        rows.append(
            {
                "customerID": f"C{i:04d}",
                "gender": "Male" if i % 2 else "Female",
                "SeniorCitizen": 1 if i % 3 == 0 else 0,
                "Partner": "Yes" if i % 2 else "No",
                "Dependents": "No",
                "tenure": i,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": ["DSL", "Fiber optic", "No"][i % 3],
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 20.0 + i,
                "TotalCharges": " " if i == 0 else str(20.0 * i),
                "Churn": churn,
            }
        )
    return pd.DataFrame(rows)
