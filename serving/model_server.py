"""
FastAPI model server for customer churn prediction.

Endpoints:
  GET  /health         liveness / readiness check
  GET  /info           model metadata
  POST /predict        single prediction
  POST /predict/batch  batch predictions

Config (environment variables):
  MODEL_DIR   folder holding churn_model.pkl and scaler.pkl (default: ./models)
"""

import logging
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
THRESHOLD = 0.5

FEATURE_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "SeniorCitizen_1",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Loaded once at startup (not per request) and shared by every request
artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model and scaler from {MODEL_DIR}")
    artifacts["model"] = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
    artifacts["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    logger.info("Model and scaler loaded")
    yield
    artifacts.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps portfolio project — IBM Telco churn prediction",
    version="1.1.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    tenure: float = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    gender_Male: int = Field(..., ge=0, le=1)
    SeniorCitizen_1: int = Field(..., ge=0, le=1)
    Partner_Yes: int = Field(..., ge=0, le=1)
    Dependents_Yes: int = Field(..., ge=0, le=1)
    PhoneService_Yes: int = Field(..., ge=0, le=1)
    MultipleLines_No_phone_service: int = Field(..., ge=0, le=1, alias="MultipleLines_No phone service")
    MultipleLines_Yes: int = Field(..., ge=0, le=1)
    InternetService_Fiber_optic: int = Field(..., ge=0, le=1, alias="InternetService_Fiber optic")
    InternetService_No: int = Field(..., ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(
        ..., ge=0, le=1, alias="OnlineSecurity_No internet service"
    )
    OnlineSecurity_Yes: int = Field(..., ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(..., ge=0, le=1, alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int = Field(..., ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(
        ..., ge=0, le=1, alias="DeviceProtection_No internet service"
    )
    DeviceProtection_Yes: int = Field(..., ge=0, le=1)
    TechSupport_No_internet_service: int = Field(..., ge=0, le=1, alias="TechSupport_No internet service")
    TechSupport_Yes: int = Field(..., ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(..., ge=0, le=1, alias="StreamingTV_No internet service")
    StreamingTV_Yes: int = Field(..., ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(
        ..., ge=0, le=1, alias="StreamingMovies_No internet service"
    )
    StreamingMovies_Yes: int = Field(..., ge=0, le=1)
    Contract_One_year: int = Field(..., ge=0, le=1, alias="Contract_One year")
    Contract_Two_year: int = Field(..., ge=0, le=1, alias="Contract_Two year")
    PaperlessBilling_Yes: int = Field(..., ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(
        ..., ge=0, le=1, alias="PaymentMethod_Credit card (automatic)"
    )
    PaymentMethod_Electronic_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Mailed check")

    model_config = ConfigDict(populate_by_name=True)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def predict_frame(df: pd.DataFrame) -> list[PredictionResponse]:
    df = df[FEATURE_COLS].copy()
    df[NUMERIC_COLS] = artifacts["scaler"].transform(df[NUMERIC_COLS])
    probs = artifacts["model"].predict_proba(df)[:, 1]
    return [
        PredictionResponse(
            churn_probability=round(float(p), 4),
            churn_prediction=int(p >= THRESHOLD),
            risk_level=risk_label(float(p)),
        )
        for p in probs
    ]


@app.get("/health")
def health():
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": "churn_model.pkl"}


@app.get("/info")
def info():
    return {
        "model_type": type(artifacts["model"]).__name__,
        "n_features": len(FEATURE_COLS),
        "numeric_cols_scaled": NUMERIC_COLS,
        "features": FEATURE_COLS,
        "threshold": THRESHOLD,
        "version": app.version,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        df = pd.DataFrame([customer.model_dump(by_alias=True)])
        return predict_frame(df)[0]
    except Exception as err:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="prediction failed") from err


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    try:
        df = pd.DataFrame([c.model_dump(by_alias=True) for c in request.customers])
        predictions = predict_frame(df)
        return BatchResponse(predictions=predictions, total=len(predictions))
    except Exception as err:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="batch prediction failed") from err
