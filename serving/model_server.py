"""
FastAPI Model Server for Customer Churn Prediction
Endpoints:
  GET  /health   — liveness check
  GET  /info     — model metadata
  POST /predict  — single prediction
  POST /predict/batch — batch predictions
"""
import logging
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

FEATURE_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Male", "SeniorCitizen_1", "Partner_Yes", "Dependents_Yes",
    "PhoneService_Yes", "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

logger.info("Loading model and scaler...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logger.info("Model and scaler loaded ✅")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps portfolio project — IBM Telco churn prediction",
    version="1.0.0",
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
    OnlineSecurity_No_internet_service: int = Field(..., ge=0, le=1, alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int = Field(..., ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(..., ge=0, le=1, alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int = Field(..., ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(..., ge=0, le=1, alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int = Field(..., ge=0, le=1)
    TechSupport_No_internet_service: int = Field(..., ge=0, le=1, alias="TechSupport_No internet service")
    TechSupport_Yes: int = Field(..., ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(..., ge=0, le=1, alias="StreamingTV_No internet service")
    StreamingTV_Yes: int = Field(..., ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(..., ge=0, le=1, alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: int = Field(..., ge=0, le=1)
    Contract_One_year: int = Field(..., ge=0, le=1, alias="Contract_One year")
    Contract_Two_year: int = Field(..., ge=0, le=1, alias="Contract_Two year")
    PaperlessBilling_Yes: int = Field(..., ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(..., ge=0, le=1, alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Mailed check")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str

class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int

def risk_label(prob: float) -> str:
    if prob >= 0.7: return "HIGH"
    if prob >= 0.4: return "MEDIUM"
    return "LOW"

def predict_single(features: CustomerFeatures) -> PredictionResponse:
    data = features.model_dump(by_alias=True)
    df = pd.DataFrame([data])[FEATURE_COLS]

    # Scale only numeric columns
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= 0.5)
    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_level=risk_label(prob),
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": "churn_model.pkl"}

@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "n_features": len(FEATURE_COLS),
        "numeric_cols_scaled": NUMERIC_COLS,
        "features": FEATURE_COLS,
        "version": "1.0.0",
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        return predict_single(customer)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    try:
        predictions = [predict_single(c) for c in request.customers]
        return BatchResponse(predictions=predictions, total=len(predictions))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
