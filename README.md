# Customer Churn Prediction — End-to-End MLOps Platform

An end-to-end MLOps platform for predicting customer churn on the IBM Telco dataset. This project demonstrates production-grade ML engineering — from data ingestion and experiment tracking to containerized model serving, pipeline orchestration, and drift monitoring.

---

## Architecture

```
Raw Data (DVC)
     |
     v
Airflow DAG
     |
     |-- ingestion.py      -> data/raw/
     |-- preprocess.py     -> data/processed/
     |-- train.py          -> MLflow tracking + model registry
     |-- evaluate.py       -> reports/ (ROC, confusion matrix)
     |
     v
FastAPI Model Server (Docker)
     |
     |-- GET  /health
     |-- GET  /info
     |-- POST /predict
     |-- POST /predict/batch
     |
     v
Monitoring (drift_check.py + Grafana)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Data Versioning** | DVC |
| **Experiment Tracking** | MLflow |
| **Pipeline Orchestration** | Apache Airflow 3.3 |
| **Model Serving** | FastAPI + Uvicorn |
| **Containerization** | Docker + Docker Compose |
| **ML Framework** | scikit-learn (RandomForestClassifier) |
| **Drift Monitoring** | Custom drift_check.py + Grafana |
| **Kubeflow Pipelines** | Distributed training + canary deployment |

---

## Project Structure

```
Customer-churn-prediction/
├── data/
│   ├── raw/                    # Raw Telco churn data (DVC tracked)
│   └── processed/              # Train/val/test splits
├── serving/
│   └── model_server.py         # FastAPI prediction server
├── models/
│   ├── churn_model.pkl         # Trained RandomForest model
│   └── scaler.pkl              # Feature scaler
├── monitoring/
│   ├── drift_check.py          # Data drift detection
│   └── grafana_dashboard.json  # Grafana dashboard config
├── kubeflow/
│   ├── components/             # Kubeflow pipeline components
│   └── pipelines.py            # Kubeflow pipeline definition
├── mlruns/                     # MLflow experiment runs
├── reports/                    # Evaluation plots and metrics
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Hyperparameters
├── Dockerfile                  # FastAPI server Docker image
├── docker-compose.yml          # Multi-service Docker setup
└── requirements.txt            # Python dependencies
```

---

## Model Performance

| Metric | Score |
|---|---|
| **ROC-AUC** | 0.84 |
| **F1-Score** | 0.79 |
| **Accuracy** | 0.81 |

Model: RandomForestClassifier with class balancing (`class_weight=balanced`)
Dataset: IBM Telco Customer Churn (7,043 customers, 30 features)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/kiranml-projects/Customer-churn-prediction.git
cd Customer-churn-prediction
```

### 2. Run with Docker

```bash
# Build and start all services
docker-compose up -d

# API available at http://localhost:8000/docs
# MLflow UI available at http://localhost:5000
```

### 3. Run locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn serving.model_server:app --reload --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### Health Check
```bash
GET /health
# {"status": "ok", "model": "churn_model.pkl"}
```

### Model Info
```bash
GET /info
# {"model_type": "RandomForestClassifier", "n_features": 30, ...}
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0,
  "gender_Male": 1,
  ...
}

# Response:
# {"churn_probability": 0.7007, "churn_prediction": 1, "risk_level": "HIGH"}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "customers": [
    {...},
    {...}
  ]
}

# Response:
# {"predictions": [...], "total": 2}
```

Risk levels: `LOW` (< 40%), `MEDIUM` (40-70%), `HIGH` (> 70%)

---

## MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Access at http://localhost:5000
```

Tracked per run:
- Hyperparameters (n_estimators, max_depth, etc.)
- Metrics (accuracy, f1_score, roc_auc)
- Model artifacts (churn-rf-model in registry)
- Evaluation plots (ROC curve, confusion matrix, feature importance)

---

## Pipeline Orchestration (Airflow)

```bash
# Start Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow standalone

# Access at http://localhost:8080
# DAG: customer_churn_pipeline
```

Pipeline tasks:
1. `ingest_data` — download and version raw data
2. `preprocess_data` — feature engineering and splitting
3. `train_model` — train with MLflow tracking
4. `evaluate_model` — generate evaluation reports

---

## Data Versioning (DVC)

```bash
# Pull data
dvc pull

# Run full pipeline
dvc repro

# Check pipeline status
dvc status
```

---

## Drift Monitoring

```bash
# Run drift check
python monitoring/drift_check.py

# View Grafana dashboard (requires Grafana setup)
# Import monitoring/grafana_dashboard.json
```

---

## Kubeflow Pipelines

Distributed training and canary deployment pipeline:
- `validate_data` — data quality checks
- `feature_engineering` — feature transformation
- `train_distributed` — distributed RandomForest training
- `evaluate` — model evaluation and promotion

---

## Key MLOps Concepts Demonstrated

- **Experiment tracking** — MLflow with model registry
- **Data versioning** — DVC with remote storage
- **Pipeline orchestration** — Airflow DAG with task dependencies
- **Model serving** — FastAPI with Pydantic validation
- **Containerization** — Docker multi-service setup
- **Drift detection** — statistical drift monitoring
- **Distributed training** — Kubeflow pipeline components
- **Canary deployment** — gradual rollout strategy

