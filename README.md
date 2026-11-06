# Customer Churn Prediction — End-to-End MLOps Pipeline

Predicts which telecom customers will cancel their service, using the public
IBM Telco Customer Churn dataset (7,043 customers). The focus is the system
around the model: a reproducible DVC pipeline, MLflow experiment tracking and
model registry, Airflow scheduling, a quality gate, a tested FastAPI service in
Docker, and a GitHub Actions CI/CD pipeline.

![CI/CD](https://github.com/kiranml-projects/Customer-churn-prediction/actions/workflows/ci_cd.yaml/badge.svg)

---

## Architecture

```
                 Airflow DAG (daily) — each task runs `dvc repro <stage>`
 ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────────┐
 │ 1 ingest │ -> │ 2 preprocess │ -> │ 3 train  │ -> │ 4 evaluate + gate│
 └──────────┘    └──────────────┘    └──────────┘    └──────────────────┘
       │                │                  │                  │
       └──── DVC: data, model, metrics ────┘   MLflow: params, metrics,
             versioned by hash in dvc.lock     plots, model registry
                                │
                     churn_model.pkl + scaler.pkl
                                │
                 FastAPI in Docker  ->  /health /info /predict /predict/batch

 GitHub Actions on every PR / push to main:
   lint -> unit tests -> dvc repro -> model tests -> docker build
        -> Trivy scan -> container smoke test -> push to GHCR (main only)
```

## Tech stack

| Layer | Technology |
|---|---|
| Data and pipeline versioning | DVC 3 |
| Experiment tracking, model registry | MLflow 2.16 (SQLite backend) |
| Orchestration | Apache Airflow 3 |
| Model | scikit-learn RandomForestClassifier (`class_weight=balanced`) |
| Serving | FastAPI + Uvicorn, Pydantic validation |
| Container | Docker (non-root, health check) |
| CI/CD | GitHub Actions, Ruff, pytest, Trivy, GHCR |

## Model performance (held-out test set, 1,057 customers)

| Metric | Score |
|---|---|
| ROC-AUC | 0.84 |
| Recall (churners) | 0.71 |
| Precision (churners) | 0.56 |
| F1 (churners) | 0.63 |
| Accuracy | 0.77 |

26.5% of customers churn, so accuracy alone is misleading. The model is tuned
for **recall on churners**: it catches about 7 in 10 customers who will leave,
which suits a retention campaign where a missed churner costs more than an
unneeded offer. Top drivers: tenure, total and monthly charges, two-year
contract, fiber-optic internet, electronic-check payment.

Quality gate: a model is promoted only if test ROC-AUC ≥ 0.80 (`params.yaml`).

## Project structure

```
├── .github/workflows/ci_cd.yaml   # CI/CD pipeline
├── airflow/dags/dag.py            # daily DAG, one task per DVC stage
├── data/scripts/
│   ├── common.py                  # shared paths, params, MLflow config
│   ├── ingestion.py               # 1. download + validate raw data
│   ├── preprocess.py              # 2. clean, encode, split, scale
│   ├── train.py                   # 3. train + log to MLflow registry
│   └── evaluate.py                # 4. test metrics, plots, quality gate
├── serving/model_server.py        # FastAPI app
├── tests/                         # pytest: ingestion, preprocessing, model, API
├── dvc.yaml / dvc.lock            # pipeline definition / pinned hashes
├── params.yaml                    # all data, model and gate settings
├── Dockerfile, docker-compose.yml
├── requirements.txt               # training pipeline
├── requirements-serve.txt         # API image only
└── requirements-dev.txt           # tests + lint
```

## Quick start

```bash
git clone https://github.com/kiranml-projects/Customer-churn-prediction.git
cd Customer-churn-prediction
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

dvc repro            # run the whole pipeline (only changed stages re-run)
dvc metrics show     # test-set metrics
pytest -q            # 21 tests
```

Change a hyperparameter without editing code:

```bash
dvc exp run -S train.max_depth=12
dvc exp show
```

### Serve the model

```bash
uvicorn serving.model_server:app --port 8000      # local
docker compose up --build                         # API on :8000, MLflow UI on :5000
```

Swagger UI: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure": 2, "MonthlyCharges": 90, "TotalCharges": 180,
  "InternetService_Fiber optic": 1, "PaymentMethod_Electronic check": 1, ...
}'
# {"churn_probability": 0.87, "churn_prediction": 1, "risk_level": "HIGH"}
```

Risk levels: `LOW` < 0.40 ≤ `MEDIUM` < 0.70 ≤ `HIGH`. Invalid input returns 422.

### MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Each run logs parameters, validation metrics and the model, and registers a
new version of `churn-rf-model`. The evaluate run logs test metrics and plots.

### Airflow

Airflow runs in its own environment. Point the DAG at this repo and its venv:

```bash
export CHURN_PROJECT_DIR=$(pwd)
export CHURN_VENV=$(pwd)/.venv
export AIRFLOW_HOME=$(pwd)/airflow      # runtime files here are git-ignored
airflow standalone                      # UI on :8080, DAG: customer_churn_pipeline
```

### DVC remote

Configure storage per machine (not committed):

```bash
dvc remote add -d --local myremote s3://<bucket>/churn-dvc   # or a local folder
dvc push
```

## Roadmap

- [ ] Promote models with an MLflow `champion` alias (challenger vs. champion); API loads `models:/churn-rf-model@champion`
- [ ] One sklearn `Pipeline` for preprocessing + model, so the API accepts raw customer fields
- [ ] Prometheus `/metrics`, Grafana dashboard, data-drift checks (Evidently) that trigger retraining
- [ ] Kubernetes deployment (Helm, probes, HPA) and canary rollout with KServe
