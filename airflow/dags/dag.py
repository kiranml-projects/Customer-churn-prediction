from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Import main functions from each script
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data", "scripts"))

from ingestion import main as ingest_main
from preprocess import main as preprocess_main
from train import main as train_main
from evaluate import main as evaluate_main

# ============================================================
# DEFAULT ARGS — applied to every task
# ============================================================
default_args = {
    "owner": "kiran",
    "depends_on_past": False,          # don't wait for previous run
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,                      # retry once on failure
    "retry_delay": timedelta(minutes=5),
}

# ============================================================
# DAG DEFINITION
# ============================================================
with DAG(
    dag_id="customer_churn_pipeline",
    description="End-to-end churn prediction pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",        # run every day
    catchup=False,                     # don't backfill missed runs
    tags=["mlops", "churn", "production"],
) as dag:

    # --------------------------------------------------------
    # TASK 1 — Ingest
    # --------------------------------------------------------
    ingest_task = PythonOperator(
        task_id="ingest",
        python_callable=ingest_main,
        doc_md="""
        Downloads the Telco churn dataset from source URL
        and saves it to data/raw/ with a timestamped filename.
        """,
    )

    # --------------------------------------------------------
    # TASK 2 — Preprocess
    # --------------------------------------------------------
    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_main,
        doc_md="""
        Loads latest raw CSV, cleans it, encodes features,
        splits into train/val/test and saves to data/processed/.
        """,
    )

    # --------------------------------------------------------
    # TASK 3 — Train
    # --------------------------------------------------------
    train_task = PythonOperator(
        task_id="train",
        python_callable=train_main,
        doc_md="""
        Trains RandomForest on train split, evaluates on val,
        logs params/metrics/model to MLflow.
        """,
    )

    # --------------------------------------------------------
    # TASK 4 — Evaluate
    # --------------------------------------------------------
    evaluate_task = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate_main,
        doc_md="""
        Evaluates model on held-out test set, generates
        confusion matrix, ROC curve, feature importance plots
        and promotes model if ROC-AUC >= 0.80.
        """,
    )

    # --------------------------------------------------------
    # TASK DEPENDENCIES — defines execution order
    # --------------------------------------------------------
    ingest_task >> preprocess_task >> train_task >> evaluate_task