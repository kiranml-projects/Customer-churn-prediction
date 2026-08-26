from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import sys

PROJECT_DIR = "/mnt/c/Users/8J3604897/Customer-churn-prediction"

default_args = {
    "owner": "kiran",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def run_script(script_path):
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=PROJECT_DIR
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

with DAG(
    dag_id="customer_churn_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "churn"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=lambda: run_script(f"{PROJECT_DIR}/data/scripts/ingestion.py"),
    )
    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=lambda: run_script(f"{PROJECT_DIR}/data/scripts/preprocess.py"),
    )
    train = PythonOperator(
        task_id="train",
        python_callable=lambda: run_script(f"{PROJECT_DIR}/data/scripts/train.py"),
    )
    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=lambda: run_script(f"{PROJECT_DIR}/data/scripts/evaluate.py"),
    )

    ingest >> preprocess >> train >> evaluate
