"""
Airflow DAG: runs the DVC pipeline stage by stage, once a day.

Airflow owns WHEN (schedule, retries, alerting); DVC owns WHAT needs to
re-run (stages whose inputs are unchanged are skipped).

Set these Airflow Variables or environment variables before enabling it:
  CHURN_PROJECT_DIR  absolute path to this repository
  CHURN_VENV         absolute path to the Python venv with requirements.txt
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = os.getenv("CHURN_PROJECT_DIR", "/opt/churn")
VENV = os.getenv("CHURN_VENV", f"{PROJECT_DIR}/.venv")


def dvc_command(stage: str) -> str:
    """Activate the venv, cd to the repo, run one DVC stage (no upstream re-run)."""
    return f"source {VENV}/bin/activate && cd {PROJECT_DIR} && dvc repro --single-item {stage}"


default_args = {
    "owner": "kiran",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="customer_churn_pipeline",
    description="End-to-end churn prediction pipeline (DVC stages)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,  # two runs writing the same files would collide
    tags=["mlops", "churn"],
) as dag:
    ingest = BashOperator(task_id="ingest", bash_command=dvc_command("ingest"))
    preprocess = BashOperator(task_id="preprocess", bash_command=dvc_command("preprocess"))
    train = BashOperator(task_id="train", bash_command=dvc_command("train"))
    evaluate = BashOperator(task_id="evaluate", bash_command=dvc_command("evaluate"))

    ingest >> preprocess >> train >> evaluate
