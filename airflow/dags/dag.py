from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = "/mnt/c/Users/8J3604897/Customer-churn-prediction"
VENV_ACTIVATE = f"source {PROJECT_DIR}/airflow-venv/bin/activate"

def dvc_command(stage: str) -> str:
    """Builds the shell command to activate the venv, cd into the
    project root, and run a single DVC pipeline stage."""
    return f"{VENV_ACTIVATE} && cd {PROJECT_DIR} && dvc repro {stage}"

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
    description="End-to-end churn prediction pipeline (DVC-orchestrated)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",        # run every day
    catchup=False,                     # don't backfill missed runs
    tags=["mlops", "churn", "production"],
) as dag:

    # --------------------------------------------------------
    # TASK 1 — Ingest
    # --------------------------------------------------------
    ingest_task = BashOperator(
        task_id="ingest",
        bash_command=dvc_command("ingest"),
        doc_md="""
        Runs the DVC `ingest` stage: downloads the Telco churn dataset
        from source URL and saves it to data/raw/ with a timestamped
        filename. Skipped by DVC if inputs haven't changed since the
        last successful run.
        """,
    )

    # --------------------------------------------------------
    # TASK 2 — Preprocess
    # --------------------------------------------------------
    preprocess_task = BashOperator(
        task_id="preprocess",
        bash_command=dvc_command("preprocess"),
        doc_md="""
        Runs the DVC `preprocess` stage: loads latest raw CSV, cleans it,
        encodes features, splits into train/val/test and saves to
        data/processed/.
        """,
    )

    # --------------------------------------------------------
    # TASK 3 — Train
    # --------------------------------------------------------
    train_task = BashOperator(
        task_id="train",
        bash_command=dvc_command("train"),
        doc_md="""
        Runs the DVC `train` stage: trains RandomForest on train split,
        evaluates on val, logs params/metrics/model to MLflow.
        """,
    )

    # --------------------------------------------------------
    # TASK 4 — Evaluate
    # --------------------------------------------------------
    evaluate_task = BashOperator(
        task_id="evaluate",
        bash_command=dvc_command("evaluate"),
        doc_md="""
        Runs the DVC `evaluate` stage: evaluates model on held-out test
        set, generates confusion matrix, ROC curve, feature importance
        plots and writes the evaluation report.
        """,
    )
