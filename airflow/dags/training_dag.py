from datetime import datetime

import requests

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator


# ==========================================================
# CONFIG
# ==========================================================

API_URL = "http://api:8000"

CATEGORIES = (
    "bottle",
    "wood",
    "pill",
)

EPOCHS = 1


# ==========================================================
# RELEASE NEXT DATA BATCH
# ==========================================================

def release_next_batch():
    """
    Simulate new data arriving.

    Run 1 -> Batch 1
    Run 2 -> Batch 2
    Run 3 -> Batch 3
    """

    response = requests.post(
        f"{API_URL}/batches/release-next",
        timeout=30,
    )

    response.raise_for_status()

    result = response.json()

    print(
        "Batch release response:",
        result,
    )

    if result["batch_id"] is None:
        raise AirflowSkipException(
            "All three batches "
            "have already been released."
        )

    return result


# ==========================================================
# TRAIN CATEGORY
# ==========================================================

def trigger_training(
    category,
):
    """
    Trigger training through FastAPI.

    training.py automatically reads all batches
    that are currently available.
    """

    payload = {
        "category": category,
        "epochs": EPOCHS,
        "save_model": True,
    }

    print(
        f"Starting training for {category}"
    )

    response = requests.post(
        f"{API_URL}/training",
        json=payload,
        timeout=7200,
    )

    response.raise_for_status()

    result = response.json()

    print(
        f"Training completed for {category}"
    )

    print(result)

    return result


# ==========================================================
# DAG
# ==========================================================

with DAG(
    dag_id="anomaly_detection_incremental_training",

    start_date=datetime(
        2026,
        9,
        22,
    ),

    # Run every hour.
    schedule="@hourly",

    catchup=False,

    max_active_runs=1,

    tags=[
        "training",
        "anomaly-detection",
        "incremental-data",
    ],

) as dag:

    # ------------------------------------------------------
    # Release new data
    # ------------------------------------------------------

    release_batch = PythonOperator(
        task_id="release_next_batch",
        python_callable=release_next_batch,
    )

    # ------------------------------------------------------
    # Train bottle
    # ------------------------------------------------------

    train_bottle = PythonOperator(
        task_id="train_bottle",
        python_callable=trigger_training,
        op_kwargs={
            "category": "bottle",
        },
    )

    # ------------------------------------------------------
    # Train wood
    # ------------------------------------------------------

    train_wood = PythonOperator(
        task_id="train_wood",
        python_callable=trigger_training,
        op_kwargs={
            "category": "wood",
        },
    )

    # ------------------------------------------------------
    # Train pill
    # ------------------------------------------------------

    train_pill = PythonOperator(
        task_id="train_pill",
        python_callable=trigger_training,
        op_kwargs={
            "category": "pill",
        },
    )

    # ------------------------------------------------------
    # WORKFLOW
    # ------------------------------------------------------

    release_batch >> train_bottle >> train_wood >> train_pill