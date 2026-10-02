from datetime import datetime

import requests

from airflow import DAG
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


# ==========================================================
# RUN EVIDENTLY DRIFT REPORT
# ==========================================================

def trigger_drift_report(
    category,
):
    """
    Ask FastAPI to compare recent prediction inputs
    with the training images (Evidently).

    The result is stored in PostgreSQL and shown
    in the Grafana "Data Drift" dashboard.
    """

    response = requests.post(
        f"{API_URL}/monitoring/drift",
        json={"category": category},
        timeout=600,
    )

    response.raise_for_status()

    result = response.json()

    print(
        f"Drift report for {category}:",
        result,
    )

    return result


# ==========================================================
# DAG
# ==========================================================

with DAG(
    dag_id="anomaly_detection_drift_monitoring",

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
        "monitoring",
        "anomaly-detection",
        "evidently",
    ],

) as dag:

    for category in CATEGORIES:

        PythonOperator(
            task_id=f"drift_{category}",
            python_callable=trigger_drift_report,
            op_kwargs={
                "category": category,
            },
        )
