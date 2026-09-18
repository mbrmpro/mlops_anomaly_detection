from datetime import datetime


import requests


from airflow import DAG

from airflow.operators.python import PythonOperator



API_URL = "http://api:8000/training"



def trigger_training():

    payload = {

        "category": "bottle",

        "epochs": 1,

        "save_model": False,

    }


    response = requests.post(

        API_URL,

        json=payload,

        timeout=3600,

    )


    response.raise_for_status()


    print(response.json())



with DAG(

    dag_id="anomaly_detection_training",

    start_date=datetime(2026, 9, 18),

    schedule="0 2 * * 0",

    catchup=False,

    tags=["training", "anomaly-detection"],

) as dag:


    train_bottle = PythonOperator(

        task_id="train_bottle",

        python_callable=trigger_training,

    )
