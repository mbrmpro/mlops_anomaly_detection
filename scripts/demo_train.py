"""Train the champion models for the demo, in the same order as Airflow.

Run through scripts/demo_setup.sh (it starts this file in the API image):

    for every training batch that is still waiting:
        POST /batches/release-next
        POST /training  bottle, wood, pill  (EPOCHS epochs each)

Training is cumulative and MLflow keeps a new version as champion only when
its test_auroc is higher - exactly what the Airflow DAG does, but with more
epochs so that the models are good enough to be judged live.

Idempotent: with all batches released, a finished training run on the newest
released batch and a champion for every category it does nothing. A category
without such a run (for example after an interrupted setup) or without
champion (for example after MLflow was cleaned) is trained once on all
released images.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = os.getenv("DEMO_API_URL", "http://127.0.0.1:8000")
MLFLOW_URL = os.getenv("DEMO_MLFLOW_URL", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "anomaly-detection-cae"
CATEGORIES = ("bottle", "wood", "pill")
EPOCHS = int(os.getenv("DEMO_EPOCHS", "30"))


def call(method, url, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"null")


def has_champion(category):
    query = urllib.parse.urlencode({"name": f"cae-{category}", "alias": "champion"})
    try:
        call("GET", f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/alias?{query}")
        return True
    except urllib.error.HTTPError as error:
        if error.code in (400, 404):
            return False
        raise


def batch_ids(released):
    batches = call("GET", f"{API_URL}/batches/status")["batches"]
    return sorted({row["batch_id"] for row in batches if bool(row["is_available"]) == released})


def trained_on(category, batch_id):
    """True when a finished MLflow run trained this category up to batch_id.

    src/training.py tags every run with category and latest_batch; a run that
    was interrupted never reaches the FINISHED status.
    """

    query = urllib.parse.urlencode({"experiment_name": EXPERIMENT_NAME})
    try:
        experiment = call("GET", f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name?{query}")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise

    answer = call(
        "POST",
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": [experiment["experiment"]["experiment_id"]],
            "filter": (
                f"tags.category = '{category}' "
                f"and tags.latest_batch = '{batch_id}' "
                "and attributes.status = 'FINISHED'"
            ),
            "max_results": 1,
        },
    )
    return bool(answer.get("runs"))


def train(category):
    print(f"{time.strftime('%H:%M:%S')}  train {category} ({EPOCHS} epochs) ...", flush=True)
    answer = call(
        "POST",
        f"{API_URL}/training",
        {"category": category, "epochs": EPOCHS},
        timeout=3600,
    )
    registry = (answer.get("result") or {}).get("model_registry") or {}
    print(
        f"           version {registry.get('model_version')}, "
        f"test_auroc {registry.get('candidate_score')}, "
        f"promoted {registry.get('promoted')}",
        flush=True,
    )


def main():
    trained = False

    for batch_id in batch_ids(released=False):
        released = call("POST", f"{API_URL}/batches/release-next")
        print(f"{time.strftime('%H:%M:%S')}  released batch {batch_id}: {released}", flush=True)

        for category in CATEGORIES:
            train(category)

        trained = True

    # An interrupted earlier run can leave the newest batch released but not
    # trained for every category: those categories are trained now.
    released_batches = batch_ids(released=True)

    if not released_batches:
        print("FAILED: no training batch in PostgreSQL (database/load_data.py has not run)")
        sys.exit(1)

    newest_batch = released_batches[-1]

    for category in CATEGORIES:
        if not trained_on(category, newest_batch) or not has_champion(category):
            train(category)
            trained = True

    missing = [category for category in CATEGORIES if not has_champion(category)]

    if missing:
        print(f"FAILED: no champion model for {', '.join(missing)}")
        sys.exit(1)

    print("champions ready" if trained else "champions already present: nothing to train")


if __name__ == "__main__":
    main()
