# Industrial Image Anomaly Detection — MLOps

This project implements an MLOps pipeline for industrial image anomaly detection using the MVTec AD dataset.

The project currently supports three categories:

* Bottle
* Wood
* Pill

A convolutional autoencoder is trained on normal images.
The reconstruction error is used to calculate an anomaly score.

---

## Architecture

The project is using:

* **PostgreSQL** — stores image metadata and training batch status
* **FastAPI** — exposes training, prediction and batch-management endpoints
* **Airflow** — orchestrates automatic data arrival and model training
* **TensorFlow** — trains the convolutional autoencoder
* **MLflow** — tracks experiments, parameters, metrics and model artifacts
* **Docker Compose** — starts and connects all services

Basic workflow:

```text
Airflow
   |
   | HTTP
   v
FastAPI
   |
   +----------> PostgreSQL
   |
   v
training.py
   |
   +----------> TensorFlow
   |
   +----------> MLflow
```

---

# Project Structure

```text
mlops_anomaly_detection/
├── airflow/
│   └── dags/
│       └── training_dag.py
├── api/
│   └── main.py
├── database/
│   ├── init.sql
│   └── load_data.py
├── data/
├── models/
├── src/
│   ├── training.py
│   ├── predict.py
│   └── evaluate.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Start the Project

Go to the project directory:

```bash
cd ~/jupyter/Anomaly_detection_project/MLops_Project/mlops_anomaly_detection
```

Start all Docker services:

```bash
docker compose up -d
```

Check the containers:

```bash
docker compose ps
```

The main containers are:

```text
anomaly_postgres
anomaly_mlflow
anomaly_api
anomaly_airflow
```

---

# Stop the Project

Stop all containers:

```bash
docker compose down
```

Do not use:

```bash
docker compose down -v
```

unless you intentionally want to delete Docker volumes.

---

# Application URLs

## FastAPI

```text
http://localhost:8000/docs
```

## Airflow

```text
http://localhost:8081
```

## MLflow

```text
http://localhost:5001
```

## PostgreSQL

```text
localhost:5432
```

---

# Check the Services

## FastAPI

```bash
curl http://localhost:8000/health/db
```

Expected:

```json
{
  "status": "ok",
  "database": "connected"
}
```

## MLflow

```bash
curl http://localhost:5001/health
```

Expected:

```text
OK
```

## Airflow

```bash
curl http://localhost:8081/health
```

The response should show healthy Airflow components.

---

# Airflow Login

List existing users:

```bash
docker compose exec airflow airflow users list
```

For a local demo, the admin password can be reset with:

```bash
docker compose exec airflow \
airflow users reset-password \
--username admin \
--password admin
```

Then log in with:

```text
Username: admin
Password: admin
```

---

# Dataset Preparation

The project uses:

```text
bottle
wood
pill
```

The data is divided into:

```text
train/
├── batch_1/
├── batch_2/
└── batch_3/

test/
├── good/
└── anomaly types...
```

The training batches simulate new data arriving over time.

The test set stays fixed.

---

# PostgreSQL Training Batches

Training images contain an `is_available` field.

Initially:

```text
Batch 1 → FALSE
Batch 2 → FALSE
Batch 3 → FALSE
```

This means the training data has not arrived yet.

Airflow changes the batches to available one by one.

---

# Reset the Training Batches

Before a demo, reset all training batches:

```bash
docker compose exec postgres \
psql -U anomaly_user -d anomaly_db \
-c "
UPDATE images
SET
    is_available = FALSE,
    released_at = NULL
WHERE split = 'train';
"
```

Check the batch status:

```bash
docker compose exec postgres \
psql -U anomaly_user -d anomaly_db \
-c "
SELECT
    category,
    batch_id,
    is_available,
    COUNT(*)
FROM images
WHERE split = 'train'
GROUP BY category, batch_id, is_available
ORDER BY category, batch_id;
"
```

Initial expected state:

```text
bottle | 1 | f | 61
bottle | 2 | f | 61
bottle | 3 | f | 61

pill   | 1 | f | 78
pill   | 2 | f | 78
pill   | 3 | f | 78

wood   | 1 | f | 71
wood   | 2 | f | 71
wood   | 3 | f | 71
```

`f` means `FALSE`.

`t` means `TRUE`.

---

# Airflow Workflow

The Airflow DAG is:

```text
anomaly_detection_incremental_training
```

The tasks run in this order:

```text
release_next_batch
        |
        v
train_bottle
        |
        v
train_wood
        |
        v
train_pill
```

Airflow communicates with FastAPI through HTTP requests.

For example:

```text
Airflow
   |
   | POST /batches/release-next
   v
FastAPI
```

FastAPI then updates PostgreSQL.

For training:

```text
Airflow
   |
   | POST /training
   v
FastAPI
   |
   v
training.py
```

---

# First Airflow Run

Trigger the DAG once from the Airflow interface.

Airflow releases:

```text
Batch 1
```

The database becomes:

```text
Batch 1 → TRUE
Batch 2 → FALSE
Batch 3 → FALSE
```

Training uses:

```text
Bottle → 61 images
Wood   → 71 images
Pill   → 78 images
```

The three models are trained one after another.

---

# Second Airflow Run

Trigger the DAG again.

Airflow releases:

```text
Batch 2
```

Training becomes cumulative:

```text
Bottle → 61 + 61 = 122 images
Wood   → 71 + 71 = 142 images
Pill   → 78 + 78 = 156 images
```

The models are retrained using:

```text
Batch 1 + Batch 2
```

---

# Third Airflow Run

Trigger the DAG again.

Airflow releases:

```text
Batch 3
```

Training now uses:

```text
Bottle → 183 images
Wood   → 213 images
Pill   → 234 images
```

The models are trained using:

```text
Batch 1 + Batch 2 + Batch 3
```

---

# Fourth Airflow Run

After all three batches are available, there is no new data.

Airflow skips the training workflow.

This prevents unnecessary retraining.

---

# Model Training

The actual machine-learning training happens in:

```text
src/training.py
```

The training pipeline:

```text
Read available training images
        |
        v
Create train / validation split
        |
        v
Train TensorFlow autoencoder
        |
        v
Calculate anomaly threshold
        |
        v
Evaluate on fixed test set
        |
        v
Log results to MLflow
```

---

# MLflow

MLflow runs inside Docker.

Inside the Docker network:

```text
http://mlflow:5000
```

From the browser:

```text
http://localhost:5001
```

Each training run logs information such as:

```text
category
available batches
latest batch
number of training images
epochs
threshold
accuracy
precision
recall
F1
AUROC
confusion matrix
```

It also stores:

```text
evaluation.json
threshold.json
model artifacts
```

Example MLflow runs:

```text
cae-bottle-batch-1
cae-wood-batch-1
cae-pill-batch-1

cae-bottle-batch-2
cae-wood-batch-2
cae-pill-batch-2

cae-bottle-batch-3
cae-wood-batch-3
cae-pill-batch-3
```

---

# Clean MLflow Before a Demo

Stop the containers:

```bash
docker compose down
```

Remove the Dockerized MLflow history:

```bash
sudo rm -rf mlflow_data
mkdir -p mlflow_data
```

Start the project again:

```bash
docker compose up -d
```

Check MLflow:

```bash
curl http://localhost:5001/health
```

---

# Manual Training

Training can also be executed without Airflow.

Example:

```bash
python src/training.py bottle --epochs 1
```

For a quick test without replacing the saved model:

```bash
python src/training.py bottle --epochs 1 --no-save
```

Airflow is preferred for the complete automated workflow.

---

# FastAPI

FastAPI provides the interface between Airflow and the ML pipeline.

Main endpoints:

```text
GET  /
GET  /health/db

GET  /batches/status

POST /batches/release-next

POST /training

POST /predict
```

Interactive documentation:

```text
http://localhost:8000/docs
```

---

# Docker Communication

Docker Compose gives each service a hostname.

Inside Docker:

```text
Airflow → http://api:8000

FastAPI → postgres:5432

FastAPI / training.py → http://mlflow:5000
```

From the local computer:

```text
FastAPI → localhost:8000

Airflow → localhost:8081

MLflow → localhost:5001

PostgreSQL → localhost:5432
```

---

# Complete MLOps Workflow

```text
1. Docker Compose starts the services
        |
        v
2. Airflow starts the DAG
        |
        v
3. Airflow requests a new batch
        |
        v
4. FastAPI updates PostgreSQL
        |
        v
5. New training data becomes available
        |
        v
6. Airflow requests model training
        |
        v
7. FastAPI calls training.py
        |
        v
8. TensorFlow trains the autoencoder
        |
        v
9. Model is evaluated on the fixed test set
        |
        v
10. Metrics and artifacts are logged in MLflow
        |
        v
11. Next Airflow run releases the next batch
```

---

# Useful Docker Commands

Show running containers:

```bash
docker compose ps
```

Show API logs:

```bash
docker compose logs -f api
```

Show Airflow logs:

```bash
docker compose logs -f airflow
```

Show MLflow logs:

```bash
docker compose logs -f mlflow
```

Show PostgreSQL logs:

```bash
docker compose logs -f postgres
```

Restart one service:

```bash
docker compose restart api
```

Rebuild the API:

```bash
docker compose up -d --build api
```

Start everything:

```bash
docker compose up -d
```

Stop everything:

```bash
docker compose down
```

---

# Git

The main source files for the data-automation pipeline are:

```text
src/training.py
api/main.py
airflow/dags/training_dag.py
docker-compose.yml
database/load_data.py
```

Runtime files should normally not be committed:

```text
data/
models/
mlflow_data/
mlflow.db
mlruns/
.venv/
```

---

# Current Status

Implemented:

* PostgreSQL dataset metadata
* Fixed test dataset
* Three simulated training batches
* TensorFlow convolutional autoencoder
* FastAPI training API
* Automatic batch release
* Airflow orchestration
* Docker Compose services
* MLflow experiment tracking
* Cumulative retraining

Next planned steps:

* MLflow Model Registry
* Compare candidate models
* Select or promote the best model
* Dataset versioning with DVC
* CI/CD improvements

---

## Authors

* Ayoub
* Mohamed
* Pavel
