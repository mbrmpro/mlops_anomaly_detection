# Anomaly Detection – MLOps Project


MLOps project for industrial image anomaly detection using a **Convolutional Autoencoder (CAE)** and the **MVTec Anomaly Detection dataset**.


The goal of the project is to build a reproducible ML pipeline that will later be extended with model tracking, data versioning, automation, deployment, and monitoring.


## Phase 1 – Foundations


The current implementation includes:


- MVTec image dataset

- PostgreSQL for image metadata

- Docker Compose for the database

- Dynamic dataset category detection

- CAE training pipeline

- Validation-based anomaly threshold

- Prediction pipeline

- FastAPI inference and training API

- Swagger API documentation


## Project Structure


```text

anomaly-detection-mlops/

├── api/

│   └── main.py

├── database/

│   ├── init.sql

│   └── load_data.py

├── src/

│   ├── training.py

│   └── predict.py

├── data/

├── models/

├── docker-compose.yml

├── requirements.txt

└── README.md

```


## Setup


Create the Python environment and install dependencies:


```bash

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

```


Start PostgreSQL:


```bash

docker compose up -d

```


Load dataset metadata:


```bash

python database/load_data.py

```


Train a model:


```bash

python src/training.py bottle

```


Start the API:


```bash

uvicorn api.main:app --host 0.0.0.0 --port 8000

```


Swagger UI:


```text

http://<HOST>:8000/docs

```


## API Endpoints


### `POST /training`


Starts model training for a selected category.


### `POST /predict`


Runs anomaly detection for an image and returns:


- anomaly score

- threshold

- prediction (`normal` or `anomaly`)


## Next Steps


Phase 2 will extend the project with:


- MLflow

- DVC

- Docker microservices

- automated training

- model versioning and registry


Later phases will add drift detection, monitoring, and a Streamlit interface.


## Team
Ayoub , Mohamed
