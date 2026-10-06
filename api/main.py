import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.monitoring import create_tables, log_prediction, run_drift_report
from src.training import train
from src.predict import predict


# ==========================================================
# CONFIG
# ==========================================================

@asynccontextmanager
async def lifespan(app):
    # Tables for prediction logs and drift reports.
    create_tables(engine)
    yield


app = FastAPI(
    title="Anomaly Detection API",
    lifespan=lifespan,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set"
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# ==========================================================
# PROMETHEUS METRICS (API PERFORMANCE)
# ==========================================================

# The four "golden signals" of the API:
#   traffic, errors, latency -> http_request_duration_seconds (_count, _bucket)
#   saturation               -> http_requests_in_progress + process CPU / memory
# route is the endpoint template, e.g. /predict ("other" = docs or 404).
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "route", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300, 1800),
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently being processed.",
)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    # Prometheus scrapes /metrics every 15 s: do not count these requests.
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()

    with REQUESTS_IN_PROGRESS.track_inprogress():
        response = await call_next(request)

    route = request.scope.get("route")

    REQUEST_DURATION.labels(
        method=request.method,
        route=route.path if route else "other",
        status=str(response.status_code),
    ).observe(time.perf_counter() - start)

    return response


@app.get("/metrics", include_in_schema=False)
def metrics():
    # Also contains process_cpu_seconds_total and process_resident_memory_bytes.
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ==========================================================
# REQUEST MODELS
# ==========================================================

class TrainingRequest(BaseModel):
    category: str
    epochs: int = 30
    save_model: bool = True


class PredictionRequest(BaseModel):
    image_path: str
    category: str


class DriftRequest(BaseModel):
    category: str


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "anomaly-detection-api",
    }


# ==========================================================
# DATABASE HEALTH
# ==========================================================

@app.get("/health/db")
def database_health():
    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return {
            "status": "ok",
            "database": "connected",
        }

    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Database connection failed: "
                f"{str(error)}"
            ),
        )


# ==========================================================
# BATCH STATUS
# ==========================================================

@app.get("/batches/status")
def batch_status():
    """
    Show the current state of the three training batches.
    """

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    SELECT
                        category,
                        batch_id,
                        BOOL_AND(is_available) AS is_available,
                        COUNT(*) AS image_count
                    FROM images
                    WHERE split = 'train'
                    GROUP BY
                        category,
                        batch_id
                    ORDER BY
                        category,
                        batch_id;
                    """
                )
            )

            rows = result.fetchall()

        return {
            "batches": [
                {
                    "category": row[0],
                    "batch_id": row[1],
                    "is_available": bool(row[2]),
                    "image_count": row[3],
                }
                for row in rows
            ]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# RELEASE NEXT BATCH
# ==========================================================

@app.post("/batches/release-next")
def release_next_batch():
    """
    Simulate the arrival of new data.

    First call:
        releases Batch 1

    Second call:
        releases Batch 2

    Third call:
        releases Batch 3

    After that:
        no additional batch is released.
    """

    try:
        with engine.begin() as connection:

            # Find the next batch that has not arrived yet.
            result = connection.execute(
                text(
                    """
                    SELECT MIN(batch_id)
                    FROM images
                    WHERE split = 'train'
                      AND is_available = FALSE;
                    """
                )
            )

            next_batch = result.scalar()

            if next_batch is None:
                return {
                    "status": "complete",
                    "message": (
                        "All training batches "
                        "have already been released."
                    ),
                    "batch_id": None,
                }

            # Release the same batch for all categories.
            result = connection.execute(
                text(
                    """
                    UPDATE images
                    SET
                        is_available = TRUE,
                        released_at = CURRENT_TIMESTAMP
                    WHERE split = 'train'
                      AND batch_id = :batch_id
                      AND is_available = FALSE;
                    """
                ),
                {
                    "batch_id": next_batch,
                },
            )

            released_images = result.rowcount

        return {
            "status": "released",
            "batch_id": int(next_batch),
            "released_images": released_images,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# TRAINING
# ==========================================================

@app.post("/training")
def training_endpoint(
    request: TrainingRequest,
):
    try:
        result = train(
            request.category,
            epochs=request.epochs,
            save_model=request.save_model,
        )

        return {
            "status": "success",
            "message": (
                f"Training completed "
                f"for {request.category}"
            ),
            "result": result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# PREDICTION
# ==========================================================

@app.post("/predict")
def prediction_endpoint(
    request: PredictionRequest,
):
    try:
        result = predict(
            request.image_path,
            request.category,
        )

        # Input features are the "current data" for drift monitoring.
        log_prediction(engine, result)

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# DATA DRIFT (EVIDENTLY)
# ==========================================================

@app.post("/monitoring/drift")
def drift_endpoint(
    request: DriftRequest,
):
    """
    Compare the inputs of recent predictions with the
    released training images and store the result for Grafana.
    """

    try:
        return run_drift_report(
            engine,
            request.category,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# PREDICTION FROM AN UPLOADED IMAGE (STREAMLIT)
# ==========================================================

@app.post("/predict/upload")
def prediction_upload_endpoint(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Predict an uploaded image.

    The image is saved to a temporary file and passed to
    the same predict() function that POST /predict uses.
    """

    try:
        suffix = Path(file.filename or "").suffix

        with tempfile.NamedTemporaryFile(suffix=suffix) as image_file:
            image_file.write(file.file.read())
            image_file.flush()

            result = predict(
                image_file.name,
                category,
            )

            # Show the uploaded file name instead of the temporary path.
            result["image_path"] = file.filename

            # Uploads are "current data" for drift monitoring too.
            # The features are read while the temporary file still exists.
            log_prediction(engine, result, image_file=image_file.name)

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================================
# TEST SET EXAMPLES (STREAMLIT)
# ==========================================================

@app.get("/images/examples")
def image_examples(
    category: str,
):
    """
    One image of the fixed test set per defect type
    (good, broken_large, ...) together with its true label.
    """

    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT DISTINCT ON (defect_type)
                        id,
                        image_path,
                        defect_type,
                        is_anomaly
                    FROM images
                    WHERE category = :category
                      AND split = 'test'
                    ORDER BY defect_type, image_path;
                    """
                ),
                {"category": category},
            ).fetchall()

        examples = [
            {
                "id": row[0],
                "image_path": row[1],
                "defect_type": row[2],
                "is_anomaly": bool(row[3]),
            }
            for row in rows
        ]

        # Normal example first, then the defect types.
        examples.sort(
            key=lambda example: (example["is_anomaly"], example["defect_type"])
        )

        return {
            "category": category,
            "examples": examples,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/images/{image_id}/file")
def image_file(
    image_id: int,
):
    """
    Return the file of an image registered in the images table.
    Only paths stored in PostgreSQL can be read.
    """

    with engine.connect() as connection:
        image_path = connection.execute(
            text(
                """
                SELECT image_path
                FROM images
                WHERE id = :image_id;
                """
            ),
            {"image_id": image_id},
        ).scalar()

    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found",
        )

    return FileResponse(image_path)