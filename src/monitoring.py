"""Data drift monitoring of prediction inputs with Evidently.

Every prediction stores simple statistics of its input image in PostgreSQL.
A drift run compares the predictions of the last WINDOW_HOURS hours
(current data) with the released training images of the same category
(reference data). Evidently decides which features drifted; the results are
stored in PostgreSQL and shown by the Grafana "Data Drift" dashboard.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import text

from src.predict import decode_image


# ==========================================================
# CONFIG
# ==========================================================

FEATURES = ["brightness", "contrast", "sharpness", "saturation"]

# Predictions of the last 24 hours are compared with the reference data.
WINDOW_HOURS = 24

# Fewer predictions are not enough for a meaningful statistical test.
MIN_CURRENT_ROWS = 10


# ==========================================================
# TABLES
# ==========================================================

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category VARCHAR(50) NOT NULL,
    image_path TEXT NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    prediction VARCHAR(10) NOT NULL,
    brightness DOUBLE PRECISION NOT NULL,
    contrast DOUBLE PRECISION NOT NULL,
    sharpness DOUBLE PRECISION NOT NULL,
    saturation DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    reference_rows INTEGER NOT NULL,
    current_rows INTEGER NOT NULL,
    drifted_features INTEGER NOT NULL,
    total_features INTEGER NOT NULL,
    drift_share DOUBLE PRECISION NOT NULL,
    dataset_drift BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_feature_results (
    report_id INTEGER NOT NULL REFERENCES drift_reports (id),
    feature VARCHAR(50) NOT NULL,
    method VARCHAR(100) NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    drift_score DOUBLE PRECISION NOT NULL,
    drifted BOOLEAN NOT NULL
);
"""


def create_tables(engine):
    """Create the monitoring tables (also for existing PostgreSQL volumes)."""

    with engine.begin() as connection:
        connection.execute(text(CREATE_TABLES_SQL))


# ==========================================================
# IMAGE FEATURES
# ==========================================================

def image_features(image_path):
    """Statistics of the model input image (128x128 RGB, values 0..1)."""

    image = decode_image(image_path).numpy()
    gray = image.mean(axis=2)

    # Laplacian: high values mean sharp edges, low values mean blur.
    laplacian = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(np.abs(laplacian).mean()),
        "saturation": float((image.max(axis=2) - image.min(axis=2)).mean()),
    }


# ==========================================================
# PREDICTION LOG (CURRENT DATA)
# ==========================================================

def log_prediction(engine, result, image_file=None):
    """Store one prediction and the features of its input image.

    image_file: file to read the features from when it is not
    result["image_path"] (for example the temporary file of an upload).
    """

    features = image_features(image_file or result["image_path"])

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO prediction_logs (
                    category, image_path, anomaly_score, prediction,
                    brightness, contrast, sharpness, saturation
                )
                VALUES (
                    :category, :image_path, :anomaly_score, :prediction,
                    :brightness, :contrast, :sharpness, :saturation
                );
                """
            ),
            {
                "category": result["category"],
                "image_path": result["image_path"],
                "anomaly_score": result["anomaly_score"],
                "prediction": result["prediction"],
                **features,
            },
        )


# ==========================================================
# DRIFT REPORT
# ==========================================================

def run_drift_report(engine, category):
    """Compare recent prediction inputs with the training images."""

    evaluated_at = datetime.now(timezone.utc)
    window_start = evaluated_at - timedelta(hours=WINDOW_HOURS)

    with engine.connect() as connection:
        current = pd.read_sql(
            text(
                """
                SELECT brightness, contrast, sharpness, saturation
                FROM prediction_logs
                WHERE category = :category
                  AND created_at >= :window_start;
                """
            ),
            connection,
            params={
                "category": category,
                "window_start": window_start,
            },
        )

        reference_paths = connection.execute(
            text(
                """
                SELECT image_path
                FROM images
                WHERE category = :category
                  AND split = 'train'
                  AND is_available = TRUE
                ORDER BY image_path;
                """
            ),
            {"category": category},
        ).scalars().all()

    summary = {
        "category": category,
        "window_start": window_start.isoformat(),
        "evaluated_at": evaluated_at.isoformat(),
        "reference_rows": len(reference_paths),
        "current_rows": len(current),
    }

    if len(current) < MIN_CURRENT_ROWS or not reference_paths:
        return {
            "status": "skipped",
            "reason": (
                f"Need at least {MIN_CURRENT_ROWS} predictions in the last "
                f"{WINDOW_HOURS} hours and released training images."
            ),
            **summary,
        }

    reference = pd.DataFrame(
        [image_features(path) for path in reference_paths]
    )

    # ------------------------------------------------------
    # EVIDENTLY
    # ------------------------------------------------------

    # include_tests=True: Evidently marks each drift check as FAIL/SUCCESS.
    report = Report(
        [DataDriftPreset(columns=FEATURES)],
        include_tests=True,
    )

    snapshot = report.run(
        current_data=current[FEATURES],
        reference_data=reference[FEATURES],
    ).dict()

    test_status = {
        test["metric_config"]["metric_id"]: test["status"]
        for test in snapshot["tests"]
    }

    # DriftedColumnsCount: number and share of drifted features.
    dataset_metric = next(
        metric
        for metric in snapshot["metrics"]
        if metric["config"]["type"] == "evidently:metric_v2:DriftedColumnsCount"
    )

    # ValueDrift: one statistical test per feature.
    # float(): Evidently returns numpy numbers, PostgreSQL needs plain floats.
    feature_results = [
        {
            "feature": metric["config"]["column"],
            "method": metric["config"]["method"],
            "threshold": float(metric["config"]["threshold"]),
            "drift_score": float(metric["value"]),
            "drifted": test_status[metric["id"]] == "FAIL",
        }
        for metric in snapshot["metrics"]
        if metric["config"]["type"] == "evidently:metric_v2:ValueDrift"
    ]

    result = {
        "status": "completed",
        **summary,
        "drifted_features": int(dataset_metric["value"]["count"]),
        "total_features": len(feature_results),
        "drift_share": float(dataset_metric["value"]["share"]),
        "dataset_drift": test_status[dataset_metric["id"]] == "FAIL",
        "features": feature_results,
    }

    # ------------------------------------------------------
    # STORE FOR GRAFANA
    # ------------------------------------------------------

    with engine.begin() as connection:
        report_id = connection.execute(
            text(
                """
                INSERT INTO drift_reports (
                    category, evaluated_at, window_start,
                    reference_rows, current_rows,
                    drifted_features, total_features,
                    drift_share, dataset_drift
                )
                VALUES (
                    :category, :evaluated_at, :window_start,
                    :reference_rows, :current_rows,
                    :drifted_features, :total_features,
                    :drift_share, :dataset_drift
                )
                RETURNING id;
                """
            ),
            {
                "category": category,
                "evaluated_at": evaluated_at,
                "window_start": window_start,
                "reference_rows": result["reference_rows"],
                "current_rows": result["current_rows"],
                "drifted_features": result["drifted_features"],
                "total_features": result["total_features"],
                "drift_share": result["drift_share"],
                "dataset_drift": result["dataset_drift"],
            },
        ).scalar_one()

        connection.execute(
            text(
                """
                INSERT INTO drift_feature_results (
                    report_id, feature, method,
                    threshold, drift_score, drifted
                )
                VALUES (
                    :report_id, :feature, :method,
                    :threshold, :drift_score, :drifted
                );
                """
            ),
            [
                {"report_id": report_id, **feature}
                for feature in feature_results
            ],
        )

    return result
