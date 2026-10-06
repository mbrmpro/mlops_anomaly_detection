"""SPOTLIGHT: optional guided walkthrough of the Streamlit story.

This module is the adapter between Streamlit and the walkthrough renderer in
spotlight/spotlight.js and spotlight/spotlight.css. It holds the English step
registry and mounts the component once.

The walkthrough is presentation only. It reads the rendered page and may select
a story tab to show a target. It never calls the API and never changes model,
database, prediction or training state.

Every step explains one page element with four texts: what, role, code and
decision. A backtick pair marks a code identifier, which is rendered as <code>.
The optional "tab" field names the story tab that must be visible first.
"""

from pathlib import Path

import streamlit as st

ASSET_DIR = Path(__file__).parent / "spotlight"

# Stable root of the story tabs: st.tabs(..., key="story-tabs") in app.py.
TAB_ROOT = ".st-key-story-tabs"

STEPS = [
    {
        "target": ".st-key-spotlight-title",
        "title": "What this app does",
        "keyword": "overview",
        "what": "Finds defective products in photos.",
        "role": "Tells the project as a short story.",
        "code": "`frontend/app.py`",
        "decision": "Streamlit only shows results. The API does the work.",
    },
    {
        "target": ".st-key-spotlight-system-status",
        "title": "System status",
        "keyword": "health",
        "what": "Green means the API and the database are running.",
        "role": "Shows the system is ready before you test it.",
        "code": "`GET /health/db`",
        "decision": "The page checks everything through the API.",
    },
    {
        "target": ".st-key-story-tabs [role=\"tablist\"]",
        "title": "Six tabs",
        "keyword": "navigation",
        "what": "The story in six steps, from problem to system.",
        "role": "The only navigation on the page.",
        "code": "`st.tabs()`",
        "decision": "Problem first, architecture last.",
    },
    {
        "target": ".st-key-spotlight-challenge",
        "tab": "1 · The challenge",
        "title": "The challenge",
        "keyword": "normal only",
        "what": "Defects are rare and every one looks different.",
        "role": "So the model learns only what normal looks like.",
        "code": "`src/training.py`",
        "decision": "What the model cannot rebuild well is a defect.",
    },
    {
        "target": ".st-key-spotlight-data",
        "tab": "2 · The data",
        "title": "The data",
        "keyword": "batches",
        "what": "Photos from the MVTec AD dataset.",
        "role": "Training images arrive in three batches, like in a factory.",
        "code": "`GET /batches/status`",
        "decision": "A fixed test set makes every new model comparable.",
    },
    {
        "target": ".st-key-spotlight-mlflow",
        "tab": "3 · MLflow",
        "title": "Training with MLflow",
        "keyword": "experiment tracking",
        "what": "Rebuilds each image pixel by pixel, epoch after epoch.",
        "role": "MLflow saves each run: settings, losses, scores, model.",
        "code": "`mlflow.log_metric()`",
        "decision": "A new model becomes champion only if it scores higher.",
    },
    {
        "target": ".st-key-spotlight-try-model",
        "tab": "4 · Try the model",
        "title": "Try the model",
        "keyword": "prediction",
        "what": "Pick an image, get a verdict: normal or anomaly.",
        "role": "Sends the image to the trained model.",
        "code": "`POST /predict`",
        "decision": "Test images have a known answer, so we can check the model.",
    },
    {
        "target": ".st-key-spotlight-decision",
        "tab": "5 · How it decides",
        "title": "How it decides",
        "keyword": "anomaly score",
        "what": "One number: how badly the image was rebuilt.",
        "role": "Above the threshold means anomaly.",
        "code": "`src/predict.py`",
        "decision": "The threshold allows about 5% false alarms.",
    },
    {
        "target": ".st-key-spotlight-system",
        "tab": "6 · Behind the scenes",
        "title": "Behind the scenes",
        "keyword": "architecture",
        "what": "All services that run the project.",
        "role": "Streamlit, FastAPI, model, PostgreSQL, MLflow and Airflow.",
        "code": "`docker-compose.yml`",
        "decision": "One job per service, so each can change alone.",
    },
]

_spotlight = st.components.v2.component(
    "spotlight",
    js=(ASSET_DIR / "spotlight.js").read_text(encoding="utf-8"),
    css=(ASSET_DIR / "spotlight.css").read_text(encoding="utf-8"),
)


def render_spotlight():
    """Mount the SPOTLIGHT button and its walkthrough. Call once per page."""

    _spotlight(
        key="spotlight",
        data={"tab_root": TAB_ROOT, "steps": STEPS},
        width="content",
    )
