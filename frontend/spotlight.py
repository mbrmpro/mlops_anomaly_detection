"""SPOTLIGHT: optional guided walkthrough of the Streamlit story.

This module is the adapter between Streamlit and the walkthrough renderer in
spotlight/spotlight.js and spotlight/spotlight.css. It holds the English step
registry and mounts the component once.

The walkthrough is presentation only. It reads the rendered page and may select
a story tab to show a target. It never calls the API and never changes model,
database, prediction or training state.

One step per tab, in story order, after one overview step. Every step explains
the highlighted part with four short texts: what, role, code and decision. A
backtick pair marks a code identifier, which is rendered as <code>. The "tab"
field names the story tab that must be visible first; it must be one of the
tab labels that app.py passes to render_spotlight().
"""

from pathlib import Path

import streamlit as st

ASSET_DIR = Path(__file__).parent / "spotlight"

# Stable root of the story tabs: st.tabs(..., key="story-tabs") in app.py.
TAB_ROOT = ".st-key-story-tabs"

STEPS = [
    {
        "target": ".st-key-spotlight-title",
        "title": "Overview",
        "keyword": "overview",
        "what": "The MLOps pipeline of an image anomaly detector.",
        "role": "One tab per component, in processing order.",
        "code": "`frontend/app.py`",
        "decision": "Streamlit presents; the API executes.",
    },
    {
        "target": ".st-key-spotlight-flow",
        "tab": "1 · Data flow",
        "title": "Data flow",
        "keyword": "three loops",
        "what": "Training, serving and monitoring.",
        "role": "Arrow numbers give the processing order.",
        "code": "`docker-compose.yml`",
        "decision": "One database and one API keep the architecture simple.",
    },
    {
        "target": ".st-key-spotlight-compose",
        "tab": "2 · Docker Compose",
        "title": "Docker Compose",
        "keyword": "infrastructure",
        "what": "Seven services in one network.",
        "role": "Start order is enforced: the API waits for healthy dependencies.",
        "code": "`docker compose up -d --build`",
        "decision": "One container per responsibility: each restarts independently.",
    },
    {
        "target": ".st-key-spotlight-postgres",
        "tab": "3 · PostgreSQL",
        "title": "PostgreSQL",
        "keyword": "data",
        "what": "All images with category and ground-truth label.",
        "role": "Releases training data in batches; logs every prediction.",
        "code": "`database/init.sql`",
        "decision": "A fixed test set makes every model comparable.",
    },
    {
        "target": ".st-key-spotlight-airflow",
        "tab": "4 · Airflow",
        "title": "Airflow",
        "keyword": "schedule",
        "what": "Hourly DAG: batch release, then training.",
        "role": "Simulates data arriving over time.",
        "code": "`airflow/dags/training_dag.py`",
        "decision": "Retraining only on new data.",
    },
    {
        "target": ".st-key-spotlight-fastapi",
        "tab": "5 · FastAPI",
        "title": "FastAPI",
        "keyword": "entry point",
        "what": "Single entry point of the system.",
        "role": "Release, train, predict, check drift.",
        "code": "`api/main.py`",
        "decision": "One place to measure, secure and change.",
    },
    {
        "target": ".st-key-spotlight-tensorflow",
        "tab": "6 · TensorFlow",
        "title": "TensorFlow",
        "keyword": "model",
        "what": "A convolutional autoencoder.",
        "role": "Reconstructs normal images; defects reconstruct poorly.",
        "code": "`src/training.py`",
        "decision": "No labelled defects required.",
    },
    {
        "target": ".st-key-spotlight-mlflow",
        "tab": "7 · MLflow",
        "title": "MLflow",
        "keyword": "registry",
        "what": "Every run: parameters, metrics, model.",
        "role": "The alias `champion` moves only to a higher `test_auroc`.",
        "code": "`models:/cae-<category>@champion`",
        "decision": "A weaker model is never promoted.",
    },
    {
        "target": ".st-key-spotlight-prediction",
        "tab": "8 · Prediction",
        "title": "Prediction",
        "keyword": "verdict",
        "what": "One click scores one test image.",
        "role": "Score above the threshold means anomaly.",
        "code": "`POST /predict`",
        "decision": "Test images carry ground truth: each verdict is verifiable.",
    },
    {
        "target": ".st-key-spotlight-streamlit",
        "tab": "9 · Streamlit",
        "title": "Streamlit",
        "keyword": "presentation layer",
        "what": "This page and this walkthrough.",
        "role": "An API client only.",
        "code": "`frontend/spotlight.py`",
        "decision": "UI changes cannot affect the model.",
    },
    {
        "target": ".st-key-spotlight-prometheus",
        "tab": "10 · Prometheus",
        "title": "Prometheus",
        "keyword": "health",
        "what": "Traffic, errors, response time, memory.",
        "role": "Scrapes the API every 15 s; three alert rules.",
        "code": "`GET /metrics`",
        "decision": "Measured inside the API: every caller is counted.",
    },
    {
        "target": ".st-key-spotlight-quality",
        "tab": "11 · Holding Quality Evidence",
        "title": "Holding Quality Evidence",
        "keyword": "quality evidence",
        "what": "Drift check of four image features with Evidently.",
        "role": "Recent inputs vs training images, hourly; every completed check is stored.",
        "code": "`POST /monitoring/drift`",
        "decision": "Data quality is evidenced, not assumed.",
    },
    {
        "target": ".st-key-spotlight-grafana",
        "tab": "12 · Grafana",
        "title": "Grafana",
        "keyword": "operations",
        "what": "Two dashboards: API health and data drift.",
        "role": "Basis for decisions on new data.",
        "code": "`monitoring/grafana/dashboards/`",
        "decision": "The system monitors itself end to end.",
    },
]

_spotlight = st.components.v2.component(
    "spotlight",
    js=(ASSET_DIR / "spotlight.js").read_text(encoding="utf-8"),
    css=(ASSET_DIR / "spotlight.css").read_text(encoding="utf-8"),
)


def check_tabs(tab_labels):
    """Fail at page start when a step names a tab that app.py does not render.

    A wrong label would otherwise leave that step pointing at a hidden tab.
    """

    for index, step in enumerate(STEPS):
        if "tab" in step and step["tab"] not in tab_labels:
            raise ValueError(
                f"Spotlight step {index} ({step['title']}): "
                f"tab '{step['tab']}' is not one of the page tabs {tab_labels}"
            )


def render_spotlight(tab_labels):
    """Mount the SPOTLIGHT button and its walkthrough. Call once per page."""

    check_tabs(tab_labels)

    _spotlight(
        key="spotlight",
        data={"tab_root": TAB_ROOT, "steps": STEPS},
        width="content",
    )
