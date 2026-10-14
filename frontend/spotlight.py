"""SPOTLIGHT: optional guided walkthrough of the Streamlit story.

This module is the adapter between Streamlit and the walkthrough renderer in
spotlight/spotlight.js and spotlight/spotlight.css. It holds the English step
registry and mounts the component once.

The walkthrough is presentation only. It reads the rendered page and may select
a story tab to show a target. It never calls the API and never changes model,
database, prediction or training state.

One step per tab, in story order, after one overview step; the card number is
the tab number. SPOTLIGHT starts at the step of the open tab (on the first tab:
the overview). Every step explains
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
        "what": "An MLOps pipeline that finds defects in product images.",
        "role": "Each tab shows one component, in the order the data flows.",
        "code": "`frontend/app.py`",
        "decision": "The page only presents; the API does all the work.",
    },
    {
        "target": ".st-key-spotlight-flow",
        "tab": "1 · Data flow",
        "title": "Data flow",
        "keyword": "three loops",
        "what": "Three loops: training, serving and monitoring.",
        "role": "Each color marks one loop; the number in brackets is the tab.",
        "code": "`docker-compose.yml`",
        "decision": "One database and one API keep the system simple.",
    },
    {
        "target": ".st-key-spotlight-compose",
        "tab": "2 · Docker Compose",
        "title": "Docker Compose",
        "keyword": "infrastructure",
        "what": "Seven services start with one command.",
        "role": "The API starts only after the database and MLflow are healthy.",
        "code": "`docker compose up -d --build`",
        "decision": "One container per job, so each one restarts on its own.",
    },
    {
        "target": ".st-key-spotlight-postgres",
        "tab": "3 · PostgreSQL",
        "title": "PostgreSQL",
        "keyword": "data",
        "what": "Every image with its category and its true label.",
        "role": "It holds the training batches and logs every prediction.",
        "code": "`database/init.sql`",
        "decision": "A fixed test set makes every model comparable.",
    },
    {
        "target": ".st-key-spotlight-airflow",
        "tab": "4 · Airflow",
        "title": "Airflow",
        "keyword": "schedule",
        "what": "Every hour: release the next batch of data, then retrain.",
        "role": "It simulates new data arriving over time.",
        "code": "`airflow/dags/training_dag.py`",
        "decision": "The model is retrained only when new data arrives.",
    },
    {
        "target": ".st-key-spotlight-fastapi",
        "tab": "5 · FastAPI",
        "title": "FastAPI",
        "keyword": "entry point",
        "what": "The single entry point of the system.",
        "role": "It releases data, trains, predicts and checks drift.",
        "code": "`api/main.py`",
        "decision": "One place to measure, secure and change the system.",
    },
    {
        "target": ".st-key-spotlight-tensorflow",
        "tab": "6 · TensorFlow",
        "title": "TensorFlow",
        "keyword": "model",
        "what": "A convolutional autoencoder trained on normal images only.",
        "role": "It rebuilds normal images well and defects badly.",
        "code": "`src/training.py`",
        "decision": "No labelled defects are needed for training.",
    },
    {
        "target": ".st-key-spotlight-mlflow",
        "tab": "7 · MLflow",
        "title": "MLflow",
        "keyword": "registry",
        "what": "Every training run is logged: parameters, metrics and model.",
        "role": "A new version becomes `champion` only with a higher `test_auroc`.",
        "code": "`models:/cae-<category>@champion`",
        "decision": "A weaker model never reaches the user.",
    },
    {
        "target": ".st-key-spotlight-prediction",
        "tab": "8 · Prediction",
        "title": "Prediction",
        "keyword": "verdict",
        "what": "One click checks one test image.",
        "role": "A score above the threshold means anomaly.",
        "code": "`POST /predict`",
        "decision": "Test images have known labels, so every verdict can be checked.",
    },
    {
        "target": ".st-key-spotlight-streamlit",
        "tab": "9 · Streamlit",
        "title": "Streamlit",
        "keyword": "presentation layer",
        "what": "This page and this walkthrough.",
        "role": "It only calls the API: no model and no database access.",
        "code": "`frontend/spotlight.py`",
        "decision": "Changing the page can never change a prediction.",
    },
    {
        "target": ".st-key-spotlight-prometheus",
        "tab": "10 · Prometheus",
        "title": "Prometheus",
        "keyword": "health",
        "what": "Traffic, errors and prediction time of the API.",
        "role": "It reads the API every 15 seconds and checks three alert rules.",
        "code": "`GET /metrics`",
        "decision": "Measured inside the API, so every request is counted.",
    },
    {
        "target": ".st-key-spotlight-quality",
        "tab": "11 · Holding Quality Evidence",
        "title": "Holding Quality Evidence",
        "keyword": "quality evidence",
        "what": "A drift check of four image features with Evidently.",
        "role": "Every hour it compares recent inputs with the training images.",
        "code": "`POST /monitoring/drift`",
        "decision": "Each completed check is stored: quality is proven, not assumed.",
    },
    {
        "target": ".st-key-spotlight-grafana",
        "tab": "12 · Grafana",
        "title": "Grafana",
        "keyword": "operations",
        "what": "Two dashboards: API health and data drift.",
        "role": "They show when the model needs new data.",
        "code": "`monitoring/grafana/dashboards/`",
        "decision": "The system watches itself from end to end.",
    },
]

_spotlight = st.components.v2.component(
    "spotlight",
    js=(ASSET_DIR / "spotlight.js").read_text(encoding="utf-8"),
    css=(ASSET_DIR / "spotlight.css").read_text(encoding="utf-8"),
)


def check_tabs(tab_labels):
    """Fail at page start when a step names a tab that app.py does not render,
    or when the steps do not follow the tabs one to one.

    A wrong label would otherwise leave that step pointing at a hidden tab.
    """

    for index, step in enumerate(STEPS):
        if "tab" in step and step["tab"] not in tab_labels:
            raise ValueError(
                f"Spotlight step {index} ({step['title']}): "
                f"tab '{step['tab']}' is not one of the page tabs {tab_labels}"
            )

    # The card shows "STEP n / 12" with n counted over the steps that have a tab:
    # n is the tab number only when the steps follow the tabs one to one.
    step_tabs = [step["tab"] for step in STEPS if "tab" in step]

    if step_tabs != list(tab_labels):
        raise ValueError(
            "Spotlight steps must follow the page tabs one to one "
            f"(steps: {step_tabs}, tabs: {list(tab_labels)})"
        )


def render_spotlight(tab_labels):
    """Mount the SPOTLIGHT button and its walkthrough. Call once per page."""

    check_tabs(tab_labels)

    _spotlight(
        key="spotlight",
        data={"tab_root": TAB_ROOT, "steps": STEPS},
        width="content",
    )
