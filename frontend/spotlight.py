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
the highlighted part with four short texts: "what" (purpose: why the element
is needed; on the overview: the goal of the project), "role" (what it does in
this project), "code" (where the shown view is built) and "decision" (what to
notice on the screen). A
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
        "what": "We demonstrate an MLOps workflow for automated visual inspection.",
        "role": "Our prototype uses MVTec AD images of bottles, wood and pills.",
        "code": "`frontend/app.py` · title band",
        "decision": "Follow the tabs from system setup to a prediction and its monitoring.",
    },
    {
        "target": ".st-key-spotlight-flow",
        "tab": "1 · Data flow",
        "title": "Data flow",
        "keyword": "architecture",
        "what": "We separate training, serving and monitoring to make each responsibility clear.",
        "role": "Airflow and Streamlit reach the data and the models only through the API.",
        "code": "`frontend/app.py` · `flow_tab`",
        "decision": "Trace one image through prediction, then follow its logs into monitoring.",
    },
    {
        "target": ".st-key-spotlight-compose",
        "tab": "2 · Docker Compose",
        "title": "Docker Compose",
        "keyword": "deployment",
        "what": "A shared Compose configuration makes the project easier to reproduce.",
        "role": "Seven services run together, with health checks controlling API startup.",
        "code": "`frontend/app.py` · `compose_tab`",
        "decision": "Read the startup dependencies separately from the data flow in the previous tab.",
    },
    {
        "target": ".st-key-spotlight-postgres",
        "tab": "3 · PostgreSQL",
        "title": "PostgreSQL",
        "keyword": "data records",
        "what": "SQL tables give training, serving and monitoring one consistent, queryable record.",
        "role": "Batch availability controls which normal images can be used for training.",
        "code": "`frontend/app.py` · `postgres_tab`",
        "decision": "Compare the sample images with the tables that track their use.",
    },
    {
        "target": ".st-key-spotlight-airflow",
        "tab": "4 · Airflow",
        "title": "Airflow",
        "keyword": "batch release",
        "what": "Scheduled tasks make batch release and retraining repeatable.",
        "role": "The demo releases stored batches to simulate new data arriving over time.",
        "code": "`frontend/app.py` · `airflow_tab`",
        "decision": "Follow the training path and the skip path when no batch remains.",
    },
    {
        "target": ".st-key-spotlight-fastapi",
        "tab": "5 · FastAPI",
        "title": "FastAPI",
        "keyword": "API access",
        "what": "A shared API gives clients a consistent way to access backend operations.",
        "role": "Endpoints expose batch release, training, prediction and drift checks.",
        "code": "`frontend/app.py` · `fastapi_tab`",
        "decision": "Match Airflow and Streamlit to their API endpoints for backend operations.",
    },
    {
        "target": ".st-key-spotlight-tensorflow",
        "tab": "6 · TensorFlow",
        "title": "TensorFlow",
        "keyword": "reconstruction",
        "what": "Learning normal appearance reduces the need to collect defect examples for training.",
        "role": "The autoencoder reconstructs an image; reconstruction error is its anomaly score.",
        "code": "`frontend/app.py` · `tensorflow_tab`",
        "decision": "Follow the rule: an anomaly is flagged when the score exceeds the threshold.",
    },
    {
        "target": ".st-key-spotlight-mlflow",
        "tab": "7 · MLflow",
        "title": "MLflow",
        "keyword": "model selection",
        "what": "Tracking runs and model versions makes model changes traceable.",
        "role": "Without a champion, the candidate wins; otherwise, it needs a higher test AUROC.",
        "code": "`frontend/app.py` · `mlflow_tab`",
        "decision": "Connect the logged run to its registered version and the model used by the API.",
    },
    {
        "target": ".st-key-spotlight-prediction",
        "tab": "8 · Prediction",
        "title": "Prediction",
        "keyword": "inference",
        "what": "A labeled test image lets us check a prediction against a known outcome.",
        "role": "The API uses the category's champion model to return a score and a prediction.",
        "code": "`frontend/app.py` · `prediction_tab`",
        "decision": "Compare prediction and ground truth; the score ratio is not a probability.",
    },
    {
        "target": ".st-key-spotlight-streamlit",
        "tab": "9 · Streamlit",
        "title": "Streamlit",
        "keyword": "interface",
        "what": "A separate interface keeps presentation code apart from model execution.",
        "role": "Streamlit calls the API and displays images, predictions and monitoring results.",
        "code": "`frontend/app.py` · `streamlit_tab`, `frontend/spotlight.py`",
        "decision": "Use the diagram to locate the boundary between the interface and the backend.",
    },
    {
        "target": ".st-key-spotlight-prometheus",
        "tab": "10 · Prometheus",
        "title": "Prometheus",
        "keyword": "service health",
        "what": "Service metrics help us detect failures and slow responses.",
        "role": "Prometheus scrapes API metrics every 15 seconds and evaluates alert rules.",
        "code": "`frontend/app.py` · `prometheus_tab`",
        "decision": "Counts cover API uptime; error alerts use 5-minute rates, latency alerts 5-minute p95.",
    },
    {
        "target": ".st-key-spotlight-quality",
        "tab": "11 · Holding Quality Evidence",
        "title": "Holding Quality Evidence",
        "keyword": "input drift",
        "what": "Input drift checks help us notice changes that may affect model behavior.",
        "role": "Evidently compares four features of recent inputs with released training images.",
        "code": "`frontend/app.py` · `quality_tab`",
        "decision": "Inspect each feature result; input drift alone does not prove an accuracy drop.",
    },
    {
        "target": ".st-key-spotlight-grafana",
        "tab": "12 · Grafana",
        "title": "Grafana",
        "keyword": "operations",
        "what": "Dashboards bring service health and input drift together for operational review.",
        "role": "Prometheus supplies service metrics; PostgreSQL supplies stored drift results.",
        "code": "`frontend/app.py` · `grafana_tab`",
        "decision": "The dashed arrow marks a human decision; drift does not automatically trigger retraining.",
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
