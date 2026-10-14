"""Streamlit frontend: the anomaly detection MLOps pipeline, one tab per component.

Built for a short presentation: every tab shows what a component does with one
diagram, one table and, where it exists, one live element, centered like a
slide. The tabs follow the order in which the components act on the data.

Streamlit is only a client. Every live number and every image on this page
comes from the FastAPI service. While the API is offline the page stays
readable and every live part shows "Live view paused" instead of an error.
Regression check: python frontend/check_app.py
"""

import os
import time

import pandas as pd
import requests
import streamlit as st

from spotlight import render_spotlight


# ==========================================================
# CONFIG
# ==========================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Browser addresses of the other tools. In GitHub Codespaces every port has its
# own forwarded address (docker-compose.yml passes CODESPACE_NAME in); anywhere
# else the published localhost ports are used.
CODESPACE_NAME = os.getenv("CODESPACE_NAME", "")
CODESPACE_DOMAIN = os.getenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN") or "app.github.dev"


def tool_url(port, path=""):
    """Address of a published port as the presenter's browser reaches it."""

    if CODESPACE_NAME:
        return f"https://{CODESPACE_NAME}-{port}.{CODESPACE_DOMAIN}{path}"

    return f"http://localhost:{port}{path}"


# Same categories as src/training.py (PROJECT_CATEGORIES).
CATEGORIES = ("bottle", "wood", "pill")

CATEGORY_ICON = {"bottle": "🍾", "wood": "🪵", "pill": "💊"}

# The use case of the presentation slides, reduced to the image source
# (the slides also name the inspection signal): category -> (group, product, camera).
USE_CASE = {
    "bottle": ("Bottles", "Manufactured bottle", "Camera before shipping"),
    "pill": ("Tablets", "Finished tablet", "Overhead line camera"),
    "wood": ("Wood", "Machined surface", "Camera before sorting"),
}

# The four image features of the drift check (src/monitoring.py):
# name -> what it measures.
DRIFT_FEATURES = {
    "brightness": "mean gray value",
    "contrast": "spread of the gray values",
    "sharpness": "edges (blur lowers it)",
    "saturation": "color strength",
}

# The story order. frontend/spotlight.py refers to these labels and
# render_spotlight() rejects any label that is not in this list.
TAB_LABELS = [
    "1 · Data flow",
    "2 · Docker Compose",
    "3 · PostgreSQL",
    "4 · Airflow",
    "5 · FastAPI",
    "6 · TensorFlow",
    "7 · MLflow",
    "8 · Prediction",
    "9 · Streamlit",
    "10 · Prometheus",
    "11 · Holding Quality Evidence",
    "12 · Grafana",
]

# Palette of the presentation slides; .streamlit/config.toml sets the same
# indigo as the primary color.
INDIGO = "#3D3B8E"
LAVENDER = "#ECEAF6"
MUTED = "#6E6C93"

# One color per loop of the pipeline. The diagrams use the same hex values.
LOOP_COLOR = {
    "setup": MUTED,
    "training": INDIGO,
    "serving": "#1F8A70",
    "monitoring": "#C8632D",
}

# Shared look of every diagram on the page. Streamlit draws all diagram text in
# the theme font; PAGE_STYLE sets its size to the 14 pt used for the layout here.
GRAPH_STYLE = """
    node [shape=box, style="rounded,filled", fillcolor="#ECEAF6", color="#B9B6DA", fontcolor="#26254A", fontname="Helvetica", fontsize=14];
    edge [fontname="Helvetica", fontsize=14, color="#8F8CC4", fontcolor="#6E6C93"];
"""

# One tab is one screen on a 1920 x 1080 display at 100 % browser zoom: the page is shown
# at 80 % (CSS zoom on the main block only, so Streamlit's header and full-height layout
# stay unscaled), where the longest tab still fits.
PAGE_ZOOM = 0.8

# Layout pixels (before the zoom) above the content of a tab (top padding, title band, two
# tab rows, tab panel padding) and below it (bottom padding). The content of every tab
# fills the screen height minus both; measured on the rendered page.
TAB_TOP = 247
TAB_BOTTOM = 20

# The block that holds the content of one story tab.
TAB_CONTENT = (
    '.st-key-story-tabs [data-testid="stTabPanel"] > [data-testid="stVerticalBlock"]'
    ' > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]'
)

PAGE_STYLE = f"""
<style>
.st-key-title-band {{ background: {INDIGO}; color: #ffffff; border-radius: 8px; padding: 18px 24px; }}
.band-title {{ font-size: 1.7rem; font-weight: 700; line-height: 1.2; }}
.band-status {{ font-size: .78rem; font-weight: 600; text-align: right; white-space: nowrap; line-height: 1.8; }}
.band-sub {{ font-size: .95rem; opacity: .85; margin-top: 4px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ zoom: {PAGE_ZOOM}; padding: 2rem 3rem 1rem; max-width: 1300px; margin: 0 auto; }}
/* Every tab fills the screen below the tabs like a slide: its parts are spread evenly and its
   takeaway line always sits at the bottom. 100vh inside the zoomed block is 80 % of the screen. */
{TAB_CONTENT} {{ min-height: calc({100 / PAGE_ZOOM:g}vh - {TAB_TOP + TAB_BOTTOM}px); justify-content: space-evenly;
  position: relative; padding-bottom: 3.5rem; }}
{TAB_CONTENT} > [data-testid="stElementContainer"]:has(.takeaway) {{ position: absolute; left: 0; right: 0; bottom: 0; }}
/* The twelve story tabs as two rows of six separate tabs (1-6 above 7-12): the labels no
   longer read as one long line, and no tab can hide behind a scroll arrow. */
.st-key-story-tabs [role="tablist"] {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0; overflow-x: visible; padding: 0 1px 1px 0; }}
.st-key-story-tabs [role="tab"] {{ justify-content: center; min-width: 0; height: auto; padding: 7px 8px;
  margin: 0 -1px -1px 0; border: 1px solid #DCD9EE; border-radius: 8px 8px 0 0; background: {LAVENDER}; }}
.st-key-story-tabs [role="tab"]:hover {{ background: #E2DFF3; }}
.st-key-story-tabs [role="tab"][aria-selected="true"] {{ position: relative; z-index: 1; background: #ffffff;
  border-color: {INDIGO}; }}
.st-key-story-tabs [role="tab"] p {{ font-size: .8rem; text-align: center; white-space: normal; }}
.st-key-story-tabs [role="tab"][aria-selected="true"] p {{ color: {INDIGO}; font-weight: 600; }}
.section-head {{ text-align: center; }}
.kicker {{ font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }}
.headline {{ color: {INDIGO}; font-size: 1.8rem; font-weight: 700; line-height: 1.2; margin: 2px 0 6px; }}
.takeaway {{ display: flex; justify-content: center; align-items: center; gap: 14px; color: {INDIGO};
  font-weight: 600; text-align: center; margin: 4px 0; }}
.takeaway::before, .takeaway::after {{ content: ""; width: 48px; height: 2px; background: #F2C4AE; flex: none; }}
.name {{ color: {INDIGO}; font-weight: 600; }}
.card {{ background: {LAVENDER}; border-radius: 10px; padding: 10px 16px; }}
.card b {{ display: block; color: {INDIGO}; }}
.thumb {{ width: 64px; height: 64px; border-radius: 50%; background: {LAVENDER};
  display: flex; align-items: center; justify-content: center; font-size: 1.8rem; }}
.st-key-use-case img {{ border-radius: 50%; }}
[data-testid="stGraphVizChart"] svg text {{ font-size: 14px !important; }}
[data-testid="stMarkdownContainer"] table {{ border-collapse: collapse; font-size: .9rem; }}
[data-testid="stMarkdownContainer"] th {{ background: {LAVENDER}; color: {INDIGO}; font-weight: 600; }}
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {{
  border: 1px solid #E2E0F1; padding: 6px 12px; }}
</style>
"""


# ==========================================================
# API CLIENT
# ==========================================================

def api_get(path, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response


def error_detail(response):
    try:
        return response.json()["detail"]
    except (ValueError, KeyError):
        return response.text


def short_error(error):
    """One readable reason instead of a raw exception dump."""

    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}"

    if isinstance(error, requests.Timeout):
        return "no answer in time"

    if isinstance(error, requests.ConnectionError):
        return "connection refused"

    return type(error).__name__


# Live reads are cached for a few seconds: Streamlit runs the code of every tab
# on each interaction, and the page should not flood the API it is measuring.
LIVE_TTL_SECONDS = 15


@st.cache_data(ttl=LIVE_TTL_SECONDS, show_spinner=False)
def check_system():
    """Ask the API (at most every 15 s): (api_online, database_online)."""

    try:
        requests.get(f"{API_URL}/", timeout=3).raise_for_status()
    except requests.RequestException:
        return False, False

    try:
        requests.get(f"{API_URL}/health/db", timeout=3).raise_for_status()
    except requests.RequestException:
        return True, False

    return True, True


@st.cache_data(ttl=60, show_spinner=False)
def load_examples(category):
    return api_get("/images/examples", category=category).json()["examples"]


@st.cache_data(show_spinner=False)
def load_image(image_id):
    return api_get(f"/images/{image_id}/file").content


@st.cache_data(ttl=LIVE_TTL_SECONDS, show_spinner=False)
def fetch_batches():
    return api_get("/batches/status").json()["batches"]


@st.cache_data(ttl=LIVE_TTL_SECONDS, show_spinner=False)
def fetch_metrics():
    return api_get("/metrics").text


@st.cache_data(ttl=LIVE_TTL_SECONDS, show_spinner=False)
def ping_api():
    start = time.perf_counter()
    api_get("/")
    return time.perf_counter() - start


def image_or_none(image_id):
    try:
        return load_image(image_id)
    except requests.RequestException:
        return None


def good_example_image(category):
    """A normal test image of the category, or None."""

    try:
        examples = load_examples(category)
    except requests.RequestException:
        return None

    for example in examples:
        if example["defect_type"] == "good":
            return image_or_none(example["id"])

    return None


def parse_metrics(text):
    """Prometheus text format -> list of (name, labels, value)."""

    samples = []

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue

        name_part, _, value = line.rpartition(" ")
        name, _, label_part = name_part.partition("{")
        labels = {}

        for pair in label_part.rstrip("}").split(","):
            if "=" in pair:
                key, _, label_value = pair.partition("=")
                labels[key] = label_value.strip('"')

        try:
            samples.append((name, labels, float(value)))
        except ValueError:
            continue

    return samples


def is_correct(result, true_label):
    return (result["prediction"] == "anomaly") == (true_label != "good")


def request_prediction(send_request, category, image_name, true_label=None, image_bytes=None):
    """Call the API, keep the answer in the session and show errors.

    The category is kept with the answer: the verdict is shown only while
    that category is selected.
    """

    start = time.perf_counter()

    try:
        with st.spinner("The API is checking the image..."):
            response = send_request()
    except requests.RequestException as error:
        st.error(f"Prediction failed: the API did not answer ({short_error(error)}).")
        return

    if response.status_code != 200:
        st.error(f"Prediction failed (HTTP {response.status_code}): {error_detail(response)}")
        return

    st.session_state["last_prediction"] = {
        "result": response.json(),
        "category": category,
        "image": image_name,
        "image_bytes": image_bytes,
        "true_label": true_label,
        "seconds": time.perf_counter() - start,
    }


# ==========================================================
# SHARED PAGE PARTS
# ==========================================================

def centered(key=None):
    """A block whose content-sized parts (diagram, table, metrics, buttons)
    sit in the middle of the page, as on a slide."""

    return st.container(key=key, horizontal_alignment="center")


def section(kicker, headline, loop):
    """Slide-style opening of a tab: a small colored kicker and one headline."""

    st.html(
        '<div class="section-head">'
        f'<div class="kicker" style="color: {LOOP_COLOR[loop]}">{kicker}</div>'
        f'<div class="headline">{headline}</div>'
        "</div>"
    )


def takeaway(text):
    """The one sentence the audience should remember from a tab."""

    st.html(f'<div class="takeaway">{text}</div>')


def status_line(label, ok):
    """One line of the system status in the title band: a green or orange dot."""

    dot = "#7FE0B4" if ok else "#F6B39A"
    return f'<span style="color: {dot}">●</span> {label}'


def live_paused():
    """Shown instead of a live part while the API is offline."""

    st.info(
        "Live view paused: the API is offline. "
        "Start the system with `docker compose up -d` to see live data here.",
        icon="⏸️",
    )


def open_links(*links):
    """One-click buttons, side by side in the middle, that open a tool's own UI
    in a new browser tab."""

    with st.container(horizontal=True, horizontal_alignment="center"):
        for label, url in links:
            st.link_button(label, url)


def metric_row(*metrics):
    """Live numbers side by side in the middle of the page: (label, value) pairs."""

    with st.container(horizontal=True, horizontal_alignment="center", gap="large"):
        for label, value in metrics:
            st.metric(label, value, width="content")


def diagram(dot):
    """Graphviz diagram at its natural size: the same text size on every tab.

    A diagram wider than the page shrinks to the page width.
    """

    st.graphviz_chart(dot, width="content")


def use_case_rows():
    """The use case as live rows: product image, product, camera."""

    with st.container(key="use-case", width=640):
        for category in ("bottle", "pill", "wood"):
            group, product, camera = USE_CASE[category]
            thumb, name, camera_column = st.columns([0.7, 1.6, 1.8], vertical_alignment="center")

            image = good_example_image(category) if api_online else None

            if image is None:
                thumb.html(f'<div class="thumb">{CATEGORY_ICON[category]}</div>')
            else:
                thumb.image(image, width=64)

            name.html(
                f'<div class="kicker" style="color: {MUTED}">{group}</div>'
                f'<div class="name">{product}</div>'
            )
            camera_column.html(f'<div class="card"><b>{camera}</b></div>')


def feature_table(results=None):
    """The drift features as one table. With the results of a completed check
    it also shows the p-value and the verdict of each feature."""

    if results is None:
        rows = [f"| `{name}` | {meaning} |" for name, meaning in DRIFT_FEATURES.items()]
        return "\n".join(["| Feature | Measures |", "|---|---|", *rows])

    rows = []

    for result in results:
        p_value = result["drift_score"]
        shown_p_value = "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"
        drifted = "yes" if result["drifted"] else "no"
        meaning = DRIFT_FEATURES.get(result["feature"], "—")
        rows.append(f"| `{result['feature']}` | {meaning} | {shown_p_value} | {drifted} |")

    return "\n".join(["| Feature | Measures | p-value | Drifted |", "|---|---|---|---|", *rows])


def show_prediction(prediction):
    result = prediction["result"]
    score = result["anomaly_score"]
    threshold = result["threshold"]

    with st.container(border=True):
        image_column, verdict_column = st.columns([1, 4], vertical_alignment="center")

        if prediction.get("image_bytes") is not None:
            image_column.image(prediction["image_bytes"], caption=prediction["image"], width="stretch")

        with verdict_column:
            if result["prediction"] == "anomaly":
                st.markdown("## Verdict: :red[ANOMALY]")
            else:
                st.markdown("## Verdict: :green[NORMAL]")

            true_label = prediction["true_label"]

            if true_label is not None:
                truth = "normal product" if true_label == "good" else f"defect ({true_label})"

                if is_correct(result, true_label):
                    st.markdown(f"**Ground truth:** {truth} → prediction :green[correct].")
                else:
                    st.markdown(f"**Ground truth:** {truth} → prediction :red[incorrect].")

            st.metric("Score / threshold", f"{score / threshold:.2f} x")

            st.caption(
                f"score {score:.6f} · threshold {threshold:.6f} · "
                f"model {result.get('model_name')} v{result.get('model_version')} (MLflow champion) · "
                f"response time {prediction['seconds']:.2f} s"
            )


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="Anomaly Detection — MLOps pipeline",
    page_icon="🔍",
    layout="wide",
)

st.html(PAGE_STYLE)

st.session_state.setdefault("last_prediction", None)

# Asked at most every 15 s. While the API is offline the page stays readable
# and every live part is paused: no request, no error dump, no dead button.
api_online, database_online = check_system()

# ----------------------------------------------------------
# Title band with the live system status
# ----------------------------------------------------------

if not api_online:
    status = status_line("API offline", False)
else:
    status = status_line("API connected", True) + "<br>" + status_line(
        "Database connected" if database_online else "Database not reachable", database_online
    )

with centered("spotlight-title"):
    # Title, SPOTLIGHT button (optional guided walkthrough, presentation only) and live
    # status share one band: the button needs no row of its own above the tabs.
    with st.container(key="title-band", horizontal=True, vertical_alignment="center", gap="medium", wrap=False):
        st.html(
            '<div class="band-title">🔍 Finding defects without ever seeing one</div>'
            '<div class="band-sub">A model learns only normal products; whatever it cannot rebuild is a defect.</div>'
        )
        render_spotlight(TAB_LABELS)
        st.html(f'<div class="band-status">{status}</div>', width="content")

    if not api_online:
        st.info(
            "**API: offline**. Live parts are paused. "
            "Start the system with `docker compose up -d`.",
            icon="⏸️",
        )
    elif not database_online:
        st.error("Database: not reachable. Live parts may fail.")

(
    flow_tab,
    compose_tab,
    postgres_tab,
    airflow_tab,
    fastapi_tab,
    tensorflow_tab,
    mlflow_tab,
    prediction_tab,
    streamlit_tab,
    prometheus_tab,
    quality_tab,
    grafana_tab,
) = st.tabs(TAB_LABELS, key="story-tabs")

# ==========================================================
# 1 · DATA FLOW
# ==========================================================

with flow_tab, centered():
    with centered("spotlight-flow"):
        section("Architecture", "Three loops, one database, one API", "setup")

        diagram(
            """
digraph {
    rankdir=LR;
    nodesep=0.3;
    ranksep=0.35;
"""
            + GRAPH_STYLE
            + """
    files [label="MVTec AD\\nimages", shape=folder, fillcolor="#ffffff"];
    user [label="User\\n(browser)", shape=oval, fillcolor="#ffffff"];

    subgraph cluster_compose {
        label="Docker Compose (tab 2)";
        fontname="Helvetica";
        fontsize=14;
        fontcolor="#6E6C93";
        style="rounded,dashed";
        color="#B9B6DA";

        postgres [label="PostgreSQL\\n(tab 3)", shape=cylinder, fillcolor="#FCEBE3"];
        airflow [label="Airflow\\n(tab 4)"];
        api [label="FastAPI\\n(tab 5)", fillcolor="#FCEBE3"];
        tensorflow [label="TensorFlow\\n(tab 6)"];
        mlflow [label="MLflow\\n(tab 7)"];
        prediction [label="Prediction\\n(tab 8)"];
        streamlit [label="Streamlit\\n(tab 9)"];
        prometheus [label="Prometheus\\n(tab 10)"];
        evidently [label="Evidently\\n(tab 11)"];
        grafana [label="Grafana\\n(tab 12)", fillcolor="#FCEBE3"];
    }

    files -> postgres;

    airflow -> api [color="#3D3B8E"];
    api -> postgres [color="#3D3B8E"];
    postgres -> tensorflow [color="#3D3B8E"];
    tensorflow -> mlflow [color="#3D3B8E"];

    user -> streamlit [color="#1F8A70"];
    streamlit -> api [color="#1F8A70"];
    mlflow -> prediction [color="#1F8A70"];
    api -> prediction [color="#1F8A70"];
    prediction -> postgres [color="#1F8A70"];

    airflow -> evidently [color="#C8632D"];
    postgres -> evidently [color="#C8632D", dir=both];
    api -> prometheus [color="#C8632D"];
    prometheus -> grafana [color="#C8632D"];
    postgres -> grafana [color="#C8632D"];
}
"""
        )

    st.markdown(
        f"""
| Loop | Tabs | Flow | Frequency |
|---|---|---|---|
| <b style="color: {LOOP_COLOR['training']}">Training</b> | 4 · 6 · 7 | image registration → batch release → training → new model version | hourly |
| <b style="color: {LOOP_COLOR['serving']}">Serving</b> | 8 · 9 | input image → champion score → logged verdict | on request |
| <b style="color: {LOOP_COLOR['monitoring']}">Monitoring</b> | 10 · 11 · 12 | drift checks and API metrics → dashboards | hourly; metrics every 15 s |
""",
        unsafe_allow_html=True,
        width="content",
    )

    takeaway("Colors mark the three loops; tab numbers are in brackets.")

# ==========================================================
# 2 · DOCKER COMPOSE
# ==========================================================

with compose_tab, centered():
    with centered("spotlight-compose"):
        section("Infrastructure", "Seven services, one command", "setup")

        st.code("docker compose up -d --build", language="bash", width="content")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    postgres [shape=cylinder, fillcolor="#FCEBE3"];
    api [fillcolor="#FCEBE3"];
    postgres -> api [label="healthy"];
    mlflow -> api [label="healthy"];
    api -> airflow;
    api -> prometheus;
    api -> streamlit;
    prometheus -> grafana;
    postgres -> grafana [label="healthy"];
}
"""
        )

    st.markdown(
        """
| Container | Responsibility |
|---|---|
| postgres | images, predictions, drift results |
| mlflow | experiment runs and model registry |
| api | single entry point |
| airflow | hourly scheduling |
| prometheus | API metrics and alerts |
| grafana | dashboards |
| streamlit | presentation layer |
""",
        width="content",
    )

    takeaway("Data persists across restarts: database volume and mlflow_data/.")

# ==========================================================
# 3 · POSTGRESQL
# ==========================================================

with postgres_tab, centered():
    with centered("spotlight-postgres"):
        section("Data", "One database: images, predictions, quality evidence", "setup")

        use_case_rows()

    st.markdown(
        """
| Table | Contents |
|---|---|
| `images` | every image with category, label, training batch and release status |
| `prediction_logs` | every prediction with four image features (tab 11) |
| `drift_reports` | every completed drift check: the quality evidence (tab 11) |
""",
        width="content",
    )

    if not api_online:
        live_paused()
    else:
        try:
            batches = pd.DataFrame(fetch_batches())
        except requests.RequestException as error:
            st.error(f"Could not load the batch status ({short_error(error)}).")
            batches = None

        if batches is not None and batches.empty:
            st.info("No images registered yet. Run database/load_data.py first.")
        elif batches is not None:
            released = batches[batches["is_available"]]
            metric_row(
                (
                    "Training batches released",
                    f"{released['batch_id'].nunique()} / {batches['batch_id'].nunique()}",
                ),
            )

    takeaway("Fixed test set: every model is evaluated on the same images.")

# ==========================================================
# 4 · AIRFLOW
# ==========================================================

with airflow_tab, centered():
    with centered("spotlight-airflow"):
        section("Training loop", "Hourly batch release and retraining", "training")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    release [label="release_next_batch", fillcolor="#FCEBE3"];
    bottle [label="train_bottle"];
    wood [label="train_wood"];
    pill [label="train_pill"];
    skip [label="no new batch\\n→ training skipped", shape=note, fillcolor="#ffffff"];
    release -> bottle -> wood -> pill [color="#3D3B8E"];
    release -> skip [style=dashed];
}
"""
        )

    st.markdown(
        """
| DAG | Schedule | Tasks (one API call each) |
|---|---|---|
| `anomaly_detection_incremental_training` | hourly | release the next training batch; train bottle, wood, pill |
| `anomaly_detection_drift_monitoring` | hourly | drift check per category (tab 11) |
""",
        width="content",
    )

    if not api_online:
        live_paused()
    else:
        try:
            waiting = [row for row in fetch_batches() if not row["is_available"]]
        except requests.RequestException as error:
            st.error(f"Could not load the batch status ({short_error(error)}).")
            waiting = None

        if waiting:
            st.info("Live: the next run releases a new batch and retrains.")
        elif waiting is not None:
            st.success("Live: all batches are released; the next run skips training.")

    open_links(("Open Airflow ↗", tool_url(8081)))

    takeaway("Retraining runs only when new data is released.")

# ==========================================================
# 5 · FASTAPI
# ==========================================================

with fastapi_tab, centered():
    with centered("spotlight-fastapi"):
        section("Interface", "A single API for every caller", "setup")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    api [label="FastAPI", fillcolor="#FCEBE3"];
    airflow [label="Airflow"];
    streamlit [label="Streamlit"];
    prometheus [label="Prometheus"];
    postgres [label="PostgreSQL", shape=cylinder];
    model [label="Champion model\\n(MLflow)"];
    evidently [label="Evidently"];
    airflow -> api [label="release · train · drift", color="#3D3B8E", fontcolor="#3D3B8E"];
    streamlit -> api [label="predict", color="#1F8A70", fontcolor="#1F8A70"];
    prometheus -> api [label="metrics", color="#C8632D", fontcolor="#C8632D"];
    api -> postgres;
    api -> model;
    api -> evidently;
}
"""
        )

    st.markdown(
        """
| Endpoint | Caller | Purpose |
|---|---|---|
| `POST /batches/release-next` | Airflow | releases the next training batch |
| `POST /training` | Airflow | trains one category |
| `POST /predict` | Streamlit | returns a verdict for one image |
| `POST /monitoring/drift` | Airflow | runs the drift check of one category |
| `GET /metrics` | Prometheus | exposes request counts and durations |
""",
        width="content",
    )

    if not api_online:
        live_paused()
    else:
        try:
            st.success(f"Live: `GET /` responded in {ping_api():.3f} s.")
        except requests.RequestException as error:
            st.error(f"The API did not respond ({short_error(error)}).")

    open_links(("Open API docs ↗", tool_url(8000, "/docs")))
    takeaway("Every caller goes through one API: one place to measure, secure and change.")

# ==========================================================
# 6 · TENSORFLOW
# ==========================================================

with tensorflow_tab, centered():
    with centered("spotlight-tensorflow"):
        section("Training loop", "Autoencoder trained on normal images only", "training")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    image [label="Image\\n128 x 128"];
    encoder [label="Encoder"];
    code [label="Code\\n8 x 8", shape=ellipse];
    decoder [label="Decoder"];
    rebuilt [label="Reconstruction"];
    error [label="Reconstruction error\\n= anomaly score", fillcolor="#FCEBE3"];
    image -> encoder -> code -> decoder -> rebuilt;
    image -> error;
    rebuilt -> error;
}
"""
        )

    st.markdown(
        """
| Aspect | Choice |
|---|---|
| Training data | released **normal** images of one category |
| Model | convolutional autoencoder (TensorFlow) |
| Anomaly score | reconstruction error of the image |
| Threshold | 95th percentile of the scores of normal training images |
| Evaluation | the fixed test set → `test_auroc` |
""",
        width="content",
    )

    takeaway("No labelled defects are required for training.")

# ==========================================================
# 7 · MLFLOW
# ==========================================================

with mlflow_tab, centered():
    with centered("spotlight-mlflow"):
        section("Training loop", "Promotion only on a higher test AUROC", "training")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    run [label="Finished run"];
    version [label="New version of\\ncae-<category>"];
    compare [label="test_auroc higher\\nthan the champion?", shape=diamond, fillcolor="#FCEBE3"];
    promoted [label="champion alias moves", fillcolor="#E3F2EC"];
    rejected [label="champion retained", fillcolor="#FBE9E7"];
    run -> version -> compare;
    compare -> promoted [label="yes"];
    compare -> rejected [label="no"];
}
"""
        )

    st.markdown(
        """
| Logged per run | Example |
|---|---|
| parameters | epochs, batch size, learning rate |
| metrics | `train_loss`, `val_loss`, `test_auroc` |
| model | a new version of `cae-<category>` |
| provenance | git commit, dataset version |
""",
        width="content",
    )

    open_links(
        ("Open MLflow ↗", tool_url(5001)),
        ("Registered models ↗", tool_url(5001, "/#/models")),
    )

    takeaway("The API serves models:/cae-&lt;category&gt;@champion without a restart.")

# ==========================================================
# 8 · PREDICTION
# ==========================================================

with prediction_tab, centered():
    with centered("spotlight-prediction"):
        section("Serving loop", "Single-image inspection", "serving")

        if not api_online:
            live_paused()
        else:
            category = st.radio(
                "Category",
                CATEGORIES,
                format_func=lambda name: f"{CATEGORY_ICON[name]} {name}",
                horizontal=True,
                label_visibility="collapsed",
            )

            try:
                examples = load_examples(category)
            except requests.RequestException as error:
                st.error(f"Could not load the test examples ({short_error(error)}).")
                examples = None

            if examples == []:
                st.info(
                    f"No test images of {category} are registered yet. "
                    "Load the dataset with `python database/load_data.py`."
                )

            if examples:
                # At least 8 columns, the test images in the middle ones: the
                # thumbnails stay small enough to keep the images, the buttons
                # and the verdict on one screen, and centered with fewer images.
                side = max(0, (8 - len(examples) + 1) // 2)
                columns = st.columns(len(examples) + 2 * side)[side : side + len(examples)]
                chosen = None

                for column, example in zip(columns, examples):
                    with column:
                        image = image_or_none(example["id"])

                        if image is None:
                            st.write("(image not available)")
                        else:
                            st.image(image, width="stretch")

                        label = example["defect_type"]
                        st.markdown(f":green[{label}]" if label == "good" else f":red[{label}]")

                        if st.button(
                            "Check",
                            key=f"predict_example_{example['id']}",
                            type="primary",
                            width="stretch",
                        ):
                            chosen = example

                # Outside the columns: spinner, errors and verdict get the full width.
                if chosen is not None:
                    request_prediction(
                        lambda: requests.post(
                            f"{API_URL}/predict",
                            json={
                                "image_path": chosen["image_path"],
                                "category": category,
                            },
                            timeout=300,
                        ),
                        category=category,
                        image_name="/".join(chosen["image_path"].split("/")[-2:]),
                        true_label=chosen["defect_type"],
                        image_bytes=image_or_none(chosen["id"]),
                    )

    if api_online:
        # Filled at the end, so that a prediction from either button shows at once,
        # right under the test images.
        result_slot = st.container()

        with st.expander("Upload your own image"):
            uploaded_file = st.file_uploader(
                "Image (PNG or JPG)",
                type=["png", "jpg", "jpeg"],
            )

            if st.button(
                "Check this image",
                type="primary",
                key="predict_upload",
                disabled=uploaded_file is None,
            ):
                request_prediction(
                    lambda: requests.post(
                        f"{API_URL}/predict/upload",
                        data={"category": category},
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        },
                        timeout=300,
                    ),
                    category=category,
                    image_name=uploaded_file.name,
                    image_bytes=uploaded_file.getvalue(),
                )

        prediction = st.session_state["last_prediction"]

        # Only the verdict of the selected category: after a switch, an older
        # verdict must not look like the verdict of the new category's images.
        if prediction is not None and prediction["category"] == category:
            with result_slot:
                show_prediction(prediction)

    takeaway("Anomaly: score above the threshold (ratio above 1.00 x).")

# ==========================================================
# 9 · STREAMLIT
# ==========================================================

with streamlit_tab, centered():
    with centered("spotlight-streamlit"):
        section("Serving loop", "Presentation layer: an API client only", "serving")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    browser [label="Browser", shape=oval, fillcolor="#ffffff"];
    streamlit [label="Streamlit\\n(this page)"];
    api [label="FastAPI", fillcolor="#FCEBE3"];
    rest [label="PostgreSQL · model · MLflow · Evidently"];
    browser -> streamlit [label="HTTP"];
    streamlit -> api [label="HTTP (API_URL)", color="#1F8A70", fontcolor="#1F8A70"];
    api -> rest;
}
"""
        )

    st.markdown(
        """
| Part | File |
|---|---|
| Page | `frontend/app.py` |
| SPOTLIGHT walkthrough | `frontend/spotlight.py`, `frontend/spotlight/` |
| Regression check | `frontend/check_app.py` |
""",
        width="content",
    )

    takeaway("No model or database access in the frontend: UI changes cannot affect predictions.")

# ==========================================================
# 10 · PROMETHEUS
# ==========================================================

with prometheus_tab, centered():
    with centered("spotlight-prometheus"):
        section("Monitoring loop", "API health and alerting", "monitoring")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    api [label="FastAPI\\n/metrics", fillcolor="#FCEBE3"];
    prometheus [label="Prometheus"];
    alerts [label="3 alert rules", shape=note, fillcolor="#ffffff"];
    grafana [label="Grafana\\n(tab 12)"];
    api -> prometheus [label="every 15 s", color="#C8632D", fontcolor="#C8632D"];
    prometheus -> alerts;
    prometheus -> grafana;
}
"""
        )

        if not api_online:
            live_paused()
        else:
            try:
                samples = parse_metrics(fetch_metrics())
            except requests.RequestException as error:
                st.error(f"Could not read the metrics ({short_error(error)}).")
                samples = []

            if samples:
                counts = [
                    (labels, value)
                    for name, labels, value in samples
                    if name == "http_request_duration_seconds_count"
                ]
                # Prediction time only: a training call takes minutes by design, and
                # an average over all routes would look like a slow API.
                predictions = sum(value for labels, value in counts if labels.get("route") == "/predict")
                prediction_seconds = sum(
                    value
                    for name, labels, value in samples
                    if name == "http_request_duration_seconds_sum" and labels.get("route") == "/predict"
                )

                total = sum(value for _, value in counts)
                errors = sum(value for labels, value in counts if labels.get("status", "").startswith("5"))

                metric_row(
                    ("Requests (traffic)", f"{total:.0f}"),
                    ("Server errors (5xx)", f"{errors:.0f}"),
                    (
                        "Average prediction time",
                        f"{prediction_seconds / predictions:.3f} s" if predictions else "—",
                    ),
                )

    st.markdown(
        """
| Alert | Fires when (for 1 minute) |
|---|---|
| `AnomalyApiDown` | Prometheus cannot read `/metrics` |
| `AnomalyApiHighErrorRate` | more than 5 % of the requests end with HTTP 5xx |
| `AnomalyApiSlowPredictions` | 95th percentile of `/predict` above 2 s |
""",
        width="content",
    )

    open_links(
        ("Open Prometheus ↗", tool_url(9090)),
        ("Alerts ↗", tool_url(9090, "/alerts")),
        ("Targets ↗", tool_url(9090, "/targets")),
    )
    takeaway("Three alert rules watch the API: down, more than 5 % errors, predictions slower than 2 s.")

# ==========================================================
# 11 · HOLDING QUALITY EVIDENCE
# ==========================================================

with quality_tab, centered():
    with centered("spotlight-quality"):
        section("Monitoring loop", "Evidence that input quality holds", "monitoring")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    logs [label="prediction_logs\\nlast 24 h", shape=cylinder];
    reference [label="training images", shape=cylinder];
    evidently [label="Evidently\\nK-S test per feature", fillcolor="#FCEBE3"];
    reports [label="drift_reports\\n(stored evidence)", shape=cylinder];
    grafana [label="Grafana\\n(tab 12)"];
    logs -> evidently [label="current", color="#C8632D", fontcolor="#C8632D"];
    reference -> evidently [label="reference"];
    evidently -> reports -> grafana;
}
"""
        )

    feature_results = None

    if not api_online:
        live_paused()
    else:
        drift_category = st.radio(
            "Category to check",
            CATEGORIES,
            format_func=lambda name: f"{CATEGORY_ICON[name]} {name}",
            horizontal=True,
            key="drift_category",
        )

        if st.button("Run Evidently", key="run_drift", type="primary"):
            try:
                # The first check of a category after an API start reads every training
                # image (up to 30 s); later checks reuse the cached image features.
                with st.spinner("Evidently compares the feature distributions (first check after a restart: up to 30 s)..."):
                    response = requests.post(
                        f"{API_URL}/monitoring/drift",
                        json={"category": drift_category},
                        timeout=600,
                    )
            except requests.RequestException as error:
                st.error(f"Drift check failed: the API did not respond ({short_error(error)}).")
                response = None

            if response is not None and response.status_code != 200:
                st.error(f"Drift check failed (HTTP {response.status_code}): {error_detail(response)}")
            elif response is not None:
                report = response.json()

                if report["status"] == "skipped":
                    st.warning(
                        f"Skipped and not stored: {report['reason']} "
                        f"(predictions in the last 24 h: {report['current_rows']})"
                    )
                else:
                    metric_row(
                        ("Dataset drift", "Yes" if report["dataset_drift"] else "No"),
                        ("Drifted features", f"{report['drifted_features']} / {report['total_features']}"),
                        ("Predictions compared", report["current_rows"]),
                    )

                    feature_results = report["features"]

    st.markdown(feature_table(feature_results), width="content")

    takeaway("Drift is flagged when at least half of the features drift (p &lt; 0.05); every completed check is stored as evidence.")

# ==========================================================
# 12 · GRAFANA
# ==========================================================

with grafana_tab, centered():
    with centered("spotlight-grafana"):
        section("Monitoring loop", "Operational overview: two dashboards", "monitoring")

        diagram(
            """
digraph {
    rankdir=LR;
"""
            + GRAPH_STYLE
            + """
    arrive [label="Data release\\n(Airflow)"];
    learn [label="Model training\\n(TensorFlow)"];
    judge [label="Champion serving\\n(MLflow)"];
    watch [label="Monitoring\\n(Prometheus · Evidently)", fillcolor="#FCEBE3"];
    arrive -> learn [color="#3D3B8E"];
    learn -> judge [color="#3D3B8E"];
    judge -> watch [color="#1F8A70"];
    watch -> arrive [label="drift → decision on new data", color="#C8632D", fontcolor="#C8632D", style=dashed, constraint=false];
}
"""
        )

    st.markdown(
        """
| Dashboard | Data source | Shows |
|---|---|---|
| **API / Application Health** | Prometheus (tab 10) | status, requests per second, response times, errors, CPU, memory |
| **Data Drift (Evidently)** | PostgreSQL (tab 11) | dataset drift, drifted features, per-feature results, history |
""",
        width="content",
    )

    open_links(
        ("Open Grafana ↗", tool_url(3000)),
        ("Dashboards ↗", tool_url(3000, "/dashboards")),
    )

    takeaway("Monitoring shows when new data is needed; retraining follows the next data release.")
