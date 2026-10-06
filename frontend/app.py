"""Streamlit frontend: the anomaly detection project told as a short story.

Streamlit is only a client. Everything on this page comes from the FastAPI
service: system status, training data, test images and predictions.
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from spotlight import render_spotlight


# ==========================================================
# CONFIG
# ==========================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Same categories as src/training.py (PROJECT_CATEGORIES).
CATEGORIES = ("bottle", "wood", "pill")

CATEGORY_TEXT = {
    "bottle": "Glass bottles seen from above. Typical defects: broken glass, contamination.",
    "wood": "Wooden surfaces. Typical defects: holes, scratches, stains, liquid.",
    "pill": "Pills. Typical defects: cracks, contamination, faulty imprint, wrong color.",
}

CATEGORY_ICON = {
    "bottle": "🍾",
    "wood": "🪵",
    "pill": "💊",
}


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


@st.cache_data(ttl=60, show_spinner=False)
def load_examples(category):
    return api_get("/images/examples", category=category).json()["examples"]


@st.cache_data(show_spinner=False)
def load_image(image_id):
    return api_get(f"/images/{image_id}/file").content


def is_correct(result, true_label):
    return (result["prediction"] == "anomaly") == (true_label != "good")


def request_prediction(send_request, source, image_name, category, true_label=None):
    """Call the API, keep the answer in the session and show errors."""

    start = time.perf_counter()

    try:
        with st.spinner("The API is checking the image..."):
            response = send_request()
    except requests.RequestException as error:
        st.error(f"Prediction failed: API not reachable ({error})")
        return

    if response.status_code != 200:
        st.error(f"Prediction failed (HTTP {response.status_code}): {error_detail(response)}")
        return

    result = response.json()

    st.session_state["last_prediction"] = {
        "result": result,
        "image": image_name,
        "true_label": true_label,
        "seconds": time.perf_counter() - start,
    }

    st.session_state["history"].append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "category": category,
            "image": image_name,
            "source": source,
            "prediction": result["prediction"],
            "anomaly_score": round(result["anomaly_score"], 6),
            "threshold": round(result["threshold"], 6),
            "true_label": true_label or "-",
            "correct": (
                "-"
                if true_label is None
                else ("yes" if is_correct(result, true_label) else "no")
            ),
        }
    )


# ==========================================================
# RESULT CARD
# ==========================================================

def show_prediction(prediction):
    result = prediction["result"]
    score = result["anomaly_score"]
    threshold = result["threshold"]

    with st.container(border=True):
        st.success(
            f"The API answered in {prediction['seconds']:.2f} s "
            f"for {prediction['image']} ({result['category']})."
        )

        if result["prediction"] == "anomaly":
            st.markdown("## Verdict: :red[ANOMALY]")
        else:
            st.markdown("## Verdict: :green[NORMAL]")

        score_column, threshold_column, ratio_column = st.columns(3)

        score_column.metric(
            "Anomaly score",
            f"{score:.6f}",
            delta=f"{score - threshold:+.6f} vs threshold",
            delta_color="inverse",
        )

        threshold_column.metric("Threshold", f"{threshold:.6f}")

        ratio_column.metric("Score / threshold", f"{score / threshold:.2f} x")

        st.caption(
            "The image is an anomaly when its score is above the threshold "
            "(ratio above 1.00 x)."
        )

        true_label = prediction["true_label"]

        if true_label is not None:
            truth = "a normal product" if true_label == "good" else f"a defect ({true_label})"

            if is_correct(result, true_label):
                st.markdown(f"**Ground truth:** {truth} → the model was :green[right].")
            else:
                st.markdown(f"**Ground truth:** {truth} → the model was :red[wrong].")

        # Shown when the API serves an MLflow registered model version.
        if "model_version" in result:
            st.markdown(
                f"**Model:** `{result.get('model_name')}` version "
                f"`{result['model_version']}`"
            )

        with st.expander("Full API response"):
            st.json(result)


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)

st.session_state.setdefault("history", [])
st.session_state.setdefault("last_prediction", None)

# ----------------------------------------------------------
# Sidebar: live system status
# ----------------------------------------------------------

with st.sidebar.container(key="spotlight-system-status"):
    st.header("System status")

    try:
        api_get("/")
        st.success("API: connected")
        api_ok = True
    except requests.RequestException as error:
        st.error(f"API: not reachable\n\n{error}")
        api_ok = False

    if api_ok:
        try:
            api_get("/health/db")
            st.success("Database: connected")
        except requests.RequestException:
            st.error("Database: not reachable")

    st.caption(f"API URL: {API_URL}")
    st.caption("Everything on this page is read from the API.")

# ----------------------------------------------------------
# Title
# ----------------------------------------------------------

with st.container(key="spotlight-title"):
    st.title("🔍 Finding defects without ever seeing one")

    st.markdown(
        "A neural network learns what a **normal** product looks like. "
        "Anything it cannot reproduce well is suspicious. "
        "Follow the story tab by tab, then test the model yourself."
    )

    # Optional guided walkthrough of this page (presentation only).
    render_spotlight()

challenge_tab, data_tab, mlflow_tab, try_tab, decision_tab, system_tab = st.tabs(
    [
        "1 · The challenge",
        "2 · The data",
        "3 · MLflow",
        "4 · Try the model",
        "5 · How it decides",
        "6 · Behind the scenes",
    ],
    key="story-tabs",
)

# ==========================================================
# 1 · THE CHALLENGE
# ==========================================================

with challenge_tab:
    with st.container(key="spotlight-challenge"):
        st.header("The challenge")

        st.markdown(
            """
In a factory most products are fine. Defects are **rare**, and every defect
looks different: a crack, a stain, a missing piece. Collecting labeled
examples of every possible defect is almost impossible.

So the model is trained **only on normal images**. It learns to rebuild them.
When it sees a defect, the rebuilt image differs from the original —
that difference is the **anomaly score**.
"""
        )

        columns = st.columns(len(CATEGORIES))

        for column, category in zip(columns, CATEGORIES):
            with column.container(border=True):
                st.subheader(f"{CATEGORY_ICON[category]} {category}")
                st.write(CATEGORY_TEXT[category])

        st.subheader("The idea: a convolutional autoencoder")

        st.graphviz_chart(
            """
digraph {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
    image [label="Input image\\n128 x 128"];
    encoder [label="Encoder\\n(compress)"];
    code [label="Compact code", shape=ellipse];
    decoder [label="Decoder\\n(rebuild)"];
    rebuilt [label="Rebuilt image"];
    error [label="Reconstruction error\\n= anomaly score", fillcolor="#fdf1dc"];
    decision [label="score > threshold ?\\nANOMALY : NORMAL", fillcolor="#e6f4ea"];
    image -> encoder -> code -> decoder -> rebuilt;
    image -> error;
    rebuilt -> error -> decision;
}
"""
        )

# ==========================================================
# 2 · THE DATA
# ==========================================================

with data_tab:
    with st.container(key="spotlight-data"):
        st.header("The data arrives in batches")

        st.markdown(
            """
The images come from the **MVTec AD** dataset. For every category:

* a **fixed test set** (normal and defective images) measures every model the same way;
* the normal training images are split into **three batches** that "arrive" over time.

Airflow releases one batch per run and retrains the models on everything released so far
(batch 1 → batches 1 + 2 → batches 1 + 2 + 3).
The numbers below are read live from the API (`GET /batches/status`).
"""
        )

        try:
            batches = pd.DataFrame(api_get("/batches/status").json()["batches"])
        except requests.RequestException as error:
            st.error(f"Could not load the batch status: {error}")
            batches = pd.DataFrame()

        if batches.empty:
            st.info("No training data registered yet. Run database/load_data.py first.")
        else:
            batches["status"] = batches["is_available"].map(
                {True: "released", False: "waiting"}
            )

            released = batches.loc[batches["is_available"], "image_count"].sum()
            total = batches["image_count"].sum()
            released_batches = batches.loc[batches["is_available"], "batch_id"].nunique()

            first, second, third = st.columns(3)
            first.metric("Training images released", f"{released} / {total}")
            second.metric("Batches released", f"{released_batches} / {batches['batch_id'].nunique()}")
            third.metric("Categories", batches["category"].nunique())

            chart = batches.pivot_table(
                index="category",
                columns="status",
                values="image_count",
                aggfunc="sum",
                fill_value=0,
            )

            st.bar_chart(chart, x_label="category", y_label="training images")

            st.dataframe(
                batches[["category", "batch_id", "status", "image_count"]],
                hide_index=True,
            )

# ==========================================================
# 3 · MLFLOW
# ==========================================================

with mlflow_tab:
    with st.container(key="spotlight-mlflow"):
        st.header("Training, recorded by MLflow")

        st.markdown(
            """
The autoencoder learns by **looping over pixels**. In every **epoch** it rebuilds
each released normal image (128 x 128 pixels, in batches of 16) and compares the
rebuilt image with the original, pixel by pixel. The difference is one number, the
**training loss**. The weights then change a little to make it smaller, and the
loop starts again. MLflow records every step of this loop.
"""
        )

        st.graphviz_chart(
            """
digraph {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
    images [label="Normal images\\n128 x 128, batches of 16"];
    rebuild [label="Autoencoder\\nrebuilds every pixel"];
    loss [label="Training loss\\n0.7 MSE + 0.3 (1 - SSIM)", fillcolor="#fdf1dc"];
    update [label="Adjust the weights"];
    mlflow [label="MLflow\\ntrain_loss and val_loss", fillcolor="#e6f4ea"];
    images -> rebuild -> loss -> update;
    update -> rebuild [label="next epoch"];
    loss -> mlflow [label="every epoch"];
}
"""
        )

        st.markdown("**Training loss** — how far the rebuilt pixels are from the original:")

        st.latex(r"\text{loss} = 0.7\,\text{MSE} + 0.3\,(1 - \text{SSIM})")

        st.markdown(
            """
| Part | What it measures |
|---|---|
| MSE | average squared difference over all pixels |
| 1 − SSIM | lost structure: edges and textures |

This loss is used only to **train** the model. The anomaly score in tab 5 is a
separate formula that judges a finished image.
"""
        )

        st.subheader("One training run")

        st.markdown(
            """
| Setting | Value |
|---|---|
| Images | released normal images of one category, 128 x 128 |
| Batch size | 16 images per weight update |
| Epochs | 30 passes over all training images (default) |
| Validation | 20 % of the images, never used to update the weights |
| Learning rate | 0.001, halved when the validation loss stops improving for 4 epochs |
"""
        )

        st.subheader("What MLflow keeps for every run")

        st.markdown(
            """
| What | Where in MLflow | Content |
|---|---|---|
| Settings | parameters | `epochs_requested`, `threshold_percentile`, ... |
| Learning curve | metrics, one value per epoch | `train_loss`, `val_loss` |
| Results on the fixed test set | metrics | `test_auroc`, `test_f1`, `test_precision`, `test_recall`, `anomaly_threshold` |
| Reports | artifacts | `reports/evaluation.json`, `reports/threshold.json` |
| Training data used | artifact and tag | `dataset/manifest.tsv`, `dataset_version` |
| The trained model | artifact and Model Registry | `model/`, a new version of `cae-<category>` |

Each run is named after its category and the newest released batch, for example
`cae-bottle-batch-2`.
"""
        )

        st.subheader("Champion selection")

        st.markdown(
            """
Every saved run becomes a new version of the registered model `cae-<category>`.
The new version is compared with the current **champion** on the fixed test set:
it takes over the alias `champion` only if its `test_auroc` is strictly higher.
The first model of a category becomes champion directly; on a tie the champion stays.

The API always predicts with `models:/cae-<category>@champion` and loads a newly
promoted champion by itself. No code change and no restart are needed.
"""
        )

        st.subheader("The files behind it")

        st.markdown(
            """
| File | Content |
|---|---|
| `mlflow_data/mlflow.db` | runs, parameters, metrics and the model registry (SQLite) |
| `mlflow_data/artifacts/` | models, reports and data manifests of every run |

Open the MLflow UI at http://localhost:5001, experiment `anomaly-detection-cae`.
"""
        )

# ==========================================================
# 4 · TRY THE MODEL
# ==========================================================

with try_tab:
    with st.container(key="spotlight-try-model"):
        st.header("Try the model")

        st.markdown(
            "Pick a category and an image. Streamlit sends it to the FastAPI service, "
            "which runs the trained model and returns the anomaly score."
        )

        category = st.selectbox(
            "Category",
            CATEGORIES,
            format_func=lambda name: f"{CATEGORY_ICON[name]} {name}",
        )

        source = st.radio(
            "Image source",
            ["Test set example (true label known)", "Upload your own image"],
            horizontal=True,
        )

        if source.startswith("Test set"):
            try:
                examples = load_examples(category)
            except requests.RequestException as error:
                st.error(f"Could not load test examples: {error}")
                examples = []

            if examples:
                st.caption("One image of the fixed test set per defect type:")

                columns = st.columns(len(examples))

                for column, example in zip(columns, examples):
                    with column:
                        try:
                            st.image(load_image(example["id"]), width="stretch")
                        except requests.RequestException:
                            st.write("(image not available)")

                        label = example["defect_type"]
                        st.markdown(f":green[{label}]" if label == "good" else f":red[{label}]")

                chosen = st.radio(
                    "Which image should the model check?",
                    range(len(examples)),
                    format_func=lambda index: examples[index]["defect_type"],
                    horizontal=True,
                )

                example = examples[chosen]

                if st.button("Predict with the API", type="primary", key="predict_example"):
                    request_prediction(
                        lambda: requests.post(
                            f"{API_URL}/predict",
                            json={
                                "image_path": example["image_path"],
                                "category": category,
                            },
                            timeout=300,
                        ),
                        source="test set",
                        image_name=example["image_path"].split("/")[-1],
                        category=category,
                        true_label=example["defect_type"],
                    )

        else:
            uploaded_file = st.file_uploader(
                "Image (PNG or JPG)",
                type=["png", "jpg", "jpeg"],
            )

            if uploaded_file is not None:
                st.image(uploaded_file, caption=uploaded_file.name, width=300)

            if st.button(
                "Predict with the API",
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
                    source="upload",
                    image_name=uploaded_file.name,
                    category=category,
                )

        if st.session_state["last_prediction"] is not None:
            st.subheader("Result")
            show_prediction(st.session_state["last_prediction"])

        history = st.session_state["history"]

        if history:
            st.subheader("Your predictions in this session")

            history_frame = pd.DataFrame(history)
            checked = history_frame[history_frame["correct"] != "-"]

            first, second, third = st.columns(3)
            first.metric("Predictions", len(history_frame))
            second.metric("Anomalies found", int((history_frame["prediction"] == "anomaly").sum()))

            if len(checked):
                third.metric(
                    "Correct on test examples",
                    f"{(checked['correct'] == 'yes').sum()} / {len(checked)}",
                )

            st.dataframe(history_frame.iloc[::-1], hide_index=True)

# ==========================================================
# 5 · HOW IT DECIDES
# ==========================================================

with decision_tab:
    with st.container(key="spotlight-decision"):
        st.header("How the model decides")

        st.markdown("**1. Anomaly score** — how badly the model rebuilt the image:")

        st.latex(
            r"\text{score} = 0.4\,\text{MSE} + 0.2\,\text{L1} "
            r"+ 0.2\,(1 - \text{SSIM}) + 0.2\,\text{BlurDiff}"
        )

        st.markdown(
            """
| Part | What it measures |
|---|---|
| MSE | squared pixel difference — punishes large errors |
| L1 | absolute pixel difference — average error |
| 1 − SSIM | loss of structure (edges, textures) |
| BlurDiff | difference after smoothing — ignores pixel noise |

**2. Threshold** — computed during training: the 95th percentile of the scores of the
normal training images. About 5% of normal training images score above it.

**3. Decision** — `score > threshold` → **ANOMALY**, otherwise **NORMAL**.

A higher threshold gives fewer false alarms but more missed defects;
a lower threshold catches more defects but raises more false alarms.
"""
        )

        last = st.session_state["last_prediction"]

        if last is None:
            st.info("Make a prediction in tab 4 to see these numbers for your own image.")
        else:
            result = last["result"]

            st.markdown(
                f"**Your last image** ({last['image']}): score "
                f"`{result['anomaly_score']:.6f}` vs threshold `{result['threshold']:.6f}` "
                f"= **{result['anomaly_score'] / result['threshold']:.2f} x** the threshold "
                f"→ **{result['prediction'].upper()}**."
            )

# ==========================================================
# 6 · BEHIND THE SCENES
# ==========================================================

with system_tab:
    with st.container(key="spotlight-system"):
        st.header("Behind the scenes")

        st.graphviz_chart(
            """
digraph {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
    browser [label="Browser"];
    streamlit [label="Streamlit\\n(this page)"];
    api [label="FastAPI", fillcolor="#fdf1dc"];
    model [label="predict.py\\nTensorFlow model"];
    postgres [label="PostgreSQL\\nimages + batches", shape=cylinder];
    airflow [label="Airflow\\nbatches + training"];
    mlflow [label="MLflow\\nruns + models"];
    browser -> streamlit -> api;
    api -> model;
    api -> postgres;
    airflow -> api;
    api -> mlflow [label="training"];
}
"""
        )

        st.markdown(
            """
Streamlit never loads a model or reads the database itself. It only calls the API:

| Endpoint | Used for |
|---|---|
| `GET /` and `GET /health/db` | system status in the sidebar |
| `GET /batches/status` | tab 2 — released training batches |
| `GET /images/examples` | tab 4 — one test image per defect type |
| `GET /images/{id}/file` | tab 4 — the test images themselves |
| `POST /predict` | tab 4 — prediction for a test set example |
| `POST /predict/upload` | tab 4 — prediction for an uploaded image |

Other user interfaces of the project:

* FastAPI documentation: http://localhost:8000/docs
* MLflow (experiments and models): http://localhost:5001
* Airflow (batches and training): http://localhost:8081
"""
        )
