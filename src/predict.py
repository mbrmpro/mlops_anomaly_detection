from pathlib import Path

import json

import os

import sys

import tempfile


import mlflow

import numpy as np

import tensorflow as tf

from mlflow import MlflowClient

from mlflow.exceptions import MlflowException



# ==========================================================

# CONFIG

# ==========================================================


IMG_SIZE = 128

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001",
)

# Same registry names as src/training.py: model "cae-<category>",
# alias "champion" on the version used for prediction.
CHAMPION_ALIAS = "champion"

# Champion already loaded per category: {category: (version, model, threshold)}.
# A new champion version is downloaded automatically after a promotion.
_loaded_champions = {}



# ==========================================================

# IMAGE LOADING

# ==========================================================


def decode_image(image_path):

    img = tf.io.read_file(image_path)

    img = tf.image.decode_image(

        img,

        channels=3,

        expand_animations=False

    )

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    return tf.cast(img, tf.float32) / 255.0



# ==========================================================

# ANOMALY SCORE

# ==========================================================


def blur(img):

    return tf.nn.avg_pool(

        img,

        ksize=3,

        strides=1,

        padding="SAME"

    )



def compute_anomaly_score(model, image):


    batch = tf.expand_dims(image, axis=0)


    reconstruction = model.predict(

        batch,

        verbose=0

    )


    batch_np = batch.numpy()


    mse = np.mean(

        np.square(batch_np - reconstruction)

    )


    l1 = np.mean(

        np.abs(batch_np - reconstruction)

    )


    ssim = tf.image.ssim(

        batch,

        reconstruction,

        max_val=1.0

    ).numpy()[0]


    blur_true = blur(batch).numpy()

    blur_recon = blur(reconstruction).numpy()


    blur_diff = np.mean(

        np.abs(blur_true - blur_recon)

    )


    score = (

        0.4 * mse

        + 0.2 * l1

        + 0.2 * (1 - ssim)

        + 0.2 * blur_diff

    )


    return float(score)



# ==========================================================

# LOAD CHAMPION MODEL + THRESHOLD FROM MLFLOW

# ==========================================================


def load_champion(category):
    """Return (model, threshold, version) of the MLflow champion."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = f"cae-{category}"

    try:
        version = MlflowClient().get_model_version_by_alias(
            model_name,
            CHAMPION_ALIAS,
        ).version
    except MlflowException as error:
        raise LookupError(
            f"No champion model in MLflow for '{category}' "
            f"({model_name}@{CHAMPION_ALIAS}). Train a model first."
        ) from error

    loaded = _loaded_champions.get(category)

    if loaded is not None and loaded[0] == version:
        return loaded[1], loaded[2], version

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{model_name}/{version}",
                dst_path=tmp_dir,
            )
        )

        # compile=False means that the custom training loss
        # is not required for inference.
        model = tf.keras.models.load_model(
            model_dir / "cae.keras",
            compile=False,
        )

        threshold = json.loads(
            (model_dir / "threshold.json").read_text()
        )["threshold"]

    _loaded_champions[category] = (version, model, threshold)

    return model, threshold, version



# ==========================================================

# PREDICTION

# ==========================================================


def predict(image_path, category):


    model, threshold, model_version = load_champion(

        category

    )


    image = decode_image(image_path)


    score = compute_anomaly_score(

        model,

        image

    )


    is_anomaly = score > threshold


    result = {

        "category": category,

        "image_path": str(image_path),

        "anomaly_score": score,

        "threshold": threshold,

        "prediction": (

            "anomaly"

            if is_anomaly

            else "normal"

        ),

        "model_name": f"cae-{category}",

        "model_version": model_version,

    }


    return result



# ==========================================================

# MAIN

# ==========================================================


if __name__ == "__main__":


    if len(sys.argv) != 3:

        print(

            "Usage: python src/predict.py "

            "<image_path> <category>"

        )

        sys.exit(1)


    image_path = sys.argv[1]

    category = sys.argv[2]


    result = predict(

        image_path,

        category

    )


    print("\n========== PREDICTION ==========")

    print(f"Category:      {result['category']}")

    print(f"Image:         {result['image_path']}")

    print(

        f"Anomaly score: "

        f"{result['anomaly_score']:.6f}"

    )

    print(

        f"Threshold:     "

        f"{result['threshold']:.6f}"

    )

    print(

        f"Prediction:    "

        f"{result['prediction'].upper()}"

    )

    print(

        f"Model:         "

        f"{result['model_name']} version {result['model_version']}"

    )
