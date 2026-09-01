from pathlib import Path

import json

import sys


import numpy as np

import tensorflow as tf



# ==========================================================

# CONFIG

# ==========================================================


IMG_SIZE = 128



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

# LOAD MODEL + THRESHOLD

# ==========================================================


def load_model_and_threshold(category):


    model_dir = Path("models") / category


    model_path = model_dir / "cae.keras"

    threshold_path = model_dir / "threshold.json"


    if not model_path.exists():

        raise FileNotFoundError(

            f"Model not found: {model_path}"

        )


    if not threshold_path.exists():

        raise FileNotFoundError(

            f"Threshold not found: {threshold_path}"

        )


    # compile=False means that the custom training loss

    # is not required for inference.

    model = tf.keras.models.load_model(

        model_path,

        compile=False

    )


    with open(threshold_path, "r") as file:

        threshold_data = json.load(file)


    threshold = threshold_data["threshold"]


    return model, threshold



# ==========================================================

# PREDICTION

# ==========================================================


def predict(image_path, category):


    model, threshold = load_model_and_threshold(

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

        )

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
