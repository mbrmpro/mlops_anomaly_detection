"""Train and evaluate category-specific convolutional autoencoders.

The structure follows the exploratory CAE notebook, while data paths and
labels are read from PostgreSQL for the MLOps application.
"""

from __future__ import annotations

# ==========================================================
# IMPORTS
# ==========================================================

import argparse
import json
import os
from pathlib import Path

import numpy as np
import psycopg2
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

import mlflow
import mlflow.tensorflow

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# ==========================================================
# CONFIG
# ==========================================================

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 30
VAL_SIZE = 0.20
CALIBRATION_SIZE = 0.30
RANDOM_STATE = 42
PROJECT_CATEGORIES = ("bottle", "wood", "pill")
LEARNING_RATE = 1e-3

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "anomaly-detection-cae",
)

# ==========================================================
# MLFLOW CONFIGURATION
# ==========================================================


def configure_mlflow():
    """Configure MLflow experiment tracking."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    mlflow.tensorflow.autolog(
        log_models=False,
        log_datasets=False,
        log_every_epoch=True,
        checkpoint=False,
    )


# ==========================================================
# POSTGRESQL CONNECTION AND QUERIES
# ==========================================================


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "anomaly_db"),
        user=os.getenv("POSTGRES_USER", "anomaly_user"),
        password=os.getenv("POSTGRES_PASSWORD", "anomaly_password"),
    )


def get_categories():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT category
                FROM images
                WHERE split = 'train' AND is_anomaly = FALSE
                ORDER BY category;
                """)
            return [row[0] for row in cursor.fetchall()]


def get_train_paths(category):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT image_path
                FROM images
                WHERE category = %s
                  AND split = 'train'
                  AND is_anomaly = FALSE
                ORDER BY image_path;
                """,
                (category,),
            )
            return [row[0] for row in cursor.fetchall()]


def get_test_data(category):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT image_path, is_anomaly
                FROM images
                WHERE category = %s AND split = 'test'
                ORDER BY image_path;
                """,
                (category,),
            )
            rows = cursor.fetchall()

    paths = np.asarray([row[0] for row in rows])
    labels = np.asarray([int(row[1]) for row in rows], dtype=np.int32)
    return paths, labels


# ==========================================================
# IMAGE LOADING
# ==========================================================


def decode_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return tf.cast(image, tf.float32) / 255.0


def autoencoder_parser(path):
    image = decode_image(path)
    return image, image


def image_parser(path):
    return decode_image(path)


# ==========================================================
# DATASETS
# ==========================================================


def build_autoencoder_dataset(paths, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        dataset = dataset.shuffle(
            len(paths), seed=RANDOM_STATE, reshuffle_each_iteration=True
        )
    return (
        dataset.map(autoencoder_parser, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_image_dataset(paths):
    return (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(image_parser, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


# ==========================================================
# CONVOLUTIONAL AUTOENCODER
# ==========================================================


def build_cae(img_size=IMG_SIZE):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 128 -> 64

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 64 -> 32

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 32 -> 16

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    encoded = layers.MaxPooling2D(2)(x)  # 16 -> 8

    x = layers.Conv2D(512, 3, padding="same")(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return models.Model(inputs, outputs, name="convolutional_autoencoder")


# ==========================================================
# LOSS: MSE + SSIM
# ==========================================================


def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def hybrid_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.7 * mse + 0.3 * ssim_loss(y_true, y_pred)


# ==========================================================
# ANOMALY SCORES: MSE + L1 + (1 - SSIM) + BLUR DIFFERENCE
# ==========================================================


def blur(images):
    return tf.nn.avg_pool(images, ksize=3, strides=1, padding="SAME")


def compute_scores(model, dataset):
    components = {"mse": [], "l1": [], "ssim": [], "blur": [], "hybrid": []}

    for batch in dataset:
        reconstruction = model(batch, training=False)
        mse = tf.reduce_mean(tf.square(batch - reconstruction), axis=(1, 2, 3))
        l1 = tf.reduce_mean(tf.abs(batch - reconstruction), axis=(1, 2, 3))
        ssim = tf.image.ssim(batch, reconstruction, max_val=1.0)
        blur_difference = tf.reduce_mean(
            tf.abs(blur(batch) - blur(reconstruction)), axis=(1, 2, 3)
        )
        hybrid = 0.4 * mse + 0.2 * l1 + 0.2 * (1.0 - ssim) + 0.2 * blur_difference

        components["mse"].extend(mse.numpy())
        components["l1"].extend(l1.numpy())
        components["ssim"].extend(ssim.numpy())
        components["blur"].extend(blur_difference.numpy())
        components["hybrid"].extend(hybrid.numpy())

    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in components.items()
    }


# ==========================================================
# THRESHOLD OPTIMIZER
# ==========================================================


def find_best_threshold(labels, scores):
    """Optimize F1 on calibration data, never on final holdout data."""
    best_threshold = float(scores[0])
    best_f1 = -1.0
    for threshold in np.unique(scores):
        predictions = (scores > threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


# ==========================================================
# FINAL EVALUATION
# ==========================================================


def evaluate(labels, scores, threshold):
    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ==========================================================
# TRAIN ONE CATEGORY
# ==========================================================


def _train_category(category, epochs=EPOCHS, save_model=True):
    print(f"\n========== TRAINING: {category} ==========")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    all_train_paths = np.asarray(get_train_paths(category))
    all_test_paths, all_test_labels = get_test_data(category)

    if len(all_train_paths) < 2:
        raise ValueError(f"Not enough training images for category '{category}'")
    if len(all_test_paths) < 4 or len(np.unique(all_test_labels)) < 2:
        raise ValueError(f"Insufficient labeled test data for category '{category}'")

    train_paths, validation_paths = train_test_split(
        all_train_paths,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )
    calibration_paths, holdout_paths, calibration_labels, holdout_labels = (
        train_test_split(
            all_test_paths,
            all_test_labels,
            train_size=CALIBRATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=all_test_labels,
        )
    )

    print(
        f"Train: {len(train_paths)} | Validation: {len(validation_paths)} | "
        f"Calibration: {len(calibration_paths)} | Holdout: {len(holdout_paths)}"
    )

    mlflow.log_params(
        {
            "category": category,
            "model_type": "convolutional_autoencoder",
            "image_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs_requested": epochs,
            "learning_rate": LEARNING_RATE,
            "validation_fraction": VAL_SIZE,
            "calibration_fraction": CALIBRATION_SIZE,
            "random_state": RANDOM_STATE,
            "train_images": len(train_paths),
            "validation_images": len(validation_paths),
            "calibration_images": len(calibration_paths),
            "holdout_images": len(holdout_paths),
        }
    )

    train_dataset = build_autoencoder_dataset(train_paths, shuffle=True)
    validation_dataset = build_autoencoder_dataset(validation_paths)
    calibration_dataset = build_image_dataset(calibration_paths)
    holdout_dataset = build_image_dataset(holdout_paths)

    model = build_cae(IMG_SIZE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=hybrid_loss,
    )
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss", patience=8, restore_best_weights=True
        # ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("\nOptimizing threshold on calibration data...")
    calibration_scores = compute_scores(model, calibration_dataset)
    threshold, calibration_f1 = find_best_threshold(
        calibration_labels, calibration_scores["hybrid"]
    )

    print("Evaluating once on untouched holdout data...")
    holdout_scores = compute_scores(model, holdout_dataset)
    metrics = evaluate(holdout_labels, holdout_scores["hybrid"], threshold)

    print("\n========== FINAL EVALUATION ==========")
    print(f"Threshold:         {threshold:.6f}")
    print(f"Calibration F1:    {calibration_f1:.4f}")
    print(f"Holdout accuracy:  {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1:                {metrics['f1']:.4f}")
    print(f"AUROC:             {metrics['auroc']:.4f}")
    print(
        f"TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )

    model_dir = Path("models") / category
    result = {
        "category": category,
        "mlflow_run_id": mlflow.active_run().info.run_id,
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(history.history["loss"])),
        "threshold": float(threshold),
        "threshold_source": "labeled_calibration_split",
        "calibration_fraction": CALIBRATION_SIZE,
        "calibration_f1": calibration_f1,
        "holdout_metrics": metrics,
        "model_saved": bool(save_model),
    }

    mlflow.log_metrics(
        {
            "epochs_completed": result["epochs_completed"],
            "best_validation_loss": float(min(history.history["val_loss"])),
            "anomaly_threshold": float(threshold),
            "calibration_f1": float(calibration_f1),
            "holdout_accuracy": metrics["accuracy"],
            "holdout_balanced_accuracy": metrics["balanced_accuracy"],
            "holdout_precision": metrics["precision"],
            "holdout_recall": metrics["recall"],
            "holdout_f1": metrics["f1"],
            "holdout_auroc": metrics["auroc"],
            "holdout_tn": metrics["tn"],
            "holdout_fp": metrics["fp"],
            "holdout_fn": metrics["fn"],
            "holdout_tp": metrics["tp"],
        }
    )

    mlflow.log_dict(
        result,
        "reports/evaluation.json",
    )

    mlflow.log_dict(
        {
            "category": category,
            "threshold": float(threshold),
            "threshold_source": "labeled_calibration_split",
            "calibration_fraction": CALIBRATION_SIZE,
            "calibration_f1": float(calibration_f1),
        },
        "reports/threshold.json",
    )

    if save_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "cae.keras"
        threshold_path = model_dir / "threshold.json"
        evaluation_path = model_dir / "evaluation.json"
        model.save(model_path)
        threshold_path.write_text(
            json.dumps(
                {
                    "category": category,
                    "threshold": float(threshold),
                    "threshold_source": "labeled_calibration_split",
                    "calibration_fraction": CALIBRATION_SIZE,
                    "calibration_f1": calibration_f1,
                },
                indent=4,
            )
        )
        evaluation_path.write_text(json.dumps(result, indent=4))
        print(f"Model:             {model_path}")
        print(f"Threshold:         {threshold_path}")
        print(f"Evaluation:        {evaluation_path}")

    return result


def train(category, epochs=EPOCHS, save_model=True):
    """Train one category inside a dedicated MLflow run."""

    configure_mlflow()

    with mlflow.start_run(run_name=f"cae-{category}") as run:
        mlflow.set_tags(
            {
                "category": category,
                "pipeline_stage": "training",
                "model_family": "CAE",
                "candidate_status": "candidate",
            }
        )

        print(f"MLflow run: {run.info.run_id}")

        return _train_category(
            category=category,
            epochs=epochs,
            save_model=save_model,
        )


# ==========================================================
# MAIN
# ==========================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "categories",
        nargs="*",
        default=list(PROJECT_CATEGORIES),
        help="Categories to train. Defaults to bottle wood pill.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train without replacing saved model artifacts.",
    )
    args = parser.parse_args()

    categories = args.categories or list(PROJECT_CATEGORIES)
    available_categories = set(get_categories())
    missing = [
        category for category in categories if category not in available_categories
    ]
    if missing:
        raise ValueError("Missing categories in PostgreSQL: " + ", ".join(missing))

    print("Categories:", ", ".join(categories))
    results = [
        train(category, epochs=args.epochs, save_model=not args.no_save)
        for category in categories
    ]

    if not args.no_save:
        summary_path = Path("models") / "training_summary.json"
        summary_path.write_text(json.dumps(results, indent=4))
        print(f"Training summary:  {summary_path}")


if __name__ == "__main__":
    main()
