from pathlib import Path

import argparse

import json

import numpy as np

import psycopg2

import tensorflow as tf

from tensorflow.keras import layers, models

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split


IMG_SIZE = 128

BATCH_SIZE = 16

EPOCHS = 30

VAL_SIZE = 0.2

THRESHOLD_PERCENTILE = 95

RANDOM_STATE = 42


def get_connection():

    return psycopg2.connect(

        host="localhost",

        port=5432,

        dbname="anomaly_db",

        user="anomaly_user",

        password="anomaly_password"

    )


def get_categories():

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute("""

                SELECT DISTINCT category

                FROM images

                WHERE split='train' AND is_anomaly=FALSE

                ORDER BY category;

            """)

            return [row[0] for row in cur.fetchall()]


def get_train_paths(category):

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute("""

                SELECT image_path

                FROM images

                WHERE category=%s AND split='train' AND is_anomaly=FALSE

                ORDER BY image_path;

            """, (category,))

            return [row[0] for row in cur.fetchall()]


def get_test_data(category):

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute("""

                SELECT image_path, is_anomaly

                FROM images

                WHERE category=%s AND split='test'

                ORDER BY image_path;

            """, (category,))

            rows = cur.fetchall()

    paths = [row[0] for row in rows]

    labels = np.array([int(row[1]) for row in rows])

    return paths, labels


def decode_image(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_image(img, channels=3, expand_animations=False)

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    return tf.cast(img, tf.float32) / 255.0


def autoencoder_parser(path):

    img = decode_image(path)

    return img, img


def image_parser(path):

    return decode_image(path)


def build_autoencoder_dataset(paths, shuffle=False):

    ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle:

        ds = ds.shuffle(len(paths), seed=RANDOM_STATE)

    return ds.map(

        autoencoder_parser,

        num_parallel_calls=tf.data.AUTOTUNE

    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_image_dataset(paths):

    return tf.data.Dataset.from_tensor_slices(paths).map(

        image_parser,

        num_parallel_calls=tf.data.AUTOTUNE

    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_cae():

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    x = inputs

    for filters in [32, 64, 128, 256]:

        x = layers.Conv2D(filters, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)

        x = layers.ReLU()(x)

        x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(512, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    for filters in [256, 128, 64, 32]:

        x = layers.Conv2DTranspose(

            filters,

            3,

            strides=2,

            padding="same"

        )(x)

        x = layers.ReLU()(x)

    outputs = layers.Conv2D(

        3,

        3,

        activation="sigmoid",

        padding="same"

    )(x)

    return models.Model(inputs, outputs)


def ssim_loss(y_true, y_pred):

    return 1 - tf.reduce_mean(

        tf.image.ssim(y_true, y_pred, max_val=1.0)

    )


def hybrid_loss(y_true, y_pred):

    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    return 0.7 * mse + 0.3 * ssim_loss(y_true, y_pred)


def blur(img):

    return tf.nn.avg_pool(

        img,

        ksize=3,

        strides=1,

        padding="SAME"

    )


def compute_scores(model, dataset):

    scores = []

    for batch in dataset:

        reconstruction = model.predict(batch, verbose=0)

        batch_np = batch.numpy()

        mse = np.mean(

            np.square(batch_np - reconstruction),

            axis=(1, 2, 3)

        )

        l1 = np.mean(

            np.abs(batch_np - reconstruction),

            axis=(1, 2, 3)

        )

        ssim = tf.image.ssim(

            batch,

            reconstruction,

            max_val=1.0

        ).numpy()

        blur_diff = np.mean(

            np.abs(

                blur(batch).numpy()

                - blur(reconstruction).numpy()

            ),

            axis=(1, 2, 3)

        )

        hybrid = (

            0.4 * mse

            + 0.2 * l1

            + 0.2 * (1 - ssim)

            + 0.2 * blur_diff

        )

        scores.extend(hybrid)

    return np.array(scores)


def train(category, epochs=EPOCHS, save_model=True):

    print(f"\n========== TRAINING: {category} ==========")

    all_train_paths = get_train_paths(category)

    test_paths, test_labels = get_test_data(category)

    if len(all_train_paths) < 2:

        raise ValueError(

            f"Not enough training images for category '{category}'"

        )

    if not test_paths:

        raise ValueError(

            f"No test images found for category '{category}'"

        )

    train_paths, val_paths = train_test_split(

        all_train_paths,

        test_size=VAL_SIZE,

        random_state=RANDOM_STATE

    )

    print(

        f"Train: {len(train_paths)} | "

        f"Validation: {len(val_paths)} | "

        f"Test: {len(test_paths)}"

    )

    train_ds = build_autoencoder_dataset(

        train_paths,

        shuffle=True

    )

    val_train_ds = build_autoencoder_dataset(val_paths)

    val_score_ds = build_image_dataset(val_paths)

    test_ds = build_image_dataset(test_paths)

    model = build_cae()

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-3

        ),

        loss=hybrid_loss

    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            min_lr=1e-6

        )

    ]

    model.fit(

        train_ds,

        validation_data=val_train_ds,

        epochs=epochs,

        callbacks=callbacks

    )

    print("\nCalculating validation threshold...")

    val_scores = compute_scores(model, val_score_ds)

    threshold = float(

        np.percentile(

            val_scores,

            THRESHOLD_PERCENTILE

        )

    )

    print("Evaluating test dataset...")

    test_scores = compute_scores(model, test_ds)

    predictions = (test_scores > threshold).astype(int)

    f1 = f1_score(

        test_labels,

        predictions,

        zero_division=0

    )

    print(f"Threshold: {threshold:.6f}")

    print(f"Test F1:   {f1:.4f}")

    if save_model:

        model_dir = Path("models") / category

        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "cae.keras"

        threshold_path = model_dir / "threshold.json"

        model.save(model_path)

        with open(threshold_path, "w") as file:

            json.dump({

                "category": category,

                "threshold": threshold,

                "threshold_percentile": THRESHOLD_PERCENTILE,

                "test_f1": float(f1)

            }, file, indent=4)

        print(f"Model:     {model_path}")

        print(f"Threshold: {threshold_path}")

    else:

        print("Model not saved (smoke test).")

    return {

        "category": category,

        "epochs": epochs,

        "threshold": threshold,

        "f1": float(f1),

        "model_saved": save_model

    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "category",

        nargs="?",

        help="Category to train. If omitted, all DB categories are trained."

    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=EPOCHS

    )

    args = parser.parse_args()

    categories = [args.category] if args.category else get_categories()

    if not categories:

        raise ValueError("No training categories found in PostgreSQL.")

    print("Categories:", ", ".join(categories))

    for category in categories:

        train(category, epochs=args.epochs)


if __name__ == "__main__":

    main()
