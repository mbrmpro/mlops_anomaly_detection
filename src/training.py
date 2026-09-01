from pathlib import Path

import json


import numpy as np

import psycopg2

import tensorflow as tf

from tensorflow.keras import layers, models

from sklearn.metrics import f1_score



# ==========================================================

# CONFIG

# ==========================================================


IMG_SIZE = 128

BATCH_SIZE = 16

EPOCHS = 2


CATEGORY = "bottle"


MODEL_DIR = Path("models") / CATEGORY

MODEL_DIR.mkdir(parents=True, exist_ok=True)



# ==========================================================

# DATABASE

# ==========================================================


def get_connection():

    return psycopg2.connect(

        host="localhost",

        port=5432,

        dbname="anomaly_db",

        user="anomaly_user",

        password="anomaly_password",

    )



def get_train_paths(category):

    connection = get_connection()

    cursor = connection.cursor()


    query = """

        SELECT image_path

        FROM images

        WHERE category = %s

          AND split = 'train'

          AND is_anomaly = FALSE

        ORDER BY image_path;

    """


    cursor.execute(query, (category,))

    rows = cursor.fetchall()


    cursor.close()

    connection.close()


    return [row[0] for row in rows]



def get_test_data(category):

    connection = get_connection()

    cursor = connection.cursor()


    query = """

        SELECT image_path, is_anomaly

        FROM images

        WHERE category = %s

          AND split = 'test'

        ORDER BY image_path;

    """


    cursor.execute(query, (category,))

    rows = cursor.fetchall()


    cursor.close()

    connection.close()


    paths = [row[0] for row in rows]

    labels = np.array([int(row[1]) for row in rows])


    return paths, labels



# ==========================================================

# IMAGE LOADING

# ==========================================================


def decode_image(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_image(

        img,

        channels=3,

        expand_animations=False

    )


    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))


    return tf.cast(img, tf.float32) / 255.0



def train_parser(path):

    img = decode_image(path)

    return img, img



def test_parser(path):

    return decode_image(path)



def build_train_dataset(paths):

    ds = tf.data.Dataset.from_tensor_slices(paths)


    ds = ds.shuffle(1000)


    ds = ds.map(

        train_parser,

        num_parallel_calls=tf.data.AUTOTUNE

    )


    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



def build_test_dataset(paths):

    ds = tf.data.Dataset.from_tensor_slices(paths)


    ds = ds.map(

        test_parser,

        num_parallel_calls=tf.data.AUTOTUNE

    )


    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# ==========================================================

# CAE MODEL

# ==========================================================


def build_cae(img_size=128):


    inputs = layers.Input(

        shape=(img_size, img_size, 3)

    )


    # Encoder

    x = layers.Conv2D(32, 3, padding="same")(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(2)(x)


    x = layers.Conv2D(64, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(2)(x)


    x = layers.Conv2D(128, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(2)(x)


    x = layers.Conv2D(256, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(2)(x)


    # Bottleneck

    x = layers.Conv2D(512, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)


    # Decoder

    x = layers.Conv2DTranspose(

        256, 3, strides=2, padding="same"

    )(x)

    x = layers.ReLU()(x)


    x = layers.Conv2DTranspose(

        128, 3, strides=2, padding="same"

    )(x)

    x = layers.ReLU()(x)


    x = layers.Conv2DTranspose(

        64, 3, strides=2, padding="same"

    )(x)

    x = layers.ReLU()(x)


    x = layers.Conv2DTranspose(

        32, 3, strides=2, padding="same"

    )(x)

    x = layers.ReLU()(x)


    outputs = layers.Conv2D(

        3,

        3,

        activation="sigmoid",

        padding="same"

    )(x)


    return models.Model(inputs, outputs)



# ==========================================================

# LOSS

# ==========================================================


def ssim_loss(y_true, y_pred):


    ssim = tf.image.ssim(

        y_true,

        y_pred,

        max_val=1.0

    )


    return 1 - tf.reduce_mean(ssim)



def hybrid_loss(y_true, y_pred):


    mse = tf.reduce_mean(

        tf.square(y_true - y_pred)

    )


    return (

        0.7 * mse

        + 0.3 * ssim_loss(y_true, y_pred)

    )



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



def compute_scores(model, dataset):


    scores = []


    for batch in dataset:


        reconstruction = model.predict(

            batch,

            verbose=0

        )


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


        blur_true = blur(batch).numpy()

        blur_recon = blur(reconstruction).numpy()


        blur_diff = np.mean(

            np.abs(blur_true - blur_recon),

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



# ==========================================================

# THRESHOLD

# ==========================================================


def find_best_threshold(labels, scores):


    best_threshold = None

    best_f1 = -1


    thresholds = np.linspace(

        scores.min(),

        scores.max(),

        300

    )


    for threshold in thresholds:


        predictions = (

            scores > threshold

        ).astype(int)


        score = f1_score(

            labels,

            predictions,

            zero_division=0

        )


        if score > best_f1:

            best_f1 = score

            best_threshold = threshold


    return best_threshold, best_f1



# ==========================================================

# TRAINING

# ==========================================================


def train(category):


    print(f"\nTraining category: {category}")


    train_paths = get_train_paths(category)

    test_paths, test_labels = get_test_data(category)


    print(

        f"Training images: {len(train_paths)}"

    )


    print(

        f"Test images: {len(test_paths)}"

    )


    train_ds = build_train_dataset(train_paths)

    test_ds = build_test_dataset(test_paths)


    model = build_cae(IMG_SIZE)


    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-3

        ),

        loss=hybrid_loss

    )


    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="loss",

            patience=10,

            restore_best_weights=True

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="loss",

            factor=0.5,

            patience=4

        )

    ]


    model.fit(

        train_ds,

        epochs=EPOCHS,

        callbacks=callbacks

    )


    print("\nCalculating anomaly scores...")


    scores = compute_scores(

        model,

        test_ds

    )


    threshold, f1 = find_best_threshold(

        test_labels,

        scores

    )


    print(f"Threshold: {threshold:.6f}")

    print(f"F1 score: {f1:.4f}")


    # Save model

    model_path = MODEL_DIR / "cae.keras"


    model.save(model_path)


    # Save threshold

    threshold_path = MODEL_DIR / "threshold.json"


    with open(threshold_path, "w") as file:

        json.dump(

            {

                "category": category,

                "threshold": float(threshold)

            },

            file,

            indent=4

        )


    print(f"\nModel saved to: {model_path}")

    print(f"Threshold saved to: {threshold_path}")



# ==========================================================

# MAIN

# ==========================================================


if __name__ == "__main__":

    train(CATEGORY)
