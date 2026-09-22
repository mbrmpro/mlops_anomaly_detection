from pathlib import Path
import os
import random
import shutil

import psycopg2


# ==========================================================
# CONFIG
# ==========================================================

SOURCE_DATA_DIR = Path(
    "/home/ayoub/jupyter/Anomaly_detection_project/"
    "mvtec_anomaly_detection"
)

DATA_DIR = Path("data")

PROJECT_CATEGORIES = (
    "bottle",
    "wood",
    "pill",
)

TEST_GOOD_FRACTION = 0.20

NUMBER_OF_BATCHES = 3

RANDOM_SEED = 42


# ==========================================================
# DATABASE CONNECTION
# ==========================================================

def get_connection():

    return psycopg2.connect(
        host=os.getenv(
            "POSTGRES_HOST",
            "localhost",
        ),
        port=int(
            os.getenv(
                "POSTGRES_PORT",
                "5432",
            )
        ),
        dbname=os.getenv(
            "POSTGRES_DB",
            "anomaly_db",
        ),
        user=os.getenv(
            "POSTGRES_USER",
            "anomaly_user",
        ),
        password=os.getenv(
            "POSTGRES_PASSWORD",
            "anomaly_password",
        ),
    )


# ==========================================================
# RESET DATABASE
# ==========================================================

def reset_database():

    print("Resetting database...")

    with get_connection() as connection:

        with connection.cursor() as cursor:

            # Remove table from the previous design
            cursor.execute(
                """
                DROP TABLE IF EXISTS dataset_assignments;
                """
            )

            # Recreate images table cleanly
            cursor.execute(
                """
                DROP TABLE IF EXISTS images CASCADE;
                """
            )

            cursor.execute(
                """
                CREATE TABLE images (

                    id SERIAL PRIMARY KEY,

                    image_path TEXT NOT NULL UNIQUE,

                    category VARCHAR(50) NOT NULL
                        CHECK (
                            category IN (
                                'bottle',
                                'wood',
                                'pill'
                            )
                        ),

                    split VARCHAR(10) NOT NULL
                        CHECK (
                            split IN (
                                'train',
                                'test'
                            )
                        ),

                    source_split VARCHAR(10) NOT NULL
                        CHECK (
                            source_split IN (
                                'train',
                                'test'
                            )
                        ),

                    batch_id SMALLINT,

                    defect_type VARCHAR(50) NOT NULL,

                    is_anomaly BOOLEAN NOT NULL,

                    is_available BOOLEAN
                        NOT NULL
                        DEFAULT FALSE,

                    released_at TIMESTAMP,

                    CHECK (
                        (
                            split = 'train'
                            AND batch_id BETWEEN 1 AND 3
                        )
                        OR
                        (
                            split = 'test'
                            AND batch_id IS NULL
                        )
                    )
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX idx_images_category
                ON images(category);
                """
            )

            cursor.execute(
                """
                CREATE INDEX idx_images_split
                ON images(category, split);
                """
            )

            cursor.execute(
                """
                CREATE INDEX idx_images_batch
                ON images(
                    category,
                    batch_id,
                    is_available
                );
                """
            )

    print("Database reset completed.")


# ==========================================================
# RESET PROJECT DATA DIRECTORY
# ==========================================================

def reset_data_directory():

    print("Resetting data directory...")

    if DATA_DIR.is_symlink():
        DATA_DIR.unlink()

    elif DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================
# INSERT DATABASE RECORD
# ==========================================================

def insert_record(
    cursor,
    image_path,
    category,
    split,
    source_split,
    batch_id,
    defect_type,
    is_anomaly,
    is_available,
):

    cursor.execute(
        """
        INSERT INTO images (
            image_path,
            category,
            split,
            source_split,
            batch_id,
            defect_type,
            is_anomaly,
            is_available
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        );
        """,
        (
            str(image_path),
            category,
            split,
            source_split,
            batch_id,
            defect_type,
            is_anomaly,
            is_available,
        ),
    )


# ==========================================================
# COPY IMAGE
# ==========================================================

def copy_image(
    source_path,
    destination_directory,
    prefix=None,
):

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if prefix:

        destination_name = (
            f"{prefix}_{source_path.name}"
        )

    else:

        destination_name = source_path.name

    destination_path = (
        destination_directory
        / destination_name
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    return destination_path


# ==========================================================
# COLLECT GOOD IMAGES
# ==========================================================

def collect_good_images(category):

    category_source = (
        SOURCE_DATA_DIR
        / category
    )

    images = []

    # Original train/good
    train_good = (
        category_source
        / "train"
        / "good"
    )

    for image_path in sorted(
        train_good.glob("*.png")
    ):

        images.append(
            (
                image_path,
                "train",
            )
        )

    # Original test/good
    test_good = (
        category_source
        / "test"
        / "good"
    )

    for image_path in sorted(
        test_good.glob("*.png")
    ):

        images.append(
            (
                image_path,
                "test",
            )
        )

    return images


# ==========================================================
# COLLECT ANOMALOUS TEST IMAGES
# ==========================================================

def collect_anomaly_images(category):

    category_test = (
        SOURCE_DATA_DIR
        / category
        / "test"
    )

    images = []

    for defect_directory in sorted(
        category_test.iterdir()
    ):

        if not defect_directory.is_dir():
            continue

        if defect_directory.name == "good":
            continue

        defect_type = defect_directory.name

        for image_path in sorted(
            defect_directory.glob("*.png")
        ):

            images.append(
                (
                    image_path,
                    defect_type,
                )
            )

    return images


# ==========================================================
# SPLIT INTO THREE BATCHES
# ==========================================================

def split_into_batches(images):

    batches = [
        [],
        [],
        [],
    ]

    for index, image in enumerate(images):

        batch_index = (
            index
            % NUMBER_OF_BATCHES
        )

        batches[
            batch_index
        ].append(image)

    return batches


# ==========================================================
# PREPARE ONE CATEGORY
# ==========================================================

def prepare_category(
    cursor,
    category,
):

    print()
    print("=" * 60)
    print(
        f"Preparing category: {category}"
    )
    print("=" * 60)

    # ------------------------------------------------------
    # Collect all normal images
    # ------------------------------------------------------

    good_images = collect_good_images(
        category
    )

    anomaly_images = collect_anomaly_images(
        category
    )

    print(
        f"Original good images: "
        f"{len(good_images)}"
    )

    print(
        f"Original anomaly images: "
        f"{len(anomaly_images)}"
    )

    if not good_images:

        raise RuntimeError(
            f"No good images found "
            f"for {category}"
        )

    # ------------------------------------------------------
    # Shuffle deterministically
    # ------------------------------------------------------

    random_generator = random.Random(
        f"{RANDOM_SEED}-{category}"
    )

    random_generator.shuffle(
        good_images
    )

    # ------------------------------------------------------
    # Fixed test GOOD set = 20%
    # ------------------------------------------------------

    number_test_good = max(
        1,
        round(
            len(good_images)
            * TEST_GOOD_FRACTION
        ),
    )

    test_good_images = (
        good_images[
            :number_test_good
        ]
    )

    training_images = (
        good_images[
            number_test_good:
        ]
    )

    # ------------------------------------------------------
    # Split remaining 80% into three batches
    # ------------------------------------------------------

    batches = split_into_batches(
        training_images
    )

    # ======================================================
    # COPY TRAINING BATCHES
    # ======================================================

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        destination = (
            DATA_DIR
            / category
            / "train"
            / f"batch_{batch_number}"
        )

        for (
            source_path,
            source_split,
        ) in batch:

            destination_path = copy_image(
                source_path,
                destination,
                prefix=source_split,
            )

            insert_record(
                cursor=cursor,
                image_path=destination_path,
                category=category,
                split="train",
                source_split=source_split,
                batch_id=batch_number,
                defect_type="good",
                is_anomaly=False,

                # Airflow will release them later
                is_available=False,
            )

    # ======================================================
    # COPY NORMAL TEST IMAGES
    # ======================================================

    test_good_destination = (
        DATA_DIR
        / category
        / "test"
        / "good"
    )

    for (
        source_path,
        source_split,
    ) in test_good_images:

        destination_path = copy_image(
            source_path,
            test_good_destination,
            prefix=source_split,
        )

        insert_record(
            cursor=cursor,
            image_path=destination_path,
            category=category,
            split="test",
            source_split=source_split,
            batch_id=None,
            defect_type="good",
            is_anomaly=False,

            # Test data is always available
            is_available=True,
        )

    # ======================================================
    # COPY ALL ANOMALOUS TEST IMAGES
    # ======================================================

    for (
        source_path,
        defect_type,
    ) in anomaly_images:

        destination = (
            DATA_DIR
            / category
            / "test"
            / defect_type
        )

        destination_path = copy_image(
            source_path,
            destination,
        )

        insert_record(
            cursor=cursor,
            image_path=destination_path,
            category=category,
            split="test",
            source_split="test",
            batch_id=None,
            defect_type=defect_type,
            is_anomaly=True,
            is_available=True,
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print(
        f"Training images: "
        f"{len(training_images)}"
    )

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"  Batch {batch_number}: "
            f"{len(batch)}"
        )

    print(
        f"Test good: "
        f"{len(test_good_images)}"
    )

    print(
        f"Test anomaly: "
        f"{len(anomaly_images)}"
    )

    print(
        f"Total test: "
        f"{len(test_good_images) + len(anomaly_images)}"
    )


# ==========================================================
# PRINT DATABASE SUMMARY
# ==========================================================

def print_database_summary():

    print()
    print("=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    category,
                    split,
                    batch_id,
                    is_available,
                    COUNT(*)
                FROM images
                GROUP BY
                    category,
                    split,
                    batch_id,
                    is_available
                ORDER BY
                    category,
                    split,
                    batch_id;
                """
            )

            rows = cursor.fetchall()

            for row in rows:

                print(
                    f"{row[0]:8} | "
                    f"{row[1]:5} | "
                    f"batch={str(row[2]):4} | "
                    f"available={str(row[3]):5} | "
                    f"images={row[4]}"
                )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print(
        "Preparing MVTec data for "
        "incremental Airflow simulation..."
    )

    # Reset everything
    reset_database()
    reset_data_directory()

    # Create prepared data + database metadata
    with get_connection() as connection:

        with connection.cursor() as cursor:

            for category in PROJECT_CATEGORIES:

                prepare_category(
                    cursor,
                    category,
                )

    print_database_summary()

    print()
    print(
        "Data preparation completed successfully."
    )


if __name__ == "__main__":
    main()