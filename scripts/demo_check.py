"""Readiness check before presenting - run it on the day of the presentation.

Run through scripts/demo_setup.sh (it starts this file in the API image).

1. Every test image shown on the Prediction tab (one per defect type) is
   judged by the API; the report says which verdicts are right or wrong, so
   the presenter knows in advance which image to click.
2. SAMPLE_SIZE further test images per category are judged. This measures the
   champion on more data and leaves enough recent predictions (at least 10
   per category within 24 hours) for the Evidently drift check on tab 11.

Exit code 1 when the API does not answer or a category has no champion.
Wrong verdicts are reported, not hidden: they are a property of the model.
"""

import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import psycopg2

API_URL = os.getenv("DEMO_API_URL", "http://127.0.0.1:8000")
CATEGORIES = ("bottle", "wood", "pill")
SAMPLE_SIZE = 12


def call(method, path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{API_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def test_images(category):
    """All test images of a category: (image_path, defect_type, is_anomaly)."""

    connection = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "anomaly_db"),
        user=os.getenv("POSTGRES_USER", "anomaly_user"),
        password=os.getenv("POSTGRES_PASSWORD", "anomaly_password"),
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT image_path, defect_type, is_anomaly FROM images "
                "WHERE category = %s AND split = 'test'",
                (category,),
            )
            return cursor.fetchall()
    finally:
        connection.close()


def judge(category, image_path, is_anomaly):
    answer = call("POST", "/predict", {"image_path": image_path, "category": category})
    correct = (answer["prediction"] == "anomaly") == bool(is_anomaly)
    ratio = answer["anomaly_score"] / answer["threshold"]
    return answer, correct, ratio


def main():
    try:
        call("GET", "/health/db", timeout=10)
    except (urllib.error.URLError, OSError) as error:
        print(f"FAILED: the API does not answer at {API_URL} ({error})")
        sys.exit(1)

    failed = False

    for category in CATEGORIES:
        print(f"\n== {category}")

        query = urllib.parse.urlencode({"category": category})
        examples = call("GET", f"/images/examples?{query}")["examples"]

        print("Prediction tab images:")
        for example in examples:
            is_anomaly = example["defect_type"] != "good"
            try:
                answer, correct, ratio = judge(category, example["image_path"], is_anomaly)
            except urllib.error.HTTPError as error:
                print(f"  FAILED {example['defect_type']}: HTTP {error.code} (no champion?)")
                failed = True
                break
            mark = "right" if correct else "WRONG"
            print(
                f"  {mark:5}  {example['defect_type']:16} -> {answer['prediction']:7} "
                f"({ratio:.2f} x threshold, model v{answer.get('model_version')})"
            )

        if failed:
            continue

        # Deterministic sample: the same images on every run.
        rows = sorted(test_images(category), key=lambda row: hashlib.md5(row[0].encode()).hexdigest())
        sample = rows[:SAMPLE_SIZE]
        right = sum(judge(category, path, is_anomaly)[1] for path, _, is_anomaly in sample)

        print(f"Sample of {len(sample)} test images: {right}/{len(sample)} right")

    if failed:
        print("\nFAILED: train the champion models first (scripts/demo_setup.sh does it).")
        sys.exit(1)

    print("\nReady: verdicts above, and enough recent predictions for the drift check (tab 11).")


if __name__ == "__main__":
    main()
