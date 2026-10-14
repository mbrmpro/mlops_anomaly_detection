"""Regression check for the drift monitoring cache (no extra dependency).

Run inside the API container:

    docker compose exec api python -m src.check_monitoring

Fails when

* the features of an unchanged training image are computed again (every
  drift run then decodes all training images again: up to 30 s per run),
* a changed file keeps its old features.
"""

import os
import sys
import tempfile

from src import monitoring


def problems():
    if not hasattr(monitoring, "reference_features"):
        return ["src/monitoring.py has no reference_features(): no feature cache"]

    computed = []

    def counting_features(image_path):
        computed.append(image_path)
        return {"brightness": float(len(computed))}

    original = monitoring.image_features
    monitoring.image_features = counting_features
    found = []

    try:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train_good.png")

            with open(path, "wb") as image_file:
                image_file.write(b"first version")

            first = monitoring.reference_features(path)
            again = monitoring.reference_features(path)

            if len(computed) != 1 or again != first:
                found.append(f"an unchanged file was computed {len(computed)} times instead of once")

            computed_before_change = len(computed)

            with open(path, "wb") as image_file:
                image_file.write(b"second, longer version")

            changed = monitoring.reference_features(path)

            if len(computed) != computed_before_change + 1 or changed == first:
                found.append("a changed file kept its old features")
    finally:
        monitoring.image_features = original

    return found


def main():
    found = problems()

    if found:
        print("FAILED (monitoring)")
        for problem in found:
            print(" -", problem)
        sys.exit(1)

    print("OK (monitoring)")


if __name__ == "__main__":
    main()
