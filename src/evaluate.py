"""Evaluate the MLflow champion CAE models without retraining them.

This script reports three views of each model:
1. the threshold saved during training;
2. an optimistic notebook-style threshold fitted on the whole test set;
3. a fair threshold fitted on a calibration subset and measured on a holdout subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from predict import load_champion
from training import build_image_dataset, compute_scores, get_test_data


RANDOM_STATE = 42
CALIBRATION_SIZE = 0.30


def find_best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose the score threshold that maximizes F1 on calibration data."""

    candidate_thresholds = np.unique(scores)
    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        predictions = (scores > threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate threshold-dependent metrics and AUROC."""

    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_category(category: str) -> dict:
    model_dir = Path("models") / category
    model, saved_threshold, model_version = load_champion(category)

    test_paths, labels = get_test_data(category)
    score_components = compute_scores(model, build_image_dataset(test_paths))
    scores = score_components["hybrid"]

    optimistic_threshold = find_best_threshold(labels, scores)

    indices = np.arange(len(labels))
    calibration_indices, holdout_indices = train_test_split(
        indices,
        train_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    calibrated_threshold = find_best_threshold(
        labels[calibration_indices],
        scores[calibration_indices],
    )

    report = {
        "category": category,
        "model_version": model_version,
        "saved_threshold_full_test": classification_metrics(
            labels,
            scores,
            saved_threshold,
        ),
        "optimistic_full_test": classification_metrics(
            labels,
            scores,
            optimistic_threshold,
        ),
        "calibrated_holdout": classification_metrics(
            labels[holdout_indices],
            scores[holdout_indices],
            calibrated_threshold,
        ),
        "calibration_samples": int(len(calibration_indices)),
        "holdout_samples": int(len(holdout_indices)),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / "evaluation.json"
    output_path.write_text(json.dumps(report, indent=4))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("categories", nargs="+", help="Saved model categories to evaluate")
    args = parser.parse_args()

    for category in args.categories:
        report = evaluate_category(category)
        saved = report["saved_threshold_full_test"]
        optimistic = report["optimistic_full_test"]
        calibrated = report["calibrated_holdout"]

        print(f"\n========== EVALUATION: {category} ==========")
        print(f"Champion version:              {report['model_version']}")
        print(f"AUROC:                         {saved['auroc']:.4f}")
        print(f"Saved-threshold F1:            {saved['f1']:.4f}")
        print(f"Optimistic notebook-style F1: {optimistic['f1']:.4f}")
        print(f"Calibrated holdout F1:         {calibrated['f1']:.4f}")
        print(f"Calibrated holdout recall:     {calibrated['recall']:.4f}")
        print(f"Report: models/{category}/evaluation.json")


if __name__ == "__main__":
    main()
