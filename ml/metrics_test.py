import numpy as np

from ml.metrics import get_metrics  # Adjust import path


def test_get_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = get_metrics(y_true, y_pred, y_proba)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "tp",
        "tn",
        "fp",
        "fn",
    }
    assert set(metrics.keys()) == expected_keys

    # Check metric values
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.67
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 0.8
    assert metrics["auc"] == 1.0  # Fix: correct AUC

    # Confusion matrix
    assert metrics["tp"] == 2
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
