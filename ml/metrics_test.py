import numpy as np

from ml.metrics import get_metrics


def test_get_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8])  # threshold 0.5 by default

    metrics = get_metrics(y_true, y_proba)

    # With threshold 0.5, predictions = [0,0,0,1]
    assert metrics["tp"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1

    # Check metric types and rounding
    assert isinstance(metrics["roc_auc"], float)
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.67


def test_get_metrics_with_threshold():
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.3, 0.6, 0.4, 0.8])

    # threshold 0.5
    metrics = get_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1


def test_get_metrics_all_zeros():
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = get_metrics(y_true, y_proba)
    assert metrics["tp"] == 0
    assert metrics["tn"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["accuracy"] == 1.0
