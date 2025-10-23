import numpy as np

from ml.metrics import get_metrics


def test_get_metrics_perfect_prediction():
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = get_metrics(y_test, y_pred, y_pred_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_get_metrics_all_wrong():
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    y_pred_proba = np.array([0.9, 0.1, 0.8, 0.2])

    metrics = get_metrics(y_test, y_pred, y_pred_proba)

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["auc"] == 0.0
    assert metrics["tp"] == 0
    assert metrics["tn"] == 0
    assert metrics["fp"] == 2
    assert metrics["fn"] == 2


def test_get_metrics_mixed_prediction():
    y_test = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_pred_proba = np.array([0.2, 0.8, 0.7, 0.4])

    metrics = get_metrics(y_test, y_pred, y_pred_proba)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1


def test_get_metrics_zero_division():
    y_test = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.3])

    metrics = get_metrics(y_test, y_pred, y_pred_proba)

    # Precision, recall, f1 should handle zero division
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["tp"] == 0
    assert metrics["tn"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
