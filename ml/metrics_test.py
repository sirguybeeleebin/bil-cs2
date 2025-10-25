import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.metrics import get_metrics


def test_get_metrics_basic():
    y_test = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = get_metrics(y_test, y_pred_proba)

    # Check that all keys exist
    expected_keys = {
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
    }
    assert set(metrics.keys()) == expected_keys

    # Check that values are of correct type
    for key in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        assert isinstance(metrics[key], float)
    for key in ["tp", "tn", "fp", "fn"]:
        assert isinstance(metrics[key], int)

    # Check actual values
    y_pred = (y_pred_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    assert metrics["tp"] == tp
    assert metrics["tn"] == tn
    assert metrics["fp"] == fp
    assert metrics["fn"] == fn
    assert metrics["roc_auc"] == round(roc_auc_score(y_test, y_pred_proba), 2)
    assert metrics["f1"] == round(f1_score(y_test, y_pred), 2)
    assert metrics["precision"] == round(precision_score(y_test, y_pred), 2)
    assert metrics["recall"] == round(recall_score(y_test, y_pred), 2)
    assert metrics["accuracy"] == round(accuracy_score(y_test, y_pred), 2)


def test_get_metrics_all_zeros():
    y_test = np.array([0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.3])

    metrics = get_metrics(y_test, y_pred_proba)
    # In this case, tp, fp, fn should be 0, tn should be len(y_test)
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 3
    assert metrics["accuracy"] == 1.0


def test_get_metrics_all_ones():
    y_test = np.array([1, 1, 1])
    y_pred_proba = np.array([0.9, 0.8, 0.7])

    metrics = get_metrics(y_test, y_pred_proba)
    assert metrics["tp"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 0
    assert metrics["accuracy"] == 1.0
