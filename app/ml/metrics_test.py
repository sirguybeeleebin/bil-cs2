import numpy as np
import pytest

from app.ml.metrics import get_metrics

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def y_test():
    return np.array([0, 1, 0, 1, 1, 0, 1, 0])


@pytest.fixture
def y_pred_proba():
    return np.array([0.1, 0.9, 0.4, 0.8, 0.6, 0.3, 0.7, 0.2])


@pytest.fixture
def y_pred_proba_all_zeros():
    return np.zeros(8)


@pytest.fixture
def y_pred_proba_all_ones():
    return np.ones(8)


# -----------------------------
# Tests
# -----------------------------


def test_get_metrics_basic(y_test, y_pred_proba):
    metrics = get_metrics(y_test, y_pred_proba)

    # Check keys
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

    # Check types
    for k in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        assert isinstance(metrics[k], float)
    for k in ["tp", "tn", "fp", "fn"]:
        assert isinstance(metrics[k], int)

    # Check some metric values are in expected range
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_get_metrics_all_zeros(y_test, y_pred_proba_all_zeros):
    metrics = get_metrics(y_test, y_pred_proba_all_zeros)
    # Predictions are all 0
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["tn"] >= 0
    assert metrics["fn"] >= 0
    # Accuracy should be <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_get_metrics_all_ones(y_test, y_pred_proba_all_ones):
    metrics = get_metrics(y_test, y_pred_proba_all_ones)
    # Predictions are all 1
    assert metrics["tn"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] >= 0
    assert metrics["fp"] >= 0
    # Accuracy should be <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_get_metrics_threshold_edge_case():
    y_test = np.array([0, 1])
    y_pred_proba = np.array([0.5, 0.5])  # Exactly 0.5
    metrics = get_metrics(y_test, y_pred_proba)
    # Both should be predicted as 1
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 0
    assert metrics["fn"] == 0
