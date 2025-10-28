import numpy as np

from train_model.metrics import get_metrics


def test_get_metrics():
    # synthetic test labels
    y_test = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    # predicted probabilities
    y_test_pred_proba = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.9, 0.2, 0.7])

    metrics = get_metrics(y_test, y_test_pred_proba)

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
    for key in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        assert isinstance(metrics[key], float)
    for key in ["tp", "tn", "fp", "fn"]:
        assert isinstance(metrics[key], int)

    # Check values (rounded)
    assert metrics["accuracy"] == 1.0  # All predictions are correct with threshold 0.5
    assert metrics["tp"] == 4
    assert metrics["tn"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
