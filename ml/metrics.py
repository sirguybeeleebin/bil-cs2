from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(y_test: np.ndarray, y_test_pred_proba: np.ndarray) -> dict[str, float]:
    y_proba = y_test_pred_proba
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float] = {
        "roc_auc": float(round(roc_auc_score(y_test, y_proba), 2)),
        "f1": float(round(f1_score(y_test, y_pred), 2)),
        "precision": float(round(precision_score(y_test, y_pred), 2)),
        "recall": float(round(recall_score(y_test, y_pred), 2)),
        "accuracy": float(round(accuracy_score(y_test, y_pred), 2)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return metrics
