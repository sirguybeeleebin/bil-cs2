import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(round(roc_auc_score(y_true, y_proba), 2)),
        "f1": float(round(f1_score(y_true, y_pred), 2)),
        "precision": float(round(precision_score(y_true, y_pred), 2)),
        "recall": float(round(recall_score(y_true, y_pred), 2)),
        "accuracy": float(round(accuracy_score(y_true, y_pred), 2)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
