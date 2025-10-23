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
    y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> dict:
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, zero_division=0), 2)
    recall = round(recall_score(y_test, y_pred, zero_division=0), 2)
    fscore = round(f1_score(y_test, y_pred, zero_division=0), 2)

    try:
        auc = round(roc_auc_score(y_test, y_pred_proba), 2)
    except ValueError:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": fscore,
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
