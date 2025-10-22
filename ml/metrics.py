import logging
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_metrics(
    y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> dict:
    log.info("Вычисление метрик модели...")

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, zero_division=0), 2)
    recall = round(recall_score(y_test, y_pred, zero_division=0), 2)
    fscore = round(f1_score(y_test, y_pred, zero_division=0), 2)
    auc = round(roc_auc_score(y_test, y_pred_proba), 2)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tn, fp, fn, tp = round(tn, 2), round(fp, 2), round(fn, 2), round(tp, 2)

    metrics = {
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

    log.info("Метрики успешно рассчитаны:")
    for k, v in metrics.items():
        log.info(f"  {k}: {v}")

    return metrics
