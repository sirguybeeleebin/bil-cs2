import warnings

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


def get_metrics(y_true, y_pred, y_proba):
    acc = float(round(accuracy_score(y_true, y_pred), 2))
    prec = float(round(precision_score(y_true, y_pred), 2))
    rec = float(round(recall_score(y_true, y_pred), 2))
    f1 = float(round(f1_score(y_true, y_pred), 2))
    auc = float(round(roc_auc_score(y_true, y_proba), 2))
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, y_pred).ravel())
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
