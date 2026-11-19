"""Metricas de classificacao implementadas a partir da matriz de confusao."""
import numpy as np
from sklearn.metrics import roc_auc_score


def confusion(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"TP": int(((y_true == 1) & (y_pred == 1)).sum()),
            "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
            "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
            "FN": int(((y_true == 1) & (y_pred == 0)).sum())}


def _div(a, b):
    return a / b if b else 0.0


def fbeta(precision, recall, beta):
    """F_beta = (1+b^2) P R / (b^2 P + R)."""
    b2 = beta ** 2
    return _div((1 + b2) * precision * recall, b2 * precision + recall)


def evaluate(y_true, y_score, threshold=0.5) -> dict:
    """Retorna todas as metricas para um dado threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    auc = roc_auc_score(y_true, y_score) if len(set(y_score)) > 1 else 0.5
    return {**cm, "Acuracia": _div(tp + tn, tp + tn + fp + fn), "Precisao": p, "Recall": r,
            "F1": fbeta(p, r, 1), "F2": fbeta(p, r, 2), "F0.5": fbeta(p, r, 0.5),
            "AUC-ROC": auc}
