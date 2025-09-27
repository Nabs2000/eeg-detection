from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score

# -----------------------------
# Metrics
# -----------------------------
def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    # AUROC can fail if only one class present; guard it
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    # default threshold 0.5 for reporting
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return {"auc": auc, "acc": acc, "prec": prec, "rec": rec, "f1": f1}

def best_threshold(y_true: np.ndarray, probs: np.ndarray, grid=None) -> Tuple[float, float]:
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1