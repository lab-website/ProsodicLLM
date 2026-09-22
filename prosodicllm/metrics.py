from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

IGNORE = -100


def _masked(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = y_true != IGNORE
    return y_true[m], y_pred[m]


def classification_metrics(y_true, y_pred, prefix="") -> Dict[str, float]:
    yt, yp = _masked(y_true, y_pred)
    if len(yt) == 0:
        return {f"{prefix}accuracy": float("nan"), f"{prefix}macro_f1": float("nan")}
    return {
        f"{prefix}accuracy": float(accuracy_score(yt, yp)),
        f"{prefix}macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def tone_error_rate(y_true, y_pred) -> float:
    yt, yp = _masked(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float((yt != yp).mean())


def rhyme_pair_f1(true_rhyme: List[List[int]], pred_rhyme: List[List[int]]) -> float:
    gold, pred = [], []
    for gt, pr in zip(true_rhyme, pred_rhyme):
        valid = [i for i, x in enumerate(gt) if x != IGNORE]
        for a in range(len(valid)):
            for b in range(a + 1, len(valid)):
                i, j = valid[a], valid[b]
                gold.append(int(gt[i] == gt[j]))
                pred.append(int(pr[i] == pr[j]))
    if not gold:
        return float("nan")
    return float(f1_score(gold, pred, zero_division=0))


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054):
    """Approximate Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    phat = correct / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return (center - half, center + half)
