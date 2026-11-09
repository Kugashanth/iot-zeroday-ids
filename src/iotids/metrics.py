"""Thresholding and evaluation metrics for anomaly detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .data import BENIGN


def benign_threshold(scores_benign_val: np.ndarray, quantile: float) -> float:
    """Unsupervised threshold: the q-th quantile of BENIGN validation scores.
    q=0.99 -> ~1% false-positive rate on benign traffic by construction."""
    return float(np.quantile(scores_benign_val, quantile))


def robust_stats(scores: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based sigma (robust to <50% contamination by attacks)."""
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826
    return med, max(mad, 1e-12)


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def fpr_at_tpr(y: np.ndarray, s: np.ndarray, tpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, tpr_target, side="left")
    return float(fpr[min(idx, len(fpr) - 1)])


def evaluate(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    """Threshold-free + thresholded metrics. y: 1=attack, 0=benign."""
    pred = (s > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "fpr": fp / max(fp + tn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": float(thr),
    }
    if len(np.unique(y)) == 2:
        out.update({
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "tpr@fpr=1%": tpr_at_fpr(y, s, 0.01),
            "fpr@tpr=95%": fpr_at_tpr(y, s, 0.95),
        })
    return out


def per_family(families: np.ndarray | pd.Series, s: np.ndarray, thr: float) -> dict:
    """Detection rate per attack family (for Benign this is the false-positive rate)."""
    fam = pd.Series(np.asarray(families))
    flagged = pd.Series(s > thr)
    return flagged.groupby(fam).mean().to_dict()


def family_auc(families: np.ndarray | pd.Series, s: np.ndarray) -> dict:
    """ROC-AUC of each attack family vs benign — threshold-independent separability."""
    fam = np.asarray(families)
    b = s[fam == BENIGN]
    out = {}
    for f in sorted(set(fam) - {BENIGN}):
        a = s[fam == f]
        y = np.r_[np.zeros(len(b)), np.ones(len(a))]
        out[f] = float(roc_auc_score(y, np.r_[b, a]))
    return out
