"""
metrics.py
----------
Shared metric computation for Phase 4 model comparison, threshold search,
and final TEST evaluation. Every metric here is computed from y_true
(0/1) and y_prob (predicted probability of the positive class) alone --
no dependence on which model produced them, so the same functions apply
identically to VALIDATION model comparison and the single TEST pass.

ACCURACY IS DELIBERATELY NOT COMPUTED AS A HEADLINE METRIC ANYWHERE IN
THIS MODULE (Phase 4 spec: "ACCURACY MUST NOT BE THE PRIMARY MODEL
SELECTION METRIC") -- at ~9% prevalence a trivial always-negative
classifier scores ~91% accuracy, which is meaningless here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    n_selected = int(tp + fp)
    return {
        "threshold": float(threshold),
        "n_selected": n_selected,
        "pct_selected": round(n_selected / len(y_true), 6) if len(y_true) else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "specificity": round(float(specificity), 6),
        "f1": round(float(f1), 6),
    }


def rank_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Threshold-independent metrics: ROC-AUC, PR-AUC, Brier, log loss."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    prevalence = float(y_true.mean())

    result = {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 6),
        "prevalence": round(prevalence, 6),
        "pr_auc_no_skill_baseline": round(prevalence, 6),
    }
    # log loss requires probabilities strictly in (0, 1) with both classes present
    try:
        clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        result["log_loss"] = round(float(log_loss(y_true, clipped, labels=[0, 1])), 6)
    except ValueError:
        result["log_loss"] = None
    return result


def full_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Convenience: rank metrics + confusion-matrix metrics at one threshold."""
    return {**rank_metrics(y_true, y_prob), **threshold_confusion_counts(y_true, y_prob, threshold)}


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float]) -> list[dict]:
    return [threshold_confusion_counts(y_true, y_prob, t) for t in thresholds]


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width probability bins: mean predicted probability vs.
    observed positive rate per bin, plus bin size -- the raw data behind a
    calibration curve, kept in a plain list-of-dicts so it serializes
    straightforwardly to calibration_bins.csv."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            "bin_index": b,
            "bin_low": round(float(edges[b]), 4),
            "bin_high": round(float(edges[b + 1]), 4),
            "n": n,
            "mean_predicted_probability": round(float(y_prob[mask].mean()), 6) if n else None,
            "observed_positive_rate": round(float(y_true[mask].mean()), 6) if n else None,
        })
    return rows
