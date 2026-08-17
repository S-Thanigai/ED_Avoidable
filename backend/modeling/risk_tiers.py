"""
risk_tiers.py
-------------
LOW / MODERATE / HIGH risk-tier assignment from a predicted probability
and two thresholds. Phase 4 scope only: these are risk tiers, NOT a care
navigation decision -- routing to PCP/Urgent Care/Telehealth/Care
Management is explicitly out of scope here (belongs to Phase 5's Care
Navigation Agent; see docs/DECISION_LOG.md and the Phase 4 spec's Step 24).

Semantics (docs/04_MODEL_DEVELOPMENT.md section 15):
  LOW      -- lower predicted likelihood; no proactive escalation by risk alone.
  MODERATE -- meaningful navigation opportunity; suitable for light-touch
              navigation or review.
  HIGH     -- strongest predicted navigation opportunity; candidate for
              stronger outreach / Care Management review LATER (Phase 5
              decides that, not this module).

moderate_threshold and high_threshold are NOT hardcoded here -- they are
derived from VALIDATION-only analysis in train.py and passed in by the
caller (and persisted in the model artifact/metadata), per the Phase 4
spec's "do not invent arbitrary values" instruction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOW = "LOW"
MODERATE = "MODERATE"
HIGH = "HIGH"


def validate_thresholds(moderate_threshold: float, high_threshold: float) -> None:
    if not (0.0 <= moderate_threshold < high_threshold <= 1.0):
        raise ValueError(
            f"Thresholds must satisfy 0 <= moderate_threshold < high_threshold <= 1; "
            f"got moderate={moderate_threshold}, high={high_threshold}"
        )


def assign_risk_tier(probability: float, moderate_threshold: float, high_threshold: float) -> str:
    """Assign a single probability to LOW / MODERATE / HIGH."""
    validate_thresholds(moderate_threshold, high_threshold)
    if probability >= high_threshold:
        return HIGH
    if probability >= moderate_threshold:
        return MODERATE
    return LOW


def assign_risk_tiers(probabilities, moderate_threshold: float, high_threshold: float) -> pd.Series:
    """Vectorized tier assignment over an array-like of probabilities."""
    validate_thresholds(moderate_threshold, high_threshold)
    probabilities = pd.Series(probabilities)
    tiers = pd.Series(LOW, index=probabilities.index, dtype="object")
    tiers[probabilities >= moderate_threshold] = MODERATE
    tiers[probabilities >= high_threshold] = HIGH
    return tiers


def tier_report(y_true, y_prob, moderate_threshold: float, high_threshold: float) -> list[dict]:
    """Per-tier member count, population %, positive count, observed
    prevalence, lift vs. overall prevalence, and mean predicted
    probability -- the table used for both the VALIDATION and TEST
    risk-tier quality reports (Phase 4 spec Steps 11/13)."""
    y_true = pd.Series(np.asarray(y_true))
    y_prob = pd.Series(np.asarray(y_prob))
    tiers = assign_risk_tiers(y_prob, moderate_threshold, high_threshold)

    overall_prevalence = float(y_true.mean())
    total = len(y_true)

    rows = []
    for tier in (LOW, MODERATE, HIGH):
        mask = tiers == tier
        n = int(mask.sum())
        positives = int(y_true[mask].sum()) if n else 0
        prevalence = positives / n if n else None
        lift = (prevalence / overall_prevalence) if (n and overall_prevalence) else None
        rows.append({
            "tier": tier,
            "member_count": n,
            "population_pct": round(n / total, 6) if total else None,
            "positive_count": positives,
            "observed_prevalence": round(prevalence, 6) if prevalence is not None else None,
            "lift_vs_overall_prevalence": round(lift, 4) if lift is not None else None,
            "mean_predicted_probability": round(float(y_prob[mask].mean()), 6) if n else None,
        })
    return rows
