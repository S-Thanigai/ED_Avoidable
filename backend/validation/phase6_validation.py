"""
phase6_validation.py
---------------------
Phase 6 -- Safety + Fairness + Robustness + End-to-End Multi-Agent
Validation.

This is NOT a model-development script. It never retrains, tunes, or
recalibrates `uc07-risk-synthetic-v1`; it loads the frozen artifact and
the frozen synthetic snapshots/raw data exactly as they are and exercises
the already-built Risk Detection / Care Navigation / Safety & Policy
agents and orchestrator against them.

Produces every artifact under artifacts/phase6_validation/. Run:
    python backend/validation/phase6_validation.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for _subdir in ("pit", "agents", "modeling"):
    _p = str(BACKEND_DIR / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import care_navigation
import safety_policy
from contracts import CurrentSafetyContext, NavigationDecision, NavigationDestination, ReasonCode, RiskTier, SafetyState
from orchestrator import UC07Orchestrator, decision_to_dict
from risk_detection import RiskDetectionAgent
from windows import build_snapshot_window

SYNTHETIC_RAW_DIR = REPO_ROOT / "data" / "synthetic"
SYNTHETIC_DERIVED_DIR = REPO_ROOT / "data" / "derived" / "synthetic"
TEST_CSV = SYNTHETIC_DERIVED_DIR / "test_snapshot.csv"
MODELS_DIR = REPO_ROOT / "backend" / "models"
EVAL_DIR = REPO_ROOT / "artifacts" / "phase6_validation"

MODERATE_THRESHOLD = 0.105986
HIGH_THRESHOLD = 0.213252
TEST_INDEX_DATE = date(2026, 4, 3)
MIN_SUBGROUP_N = 100  # Step 10: below this, metrics are descriptive/unstable only

# --- these constants are validation-script-local copies for reporting
# convenience ONLY. The single authoritative source of the thresholds is
# always backend/models/uc07_risk_synthetic_v1_model_metadata.json, loaded
# and asserted equal to these below before anything else runs.


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frozen_set() -> dict:
    files = {
        "raw_members.csv": REPO_ROOT / "raw_members.csv",
        "raw_ed_visits.csv": REPO_ROOT / "raw_ed_visits.csv",
        "raw_care_history.csv": REPO_ROOT / "raw_care_history.csv",
        "synthetic_raw_members.csv": SYNTHETIC_RAW_DIR / "raw_members.csv",
        "synthetic_raw_ed_visits.csv": SYNTHETIC_RAW_DIR / "raw_ed_visits.csv",
        "synthetic_raw_care_history.csv": SYNTHETIC_RAW_DIR / "raw_care_history.csv",
        "train_snapshot.csv": SYNTHETIC_DERIVED_DIR / "train_snapshot.csv",
        "validation_snapshot.csv": SYNTHETIC_DERIVED_DIR / "validation_snapshot.csv",
        "test_snapshot.csv": TEST_CSV,
        "uc07_risk_v1_model.joblib": MODELS_DIR / "uc07_risk_v1_model.joblib",
        "uc07_risk_synthetic_v1_model.joblib": MODELS_DIR / "uc07_risk_synthetic_v1_model.joblib",
    }
    return {name: sha256_file(p) for name, p in files.items()}


# =============================================================================
# Threshold-dependent metric helpers (Accuracy/Balanced Accuracy included,
# never as a selection metric -- descriptive only, same discipline as
# Phase 4E's train_phase4e_tree_comparison.py)
# =============================================================================

def group_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    positives = int(y_true.sum())
    prevalence = positives / n if n else None

    result = {
        "n": n,
        "positives": positives,
        "prevalence": round(prevalence, 6) if prevalence is not None else None,
        "mean_probability": round(float(y_prob.mean()), 6) if n else None,
        "median_probability": round(float(np.median(y_prob)), 6) if n else None,
    }

    if n == 0:
        return result

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result.update({
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "specificity": round(float(tn / (tn + fp)), 6) if (tn + fp) > 0 else None,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "fpr": round(float(fp / (fp + tn)), 6) if (fp + tn) > 0 else None,
        "fnr": round(float(fn / (fn + tp)), 6) if (fn + tp) > 0 else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })
    # rank metrics need both classes present
    if 0 < positives < n:
        result["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        result["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 6)
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    result["brier"] = round(float(brier_score_loss(y_true, y_prob)), 6)

    tiers = pd.Series(np.where(y_prob >= HIGH_THRESHOLD, "HIGH", np.where(y_prob >= MODERATE_THRESHOLD, "MODERATE", "LOW")))
    result["low_pct"] = round(float((tiers == "LOW").mean()), 6)
    result["moderate_pct"] = round(float((tiers == "MODERATE").mean()), 6)
    result["high_pct"] = round(float((tiers == "HIGH").mean()), 6)
    result["moderate_plus_selection_rate"] = round(float((tiers != "LOW").mean()), 6)
    result["high_selection_rate"] = round(float((tiers == "HIGH").mean()), 6)
    return result


def classify_disparity(delta: float, kind: str) -> str:
    """Defensible, explicit, documented thresholds (Step 11). Not a
    statistical-significance test -- a descriptive triage rule to decide
    which observed differences deserve a closer look. NEVER interpreted
    as "the model is/isn't fair"."""
    magnitude = abs(delta)
    if magnitude >= 0.30:
        return "INVESTIGATE"
    if magnitude >= 0.10:
        return "MONITOR"
    return "NO MATERIAL SIGNAL DETECTED"


# =============================================================================
# STEP 2/3 -- Safety override matrix + missing-context matrix
# =============================================================================

def _nav_row(destination_label: str, member_id: str, risk_tier: RiskTier, feature_dict: dict) -> tuple[NavigationDecision, dict]:
    row = pd.Series(feature_dict)
    nav = care_navigation.decide(member_id, risk_tier, row)
    return nav, feature_dict


NAV_SCENARIOS = {
    "CARE_MANAGEMENT": (RiskTier.HIGH, {
        "transportation_barrier": 1, "telehealth_available": 0, "pcp_distance_miles": 5, "urgent_care_distance_miles": 5,
        "clinical_burden": 3, "prior_ed_count_270d": 3, "prior_potentially_avoidable_ed_count_270d": 3,
        "prior_pcp_count_270d": 1, "has_prior_care_management": 0,
    }),
    "TELEHEALTH": (RiskTier.LOW, {
        "transportation_barrier": 1, "telehealth_available": 1, "pcp_distance_miles": 15, "urgent_care_distance_miles": 15,
        "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
        "prior_pcp_count_270d": 0, "has_prior_care_management": 0,
    }),
    "URGENT_CARE": (RiskTier.LOW, {
        "transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 8, "urgent_care_distance_miles": 3,
        "clinical_burden": 0, "prior_ed_count_270d": 2, "prior_potentially_avoidable_ed_count_270d": 0,
        "prior_pcp_count_270d": 0, "has_prior_care_management": 0,
    }),
    "PRIMARY_CARE": (RiskTier.LOW, {
        "transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 5, "urgent_care_distance_miles": 20,
        "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
        "prior_pcp_count_270d": 1, "has_prior_care_management": 0,
    }),
    "NO_PROACTIVE_NAVIGATION": (RiskTier.LOW, {
        "transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 20, "urgent_care_distance_miles": 20,
        "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
        "prior_pcp_count_270d": 0, "has_prior_care_management": 0,
    }),
}

INDIVIDUAL_TRIGGERS = {
    "red_flag=1": dict(red_flag=1),
    "icu=1": dict(icu=1),
    "admitted=1": dict(admitted=1),
    "major_procedure=1": dict(major_procedure=1),
    "triage_level=1": dict(triage_level=1),
    "triage_level=2": dict(triage_level=2),
}

COMBINATION_TRIGGERS = {
    "red_flag + HIGH risk": (dict(red_flag=1), "CARE_MANAGEMENT"),  # HIGH-tier scenario above
    "ICU + telehealth available": (dict(icu=1), "TELEHEALTH"),
    "admitted + strong PCP access": (dict(admitted=1), "PRIMARY_CARE"),
    "major procedure + LOW risk": (dict(major_procedure=1), "NO_PROACTIVE_NAVIGATION"),
    "triage 1 + care-management eligibility": (dict(triage_level=1), "CARE_MANAGEMENT"),
}


def build_safety_override_matrix() -> pd.DataFrame:
    rows = []

    # ---- verify each scenario's pre-safety destination first (sanity) ----
    pre_safety = {}
    for label, (tier, feats) in NAV_SCENARIOS.items():
        nav, _ = _nav_row(label, "SCEN", tier, feats)
        pre_safety[label] = (nav, tier)

    # ---- individual triggers x every destination x LOW/MODERATE/HIGH nav scenario ----
    for trigger_name, trigger_kwargs in INDIVIDUAL_TRIGGERS.items():
        for dest_label, (nav, tier) in pre_safety.items():
            context = CurrentSafetyContext(**trigger_kwargs)
            safety, final_nav = safety_policy.decide(nav, context)
            rows.append({
                "scenario_group": "individual_trigger_x_destination",
                "trigger": trigger_name, "risk_tier": tier.value,
                "preliminary_destination": nav.destination.value,
                "safety_state": safety.state.value, "override": safety.override,
                "final_destination": final_nav.destination.value if final_nav.destination else None,
                "expected": "OVERRIDE", "passed": safety.state == SafetyState.OVERRIDE and final_nav.destination is None,
            })

    # ---- individual triggers x explicit risk tier sweep (fixed feature row) ----
    fixed_feats = NAV_SCENARIOS["CARE_MANAGEMENT"][1]
    for trigger_name, trigger_kwargs in INDIVIDUAL_TRIGGERS.items():
        for tier in (RiskTier.LOW, RiskTier.MODERATE, RiskTier.HIGH):
            nav, _ = _nav_row("SCEN", "SCEN", tier, fixed_feats)
            context = CurrentSafetyContext(**trigger_kwargs)
            safety, final_nav = safety_policy.decide(nav, context)
            rows.append({
                "scenario_group": "individual_trigger_x_risk_tier",
                "trigger": trigger_name, "risk_tier": tier.value,
                "preliminary_destination": nav.destination.value,
                "safety_state": safety.state.value, "override": safety.override,
                "final_destination": final_nav.destination.value if final_nav.destination else None,
                "expected": "OVERRIDE", "passed": safety.state == SafetyState.OVERRIDE and final_nav.destination is None,
            })

    # ---- named combination scenarios from the Phase 6 spec ----
    for combo_name, (trigger_kwargs, dest_label) in COMBINATION_TRIGGERS.items():
        nav, tier = pre_safety[dest_label]
        context = CurrentSafetyContext(**trigger_kwargs)
        safety, final_nav = safety_policy.decide(nav, context)
        rows.append({
            "scenario_group": "named_combination",
            "trigger": combo_name, "risk_tier": tier.value,
            "preliminary_destination": nav.destination.value,
            "safety_state": safety.state.value, "override": safety.override,
            "final_destination": final_nav.destination.value if final_nav.destination else None,
            "expected": "OVERRIDE", "passed": safety.state == SafetyState.OVERRIDE and final_nav.destination is None,
        })

    # ---- multi-trigger combinations (two triggers at once) ----
    for (n1, k1), (n2, k2) in itertools.combinations(INDIVIDUAL_TRIGGERS.items(), 2):
        merged = {**k1, **k2}
        nav, tier = pre_safety["CARE_MANAGEMENT"]
        context = CurrentSafetyContext(**merged)
        safety, final_nav = safety_policy.decide(nav, context)
        rows.append({
            "scenario_group": "multi_trigger_combination",
            "trigger": f"{n1} + {n2}", "risk_tier": tier.value,
            "preliminary_destination": nav.destination.value,
            "safety_state": safety.state.value, "override": safety.override,
            "final_destination": final_nav.destination.value if final_nav.destination else None,
            "expected": "OVERRIDE", "passed": safety.state == SafetyState.OVERRIDE and final_nav.destination is None,
        })

    return pd.DataFrame(rows)


def build_missing_context_matrix() -> pd.DataFrame:
    nav, _ = _nav_row("TELEHEALTH", "SCEN", RiskTier.LOW, NAV_SCENARIOS["TELEHEALTH"][1])
    rows = []

    def _row(label: str, kwargs: dict, expected: str):
        context = CurrentSafetyContext(**kwargs)
        safety, final_nav = safety_policy.decide(nav, context)
        rows.append({
            "scenario": label, "fields_supplied": json.dumps(kwargs),
            "safety_state": safety.state.value,
            "final_destination": final_nav.destination.value if final_nav.destination else None,
            "expected": expected, "passed": safety.state.value == expected,
        })

    _row("all fields missing", {}, "CAUTION")

    for field, safe_value in (("red_flag", 0), ("icu", 0), ("admitted", 0), ("major_procedure", 0), ("triage_level", 4)):
        _row(f"only {field} supplied (safe value)", {field: safe_value}, "CAUTION")

    all_safe = dict(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4)
    for omit in ("red_flag", "icu", "admitted", "major_procedure", "triage_level"):
        partial = {k: v for k, v in all_safe.items() if k != omit}
        _row(f"all safe except {omit} missing", partial, "CAUTION")

    _row("fully known, all safe", all_safe, "CLEAR")
    _row("red_flag=1 + triage missing", {"red_flag": 1}, "OVERRIDE")
    _row("icu=1 + everything else missing", {"icu": 1}, "OVERRIDE")
    _row("major_procedure=1 + everything else missing", {"major_procedure": 1}, "OVERRIDE")
    _row("triage_level=1 + everything else missing", {"triage_level": 1}, "OVERRIDE")
    _row("mixed: red_flag=0 known-safe, triage=1 known-override", {"red_flag": 0, "triage_level": 1}, "OVERRIDE")
    _row("mixed: 3 known-safe, 2 missing", {"red_flag": 0, "icu": 0, "admitted": 0}, "CAUTION")

    return pd.DataFrame(rows)


# =============================================================================
# STEP 5 -- Prohibited language scan
# =============================================================================

def scan_prohibited_language(agent: RiskDetectionAgent, features_df: pd.DataFrame) -> dict:
    violations = []

    # (a) every reachable NavigationDecision explanation, generated from
    # the module's own finite phrase/label templates (exhaustive, not
    # sample-dependent) -- catches anything a future template edit could
    # introduce even if it never happens to occur in this population.
    reason_powerset_texts = set()
    for destination in NavigationDestination:
        label = care_navigation._DESTINATION_LABELS.get(destination, "")
        for r in ReasonCode:
            phrase = care_navigation._REASON_PHRASES[r]
            text = f"Based on {phrase}, {label} may be a useful future, non-emergency option."
            reason_powerset_texts.add(text)
    reason_powerset_texts.add(f"Based on this member's current pattern, {care_navigation._REASON_PHRASES[ReasonCode.NO_OPPORTUNITY_IDENTIFIED]}.")

    for text in reason_powerset_texts:
        hits = safety_policy.check_text(text)
        if hits:
            violations.append({"source": "template_enumeration", "text": text, "phrases": hits})

    # (b) static safety/disclaimer constants
    for name, text in (
        ("BASE_DISCLAIMER", safety_policy.BASE_DISCLAIMER),
        ("OVERRIDE_MESSAGE", safety_policy.OVERRIDE_MESSAGE),
        ("CAUTION_MESSAGE", safety_policy.CAUTION_MESSAGE),
        ("CLEAR_MESSAGE", safety_policy.CLEAR_MESSAGE),
    ):
        hits = safety_policy.check_text(text)
        if hits:
            violations.append({"source": f"constant:{name}", "text": text, "phrases": hits})

    # (c) full synthetic population: actual generated NavigationDecision
    # explanations at the member's real model-derived risk tier.
    X = features_df[agent.feature_columns]
    probs = agent.pipeline.predict_proba(X)[:, 1]
    tiers = [RiskTier(t) for t in np.where(probs >= agent.high_threshold, "HIGH", np.where(probs >= agent.moderate_threshold, "MODERATE", "LOW"))]

    n_scanned = 0
    for i in range(len(features_df)):
        row = features_df.iloc[i]
        nav = care_navigation.decide(str(row["member_id"]), tiers[i], row)
        n_scanned += 1
        hits = safety_policy.check_text(nav.explanation)
        if hits:
            violations.append({"source": "population_navigation", "member_id": row["member_id"], "text": nav.explanation, "phrases": hits})
        # also run the language check the way Safety Agent itself would,
        # under every safety state, to prove no state lets a violation through
        for context in (CurrentSafetyContext(), CurrentSafetyContext(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4), CurrentSafetyContext(red_flag=1)):
            safety, final_nav = safety_policy.decide(nav, context)
            hits2 = safety_policy.check_text(final_nav.explanation) + safety_policy.check_text(safety.message)
            if hits2:
                violations.append({"source": "population_safety_reviewed", "member_id": row["member_id"], "safety_state": safety.state.value, "phrases": hits2})

    return {
        "template_combinations_scanned": len(reason_powerset_texts),
        "population_members_scanned": n_scanned,
        "total_checks": len(reason_powerset_texts) + 4 + n_scanned * 4,
        "violations": violations,
        "violation_count": len(violations),
    }


# =============================================================================
# STEP 6/7 -- Transportation barrier investigation + counterfactual
# =============================================================================

def transportation_barrier_analysis(agent: RiskDetectionAgent, test_df: pd.DataFrame) -> dict:
    X = test_df[agent.feature_columns]
    y = test_df["future_potentially_avoidable_ed_90d"].astype(int).to_numpy()
    probs = agent.pipeline.predict_proba(X)[:, 1]

    groups = {}
    for val in (0, 1):
        mask = (test_df["transportation_barrier"] == val).to_numpy()
        groups[str(val)] = {
            "at_0.50": group_metrics(y[mask], probs[mask], 0.50),
            "at_MODERATE": group_metrics(y[mask], probs[mask], MODERATE_THRESHOLD),
            "at_HIGH": group_metrics(y[mask], probs[mask], HIGH_THRESHOLD),
        }

    # coefficient / contribution
    model = agent.pipeline.named_steps["model"]
    preprocessor = agent.pipeline.named_steps["preprocessor"]
    feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    coefs = dict(zip(feature_names, model.coef_[0]))
    ranked = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    rank = next((i for i, (name, _) in enumerate(ranked, start=1) if name == "transportation_barrier"), None)

    # correlation with other access/history features
    corr_cols = [c for c in agent.feature_columns if c != "transportation_barrier" and pd.api.types.is_numeric_dtype(test_df[c])]
    correlations = test_df[["transportation_barrier"] + corr_cols].corr()["transportation_barrier"].drop("transportation_barrier")
    top_correlated = correlations.abs().sort_values(ascending=False).head(10)
    correlation_report = {name: round(float(correlations[name]), 4) for name in top_correlated.index}

    return {
        "group_0": groups["0"], "group_1": groups["1"],
        "coefficient": {
            "standardized_coefficient": round(float(coefs.get("transportation_barrier", float("nan"))), 6),
            "rank_by_abs_magnitude": rank, "total_features": len(ranked),
            "top_5_by_abs_magnitude": [{"feature": n, "coefficient": round(float(c), 6)} for n, c in ranked[:5]],
        },
        "top_correlated_features": correlation_report,
    }


def counterfactual_transportation(agent: RiskDetectionAgent, test_df: pd.DataFrame) -> pd.DataFrame:
    """In-memory only -- never writes back to test_snapshot.csv."""
    rows = []
    X_base = test_df[agent.feature_columns].copy()
    base_probs = agent.pipeline.predict_proba(X_base)[:, 1]
    base_tiers = np.where(base_probs >= HIGH_THRESHOLD, "HIGH", np.where(base_probs >= MODERATE_THRESHOLD, "MODERATE", "LOW"))

    for direction, from_val, to_val in (("0_to_1", 0, 1), ("1_to_0", 1, 0)):
        mask = (test_df["transportation_barrier"] == from_val).to_numpy()
        if mask.sum() == 0:
            continue
        X_cf = X_base.copy()
        X_cf.loc[mask, "transportation_barrier"] = to_val
        cf_probs = agent.pipeline.predict_proba(X_cf)[:, 1]
        cf_tiers = np.where(cf_probs >= HIGH_THRESHOLD, "HIGH", np.where(cf_probs >= MODERATE_THRESHOLD, "MODERATE", "LOW"))

        delta = cf_probs[mask] - base_probs[mask]
        b_tiers = base_tiers[mask]
        c_tiers = cf_tiers[mask]

        def _crossings(a, b, lo, hi):
            return int(((a == lo) & (b == hi)).sum())

        rows.append({
            "direction": f"transportation_barrier {from_val}->{to_val}",
            "n_affected": int(mask.sum()),
            "mean_probability_change": round(float(delta.mean()), 6),
            "median_probability_change": round(float(np.median(delta)), 6),
            "max_probability_change": round(float(delta.max()), 6),
            "min_probability_change": round(float(delta.min()), 6),
            "n_LOW_to_MODERATE": _crossings(b_tiers, c_tiers, "LOW", "MODERATE"),
            "n_LOW_to_HIGH": _crossings(b_tiers, c_tiers, "LOW", "HIGH"),
            "n_MODERATE_to_HIGH": _crossings(b_tiers, c_tiers, "MODERATE", "HIGH"),
            "n_HIGH_to_lower": int(((b_tiers == "HIGH") & (c_tiers != "HIGH")).sum()),
            "n_MODERATE_to_LOW": int(((b_tiers == "MODERATE") & (c_tiers == "LOW")).sum()),
            "label": "model sensitivity / counterfactual feature perturbation -- NOT clinical causality",
        })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 8 -- Access feature sensitivity
# =============================================================================

def access_feature_sensitivity(agent: RiskDetectionAgent, test_df: pd.DataFrame) -> pd.DataFrame:
    X_base = test_df[agent.feature_columns].copy()
    base_probs = agent.pipeline.predict_proba(X_base)[:, 1]
    base_tiers = np.where(base_probs >= HIGH_THRESHOLD, "HIGH", np.where(base_probs >= MODERATE_THRESHOLD, "MODERATE", "LOW"))
    rows = []

    def _perturb_and_measure(label: str, col: str, transform):
        X_cf = X_base.copy()
        X_cf[col] = transform(X_cf[col])
        cf_probs = agent.pipeline.predict_proba(X_cf)[:, 1]
        cf_tiers = np.where(cf_probs >= HIGH_THRESHOLD, "HIGH", np.where(cf_probs >= MODERATE_THRESHOLD, "MODERATE", "LOW"))
        delta = cf_probs - base_probs
        crossings = int((base_tiers != cf_tiers).sum())
        rows.append({
            "perturbation": label,
            "n": len(X_base),
            "mean_probability_change": round(float(delta.mean()), 6),
            "median_probability_change": round(float(np.median(delta)), 6),
            "max_abs_probability_change": round(float(np.abs(delta).max()), 6),
            "n_tier_crossings": crossings,
            "pct_tier_crossings": round(crossings / len(X_base), 6),
            "label": "model sensitivity / counterfactual feature perturbation -- NOT clinical causality",
        })

    _perturb_and_measure("telehealth_available: flip 0<->1", "telehealth_available", lambda s: 1 - s)
    _perturb_and_measure("pcp_distance_miles: +1 mile (tiny)", "pcp_distance_miles", lambda s: s + 1.0)
    _perturb_and_measure("pcp_distance_miles: +5 miles", "pcp_distance_miles", lambda s: s + 5.0)
    _perturb_and_measure("pcp_distance_miles: +10 miles (large)", "pcp_distance_miles", lambda s: s + 10.0)
    _perturb_and_measure("urgent_care_distance_miles: +1 mile (tiny)", "urgent_care_distance_miles", lambda s: s + 1.0)
    _perturb_and_measure("urgent_care_distance_miles: +5 miles", "urgent_care_distance_miles", lambda s: s + 5.0)
    _perturb_and_measure("urgent_care_distance_miles: +10 miles (large)", "urgent_care_distance_miles", lambda s: s + 10.0)

    return pd.DataFrame(rows)


# =============================================================================
# STEP 9/10/11 -- Subgroup assessment, small-group protection, fairness interpretation
# =============================================================================

def subgroup_assessment(agent: RiskDetectionAgent, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = test_df[agent.feature_columns]
    y = test_df["future_potentially_avoidable_ed_90d"].astype(int).to_numpy()
    probs = agent.pipeline.predict_proba(X)[:, 1]

    groups: list[tuple[str, str, np.ndarray]] = []

    age_bins = [(18, 35, "age_18_34"), (35, 50, "age_35_49"), (50, 65, "age_50_64"), (65, 130, "age_65_plus")]
    for lo, hi, label in age_bins:
        groups.append(("age", label, ((test_df["age"] >= lo) & (test_df["age"] < hi)).to_numpy()))

    for g in sorted(test_df["gender"].dropna().unique()):
        groups.append(("gender", f"gender_{g}", (test_df["gender"] == g).to_numpy()))

    burden_bins = [(0, 1, "clinical_burden_0"), (1, 2, "clinical_burden_1"), (2, 3, "clinical_burden_2"), (3, 99, "clinical_burden_3_plus")]
    for lo, hi, label in burden_bins:
        groups.append(("clinical_burden", label, ((test_df["clinical_burden"] >= lo) & (test_df["clinical_burden"] < hi)).to_numpy()))

    for v in (0, 1):
        groups.append(("transportation_barrier", f"transportation_barrier_{v}", (test_df["transportation_barrier"] == v).to_numpy()))
        groups.append(("telehealth_available", f"telehealth_available_{v}", (test_df["telehealth_available"] == v).to_numpy()))

    dist_bins = [(0, 5, "pcp_distance_0_5"), (5, 10, "pcp_distance_5_10"), (10, 999, "pcp_distance_10_plus")]
    for lo, hi, label in dist_bins:
        groups.append(("pcp_distance_band", label, ((test_df["pcp_distance_miles"] >= lo) & (test_df["pcp_distance_miles"] < hi)).to_numpy()))

    rows = []
    for dimension, label, mask in groups:
        n = int(mask.sum())
        sufficient = n >= MIN_SUBGROUP_N
        m = group_metrics(y[mask], probs[mask], MODERATE_THRESHOLD)
        m50 = group_metrics(y[mask], probs[mask], 0.50)
        rows.append({
            "dimension": dimension, "subgroup": label, "sufficient_sample": sufficient,
            "n": n, "positives": m["positives"], "prevalence": m["prevalence"], "mean_probability": m["mean_probability"],
            "accuracy_at_0.50": m50.get("accuracy"), "accuracy_at_MODERATE": m.get("accuracy"),
            "balanced_accuracy_at_MODERATE": m.get("balanced_accuracy"),
            "precision_at_MODERATE": m.get("precision"), "recall_at_MODERATE": m.get("recall"),
            "specificity_at_MODERATE": m.get("specificity"), "f1_at_MODERATE": m.get("f1"),
            "fpr_at_MODERATE": m.get("fpr"), "fnr_at_MODERATE": m.get("fnr"),
            "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"), "brier": m.get("brier"),
            "moderate_plus_selection_rate": m.get("moderate_plus_selection_rate"),
            "high_selection_rate": m.get("high_selection_rate"),
            "note": None if sufficient else f"n<{MIN_SUBGROUP_N}: descriptive only / unstable, not used for disparity conclusions",
        })
    subgroup_df = pd.DataFrame(rows)

    # ---- pairwise disparity summary for the binary/paired dimensions ----
    disparity_rows = []

    def _pair(dimension: str, label_a: str, label_b: str):
        a = subgroup_df[subgroup_df["subgroup"] == label_a].iloc[0]
        b = subgroup_df[subgroup_df["subgroup"] == label_b].iloc[0]
        if not (a["sufficient_sample"] and b["sufficient_sample"]):
            return
        for metric in ("recall_at_MODERATE", "fpr_at_MODERATE", "fnr_at_MODERATE", "moderate_plus_selection_rate", "brier"):
            delta = (a[metric] if a[metric] is not None else float("nan")) - (b[metric] if b[metric] is not None else float("nan"))
            disparity_rows.append({
                "dimension": dimension, "group_a": label_a, "group_b": label_b, "metric": metric,
                "value_a": a[metric], "value_b": b[metric], "delta_a_minus_b": round(float(delta), 6) if delta == delta else None,
                "classification": classify_disparity(delta, metric) if delta == delta else "NO MATERIAL SIGNAL DETECTED",
            })

    _pair("transportation_barrier", "transportation_barrier_1", "transportation_barrier_0")
    _pair("telehealth_available", "telehealth_available_1", "telehealth_available_0")
    if {"gender_F", "gender_M"}.issubset(set(subgroup_df["subgroup"])):
        _pair("gender", "gender_F", "gender_M")
    age_labels = [l for _, l, _ in groups if l.startswith("age_")]
    for a, b in itertools.combinations(age_labels, 2):
        _pair("age", a, b)
    burden_labels = [l for _, l, _ in groups if l.startswith("clinical_burden_")]
    for a, b in itertools.combinations(burden_labels, 2):
        _pair("clinical_burden", a, b)

    disparity_df = pd.DataFrame(disparity_rows)
    return subgroup_df, disparity_df


# =============================================================================
# STEP 12-18 -- Navigation policy matrix, CM/telehealth/UC/PCP/no-nav validation,
# conflict scenarios
# =============================================================================

def navigation_policy_matrix() -> pd.DataFrame:
    rows = []

    def _add(scenario, tier, feats, why):
        nav = care_navigation.decide("SCEN", tier, pd.Series(feats))
        rows.append({
            "scenario": scenario, "risk_tier": tier.value, "inputs": json.dumps(feats),
            "destination": nav.destination.value, "reason_codes": [r.value for r in nav.reason_codes],
            "explanation": nav.explanation, "why_selected": why,
        })
        return nav

    # ---- Step 12: one representative scenario per reachable destination ----
    for label, (tier, feats) in NAV_SCENARIOS.items():
        _add(f"reachability: {label}", tier, feats, f"Constructed to satisfy exactly the {label} branch of the deterministic rule tree.")

    # ---- Step 13: Care Management validation ----
    _add("CM: HIGH risk only, no complexity/access/history", RiskTier.HIGH,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0},
         "EXPECTED: NOT CARE_MANAGEMENT -- risk alone never triggers CM (no complexity/access/history signal present); should fall through to PRIMARY_CARE.")
    _add("CM: HIGH + chronic complexity", RiskTier.HIGH,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 20,
          "clinical_burden": 2, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0},
         "EXPECTED: CARE_MANAGEMENT -- elevated risk + clinical_burden>=2 complexity signal.")
    _add("CM: HIGH + transportation barrier", RiskTier.HIGH,
         {"transportation_barrier": 1, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0},
         "EXPECTED: CARE_MANAGEMENT -- elevated risk + transportation_barrier complexity signal.")
    _add("CM: HIGH + previous CM engagement", RiskTier.HIGH,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 1},
         "EXPECTED: CARE_MANAGEMENT -- elevated risk + prior CM engagement complexity/continuity signal.")
    _add("CM: MODERATE + repeated utilization + access barrier", RiskTier.MODERATE,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 15, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 3, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0},
         "EXPECTED: CARE_MANAGEMENT -- repeated utilization (>=2) + pcp_distance>10 access-barrier complexity signal.")
    _add("CM: LOW + complexity only, no risk/utilization", RiskTier.LOW,
         {"transportation_barrier": 1, "telehealth_available": 0, "pcp_distance_miles": 15, "urgent_care_distance_miles": 20,
          "clinical_burden": 3, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: NOT CARE_MANAGEMENT -- complexity/access signals present but LOW risk + no repeated utilization; proves CM requires the risk/utilization side too, not complexity alone.")

    # ---- Step 14: Telehealth validation ----
    _add("Telehealth: available + transportation barrier", RiskTier.LOW,
         {"transportation_barrier": 1, "telehealth_available": 1, "pcp_distance_miles": 15, "urgent_care_distance_miles": 15,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: TELEHEALTH -- available and useful given the access barrier.")
    _add("Telehealth: unavailable + transportation barrier", RiskTier.LOW,
         {"transportation_barrier": 1, "telehealth_available": 0, "pcp_distance_miles": 15, "urgent_care_distance_miles": 15,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: NOT TELEHEALTH (unavailable) -- falls through toward NO_PROACTIVE_NAVIGATION (no elevated risk/utilization to trigger URGENT_CARE/PRIMARY_CARE either).")
    _add("Telehealth: available, no access barrier", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 1, "pcp_distance_miles": 3, "urgent_care_distance_miles": 5,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: NOT TELEHEALTH -- available but no access barrier (both distances <=10mi) makes it not useful; falls through toward NO_PROACTIVE_NAVIGATION.")
    _add("Telehealth: HIGH risk + telehealth available (with access barrier)", RiskTier.HIGH,
         {"transportation_barrier": 1, "telehealth_available": 1, "pcp_distance_miles": 15, "urgent_care_distance_miles": 15,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: CARE_MANAGEMENT, not TELEHEALTH -- elevated risk + transportation_barrier complexity signal outranks TELEHEALTH in priority order.")

    # ---- Step 15: Urgent care validation ----
    _add("UC: urgent meaningfully closer than PCP", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 8, "urgent_care_distance_miles": 2,
          "clinical_burden": 0, "prior_ed_count_270d": 2, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: URGENT_CARE -- urgent_care_distance < pcp_distance, PCP itself stays within the 10mi access-barrier threshold (so CARE_MANAGEMENT's LIMITED_PCP_ACCESS signal does not also fire), and repeated utilization provides a navigation opportunity.")
    _add("UC: urgent farther than PCP", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 8,
          "clinical_burden": 0, "prior_ed_count_270d": 2, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0},
         "EXPECTED: NOT URGENT_CARE (urgent not closer) -- falls to PRIMARY_CARE given close PCP access + continuity opportunity.")
    _add("UC: both nearby", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 2,
          "clinical_burden": 0, "prior_ed_count_270d": 2, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0},
         "EXPECTED: URGENT_CARE -- urgent_care_distance < pcp_distance still holds even though both are close.")
    _add("UC: both distant (>10mi) -- CARE_MANAGEMENT pre-empts URGENT_CARE", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 15, "urgent_care_distance_miles": 12,
          "clinical_burden": 0, "prior_ed_count_270d": 2, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: CARE_MANAGEMENT, not URGENT_CARE -- once pcp_distance itself exceeds the 10mi access-barrier threshold, CARE_MANAGEMENT's LIMITED_PCP_ACCESS complexity signal (combined with the repeated-utilization signal) takes priority over a plain URGENT_CARE suggestion, even though urgent care is closer than PCP. Documented system behavior: URGENT_CARE is only reachable when PCP access itself is within the access-barrier threshold.")
    _add("UC: missing distances (defaults to 99.0 = far, no other opportunity signal)", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: NOT URGENT_CARE -- both distances default to 99.0 (equal), so urgent_distance < pcp_distance is False; with no risk/utilization/complexity signal at all, falls through to NO_PROACTIVE_NAVIGATION (missing distances default to 'far', never to a falsely favorable value).")

    # ---- Step 16: Primary care validation ----
    _add("PCP: good access + recent engagement", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 4, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 2, "has_prior_care_management": 0},
         "EXPECTED: PRIMARY_CARE -- close access + continuity opportunity from recent PCP engagement.")
    _add("PCP: no prior engagement", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 4, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         "EXPECTED: NOT PRIMARY_CARE -- no continuity opportunity (prior_pcp_count=0 and prior_ed_count=0); falls to NO_PROACTIVE_NAVIGATION.")
    _add("PCP: telehealth unavailable, urgent care not advantageous", RiskTier.MODERATE,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 4, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 1, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0},
         "EXPECTED: PRIMARY_CARE -- MODERATE risk + close PCP access + continuity opportunity, and urgent care is not closer.")

    # ---- Step 17: NO_PROACTIVE_NAVIGATION validation ----
    _add("No-nav: LOW risk, no opportunity", RiskTier.LOW, NAV_SCENARIOS["NO_PROACTIVE_NAVIGATION"][1],
         "EXPECTED: NO_PROACTIVE_NAVIGATION -- proves the system does not force an intervention on every member.")

    return pd.DataFrame(rows)


def conflict_scenarios() -> pd.DataFrame:
    rows = []

    def _add(scenario, tier, feats, context_kwargs=None):
        nav = care_navigation.decide("SCEN", tier, pd.Series(feats))
        context = CurrentSafetyContext(**(context_kwargs or {}))
        safety, final_nav = safety_policy.decide(nav, context)
        rows.append({
            "scenario": scenario, "risk_tier": tier.value, "inputs": json.dumps(feats),
            "safety_context": json.dumps(context_kwargs or {}),
            "preliminary_destination": nav.destination.value,
            "safety_state": safety.state.value,
            "final_destination": final_nav.destination.value if final_nav.destination else None,
            "explanation": final_nav.explanation,
        })

    _add("HIGH risk + perfect PCP access + transportation barrier", RiskTier.HIGH,
         {"transportation_barrier": 1, "telehealth_available": 0, "pcp_distance_miles": 1, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 3, "has_prior_care_management": 0})
    _add("LOW risk + heavy chronic burden", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 20, "urgent_care_distance_miles": 20,
          "clinical_burden": 5, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0})
    _add("MODERATE risk + no outpatient access", RiskTier.MODERATE,
         {"transportation_barrier": 1, "telehealth_available": 0, "pcp_distance_miles": 30, "urgent_care_distance_miles": 30,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0})
    _add("HIGH risk + no prior utilization", RiskTier.HIGH,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 20, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0})
    _add("LOW risk + repeated care-management history", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 1})
    _add("Telehealth available but extreme PCP proximity", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 1, "pcp_distance_miles": 0.5, "urgent_care_distance_miles": 20,
          "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 1, "has_prior_care_management": 0})
    _add("Urgent care closer but emergency safety trigger present", RiskTier.LOW,
         {"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 12, "urgent_care_distance_miles": 2,
          "clinical_burden": 0, "prior_ed_count_270d": 2, "prior_potentially_avoidable_ed_count_270d": 0,
          "prior_pcp_count_270d": 0, "has_prior_care_management": 0},
         context_kwargs={"red_flag": 1})

    return pd.DataFrame(rows)


# =============================================================================
# STEP 19 -- Input validation (via FastAPI TestClient)
# =============================================================================

def input_validation_tests() -> pd.DataFrame:
    sys.path.insert(0, str(BACKEND_DIR))
    import main as main_mod
    from fastapi.testclient import TestClient

    client = TestClient(main_mod.app)
    members = pd.read_csv(SYNTHETIC_RAW_DIR / "raw_members.csv")
    ed = pd.read_csv(SYNTHETIC_RAW_DIR / "raw_ed_visits.csv")
    care = pd.read_csv(SYNTHETIC_RAW_DIR / "raw_care_history.csv")

    def _files(m=None, e=None, c=None):
        return {
            "members_file": ("m.csv", (m if m is not None else members).to_csv(index=False).encode(), "text/csv"),
            "ed_visits_file": ("e.csv", (e if e is not None else ed).to_csv(index=False).encode(), "text/csv"),
            "care_file": ("c.csv", (c if c is not None else care).to_csv(index=False).encode(), "text/csv"),
        }

    rows = []

    def _case(name, files=None, data=None, expect_clean_rejection=None):
        payload = {"member_id": "M00001", "index_date": "2026-07-03"}  # fixed date inside the synthetic data era,
        # so a malformed ED-visit-level row (e.g. an invalid triage_level) actually falls inside the observation
        # window and reaches classify_ed_encounters() -- an unpinned "today" index_date would non-deterministically
        # place such a row outside the window, silently skipping the very code path being tested.
        if data:
            payload.update(data)
        payload = {k: v for k, v in payload.items() if v is not None}  # None means "omit this field"
        try:
            resp = client.post("/uc07/decide", files=files or _files(), data=payload)
            status = resp.status_code
            body_ok = True
            try:
                resp.json()
            except Exception:
                body_ok = False
            rows.append({
                "case": name, "status_code": status,
                "clean_4xx_or_2xx": status < 500,
                "json_parseable": body_ok,
                "no_raw_traceback_leaked": "Traceback" not in resp.text,
            })
        except Exception as exc:
            rows.append({"case": name, "status_code": None, "clean_4xx_or_2xx": False, "json_parseable": False,
                          "no_raw_traceback_leaked": None, "unhandled_exception": str(exc)})

    # negative / extreme distances
    m1 = members.copy(); m1.loc[0, "pcp_distance_miles"] = -5.0
    _case("negative pcp_distance_miles", files=_files(m=m1))
    m2 = members.copy(); m2.loc[0, "urgent_care_distance_miles"] = 999999.0
    _case("extreme urgent_care_distance_miles", files=_files(m=m2))

    # invalid binary values
    m3 = members.copy(); m3.loc[0, "transportation_barrier"] = 7
    _case("invalid binary transportation_barrier=7", files=_files(m=m3))

    # invalid triage -- must land inside the observation window for
    # index_date=2026-07-03 (270 days back = 2025-10-07..2026-07-03) or
    # classify_ed_encounters() never sees it at all.
    e1 = ed.copy()
    in_window_idx = e1.index[(e1["visit_date"] >= "2025-10-07") & (e1["visit_date"] < "2026-07-03")][0]
    e1.loc[in_window_idx, "triage_level"] = 9
    _case("invalid triage_level=9 in ed_visits (in-window row)", files=_files(e=e1))

    # NaN in required numeric column
    m4 = members.copy(); m4.loc[0, "age"] = float("nan")
    _case("NaN age", files=_files(m=m4))

    # extreme age
    m5 = members.copy(); m5.loc[0, "age"] = -5
    _case("negative age", files=_files(m=m5))
    m6 = members.copy(); m6.loc[0, "age"] = 999
    _case("extreme age=999", files=_files(m=m6))

    # missing member entirely / unknown member id
    _case("unknown member_id", data={"member_id": "NOT_A_REAL_MEMBER"})

    # missing required column
    m7 = members.drop(columns=["transportation_barrier"])
    _case("missing required column transportation_barrier", files=_files(m=m7))

    # empty file
    _case("empty members file", files={
        "members_file": ("m.csv", b"", "text/csv"),
        "ed_visits_file": ("e.csv", ed.to_csv(index=False).encode(), "text/csv"),
        "care_file": ("c.csv", care.to_csv(index=False).encode(), "text/csv"),
    })

    # extra unexpected fields
    m8 = members.copy(); m8["unexpected_extra_column"] = "x"
    _case("extra unexpected column", files=_files(m=m8))

    # wrong / malformed current_safety_context values
    _case("current_safety_context invalid binary (2)", data={"member_id": "M00001", "current_safety_context": json.dumps({"M00001": {"red_flag": 2}})})
    _case("current_safety_context invalid triage (0)", data={"member_id": "M00001", "current_safety_context": json.dumps({"M00001": {"triage_level": 0}})})
    _case("current_safety_context invalid triage (6)", data={"member_id": "M00001", "current_safety_context": json.dumps({"M00001": {"triage_level": 6}})})
    _case("current_safety_context not an object", data={"member_id": "M00001", "current_safety_context": json.dumps([1, 2, 3])})

    # empty request (no member_id, defaults to full population -- valid, not an error)
    _case("no member_id (full population)", data={"member_id": None})

    return pd.DataFrame(rows)


# =============================================================================
# STEP 20/21 -- Probability/tier invariants + explanation validation
# =============================================================================

def probability_tier_invariants(agent: RiskDetectionAgent, test_df: pd.DataFrame) -> dict:
    X = test_df[agent.feature_columns]
    probs = agent.pipeline.predict_proba(X)[:, 1]

    bounds_ok = bool(((probs >= 0.0) & (probs <= 1.0)).all())

    tiers = []
    for p in probs:
        if p >= agent.high_threshold:
            tiers.append("HIGH")
        elif p >= agent.moderate_threshold:
            tiers.append("MODERATE")
        else:
            tiers.append("LOW")
    tiers = np.array(tiers)

    low_ok = bool((probs[tiers == "LOW"] < agent.moderate_threshold).all()) if (tiers == "LOW").any() else True
    mod_ok = bool(((probs[tiers == "MODERATE"] >= agent.moderate_threshold) & (probs[tiers == "MODERATE"] < agent.high_threshold)).all()) if (tiers == "MODERATE").any() else True
    high_ok = bool((probs[tiers == "HIGH"] >= agent.high_threshold).all()) if (tiers == "HIGH").any() else True

    thresholds_from_metadata = agent.moderate_threshold == MODERATE_THRESHOLD and agent.high_threshold == HIGH_THRESHOLD

    return {
        "n_scored": len(probs),
        "probability_bounds_ok": bounds_ok,
        "low_tier_invariant_ok": low_ok, "moderate_tier_invariant_ok": mod_ok, "high_tier_invariant_ok": high_ok,
        "thresholds_loaded_from_metadata_match": thresholds_from_metadata,
        "moderate_threshold": agent.moderate_threshold, "high_threshold": agent.high_threshold,
        "all_passed": bounds_ok and low_ok and mod_ok and high_ok and thresholds_from_metadata,
    }


def explanation_validation(agent: RiskDetectionAgent, test_df: pd.DataFrame) -> dict:
    X = test_df[agent.feature_columns].head(200)
    ids = test_df["member_id"].head(200)
    max_factors_seen = 0
    causal_violations = []
    leakage_violations = []
    forbidden_causal_phrases = ["will cause", "will result in", "guaranteed to", "definitely will"]
    forbidden_leakage_tokens = ["future_potentially_avoidable_ed_90d", "member_id", "index_date"]

    for i in range(len(X)):
        row = X.iloc[[i]]
        assessment = agent.assess(str(ids.iloc[i]), row, TEST_INDEX_DATE)
        max_factors_seen = max(max_factors_seen, len(assessment.contributing_factors))
        for factor in assessment.contributing_factors:
            lowered = factor.lower()
            for phrase in forbidden_causal_phrases:
                if phrase in lowered:
                    causal_violations.append({"member_id": ids.iloc[i], "factor": factor})
            for token in forbidden_leakage_tokens:
                if token in lowered:
                    leakage_violations.append({"member_id": ids.iloc[i], "factor": factor})

    return {
        "members_checked": len(X),
        "max_factor_count_observed": max_factors_seen,
        "max_factor_count_allowed": 3,
        "max_factor_count_respected": max_factors_seen <= 3,
        "causal_wording_violations": causal_violations,
        "leakage_token_violations": leakage_violations,
        "passed": max_factors_seen <= 3 and not causal_violations and not leakage_violations,
    }


# =============================================================================
# STEP 23/24 -- End-to-end population run + cross-agent consistency
# =============================================================================

def end_to_end_population_run(orchestrator: UC07Orchestrator, members: pd.DataFrame, ed: pd.DataFrame, care: pd.DataFrame) -> dict:
    decisions = orchestrator.decide_for_all_members(members, ed, care, TEST_INDEX_DATE, current_contexts={})

    tier_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    dest_counts = {d.value: 0 for d in NavigationDestination}
    dest_counts["NONE_OVERRIDE"] = 0
    safety_counts = {"CLEAR": 0, "CAUTION": 0, "OVERRIDE": 0}

    model_versions, dataset_ids, synthetic_flags, moderate_thrs, high_thrs = set(), set(), set(), set(), set()
    consistency_violations = []

    for d in decisions:
        tier_counts[d.risk.tier.value] += 1
        safety_counts[d.safety.state.value] += 1
        if d.navigation.destination is None:
            dest_counts["NONE_OVERRIDE"] += 1
        else:
            dest_counts[d.navigation.destination.value] += 1

        model_versions.add(d.risk.model_version)
        dataset_ids.add(d.risk.dataset_id)
        synthetic_flags.add(d.risk.synthetic_model)
        moderate_thrs.add(d.risk.moderate_threshold)
        high_thrs.add(d.risk.high_threshold)

        if not (0.0 <= d.risk.probability <= 1.0):
            consistency_violations.append({"member_id": d.member_id, "issue": "probability out of [0,1]"})
        if d.safety.state == SafetyState.OVERRIDE and d.navigation.destination is not None:
            consistency_violations.append({"member_id": d.member_id, "issue": "OVERRIDE but destination not suppressed"})

    return {
        "n_members": len(decisions),
        "risk_tier_counts": tier_counts,
        "navigation_destination_counts": dest_counts,
        "safety_state_counts": safety_counts,
        "note_on_safety_counts": "current_safety_context intentionally not supplied for the static population run (no such field exists in the frozen TEST snapshot); every decision therefore resolves to CAUTION unless already OVERRIDE-eligible from another path. Scenario-based safety testing (safety_override_matrix.csv / missing_safety_context_tests.csv) covers OVERRIDE/CLEAR explicitly.",
        "cross_agent_consistency": {
            "distinct_model_versions": list(model_versions), "distinct_dataset_ids": list(dataset_ids),
            "distinct_synthetic_flags": list(synthetic_flags), "distinct_moderate_thresholds": list(moderate_thrs),
            "distinct_high_thresholds": list(high_thrs),
            "all_consistent": len(model_versions) == 1 and len(dataset_ids) == 1 and len(synthetic_flags) == 1
            and len(moderate_thrs) == 1 and len(high_thrs) == 1,
            "violations": consistency_violations,
        },
    }


# =============================================================================
# STEP 27 -- Determinism
# =============================================================================

def determinism_check(orchestrator: UC07Orchestrator, members: pd.DataFrame, ed: pd.DataFrame, care: pd.DataFrame, n_members: int = 25, repeats: int = 5) -> dict:
    sample_ids = members["member_id"].head(n_members).tolist()
    mismatches = []
    for member_id in sample_ids:
        results = [orchestrator.decide_for_member(member_id, members, ed, care, TEST_INDEX_DATE) for _ in range(repeats)]
        first = decision_to_dict(results[0])
        for r in results[1:]:
            if decision_to_dict(r) != first:
                mismatches.append(member_id)
                break
    return {"n_members_checked": n_members, "repeats_per_member": repeats, "mismatches": mismatches, "deterministic": len(mismatches) == 0}


# =============================================================================
# main
# =============================================================================

def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 6: Safety + Fairness + Robustness + End-to-End Validation ===")

    hashes_before = hash_frozen_set()

    agent = RiskDetectionAgent()
    assert agent.model_version == "uc07-risk-synthetic-v1"
    assert agent.moderate_threshold == MODERATE_THRESHOLD
    assert agent.high_threshold == HIGH_THRESHOLD
    print(f"Frozen model confirmed: {agent.model_version} thresholds MODERATE={agent.moderate_threshold} HIGH={agent.high_threshold}")

    test_df = pd.read_csv(TEST_CSV)
    members = pd.read_csv(SYNTHETIC_RAW_DIR / "raw_members.csv")
    ed = pd.read_csv(SYNTHETIC_RAW_DIR / "raw_ed_visits.csv")
    care = pd.read_csv(SYNTHETIC_RAW_DIR / "raw_care_history.csv")
    orchestrator = UC07Orchestrator(risk_agent=agent)

    print("\n--- Step 2/3: safety override + missing-context matrices ---")
    override_df = build_safety_override_matrix()
    override_df.to_csv(EVAL_DIR / "safety_override_matrix.csv", index=False)
    missing_df = build_missing_context_matrix()
    missing_df.to_csv(EVAL_DIR / "missing_safety_context_tests.csv", index=False)
    print(f"Override matrix: {len(override_df)} rows, all passed={override_df['passed'].all()}")
    print(f"Missing-context matrix: {len(missing_df)} rows, all passed={missing_df['passed'].all()}")

    print("\n--- Step 5: prohibited language scan ---")
    language_report = scan_prohibited_language(agent, test_df)
    (EVAL_DIR / "prohibited_language_scan.json").write_text(json.dumps(language_report, indent=2, default=str))
    print(f"Scanned {language_report['total_checks']} texts, violations={language_report['violation_count']}")

    print("\n--- Step 6/7: transportation barrier investigation + counterfactual ---")
    transport_report = transportation_barrier_analysis(agent, test_df)
    (EVAL_DIR / "transportation_barrier_analysis.json").write_text(json.dumps(transport_report, indent=2, default=str))
    counterfactual_df = counterfactual_transportation(agent, test_df)
    counterfactual_df.to_csv(EVAL_DIR / "transportation_counterfactual_analysis.csv", index=False)
    print(f"Group 1 recall@MODERATE={transport_report['group_1']['at_MODERATE']['recall']} "
          f"Group 0 recall@MODERATE={transport_report['group_0']['at_MODERATE']['recall']}")

    print("\n--- Step 8: access feature sensitivity ---")
    access_df = access_feature_sensitivity(agent, test_df)
    access_df.to_csv(EVAL_DIR / "access_feature_sensitivity.csv", index=False)

    print("\n--- Step 9/10/11: subgroup assessment + fairness interpretation ---")
    subgroup_df, disparity_df = subgroup_assessment(agent, test_df)
    subgroup_df.to_csv(EVAL_DIR / "subgroup_metrics.csv", index=False)
    disparity_df.to_csv(EVAL_DIR / "subgroup_disparity_summary.csv", index=False)
    print(f"Subgroups evaluated: {len(subgroup_df)}; disparity comparisons: {len(disparity_df)}")

    print("\n--- Step 12-18: navigation policy matrix + conflict scenarios ---")
    nav_matrix_df = navigation_policy_matrix()
    nav_matrix_df.to_csv(EVAL_DIR / "navigation_policy_matrix.csv", index=False)
    conflict_df = conflict_scenarios()
    conflict_df.to_csv(EVAL_DIR / "conflict_scenarios.csv", index=False)

    print("\n--- Step 19: input validation ---")
    input_val_df = input_validation_tests()
    input_val_df.to_csv(EVAL_DIR / "input_validation_results.csv", index=False)
    print(f"Input validation cases: {len(input_val_df)}, all handled cleanly (status<500)={input_val_df['clean_4xx_or_2xx'].all()}")

    print("\n--- Step 20/21: probability/tier invariants + explanation validation ---")
    invariants_report = probability_tier_invariants(agent, test_df)
    explanation_report = explanation_validation(agent, test_df)
    (EVAL_DIR / "probability_tier_invariants.json").write_text(json.dumps(invariants_report, indent=2, default=str))
    (EVAL_DIR / "explanation_validation.json").write_text(json.dumps(explanation_report, indent=2, default=str))

    print("\n--- Step 22: synthetic disclosure ---")
    disclosure_report = {
        "synthetic_flag_true": agent.synthetic_model is True,
        "disclaimer_present": bool(agent.metadata.get("disclaimer")),
        "disclaimer_text": agent.metadata.get("disclaimer"),
        "intended_use": agent.metadata.get("intended_use"),
        "mentions_synthetic": "synthetic" in (agent.metadata.get("disclaimer") or "").lower(),
        "mentions_not_clinically_validated": "clinically validated" in (agent.metadata.get("disclaimer") or "").lower(),
    }
    disclosure_report["passed"] = (
        disclosure_report["synthetic_flag_true"] and disclosure_report["disclaimer_present"]
        and disclosure_report["mentions_synthetic"] and disclosure_report["mentions_not_clinically_validated"]
    )
    (EVAL_DIR / "synthetic_disclosure_check.json").write_text(json.dumps(disclosure_report, indent=2, default=str))

    print("\n--- Step 23/24: end-to-end population run + cross-agent consistency ---")
    population_report = end_to_end_population_run(orchestrator, members, ed, care)
    (EVAL_DIR / "population_decision_summary.json").write_text(json.dumps(population_report, indent=2, default=str))
    print(f"Population: n={population_report['n_members']} tiers={population_report['risk_tier_counts']}")
    print(f"Navigation destinations: {population_report['navigation_destination_counts']}")
    print(f"Safety states: {population_report['safety_state_counts']}")
    print(f"Cross-agent consistency all_consistent={population_report['cross_agent_consistency']['all_consistent']}")

    print("\n--- Step 27: determinism ---")
    determinism_report = determinism_check(orchestrator, members, ed, care)
    (EVAL_DIR / "determinism_check.json").write_text(json.dumps(determinism_report, indent=2, default=str))
    print(f"Determinism: {determinism_report['deterministic']}")

    hashes_after = hash_frozen_set()
    immutability_ok = hashes_after == hashes_before
    (EVAL_DIR / "immutability_check.json").write_text(json.dumps(
        {"hashes_before": hashes_before, "hashes_after": hashes_after, "unchanged": immutability_ok}, indent=2))
    print(f"\nImmutability check: {'PASS' if immutability_ok else 'FAIL'}")
    if not immutability_ok:
        raise SystemExit("CRITICAL: frozen files changed during Phase 6 validation run.")

    summary = {
        "phase": "6",
        "model_frozen": "uc07-risk-synthetic-v1",
        "moderate_threshold": MODERATE_THRESHOLD, "high_threshold": HIGH_THRESHOLD,
        "safety_override_matrix": {"rows": len(override_df), "all_passed": bool(override_df["passed"].all())},
        "missing_context_matrix": {"rows": len(missing_df), "all_passed": bool(missing_df["passed"].all())},
        "prohibited_language_violations": language_report["violation_count"],
        "transportation_barrier": {
            "group_0_recall_at_moderate": transport_report["group_0"]["at_MODERATE"]["recall"],
            "group_1_recall_at_moderate": transport_report["group_1"]["at_MODERATE"]["recall"],
        },
        "subgroups_evaluated": len(subgroup_df),
        "input_validation_all_clean": bool(input_val_df["clean_4xx_or_2xx"].all()),
        "probability_tier_invariants_passed": invariants_report["all_passed"],
        "explanation_validation_passed": explanation_report["passed"],
        "synthetic_disclosure_passed": disclosure_report["passed"],
        "population_run": {
            "n": population_report["n_members"],
            "cross_agent_consistency": population_report["cross_agent_consistency"]["all_consistent"],
        },
        "determinism": determinism_report["deterministic"],
        "immutability_ok": immutability_ok,
    }
    (EVAL_DIR / "phase6_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== Phase 6 validation script complete ===")
    return summary


if __name__ == "__main__":
    main()
