"""
manifest.py
-----------
Builds the machine-readable feature manifest (data/derived/feature_manifest.json)
describing every column in a Phase 3 snapshot: its source, category,
temporal window, static/longitudinal status, whether it is a model
candidate, its leakage status, and a human-readable description.

This manifest is intended to be packaged with the model in a later phase
so consumers never have to re-derive "is this column safe to use as a
feature" by reading source code.
"""
from __future__ import annotations

import re

import pandas as pd

IDENTIFIER_COLUMNS = {"member_id"}
METADATA_COLUMNS = {"index_date"}
TARGET_COLUMNS = {"future_potentially_avoidable_ed_90d"}

STATIC_DESCRIPTIONS = {
    "age": ("demographic", "raw_members.csv", "Member age in years."),
    "gender": ("demographic", "raw_members.csv", "Member gender (M/F). Retained as a candidate predictor; flagged for subgroup/fairness validation before production use -- see docs/02_UC07_AND_DATA_DESIGN.md section 13."),
    "diabetes": ("chronic_condition", "raw_members.csv", "Chronic diabetes flag (0/1)."),
    "copd": ("chronic_condition", "raw_members.csv", "Chronic COPD flag (0/1)."),
    "hypertension": ("chronic_condition", "raw_members.csv", "Chronic hypertension flag (0/1)."),
    "chf": ("chronic_condition", "raw_members.csv", "Chronic heart failure flag (0/1)."),
    "asthma": ("chronic_condition", "raw_members.csv", "Chronic asthma flag (0/1)."),
    "ckd": ("chronic_condition", "raw_members.csv", "Chronic kidney disease flag (0/1)."),
    "num_chronic_conditions": ("chronic_condition", "raw_members.csv", "Count of chronic conditions (redundant with the 6 individual flags; retained per Phase 2 approval -- monitor for multicollinearity in modeling)."),
    "clinical_burden": ("chronic_condition", "derived (raw_members.csv)", "Sum of the 6 chronic-condition flags."),
    "transportation_barrier": ("access", "raw_members.csv", "Member-reported transportation barrier flag (0/1)."),
    "telehealth_available": ("access", "raw_members.csv", "Whether telehealth is available to this member (0/1)."),
    "pcp_distance_miles": ("access", "raw_members.csv", "Distance in miles to the member's primary care provider."),
    "urgent_care_distance_miles": ("access", "raw_members.csv", "Distance in miles to the nearest urgent care."),
    "access_burden": ("access", "derived (raw_members.csv)", "transportation_barrier + (pcp_distance_miles>10) + (urgent_care_distance_miles>10)."),
}

STATE_SLUG_TO_DESC = {
    "potentially_avoidable": "encounters classified POTENTIALLY_AVOIDABLE",
    "protected": "encounters classified PROTECTED_OR_HIGH_ACUITY",
    "uncertain": "encounters classified UNCERTAIN",
}
CARE_SLUG_TO_LABEL = {
    "pcp": "Primary Care (PCP)",
    "urgent_care": "Urgent Care",
    "telehealth": "Telehealth",
    "care_management": "Care Management",
}


def _classify_column(name: str) -> dict:
    if name in IDENTIFIER_COLUMNS:
        return {
            "category": "identifier",
            "source_dataset": "raw_members.csv",
            "temporal_window": "n/a",
            "static_or_longitudinal": "static",
            "model_candidate": False,
            "leakage_status": "identifier_excluded",
            "description": "Member identifier. Never a model feature.",
        }
    if name in METADATA_COLUMNS:
        return {
            "category": "metadata",
            "source_dataset": "derived (snapshot definition)",
            "temporal_window": "n/a",
            "static_or_longitudinal": "static",
            "model_candidate": False,
            "leakage_status": "metadata_excluded",
            "description": "The snapshot's fixed point-in-time index date. Never a model feature.",
        }
    if name in TARGET_COLUMNS:
        return {
            "category": "target",
            "source_dataset": "derived (raw_ed_visits.csv, outcome window)",
            "temporal_window": "outcome (90d forward)",
            "static_or_longitudinal": "n/a",
            "model_candidate": False,
            "leakage_status": "target_outcome_window",
            "description": "1 if the member has >=1 POTENTIALLY_AVOIDABLE ED encounter in the 90-day outcome window after index_date, else 0.",
        }
    if name in STATIC_DESCRIPTIONS:
        category, source, desc = STATIC_DESCRIPTIONS[name]
        return {
            "category": category,
            "source_dataset": source,
            "temporal_window": "static",
            "static_or_longitudinal": "static",
            "model_candidate": True,
            "leakage_status": "safe_static",
            "description": desc,
        }

    m = re.match(r"^prior_(potentially_avoidable|protected|uncertain)_ed_count_(\d+)d$", name)
    if m:
        state_slug, days = m.groups()
        return {
            "category": "prior_ed_utilization",
            "source_dataset": "raw_ed_visits.csv",
            "temporal_window": f"{days}d (observation, pre-index)",
            "static_or_longitudinal": "longitudinal",
            "model_candidate": True,
            "leakage_status": "safe_historical_pre_index",
            "description": f"Count of prior ED {STATE_SLUG_TO_DESC[state_slug]} in the {days}-day window strictly before index_date.",
        }

    m = re.match(r"^prior_ed_count_(\d+)d$", name)
    if m:
        days = m.group(1)
        return {
            "category": "prior_ed_utilization",
            "source_dataset": "raw_ed_visits.csv",
            "temporal_window": f"{days}d (observation, pre-index)",
            "static_or_longitudinal": "longitudinal",
            "model_candidate": True,
            "leakage_status": "safe_historical_pre_index",
            "description": f"Total count of ED encounters (any state) in the {days}-day window strictly before index_date.",
        }

    if name in ("days_since_prior_ed", "days_since_prior_potentially_avoidable_ed", "days_since_prior_protected_ed"):
        which = {
            "days_since_prior_ed": "any ED encounter",
            "days_since_prior_potentially_avoidable_ed": "an ED encounter classified POTENTIALLY_AVOIDABLE",
            "days_since_prior_protected_ed": "an ED encounter classified PROTECTED_OR_HIGH_ACUITY",
        }[name]
        return {
            "category": "prior_ed_recency",
            "source_dataset": "raw_ed_visits.csv",
            "temporal_window": "270d (observation, pre-index)",
            "static_or_longitudinal": "longitudinal",
            "model_candidate": True,
            "leakage_status": "safe_historical_pre_index",
            "description": f"Days between the member's most recent {which} strictly before index_date and index_date, capped by the 270-day observation window. NaN if no such encounter falls within the observation window (see companion has_prior_* flag).",
        }

    HAS_PRIOR_ED_FLAGS = {"has_prior_ed", "has_prior_potentially_avoidable_ed", "has_prior_protected_ed"}
    HAS_PRIOR_CARE_FLAGS = {f"has_prior_{slug}" for slug in CARE_SLUG_TO_LABEL}
    if name in HAS_PRIOR_ED_FLAGS or name in HAS_PRIOR_CARE_FLAGS:
        is_ed = name in HAS_PRIOR_ED_FLAGS
        return {
            "category": "prior_ed_recency" if is_ed else "prior_care_recency",
            "source_dataset": "raw_ed_visits.csv" if is_ed else "raw_care_history.csv",
            "temporal_window": "270d (observation, pre-index)",
            "static_or_longitudinal": "longitudinal",
            "model_candidate": True,
            "leakage_status": "safe_historical_pre_index",
            "description": f"Explicit 0/1 flag: whether {name.replace('has_prior_', '').replace('_', ' ')} has >=1 qualifying prior event within the 270-day observation window. Companion to the paired days_since_prior_* column's NaN representation.",
        }

    m = re.match(r"^prior_(pcp|urgent_care|telehealth|care_management)_count_(\d+)d$", name)
    if m:
        slug, days = m.groups()
        return {
            "category": "prior_care_utilization",
            "source_dataset": "raw_care_history.csv",
            "temporal_window": f"{days}d (observation, pre-index)",
            "static_or_longitudinal": "longitudinal",
            "model_candidate": True,
            "leakage_status": "safe_historical_pre_index",
            "description": f"Count of prior {CARE_SLUG_TO_LABEL[slug]} care-history visits in the {days}-day window strictly before index_date.",
        }

    m = re.match(r"^days_since_prior_(pcp|urgent_care|telehealth|care_management)$", name)
    if m:
        slug = m.group(1)
        return {
            "category": "prior_care_recency",
            "source_dataset": "raw_care_history.csv",
            "temporal_window": "270d (observation, pre-index)",
            "static_or_longitudinal": "longitudinal",
            "model_candidate": True,
            "leakage_status": "safe_historical_pre_index",
            "description": f"Days between the member's most recent {CARE_SLUG_TO_LABEL[slug]} visit strictly before index_date and index_date, capped by the 270-day observation window. NaN if none within the window (see companion has_prior_{slug} flag).",
        }

    if name in ("ed_utilization_velocity_30_over_180", "potentially_avoidable_ed_velocity_90_over_270"):
        formula = {
            "ed_utilization_velocity_30_over_180": "prior_ed_count_30d / max(prior_ed_count_180d, 1)",
            "potentially_avoidable_ed_velocity_90_over_270": "prior_potentially_avoidable_ed_count_90d / max(prior_potentially_avoidable_ed_count_270d, 1)",
        }[name]
        return {
            "category": "velocity",
            "source_dataset": "derived (raw_ed_visits.csv)",
            "temporal_window": "270d (observation, pre-index)",
            "static_or_longitudinal": "longitudinal",
            "model_candidate": True,
            "leakage_status": "safe_historical_pre_index",
            "description": f"Recent-vs-baseline utilization ratio. Formula: {formula}.",
        }

    return {
        "category": "unclassified",
        "source_dataset": "unknown",
        "temporal_window": "unknown",
        "static_or_longitudinal": "unknown",
        "model_candidate": False,
        "leakage_status": "unclassified_review_required",
        "description": "Column did not match any known naming pattern -- flagged for manual review before use.",
    }


def build_feature_manifest(snapshot: pd.DataFrame) -> dict:
    entries = []
    for col in snapshot.columns:
        info = _classify_column(col)
        entries.append({"feature_name": col, **info})

    return {
        "identifier_columns": sorted(IDENTIFIER_COLUMNS),
        "metadata_columns": sorted(METADATA_COLUMNS),
        "target_columns": sorted(TARGET_COLUMNS),
        "fairness_audit_columns": ["gender"],
        "excluded_feature_groups": [
            {
                "group": "diagnosis_*",
                "reason": "Excluded from the Phase 3 baseline feature set. Phase 1 found the previous unwindowed diagnosis crosstab leaking ED-utilization volume (~36% of that model's feature importance); Phase 2 additionally verified diagnosis carries no measurable acuity signal in this dataset. Predictive value as a properly point-in-time-windowed feature is unverified and is a candidate for controlled future experimentation, not part of this baseline.",
            },
            {
                "group": "outcome-window encounter fields",
                "reason": "triage_level, red_flag, admitted, icu, major_procedure, diagnosis, and cost of outcome-window encounters are label-only by design (docs/02_UC07_AND_DATA_DESIGN.md section 12) and never appear as features.",
            },
        ],
        "features": entries,
    }
