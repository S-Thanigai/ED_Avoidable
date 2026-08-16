"""
features.py
-----------
Point-in-time (observation-window-only) feature engineering for UC07
(Phase 3).

Every longitudinal feature in this module is computed strictly from
records with `observation_start <= event_date < index_date` for the
snapshot's own window (backend/pit/windows.py). No field from an
outcome-window (future) encounter, and no per-encounter severity field of
any kind (triage_level, red_flag, admitted, icu, major_procedure,
diagnosis, cost), ever enters this module's output -- those are
label-only (docs/02_UC07_AND_DATA_DESIGN.md section 12).

Diagnosis-derived predictors are intentionally EXCLUDED from this Phase 3
baseline feature set. Phase 1/2 found the previous implementation's
unwindowed `diagnosis_*` crosstab leaking utilization information (~36%
of that model's feature importance) because it was computed over ALL
encounters relative to a single global reference date rather than
strictly-prior encounters relative to each row's own index date.
Diagnosis's predictive *value* as a properly-windowed feature is
unverified; it is documented as a candidate for controlled future
experimentation (docs/02_UC07_AND_DATA_DESIGN.md section 13, "OPTIONAL"),
not part of this baseline.

`member_id` is an identifier, never a model feature. `index_date` is
snapshot metadata, never a model feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from encounter_classification import (
    POTENTIALLY_AVOIDABLE,
    PROTECTED_OR_HIGH_ACUITY,
    UNCERTAIN,
    classify_ed_encounters,
)
from windows import SnapshotWindow, in_observation_window

# Windows (days) for total and potentially-avoidable ED utilization counts.
# All fit inside the 270-day observation period.
ED_COUNT_WINDOWS_DAYS = (30, 90, 180, 270)

# Windows (days) for protected/uncertain ED subcounts. Restricted to a
# short + full-window pair (rather than the full 30/90/180/270 ladder used
# for total/potentially-avoidable counts) to avoid a proliferation of
# near-collinear, low-count features for states that are not themselves
# the primary navigation/utilization signal.
STATE_SUBCOUNT_WINDOWS_DAYS = (90, 270)

# Windows (days) for prior alternative-care utilization counts.
CARE_COUNT_WINDOWS_DAYS = (30, 90, 180, 270)

# Actual raw care_type category values present in raw_care_history.csv,
# mapped to the feature-name slug used below. No invented categories.
CARE_TYPE_TO_FEATURE_SLUG = {
    "PCP": "pcp",
    "Urgent Care": "urgent_care",
    "Telehealth": "telehealth",
    "Care Management": "care_management",
}

# Static member-level fields actually present in raw_members.csv (every
# column except the member_id identifier).
STATIC_MEMBER_COLUMNS = [
    "age",
    "gender",  # retained as a candidate predictor; flagged for
               # subgroup/fairness review in the feature manifest
               # (docs/02_UC07_AND_DATA_DESIGN.md section 13).
    "diabetes", "copd", "hypertension", "chf", "asthma", "ckd",
    "num_chronic_conditions",
    "transportation_barrier", "telehealth_available",
    "pcp_distance_miles", "urgent_care_distance_miles",
]
CLINICAL_FLAG_COLUMNS = ["diabetes", "copd", "hypertension", "chf", "asthma", "ckd"]


def _static_member_features(members: pd.DataFrame) -> pd.DataFrame:
    """Static, always-safe member fields (docs/02_UC07_AND_DATA_DESIGN.md
    section 13, MUST HAVE). No time dependency, no leakage risk."""
    missing = [c for c in STATIC_MEMBER_COLUMNS if c not in members.columns]
    if missing:
        raise KeyError(f"raw_members.csv is missing expected columns: {missing}")

    out = members[["member_id"] + STATIC_MEMBER_COLUMNS].drop_duplicates("member_id").copy()

    out["clinical_burden"] = out[CLINICAL_FLAG_COLUMNS].sum(axis=1)
    out["access_burden"] = (
        out["transportation_barrier"].fillna(0).astype(int)
        + (out["pcp_distance_miles"] > 10).astype(int)
        + (out["urgent_care_distance_miles"] > 10).astype(int)
    )
    return out


def _windowed_counts(events: pd.DataFrame, member_order: pd.Series, index_date: pd.Timestamp, windows_days) -> dict[int, pd.Series]:
    """For each window length, count events per member with
    visit_date >= index_date - window_days (events are already filtered
    to the observation window by the caller, so this only narrows further
    within it). Returns {days: Series aligned to member_order}."""
    result = {}
    for days in windows_days:
        cutoff = index_date - pd.Timedelta(days=days)
        subset = events.loc[events["visit_date"] >= cutoff]
        counts = subset.groupby("member_id").size().reindex(member_order).fillna(0).astype(int)
        result[days] = counts
    return result


def _recency(events: pd.DataFrame, member_order: pd.Series, index_date: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Days since the most recent event strictly before index_date, using
    ONLY events already filtered to the observation window (so recency is
    capped at the 270-day observation horizon by construction -- an event
    older than that is treated as "no prior event in this window", not
    reached via unbounded lookback). Returns (days_since, has_prior_flag).
    Explicit missing-value representation: NaN + a companion 0/1 flag,
    rather than a magic sentinel number."""
    last_visit = events.groupby("member_id")["visit_date"].max().reindex(member_order)
    days_since = (index_date - last_visit).dt.days
    has_prior = (~last_visit.isna()).astype(int)
    return days_since, has_prior


def _prior_ed_features(ed: pd.DataFrame, member_order: pd.Series, window: SnapshotWindow) -> pd.DataFrame:
    ed = ed.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="raise")
    prior = ed.loc[in_observation_window(ed["visit_date"], window)].copy()
    prior["encounter_state"] = classify_ed_encounters(prior)

    out = pd.DataFrame({"member_id": member_order})

    total_counts = _windowed_counts(prior, member_order, window.index_date, ED_COUNT_WINDOWS_DAYS)
    for days, counts in total_counts.items():
        out[f"prior_ed_count_{days}d"] = counts.values

    state_window_map = {
        POTENTIALLY_AVOIDABLE: ("potentially_avoidable", ED_COUNT_WINDOWS_DAYS),
        PROTECTED_OR_HIGH_ACUITY: ("protected", STATE_SUBCOUNT_WINDOWS_DAYS),
        UNCERTAIN: ("uncertain", STATE_SUBCOUNT_WINDOWS_DAYS),
    }
    for state, (slug, windows_days) in state_window_map.items():
        state_rows = prior.loc[prior["encounter_state"] == state]
        counts = _windowed_counts(state_rows, member_order, window.index_date, windows_days)
        for days, c in counts.items():
            out[f"prior_{slug}_ed_count_{days}d"] = c.values

    days_since_ed, has_prior_ed = _recency(prior, member_order, window.index_date)
    out["days_since_prior_ed"] = days_since_ed.values
    out["has_prior_ed"] = has_prior_ed.values

    avoidable_rows = prior.loc[prior["encounter_state"] == POTENTIALLY_AVOIDABLE]
    days_since_avoid, has_prior_avoid = _recency(avoidable_rows, member_order, window.index_date)
    out["days_since_prior_potentially_avoidable_ed"] = days_since_avoid.values
    out["has_prior_potentially_avoidable_ed"] = has_prior_avoid.values

    protected_rows = prior.loc[prior["encounter_state"] == PROTECTED_OR_HIGH_ACUITY]
    days_since_prot, has_prior_prot = _recency(protected_rows, member_order, window.index_date)
    out["days_since_prior_protected_ed"] = days_since_prot.values
    out["has_prior_protected_ed"] = has_prior_prot.values

    # Utilization velocity: small, interpretable, documented formulas only.
    # Denominator floored at 1 to avoid divide-by-zero; this makes the
    # ratio read as "recent count relative to a longer baseline count
    # (at least 1)", not a literal rate.
    out["ed_utilization_velocity_30_over_180"] = (
        out["prior_ed_count_30d"] / out["prior_ed_count_180d"].clip(lower=1)
    )
    out["potentially_avoidable_ed_velocity_90_over_270"] = (
        out["prior_potentially_avoidable_ed_count_90d"]
        / out["prior_potentially_avoidable_ed_count_270d"].clip(lower=1)
    )

    return out


def _prior_care_features(care: pd.DataFrame, member_order: pd.Series, window: SnapshotWindow) -> pd.DataFrame:
    care = care.copy()
    care["visit_date"] = pd.to_datetime(care["visit_date"], errors="raise")
    prior = care.loc[in_observation_window(care["visit_date"], window)].copy()

    unknown_types = sorted(set(prior["care_type"].unique()) - set(CARE_TYPE_TO_FEATURE_SLUG))
    if unknown_types:
        raise ValueError(f"Unrecognized care_type values in raw_care_history.csv: {unknown_types}")

    out = pd.DataFrame({"member_id": member_order})

    for care_type, slug in CARE_TYPE_TO_FEATURE_SLUG.items():
        type_rows = prior.loc[prior["care_type"] == care_type]

        counts = _windowed_counts(type_rows, member_order, window.index_date, CARE_COUNT_WINDOWS_DAYS)
        for days, c in counts.items():
            out[f"prior_{slug}_count_{days}d"] = c.values

        days_since, has_prior = _recency(type_rows, member_order, window.index_date)
        out[f"days_since_prior_{slug}"] = days_since.values
        out[f"has_prior_{slug}"] = has_prior.values

    return out


def build_observation_features(
    members: pd.DataFrame,
    ed: pd.DataFrame,
    care: pd.DataFrame,
    window: SnapshotWindow,
) -> pd.DataFrame:
    """
    Build the full point-in-time feature frame for one snapshot: one row
    per member in `members`, containing only static member fields and
    strictly-prior (observation-window) longitudinal features. Includes
    `member_id` (identifier) and `index_date` (metadata) columns for
    traceability -- neither is a model feature (see backend/pit/manifest.py).
    """
    member_order = members["member_id"].drop_duplicates().reset_index(drop=True)

    static = _static_member_features(members)
    ed_feats = _prior_ed_features(ed, member_order, window)
    care_feats = _prior_care_features(care, member_order, window)

    out = static.merge(ed_feats, on="member_id", how="left").merge(care_feats, on="member_id", how="left")
    out.insert(1, "index_date", window.index_date)
    return out
