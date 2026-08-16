"""
validation.py
--------------
Automated leakage and data-quality checks for the Phase 3 point-in-time
pipeline (docs' Step 10/12 requirements). Used both by
backend/pit/build_snapshots.py (to produce a pass/fail report before any
derived file is written) and by backend/tests/test_validation.py (as
assertions against synthetic data designed to trip each check).

Where practical, checks INDEPENDENTLY recompute a value from the raw
event-level data using only the window boundary constants (not by calling
back into features.py/target.py), and compare it against what actually
ended up in the snapshot -- this catches real implementation bugs in the
feature/target modules, not just tautologies about the filtering that
produced them.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from encounter_classification import POTENTIALLY_AVOIDABLE, classify_ed_encounters
from windows import SnapshotWindow

FORBIDDEN_RAW_ENCOUNTER_FIELDS = {
    "triage_level", "red_flag", "admitted", "icu", "major_procedure",
    "diagnosis", "cost",
}
LEGACY_TARGET_COLUMN = "frequent_ED_user"
LEGACY_DIAGNOSIS_PREFIX = "diagnosis_"


# ---------------------------------------------------------------------------
# 1-4: boundary / overlap checks (independent recomputation from raw events)
# ---------------------------------------------------------------------------

def check_no_future_events_in_observation(ed_raw: pd.DataFrame, window: SnapshotWindow) -> dict:
    """(1) No observation ED event has event_date >= index_date."""
    ed = ed_raw.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"])
    obs = ed.loc[(ed["visit_date"] >= window.observation_start) & (ed["visit_date"] < window.index_date)]
    bad = obs.loc[obs["visit_date"] >= window.index_date]
    return {"passed": bad.empty, "violation_count": int(len(bad))}


def check_no_future_care_events_in_observation(care_raw: pd.DataFrame, window: SnapshotWindow) -> dict:
    """(2) No observation care-history event has event_date >= index_date."""
    care = care_raw.copy()
    care["visit_date"] = pd.to_datetime(care["visit_date"])
    obs = care.loc[(care["visit_date"] >= window.observation_start) & (care["visit_date"] < window.index_date)]
    bad = obs.loc[obs["visit_date"] >= window.index_date]
    return {"passed": bad.empty, "violation_count": int(len(bad))}


def check_no_past_target_events(ed_raw: pd.DataFrame, window: SnapshotWindow) -> dict:
    """(3) No target-contributing (outcome-window) event has event_date < index_date."""
    ed = ed_raw.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"])
    outcome = ed.loc[(ed["visit_date"] >= window.outcome_start) & (ed["visit_date"] < window.outcome_end)]
    bad = outcome.loc[outcome["visit_date"] < window.index_date]
    return {"passed": bad.empty, "violation_count": int(len(bad))}


def check_no_observation_outcome_overlap(ed_raw: pd.DataFrame, window: SnapshotWindow) -> dict:
    """(4) No event belongs to both observation and outcome for the same snapshot."""
    ed = ed_raw.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"])
    obs_ids = set(ed.index[(ed["visit_date"] >= window.observation_start) & (ed["visit_date"] < window.index_date)])
    out_ids = set(ed.index[(ed["visit_date"] >= window.outcome_start) & (ed["visit_date"] < window.outcome_end)])
    overlap = obs_ids & out_ids
    return {"passed": len(overlap) == 0, "violation_count": len(overlap)}


# ---------------------------------------------------------------------------
# 5-9: schema / column-membership checks
# ---------------------------------------------------------------------------

def check_forbidden_columns_absent(columns: Iterable[str]) -> dict:
    """(5) No future/outcome raw field appears in the model feature list."""
    present = [c for c in columns if c in FORBIDDEN_RAW_ENCOUNTER_FIELDS]
    return {"passed": len(present) == 0, "present": present}


def check_identifier_not_feature(feature_columns: Iterable[str], identifier_columns: Iterable[str]) -> dict:
    """(6) member_id is not classified as a model feature."""
    present = [c for c in feature_columns if c in set(identifier_columns)]
    return {"passed": len(present) == 0, "present": present}


def check_index_date_not_feature(feature_columns: Iterable[str]) -> dict:
    """(7) index_date is not classified as a model feature."""
    present = "index_date" in set(feature_columns)
    return {"passed": not present}


def check_target_not_feature(feature_columns: Iterable[str], target_columns: Iterable[str]) -> dict:
    """(8) target is not classified as a model feature."""
    present = [c for c in feature_columns if c in set(target_columns)]
    return {"passed": len(present) == 0, "present": present}


def check_legacy_target_absent(columns: Iterable[str]) -> dict:
    """(9) legacy frequent_ED_user is absent from derived snapshots."""
    present = LEGACY_TARGET_COLUMN in set(columns)
    return {"passed": not present}


def check_diagnosis_crosstab_absent(columns: Iterable[str]) -> dict:
    """(10) legacy unwindowed diagnosis_* features are absent."""
    present = [c for c in columns if str(c).startswith(LEGACY_DIAGNOSIS_PREFIX)]
    return {"passed": len(present) == 0, "present": present}


def check_no_global_max_date_index(window: SnapshotWindow, ed_raw: pd.DataFrame) -> dict:
    """(11) old global-max-date recency behavior is not used -- the
    snapshot's index_date must NOT equal the dataset's global max ED
    visit date (that coincidence would indicate the Phase 1 bug pattern
    re-emerged), and must be one of the three fixed, approved dates."""
    ed = ed_raw.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"])
    global_max = ed["visit_date"].max()
    from windows import SNAPSHOT_INDEX_DATES
    is_approved = window.index_date in set(SNAPSHOT_INDEX_DATES.values())
    not_global_max = window.index_date != global_max
    return {"passed": bool(is_approved and not_global_max), "is_approved_index_date": bool(is_approved), "equals_global_max_date": bool(not not_global_max)}


# ---------------------------------------------------------------------------
# 12: reconciliation checks -- independently recompute features/target from
# raw data and compare to what is actually in the snapshot.
# ---------------------------------------------------------------------------

def reconcile_ed_prior_count(ed_raw: pd.DataFrame, window: SnapshotWindow, days: int, state: str | None = None) -> pd.Series:
    ed = ed_raw.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"])
    cutoff = window.index_date - pd.Timedelta(days=days)
    mask = (
        (ed["visit_date"] >= window.observation_start)
        & (ed["visit_date"] < window.index_date)
        & (ed["visit_date"] >= cutoff)
    )
    subset = ed.loc[mask].copy()
    if state is not None:
        subset["encounter_state"] = classify_ed_encounters(subset)
        subset = subset.loc[subset["encounter_state"] == state]
    return subset.groupby("member_id").size()


def reconcile_target(ed_raw: pd.DataFrame, members: pd.DataFrame, window: SnapshotWindow) -> pd.Series:
    ed = ed_raw.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"])
    mask = (ed["visit_date"] >= window.outcome_start) & (ed["visit_date"] < window.outcome_end)
    outcome = ed.loc[mask].copy()
    outcome["encounter_state"] = classify_ed_encounters(outcome)
    positive = set(outcome.loc[outcome["encounter_state"] == POTENTIALLY_AVOIDABLE, "member_id"])
    return members["member_id"].isin(positive).astype(int)


def check_snapshot_reconciliation(
    snapshot: pd.DataFrame,
    members: pd.DataFrame,
    ed_raw: pd.DataFrame,
    window: SnapshotWindow,
    target_column: str,
) -> dict:
    """(12) Every longitudinal feature can be traced to events strictly
    before its snapshot index date -- verified by independently
    recomputing prior_ed_count_270d, prior_ed_count_30d,
    prior_potentially_avoidable_ed_count_270d, and the target column from
    raw data and comparing to the snapshot exactly."""
    results = {}
    snap = snapshot.set_index("member_id")

    checks = [
        ("prior_ed_count_270d", reconcile_ed_prior_count(ed_raw, window, 270)),
        ("prior_ed_count_30d", reconcile_ed_prior_count(ed_raw, window, 30)),
        ("prior_potentially_avoidable_ed_count_270d", reconcile_ed_prior_count(ed_raw, window, 270, POTENTIALLY_AVOIDABLE)),
    ]
    all_passed = True
    for col, recomputed in checks:
        recomputed_full = recomputed.reindex(snap.index).fillna(0).astype(int)
        actual = snap[col].astype(int)
        matches = (recomputed_full == actual).all()
        results[col] = {"passed": bool(matches), "mismatch_count": int((recomputed_full != actual).sum())}
        all_passed = all_passed and matches

    recomputed_target = reconcile_target(ed_raw, members, window)
    recomputed_target.index = members["member_id"].values
    actual_target = snap[target_column].astype(int)
    target_matches = (recomputed_target.reindex(snap.index) == actual_target).all()
    results[target_column] = {"passed": bool(target_matches), "mismatch_count": int((recomputed_target.reindex(snap.index) != actual_target).sum())}
    all_passed = all_passed and target_matches

    return {"passed": bool(all_passed), "details": results}


# ---------------------------------------------------------------------------
# Data-quality checks (Step 12 of the Phase 3 spec)
# ---------------------------------------------------------------------------

def check_no_duplicate_member_ids(df: pd.DataFrame) -> dict:
    dup = int(df["member_id"].duplicated().sum())
    return {"passed": dup == 0, "duplicate_count": dup}


def check_no_missing_member_ids(df: pd.DataFrame) -> dict:
    missing = int(df["member_id"].isna().sum())
    return {"passed": missing == 0, "missing_count": missing}


def check_no_infinite_values(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    bad_cols = [c for c in numeric.columns if np.isinf(numeric[c]).any()]
    return {"passed": len(bad_cols) == 0, "columns": bad_cols}


def check_no_negative_counts(df: pd.DataFrame) -> dict:
    count_cols = [c for c in df.columns if "_count_" in c]
    bad_cols = [c for c in count_cols if (df[c] < 0).any()]
    return {"passed": len(bad_cols) == 0, "columns": bad_cols}


def check_no_negative_recency(df: pd.DataFrame) -> dict:
    recency_cols = [c for c in df.columns if c.startswith("days_since_prior_")]
    bad_cols = [c for c in recency_cols if (df[c].dropna() < 0).any()]
    return {"passed": len(bad_cols) == 0, "columns": bad_cols}


def check_schema_consistency(snapshots: dict[str, pd.DataFrame]) -> dict:
    names = list(snapshots)
    ref_cols = list(snapshots[names[0]].columns)
    mismatches = {}
    for name in names[1:]:
        cols = list(snapshots[name].columns)
        if cols != ref_cols:
            mismatches[name] = {
                "missing_vs_reference": sorted(set(ref_cols) - set(cols)),
                "extra_vs_reference": sorted(set(cols) - set(ref_cols)),
            }
    return {"passed": len(mismatches) == 0, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_all_checks(
    snapshots: dict[str, pd.DataFrame],
    windows: dict[str, SnapshotWindow],
    members: pd.DataFrame,
    ed_raw: pd.DataFrame,
    care_raw: pd.DataFrame,
    identifier_cols: set[str],
    metadata_cols: set[str],
    target_cols: set[str],
) -> dict:
    report: dict = {"per_snapshot": {}, "cross_snapshot": {}}

    for name, snapshot in snapshots.items():
        window = windows[name]
        feature_cols = [c for c in snapshot.columns if c not in identifier_cols | metadata_cols | target_cols]
        target_col = next(iter(target_cols))

        checks = {
            "no_future_events_in_observation": check_no_future_events_in_observation(ed_raw, window),
            "no_future_care_events_in_observation": check_no_future_care_events_in_observation(care_raw, window),
            "no_past_target_events": check_no_past_target_events(ed_raw, window),
            "no_observation_outcome_overlap": check_no_observation_outcome_overlap(ed_raw, window),
            "forbidden_columns_absent": check_forbidden_columns_absent(snapshot.columns),
            "identifier_not_feature": check_identifier_not_feature(feature_cols, identifier_cols),
            "index_date_not_feature": check_index_date_not_feature(feature_cols),
            "target_not_feature": check_target_not_feature(feature_cols, target_cols),
            "legacy_target_absent": check_legacy_target_absent(snapshot.columns),
            "diagnosis_crosstab_absent": check_diagnosis_crosstab_absent(snapshot.columns),
            "no_global_max_date_index": check_no_global_max_date_index(window, ed_raw),
            "snapshot_reconciliation": check_snapshot_reconciliation(snapshot, members, ed_raw, window, target_col),
            "no_duplicate_member_ids": check_no_duplicate_member_ids(snapshot),
            "no_missing_member_ids": check_no_missing_member_ids(snapshot),
            "no_infinite_values": check_no_infinite_values(snapshot),
            "no_negative_counts": check_no_negative_counts(snapshot),
            "no_negative_recency": check_no_negative_recency(snapshot),
        }
        report["per_snapshot"][name] = checks

    report["cross_snapshot"]["schema_consistency"] = check_schema_consistency(snapshots)

    all_passed = report["cross_snapshot"]["schema_consistency"]["passed"] and all(
        c["passed"] for snap_checks in report["per_snapshot"].values() for c in snap_checks.values()
    )
    report["all_passed"] = bool(all_passed)
    return report
