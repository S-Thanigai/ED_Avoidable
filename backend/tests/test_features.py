"""Tests for backend/pit/features.py, using small synthetic in-memory data."""
import numpy as np
import pandas as pd

from features import build_observation_features
from windows import build_snapshot_window

WINDOW = build_snapshot_window("train", "2025-10-05")


def _members(rows):
    base = {
        "member_id": [], "age": [], "gender": [],
        "diabetes": [], "copd": [], "hypertension": [], "chf": [], "asthma": [], "ckd": [],
        "num_chronic_conditions": [], "transportation_barrier": [], "telehealth_available": [],
        "pcp_distance_miles": [], "urgent_care_distance_miles": [],
    }
    for r in rows:
        defaults = dict(age=50, gender="F", diabetes=0, copd=0, hypertension=0, chf=0, asthma=0, ckd=0,
                         num_chronic_conditions=0, transportation_barrier=0, telehealth_available=1,
                         pcp_distance_miles=2.0, urgent_care_distance_miles=2.0)
        defaults.update(r)
        for k, v in defaults.items():
            base[k].append(v)
    return pd.DataFrame(base)


def _ed(rows):
    cols = ["visit_id", "member_id", "visit_date", "triage_level", "red_flag", "admitted", "icu", "major_procedure"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = []
    for i, r in enumerate(rows):
        defaults = dict(visit_id=f"V{i}", red_flag=0, admitted=0, icu=0, major_procedure=0)
        defaults.update(r)
        out.append(defaults)
    return pd.DataFrame(out)[cols]


def _care(rows):
    cols = ["care_id", "member_id", "visit_date", "care_type"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = []
    for i, r in enumerate(rows):
        defaults = dict(care_id=f"C{i}")
        defaults.update(r)
        out.append(defaults)
    return pd.DataFrame(out)[cols]


def test_only_observation_window_ed_events_counted():
    members = _members([{"member_id": "M1"}])
    ed = _ed([
        {"member_id": "M1", "visit_date": "2025-08-01", "triage_level": 4},  # inside observation
        {"member_id": "M1", "visit_date": "2025-10-05", "triage_level": 4},  # == index_date, excluded (outcome, not observation)
        {"member_id": "M1", "visit_date": "2025-11-01", "triage_level": 4},  # future, excluded entirely
    ])
    care = _care([])
    feats = build_observation_features(members, ed, care, WINDOW)
    assert feats.loc[feats["member_id"] == "M1", "prior_ed_count_270d"].iloc[0] == 1


def test_event_before_observation_start_excluded():
    members = _members([{"member_id": "M1"}])
    too_old = WINDOW.observation_start - pd.Timedelta(days=5)
    ed = _ed([{"member_id": "M1", "visit_date": str(too_old.date()), "triage_level": 4}])
    feats = build_observation_features(members, ed, _care([]), WINDOW)
    assert feats.loc[feats["member_id"] == "M1", "prior_ed_count_270d"].iloc[0] == 0


def test_windowed_counts_respect_their_own_window_length():
    members = _members([{"member_id": "M1"}])
    ed = _ed([
        {"member_id": "M1", "visit_date": str((WINDOW.index_date - pd.Timedelta(days=10)).date()), "triage_level": 4},  # within 30d
        {"member_id": "M1", "visit_date": str((WINDOW.index_date - pd.Timedelta(days=100)).date()), "triage_level": 4},  # within 180d not 90d
        {"member_id": "M1", "visit_date": str((WINDOW.index_date - pd.Timedelta(days=250)).date()), "triage_level": 4},  # within 270d only
    ])
    feats = build_observation_features(members, ed, _care([]), WINDOW)
    row = feats.loc[feats["member_id"] == "M1"].iloc[0]
    assert row["prior_ed_count_30d"] == 1
    assert row["prior_ed_count_90d"] == 1
    assert row["prior_ed_count_180d"] == 2
    assert row["prior_ed_count_270d"] == 3


def test_recency_relative_to_index_date_not_global_max():
    members = _members([{"member_id": "M1"}])
    last_visit = WINDOW.index_date - pd.Timedelta(days=17)
    ed = _ed([
        {"member_id": "M1", "visit_date": str((WINDOW.index_date - pd.Timedelta(days=200)).date()), "triage_level": 4},
        {"member_id": "M1", "visit_date": str(last_visit.date()), "triage_level": 4},
    ])
    feats = build_observation_features(members, ed, _care([]), WINDOW)
    row = feats.loc[feats["member_id"] == "M1"].iloc[0]
    assert row["days_since_prior_ed"] == 17
    assert row["has_prior_ed"] == 1


def test_missing_history_uses_explicit_nan_and_zero_flag():
    members = _members([{"member_id": "M1"}])
    feats = build_observation_features(members, _ed([]), _care([]), WINDOW)
    row = feats.loc[feats["member_id"] == "M1"].iloc[0]
    assert pd.isna(row["days_since_prior_ed"])
    assert row["has_prior_ed"] == 0
    assert row["prior_ed_count_270d"] == 0


def test_care_history_only_observation_window_and_correct_type_mapping():
    members = _members([{"member_id": "M1"}])
    care = _care([
        {"member_id": "M1", "visit_date": "2025-08-01", "care_type": "PCP"},          # observation
        {"member_id": "M1", "visit_date": "2025-08-05", "care_type": "Urgent Care"},   # observation
        {"member_id": "M1", "visit_date": "2025-11-01", "care_type": "Telehealth"},    # future -> excluded
    ])
    feats = build_observation_features(members, _ed([]), care, WINDOW)
    row = feats.loc[feats["member_id"] == "M1"].iloc[0]
    assert row["prior_pcp_count_270d"] == 1
    assert row["prior_urgent_care_count_270d"] == 1
    assert row["prior_telehealth_count_270d"] == 0


def test_unknown_care_type_raises():
    members = _members([{"member_id": "M1"}])
    care = _care([{"member_id": "M1", "visit_date": "2025-08-01", "care_type": "Something Else"}])
    try:
        build_observation_features(members, _ed([]), care, WINDOW)
        assert False, "expected ValueError for unrecognized care_type"
    except ValueError:
        pass


def test_velocity_formula():
    members = _members([{"member_id": "M1"}])
    recent = WINDOW.index_date - pd.Timedelta(days=5)
    ed = _ed([{"member_id": "M1", "visit_date": str(recent.date()), "triage_level": 4}] * 3)
    feats = build_observation_features(members, ed, _care([]), WINDOW)
    row = feats.loc[feats["member_id"] == "M1"].iloc[0]
    expected = row["prior_ed_count_30d"] / max(row["prior_ed_count_180d"], 1)
    assert row["ed_utilization_velocity_30_over_180"] == expected


def test_no_diagnosis_columns_produced():
    members = _members([{"member_id": "M1"}])
    feats = build_observation_features(members, _ed([]), _care([]), WINDOW)
    diag_cols = [c for c in feats.columns if c.startswith("diagnosis_")]
    assert diag_cols == []


def test_no_forbidden_encounter_fields_in_output():
    members = _members([{"member_id": "M1"}])
    feats = build_observation_features(members, _ed([]), _care([]), WINDOW)
    forbidden = {"triage_level", "red_flag", "admitted", "icu", "major_procedure", "diagnosis", "cost"}
    assert forbidden.isdisjoint(set(feats.columns))


def test_static_features_present_and_no_infinite_values():
    members = _members([{"member_id": "M1", "diabetes": 1, "hypertension": 1}])
    feats = build_observation_features(members, _ed([]), _care([]), WINDOW)
    row = feats.loc[feats["member_id"] == "M1"].iloc[0]
    assert row["clinical_burden"] == 2
    numeric = feats.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()
