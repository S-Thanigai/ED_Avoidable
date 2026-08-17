"""
Tests for backend/pit/validation.py.

Each check is exercised against BOTH clean data (must pass) and
intentionally corrupted/leaky data (must fail) -- this proves the checks
actually detect the problems they claim to detect, not just that they
always return True.
"""
import pandas as pd

import validation as val
from features import build_observation_features
from target import TARGET_COLUMN, build_member_target
from windows import build_snapshot_window

WINDOW = build_snapshot_window("train", "2025-10-05")


def _members(ids):
    return pd.DataFrame({
        "member_id": ids, "age": [50] * len(ids), "gender": ["F"] * len(ids),
        "diabetes": [0] * len(ids), "copd": [0] * len(ids), "hypertension": [0] * len(ids),
        "chf": [0] * len(ids), "asthma": [0] * len(ids), "ckd": [0] * len(ids),
        "num_chronic_conditions": [0] * len(ids), "transportation_barrier": [0] * len(ids),
        "telehealth_available": [1] * len(ids), "pcp_distance_miles": [2.0] * len(ids),
        "urgent_care_distance_miles": [2.0] * len(ids),
    })


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


def _care():
    return pd.DataFrame(columns=["care_id", "member_id", "visit_date", "care_type"])


# ---- schema / column checks: clean vs. corrupted ----

def test_forbidden_columns_check_passes_on_clean_snapshot():
    members = _members(["M1"])
    feats = build_observation_features(members, _ed([]), _care(), WINDOW)
    assert val.check_forbidden_columns_absent(feats.columns)["passed"]


def test_forbidden_columns_check_fails_when_triage_level_present():
    df = pd.DataFrame({"member_id": ["M1"], "triage_level": [4]})
    result = val.check_forbidden_columns_absent(df.columns)
    assert not result["passed"]
    assert "triage_level" in result["present"]


def test_legacy_target_absent_passes_on_clean_columns():
    assert val.check_legacy_target_absent(["member_id", TARGET_COLUMN])["passed"]


def test_legacy_target_absent_fails_when_present():
    result = val.check_legacy_target_absent(["member_id", "frequent_ED_user"])
    assert not result["passed"]


def test_diagnosis_crosstab_absent_passes_on_clean_columns():
    assert val.check_diagnosis_crosstab_absent(["prior_ed_count_30d"])["passed"]


def test_diagnosis_crosstab_absent_fails_when_diagnosis_columns_present():
    result = val.check_diagnosis_crosstab_absent(["diagnosis_UTI", "diagnosis_Fever"])
    assert not result["passed"]
    assert set(result["present"]) == {"diagnosis_UTI", "diagnosis_Fever"}


def test_identifier_not_feature_passes_when_excluded():
    assert val.check_identifier_not_feature(["age", "prior_ed_count_30d"], {"member_id"})["passed"]


def test_identifier_not_feature_fails_when_included():
    result = val.check_identifier_not_feature(["age", "member_id"], {"member_id"})
    assert not result["passed"]


def test_index_date_not_feature_fails_when_included():
    assert not val.check_index_date_not_feature(["age", "index_date"])["passed"]


def test_target_not_feature_fails_when_included():
    result = val.check_target_not_feature(["age", TARGET_COLUMN], {TARGET_COLUMN})
    assert not result["passed"]


# ---- data-quality checks: clean vs. corrupted ----

def test_no_duplicate_member_ids_passes_and_fails():
    clean = pd.DataFrame({"member_id": ["M1", "M2"]})
    dirty = pd.DataFrame({"member_id": ["M1", "M1"]})
    assert val.check_no_duplicate_member_ids(clean)["passed"]
    assert not val.check_no_duplicate_member_ids(dirty)["passed"]


def test_no_infinite_values_passes_and_fails():
    clean = pd.DataFrame({"x": [1.0, 2.0]})
    dirty = pd.DataFrame({"x": [1.0, float("inf")]})
    assert val.check_no_infinite_values(clean)["passed"]
    assert not val.check_no_infinite_values(dirty)["passed"]


def test_no_negative_counts_passes_and_fails():
    clean = pd.DataFrame({"prior_ed_count_30d": [0, 3]})
    dirty = pd.DataFrame({"prior_ed_count_30d": [0, -1]})
    assert val.check_no_negative_counts(clean)["passed"]
    assert not val.check_no_negative_counts(dirty)["passed"]


def test_no_negative_recency_passes_and_fails():
    clean = pd.DataFrame({"days_since_prior_ed": [5.0, None]})
    dirty = pd.DataFrame({"days_since_prior_ed": [5.0, -2.0]})
    assert val.check_no_negative_recency(clean)["passed"]
    assert not val.check_no_negative_recency(dirty)["passed"]


def test_schema_consistency_passes_and_fails():
    a = pd.DataFrame({"member_id": ["M1"], "age": [1]})
    b = pd.DataFrame({"member_id": ["M1"], "age": [1]})
    c = pd.DataFrame({"member_id": ["M1"], "age": [1], "extra": [1]})
    assert val.check_schema_consistency({"train": a, "validation": b})["passed"]
    result = val.check_schema_consistency({"train": a, "validation": c})
    assert not result["passed"]
    assert "extra" in result["mismatches"]["validation"]["extra_vs_reference"]


# ---- reconciliation: proves features.py/target.py boundary math is correct,
# and that the check catches a deliberately corrupted snapshot ----

def test_reconciliation_passes_on_genuine_pipeline_output():
    members = _members(["M1", "M2"])
    ed = _ed([
        {"member_id": "M1", "visit_date": "2025-09-01", "triage_level": 4},
        {"member_id": "M2", "visit_date": "2025-11-01", "triage_level": 5},  # outcome window -> positive
    ])
    feats = build_observation_features(members, ed, _care(), WINDOW)
    target_df, _ = build_member_target(members, ed, WINDOW)
    snapshot = feats.merge(target_df, on="member_id", how="left")

    result = val.check_snapshot_reconciliation(snapshot, members, ed, WINDOW, TARGET_COLUMN)
    assert result["passed"], result


def test_reconciliation_fails_on_corrupted_count_column():
    members = _members(["M1"])
    ed = _ed([{"member_id": "M1", "visit_date": "2025-09-01", "triage_level": 4}])
    feats = build_observation_features(members, ed, _care(), WINDOW)
    target_df, _ = build_member_target(members, ed, WINDOW)
    snapshot = feats.merge(target_df, on="member_id", how="left")

    corrupted = snapshot.copy()
    corrupted["prior_ed_count_270d"] = corrupted["prior_ed_count_270d"] + 5  # inject a bug

    result = val.check_snapshot_reconciliation(corrupted, members, ed, WINDOW, TARGET_COLUMN)
    assert not result["passed"]
    assert not result["details"]["prior_ed_count_270d"]["passed"]


def test_reconciliation_fails_on_corrupted_target():
    members = _members(["M1"])
    ed = _ed([{"member_id": "M1", "visit_date": "2025-09-01", "triage_level": 4}])
    feats = build_observation_features(members, ed, _care(), WINDOW)
    target_df, _ = build_member_target(members, ed, WINDOW)
    snapshot = feats.merge(target_df, on="member_id", how="left")

    corrupted = snapshot.copy()
    corrupted[TARGET_COLUMN] = 1 - corrupted[TARGET_COLUMN]  # flip the label

    result = val.check_snapshot_reconciliation(corrupted, members, ed, WINDOW, TARGET_COLUMN)
    assert not result["passed"]
    assert not result["details"][TARGET_COLUMN]["passed"]


def test_no_global_max_date_index_detects_bug_pattern():
    """If a snapshot's index_date were (incorrectly) set to the dataset's
    global max ED visit date, this check must fail -- that is exactly the
    Phase 1 leakage bug pattern this pipeline is designed to avoid."""
    ed = _ed([
        {"member_id": "M1", "visit_date": "2025-10-05", "triage_level": 4},
        {"member_id": "M1", "visit_date": "2026-04-03", "triage_level": 4},  # global max
    ])
    from windows import build_snapshot_window
    bad_window = build_snapshot_window("bogus", "2026-04-03")  # == global max date
    result = val.check_no_global_max_date_index(bad_window, ed)
    assert not result["passed"]
    assert result["equals_global_max_date"] is True

    good_window = build_snapshot_window("train", "2025-10-05")
    good_result = val.check_no_global_max_date_index(good_window, ed)
    assert good_result["passed"]
