"""
Phase 7 tests: static input validation (age/binary/distance/chronic-
count consistency), the safety-context completeness/provenance contract
(COMPLETE/PARTIAL/ABSENT, CALLER_SUPPLIED/NOT_AVAILABLE), invalid
safety-context value rejection, and the API validation matrix (Step 21).

Does not remove or weaken any Phase 5/6 test; safety invariants are
re-verified by backend/tests/test_phase6_safety_invariants.py, unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import safety_policy
from contracts import ContextCompleteness, ContextSource, CurrentSafetyContext, SafetyState
from input_validation import (
    EdVisitDataValidationError,
    MemberDataValidationError,
    validate_and_normalize_members_df,
    validate_ed_visits_df,
)
from safety_context_schema import SafetyContextEntry, SafetyContextPayload

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"


@pytest.fixture(scope="module")
def valid_members_df():
    return pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")


@pytest.fixture(scope="module")
def valid_ed_df():
    return pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")


@pytest.fixture(scope="module")
def client():
    import sys
    if str(REPO_ROOT / "backend") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "backend"))
    import main
    return TestClient(main.app)


def _files():
    return {
        "members_file": ("m.csv", (SYNTHETIC_DIR / "raw_members.csv").read_bytes(), "text/csv"),
        "ed_visits_file": ("e.csv", (SYNTHETIC_DIR / "raw_ed_visits.csv").read_bytes(), "text/csv"),
        "care_file": ("c.csv", (SYNTHETIC_DIR / "raw_care_history.csv").read_bytes(), "text/csv"),
    }


def _files_with_members(df: pd.DataFrame):
    return {
        "members_file": ("m.csv", df.to_csv(index=False).encode(), "text/csv"),
        "ed_visits_file": ("e.csv", (SYNTHETIC_DIR / "raw_ed_visits.csv").read_bytes(), "text/csv"),
        "care_file": ("c.csv", (SYNTHETIC_DIR / "raw_care_history.csv").read_bytes(), "text/csv"),
    }


def _files_with_ed(df: pd.DataFrame):
    return {
        "members_file": ("m.csv", (SYNTHETIC_DIR / "raw_members.csv").read_bytes(), "text/csv"),
        "ed_visits_file": ("e.csv", df.to_csv(index=False).encode(), "text/csv"),
        "care_file": ("c.csv", (SYNTHETIC_DIR / "raw_care_history.csv").read_bytes(), "text/csv"),
    }


# =============================================================================
# Static member input validation
# =============================================================================

def test_valid_members_df_passes_unchanged(valid_members_df):
    out = validate_and_normalize_members_df(valid_members_df)
    assert len(out) == len(valid_members_df)
    assert (out["num_chronic_conditions"] == valid_members_df["num_chronic_conditions"]).all()


def test_age_negative_rejected(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "age"] = -5
    with pytest.raises(MemberDataValidationError) as exc:
        validate_and_normalize_members_df(df)
    assert any("age" in issue for issue in exc.value.issues)


def test_age_above_max_rejected(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "age"] = 999
    with pytest.raises(MemberDataValidationError):
        validate_and_normalize_members_df(df)


def test_age_nan_rejected(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "age"] = float("nan")
    with pytest.raises(MemberDataValidationError):
        validate_and_normalize_members_df(df)


def test_age_non_integer_rejected(valid_members_df):
    df = valid_members_df.copy()
    df["age"] = df["age"].astype(object)
    df.loc[0, "age"] = 45.5
    with pytest.raises(MemberDataValidationError):
        validate_and_normalize_members_df(df)


def test_age_within_supported_range_at_boundaries_passes(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "age"] = 0
    df.loc[1, "age"] = 120
    out = validate_and_normalize_members_df(df)
    assert out.loc[0, "age"] == 0
    assert out.loc[1, "age"] == 120


@pytest.mark.parametrize("field_name", ["diabetes", "copd", "hypertension", "chf", "asthma", "ckd",
                                          "transportation_barrier", "telehealth_available"])
@pytest.mark.parametrize("bad_value", [2, -1, float("nan"), float("inf"), float("-inf"), "yes"])
def test_binary_member_field_invalid_values_rejected(valid_members_df, field_name, bad_value):
    df = valid_members_df.copy()
    df[field_name] = df[field_name].astype(object)
    df.loc[0, field_name] = bad_value
    with pytest.raises(MemberDataValidationError) as exc:
        validate_and_normalize_members_df(df)
    assert any(field_name in issue for issue in exc.value.issues)


@pytest.mark.parametrize("field_name", ["transportation_barrier", "telehealth_available"])
def test_binary_member_field_valid_values_pass(valid_members_df, field_name):
    df = valid_members_df.copy()
    df.loc[0, field_name] = 0
    df.loc[1, field_name] = 1
    validate_and_normalize_members_df(df)  # must not raise


@pytest.mark.parametrize("field_name", ["pcp_distance_miles", "urgent_care_distance_miles"])
@pytest.mark.parametrize("bad_value", [-5.0, 999999.0, float("nan"), float("inf")])
def test_distance_field_invalid_values_rejected(valid_members_df, field_name, bad_value):
    df = valid_members_df.copy()
    df.loc[0, field_name] = bad_value
    with pytest.raises(MemberDataValidationError) as exc:
        validate_and_normalize_members_df(df)
    assert any(field_name in issue for issue in exc.value.issues)


def test_distance_field_at_supported_boundaries_passes(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "pcp_distance_miles"] = 0.0
    df.loc[1, "pcp_distance_miles"] = 500.0
    out = validate_and_normalize_members_df(df)
    assert out.loc[0, "pcp_distance_miles"] == 0.0


# ---- chronic-count consistency ----

def test_num_chronic_conditions_derived_when_absent(valid_members_df):
    df = valid_members_df.drop(columns=["num_chronic_conditions"]).copy()
    out = validate_and_normalize_members_df(df)
    expected = df[["diabetes", "copd", "hypertension", "chf", "asthma", "ckd"]].sum(axis=1)
    assert (out["num_chronic_conditions"] == expected).all()


def test_num_chronic_conditions_consistent_value_passes(valid_members_df):
    validate_and_normalize_members_df(valid_members_df)  # must not raise -- source data is self-consistent


def test_num_chronic_conditions_inconsistent_value_rejected(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "num_chronic_conditions"] = df.loc[0, "num_chronic_conditions"] + 3
    with pytest.raises(MemberDataValidationError) as exc:
        validate_and_normalize_members_df(df)
    assert any("num_chronic_conditions" in issue and "does not equal the sum" in issue for issue in exc.value.issues)


def test_num_chronic_conditions_negative_rejected(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "num_chronic_conditions"] = -1
    with pytest.raises(MemberDataValidationError):
        validate_and_normalize_members_df(df)


def test_multiple_issues_all_collected_not_just_first(valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "age"] = -5
    df.loc[1, "pcp_distance_miles"] = -1.0
    df.loc[2, "transportation_barrier"] = 7
    with pytest.raises(MemberDataValidationError) as exc:
        validate_and_normalize_members_df(df)
    assert len(exc.value.issues) >= 3


# =============================================================================
# ED-visit binary + triage validation (closes the Phase 6 out-of-window gap)
# =============================================================================

def test_valid_ed_df_passes(valid_ed_df):
    validate_ed_visits_df(valid_ed_df)  # must not raise


def test_invalid_triage_level_rejected_regardless_of_observation_window(valid_ed_df):
    """Phase 6 found that an invalid triage_level outside every
    snapshot's observation window silently passed through (never reached
    classify_ed_encounters()). This validates every row, not just
    windowed ones."""
    df = valid_ed_df.copy()
    df.loc[0, "triage_level"] = 9  # row 0's visit_date (2025-01-01) is outside any snapshot's window
    with pytest.raises(EdVisitDataValidationError) as exc:
        validate_ed_visits_df(df)
    assert any("triage_level" in issue for issue in exc.value.issues)


@pytest.mark.parametrize("bad_triage", [0, 6, -1, float("nan")])
def test_various_invalid_triage_values_rejected(valid_ed_df, bad_triage):
    df = valid_ed_df.copy()
    df.loc[0, "triage_level"] = bad_triage
    with pytest.raises(EdVisitDataValidationError):
        validate_ed_visits_df(df)


@pytest.mark.parametrize("field_name", ["admitted", "icu", "major_procedure", "red_flag"])
def test_ed_binary_field_invalid_value_rejected(valid_ed_df, field_name):
    df = valid_ed_df.copy()
    df[field_name] = df[field_name].astype(object)
    df.loc[0, field_name] = 2
    with pytest.raises(EdVisitDataValidationError) as exc:
        validate_ed_visits_df(df)
    assert any(field_name in issue for issue in exc.value.issues)


# =============================================================================
# Safety-context contract: COMPLETE / PARTIAL / ABSENT, missing != false
# =============================================================================

def test_absent_context_completeness():
    assert CurrentSafetyContext().completeness == ContextCompleteness.ABSENT
    assert CurrentSafetyContext().source == ContextSource.NOT_AVAILABLE


def test_partial_context_completeness():
    assert CurrentSafetyContext(triage_level=4).completeness == ContextCompleteness.PARTIAL
    assert CurrentSafetyContext(triage_level=4).source == ContextSource.CALLER_SUPPLIED


def test_complete_context_completeness():
    ctx = CurrentSafetyContext(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4)
    assert ctx.completeness == ContextCompleteness.COMPLETE
    assert ctx.source == ContextSource.CALLER_SUPPLIED


def test_missing_field_is_none_not_zero():
    ctx = CurrentSafetyContext(red_flag=1)
    assert ctx.icu is None  # never coerced to 0
    assert ctx.admitted is None
    assert ctx.major_procedure is None
    assert ctx.triage_level is None


def test_safety_decision_carries_completeness_and_source_metadata():
    from contracts import NavigationDecision, NavigationDestination, ReasonCode
    nav = NavigationDecision(member_id="M1", destination=NavigationDestination.TELEHEALTH,
                              reason_codes=[ReasonCode.TELEHEALTH_AVAILABLE], explanation="x")
    safety, _ = safety_policy.decide(nav, CurrentSafetyContext(triage_level=4))
    assert safety.context_completeness == ContextCompleteness.PARTIAL
    assert safety.context_source == ContextSource.CALLER_SUPPLIED
    assert safety.state == SafetyState.CAUTION

    safety2, _ = safety_policy.decide(nav, CurrentSafetyContext())
    assert safety2.context_completeness == ContextCompleteness.ABSENT
    assert safety2.context_source == ContextSource.NOT_AVAILABLE

    safety3, _ = safety_policy.decide(nav, CurrentSafetyContext(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4))
    assert safety3.context_completeness == ContextCompleteness.COMPLETE
    assert safety3.state == SafetyState.CLEAR


def test_known_override_wins_even_with_partial_completeness():
    from contracts import NavigationDecision, NavigationDestination, ReasonCode
    nav = NavigationDecision(member_id="M1", destination=NavigationDestination.CARE_MANAGEMENT,
                              reason_codes=[ReasonCode.ELEVATED_FUTURE_RISK], explanation="x")
    safety, final_nav = safety_policy.decide(nav, CurrentSafetyContext(red_flag=1))
    assert safety.state == SafetyState.OVERRIDE
    assert safety.context_completeness == ContextCompleteness.PARTIAL  # only 1 of 5 fields known
    assert final_nav.destination is None


# =============================================================================
# Invalid safety-context values (Pydantic schema)
# =============================================================================

@pytest.mark.parametrize("bad_value", [2, -1, "yes", float("nan"), float("inf")])
def test_schema_rejects_invalid_binary_value(bad_value):
    with pytest.raises(ValidationError):
        SafetyContextEntry(red_flag=bad_value)


@pytest.mark.parametrize("bad_value", [0, 6, -1, "high", float("nan")])
def test_schema_rejects_invalid_triage_value(bad_value):
    with pytest.raises(ValidationError):
        SafetyContextEntry(triage_level=bad_value)


def test_schema_accepts_valid_complete_entry():
    entry = SafetyContextEntry(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4)
    assert entry.red_flag == 0 and entry.triage_level == 4


def test_schema_omitted_field_stays_none():
    entry = SafetyContextEntry(triage_level=4)
    assert entry.red_flag is None
    assert entry.icu is None


def test_schema_rejects_unknown_extra_field():
    with pytest.raises(ValidationError):
        SafetyContextEntry(red_flag=0, made_up_field=1)


def test_schema_rejects_non_dict_payload():
    with pytest.raises(ValidationError):
        SafetyContextPayload.model_validate([1, 2, 3])


def test_schema_boolean_binary_values_map_cleanly():
    entry = SafetyContextEntry(red_flag=True, icu=False)
    assert entry.red_flag == 1
    assert entry.icu == 0


# =============================================================================
# API validation matrix (Step 21)
# =============================================================================

def test_api_valid_complete_request_succeeds(client):
    resp = client.post("/uc07/decide", files=_files(), data={"member_id": "M00001", "index_date": "2026-07-03"})
    assert resp.status_code == 200


def test_api_valid_no_safety_context_succeeds(client):
    resp = client.post("/uc07/decide", files=_files(), data={"member_id": "M00001", "index_date": "2026-07-03"})
    assert resp.status_code == 200
    decision = resp.json()["decisions"][0]
    assert decision["safety"]["state"] == "CAUTION"
    assert decision["safety"]["context_completeness"] == "ABSENT"


def test_api_valid_partial_safety_context_succeeds_as_caution(client):
    context = json.dumps({"M00001": {"triage_level": 4}})
    resp = client.post("/uc07/decide", files=_files(), data={"member_id": "M00001", "index_date": "2026-07-03", "current_safety_context": context})
    assert resp.status_code == 200
    decision = resp.json()["decisions"][0]
    assert decision["safety"]["state"] == "CAUTION"
    assert decision["safety"]["context_completeness"] == "PARTIAL"


def test_api_valid_override_context_succeeds(client):
    context = json.dumps({"M00001": {"red_flag": 1}})
    resp = client.post("/uc07/decide", files=_files(), data={"member_id": "M00001", "index_date": "2026-07-03", "current_safety_context": context})
    assert resp.status_code == 200
    decision = resp.json()["decisions"][0]
    assert decision["safety"]["state"] == "OVERRIDE"
    assert decision["navigation"]["destination"] is None


def test_api_invalid_age_returns_422(client, valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "age"] = -5
    resp = client.post("/uc07/decide", files=_files_with_members(df), data={"member_id": df.loc[0, "member_id"]})
    assert resp.status_code == 422
    assert "Traceback" not in resp.text


def test_api_negative_distance_returns_422(client, valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "pcp_distance_miles"] = -5.0
    resp = client.post("/uc07/decide", files=_files_with_members(df), data={"member_id": df.loc[0, "member_id"]})
    assert resp.status_code == 422


def test_api_invalid_binary_flag_in_members_returns_422(client, valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "transportation_barrier"] = 7
    resp = client.post("/uc07/decide", files=_files_with_members(df), data={"member_id": df.loc[0, "member_id"]})
    assert resp.status_code == 422


def test_api_invalid_triage_in_ed_returns_422(client, valid_ed_df):
    df = valid_ed_df.copy()
    in_window_idx = df.index[(df["visit_date"] >= "2025-10-07") & (df["visit_date"] < "2026-07-03")][0]
    df.loc[in_window_idx, "triage_level"] = 9
    resp = client.post("/uc07/decide", files=_files_with_ed(df), data={"member_id": "M00001", "index_date": "2026-07-03"})
    assert resp.status_code == 422


def test_api_inconsistent_chronic_count_returns_422(client, valid_members_df):
    df = valid_members_df.copy()
    df.loc[0, "num_chronic_conditions"] = df.loc[0, "num_chronic_conditions"] + 5
    resp = client.post("/uc07/decide", files=_files_with_members(df), data={"member_id": df.loc[0, "member_id"]})
    assert resp.status_code == 422


def test_api_missing_num_chronic_conditions_column_is_safely_derived(client, valid_members_df):
    """Step 12: omitting a genuinely derivable column succeeds, rather
    than requiring a redundant client-supplied value."""
    df = valid_members_df.drop(columns=["num_chronic_conditions"])
    resp = client.post("/uc07/decide", files=_files_with_members(df), data={"member_id": df.loc[0, "member_id"]})
    assert resp.status_code == 200


def test_api_unknown_member_returns_404(client):
    resp = client.post("/uc07/decide", files=_files(), data={"member_id": "NOT_A_REAL_MEMBER"})
    assert resp.status_code == 404


def test_api_missing_required_column_returns_422(client, valid_members_df):
    df = valid_members_df.drop(columns=["transportation_barrier"])
    resp = client.post("/uc07/decide", files=_files_with_members(df), data={"member_id": df.loc[0, "member_id"]})
    assert resp.status_code == 422


def test_api_invalid_safety_context_triage_returns_422(client):
    context = json.dumps({"M00001": {"triage_level": 0}})
    resp = client.post("/uc07/decide", files=_files(), data={"member_id": "M00001", "current_safety_context": context})
    assert resp.status_code == 422


def test_api_safety_context_unknown_field_returns_422(client):
    context = json.dumps({"M00001": {"trige_level": 4}})  # typo -- must be rejected, not silently ignored
    resp = client.post("/uc07/decide", files=_files(), data={"member_id": "M00001", "current_safety_context": context})
    assert resp.status_code == 422


def test_api_no_response_leaks_a_traceback_across_the_validation_matrix(client, valid_members_df):
    cases = []
    df1 = valid_members_df.copy(); df1.loc[0, "age"] = -5
    cases.append(_files_with_members(df1))
    df2 = valid_members_df.copy(); df2.loc[0, "pcp_distance_miles"] = -1.0
    cases.append(_files_with_members(df2))
    for files in cases:
        resp = client.post("/uc07/decide", files=files, data={"member_id": "M00001"})
        assert "Traceback" not in resp.text
        assert resp.status_code < 500
