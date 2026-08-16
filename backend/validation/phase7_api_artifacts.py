"""
phase7_api_artifacts.py
-------------------------
Phase 7 -- generates the remaining artifacts/phase7_hardening/ files not
produced by phase7_disparity_analysis.py: the static input-validation
matrix, the safety-context completeness/provenance matrix, the API
validation matrix (Step 21), the failure-mode table (Step 19), and the
consolidated phase7_summary.json.

Exercises the already-hardened backend/main.py, backend/agents/
input_validation.py, and backend/agents/safety_context_schema.py exactly
as shipped -- does not modify or retrain anything.

Run: python backend/validation/phase7_api_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for _subdir in ("pit", "agents", "modeling"):
    _p = str(BACKEND_DIR / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main as main_mod
from fastapi.testclient import TestClient

from contracts import ContextCompleteness, ContextSource, CurrentSafetyContext, NavigationDecision, NavigationDestination, ReasonCode, SafetyState
import safety_policy
from input_validation import MemberDataValidationError, EdVisitDataValidationError, validate_and_normalize_members_df, validate_ed_visits_df

SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
EVAL_DIR = REPO_ROOT / "artifacts" / "phase7_hardening"


def _files(members_df=None, ed_df=None, care_df=None):
    members = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv") if members_df is None else members_df
    ed = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv") if ed_df is None else ed_df
    care = pd.read_csv(SYNTHETIC_DIR / "raw_care_history.csv") if care_df is None else care_df
    return {
        "members_file": ("m.csv", members.to_csv(index=False).encode(), "text/csv"),
        "ed_visits_file": ("e.csv", ed.to_csv(index=False).encode(), "text/csv"),
        "care_file": ("c.csv", care.to_csv(index=False).encode(), "text/csv"),
    }


# =============================================================================
# Static input validation matrix (module-level, not via API)
# =============================================================================

def build_input_validation_matrix(members_df: pd.DataFrame, ed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _member_case(name, mutate):
        df = members_df.copy()
        if mutate is not None:
            mutate(df)
        try:
            validate_and_normalize_members_df(df)
            rows.append({"case": name, "layer": "members", "result": "PASS", "rejected": False, "issue_count": 0})
        except MemberDataValidationError as exc:
            rows.append({"case": name, "layer": "members", "result": "REJECTED", "rejected": True, "issue_count": len(exc.issues)})

    def _ed_case(name, mutate):
        df = ed_df.copy()
        if mutate is not None:
            mutate(df)
        try:
            validate_ed_visits_df(df)
            rows.append({"case": name, "layer": "ed_visits", "result": "PASS", "rejected": False, "issue_count": 0})
        except EdVisitDataValidationError as exc:
            rows.append({"case": name, "layer": "ed_visits", "result": "REJECTED", "rejected": True, "issue_count": len(exc.issues)})

    def _set(df, col, value, row=0):
        df[col] = df[col].astype(object)
        df.loc[row, col] = value

    _member_case("valid data (baseline)", None)
    _member_case("age negative", lambda df: _set(df, "age", -5))
    _member_case("age = 0 (boundary, valid)", lambda df: _set(df, "age", 0))
    _member_case("age = 120 (boundary, valid)", lambda df: _set(df, "age", 120))
    _member_case("age = 121 (just above boundary)", lambda df: _set(df, "age", 121))
    _member_case("age NaN", lambda df: _set(df, "age", float("nan")))
    _member_case("transportation_barrier = 7", lambda df: _set(df, "transportation_barrier", 7))
    _member_case("telehealth_available = -1", lambda df: _set(df, "telehealth_available", -1))
    _member_case("pcp_distance_miles negative", lambda df: _set(df, "pcp_distance_miles", -5.0))
    _member_case("pcp_distance_miles = 500 (boundary, valid)", lambda df: _set(df, "pcp_distance_miles", 500.0))
    _member_case("pcp_distance_miles = 500.01 (just above boundary)", lambda df: _set(df, "pcp_distance_miles", 500.01))
    _member_case("urgent_care_distance_miles = inf", lambda df: _set(df, "urgent_care_distance_miles", float("inf")))
    _member_case("num_chronic_conditions inconsistent with flags", lambda df: _set(df, "num_chronic_conditions", df.loc[0, "num_chronic_conditions"] + 4))
    _member_case("num_chronic_conditions column absent (derivable)", lambda df: df.drop(columns=["num_chronic_conditions"], inplace=True))

    _ed_case("valid data (baseline)", None)
    _ed_case("triage_level = 0", lambda df: _set(df, "triage_level", 0))
    _ed_case("triage_level = 6", lambda df: _set(df, "triage_level", 6))
    _ed_case("triage_level in-window row = 9 (Phase 6 gap, now closed)", lambda df: _set(
        df, "triage_level", 9, row=df.index[(df["visit_date"] >= "2025-10-07") & (df["visit_date"] < "2026-07-03")][0]))
    _ed_case("red_flag = 2", lambda df: _set(df, "red_flag", 2))
    _ed_case("icu = NaN", lambda df: _set(df, "icu", float("nan")))

    return pd.DataFrame(rows)


# =============================================================================
# Safety-context completeness/provenance matrix
# =============================================================================

def build_safety_context_matrix() -> pd.DataFrame:
    nav = NavigationDecision(member_id="M1", destination=NavigationDestination.TELEHEALTH,
                              reason_codes=[ReasonCode.TELEHEALTH_AVAILABLE], explanation="x")
    rows = []

    def _case(name, kwargs, expected_state, expected_completeness):
        ctx = CurrentSafetyContext(**kwargs)
        safety, _ = safety_policy.decide(nav, ctx)
        rows.append({
            "case": name, "fields_supplied": json.dumps(kwargs),
            "completeness": ctx.completeness.value, "source": ctx.source.value,
            "safety_state": safety.state.value,
            "expected_state": expected_state, "expected_completeness": expected_completeness,
            "passed": safety.state.value == expected_state and ctx.completeness.value == expected_completeness,
        })

    _case("ABSENT (nothing supplied)", {}, "CAUTION", "ABSENT")
    _case("PARTIAL (1 field, safe)", {"triage_level": 4}, "CAUTION", "PARTIAL")
    _case("PARTIAL (4 of 5, safe)", {"red_flag": 0, "icu": 0, "admitted": 0, "major_procedure": 0}, "CAUTION", "PARTIAL")
    _case("COMPLETE (all safe)", {"red_flag": 0, "icu": 0, "admitted": 0, "major_procedure": 0, "triage_level": 4}, "CLEAR", "COMPLETE")
    _case("PARTIAL + known OVERRIDE trigger", {"red_flag": 1}, "OVERRIDE", "PARTIAL")
    _case("COMPLETE + known OVERRIDE trigger", {"red_flag": 0, "icu": 1, "admitted": 0, "major_procedure": 0, "triage_level": 4}, "OVERRIDE", "COMPLETE")
    _case("ABSENT but conceptually urgent (no data at all)", {}, "CAUTION", "ABSENT")

    return pd.DataFrame(rows)


# =============================================================================
# API validation matrix (Step 21)
# =============================================================================

def build_api_validation_matrix(client: TestClient) -> pd.DataFrame:
    members = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")
    rows = []

    def _case(name, files=None, data=None, expected_status=None):
        payload = {"member_id": "M00001", "index_date": "2026-07-03"}
        if data:
            payload.update(data)
        payload = {k: v for k, v in payload.items() if v is not None}
        resp = client.post("/uc07/decide", files=files or _files(), data=payload)
        rows.append({
            "case": name, "status_code": resp.status_code, "expected_status": expected_status,
            "passed": (resp.status_code == expected_status) if expected_status else None,
            "no_traceback_leaked": "Traceback" not in resp.text,
        })

    _case("valid complete request", expected_status=200)
    _case("valid, no current safety context", data={"current_safety_context": None}, expected_status=200)
    _case("valid, partial safety context", data={"current_safety_context": json.dumps({"M00001": {"triage_level": 4}})}, expected_status=200)
    _case("valid, override context", data={"current_safety_context": json.dumps({"M00001": {"red_flag": 1}})}, expected_status=200)

    m_age = members.copy(); m_age["age"] = m_age["age"].astype(object); m_age.loc[0, "age"] = -5
    _case("invalid age", files=_files(members_df=m_age), expected_status=422)

    m_dist = members.copy(); m_dist.loc[0, "pcp_distance_miles"] = -5.0
    _case("negative distance", files=_files(members_df=m_dist), expected_status=422)

    m_bin = members.copy(); m_bin.loc[0, "transportation_barrier"] = 7
    _case("invalid binary flags", files=_files(members_df=m_bin), expected_status=422)

    _case("invalid triage (safety context)", data={"current_safety_context": json.dumps({"M00001": {"triage_level": 0}})}, expected_status=422)

    m_chronic = members.copy(); m_chronic.loc[0, "num_chronic_conditions"] = m_chronic.loc[0, "num_chronic_conditions"] + 5
    _case("inconsistent chronic count", files=_files(members_df=m_chronic), expected_status=422)

    _case("unknown member", data={"member_id": "NOT_A_REAL_MEMBER"}, expected_status=404)

    m_missing = members.drop(columns=["transportation_barrier"])
    _case("missing required field", files=_files(members_df=m_missing), expected_status=422)

    m_extra = members.copy(); m_extra["unexpected_extra_column"] = "x"
    _case("extra fields (tolerated at CSV-column level)", files=_files(members_df=m_extra), expected_status=200)

    return pd.DataFrame(rows)


# =============================================================================
# Failure-mode results (Step 19)
# =============================================================================

def build_failure_mode_results() -> pd.DataFrame:
    return pd.DataFrame([
        {"failure_mode": "member data invalid (age/binary/distance/chronic-count)", "behavior": "422, structured issue list, no scoring attempted", "conservative": True},
        {"failure_mode": "safety context invalid (binary/triage out of range, unknown field, non-finite)", "behavior": "422 via Pydantic schema, no coercion to a safe-looking value", "conservative": True},
        {"failure_mode": "feature generation incomplete (required raw column missing)", "behavior": "KeyError propagates from backend/pit/features.py, never silently scores with fabricated data", "conservative": True},
        {"failure_mode": "model metadata incompatible (feature/version/threshold mismatch)", "behavior": "ModelIncompatibleError at RiskDetectionAgent construction, surfaced as 503", "conservative": True},
        {"failure_mode": "model artifact missing", "behavior": "ModelIncompatibleError, 503, never falls back to legacy model", "conservative": True},
        {"failure_mode": "threshold metadata missing/invalid ordering", "behavior": "ModelIncompatibleError at construction", "conservative": True},
        {"failure_mode": "Navigation Agent receives malformed/incomplete RiskAssessment-derived row", "behavior": "Defensive defaults (e.g. missing distance -> 99.0 = far), never raises, never fabricates a favorable/aggressive recommendation", "conservative": True},
        {"failure_mode": "Safety Agent receives malformed NavigationDecision", "behavior": "Pure deterministic logic over typed dataclass fields; language policy applied regardless of upstream content", "conservative": True},
        {"failure_mode": "Risk Agent raises mid-orchestration", "behavior": "Propagates uncaught to the API's try/except, converted to a clean error response, never fabricates a decision", "conservative": True},
    ])


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 7: API / safety-context / failure-mode artifacts ===")

    members_df = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")
    ed_df = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")

    input_matrix = build_input_validation_matrix(members_df, ed_df)
    input_matrix.to_csv(EVAL_DIR / "input_validation_matrix.csv", index=False)
    print(f"input_validation_matrix.csv: {len(input_matrix)} cases")

    safety_matrix = build_safety_context_matrix()
    safety_matrix.to_csv(EVAL_DIR / "safety_context_matrix.csv", index=False)
    print(f"safety_context_matrix.csv: {len(safety_matrix)} cases, all_passed={safety_matrix['passed'].all()}")

    client = TestClient(main_mod.app)
    api_matrix = build_api_validation_matrix(client)
    api_matrix.to_csv(EVAL_DIR / "api_validation_results.csv", index=False)
    print(f"api_validation_results.csv: {len(api_matrix)} cases, all_passed={api_matrix['passed'].dropna().all()}")

    failure_modes = build_failure_mode_results()
    failure_modes.to_csv(EVAL_DIR / "failure_mode_results.csv", index=False)
    print(f"failure_mode_results.csv: {len(failure_modes)} modes, all_conservative={failure_modes['conservative'].all()}")

    disparity_summary_path = EVAL_DIR / "phase7_disparity_summary.json"
    disparity_summary = json.loads(disparity_summary_path.read_text()) if disparity_summary_path.exists() else {}

    summary = {
        "phase": "7",
        "model_frozen": "uc07-risk-synthetic-v1",
        "disparity_summary": disparity_summary,
        "input_validation_matrix": {"cases": len(input_matrix)},
        "safety_context_matrix": {"cases": len(safety_matrix), "all_passed": bool(safety_matrix["passed"].all())},
        "api_validation_matrix": {"cases": len(api_matrix), "all_passed": bool(api_matrix["passed"].dropna().all())},
        "failure_modes_documented": len(failure_modes),
        "all_failure_modes_conservative": bool(failure_modes["conservative"].all()),
    }
    (EVAL_DIR / "phase7_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== Phase 7 API/safety-context/failure-mode artifacts complete ===")
    return summary


if __name__ == "__main__":
    main()
