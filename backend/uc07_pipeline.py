"""
uc07_pipeline.py
------------------
Shared UC07 request-parsing + orchestrator-invocation helpers, factored
out of backend/main.py so that BOTH `POST /uc07/decide`
(main.uc07_decide_endpoint) and `POST /populations/save-analysis`
(backend/routers/populations.py) call through the exact same code path
-- there is deliberately no second/duplicate prediction implementation
for the "save to database" flow (see docs/05_MULTI_AGENT_SYSTEM.md and
this feature's plan: "Do not create two independent prediction
implementations").

This is a pure extraction: every function below has the same body it
had inline in main.py before this module existed. No decision logic,
validation rule, or default changed.
"""
from __future__ import annotations

import io
import json
from datetime import date, datetime

import pandas as pd
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from contracts import CurrentSafetyContext
from input_validation import (
    EdVisitDataValidationError,
    MemberDataValidationError,
    validate_and_normalize_members_df,
    validate_ed_visits_df,
)
from orchestrator import UC07Orchestrator
from risk_detection import ModelIncompatibleError
from safety_context_csv import SafetyContextCsvValidationError, parse_safety_context_csv
from safety_context_schema import SafetyContextPayload

UC07_MEMBERS_REQUIRED = [
    # num_chronic_conditions is deliberately NOT required here: Phase 7
    # (docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md section 12) derives it
    # safely from diabetes+copd+hypertension+chf+asthma+ckd when a caller
    # omits it, rather than requiring a redundant client-supplied value;
    # validate_and_normalize_members_df() enforces consistency when it
    # IS supplied.
    "member_id", "age", "gender", "diabetes", "copd", "hypertension", "chf", "asthma", "ckd",
    "transportation_barrier", "telehealth_available",
    "pcp_distance_miles", "urgent_care_distance_miles",
]
UC07_ED_REQUIRED = [
    "member_id", "visit_date", "diagnosis", "triage_level", "admitted", "icu", "major_procedure", "cost", "red_flag",
]
UC07_CARE_REQUIRED = ["member_id", "visit_date", "care_type"]

_uc07_orchestrator: UC07Orchestrator | None = None
_uc07_orchestrator_error: str | None = None


def get_uc07_orchestrator() -> UC07Orchestrator:
    """Lazily construct and cache the orchestrator (single shared
    instance for the whole app -- backend/main.py and
    backend/routers/populations.py both call this same function, never
    construct their own)."""
    global _uc07_orchestrator, _uc07_orchestrator_error
    if _uc07_orchestrator is not None:
        return _uc07_orchestrator
    try:
        _uc07_orchestrator = UC07Orchestrator()
        _uc07_orchestrator_error = None
        return _uc07_orchestrator
    except ModelIncompatibleError as exc:
        _uc07_orchestrator_error = str(exc)
        raise HTTPException(status_code=503, detail=f"UC07 risk model unavailable: {exc}") from exc


def read_csv(upload: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame."""
    content = upload.file.read()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse '{upload.filename}' as CSV: {exc}",
        ) from exc


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"{name} is missing columns: {missing}",
        )


def parse_current_safety_context(raw: str | None) -> dict[str, CurrentSafetyContext]:
    """Parses the optional `current_safety_context` JSON form field:
    {"<member_id>": {"red_flag":0|1,"icu":0|1,"admitted":0|1,"major_procedure":0|1,"triage_level":1-5}, ...}
    A member absent from this mapping is treated as having NO current
    safety context supplied at all (CAUTION), never as CLEAR -- schema
    validation (backend/agents/safety_context_schema.py) never invents a
    default of 0 for a field the caller omitted for a given member; it
    only fills in defaults within an ENTRY the caller did provide, field
    by field. Business validation logic lives in that module, not here."""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"current_safety_context is not valid JSON: {exc}") from exc

    try:
        validated = SafetyContextPayload.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"current_safety_context is invalid: {exc.errors()}") from exc

    return {
        member_id: CurrentSafetyContext(
            red_flag=entry.red_flag, icu=entry.icu, admitted=entry.admitted,
            major_procedure=entry.major_procedure, triage_level=entry.triage_level,
        )
        for member_id, entry in validated.root.items()
    }


def parse_index_date(raw: str | None) -> date:
    if not raw:
        return datetime.now().date()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"index_date must be an ISO date (YYYY-MM-DD): {exc}") from exc


def validate_and_parse_uc07_inputs(
    members_df: pd.DataFrame,
    ed_df: pd.DataFrame,
    care_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Column-presence + value-level validation shared by
    /uc07/decide and /populations/save-analysis. Returns the
    (possibly-normalized) members_df and the unchanged ed_df -- care_df
    has no value-level validation step today, matching the existing
    endpoint exactly."""
    validate_columns(members_df, UC07_MEMBERS_REQUIRED, "members_file")
    validate_columns(ed_df, UC07_ED_REQUIRED, "ed_visits_file")
    validate_columns(care_df, UC07_CARE_REQUIRED, "care_file")

    try:
        members_df = validate_and_normalize_members_df(members_df)
    except MemberDataValidationError as exc:
        raise HTTPException(status_code=422, detail=f"members_file failed validation: {exc.issues}") from exc
    try:
        validate_ed_visits_df(ed_df)
    except EdVisitDataValidationError as exc:
        raise HTTPException(status_code=422, detail=f"ed_visits_file failed validation: {exc.issues}") from exc

    return members_df, ed_df


def resolve_safety_contexts(
    members_df: pd.DataFrame,
    current_safety_context_raw: str | None,
    safety_context_df: pd.DataFrame | None,
) -> dict[str, CurrentSafetyContext]:
    """Merges the optional batch CSV with the optional JSON field, JSON
    winning per-member on conflict -- identical merge rule to the
    existing /uc07/decide endpoint."""
    contexts = parse_current_safety_context(current_safety_context_raw)
    if safety_context_df is not None:
        try:
            csv_contexts = parse_safety_context_csv(safety_context_df, set(members_df["member_id"]))
        except SafetyContextCsvValidationError as exc:
            raise HTTPException(status_code=422, detail=f"safety_context_file failed validation: {exc.issues}") from exc
        contexts = {**csv_contexts, **contexts}
    return contexts


def decide_for_population(
    orchestrator: UC07Orchestrator,
    members_df: pd.DataFrame,
    ed_df: pd.DataFrame,
    care_df: pd.DataFrame,
    index_date: date,
    contexts: dict[str, CurrentSafetyContext],
    member_id: str | None = None,
):
    """The exact try/except call-and-error-mapping /uc07/decide already
    performs, extracted so it is not duplicated for the save-analysis
    endpoint."""
    try:
        if member_id is not None:
            if member_id not in set(members_df["member_id"]):
                raise HTTPException(status_code=404, detail=f"member_id '{member_id}' not found in members_file")
            return [
                orchestrator.decide_for_member(
                    member_id, members_df, ed_df, care_df, index_date, contexts.get(member_id),
                )
            ]
        return orchestrator.decide_for_all_members(members_df, ed_df, care_df, index_date, contexts)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelIncompatibleError as exc:
        raise HTTPException(status_code=503, detail=f"UC07 risk model unavailable: {exc}") from exc
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid data in uploaded files: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"UC07 decision pipeline failed: {exc}") from exc
