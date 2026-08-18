"""
main.py
-------
FastAPI backend for ED Risk Prediction.

Endpoint:
    POST /predict
        - members_file    : multipart CSV upload (raw_members.csv format)
        - ed_visits_file  : multipart CSV upload (raw_ed_visits.csv format)
        - care_file       : multipart CSV upload (raw_care_history.csv format)

    Returns:
        An Excel file (application/vnd.openxmlformats-officedocument.spreadsheetml.sheet)
        with columns: member_id, age, gender, risk_probability, risk_score, risk_category
"""

import asyncio
import functools
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load backend/.env (if present) into the process environment BEFORE
# anything below reads os.environ -- this is what lets GENAI_ENABLED/
# OLLAMA_BASE_URL/OLLAMA_MODEL/GENAI_TIMEOUT_SECONDS/CORS_ORIGINS be set
# once in a local file instead of re-exporting them in every shell
# session. Uses an explicit path (not the CWD-relative default
# load_dotenv() would use) so `uvicorn main:app` works the same whether
# it's launched from backend/ or the repo root. override=False (the
# default) means a variable already set in the real environment always
# wins over the .env file -- this is deliberate: it keeps `monkeypatch.
# setenv(...)` in tests, and any real deployment env var, authoritative
# over a local .env. backend/.env is untracked (see .gitignore) and never
# committed -- backend/.env.example documents the expected keys with
# safe, non-secret example values.
load_dotenv(Path(__file__).resolve().parent / ".env")

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from feature_engineering import extract_features
from predict import MODEL_PATH, explain_member, predict

# ---------------------------------------------------------------------------
# Phase 5 UC07 multi-agent decision system (backend/agents/) -- the
# authoritative path for UC07 risk/navigation/safety decisions. This is
# separate from, and does not replace, the legacy /predict* endpoints
# above, which remain wired to the pre-Phase-2 ed_risk_model.pkl exactly
# as before. See docs/05_MULTI_AGENT_SYSTEM.md section 17 for why both
# exist and which one is authoritative for UC07 going forward.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
for _subdir in ("pit", "agents"):
    _p = str(_BACKEND_DIR / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contracts import CurrentSafetyContext  # noqa: E402
from input_validation import (  # noqa: E402
    EdVisitDataValidationError,
    MemberDataValidationError,
    validate_and_normalize_members_df,
    validate_ed_visits_df,
)
from orchestrator import UC07Orchestrator, decision_to_dict  # noqa: E402
import genai_explanation  # noqa: E402
from risk_detection import DEFAULT_ARTIFACT_PATH as UC07_ARTIFACT_PATH  # noqa: E402
from risk_detection import DEFAULT_METADATA_PATH as UC07_METADATA_PATH  # noqa: E402
from risk_detection import ModelIncompatibleError  # noqa: E402
from safety_context_csv import SafetyContextCsvValidationError, parse_safety_context_csv  # noqa: E402
from safety_context_schema import SafetyContextPayload  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from services import report_service  # noqa: E402
from services.audit import record_communication_event  # noqa: E402
from services.email_service import EmailService, load_email_config, mask_email  # noqa: E402

_uc07_orchestrator: UC07Orchestrator | None = None
_uc07_orchestrator_error: str | None = None


def _get_uc07_orchestrator() -> UC07Orchestrator:
    """Lazily construct and cache the orchestrator. Mirrors the legacy
    endpoints' lazy MODEL_PATH check -- the app can still start even if
    the UC07 model artifact is temporarily missing/incompatible; the
    failure surfaces as a clean 503 on first use, not a startup crash."""
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


def _parse_current_safety_context(raw: str | None) -> dict[str, CurrentSafetyContext]:
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


def _parse_index_date(raw: str | None) -> date:
    if not raw:
        return datetime.now().date()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"index_date must be an ISO date (YYYY-MM-DD): {exc}") from exc

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# Privacy-safe GenAI provider observability (Phase 8C/8D + Groq
# integration): genai_explanation.py itself performs no logging at all
# (see its module docstring and tests/test_genai_privacy.py) -- it only
# offers an `on_event` hook that fires with short, fixed event-name
# strings ("groq_attempted", "groq_succeeded", "ollama_attempted",
# "deterministic_fallback_used", etc). This logger is where those events
# actually get written; it NEVER receives payload contents, raw model
# output, or GROQ_API_KEY -- only that one static string per event, plus
# the member-agnostic explanation_source/model_used already returned to
# the caller. See POST /uc07/explain below for where this is used.
#
# logging.basicConfig is a no-op if the root logger already has a handler
# (e.g. uvicorn's own), so this only ensures INFO-level messages are
# actually emitted somewhere (stderr) when nothing else has configured
# logging yet -- it never duplicates or overrides an existing setup.
logging.basicConfig(level=logging.INFO)
genai_logger = logging.getLogger("uc07.genai")
# Phase 9 -- safe timing/result observability for POST /uc07/email, at
# the level this handler itself measures (report generation). SMTP
# stage-level timing/errors are logged separately by
# backend/services/email_service.py's own "uc07.communication.email"
# logger, closer to where they actually happen.
comm_logger = logging.getLogger("uc07.communication")

app = FastAPI(
    title="ED Risk Prediction API",
    description=(
        "Upload three CSV files (members, ED visits, care history) "
        "and receive an Excel report with avoidable ED risk scores."
    ),
    version="1.0.0",
)

# Phase 8D Part 11 -- CORS is now environment-configured rather than
# hard-coded fully-open ("*"). This is preparation for Phase 9, not a full
# production hardening pass (no auth/credential handling changes here):
# CORS_ORIGINS accepts a comma-separated list of allowed origins; unset
# defaults to the two local Vite dev server addresses so `npm run dev`
# keeps working out of the box with no configuration required. No Azure
# URL is hard-coded here -- Phase 9 sets CORS_ORIGINS at deploy time.
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Phase 8D Part 7 -- legacy /predict concurrency fix.
#
# extract_features() is offloaded via starlette's run_in_threadpool, which
# is sufficient for it (verified: a real thread-pool call to
# extract_features() does NOT starve the event loop -- it is vectorized
# pandas/numpy work that releases the GIL normally between calls).
#
# predict(include_shap=True) is NOT offloaded via run_in_threadpool alone.
# This was verified empirically, not assumed: a controlled test (asyncio
# task ticking every 0.5s, concurrent with `await run_in_threadpool(predict,
# ...)` on real data) showed the ticker starved for the ENTIRE ~22s
# duration of a 300-row predict() call, even though it was correctly
# running on a separate OS thread. Root cause: predict()'s per-row legacy
# SHAP TreeExplainer loop (predict.py) holds the GIL near-continuously:
# CPython threads share ONE global interpreter lock, so a CPU-bound thread
# that doesn't yield it via I/O or a GIL-releasing C call can still starve
# every other thread in the process -- including the one running the
# asyncio event loop -- regardless of which OS thread it happens to run on.
# A thread pool cannot fix that; only a separate OS PROCESS (its own,
# independent GIL) can. Hence: a small ProcessPoolExecutor, used only for
# this specific call.
_predict_process_pool: ProcessPoolExecutor | None = None


def _get_predict_process_pool() -> ProcessPoolExecutor:
    global _predict_process_pool
    if _predict_process_pool is None:
        # max_workers=1: this is a local-demo endpoint the UC07 frontend
        # never calls (see test_legacy_isolation.py); a second concurrent
        # /predict call queuing behind the first is an acceptable
        # trade-off for not spawning multiple heavy Python processes on a
        # small deployment. The goal here is protecting every OTHER
        # endpoint's responsiveness while /predict runs, not making
        # multiple simultaneous /predict calls fast.
        _predict_process_pool = ProcessPoolExecutor(max_workers=1)
    return _predict_process_pool


@app.on_event("shutdown")
def _shutdown_predict_process_pool() -> None:
    global _predict_process_pool
    if _predict_process_pool is not None:
        _predict_process_pool.shutdown(wait=False, cancel_futures=True)
        _predict_process_pool = None

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _read_csv(upload: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame."""
    content = upload.file.read()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse '{upload.filename}' as CSV: {exc}",
        ) from exc


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"{name} is missing columns: {missing}",
        )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
# The React app (frontend/) is built with `npm run build`, which emits
# frontend/dist. In local dev, run the Vite dev server instead
# (`npm run dev` inside frontend/) and skip this static mount.
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="dashboard")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "ED Risk Prediction API is running."}


@app.get("/dashboard", include_in_schema=False)
def dashboard_index():
    if not FRONTEND_DIST.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Dashboard build not found. Run 'npm install && npm run build' "
                "inside the frontend/ directory, or use 'npm run dev' for local "
                "development instead."
            ),
        )
    return FileResponse(FRONTEND_DIST / "index.html")


@app.get("/health", tags=["Health"])
def health():
    """Liveness + readiness. `model_loaded` (legacy field, unchanged) only
    checks the legacy artifact's file presence, as before. `uc07_model_loaded`
    actually attempts to load and cross-validate the Phase 5 UC07 model
    artifact against its metadata -- a real readiness check, not just a
    file-existence check."""
    model_ready = MODEL_PATH.exists()
    try:
        orchestrator = _get_uc07_orchestrator()
        uc07_ready = True
        uc07_error = None
        uc07_model_version = orchestrator.risk_agent.model_version
    except HTTPException as exc:
        uc07_ready = False
        uc07_error = exc.detail
        uc07_model_version = None
    email_config = load_email_config()
    return {
        "status": "ok",
        "model_loaded": model_ready,
        "uc07_model_loaded": uc07_ready,
        "uc07_model_version": uc07_model_version,
        "uc07_model_error": uc07_error,
        # Phase 9 -- safe configuration status only; never credentials.
        "email_configured": email_config.configured,
        "email_provider": email_config.provider,
    }


@app.get("/model-info", tags=["Health"])
def model_info():
    """Non-sensitive model identity/configuration for auditability and
    synthetic-model demo disclosure. Never exposes member-level data."""
    orchestrator = _get_uc07_orchestrator()
    agent = orchestrator.risk_agent
    return {
        "model_version": agent.model_version,
        "dataset_id": agent.dataset_id,
        "synthetic_model": agent.synthetic_model,
        "target": agent.metadata.get("target"),
        "target_definition": agent.metadata.get("target_definition"),
        "prediction_horizon_days": agent.metadata.get("prediction_horizon_days"),
        "observation_window_days": agent.metadata.get("observation_window_days"),
        "moderate_threshold": agent.moderate_threshold,
        "high_threshold": agent.high_threshold,
        "feature_count": len(agent.feature_columns),
        "algorithm": agent.metadata.get("algorithm"),
        "intended_use": agent.metadata.get("intended_use"),
        "disclaimer": agent.metadata.get("disclaimer"),
    }


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------


async def _extract_all(
    members_file: UploadFile,
    ed_visits_file: UploadFile,
    care_file: UploadFile,
):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please run train_model.py first to generate ed_risk_model.pkl."
            ),
        )

    members_df = _read_csv(members_file)
    ed_visits_df = _read_csv(ed_visits_file)
    care_df = _read_csv(care_file)

    _validate_columns(
        members_df,
        required=["member_id", "age", "gender", "transportation_barrier",
                  "pcp_distance_miles", "urgent_care_distance_miles"],
        name="members_file",
    )
    _validate_columns(
        ed_visits_df,
        required=["member_id", "visit_date", "diagnosis",
                  "admitted", "icu", "major_procedure", "cost", "red_flag"],
        name="ed_visits_file",
    )
    _validate_columns(
        care_df,
        required=["member_id", "visit_date", "care_type"],
        name="care_file",
    )

    try:
        # Phase 8D: offloaded to a worker thread -- extract_features() is
        # synchronous, CPU-bound pandas work over the whole uploaded
        # population. Calling it directly inside this `async def` handler
        # would block the single-process event loop for its full
        # duration, freezing every OTHER concurrent request (including
        # unrelated ones like GET /health or POST /uc07/decide) until it
        # returns. run_in_threadpool runs the exact same function,
        # unmodified, on a worker thread instead -- same inputs, same
        # outputs, same legacy behavior, just off the event loop.
        X, member_ids = await run_in_threadpool(extract_features, members_df, ed_visits_df, care_df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature engineering failed: {exc}",
        ) from exc

    return X, member_ids, members_df, ed_visits_df, care_df


async def _run_prediction(
    members_file: UploadFile,
    ed_visits_file: UploadFile,
    care_file: UploadFile,
    include_shap: bool = True,
):
    X, member_ids, members_df, ed_visits_df, care_df = await _extract_all(
        members_file, ed_visits_file, care_file
    )

    try:
        # Phase 8D: routed through a PROCESS pool, not a thread pool --
        # see the module-level comment above _get_predict_process_pool()
        # for why. Same function, same arguments, same legacy output;
        # only the execution context changes. functools.partial is used
        # because ProcessPoolExecutor.submit()/loop.run_in_executor()
        # takes positional args only, and predict()'s `include_shap` is a
        # keyword-only-by-convention parameter here.
        loop = asyncio.get_running_loop()
        pool = _get_predict_process_pool()
        result_df = await loop.run_in_executor(
            pool,
            functools.partial(
                predict, X, member_ids, members_df, ed_visits_df, care_df,
                model_path=MODEL_PATH, include_shap=include_shap,
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc

    return result_df


@app.post(
    "/predict",
    summary="Run ED risk prediction",
    description=(
        "Upload three CSV files. Returns an Excel file with risk scores "
        "and risk categories for each member."
    ),
    tags=["Prediction"],
    response_class=StreamingResponse,
)
async def predict_endpoint(
    members_file: UploadFile = File(..., description="raw_members.csv (or similar unseen data)"),
    ed_visits_file: UploadFile = File(..., description="raw_ed_visits.csv (or similar unseen data)"),
    care_file: UploadFile = File(..., description="raw_care_history.csv (or similar unseen data)"),
):
    result_df = await _run_prediction(members_file, ed_visits_file, care_file)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Risk_Predictions")

        ws = writer.sheets["Risk_Predictions"]
        for col in ws.columns:
            max_len = max(len(str(cell.value)) for cell in col if cell.value) + 2
            ws.column_dimensions[col[0].column_letter].width = max_len

    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=ed_risk_predictions.xlsx"
        },
    )


@app.post(
    "/predict-json",
    summary="Run ED risk prediction and return JSON for dashboard rendering",
    description=(
        "Scores every member but skips per-patient SHAP explanations, which "
        "are the expensive part (~70ms/patient against the shipped model — "
        "minutes for a full census). Fetch a single patient's explanation "
        "on demand via /explain-member once the table has rendered."
    ),
    tags=["Prediction"],
)
async def predict_json_endpoint(
    members_file: UploadFile = File(..., description="raw_members.csv (or similar unseen data)"),
    ed_visits_file: UploadFile = File(..., description="raw_ed_visits.csv (or similar unseen data)"),
    care_file: UploadFile = File(..., description="raw_care_history.csv (or similar unseen data)"),
):
    result_df = await _run_prediction(members_file, ed_visits_file, care_file, include_shap=False)

    # Members with no ED/care history leave NaN in merged columns (e.g.
    # days_since_last_care); strict JSON has no NaN/Infinity, so normalize
    # to null before serializing.
    # .astype(object) first: pandas no longer upcasts float64 columns when
    # assigning None via .where(), so without it NaN silently survives.
    json_safe_df = result_df.replace([np.inf, -np.inf], np.nan).astype(object)
    json_safe_df = json_safe_df.where(pd.notnull(json_safe_df), None)

    return JSONResponse(
        {
            "columns": result_df.columns.tolist(),
            "rows": json_safe_df.to_dict(orient="records"),
            "count": len(result_df),
        }
    )


@app.post(
    "/predict-demo",
    summary="Run ED risk prediction on local demo CSVs",
    tags=["Prediction"],
)
async def predict_demo_endpoint():
    members_path = BASE_DIR / "raw_members.csv"
    ed_visits_path = BASE_DIR / "raw_ed_visits.csv"
    care_path = BASE_DIR / "raw_care_history.csv"

    if not (members_path.exists() and ed_visits_path.exists() and care_path.exists()):
        raise HTTPException(
            status_code=404,
            detail="Demo data CSV files not found in the repository."
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please run train_model.py first to generate ed_risk_model.pkl."
            ),
        )

    try:
        members_df = pd.read_csv(members_path)
        ed_visits_df = pd.read_csv(ed_visits_path)
        care_df = pd.read_csv(care_path)
        X, member_ids = extract_features(members_df, ed_visits_df, care_df)
        result_df = predict(X, member_ids, members_df, ed_visits_df, care_df, include_shap=False)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Demo prediction failed: {exc}",
        ) from exc

    json_safe_df = result_df.replace([np.inf, -np.inf], np.nan).astype(object)
    json_safe_df = json_safe_df.where(pd.notnull(json_safe_df), None)

    return JSONResponse(
        {
            "columns": result_df.columns.tolist(),
            "rows": json_safe_df.to_dict(orient="records"),
            "count": len(result_df),
        }
    )


@app.post(
    "/explain-member",
    summary="Compute the SHAP explanation for one member on demand",
    tags=["Prediction"],
)
async def explain_member_endpoint(
    member_id: str = Form(...),
    members_file: UploadFile = File(None, description="raw_members.csv (or similar unseen data)"),
    ed_visits_file: UploadFile = File(None, description="raw_ed_visits.csv (or similar unseen data)"),
    care_file: UploadFile = File(None, description="raw_care_history.csv (or similar unseen data)"),
):
    if members_file is None or ed_visits_file is None or care_file is None:
        # Fallback to local files
        members_path = BASE_DIR / "raw_members.csv"
        ed_visits_path = BASE_DIR / "raw_ed_visits.csv"
        care_path = BASE_DIR / "raw_care_history.csv"
        if not (members_path.exists() and ed_visits_path.exists() and care_path.exists()):
            raise HTTPException(
                status_code=400,
                detail="Uploaded files are required because local demo files are missing."
            )
        try:
            members_df = pd.read_csv(members_path)
            ed_visits_df = pd.read_csv(ed_visits_path)
            care_df = pd.read_csv(care_path)
            X, member_ids = extract_features(members_df, ed_visits_df, care_df)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Feature engineering failed on demo data: {exc}",
            ) from exc
    else:
        X, member_ids, *_ = await _extract_all(members_file, ed_visits_file, care_file)

    try:
        # Phase 8D: same offload as /predict above, applied for
        # consistency -- this endpoint's SHAP call is per-member (much
        # cheaper, ~70ms) rather than per-population, so it was not the
        # reported freeze source, but it is the same class of synchronous
        # CPU-bound call inside an async handler.
        explanation = await run_in_threadpool(explain_member, member_id, X, member_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {exc}",
        ) from exc

    safe_explanation = {
        k: (None if isinstance(v, float) and (v != v) else v) for k, v in explanation.items()
    }
    return JSONResponse({"member_id": member_id, **safe_explanation})


# ---------------------------------------------------------------------------
# Phase 5 UC07 multi-agent decision endpoint
# ---------------------------------------------------------------------------


def _read_uc07_csv(upload: UploadFile) -> pd.DataFrame:
    return _read_csv(upload)


@app.post(
    "/uc07/decide",
    summary="Run the UC07 Risk Detection -> Care Navigation -> Safety & Policy agent pipeline",
    description=(
        "Uploads members/ED-visits/care-history CSVs (synthetic_uc07_v1 schema), reconstructs "
        "leakage-safe point-in-time features as of `index_date` (default: today), and returns a "
        "FinalUC07Decision for every member (or a single member, if `member_id` is given). "
        "The Safety & Policy Agent always runs last and cannot be bypassed. Uses "
        "uc07-risk-synthetic-v1 -- a SYNTHETIC DATA MODEL, DEMONSTRATION ONLY."
    ),
    tags=["UC07"],
)
async def uc07_decide_endpoint(
    members_file: UploadFile = File(..., description="raw_members.csv-shaped upload"),
    ed_visits_file: UploadFile = File(..., description="raw_ed_visits.csv-shaped upload (must include triage_level)"),
    care_file: UploadFile = File(..., description="raw_care_history.csv-shaped upload"),
    member_id: str | None = Form(None, description="If set, return a decision for this one member only"),
    index_date: str | None = Form(None, description="ISO date (YYYY-MM-DD); defaults to today"),
    current_safety_context: str | None = Form(
        None,
        description=(
            'JSON object keyed by member_id, e.g. {"M00001": {"red_flag":1,"triage_level":2}}. '
            "A member with no entry is treated as having NO current safety context (CAUTION)."
        ),
    ),
    safety_context_file: UploadFile | None = File(
        None,
        description=(
            "Optional current_safety_context.csv (columns: member_id, red_flag, icu, admitted, "
            "major_procedure, triage_level). Used only by the Safety & Policy Agent -- never a "
            "model feature. A blank cell means unknown, never 0/false. If a member_id also has "
            "an entry in current_safety_context (JSON), the JSON entry wins for that member."
        ),
    ),
):
    orchestrator = _get_uc07_orchestrator()

    members_df = _read_uc07_csv(members_file)
    ed_df = _read_uc07_csv(ed_visits_file)
    care_df = _read_uc07_csv(care_file)

    _validate_columns(members_df, UC07_MEMBERS_REQUIRED, "members_file")
    _validate_columns(ed_df, UC07_ED_REQUIRED, "ed_visits_file")
    _validate_columns(care_df, UC07_CARE_REQUIRED, "care_file")

    # Phase 7 value-level validation (column PRESENCE was just checked
    # above; this checks value PLAUSIBILITY -- age/binary/distance
    # ranges, chronic-count consistency, and every triage_level, not just
    # rows inside some snapshot's observation window). Never silently
    # scores an impossible input.
    try:
        members_df = validate_and_normalize_members_df(members_df)
    except MemberDataValidationError as exc:
        raise HTTPException(status_code=422, detail=f"members_file failed validation: {exc.issues}") from exc
    try:
        validate_ed_visits_df(ed_df)
    except EdVisitDataValidationError as exc:
        raise HTTPException(status_code=422, detail=f"ed_visits_file failed validation: {exc.issues}") from exc

    parsed_index_date = _parse_index_date(index_date)
    contexts = _parse_current_safety_context(current_safety_context)

    # Phase 8B: optional batch current-safety-context CSV. Joined ONLY by
    # member_id (never row order); any member_id in the CSV that isn't in
    # members_file is a validation error, never silently ignored. A
    # member_id also present in the JSON `current_safety_context` field
    # (e.g. a single-member ad-hoc override) takes precedence over the
    # CSV for that member.
    if safety_context_file is not None:
        safety_context_df = _read_uc07_csv(safety_context_file)
        try:
            csv_contexts = parse_safety_context_csv(safety_context_df, set(members_df["member_id"]))
        except SafetyContextCsvValidationError as exc:
            raise HTTPException(status_code=422, detail=f"safety_context_file failed validation: {exc.issues}") from exc
        contexts = {**csv_contexts, **contexts}

    try:
        if member_id is not None:
            if member_id not in set(members_df["member_id"]):
                raise HTTPException(status_code=404, detail=f"member_id '{member_id}' not found in members_file")
            decisions = [
                orchestrator.decide_for_member(
                    member_id, members_df, ed_df, care_df, parsed_index_date, contexts.get(member_id),
                )
            ]
        else:
            decisions = orchestrator.decide_for_all_members(members_df, ed_df, care_df, parsed_index_date, contexts)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelIncompatibleError as exc:
        raise HTTPException(status_code=503, detail=f"UC07 risk model unavailable: {exc}") from exc
    except HTTPException:
        raise
    except ValueError as exc:
        # e.g. an unrecognized triage_level/red_flag/etc. value in the
        # uploaded data -- a client input problem, not a server fault.
        raise HTTPException(status_code=422, detail=f"Invalid data in uploaded files: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"UC07 decision pipeline failed: {exc}") from exc

    agent = orchestrator.risk_agent
    return JSONResponse({
        "model_version": agent.model_version,
        "dataset_id": agent.dataset_id,
        "synthetic_model": agent.synthetic_model,
        "index_date": parsed_index_date.isoformat(),
        "count": len(decisions),
        "decisions": [decision_to_dict(d) for d in decisions],
    })


# ---------------------------------------------------------------------------
# Phase 8C -- POST /uc07/explain: AGENT 4, the GenAI Explanation Agent.
#
# Strict input boundary (Phase 8C Part 5): this endpoint accepts ONLY the
# minimal, already-computed decision summary below -- never a CSV upload,
# never a member_id lookup against member_data/ed_visit data, and it never
# calls UC07Orchestrator, RiskDetectionAgent, care_navigation, or
# safety_policy.decide(). It is structurally incapable of producing (or
# being influenced by) a NEW risk/navigation/safety decision: the only
# thing it does with its input is hand an allow-listed dict to
# genai_explanation.generate_explanation(), which itself never imports the
# decision agents either (see genai_explanation.py's module docstring).
# This one-way architecture -- decision agents produce a FinalUC07Decision,
# the frontend echoes its already-rendered fields back here, and nothing
# flows the other way -- is covered by
# tests/test_genai_explanation_authority.py.
#
# GenAI failure (Ollama not running, timeout, malformed output, policy
# violation) can only affect the fields returned by *this* endpoint; it
# can never affect /uc07/decide, which does not call this code at all.
# ---------------------------------------------------------------------------

class ExplainFactorPayload(BaseModel):
    display_name: str
    direction: Literal["INCREASES_RISK", "DECREASES_RISK"]


class ExplainRiskPayload(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    tier: Literal["LOW", "MODERATE", "HIGH"]
    model_version: str
    factors: list[ExplainFactorPayload] = Field(default_factory=list)


class ExplainNavigationPayload(BaseModel):
    destination: Literal[
        "PRIMARY_CARE", "URGENT_CARE", "TELEHEALTH", "CARE_MANAGEMENT", "NO_PROACTIVE_NAVIGATION"
    ] | None = None
    reason_codes: list[str] = Field(default_factory=list)


class ExplainSafetyPayload(BaseModel):
    state: Literal["CLEAR", "CAUTION", "OVERRIDE"]
    context_completeness: Literal["COMPLETE", "PARTIAL", "ABSENT"]
    context_source: Literal["CALLER_SUPPLIED", "SYSTEM_DERIVED", "NOT_AVAILABLE"]


class ExplainRequest(BaseModel):
    risk: ExplainRiskPayload
    navigation: ExplainNavigationPayload
    safety: ExplainSafetyPayload
    synthetic_model: bool


@app.post("/uc07/explain")
def uc07_explain(request: ExplainRequest):
    """Lazy, on-demand, single-member only (Phase 8C Part 14) -- the
    frontend must call this at most once per member the user actually
    opens, never for a whole uploaded population. Always returns 200 with
    a MemberExplanation (GenAI or DETERMINISTIC_FALLBACK); GenAI failure
    is never surfaced as an HTTP error here."""
    payload = request.model_dump()
    events: list[str] = []
    result = genai_explanation.generate_explanation(payload, on_event=events.append)
    if events:
        genai_logger.info("uc07.explain provider_events=%s", ",".join(events))
    genai_logger.info(
        "uc07.explain result explanation_source=%s model_used=%s",
        result.explanation_source.value,
        result.model_used,
    )
    return JSONResponse({
        "summary": result.summary,
        "risk_explanation": result.risk_explanation,
        "navigation_explanation": result.navigation_explanation,
        "safety_explanation": result.safety_explanation,
        "disclaimer": result.disclaimer,
        "explanation_source": result.explanation_source.value,
        "model_used": result.model_used,
        "generation_time_ms": result.generation_time_ms,
    })


# ---------------------------------------------------------------------------
# Phase 9 -- Member Communication & Reporting: PDF report + email.
#
# Same strict input boundary as POST /uc07/explain above: these
# endpoints accept ONLY an already-computed decision summary (the same
# risk/navigation/safety shape ExplainRequest uses, reused here via
# subclassing) plus member contact fields and an already-approved
# explanation. Neither endpoint uploads/parses a CSV, looks up a member,
# or calls UC07Orchestrator/RiskDetectionAgent/care_navigation/
# safety_policy.decide() -- backend/services/report_service.py and
# email_service.py are pure renderer/transport layers with zero
# decision authority (see their module docstrings). This is an
# operational communication/reporting layer that CONSUMES the final
# decision; it does not create one.
# ---------------------------------------------------------------------------


class ReportMemberPayload(BaseModel):
    """Contact/profile fields for the PDF report and email recipient.
    `email` is NEVER an ML feature and is never read by any agent --
    the member dataset (raw_members.csv) has no email column at all
    (confirmed by inspection); this is purely a communication-layer
    field the care manager supplies from a trusted contact source or
    types in directly (see frontend/src/uc07/memberContacts.ts)."""
    member_id: str = Field(min_length=1, max_length=64)
    name: str | None = Field(default=None, max_length=200)
    email: str | None = Field(default=None, max_length=254)
    age: int | None = Field(default=None, ge=0, le=130)
    gender: str | None = Field(default=None, max_length=50)


class ReportSafetyPayload(ExplainSafetyPayload):
    """Same fields as ExplainSafetyPayload plus `message` -- the exact,
    already-produced SafetyDecision.message text the frontend already
    displays in SafetyCard, reused verbatim here rather than
    re-deriving safety wording in a second place."""
    message: str = Field(min_length=1, max_length=2000)


class ReportExplanationPayload(BaseModel):
    summary: str
    risk_explanation: str
    navigation_explanation: str
    safety_explanation: str
    disclaimer: str
    explanation_source: Literal["GENAI", "DETERMINISTIC_FALLBACK"]
    model_used: str | None = None


class ReportRequest(BaseModel):
    member: ReportMemberPayload
    risk: ExplainRiskPayload
    navigation: ExplainNavigationPayload
    safety: ReportSafetyPayload
    synthetic_model: bool
    dataset_id: str | None = None
    # If omitted, the report is built with the deterministic fallback
    # explanation (see _resolve_report_explanation) -- never a fresh
    # GenAI call. If given, it MUST be an explanation this app already
    # obtained from POST /uc07/explain (GenAI-validated or
    # deterministic); this endpoint does not re-validate its text
    # against safety_policy, matching the trust model already
    # established for MemberExplanation once it leaves that endpoint.
    explanation: ReportExplanationPayload | None = None


def _resolve_report_explanation(request: ReportRequest) -> ReportExplanationPayload:
    """Section 4-G / Section 14: never calls a NEW LLM solely for PDF or
    email generation. If the caller already has an approved explanation
    (from a prior POST /uc07/explain call), it is reused as-is.
    Otherwise this builds the SAME deterministic, template-based
    explanation genai_explanation.py always falls back to, with
    GENAI_ENABLED forced off for this one call -- so no network/LLM call
    is ever made from this code path."""
    if request.explanation is not None:
        return request.explanation

    payload = {
        "risk": {
            "probability": request.risk.probability,
            "tier": request.risk.tier,
            "model_version": request.risk.model_version,
            "factors": [f.model_dump() for f in request.risk.factors],
        },
        "navigation": {
            "destination": request.navigation.destination,
            "reason_codes": request.navigation.reason_codes,
        },
        "safety": {
            "state": request.safety.state,
            "context_completeness": request.safety.context_completeness,
            "context_source": request.safety.context_source,
        },
        "synthetic_model": request.synthetic_model,
    }
    forced_deterministic_config = genai_explanation.GenAIConfig()
    forced_deterministic_config.enabled = False
    result = genai_explanation.generate_explanation(payload, config=forced_deterministic_config)
    return ReportExplanationPayload(
        summary=result.summary,
        risk_explanation=result.risk_explanation,
        navigation_explanation=result.navigation_explanation,
        safety_explanation=result.safety_explanation,
        disclaimer=result.disclaimer,
        explanation_source=result.explanation_source.value,
        model_used=result.model_used,
    )


def _build_report_context(request: ReportRequest) -> report_service.ReportContext:
    """The ONE place a ReportRequest is turned into a report_service.
    ReportContext -- used identically by both POST /uc07/report and
    POST /uc07/email, so the downloaded PDF and the emailed attachment
    are always produced by the exact same rendering call (Section 15)."""
    explanation = _resolve_report_explanation(request)
    return report_service.ReportContext(
        member_id=request.member.member_id,
        member_name=request.member.name,
        member_email=request.member.email,
        member_age=request.member.age,
        member_gender=request.member.gender,
        risk_probability=request.risk.probability,
        risk_tier=request.risk.tier,
        model_version=request.risk.model_version,
        synthetic_model=request.synthetic_model,
        dataset_id=request.dataset_id,
        navigation_destination=request.navigation.destination,
        navigation_reason_codes=list(request.navigation.reason_codes),
        factors=[
            report_service.ReportFactor(display_name=f.display_name, direction=f.direction)
            for f in request.risk.factors
        ],
        safety_state=request.safety.state,
        safety_message=request.safety.message,
        context_completeness=request.safety.context_completeness,
        context_source=request.safety.context_source,
        explanation_summary=explanation.summary,
        explanation_risk=explanation.risk_explanation,
        explanation_navigation=explanation.navigation_explanation,
        explanation_safety=explanation.safety_explanation,
        explanation_disclaimer=explanation.disclaimer,
        explanation_source=explanation.explanation_source,
        explanation_model_used=explanation.model_used,
    )


@app.post(
    "/uc07/report",
    summary="Generate the Member Care Navigation & Risk Summary PDF report",
    tags=["Communication"],
)
def uc07_report(request: ReportRequest):
    """Renders and returns the PDF report for one member as
    application/pdf. Pure rendering over already-computed decision data
    (see module comment above) -- no CSV upload, no model inference, no
    email sent. Same rendering call POST /uc07/email uses for its
    attachment (Section 15)."""
    ctx = _build_report_context(request)
    pdf_bytes = report_service.generate_report_pdf(ctx)
    filename = report_service.build_report_filename(request.member.member_id)

    record_communication_event(
        action="PDF_GENERATED",
        member_id=request.member.member_id,
        report_id=ctx.report_id,
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class EmailSendRequest(BaseModel):
    report: ReportRequest
    to_email: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10000)


@app.post(
    "/uc07/email",
    summary="Email the Member Care Navigation & Risk Summary PDF report to the member",
    tags=["Communication"],
)
def uc07_email(request: EmailSendRequest):
    """Builds the SAME PDF POST /uc07/report would return (Section 15,
    16) and sends it as an attachment via the configured EmailProvider.
    Always returns HTTP 200 with a structured {sent, message,
    error_code} body -- matching this codebase's existing convention
    (POST /uc07/explain never surfaces a GenAI failure as an HTTP
    error) -- so a disabled/misconfigured/unreachable email provider is
    reported cleanly to the caller instead of as a 5xx. NEVER returns an
    SMTP stack trace or credential; NEVER changes a risk/navigation/
    safety value (this endpoint has no access to the decision agents at
    all -- see module comment above).

    Section 10 (Phase 9 SMTP reliability pass) -- this response shape
    was deliberately KEPT as-is rather than switched to per-failure HTTP
    status codes (e.g. 504 for a timeout): it already IS the "explicit
    structured status" fallback the spec asks for (`sent`/`error_code`/
    `message`, not just a bare 200), the frontend (EmailComposerModal.tsx)
    already branches on `result.sent`/`result.message` rather than the
    HTTP status, and changing status codes now would be an unforced
    contract break for no behavioral gain.

    Section 8 -- this is a plain `def` (not `async def`) handler, same
    as POST /uc07/report above: FastAPI/Starlette automatically runs a
    synchronous route function in its own worker thread
    (`starlette.concurrency.run_in_threadpool`, applied by the framework
    itself for any non-async endpoint), so the blocking SMTP I/O inside
    `EmailService.send_report_email()` does NOT run on -- and cannot
    starve -- the main asyncio event loop. Verified live: a concurrent
    GET /health kept responding immediately while a POST /uc07/email
    call was blocked for several seconds against an unreachable SMTP
    host. No `ProcessPoolExecutor` is needed here (unlike the legacy
    `/predict` SHAP path, docs/08D_CRITICAL_FIXES.md §5) -- SMTP is
    ordinary blocking network I/O, which a thread handles correctly
    (it releases the GIL while waiting on the socket), not CPU-bound
    work that holds the GIL."""
    report_start = time.monotonic()
    ctx = _build_report_context(request.report)
    pdf_bytes = report_service.generate_report_pdf(ctx)
    report_generation_ms = (time.monotonic() - report_start) * 1000
    filename = report_service.build_report_filename(request.report.member.member_id)

    service = EmailService()
    result = service.send_report_email(
        to_email=request.to_email,
        subject=request.subject,
        body=request.body,
        attachment_bytes=pdf_bytes,
        attachment_filename=filename,
    )

    # Safe timing only (Section 12) -- never report/email content. SMTP
    # connection/send timings are logged separately, at the point they
    # are actually measured, by email_service.py's own "smtp_send" line.
    comm_logger.info(
        "uc07.email report_generation_ms=%.1f result=%s error_code=%s",
        report_generation_ms, "SENT" if result.sent else "FAILED", result.error_code,
    )

    try:
        masked_recipient = mask_email(request.to_email)
    except Exception:
        masked_recipient = "***"

    record_communication_event(
        action="EMAIL_SENT" if result.sent else "EMAIL_FAILED",
        member_id=request.report.member.member_id,
        report_id=ctx.report_id,
        masked_recipient=masked_recipient,
        provider=result.provider,
        result_status="OK" if result.sent else (result.error_code or "FAILED"),
    )

    return JSONResponse({
        "sent": result.sent,
        "provider": result.provider,
        "message": result.message,
        "error_code": result.error_code,
        "report_id": ctx.report_id,
    })
