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

import io
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from feature_engineering import extract_features
from predict import MODEL_PATH, explain_member, predict

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ED Risk Prediction API",
    description=(
        "Upload three CSV files (members, ED visits, care history) "
        "and receive an Excel report with avoidable ED risk scores."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    model_ready = MODEL_PATH.exists()
    return {"status": "ok", "model_loaded": model_ready}


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
        X, member_ids = extract_features(members_df, ed_visits_df, care_df)
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
        result_df = predict(
            X, member_ids, members_df, ed_visits_df, care_df, include_shap=include_shap
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
        explanation = explain_member(member_id, X, member_ids)
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
