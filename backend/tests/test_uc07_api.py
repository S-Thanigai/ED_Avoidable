"""
FastAPI-level tests for the Phase 5 UC07 endpoints (backend/main.py).
Uses starlette's TestClient (no live server needed). Also verifies the
legacy endpoints/app still boot correctly (no regression from the
Phase 5 additions) and that no response leaks a raw Python traceback.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402

REPO_ROOT = BACKEND_DIR.parent
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def _synthetic_files():
    return {
        "members_file": ("raw_members.csv", open(SYNTHETIC_DIR / "raw_members.csv", "rb"), "text/csv"),
        "ed_visits_file": ("raw_ed_visits.csv", open(SYNTHETIC_DIR / "raw_ed_visits.csv", "rb"), "text/csv"),
        "care_file": ("raw_care_history.csv", open(SYNTHETIC_DIR / "raw_care_history.csv", "rb"), "text/csv"),
    }


def _close_files(files):
    for _, fh, _ in files.values():
        fh.close()


# ---- app boots, legacy endpoints unaffected ----

def test_app_boots_and_root_works(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_legacy_predict_endpoints_still_registered(client):
    paths = {route.path for route in main.app.routes}
    assert "/predict" in paths
    assert "/predict-json" in paths
    assert "/explain-member" in paths


# ---- health / model-info ----

def test_health_reports_both_models(client):
    resp = client.get("/health")
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert "model_loaded" in body  # legacy field, unchanged
    assert body["uc07_model_loaded"] is True
    assert body["uc07_model_version"] == "uc07-risk-synthetic-v1"


def test_model_info_exposes_identity_without_member_data(client):
    resp = client.get("/model-info")
    body = resp.json()
    assert resp.status_code == 200
    assert body["model_version"] == "uc07-risk-synthetic-v1"
    assert body["dataset_id"] == "synthetic_uc07_v1"
    assert body["synthetic_model"] is True
    assert body["moderate_threshold"] == pytest.approx(0.105986)
    assert body["high_threshold"] == pytest.approx(0.213252)
    assert body["feature_count"] == 59
    assert "member_id" not in json.dumps(body)  # no member-level data present


# ---- /uc07/decide happy paths ----

def test_uc07_decide_single_member_no_context(client):
    files = _synthetic_files()
    try:
        resp = client.post("/uc07/decide", files=files, data={"member_id": "M00001", "index_date": "2026-07-03"})
    finally:
        _close_files(files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "uc07-risk-synthetic-v1"
    assert body["dataset_id"] == "synthetic_uc07_v1"
    assert body["synthetic_model"] is True
    assert body["count"] == 1
    decision = body["decisions"][0]
    assert decision["member_id"] == "M00001"
    assert decision["safety"]["state"] == "CAUTION"  # no current_safety_context supplied
    assert 0.0 <= decision["risk"]["probability"] <= 1.0
    assert decision["risk"]["tier"] in ("LOW", "MODERATE", "HIGH")


def test_uc07_decide_with_override_context(client):
    files = _synthetic_files()
    context = json.dumps({"M00001": {"red_flag": 1, "triage_level": 2}})
    try:
        resp = client.post(
            "/uc07/decide", files=files,
            data={"member_id": "M00001", "index_date": "2026-07-03", "current_safety_context": context},
        )
    finally:
        _close_files(files)
    assert resp.status_code == 200
    decision = resp.json()["decisions"][0]
    assert decision["safety"]["state"] == "OVERRIDE"
    assert decision["navigation"]["destination"] is None


def test_uc07_decide_with_clear_context(client):
    files = _synthetic_files()
    context = json.dumps({"M00001": {"red_flag": 0, "icu": 0, "admitted": 0, "major_procedure": 0, "triage_level": 4}})
    try:
        resp = client.post(
            "/uc07/decide", files=files,
            data={"member_id": "M00001", "index_date": "2026-07-03", "current_safety_context": context},
        )
    finally:
        _close_files(files)
    decision = resp.json()["decisions"][0]
    assert decision["safety"]["state"] == "CLEAR"


def test_uc07_decide_full_population_batch(client):
    files = _synthetic_files()
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close_files(files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 10000


# ---- error handling ----

def test_uc07_decide_unknown_member_returns_404(client):
    files = _synthetic_files()
    try:
        resp = client.post("/uc07/decide", files=files, data={"member_id": "NOT_REAL", "index_date": "2026-07-03"})
    finally:
        _close_files(files)
    assert resp.status_code == 404
    assert "Traceback" not in resp.text


def test_uc07_decide_malformed_context_json_returns_422(client):
    files = _synthetic_files()
    try:
        resp = client.post(
            "/uc07/decide", files=files,
            data={"member_id": "M00001", "current_safety_context": "{not valid json"},
        )
    finally:
        _close_files(files)
    assert resp.status_code == 422
    assert "Traceback" not in resp.text


def test_uc07_decide_invalid_binary_flag_returns_422(client):
    files = _synthetic_files()
    context = json.dumps({"M00001": {"red_flag": 5}})
    try:
        resp = client.post(
            "/uc07/decide", files=files,
            data={"member_id": "M00001", "current_safety_context": context},
        )
    finally:
        _close_files(files)
    assert resp.status_code == 422


def test_uc07_decide_invalid_triage_level_returns_422(client):
    files = _synthetic_files()
    context = json.dumps({"M00001": {"triage_level": 9}})
    try:
        resp = client.post(
            "/uc07/decide", files=files,
            data={"member_id": "M00001", "current_safety_context": context},
        )
    finally:
        _close_files(files)
    assert resp.status_code == 422


def test_uc07_decide_invalid_index_date_returns_422(client):
    files = _synthetic_files()
    try:
        resp = client.post("/uc07/decide", files=files, data={"member_id": "M00001", "index_date": "not-a-date"})
    finally:
        _close_files(files)
    assert resp.status_code == 422


def test_uc07_decide_missing_required_column_returns_422(client, tmp_path):
    import pandas as pd
    ed = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")
    ed_missing_triage = ed.drop(columns=["triage_level"])
    bad_ed_path = tmp_path / "bad_ed.csv"
    ed_missing_triage.to_csv(bad_ed_path, index=False)

    files = {
        "members_file": ("raw_members.csv", open(SYNTHETIC_DIR / "raw_members.csv", "rb"), "text/csv"),
        "ed_visits_file": ("raw_ed_visits.csv", open(bad_ed_path, "rb"), "text/csv"),
        "care_file": ("raw_care_history.csv", open(SYNTHETIC_DIR / "raw_care_history.csv", "rb"), "text/csv"),
    }
    try:
        resp = client.post("/uc07/decide", files=files, data={"member_id": "M00001"})
    finally:
        _close_files(files)
    assert resp.status_code == 422
    assert "triage_level" in resp.text


# ---- adversarial language check across the whole response ----

def test_full_population_response_contains_no_prohibited_language(client):
    import safety_policy
    files = _synthetic_files()
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close_files(files)
    body = resp.json()
    violations = []
    for decision in body["decisions"]:
        for text in (decision["navigation"]["explanation"], decision["safety"]["message"]):
            hits = safety_policy.check_text(text)
            if hits:
                violations.append((decision["member_id"], text, hits))
    assert violations == [], f"prohibited language found in {len(violations)} responses: {violations[:5]}"


def test_all_navigation_destinations_reachable_across_full_population(client):
    files = _synthetic_files()
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close_files(files)
    destinations = {d["navigation"]["destination"] for d in resp.json()["decisions"]}
    # None (OVERRIDE) shouldn't occur here since no current_safety_context was supplied for anyone
    assert None not in destinations
    expected = {"PRIMARY_CARE", "URGENT_CARE", "TELEHEALTH", "CARE_MANAGEMENT", "NO_PROACTIVE_NAVIGATION"}
    assert destinations.issubset(expected)
    assert len(destinations) >= 3  # real population should hit several branches, not just one
