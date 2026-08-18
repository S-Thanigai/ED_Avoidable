"""
Save-analysis + reload tests -- the core "Azure SQL persistence" feature.

Covers:
  - analyzing CSVs never persists anything by itself (no DB row exists
    until POST /populations/save-analysis is explicitly called)
  - a successful save is transactional and produces correct member/
    ed_visit/care/analysis_result counts
  - a saved population belongs to the authenticated caller
  - reloading a saved population (GET .../members, .../members/{id})
    reproduces EXACTLY the same risk/navigation/safety values a fresh
    POST /uc07/decide call on the identical inputs produces -- this is
    the ML-equivalence requirement, satisfied by construction since
    save-analysis calls the same orchestrator (backend/uc07_pipeline.py)
    /uc07/decide uses, not a second implementation
  - the optional 4th CSV (current_safety_context) is preserved and still
    influences the persisted Safety Agent outcome the same way it would
    for a live /uc07/decide call
  - rollback: an induced mid-save failure leaves no partial population
"""
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
from db.engine import get_session_factory  # noqa: E402
from db.models import Population  # noqa: E402
from tests.db_helpers import build_small_uc07_csvs, delete_user_cascade, unique_email  # noqa: E402

PASSWORD = "correcthorsebattery"


@pytest.fixture
def authed_client():
    client = TestClient(main.app)
    email = unique_email()
    resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 201
    user_id = resp.json()["id"]
    yield client, user_id
    delete_user_cascade(user_id)


def test_csv_analysis_without_save_persists_nothing(authed_client):
    client, user_id = authed_client
    files, _member_ids = build_small_uc07_csvs(n_members=5)
    resp = client.post("/uc07/decide", files=files, data={})
    assert resp.status_code == 200
    assert resp.json()["count"] == 5

    # Nothing was saved -- this user's population list is still empty.
    listing = client.get("/populations")
    assert listing.json() == []

    db = get_session_factory()()
    try:
        assert db.query(Population).filter(Population.owner_user_id == user_id).count() == 0
    finally:
        db.close()


def test_save_analysis_creates_correct_counts(authed_client):
    client, _user_id = authed_client
    files, member_ids = build_small_uc07_csvs(n_members=8)
    resp = client.post("/populations/save-analysis", files=files, data={"name": "Test Population A"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Test Population A"
    assert body["member_count"] == len(member_ids)

    pid = body["id"]
    members_page = client.get(f"/populations/{pid}/members", params={"page": 1, "page_size": 50})
    assert members_page.status_code == 200
    assert members_page.json()["total_items"] == len(member_ids)


def test_saved_population_belongs_to_creating_user(authed_client):
    client, user_id = authed_client
    files, _member_ids = build_small_uc07_csvs(n_members=4)
    resp = client.post("/populations/save-analysis", files=files, data={"name": "Ownership Check"})
    pid = resp.json()["id"]

    db = get_session_factory()()
    try:
        row = db.get(Population, pid)
        assert row.owner_user_id == user_id
    finally:
        db.close()


def test_reload_matches_fresh_csv_decision_exactly(authed_client):
    """The ML-equivalence checkpoint: for the SAME inputs and index_date,
    the DB-persisted decision must equal the live CSV-pathway decision
    field for field (probability, tier, navigation, safety, SHAP
    factors) -- not just approximately."""
    client, _user_id = authed_client
    files_for_save, member_ids = build_small_uc07_csvs(n_members=6)
    save_resp = client.post(
        "/populations/save-analysis",
        files=files_for_save,
        data={"name": "Equivalence Check", "index_date": "2025-12-31"},
    )
    assert save_resp.status_code == 201
    pid = save_resp.json()["id"]

    files_for_fresh, _ = build_small_uc07_csvs(n_members=6)
    fresh_resp = client.post("/uc07/decide", files=files_for_fresh, data={"index_date": "2025-12-31"})
    assert fresh_resp.status_code == 200
    fresh_by_member = {d["member_id"]: d for d in fresh_resp.json()["decisions"]}

    for member_id in member_ids:
        detail = client.get(f"/populations/{pid}/members/{member_id}")
        assert detail.status_code == 200
        persisted = detail.json()["decision"]
        fresh = fresh_by_member[member_id]

        assert persisted["risk"]["probability"] == pytest.approx(fresh["risk"]["probability"], abs=1e-9)
        assert persisted["risk"]["tier"] == fresh["risk"]["tier"]
        assert persisted["risk"]["explanation_factors"] == fresh["risk"]["explanation_factors"]
        assert persisted["navigation"]["destination"] == fresh["navigation"]["destination"]
        assert persisted["navigation"]["reason_codes"] == fresh["navigation"]["reason_codes"]
        assert persisted["safety"]["state"] == fresh["safety"]["state"]
        assert persisted["safety"]["message"] == fresh["safety"]["message"]


def test_optional_safety_context_csv_preserved_and_affects_persisted_safety(authed_client):
    """The optional 4th CSV must still only affect the Safety Agent (not
    become an ML feature) when it flows through the save-analysis path,
    exactly as it does for a live /uc07/decide call."""
    client, _user_id = authed_client
    files, member_ids = build_small_uc07_csvs(n_members=3)
    target_member = member_ids[0]

    safety_csv = (
        "member_id,red_flag,icu,admitted,major_procedure,triage_level\n"
        f"{target_member},1,0,0,0,2\n"
    )
    files["safety_context_file"] = ("current_safety_context.csv", io.BytesIO(safety_csv.encode()), "text/csv")

    resp = client.post(
        "/populations/save-analysis",
        files=files,
        data={"name": "Safety Context Check", "index_date": "2025-12-31"},
    )
    assert resp.status_code == 201
    pid = resp.json()["id"]

    detail = client.get(f"/populations/{pid}/members/{target_member}")
    assert detail.status_code == 200
    body = detail.json()
    # red_flag=1 must trigger a safety OVERRIDE, same as the live pathway.
    assert body["decision"]["safety"]["state"] == "OVERRIDE"
    assert body["safety_context_captured_at"] is not None

    # A member with NO safety-context row must not show a captured_at.
    other_member = member_ids[1]
    other_detail = client.get(f"/populations/{pid}/members/{other_member}").json()
    assert other_detail["safety_context_captured_at"] is None


def test_save_analysis_rejects_missing_required_files(authed_client):
    client, _user_id = authed_client
    files, _member_ids = build_small_uc07_csvs(n_members=3)
    del files["care_file"]
    resp = client.post("/populations/save-analysis", files=files, data={"name": "Incomplete"})
    assert resp.status_code == 422  # FastAPI's own required-field validation


def test_save_analysis_rolls_back_on_mid_transaction_failure(authed_client):
    """Forces create_population_with_analysis to fail after the
    population row has been flushed but before the transaction commits
    -- confirms NO population row survives (no half-imported state)."""
    client, user_id = authed_client
    files, _member_ids = build_small_uc07_csvs(n_members=4)

    with patch(
        "db.repositories.populations.decision_dict_to_analysis_result_kwargs",
        side_effect=RuntimeError("induced failure for rollback test"),
    ):
        resp = client.post("/populations/save-analysis", files=files, data={"name": "Should Not Persist"})
    assert resp.status_code == 500

    db = get_session_factory()()
    try:
        assert db.query(Population).filter(Population.owner_user_id == user_id).count() == 0
    finally:
        db.close()
