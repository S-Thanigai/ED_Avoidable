"""
Security tests: SQL injection resistance (parameterized queries only,
never string-concatenated SQL) and credential-exposure checks (DB
password never reaches the frontend/response body/error text).
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
from db.engine import get_session_factory  # noqa: E402
from db.models import Population, User  # noqa: E402
from tests.db_helpers import build_small_uc07_csvs, delete_user_cascade, unique_email  # noqa: E402

PASSWORD = "correcthorsebattery"

SQLI_PAYLOADS = [
    "'; DROP TABLE users; --",
    "' OR '1'='1",
    "'; UPDATE populations SET owner_user_id=1; --",
    "Robert'); DROP TABLE population_members;--",
]


@pytest.fixture
def authed_client_with_population():
    client = TestClient(main.app)
    email = unique_email("sectest")
    resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    user_id = resp.json()["id"]

    files, member_ids = build_small_uc07_csvs(n_members=4)
    save_resp = client.post("/populations/save-analysis", files=files, data={"name": "Security Test Population"})
    pid = save_resp.json()["id"]

    yield client, pid, member_ids

    delete_user_cascade(user_id)


@pytest.mark.parametrize("payload", SQLI_PAYLOADS)
def test_search_param_is_safe_against_sql_injection(authed_client_with_population, payload):
    client, pid, _member_ids = authed_client_with_population
    resp = client.get(f"/populations/{pid}/members", params={"search": payload})
    # Must not error -- a parameterized query treats this as a literal
    # string with no matches, never executes it as SQL.
    assert resp.status_code == 200
    assert resp.json()["total_items"] == 0

    # The users/populations tables must still exist and be intact.
    db = get_session_factory()()
    try:
        assert db.query(User).count() >= 1
        assert db.get(Population, pid) is not None
    finally:
        db.close()


@pytest.mark.parametrize("payload", SQLI_PAYLOADS)
def test_population_name_is_safe_against_sql_injection(authed_client_with_population, payload):
    client, _pid, _member_ids = authed_client_with_population
    files, _ = build_small_uc07_csvs(n_members=2)
    resp = client.post("/populations/save-analysis", files=files, data={"name": payload})
    assert resp.status_code == 201
    assert resp.json()["name"] == payload  # stored/echoed verbatim as data, never executed

    db = get_session_factory()()
    try:
        assert db.query(User).count() >= 1
    finally:
        db.close()


@pytest.mark.parametrize("payload", SQLI_PAYLOADS)
def test_login_email_field_is_safe_against_sql_injection(payload):
    client = TestClient(main.app)
    resp = client.post("/auth/login", json={"email": payload, "password": "irrelevant"})
    assert resp.status_code in (401, 422)


def test_health_endpoint_never_leaks_credentials():
    client = TestClient(main.app)
    resp = client.get("/health")
    body = resp.json()
    text = str(body)
    assert "AZURE_SQL_PASSWORD" not in text
    assert "PWD=" not in text
    assert "://" not in text  # no connection string / URL fragment at all
    assert set(body.keys()) >= {"database_configured", "database_provider"}


def test_password_never_appears_in_signup_response():
    client = TestClient(main.app)
    email = unique_email("nopwleak")
    resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    try:
        assert PASSWORD not in str(resp.json())
        assert "password" not in resp.json()
        assert "password_hash" not in resp.json()
    finally:
        delete_user_cascade(resp.json()["id"])


def test_unauthorized_ids_return_401_not_500():
    """A non-existent/garbage population id must never surface a raw
    exception/traceback -- always a clean 401 (unauthenticated) here,
    since there's no session at all."""
    client = TestClient(main.app)
    resp = client.get("/populations/999999999")
    assert resp.status_code == 401
    assert "Traceback" not in resp.text
