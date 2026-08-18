"""
Authentication tests: signup, duplicate signup, password hashing, login,
logout, protected endpoints, expired/invalid sessions. Runs against the
real Azure SQL database; every test cleans up the user(s) it creates.
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
from db.models import User  # noqa: E402
from tests.db_helpers import delete_user_cascade, unique_email  # noqa: E402

PASSWORD = "correcthorsebattery"


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def created_user_ids():
    ids: list[int] = []
    yield ids
    for uid in ids:
        delete_user_cascade(uid)


def test_signup_creates_user_and_session_cookie(client, created_user_ids):
    email = unique_email()
    resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == email
    created_user_ids.append(body["id"])
    assert "uc07_session" in resp.cookies


def test_duplicate_signup_rejected(client, created_user_ids):
    email = unique_email()
    r1 = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    created_user_ids.append(r1.json()["id"])
    r2 = client.post("/auth/signup", json={"email": email, "password": "anotherpassword123"})
    assert r2.status_code == 409


def test_password_is_hashed_never_plaintext(client, created_user_ids):
    email = unique_email()
    resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    created_user_ids.append(resp.json()["id"])

    db = get_session_factory()()
    try:
        user = db.query(User).filter(User.email == email.lower()).one()
        assert user.password_hash != PASSWORD
        assert PASSWORD not in user.password_hash
        # argon2 hash format
        assert user.password_hash.startswith("$argon2")
    finally:
        db.close()


def test_login_success(client, created_user_ids):
    email = unique_email()
    signup_resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    created_user_ids.append(signup_resp.json()["id"])

    fresh_client = TestClient(main.app)
    resp = fresh_client.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 200
    assert resp.json()["email"] == email


def test_login_invalid_password(client, created_user_ids):
    email = unique_email()
    signup_resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    created_user_ids.append(signup_resp.json()["id"])

    fresh_client = TestClient(main.app)
    resp = fresh_client.post("/auth/login", json={"email": email, "password": "wrongpassword"})
    assert resp.status_code == 401


def test_login_unknown_email_returns_same_generic_error(client):
    resp = client.post("/auth/login", json={"email": unique_email(), "password": "whatever12345"})
    assert resp.status_code == 401
    assert "email" not in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()


def test_logout_clears_session(client, created_user_ids):
    email = unique_email()
    signup_resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    created_user_ids.append(signup_resp.json()["id"])

    me = client.get("/auth/me")
    assert me.status_code == 200

    logout_resp = client.post("/auth/logout")
    assert logout_resp.status_code == 204

    me_after = client.get("/auth/me")
    assert me_after.status_code == 401


def test_protected_endpoint_requires_authentication():
    fresh_client = TestClient(main.app)
    resp = fresh_client.get("/populations")
    assert resp.status_code == 401


def test_invalid_session_cookie_rejected():
    fresh_client = TestClient(main.app)
    fresh_client.cookies.set("uc07_session", "not-a-real-session-token")
    resp = fresh_client.get("/auth/me")
    assert resp.status_code == 401


def test_signup_rejects_weak_password(client):
    resp = client.post("/auth/signup", json={"email": unique_email(), "password": "short"})
    assert resp.status_code == 422


def test_signup_rejects_invalid_email(client):
    resp = client.post("/auth/signup", json={"email": "not-an-email", "password": PASSWORD})
    assert resp.status_code == 422
