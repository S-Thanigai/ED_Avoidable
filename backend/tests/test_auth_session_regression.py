"""
Regression tests for the auth-session bug: signup/login returned 201/200
but the very next authenticated request (GET /populations) 401'd in a
real browser.

Root cause (see frontend/src/apiConfig.ts's API_BASE_URL comment and
backend/auth.py's issue_session_cookie): the frontend's default backend
URL was "http://127.0.0.1:8001" while Vite serves the app from
"http://localhost:5173". Browsers treat "localhost" and "127.0.0.1" as
different SITES, so with the (correct, for local http:// dev)
SameSite=Lax cookie, the browser accepted+stored the Set-Cookie from
signup/login (SameSite never blocks *receiving* a cookie) but then
refused to attach it to the next fetch/XHR, which was cross-site. Fixed
by pointing the frontend's default API_BASE_URL at "localhost" too, so
frontend and backend are same-site (same host, different port -- port
is irrelevant to "site").

IMPORTANT CAVEAT this test file's own docstring must be honest about:
starlette's TestClient (httpx) does NOT enforce real browser cross-site
cookie rules -- it stores and resends whatever cookies the server set,
regardless of "hostname". So the functional tests below (which all use
one TestClient/cookie-jar per user, matching real frontend usage)
CANNOT by themselves reproduce or catch the SameSite/cross-host bug --
they only prove the server-side session mechanics (cookie issued ->
looked up -> ownership resolved) are correct. The header-attribute test
at the bottom is what actually guards against a regression of the real
bug: it asserts the Set-Cookie response has no explicit Domain (so it
is host-scoped, never Domain-scoped) and is SameSite=Lax, both
preconditions for "putting frontend and backend on the same hostname
fixes it" to actually hold.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
from tests.db_helpers import delete_user_cascade, unique_email  # noqa: E402

PASSWORD = "correcthorsebattery"


@pytest.fixture
def created_user_ids():
    ids: list[int] = []
    yield ids
    for uid in ids:
        delete_user_cascade(uid)


def test_signup_then_me_returns_200(created_user_ids):
    client = TestClient(main.app)
    signup = client.post("/auth/signup", json={"email": unique_email(), "password": PASSWORD})
    assert signup.status_code == 201
    created_user_ids.append(signup.json()["id"])

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["id"] == signup.json()["id"]


def test_signup_then_populations_returns_200(created_user_ids):
    client = TestClient(main.app)
    signup = client.post("/auth/signup", json={"email": unique_email(), "password": PASSWORD})
    created_user_ids.append(signup.json()["id"])

    populations = client.get("/populations")
    assert populations.status_code == 200
    assert populations.json() == []


def test_login_then_me_returns_200(created_user_ids):
    signup_client = TestClient(main.app)
    email = unique_email()
    signup = signup_client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    created_user_ids.append(signup.json()["id"])

    login_client = TestClient(main.app)  # fresh cookie jar, simulating a new browser session
    login = login_client.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200

    me = login_client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_login_then_populations_returns_200(created_user_ids):
    signup_client = TestClient(main.app)
    email = unique_email()
    signup = signup_client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    created_user_ids.append(signup.json()["id"])

    login_client = TestClient(main.app)
    login_client.post("/auth/login", json={"email": email, "password": PASSWORD})

    populations = login_client.get("/populations")
    assert populations.status_code == 200


def test_logout_then_me_returns_401(created_user_ids):
    client = TestClient(main.app)
    signup = client.post("/auth/signup", json={"email": unique_email(), "password": PASSWORD})
    created_user_ids.append(signup.json()["id"])

    assert client.get("/auth/me").status_code == 200
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/auth/me").status_code == 401


def test_unauthenticated_populations_returns_401():
    client = TestClient(main.app)
    resp = client.get("/populations")
    assert resp.status_code == 401


# ---- The test that actually guards against a regression of THIS bug ----


def test_session_cookie_attributes_are_host_scoped_and_lax(created_user_ids):
    """Asserts the properties that make 'put frontend and backend on the
    same hostname' the correct fix:
      - no explicit Domain attribute -> the cookie is scoped to the
        exact host that set it, never broadened/mismatched
      - SameSite=Lax -> correct, safe default for local http:// dev,
        but ONLY works when the requesting page is same-site with the
        API host (this is the actual invariant the apiConfig.ts fix
        upholds)
      - HttpOnly -> never readable/stealable via JS
      - Path=/ -> sent on every API path, not scoped to /auth only
    A raw Set-Cookie header is inspected directly (not requests'
    higher-level cookie jar) so a future change to any of these
    attributes fails this test immediately."""
    client = TestClient(main.app)
    resp = client.post("/auth/signup", json={"email": unique_email(), "password": PASSWORD})
    created_user_ids.append(resp.json()["id"])

    set_cookie = resp.headers.get("set-cookie")
    assert set_cookie is not None, "signup must return a Set-Cookie header"
    lowered = set_cookie.lower()

    assert "uc07_session=" in set_cookie
    assert "domain=" not in lowered, "an explicit Domain would broaden/mismatch cookie scope -- must not be set"
    assert "samesite=lax" in lowered
    assert "httponly" in lowered
    assert "path=/" in lowered
    # Secure must be OFF by default for local http:// dev (COOKIE_SECURE
    # unset) -- a Secure cookie is silently never sent over plain http,
    # which would be an even worse variant of this same bug.
    assert "secure" not in lowered
