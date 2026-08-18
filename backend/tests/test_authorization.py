"""
Per-user data isolation tests. User A saves a population; User B (a
completely different account) must be unable to see, read, search,
paginate, or delete anything belonging to User A, even when directly
addressing User A's population/member IDs. Every ownership check is
server-side (backend/db/repositories/populations.py takes owner_user_id
as an explicit argument derived only from the authenticated session --
see backend/auth.py's get_current_user) -- there is no `owner_user_id`
field anywhere in a request body/query/path a client could forge.

Note on communication endpoints: POST /uc07/report, POST /uc07/email,
and POST /uc07/explain never accept a population_id or do a database
lookup at all (see backend/main.py's module docstring above those
routes) -- they only render/transmit a decision summary the CALLER
already obtained. The ownership boundary for a SAVED member's data is
therefore enforced entirely at the point that summary is fetched (GET
/populations/{id}/members/{member_id}, covered below by
test_user_b_cannot_read_user_a_member) -- User B can never legitimately
obtain User A's member data to build a report/email request in the
first place.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
from tests.db_helpers import build_small_uc07_csvs, delete_user_cascade, unique_email  # noqa: E402

PASSWORD = "correcthorsebattery"


@pytest.fixture
def two_users():
    """Two independent, authenticated TestClients (separate cookie jars
    == separate sessions == separate identities), each owning nothing
    initially. Cleans up both users (and anything they created, via
    ON DELETE CASCADE) afterward."""
    client_a = TestClient(main.app)
    client_b = TestClient(main.app)

    email_a, email_b = unique_email("usera"), unique_email("userb")
    resp_a = client_a.post("/auth/signup", json={"email": email_a, "password": PASSWORD})
    resp_b = client_b.post("/auth/signup", json={"email": email_b, "password": PASSWORD})
    assert resp_a.status_code == 201 and resp_b.status_code == 201

    yield client_a, client_b

    delete_user_cascade(resp_a.json()["id"])
    delete_user_cascade(resp_b.json()["id"])


@pytest.fixture
def population_owned_by_a(two_users):
    client_a, client_b = two_users
    files, member_ids = build_small_uc07_csvs(n_members=5)
    resp = client_a.post("/populations/save-analysis", files=files, data={"name": "User A's Population"})
    assert resp.status_code == 201
    return client_a, client_b, resp.json()["id"], member_ids[0]


def test_user_b_does_not_see_user_a_population_in_list(population_owned_by_a):
    _client_a, client_b, _pid, _member_id = population_owned_by_a
    resp = client_b.get("/populations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_user_b_cannot_read_user_a_population_summary(population_owned_by_a):
    _client_a, client_b, pid, _member_id = population_owned_by_a
    resp = client_b.get(f"/populations/{pid}")
    assert resp.status_code == 404


def test_user_b_cannot_list_user_a_members(population_owned_by_a):
    _client_a, client_b, pid, _member_id = population_owned_by_a
    resp = client_b.get(f"/populations/{pid}/members")
    assert resp.status_code == 404


def test_user_b_cannot_read_user_a_member(population_owned_by_a):
    _client_a, client_b, pid, member_id = population_owned_by_a
    resp = client_b.get(f"/populations/{pid}/members/{member_id}")
    assert resp.status_code == 404


def test_user_b_cannot_delete_user_a_population(population_owned_by_a):
    client_a, client_b, pid, _member_id = population_owned_by_a
    resp = client_b.delete(f"/populations/{pid}")
    assert resp.status_code == 404

    # Confirm it was NOT actually deleted -- User A can still see it.
    still_there = client_a.get(f"/populations/{pid}")
    assert still_there.status_code == 200


def test_user_a_can_read_and_delete_own_population(population_owned_by_a):
    client_a, _client_b, pid, member_id = population_owned_by_a
    assert client_a.get(f"/populations/{pid}").status_code == 200
    assert client_a.get(f"/populations/{pid}/members/{member_id}").status_code == 200
    assert client_a.delete(f"/populations/{pid}").status_code == 204
    assert client_a.get(f"/populations/{pid}").status_code == 404


def test_anonymous_caller_cannot_access_any_population(population_owned_by_a):
    _client_a, _client_b, pid, _member_id = population_owned_by_a
    anon = TestClient(main.app)
    assert anon.get("/populations").status_code == 401
    assert anon.get(f"/populations/{pid}").status_code == 401
    assert anon.delete(f"/populations/{pid}").status_code == 401
