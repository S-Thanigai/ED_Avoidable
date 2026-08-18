"""
Server-side pagination/search/filter tests for GET
/populations/{id}/members. Uses a moderately-sized (60-member) synthetic
population -- large enough to exercise real multi-page pagination math
(the SQL OFFSET/FETCH + COUNT logic behaves identically at 60 rows as it
does at 10,000; a 60-row fixture keeps this suite fast) while still
proving the endpoint never returns more than page_size rows at once.
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
N_MEMBERS = 62


@pytest.fixture(scope="module")
def saved_population():
    client = TestClient(main.app)
    email = unique_email("pagination")
    resp = client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    user_id = resp.json()["id"]

    files, member_ids = build_small_uc07_csvs(n_members=N_MEMBERS)
    save_resp = client.post("/populations/save-analysis", files=files, data={"name": "Pagination Fixture"})
    assert save_resp.status_code == 201
    pid = save_resp.json()["id"]

    yield client, pid, member_ids

    delete_user_cascade(user_id)


def test_default_page_size_is_15(saved_population):
    client, pid, _member_ids = saved_population
    resp = client.get(f"/populations/{pid}/members")
    assert resp.status_code == 200
    body = resp.json()
    assert body["page_size"] == 15
    assert len(body["items"]) == 15
    assert body["total_items"] == N_MEMBERS
    assert body["total_pages"] == 5  # ceil(62/15)


def test_pagination_never_returns_more_than_page_size(saved_population):
    client, pid, _member_ids = saved_population
    resp = client.get(f"/populations/{pid}/members", params={"page": 1, "page_size": 10})
    body = resp.json()
    assert len(body["items"]) == 10
    assert body["total_items"] == N_MEMBERS


def test_pagination_pages_do_not_overlap_and_cover_everything(saved_population):
    client, pid, _member_ids = saved_population
    seen_member_ids: set[str] = set()
    page = 1
    page_size = 20
    while True:
        resp = client.get(f"/populations/{pid}/members", params={"page": page, "page_size": page_size, "sort_key": "member_id"})
        body = resp.json()
        page_ids = {item["member_id"] for item in body["items"]}
        assert not (page_ids & seen_member_ids), "pages must not overlap"
        seen_member_ids |= page_ids
        if page >= body["total_pages"]:
            break
        page += 1
    assert len(seen_member_ids) == N_MEMBERS


def test_search_by_member_id_substring(saved_population):
    client, pid, member_ids = saved_population
    target = member_ids[3]
    resp = client.get(f"/populations/{pid}/members", params={"search": target})
    body = resp.json()
    assert body["total_items"] == 1
    assert body["items"][0]["member_id"] == target


def test_search_with_no_matches_returns_empty_not_error(saved_population):
    client, pid, _member_ids = saved_population
    resp = client.get(f"/populations/{pid}/members", params={"search": "NO_SUCH_MEMBER_ID_XYZ"})
    assert resp.status_code == 200
    assert resp.json()["total_items"] == 0
    assert resp.json()["items"] == []


def test_filter_by_tier(saved_population):
    client, pid, _member_ids = saved_population
    resp = client.get(f"/populations/{pid}/members", params={"tier": "LOW", "page_size": 100})
    body = resp.json()
    assert all(item["risk"]["tier"] == "LOW" for item in body["items"])


def test_filter_by_probability_range(saved_population):
    client, pid, _member_ids = saved_population
    resp = client.get(f"/populations/{pid}/members", params={"prob_min": 0.0, "prob_max": 0.05, "page_size": 100})
    body = resp.json()
    assert all(0.0 <= item["risk"]["probability"] <= 0.05 for item in body["items"])


def test_sort_by_probability_ascending(saved_population):
    client, pid, _member_ids = saved_population
    resp = client.get(f"/populations/{pid}/members", params={"sort_key": "probability", "sort_dir": "asc", "page_size": 100})
    probs = [item["risk"]["probability"] for item in resp.json()["items"]]
    assert probs == sorted(probs)


def test_page_size_cannot_be_forced_unbounded(saved_population):
    """A client cannot bypass server-side pagination by requesting an
    enormous page_size -- the endpoint clamps it."""
    client, pid, _member_ids = saved_population
    resp = client.get(f"/populations/{pid}/members", params={"page_size": 100000})
    assert resp.status_code == 200
    assert resp.json()["page_size"] <= 200
