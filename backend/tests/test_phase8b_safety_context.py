"""
Phase 8B tests: single-member current-safety-context evaluation (via the
existing JSON `current_safety_context` field), the new optional batch
`safety_context_file` CSV upload, and the architectural invariants that
must hold across both (risk/navigation unchanged by safety context,
Safety Agent remains final authority, no forbidden language, datasets/
model untouched).

SYNTHETIC DATA MODEL -- DEMONSTRATION ONLY.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
import safety_policy  # noqa: E402

REPO_ROOT = BACKEND_DIR.parent
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture(scope="module")
def members_df():
    return pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")


def _files():
    return {
        "members_file": ("m.csv", open(SYNTHETIC_DIR / "raw_members.csv", "rb"), "text/csv"),
        "ed_visits_file": ("e.csv", open(SYNTHETIC_DIR / "raw_ed_visits.csv", "rb"), "text/csv"),
        "care_file": ("c.csv", open(SYNTHETIC_DIR / "raw_care_history.csv", "rb"), "text/csv"),
    }


def _close(files):
    for _, fh, _ in files.values():
        fh.close()


def _csv_bytes(text: str) -> bytes:
    return text.encode()


def _decide_one(client, member_id, context=None, index_date="2026-07-03"):
    files = _files()
    data = {"member_id": member_id, "index_date": index_date}
    if context is not None:
        data["current_safety_context"] = json.dumps({member_id: context})
    try:
        resp = client.post("/uc07/decide", files=files, data=data)
    finally:
        _close(files)
    return resp


@pytest.fixture(scope="module")
def low_and_high_members(client):
    """Find one real LOW-tier and one real HIGH-tier member from the
    actual population (never hardcode an assumed tier)."""
    files = _files()
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close(files)
    decisions = resp.json()["decisions"]
    low = next(d for d in decisions if d["risk"]["tier"] == "LOW")
    high = next(d for d in decisions if d["risk"]["tier"] == "HIGH")
    return low["member_id"], high["member_id"]


ALL_SAFE = {"red_flag": 0, "icu": 0, "admitted": 0, "major_procedure": 0, "triage_level": 4}

# ---------------------------------------------------------------------------
# SINGLE MEMBER (Part A) -- via the existing JSON current_safety_context path
# ---------------------------------------------------------------------------

def test_1_complete_safe_context_is_clear(client):
    resp = _decide_one(client, "M00001", ALL_SAFE)
    assert resp.status_code == 200
    assert resp.json()["decisions"][0]["safety"]["state"] == "CLEAR"


def test_2_missing_context_is_caution(client):
    resp = _decide_one(client, "M00001", None)
    assert resp.status_code == 200
    d = resp.json()["decisions"][0]["safety"]
    assert d["state"] == "CAUTION"
    assert d["context_completeness"] == "ABSENT"


def test_3_partial_safe_context_is_caution(client):
    resp = _decide_one(client, "M00001", {"red_flag": 0, "triage_level": 4})
    assert resp.status_code == 200
    d = resp.json()["decisions"][0]["safety"]
    assert d["state"] == "CAUTION"
    assert d["context_completeness"] == "PARTIAL"


@pytest.mark.parametrize("trigger", [
    {"red_flag": 1},
    {"icu": 1},
    {"admitted": 1},
    {"major_procedure": 1},
    {"triage_level": 1},
    {"triage_level": 2},
])
def test_4_to_9_each_override_trigger(client, trigger):
    resp = _decide_one(client, "M00001", trigger)
    assert resp.status_code == 200
    decision = resp.json()["decisions"][0]
    assert decision["safety"]["state"] == "OVERRIDE"
    assert decision["safety"]["override"] is True
    assert decision["navigation"]["destination"] is None


def test_10_low_risk_member_plus_emergency_context_is_override(client, low_and_high_members):
    low_member, _ = low_and_high_members
    resp = _decide_one(client, low_member, {"red_flag": 1})
    assert resp.status_code == 200
    decision = resp.json()["decisions"][0]
    assert decision["risk"]["tier"] == "LOW"
    assert decision["safety"]["state"] == "OVERRIDE"
    assert decision["navigation"]["destination"] is None


def test_11_high_risk_member_plus_safe_complete_context_is_clear(client, low_and_high_members):
    _, high_member = low_and_high_members
    resp = _decide_one(client, high_member, ALL_SAFE)
    assert resp.status_code == 200
    decision = resp.json()["decisions"][0]
    assert decision["risk"]["tier"] == "HIGH"
    assert decision["safety"]["state"] == "CLEAR"
    # HIGH risk + CLEAR safety is valid and navigation is NOT suppressed
    assert decision["navigation"]["destination"] is not None


# ---------------------------------------------------------------------------
# BATCH CSV (Part B)
# ---------------------------------------------------------------------------

def test_12_mixed_clear_caution_override_population(client):
    """Matches the spec's own worked example exactly."""
    csv_text = (
        "member_id,red_flag,icu,admitted,major_procedure,triage_level\n"
        "M00001,0,0,0,0,4\n"
        "M00002,,,,,\n"
        "M00003,1,0,0,0,2\n"
        "M00004,0,0,0,0,5\n"
        "M00005,0,1,1,0,2\n"
    )
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert resp.status_code == 200
    by_id = {d["member_id"]: d["safety"] for d in resp.json()["decisions"]}
    assert by_id["M00001"]["state"] == "CLEAR"
    assert by_id["M00002"]["state"] == "CAUTION"
    assert by_id["M00003"]["state"] == "OVERRIDE"
    assert by_id["M00004"]["state"] == "CLEAR"
    assert by_id["M00005"]["state"] == "OVERRIDE"

    states = [d["safety"]["state"] for d in resp.json()["decisions"] if d["member_id"] not in
              ("M00001", "M00002", "M00003", "M00004", "M00005")]
    # Every member with no row in the CSV also gets CAUTION -- not artificially forced equal counts.
    assert all(s == "CAUTION" for s in states)


def test_13_member_missing_from_csv_row_is_caution(client):
    csv_text = "member_id,red_flag\nM00001,0\n"
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"member_id": "M00002", "index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert resp.status_code == 200
    assert resp.json()["decisions"][0]["safety"]["state"] == "CAUTION"


def test_14_blank_safety_values_are_caution(client):
    csv_text = "member_id,red_flag,icu,admitted,major_procedure,triage_level\nM00001,,,,,\n"
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"member_id": "M00001", "index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert resp.status_code == 200
    d = resp.json()["decisions"][0]["safety"]
    assert d["state"] == "CAUTION"
    assert d["context_completeness"] == "ABSENT"


def test_15_invalid_binary_value_returns_4xx(client):
    csv_text = "member_id,red_flag\nM00001,7\n"
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert 400 <= resp.status_code < 500
    assert "Traceback" not in resp.text


def test_16_invalid_triage_returns_4xx(client):
    csv_text = "member_id,triage_level\nM00001,9\n"
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert 400 <= resp.status_code < 500


def test_17_unknown_member_id_in_csv_returns_4xx(client):
    csv_text = "member_id,red_flag\nNOT_A_REAL_MEMBER,1\n"
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert 400 <= resp.status_code < 500


def test_18_duplicate_ambiguous_member_returns_4xx(client):
    csv_text = "member_id,red_flag\nM00001,1\nM00001,0\n"
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert 400 <= resp.status_code < 500


def test_extra_unrecognized_column_in_csv_returns_4xx(client):
    csv_text = "member_id,made_up_column\nM00001,x\n"
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post("/uc07/decide", files=files, data={"index_date": "2026-07-03"})
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert 400 <= resp.status_code < 500


def test_json_context_overrides_csv_for_same_member(client):
    """A member present in BOTH the CSV and the JSON current_safety_context
    field resolves using the JSON entry (the more specific, ad-hoc
    single-member override)."""
    csv_text = "member_id,red_flag\nM00001,1\n"  # CSV says OVERRIDE
    files = _files()
    files["safety_context_file"] = ("safety.csv", io.BytesIO(_csv_bytes(csv_text)), "text/csv")
    try:
        resp = client.post(
            "/uc07/decide", files=files,
            data={
                "member_id": "M00001", "index_date": "2026-07-03",
                "current_safety_context": json.dumps({"M00001": ALL_SAFE}),  # JSON says CLEAR
            },
        )
    finally:
        _close({k: v for k, v in files.items() if k != "safety_context_file"})
    assert resp.status_code == 200
    assert resp.json()["decisions"][0]["safety"]["state"] == "CLEAR"


# ---------------------------------------------------------------------------
# ARCHITECTURE invariants
# ---------------------------------------------------------------------------

def test_19_frontend_has_no_independent_safety_decision_logic():
    """Grep-based, mirrors backend/tests/test_legacy_isolation.py's style:
    the frontend must never itself COMPUTE/ASSIGN a CLEAR/CAUTION/
    OVERRIDE string at runtime from other inputs -- that decision must
    come only from a backend response. Reading/comparing an
    already-backend-provided `safety.state === "OVERRIDE"` for display
    purposes (e.g. choosing an icon) is fine and intentionally NOT
    flagged; only a true assignment operator (single `=`, never `===`/
    `!==`) is forbidden, and TypeScript `type X = "A" | "B"` union
    declarations are excluded (a real assignment never contains a `|`
    union separator on the line)."""
    import re

    frontend_src = REPO_ROOT / "frontend" / "src"
    # single `=` (not part of ==, ===, !=, !==, <=, >=) immediately
    # followed by a CLEAR/CAUTION/OVERRIDE string literal
    assignment_re = re.compile(r'(?<![=!<>])=(?!=)\s*["\'](CLEAR|CAUTION|OVERRIDE)["\']')

    offenders = []
    for path in frontend_src.rglob("*.ts*"):
        if "__tests__" in path.parts or "/test/" in str(path).replace("\\", "/") or path.name == "types.ts":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if (
                stripped.startswith("//") or stripped.startswith("*") or "|" in stripped
                or "type " in stripped or stripped.startswith("<")
            ):
                # union-type declarations, comments, and JSX attribute/element
                # lines (e.g. `<option value="CLEAR">`, a static dropdown
                # choice -- not computed decision logic) are not runtime logic
                continue
            if assignment_re.search(line):
                offenders.append((str(path), lineno, line.strip()))
    assert offenders == [], f"frontend source assigns a safety-state literal directly: {offenders}"


def test_20_safety_agent_remains_final_authority_override_suppresses_navigation(client):
    resp = _decide_one(client, "M00001", {"red_flag": 1})
    decision = resp.json()["decisions"][0]
    assert decision["safety"]["state"] == "OVERRIDE"
    assert decision["navigation"]["destination"] is None
    assert decision["navigation"]["reason_codes"] == []


def test_21_22_23_risk_and_navigation_unchanged_by_safety_context(client):
    baseline = _decide_one(client, "M00001", None).json()["decisions"][0]
    clear = _decide_one(client, "M00001", ALL_SAFE).json()["decisions"][0]
    override = _decide_one(client, "M00001", {"red_flag": 1}).json()["decisions"][0]

    # 21: risk probability identical across all three safety contexts
    assert baseline["risk"]["probability"] == clear["risk"]["probability"] == override["risk"]["probability"]
    # 22: risk tier identical
    assert baseline["risk"]["tier"] == clear["risk"]["tier"] == override["risk"]["tier"]
    # 23: navigation logic (destination/reason_codes/explanation) unchanged
    # between CAUTION and CLEAR (neither suppresses navigation); OVERRIDE
    # suppresses navigation by design (not a "navigation logic change").
    assert baseline["navigation"]["destination"] == clear["navigation"]["destination"]
    assert baseline["navigation"]["reason_codes"] == clear["navigation"]["reason_codes"]
    assert baseline["navigation"]["explanation"] == clear["navigation"]["explanation"]
    assert override["navigation"]["destination"] is None


def test_24_25_26_model_and_dataset_hashes_unchanged():
    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    pre = json.loads((REPO_ROOT / "artifacts" / "phase8b_current_safety_context" / "pre_work_hashes.json").read_text())
    for rel, expected in pre.items():
        assert sha256(REPO_ROOT / rel) == expected, f"{rel} changed during Phase 8B"


def test_28_no_prohibited_language_in_any_new_safety_message(client):
    responses = [
        _decide_one(client, "M00001", None),
        _decide_one(client, "M00001", ALL_SAFE),
        _decide_one(client, "M00001", {"red_flag": 1}),
        _decide_one(client, "M00001", {"icu": 1}),
    ]
    for resp in responses:
        decision = resp.json()["decisions"][0]
        assert safety_policy.check_text(decision["safety"]["message"]) == []
        assert safety_policy.check_text(decision["navigation"]["explanation"]) == []
