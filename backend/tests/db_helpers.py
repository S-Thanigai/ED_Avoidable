"""
db_helpers.py
--------------
Shared helpers for the Azure SQL persistence test suite (test_db_config,
test_auth, test_authorization, test_populations_save_and_reload,
test_pagination, test_sql_injection). Not a test module itself (no
test_ prefix) -- pytest will not collect it.

These tests run against the REAL Azure SQL database configured in
backend/.env (there is no local/mocked substitute -- the whole point of
this feature is Azure SQL integration). Every test that creates a user
MUST use `unique_email()` and clean up via `delete_user_cascade()` in a
finally/fixture teardown, so repeated test runs never accumulate junk in
the shared database.
"""
from __future__ import annotations

import io
import sys
import uuid
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
for _subdir in ("pit", "agents"):
    _p = str(BACKEND_DIR / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from db.engine import get_session_factory  # noqa: E402
from db.models import User  # noqa: E402


def unique_email(prefix: str = "test") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}@example.com"


def delete_user_cascade(user_id: int) -> None:
    """Deletes a test user and (via ON DELETE CASCADE) every session and
    population -- and everything under each population -- it owns. Test
    teardown only; never used by application code."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        user = db.get(User, user_id)
        if user is not None:
            db.delete(user)
            db.commit()
    finally:
        db.close()


def _csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def build_small_uc07_csvs(n_members: int = 6, index_date: str = "2025-12-31"):
    """Builds small, valid members/ed_visits/care CSVs (in-memory,
    multipart-ready) satisfying every constraint
    backend/agents/input_validation.py enforces -- deterministic and
    small so tests run quickly, but real enough to exercise the full
    orchestrator pipeline (not a mocked/stubbed model)."""
    member_ids = [f"TST{i:04d}" for i in range(n_members)]

    members_df = pd.DataFrame(
        {
            "member_id": member_ids,
            "age": [30 + (i * 7) % 60 for i in range(n_members)],
            "gender": ["F" if i % 2 == 0 else "M" for i in range(n_members)],
            "diabetes": [i % 2 for i in range(n_members)],
            "copd": [0] * n_members,
            "hypertension": [i % 3 == 0 for i in range(n_members)],
            "chf": [0] * n_members,
            "asthma": [0] * n_members,
            "ckd": [0] * n_members,
            "transportation_barrier": [i % 2 for i in range(n_members)],
            "telehealth_available": [1] * n_members,
            "pcp_distance_miles": [round(1.0 + i * 0.5, 1) for i in range(n_members)],
            "urgent_care_distance_miles": [round(2.0 + i * 0.3, 1) for i in range(n_members)],
        }
    )
    members_df["hypertension"] = members_df["hypertension"].astype(int)
    members_df["num_chronic_conditions"] = members_df[
        ["diabetes", "copd", "hypertension", "chf", "asthma", "ckd"]
    ].sum(axis=1)

    ed_rows = []
    for i, member_id in enumerate(member_ids):
        # Roughly half the members get an ED visit history; deterministic,
        # not random, so test assertions are stable across runs.
        if i % 2 == 0:
            ed_rows.append(
                {
                    "visit_id": f"V{i:04d}",
                    "member_id": member_id,
                    "visit_date": "2025-10-05",
                    "diagnosis": "UTI",
                    "triage_level": 3,
                    "admitted": 0,
                    "icu": 0,
                    "major_procedure": 0,
                    "cost": 500.0 + i,
                    "red_flag": 0,
                }
            )
    ed_df = pd.DataFrame(
        ed_rows,
        columns=[
            "visit_id", "member_id", "visit_date", "diagnosis", "triage_level",
            "admitted", "icu", "major_procedure", "cost", "red_flag",
        ],
    )

    care_rows = []
    for i, member_id in enumerate(member_ids):
        if i % 3 == 0:
            care_rows.append(
                {"care_id": f"C{i:04d}", "member_id": member_id, "visit_date": "2025-11-01", "care_type": "PCP"}
            )
    care_df = pd.DataFrame(care_rows, columns=["care_id", "member_id", "visit_date", "care_type"])

    return {
        "members_file": ("members.csv", _csv_bytes(members_df), "text/csv"),
        "ed_visits_file": ("ed_visits.csv", _csv_bytes(ed_df), "text/csv"),
        "care_file": ("care.csv", _csv_bytes(care_df), "text/csv"),
    }, member_ids


def get_client():
    import main

    return main
