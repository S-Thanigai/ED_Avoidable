"""
Phase 8D Part 15 -- LEGACY CONCURRENCY tests (numbered 12-13 in the spec).

Verifies POST /predict (legacy, include_shap=True) no longer blocks the
rest of the FastAPI process while it runs. This was a real, demonstrated
bug found by the Phase 8C health check: predict()'s per-row legacy SHAP
TreeExplainer loop (predict.py) holds the GIL near-continuously, so even
wrapping it in starlette.concurrency.run_in_threadpool (a normal, usually-
sufficient fix) was empirically PROVEN insufficient during Phase 8D's own
investigation -- a controlled test showed a concurrent asyncio task
starved for the full duration of a threadpool-wrapped predict() call.
The actual fix (backend/main.py) routes predict() through a small
ProcessPoolExecutor instead, since a separate OS process has its own,
independent GIL.

These tests make REAL concurrent ASGI requests against the actual
FastAPI app (httpx.AsyncClient + ASGITransport, no live server process
needed) with a genuinely small population slice, so this stays fast
(a few seconds) while still exercising the real predict()/SHAP code path,
not a mock.
"""
import asyncio
import io
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

import sys
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402


def _small_legacy_files(n_members: int = 20) -> dict[str, bytes]:
    """A genuinely small slice of the real legacy dataset -- enough rows
    to trigger the real predict()/SHAP code path (not mocked), small
    enough to keep this test fast. Only `members_file` is trimmed (so
    predict() only scores a handful of members, keeping runtime short);
    ed_visits/care are left FULL, unfiltered -- the legacy feature
    engineering builds its diagnosis one-hot columns from whatever
    categories are PRESENT in the uploaded ed_visits file, so trimming it
    down to just the selected members' rows can accidentally drop a rare
    diagnosis category entirely and break column alignment. That would be
    a real behavior difference from a normal /predict call, not something
    this test is meant to exercise."""
    members = pd.read_csv(BACKEND_DIR.parent / "raw_members.csv").head(n_members)
    ed = pd.read_csv(BACKEND_DIR.parent / "raw_ed_visits.csv")
    care = pd.read_csv(BACKEND_DIR.parent / "raw_care_history.csv")

    def _to_bytes(df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    return {
        "members_file": _to_bytes(members),
        "ed_visits_file": _to_bytes(ed),
        "care_file": _to_bytes(care),
    }


async def _predict_request(client: httpx.AsyncClient, files: dict[str, bytes]) -> httpx.Response:
    upload = {
        "members_file": ("members.csv", files["members_file"], "text/csv"),
        "ed_visits_file": ("ed.csv", files["ed_visits_file"], "text/csv"),
        "care_file": ("care.csv", files["care_file"], "text/csv"),
    }
    return await client.post("/predict", files=upload, timeout=120)


async def _run_concurrency_check(check_path: str, check_kwargs: dict | None = None):
    """Fires a real /predict request, then -- WHILE it is still in
    flight -- fires a request to `check_path` and measures how long it
    takes. Returns (predict_response, check_response, check_elapsed_s)."""
    files = _small_legacy_files()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        predict_task = asyncio.create_task(_predict_request(client, files))
        await asyncio.sleep(0.3)  # let /predict actually start running

        t0 = time.monotonic()
        if check_kwargs and check_kwargs.get("method") == "POST":
            check_resp = await client.post(check_path, **{k: v for k, v in check_kwargs.items() if k != "method"})
        else:
            check_resp = await client.get(check_path)
        check_elapsed = time.monotonic() - t0

        predict_resp = await predict_task
    return predict_resp, check_resp, check_elapsed


# ---- 12. /predict does not block /health ----

def test_12_predict_does_not_block_health():
    predict_resp, health_resp, elapsed = asyncio.run(_run_concurrency_check("/health"))
    assert predict_resp.status_code == 200
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"
    # generous bound -- the point is "responds promptly", not a tight SLA;
    # the PRE-FIX behavior was "does not respond at all for 20+ seconds"
    assert elapsed < 5.0, f"/health took {elapsed:.2f}s while /predict was running (should be near-instant)"


# ---- 13. /predict does not block /uc07/decide ----

def test_13_predict_does_not_block_uc07_decide():
    async def _uc07_decide_check(client: httpx.AsyncClient) -> httpx.Response:
        synthetic_dir = REPO_ROOT / "data" / "synthetic"
        upload = {
            "members_file": ("m.csv", (synthetic_dir / "raw_members.csv").read_bytes(), "text/csv"),
            "ed_visits_file": ("e.csv", (synthetic_dir / "raw_ed_visits.csv").read_bytes(), "text/csv"),
            "care_file": ("c.csv", (synthetic_dir / "raw_care_history.csv").read_bytes(), "text/csv"),
        }
        return await client.post(
            "/uc07/decide",
            files=upload,
            data={"index_date": "2025-01-01", "member_id": "M00001"},
            timeout=60,
        )

    files = _small_legacy_files()
    transport = httpx.ASGITransport(app=main.app)

    async def _run():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            predict_task = asyncio.create_task(_predict_request(client, files))
            await asyncio.sleep(0.3)
            t0 = time.monotonic()
            decide_resp = await _uc07_decide_check(client)
            elapsed = time.monotonic() - t0
            predict_resp = await predict_task
            return predict_resp, decide_resp, elapsed

    predict_resp, decide_resp, elapsed = asyncio.run(_run())
    assert predict_resp.status_code == 200
    assert decide_resp.status_code == 200
    assert decide_resp.json()["decisions"][0]["member_id"] == "M00001"
    # /uc07/decide's own baseline latency (SHAP explainer build, unrelated
    # to /predict) is a few seconds -- bound generously above that, since
    # the point is "not frozen for the duration of /predict", not a tight SLA
    assert elapsed < 30.0, f"/uc07/decide took {elapsed:.2f}s while /predict was running"


def test_predict_output_unchanged_by_process_pool_routing():
    """The Phase 8D fix changes WHERE predict() runs, never WHAT it
    returns -- confirms the real legacy output shape/columns survive the
    process-pool round trip unmodified."""
    predict_resp, _, _ = asyncio.run(_run_concurrency_check("/health"))
    assert predict_resp.status_code == 200
    assert predict_resp.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    df = pd.read_excel(io.BytesIO(predict_resp.content))
    assert len(df) == 20
    for col in ("member_id", "risk_probability", "risk_score", "risk_category", "shap_explanation_summary"):
        assert col in df.columns
