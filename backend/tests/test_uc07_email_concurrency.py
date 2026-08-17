"""
Phase 9 SMTP reliability pass -- Section 8: proves POST /uc07/email does
NOT block the rest of the FastAPI process while a slow SMTP send is in
flight. Same methodology as test_phase8d_legacy_concurrency.py (real
concurrent ASGI requests via httpx.AsyncClient + ASGITransport, no live
server process, no real network) -- here the "slow" part is a mocked
SmtpEmailProvider.send() that sleeps, standing in for a slow/unreachable
real SMTP server, so this never depends on network conditions or
real credentials.

POST /uc07/email and POST /uc07/report are plain `def` (not `async
def`) handlers -- FastAPI/Starlette automatically runs a synchronous
route function via `starlette.concurrency.run_in_threadpool`, off the
main asyncio event loop. This test verifies that guarantee holds in
practice for the blocking SMTP I/O inside EmailService, not just in
principle.
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
from services.email_service import SmtpSendTiming  # noqa: E402

_SLOW_SEND_SECONDS = 1.5


def _report_request() -> dict:
    return {
        "member": {"member_id": "M00042", "name": "Taylor Smith", "email": "taylor.smith@example.com", "age": 57, "gender": "F"},
        "risk": {
            "probability": 0.317, "tier": "MODERATE", "model_version": "uc07-risk-synthetic-v1",
            "factors": [{"display_name": "Prior ED visits (90d)", "direction": "INCREASES_RISK"}],
        },
        "navigation": {"destination": "TELEHEALTH", "reason_codes": ["TELEHEALTH_AVAILABLE"]},
        "safety": {
            "state": "CLEAR", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED",
            "message": "No configured safety override was triggered for this encounter.",
        },
        "synthetic_model": True, "dataset_id": "synthetic_uc07_v1",
    }


def _slow_provider_send(self, message, timeout_seconds):
    time.sleep(_SLOW_SEND_SECONDS)
    return SmtpSendTiming(connection_ms=1.0, send_ms=_SLOW_SEND_SECONDS * 1000)


async def _email_request(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post(
        "/uc07/email",
        json={
            "report": _report_request(), "to_email": "taylor.smith@example.com",
            "subject": "Your Care Navigation Summary", "body": "Hello, please see the attached report.",
        },
        timeout=30,
    )


async def _run_concurrency_check():
    with patch("services.email_service.SmtpEmailProvider.send", _slow_provider_send), \
         patch.dict("os.environ", {
             "EMAIL_ENABLED": "true", "SMTP_HOST": "smtp.example.com", "SMTP_FROM_EMAIL": "care@example.com",
             "SMTP_USERNAME": "", "SMTP_PASSWORD": "",
         }):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            email_task = asyncio.create_task(_email_request(client))
            await asyncio.sleep(0.2)  # let /uc07/email actually start running

            t0 = time.monotonic()
            health_resp = await client.get("/health")
            health_elapsed = time.monotonic() - t0

            email_resp = await email_task
    return email_resp, health_resp, health_elapsed


def test_slow_smtp_send_does_not_block_health():
    email_resp, health_resp, elapsed = asyncio.run(_run_concurrency_check())
    assert email_resp.status_code == 200
    assert email_resp.json()["sent"] is True
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"
    assert elapsed < 1.0, (
        f"/health took {elapsed:.2f}s while a {_SLOW_SEND_SECONDS}s SMTP send was in flight "
        "(should be near-instant if /uc07/email isn't blocking the event loop)"
    )
