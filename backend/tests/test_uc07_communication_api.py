"""
Phase 9 -- HTTP-level tests for POST /uc07/report and POST /uc07/email
(backend/main.py). Complements test_report_service.py (pure PDF
rendering) and test_email_service.py (pure email sending, SMTP always
mocked) with the actual FastAPI request/response contract.
"""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
from services.email_service import EmailSendResult  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def _report_request(**overrides) -> dict:
    base = {
        "member": {
            "member_id": "M00042", "name": "Taylor Smith",
            "email": "taylor.smith@example.com", "age": 57, "gender": "F",
        },
        "risk": {
            "probability": 0.317, "tier": "MODERATE", "model_version": "uc07-risk-synthetic-v1",
            "factors": [
                {"display_name": "Prior ED visits (90d)", "direction": "INCREASES_RISK"},
                {"display_name": "Telehealth available", "direction": "DECREASES_RISK"},
            ],
        },
        "navigation": {"destination": "TELEHEALTH", "reason_codes": ["TELEHEALTH_AVAILABLE"]},
        "safety": {
            "state": "CLEAR", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED",
            "message": "No configured safety override was triggered for this encounter.",
        },
        "synthetic_model": True,
        "dataset_id": "synthetic_uc07_v1",
    }
    base.update(overrides)
    return base


def _pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---- POST /uc07/report ----

def test_report_endpoint_returns_pdf(client):
    resp = client.post("/uc07/report", json=_report_request())
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:5] == b"%PDF-"
    assert "Member_Care_Navigation_Report_M00042.pdf" in resp.headers["content-disposition"]


def test_report_endpoint_rejects_invalid_input(client):
    bad = _report_request()
    bad["risk"]["tier"] = "NOT_A_TIER"
    resp = client.post("/uc07/report", json=bad)
    assert resp.status_code == 422


def test_report_endpoint_rejects_missing_member(client):
    bad = _report_request()
    del bad["member"]
    resp = client.post("/uc07/report", json=bad)
    assert resp.status_code == 422


def test_report_endpoint_uses_deterministic_fallback_when_no_explanation_given(client):
    """No `explanation` field supplied -> report_service still gets a
    full deterministic explanation, and NO network/LLM call is made
    (Groq's HTTP call is monkeypatched to fail loudly if invoked)."""
    with patch("genai_explanation._call_groq", side_effect=AssertionError("must not call Groq for PDF generation")):
        with patch("genai_explanation._call_ollama", side_effect=AssertionError("must not call Ollama for PDF generation")):
            resp = client.post("/uc07/report", json=_report_request())
    assert resp.status_code == 200
    text = _pdf_text(resp.content)
    assert "G. Explanation" in text
    assert "Deterministic template explanation" in text


def test_report_endpoint_reuses_provided_explanation_verbatim(client):
    req = _report_request(explanation={
        "summary": "A distinctive pre-approved AI summary sentence for this test.",
        "risk_explanation": "Distinctive risk sentence.",
        "navigation_explanation": "Distinctive navigation sentence.",
        "safety_explanation": "Distinctive safety sentence.",
        "disclaimer": "Distinctive disclaimer sentence. Call 911 in an emergency.",
        "explanation_source": "GENAI",
        "model_used": "openai/gpt-oss-120b",
    })
    resp = client.post("/uc07/report", json=req)
    assert resp.status_code == 200
    text = _pdf_text(resp.content)
    assert "A distinctive pre-approved AI summary sentence for this test." in text
    assert "AI-generated explanation (model: openai/gpt-oss-120b)" in text


def test_report_endpoint_synthetic_disclosure_shown(client):
    resp = client.post("/uc07/report", json=_report_request(synthetic_model=True))
    text = _pdf_text(resp.content)
    assert "not clinically validated" in text.lower()


# ---- POST /uc07/email ----

def test_email_endpoint_disabled_by_default(client):
    resp = client.post("/uc07/email", json={
        "report": _report_request(),
        "to_email": "taylor.smith@example.com",
        "subject": "Your Care Navigation Summary",
        "body": "Hello Taylor, please see the attached report.",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["sent"] is False
    assert body["error_code"] == "EMAIL_DISABLED"
    assert "password" not in str(body).lower()


def test_email_endpoint_rejects_invalid_recipient_before_sending(client, monkeypatch):
    monkeypatch.setenv("EMAIL_ENABLED", "true")
    monkeypatch.setenv("SMTP_FROM_EMAIL", "care@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.invalid")
    with patch("services.email_service.smtplib.SMTP") as smtp_cls:
        resp = client.post("/uc07/email", json={
            "report": _report_request(),
            "to_email": "not-an-email",
            "subject": "Subject",
            "body": "Body",
        })
    smtp_cls.assert_not_called()  # rejected before any network attempt
    assert resp.status_code == 200
    body = resp.json()
    assert body["sent"] is False
    assert body["error_code"] == "INVALID_INPUT"
    assert "Traceback" not in str(body)


def test_email_endpoint_success_attaches_same_pdf_as_report_endpoint(client):
    report_req = _report_request()
    report_resp = client.post("/uc07/report", json=report_req)
    assert report_resp.status_code == 200

    captured: dict = {}

    def _fake_send_report_email(self, **kwargs):
        captured.update(kwargs)
        return EmailSendResult(sent=True, provider="smtp", message="Report sent to t***@example.com.")

    with patch("main.EmailService.send_report_email", _fake_send_report_email):
        email_resp = client.post("/uc07/email", json={
            "report": report_req,
            "to_email": "taylor.smith@example.com",
            "subject": "Your Care Navigation Summary",
            "body": "Hello Taylor, please see the attached report.",
        })

    assert email_resp.status_code == 200
    body = email_resp.json()
    assert body["sent"] is True
    assert body["error_code"] is None
    assert "password" not in str(body).lower()

    assert captured["attachment_filename"] == "Member_Care_Navigation_Report_M00042.pdf"
    assert captured["attachment_bytes"][:5] == b"%PDF-"

    # Same CONTENT (report_id/timestamp naturally differ between the two
    # independent calls -- Section 15 requires the same rendering
    # pipeline/content, not byte-identical output across two requests).
    downloaded_text = _pdf_text(report_resp.content)
    emailed_text = _pdf_text(captured["attachment_bytes"])
    strip = lambda t: "\n".join(
        line for line in t.splitlines()
        if "Report ID" not in line and "Generated" not in line
        and "UC07-RPT-" not in line and "UTC" not in line
    )
    assert strip(downloaded_text) == strip(emailed_text)


def test_email_endpoint_editable_subject_and_body_reach_service(client):
    captured: dict = {}

    def _fake_send_report_email(self, **kwargs):
        captured.update(kwargs)
        return EmailSendResult(sent=True, provider="smtp", message="sent")

    with patch("main.EmailService.send_report_email", _fake_send_report_email):
        resp = client.post("/uc07/email", json={
            "report": _report_request(),
            "to_email": "taylor.smith@example.com",
            "subject": "A care-manager-edited subject",
            "body": "A care-manager-edited message body.",
        })
    assert resp.status_code == 200
    assert captured["subject"] == "A care-manager-edited subject"
    assert captured["body"] == "A care-manager-edited message body."


def test_email_endpoint_never_returns_smtp_password(client):
    resp = client.post("/uc07/email", json={
        "report": _report_request(), "to_email": "taylor.smith@example.com",
        "subject": "Subject", "body": "Body",
    })
    assert "SMTP_PASSWORD" not in resp.text
    assert "smtp_password" not in resp.text.lower()


def test_email_endpoint_never_changes_decision_values(client):
    """This endpoint has no access to the decision agents at all -- the
    response body never carries a risk/navigation/safety field, and
    calling it cannot alter a FinalUC07Decision anywhere else."""
    resp = client.post("/uc07/email", json={
        "report": _report_request(), "to_email": "taylor.smith@example.com",
        "subject": "Subject", "body": "Body",
    })
    body = resp.json()
    assert set(body.keys()) == {"sent", "provider", "message", "error_code", "report_id"}


# ---- GET /health exposes safe email configuration status ----

def test_health_reports_email_configuration_without_leaking_credentials(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "email_configured" in body
    assert "email_provider" in body
    assert body["email_provider"] == "smtp"
    assert isinstance(body["email_configured"], bool)
    assert "password" not in str(body).lower()


# ---- 23. OVERRIDE safety wording preserved through the email path ----

def test_23_override_email_attachment_keeps_safety_priority_wording(client):
    override_req = _report_request(
        navigation={"destination": None, "reason_codes": []},
        safety={
            "state": "OVERRIDE", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED",
            "message": "Emergency care should not be delayed when emergency symptoms or high-acuity conditions are present.",
        },
    )
    captured: dict = {}

    def _fake_send_report_email(self, **kwargs):
        captured.update(kwargs)
        return EmailSendResult(sent=True, provider="smtp", message="sent")

    with patch("main.EmailService.send_report_email", _fake_send_report_email):
        resp = client.post("/uc07/email", json={
            "report": override_req, "to_email": "taylor.smith@example.com",
            "subject": "Your Care Navigation Summary", "body": "Hello, please see the attached report.",
        })
    assert resp.status_code == 200
    assert resp.json()["sent"] is True

    text = _pdf_text(captured["attachment_bytes"])
    safety_idx = text.index("F. Current Safety Status")
    risk_idx = text.index("C. 90-Day Risk Assessment")
    assert safety_idx < risk_idx, "OVERRIDE must keep Safety ahead of Risk in the EMAILED report too"
    flat = " ".join(text.split())
    assert "must not delay appropriate emergency evaluation" in flat


# ---- 24. /uc07/decide is completely unaffected by this endpoint's existence ----

def test_24_uc07_decide_route_and_behavior_unaffected(client):
    """Not a full re-test of /uc07/decide (see test_uc07_api.py for
    that) -- just confirms adding /uc07/report and /uc07/email did not
    remove, shadow, or otherwise disturb the existing route."""
    routes = {getattr(r, "path", None) for r in main.app.routes}
    assert "/uc07/decide" in routes
    assert "/uc07/explain" in routes
    resp = client.post("/uc07/decide", data={})
    # No files supplied -> a clean 422 (FastAPI's own validation), never
    # a 404/500 -- proves the route is still wired up and behaving as
    # before this phase.
    assert resp.status_code == 422


# ---- friendly, non-leaking error messages surfaced end-to-end ----

def test_email_endpoint_error_message_never_leaks_smtp_internals(client):
    def _fake_send_report_email(self, **kwargs):
        from services.email_service import EmailSendResult
        return EmailSendResult(
            sent=False, provider="smtp",
            message="The email provider rejected the configured credentials.",
            error_code="AUTH_FAILED",
        )

    with patch("main.EmailService.send_report_email", _fake_send_report_email):
        resp = client.post("/uc07/email", json={
            "report": _report_request(), "to_email": "taylor.smith@example.com",
            "subject": "Subject", "body": "Body",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["error_code"] == "AUTH_FAILED"
    for leak in ("Traceback", "smtplib", "SMTPAuthenticationError", "535"):
        assert leak not in resp.text
