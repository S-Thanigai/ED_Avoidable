"""
Phase 9 -- backend/services/report_service.py unit tests.

Covers Section 28's PDF checklist directly against `generate_report_pdf`
(no HTTP layer involved here -- see test_uc07_communication_api.py for
the endpoint-level equivalents). Uses pypdf to extract real text from
the rendered PDF bytes rather than grepping the raw bytes, so these
tests are robust to ReportLab's internal stream encoding/compression.
"""
from pathlib import Path
from datetime import datetime, timezone

import pytest
from pypdf import PdfReader
import io

import sys
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BACKEND_DIR / "services") not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR / "services"))

from services import report_service  # noqa: E402
from services.report_service import ReportContext, ReportFactor  # noqa: E402


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _flat(text: str) -> str:
    """Collapses all whitespace (including pypdf's line-wrap newlines,
    which do not always correspond to a real paragraph break) to single
    spaces -- for asserting a long PHRASE is present regardless of where
    ReportLab happened to wrap it onto a new line."""
    return " ".join(text.split())


def _base_ctx(**overrides) -> ReportContext:
    defaults = dict(
        member_id="M00042",
        member_name="Taylor Smith",
        member_email="taylor.smith@example.com",
        member_age=57,
        member_gender="F",
        risk_probability=0.317,
        risk_tier="MODERATE",
        model_version="uc07-risk-synthetic-v1",
        synthetic_model=True,
        dataset_id="synthetic_uc07_v1",
        navigation_destination="TELEHEALTH",
        navigation_reason_codes=["TELEHEALTH_AVAILABLE"],
        factors=[
            ReportFactor(display_name="Prior ED visits (90d)", direction="INCREASES_RISK"),
            ReportFactor(display_name="Telehealth available", direction="DECREASES_RISK"),
        ],
        safety_state="CLEAR",
        safety_message="No configured safety override was triggered for this encounter.",
        context_completeness="COMPLETE",
        context_source="CALLER_SUPPLIED",
        explanation_summary="Moderate modeled risk tier; suggested navigation: Telehealth; safety state: Clear.",
        explanation_risk="This member's modeled risk tier is MODERATE.",
        explanation_navigation="The suggested navigation destination is Telehealth.",
        explanation_safety="Complete current safety context was supplied and no override was triggered.",
        explanation_disclaimer=(
            "For care navigation only -- never a reason to delay care. If you or a member may be "
            "experiencing a medical emergency, call 911 or go to the nearest emergency department immediately."
        ),
        explanation_source="DETERMINISTIC_FALLBACK",
        explanation_model_used=None,
    )
    defaults.update(overrides)
    return ReportContext(**defaults)


# ---- 1-3. generated successfully / valid PDF / signature ----

def test_1_pdf_generated_successfully():
    pdf_bytes = report_service.generate_report_pdf(_base_ctx())
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000


def test_2_3_starts_with_pdf_signature():
    pdf_bytes = report_service.generate_report_pdf(_base_ctx())
    assert pdf_bytes[:5] == b"%PDF-"


# ---- 4-9. required fields present ----

def test_4_member_id_present():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(member_id="M00042")))
    assert "M00042" in text


def test_5_risk_tier_present():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(risk_tier="HIGH")))
    assert "HIGH" in text


def test_6_probability_present():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(risk_probability=0.317)))
    assert "31.7%" in text


def test_7_navigation_present():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(navigation_destination="URGENT_CARE")))
    assert "Urgent Care" in text


def test_8_safety_state_present():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(safety_state="CAUTION")))
    assert "CAUTION" in text


def test_9_disclaimer_present():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx()))
    assert "call 911" in text.lower()


# ---- 10. synthetic disclosure ----

def test_10_synthetic_disclosure_present_when_synthetic():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(synthetic_model=True)))
    assert "synthetic" in text.lower()
    assert "not clinically validated" in text.lower()


def test_10_synthetic_disclosure_absent_when_not_synthetic():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(synthetic_model=False)))
    assert "not clinically validated" not in text.lower()


# ---- 11. SHAP/model factors included, human-readable only ----

def test_11_factors_included_human_readable():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx()))
    assert "Prior ED visits (90d)" in text
    assert "Telehealth available" in text
    # no raw internal feature slugs (e.g. the underscored model column name)
    assert "prior_ed_count_90d" not in text


def test_11_no_factors_handled_gracefully():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(factors=[])))
    assert "No individual model-attribution factors were available." in text


# ---- 12. multipage support ----

def test_12_multipage_content_works():
    many_factors = [
        ReportFactor(display_name=f"Factor increasing #{i}", direction="INCREASES_RISK") for i in range(15)
    ] + [
        ReportFactor(display_name=f"Factor decreasing #{i}", direction="DECREASES_RISK") for i in range(15)
    ]
    pdf_bytes = report_service.generate_report_pdf(_base_ctx(factors=many_factors))
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 2
    text = _pdf_text(pdf_bytes)
    assert "Factor increasing #14" in text
    assert "Factor decreasing #14" in text


# ---- 13. OVERRIDE report prioritizes safety wording ----

def test_13_override_report_prioritizes_safety_section():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(
        safety_state="OVERRIDE",
        safety_message="Emergency care should not be delayed when emergency symptoms or high-acuity conditions are present.",
        navigation_destination=None,
        navigation_reason_codes=[],
    )))
    safety_idx = text.index("F. Current Safety Status")
    risk_idx = text.index("C. 90-Day Risk Assessment")
    assert safety_idx < risk_idx, "Safety section must appear before the risk section for OVERRIDE"
    assert "reference only" in text.lower()
    assert "must not delay appropriate emergency evaluation" in _flat(text).lower()


def test_13_override_navigation_never_worded_as_emergency_alternative():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(
        safety_state="OVERRIDE",
        safety_message="Emergency care should not be delayed.",
        navigation_destination=None,
        navigation_reason_codes=[],
    )))
    assert report_service.OVERRIDE_NAVIGATION_NOTICE in _flat(text)


def test_13_non_override_report_keeps_normal_order():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx(safety_state="CLEAR")))
    safety_idx = text.index("F. Current Safety Status")
    risk_idx = text.index("C. 90-Day Risk Assessment")
    assert risk_idx < safety_idx


# ---- 14. deterministic enough for content checks ----

def test_14_deterministic_content_given_same_input():
    fixed_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ctx1 = _base_ctx(report_id="UC07-RPT-FIXED", generated_at=fixed_ts)
    ctx2 = _base_ctx(report_id="UC07-RPT-FIXED", generated_at=fixed_ts)
    text1 = _pdf_text(report_service.generate_report_pdf(ctx1))
    text2 = _pdf_text(report_service.generate_report_pdf(ctx2))
    assert text1 == text2


# ---- filename helper ----

def test_filename_uses_member_id_and_sanitizes_unsafe_characters():
    assert report_service.build_report_filename("M00042") == "Member_Care_Navigation_Report_M00042.pdf"
    unsafe = report_service.build_report_filename('M1"; rm -rf /\n')
    assert '"' not in unsafe and "\n" not in unsafe and ";" not in unsafe
    assert unsafe.startswith("Member_Care_Navigation_Report_") and unsafe.endswith(".pdf")


# ---- never a clinical/diagnosis framing ----

def test_report_title_is_care_navigation_not_diagnosis():
    text = _pdf_text(report_service.generate_report_pdf(_base_ctx()))
    assert "MEMBER CARE NAVIGATION & RISK SUMMARY" in text
    assert "diagnosis report" not in text.lower()
