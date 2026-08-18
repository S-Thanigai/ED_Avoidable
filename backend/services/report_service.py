"""
report_service.py
------------------
Renders the "Member Care Navigation & Risk Summary" PDF -- a care-
management decision-support report, NOT a clinical diagnosis document.

Responsibilities (and ONLY these):
    - accept a plain, structured dict describing an ALREADY-COMPUTED
      decision (risk/navigation/safety) plus an ALREADY-APPROVED
      explanation (GenAI-validated or deterministic-fallback -- see
      backend/agents/genai_explanation.py) and member contact/profile
      fields
    - render that data into PDF bytes, deterministically with respect
      to its input (the only per-call variation is the generated
      timestamp/report_id, which is expected -- see module docstring
      of backend/services/__init__.py)
    - return (pdf_bytes, filename)

This module performs NO model inference, NO SHAP computation, NO risk/
navigation/safety decision logic, and NO email sending. It never
imports risk_detection.py, care_navigation.py, safety_policy.decide(),
model_explainability.py, or orchestrator.py. It is purely a renderer
over data the caller (backend/main.py's POST /uc07/report and
POST /uc07/email handlers) already assembled from a FinalUC07Decision
and a MemberExplanation the frontend already has.

POST /uc07/report and POST /uc07/email both call `generate_report_pdf`
with the SAME payload-building logic (see main.py's
`_build_report_payload`) so the downloaded report and the emailed
attachment are always the same document for the same decision -- there
is exactly one PDF-rendering code path in this project, not two
independently maintained ones.
"""
from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPORT_TITLE = "MEMBER CARE NAVIGATION & RISK SUMMARY"

IMPORTANT_SAFETY_NOTICE = (
    "This report supports care-management navigation and does not replace clinical "
    "evaluation. Emergency care should not be delayed when emergency symptoms or "
    "high-acuity conditions are present."
)

MODEL_DISCLOSURE_SYNTHETIC = (
    "Demo model trained on synthetic data; not clinically validated."
)

# Phase 9 -- mirrors genai_explanation.py's _SAFETY_STATE_TEMPLATES wording
# exactly in meaning (never "patient is safe" / "no emergency exists" for
# CLEAR; never implies confirmed safety for CAUTION/OVERRIDE).
SAFETY_STATE_DEFINITIONS = {
    "CLEAR": "Complete supplied safety context did not trigger configured high-acuity rules.",
    "CAUTION": "Current safety information is missing or incomplete.",
    "OVERRIDE": "Current information triggered a configured high-acuity/emergency safety rule.",
}

OVERRIDE_NAVIGATION_NOTICE = (
    "Current supplied information triggered a safety override. Proactive navigation "
    "information must not delay appropriate emergency evaluation."
)

FACTOR_ATTRIBUTION_NOTE = "These are model-attribution signals and do not establish causation."

RISK_WORDING = "Predicted likelihood of potentially avoidable ED utilization within the next 90 days"

_DESTINATION_LABELS = {
    "PRIMARY_CARE": "Primary Care",
    "URGENT_CARE": "Urgent Care",
    "TELEHEALTH": "Telehealth",
    "CARE_MANAGEMENT": "Care Management",
    "NO_PROACTIVE_NAVIGATION": "No Proactive Navigation",
}

_TIER_COLORS = {
    "LOW": (colors.HexColor("#1a7f37"), colors.HexColor("#e6f4ea")),
    "MODERATE": (colors.HexColor("#9a6700"), colors.HexColor("#fff6df")),
    "HIGH": (colors.HexColor("#b42318"), colors.HexColor("#fdecea")),
}
_SAFETY_COLORS = {
    "CLEAR": (colors.HexColor("#1a7f37"), colors.HexColor("#e6f4ea")),
    "CAUTION": (colors.HexColor("#9a6700"), colors.HexColor("#fff6df")),
    "OVERRIDE": (colors.HexColor("#b42318"), colors.HexColor("#fdecea")),
}
_EXPLAIN_ACCENT = colors.HexColor("#5925dc")
_EXPLAIN_BG = colors.HexColor("#f3f0fe")
_HEADER_NAVY = colors.HexColor("#1f2a44")
_MUTED_GREY = colors.HexColor("#6b7280")
_BORDER_GREY = colors.HexColor("#d9dde3")


def _title(token: str) -> str:
    return token.replace("_", " ").title()


@dataclass(frozen=True)
class ReportFactor:
    display_name: str
    direction: str  # "INCREASES_RISK" | "DECREASES_RISK"


@dataclass(frozen=True)
class ReportContext:
    """Everything `generate_report_pdf` needs, already flattened out of
    whatever request schema the caller used. Deliberately a plain
    dataclass (not a Pydantic/FastAPI model) so this module has zero
    dependency on the web framework and can be unit-tested directly."""

    member_id: str
    risk_probability: float
    risk_tier: str
    model_version: str
    synthetic_model: bool
    safety_state: str
    safety_message: str
    context_completeness: str
    context_source: str
    explanation_summary: str
    explanation_risk: str
    explanation_navigation: str
    explanation_safety: str
    explanation_disclaimer: str
    explanation_source: str
    member_name: str | None = None
    member_email: str | None = None
    member_age: int | None = None
    member_gender: str | None = None
    dataset_id: str | None = None
    explanation_model_used: str | None = None
    navigation_destination: str | None = None
    navigation_reason_codes: list[str] = field(default_factory=list)
    factors: list[ReportFactor] = field(default_factory=list)
    report_id: str | None = None
    generated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.report_id is None:
            object.__setattr__(self, "report_id", f"UC07-RPT-{uuid.uuid4().hex[:12].upper()}")
        if self.generated_at is None:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc))


def build_report_filename(member_id: str) -> str:
    """`Member_Care_Navigation_Report_<member_id>.pdf`, with member_id
    sanitized to filesystem/header-safe characters only -- an untrusted
    member_id must never be able to inject a path separator or break the
    Content-Disposition header."""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", member_id) or "member"
    return f"Member_Care_Navigation_Report_{safe_id}.pdf"


def _styles() -> dict:
    base = getSampleStyleSheet()
    styles = dict(base.byName)
    styles["ReportTitle"] = ParagraphStyle(
        "ReportTitle", parent=base["Title"], fontSize=16, leading=20,
        textColor=_HEADER_NAVY, alignment=TA_LEFT, spaceAfter=2,
    )
    styles["SectionHeading"] = ParagraphStyle(
        "SectionHeading", parent=base["Heading2"], fontSize=12.5, leading=16,
        textColor=_HEADER_NAVY, spaceBefore=14, spaceAfter=6,
    )
    styles["SubHeading"] = ParagraphStyle(
        "SubHeading", parent=base["Heading3"], fontSize=10.5, leading=13,
        textColor=_HEADER_NAVY, spaceBefore=6, spaceAfter=4,
    )
    styles["Body"] = ParagraphStyle(
        "Body", parent=base["BodyText"], fontSize=9.5, leading=13.5, textColor=colors.black,
    )
    styles["Muted"] = ParagraphStyle(
        "Muted", parent=base["BodyText"], fontSize=8.5, leading=12, textColor=_MUTED_GREY,
    )
    styles["Notice"] = ParagraphStyle(
        "Notice", parent=base["BodyText"], fontSize=9.5, leading=13.5,
        textColor=colors.black, spaceBefore=4, spaceAfter=4,
    )
    styles["MetaLabel"] = ParagraphStyle(
        "MetaLabel", parent=base["BodyText"], fontSize=8, leading=10, textColor=_MUTED_GREY,
    )
    return styles


def _badge_table(text: str, fg: colors.Color, bg: colors.Color, width: float = 60 * mm) -> Table:
    t = Table([[text]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("TEXTCOLOR", (0, 0), (-1, -1), fg),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, fg),
    ]))
    return t


def _kv_table(rows: list[tuple[str, str]], styles: dict) -> Table:
    data = [
        [Paragraph(f"<b>{label}</b>", styles["Muted"]), Paragraph(str(value) if value else "—", styles["Body"])]
        for label, value in rows
    ]
    t = Table(data, colWidths=[42 * mm, 108 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
    ]))
    return t


def _factors_table(factors: list[ReportFactor], styles: dict) -> Table | None:
    increasing = [f for f in factors if f.direction == "INCREASES_RISK"]
    decreasing = [f for f in factors if f.direction == "DECREASES_RISK"]
    if not increasing and not decreasing:
        return None

    rows = [[
        Paragraph("<b>Factors increasing estimate</b>", styles["Body"]),
        Paragraph("<b>Factors decreasing estimate</b>", styles["Body"]),
    ]]
    max_len = max(len(increasing), len(decreasing), 1)
    for i in range(max_len):
        inc = increasing[i].display_name if i < len(increasing) else ""
        dec = decreasing[i].display_name if i < len(decreasing) else ""
        rows.append([
            Paragraph(f"(+) {inc}" if inc else "—", styles["Body"]),
            Paragraph(f"(-) {dec}" if dec else "—", styles["Body"]),
        ])
    t = Table(rows, colWidths=[75 * mm, 75 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _EXPLAIN_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, _EXPLAIN_ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, _BORDER_GREY),
    ]))
    return t


def _risk_flowables(ctx: ReportContext, styles: dict, muted: bool) -> list:
    fg, bg = _TIER_COLORS.get(ctx.risk_tier, (_MUTED_GREY, colors.whitesmoke))
    heading = "C. 90-Day Risk Assessment" + (" (reference only — see Safety, above)" if muted else "")
    flow: list = [Paragraph(heading, styles["SubHeading" if muted else "SectionHeading"])]
    flow.append(Paragraph(RISK_WORDING + ":", styles["Muted" if muted else "Body"]))
    flow.append(Spacer(1, 4))

    row = Table(
        [[
            _badge_table(ctx.risk_tier, fg, bg, width=45 * mm),
            Paragraph(f"Predicted probability: <b>{ctx.risk_probability:.1%}</b>", styles["Muted" if muted else "Body"]),
        ]],
        colWidths=[48 * mm, 102 * mm],
    )
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    flow.append(row)
    flow.append(Spacer(1, 6))
    return flow


def _navigation_flowables(ctx: ReportContext, styles: dict, is_override: bool) -> list:
    flow: list = [Paragraph("E. Care Navigation", styles["SectionHeading"])]
    if is_override:
        flow.append(Paragraph(f"<b>{OVERRIDE_NAVIGATION_NOTICE}</b>", styles["Notice"]))
        flow.append(Spacer(1, 4))

    dest_label = _DESTINATION_LABELS.get(ctx.navigation_destination or "", ctx.navigation_destination or "None")
    flow.append(_kv_table([("Selected destination", dest_label)], styles))
    if ctx.navigation_reason_codes:
        reasons = ", ".join(_title(rc) for rc in ctx.navigation_reason_codes)
        flow.append(Spacer(1, 3))
        flow.append(Paragraph(f"<b>Reason codes:</b> {reasons}", styles["Body"]))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(ctx.explanation_navigation, styles["Body"]))
    flow.append(Paragraph(
        "This is a non-emergency, proactive care-navigation suggestion only.", styles["Muted"],
    ))
    return flow


def _safety_flowables(ctx: ReportContext, styles: dict, prominent: bool) -> list:
    fg, bg = _SAFETY_COLORS.get(ctx.safety_state, (_MUTED_GREY, colors.whitesmoke))
    heading = "F. Current Safety Status" + ("  — PRIORITY" if prominent else "")
    flow: list = [Paragraph(heading, styles["SectionHeading"])]

    row = Table(
        [[
            _badge_table(ctx.safety_state, fg, bg, width=45 * mm),
            Paragraph(ctx.safety_message, styles["Body"]),
        ]],
        colWidths=[48 * mm, 102 * mm],
    )
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    flow.append(row)
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(SAFETY_STATE_DEFINITIONS.get(ctx.safety_state, ""), styles["Muted"]))
    flow.append(Spacer(1, 3))
    flow.append(_kv_table([
        ("Context completeness", _title(ctx.context_completeness)),
        ("Context source", _title(ctx.context_source)),
    ], styles))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(ctx.explanation_safety, styles["Body"]))
    return flow


def _header_footer(canvas_obj, doc, ctx: ReportContext) -> None:
    canvas_obj.saveState()
    page_w, page_h = A4

    canvas_obj.setFillColor(_HEADER_NAVY)
    canvas_obj.rect(0, page_h - 14 * mm, page_w, 14 * mm, stroke=0, fill=1)
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 11)
    canvas_obj.drawString(15 * mm, page_h - 9.5 * mm, REPORT_TITLE)
    canvas_obj.setFont("Helvetica", 7.5)
    canvas_obj.drawRightString(
        page_w - 15 * mm, page_h - 9.5 * mm,
        f"Report ID: {ctx.report_id}",
    )

    canvas_obj.setFillColor(_MUTED_GREY)
    canvas_obj.setFont("Helvetica", 7.5)
    canvas_obj.drawString(
        15 * mm, 10 * mm,
        f"Generated {ctx.generated_at.strftime('%Y-%m-%d %H:%M UTC')} · Model {ctx.model_version} "
        f"{'· SYNTHETIC / DEMO DATA' if ctx.synthetic_model else ''}",
    )
    canvas_obj.drawRightString(page_w - 15 * mm, 10 * mm, f"Page {doc.page}")
    canvas_obj.setStrokeColor(_BORDER_GREY)
    canvas_obj.line(15 * mm, 13 * mm, page_w - 15 * mm, 13 * mm)
    canvas_obj.restoreState()


def generate_report_pdf(ctx: ReportContext) -> bytes:
    """Renders `ctx` into PDF bytes. Never raises for well-formed input;
    callers are responsible for validating the input BEFORE calling this
    (see main.py's Pydantic ReportRequest) -- this function assumes a
    valid ReportContext, matching the rest of this codebase's convention
    of validating at the API boundary, not deep in a rendering helper."""
    styles = _styles()
    is_override = ctx.safety_state == "OVERRIDE"
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=20 * mm, bottomMargin=18 * mm, leftMargin=15 * mm, rightMargin=15 * mm,
        title=REPORT_TITLE, author="UC07 Care Management",
    )

    story: list = []

    # A. Report metadata
    story.append(Paragraph(REPORT_TITLE, styles["ReportTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_HEADER_NAVY, spaceAfter=6))
    story.append(_kv_table([
        ("Report ID", ctx.report_id),
        ("Generated", ctx.generated_at.strftime("%Y-%m-%d %H:%M UTC")),
        ("Model version", ctx.model_version),
        ("Dataset", ctx.dataset_id or "—"),
        ("Data disclosure", "Synthetic / demonstration data — not real patient data" if ctx.synthetic_model else "—"),
    ], styles))
    story.append(Spacer(1, 6))

    # B. Member information
    story.append(Paragraph("B. Member Information", styles["SectionHeading"]))
    story.append(_kv_table([
        ("Member ID", ctx.member_id),
        ("Name", ctx.member_name or "—"),
        ("Email", ctx.member_email or "—"),
        ("Age", str(ctx.member_age) if ctx.member_age is not None else "—"),
        ("Gender", ctx.member_gender or "—"),
    ], styles))

    if is_override:
        # Safety-first ordering: the Safety section moves ahead of the
        # risk score and is visually flagged as priority (Part 5).
        story.append(KeepTogether(_safety_flowables(ctx, styles, prominent=True)))
        story.append(Spacer(1, 4))
        story.extend(_risk_flowables(ctx, styles, muted=True))
    else:
        story.extend(_risk_flowables(ctx, styles, muted=False))

    # D. Key model factors
    factors_table = _factors_table(ctx.factors, styles)
    story.append(Paragraph("D. Key Model Factors", styles["SectionHeading"]))
    if factors_table is not None:
        story.append(factors_table)
    else:
        story.append(Paragraph("No individual model-attribution factors were available.", styles["Body"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(FACTOR_ATTRIBUTION_NOTE, styles["Muted"]))
    story.append(Spacer(1, 4))

    # E. Care navigation
    story.extend(_navigation_flowables(ctx, styles, is_override))
    story.append(Spacer(1, 4))

    if not is_override:
        story.extend(_safety_flowables(ctx, styles, prominent=False))
        story.append(Spacer(1, 4))

    # G. Explanation
    story.append(Paragraph("G. Explanation", styles["SectionHeading"]))
    explain_box = Table(
        [[Paragraph(ctx.explanation_summary, styles["Body"])]],
        colWidths=[150 * mm],
    )
    explain_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _EXPLAIN_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, _EXPLAIN_ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(explain_box)
    story.append(Spacer(1, 4))
    story.append(Paragraph(ctx.explanation_risk, styles["Body"]))
    source_note = (
        f"AI-generated explanation (model: {ctx.explanation_model_used})"
        if ctx.explanation_source == "GENAI" and ctx.explanation_model_used
        else "Deterministic template explanation"
    )
    story.append(Paragraph(source_note, styles["Muted"]))
    story.append(Spacer(1, 6))

    # H. Important safety notice
    story.append(Paragraph("H. Important Safety Notice", styles["SectionHeading"]))
    notice_box = Table([[Paragraph(f"<b>{IMPORTANT_SAFETY_NOTICE}</b>", styles["Notice"])]], colWidths=[150 * mm])
    notice_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _SAFETY_COLORS["OVERRIDE"][1]),
        ("BOX", (0, 0), (-1, -1), 0.75, _SAFETY_COLORS["OVERRIDE"][0]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(notice_box)
    story.append(Spacer(1, 4))
    story.append(Paragraph(ctx.explanation_disclaimer, styles["Muted"]))

    # I. Model disclosure
    if ctx.synthetic_model:
        story.append(Spacer(1, 6))
        story.append(Paragraph("I. Model Disclosure", styles["SectionHeading"]))
        story.append(Paragraph(MODEL_DISCLOSURE_SYNTHETIC, styles["Muted"]))

    def _on_page(canvas_obj, doc_obj):
        _header_footer(canvas_obj, doc_obj, ctx)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buffer.getvalue()
