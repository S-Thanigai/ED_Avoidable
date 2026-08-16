"""
Phase 8C Part 17 -- FRONTEND architecture tests (subset of the numbered
28-36 spec items that are best verified structurally, by inspecting the
actual frontend source, rather than via a rendered-component test).
Mirrors backend/tests/test_phase8b_safety_context.py's
test_19_frontend_has_no_independent_safety_decision_logic in style and
intent, extended to Phase 8C's new explainability/GenAI surface.

Component-level rendering/loading/fallback/source-label behavior (also
part of 28-36) is covered separately by
frontend/src/uc07/__tests__/WhyFlaggedSection.test.tsx and
AiExplanationSection.test.tsx.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def _ts_files():
    for path in FRONTEND_SRC.rglob("*.ts*"):
        if "__tests__" in path.parts or path.name == "types.ts":
            continue
        yield path


def _non_comment_lines(path: Path) -> list[str]:
    """Strips single-line (//) and block (/* ... */) comments so a
    grep-style check only sees actual code -- these components'
    docstring-style comments legitimately DISCUSS the very tokens
    ("thinking", "Ollama") the checks below forbid in executable code,
    while explaining why the code never uses them."""
    text = path.read_text(encoding="utf-8")
    without_blocks = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = []
    for line in without_blocks.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        code_part = line.split("//", 1)[0]
        lines.append(code_part)
    return lines


# ---- no frontend decision logic: explanation_method / explanation_source ----

def test_frontend_never_assigns_an_explanation_method_or_source_literal():
    """The frontend must never itself compute/assign SHAP_LINEAR,
    LINEAR_CONTRIBUTION, GENAI, or DETERMINISTIC_FALLBACK -- those values
    only ever come from a backend response, exactly like the existing
    CLEAR/CAUTION/OVERRIDE check this test mirrors."""
    assignment_re = re.compile(
        r'(?<![=!<>])=(?!=)\s*["\'](SHAP_LINEAR|LINEAR_CONTRIBUTION|GENAI|DETERMINISTIC_FALLBACK|'
        r'INCREASES_RISK|DECREASES_RISK)["\']'
    )
    offenders = []
    for path in _ts_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*") or "|" in stripped or "type " in stripped:
                continue
            if assignment_re.search(line):
                offenders.append((str(path), lineno, line.strip()))
    assert offenders == [], f"frontend source assigns an explanation-method/source literal directly: {offenders}"


# ---- no hidden reasoning/thinking is ever referenced by the frontend ----

def test_frontend_never_references_a_thinking_or_reasoning_field():
    """The backend's MemberExplanationResponse (types.ts) has exactly 8
    fields, none of which is a "thinking"/"reasoning"/"chain_of_thought"
    field -- there is nothing for the frontend to accidentally render.
    This asserts the frontend source doesn't reference any such field
    name at all, which would indicate an attempt to surface internal
    model reasoning."""
    forbidden_tokens = ("thinking", "chain_of_thought", "reasoning_trace", "internal_reasoning")
    offenders = []
    for path in _ts_files():
        code = "\n".join(_non_comment_lines(path)).lower()
        for token in forbidden_tokens:
            if token in code:
                offenders.append((str(path), token))
    assert offenders == [], f"frontend source references a hidden-reasoning field: {offenders}"


# ---- Frontend -> FastAPI -> Explanation Agent -> Ollama, never Frontend -> Ollama ----

def test_frontend_never_calls_ollama_directly():
    """Phase 8C Part 16: the frontend must call this app's own FastAPI
    backend only -- it must never hold a direct URL/port to Ollama."""
    offenders = []
    for path in _ts_files():
        code = "\n".join(_non_comment_lines(path)).lower()
        if "11434" in code or "ollama" in code:
            offenders.append(str(path))
    assert offenders == [], f"frontend source references Ollama directly: {offenders}"


def test_explain_member_calls_the_backend_explain_endpoint_only():
    api_ts = (FRONTEND_SRC / "uc07" / "api.ts").read_text(encoding="utf-8")
    assert '"/uc07/explain"' in api_ts
    # the request function must be built from an already-fetched decision,
    # not from raw member data / CSV content
    assert "buildExplainRequest" in api_ts


# ---- lazy/on-demand generation: no population-wide GenAI trigger ----

def test_decide_uc07_batch_path_never_calls_explain_member():
    """Phase 8C Part 14: population-wide decide must never trigger a
    per-member GenAI explanation call itself -- explainMember is only
    ever invoked from the single-member drawer (AiExplanationSection),
    never from Uc07View's batch decide flow."""
    uc07_view = (FRONTEND_SRC / "uc07" / "Uc07View.tsx")
    if uc07_view.exists():
        text = uc07_view.read_text(encoding="utf-8")
        assert "explainMember" not in text
