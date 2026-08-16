"""
genai_explanation.py
---------------------
Phase 8C -- AGENT 4: GenAI Explanation Agent. NOT a decision-making agent.

Responsibility: translate an ALREADY-COMPUTED, already-safety-reviewed
UC07 decision summary into a short, plain-English explanation, using a
locally hosted Ollama model (qwen3:8b by default). This agent has ZERO
decision authority: it cannot change a risk probability, risk tier,
navigation destination, or safety state -- it only receives those values
after Agents 1-3 (and the Safety & Policy Agent, which is ALWAYS final)
have already produced them, and it can never feed anything back into
those agents. There is no import anywhere in this module of
risk_detection.py, care_navigation.py, safety_policy.decide(), or
orchestrator.py's decision-producing functions -- only
safety_policy.check_text()/PROHIBITED_PHRASES and BASE_DISCLAIMER are
reused, as a read-only policy/wording source, never as a way to call back
into the decision pipeline. This is enforced BY CONSTRUCTION (no such
import exists) and covered by dedicated regression tests
(tests/test_genai_explanation_authority.py) -- not merely asserted here.

Strict LLM input boundary (Phase 8C Part 5): this module NEVER receives
or has access to raw CSV data, a member's full history, or any patient
data beyond the minimal `payload` dict the caller (backend/main.py's
POST /uc07/explain) passes in -- itself limited by a Pydantic schema to
exactly: risk.{probability, tier, model_version, factors[]},
navigation.{destination, reason_codes}, safety.{state,
context_completeness, context_source}, synthetic_model. (The Phase 8C
spec's illustrative example payload also shows a `safety.reason_codes`
field; SafetyDecision -- contracts.py -- has no such field, so it is
deliberately omitted here rather than fabricated. `context_completeness`
/ `context_source`, which DO exist on SafetyDecision, are included
instead, since they let the model explain uncertainty/missing context,
one of its permitted responsibilities.)

What the model MUST NEVER do (Phase 8C Part 7) is enforced here by
APPLICATION CODE, not only by the system prompt below:
    - a JSON schema (`format`) forces structured output; malformed JSON
      or a missing required key is rejected outright (_validate_genai_output)
    - safety_policy.check_text() (the SAME centralized prohibited-phrase
      list the Safety & Policy Agent itself enforces) is run against
      every generated sentence
    - an additional GenAI-specific prohibited-content check
      (_genai_prohibited_violation) blocks diagnosis/medication/dosage/
      causal-claim language
    - a tier/destination/safety-state CONSISTENCY check
      (_tier_consistency_violation / _destination_consistency_violation /
      _safety_state_consistency_violation) rejects output that names a
      DIFFERENT risk tier, navigation destination, or safety state than
      the one actually supplied in `payload`
    - the returned `disclaimer` field is NEVER the model's own text --
      it is always overwritten with safety_policy.BASE_DISCLAIMER, so the
      model cannot weaken or omit the mandatory disclaimer
    - "think": false suppresses qwen3's chain-of-thought in the raw
      Ollama response, and only the four structured content fields are
      ever read out of that response -- no internal reasoning field is
      ever surfaced to a caller
Any rejection at any of these checks (or any network/timeout/HTTP/parse
failure at all) falls through to `_deterministic_explanation`, which
never raises and never depends on Ollama being installed, running, or
reachable. `generate_explanation()` itself never raises for any reason --
GenAI failure must never break the caller (Phase 8C Part 10).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import time

import httpx

import safety_policy
from contracts import ExplanationSource, MemberExplanation

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_SECONDS = 20.0

_REQUIRED_KEYS = ("summary", "risk_explanation", "navigation_explanation", "safety_explanation")
_MAX_FIELD_CHARS = 600
_MAX_COMBINED_CHARS = 1500

_SAFETY_STATE_TEMPLATES = {
    "CLEAR": "Complete current safety context was supplied and no configured safety override was triggered.",
    "CAUTION": "Current safety information is absent or incomplete, so the system cannot confirm the safety-rule state fully.",
    "OVERRIDE": "Supplied current information triggered a configured high-acuity/emergency safety rule.",
}

_TIER_PHRASES = {
    "LOW": ("low risk",),
    "MODERATE": ("moderate risk",),
    "HIGH": ("high risk",),
}

_DESTINATION_PHRASES = {
    "PRIMARY_CARE": ("primary care",),
    "URGENT_CARE": ("urgent care",),
    "TELEHEALTH": ("telehealth",),
    "CARE_MANAGEMENT": ("care management",),
}

# A navigation ReasonCode legitimately mentions a destination-adjacent
# concept as CONTEXT (e.g. "URGENT_CARE_ACCESS_ADVANTAGE" is a reason a
# member might be navigated to CARE_MANAGEMENT, not a claim the
# destination IS Urgent Care) -- a faithful paraphrase of a reason code
# that is actually present in payload.navigation.reason_codes must not be
# treated as an unsupported destination claim.
_REASON_CODE_JUSTIFIED_DESTINATION_PHRASES: dict[str, tuple[str, ...]] = {
    "LIMITED_PCP_ACCESS": ("primary care",),
    "TELEHEALTH_AVAILABLE": ("telehealth",),
    "URGENT_CARE_ACCESS_ADVANTAGE": ("urgent care",),
    "PRIOR_CM_ENGAGEMENT": ("care management",),
    "OUTPATIENT_CONTINUITY_OPPORTUNITY": ("care management", "primary care"),
}

_SAFETY_STATE_PHRASES = {
    "CLEAR": ("safety state is clear", "marked as clear", "no safety override"),
    "CAUTION": ("safety state is caution", "marked as caution"),
    "OVERRIDE": ("safety override was triggered", "triggered a safety override", "safety state is override"),
}

# GenAI-specific prohibited content -- beyond safety_policy's centralized
# ED-avoidance phrase list (which the Safety & Policy Agent also enforces),
# an explanation-generating LLM has additional failure modes: diagnosing,
# prescribing, inventing causal claims. Substring-matched, case-insensitive
# -- deliberately simple and auditable rather than a second ML model.
GENAI_PROHIBITED_SUBSTRINGS: tuple[str, ...] = (
    "diagnos",  # diagnose / diagnosis / diagnosed
    "prescri",  # prescribe / prescription
    "take ibuprofen", "take tylenol", "take acetaminophen", "take aspirin",
    "mg of", "milligrams", "dosage",
    "your medication", "recommend you take", "recommend taking",
    "this proves", "this confirms", "will definitely", "will cause",
    "causes this member", "caused by",
)

# A model is explicitly ALLOWED to say things like "this tool does not
# diagnose any condition" (that is the correct, safe disclaimer-style
# framing) -- only an UN-negated diagnos.../prescri... claim is a real
# violation. Strip any negated occurrence (within a short word window)
# before running the substring check above, rather than naively banning
# "diagnos"/"prescri" as bare substrings.
_NEGATED_CLINICAL_CLAIM_RE = re.compile(
    r"\b(not|no|never|n't|cannot|can't|won't)\b[^.]{0,40}?\b(diagnos\w*|prescri\w*)"
)


class GenAIConfig:
    """Read fresh from the environment on every construction (not cached
    at import time) so tests can monkeypatch os.environ per-test and so a
    running server picks up a changed GENAI_ENABLED without a restart."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("GENAI_ENABLED", "false").strip().lower() == "true"
        self.base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        try:
            self.timeout_seconds = float(os.environ.get("GENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        except ValueError:
            self.timeout_seconds = DEFAULT_TIMEOUT_SECONDS


def load_config() -> GenAIConfig:
    return GenAIConfig()


# ---------------------------------------------------------------------------
# Deterministic fallback -- mandatory, always available, zero external
# dependencies. Uses ONLY the model-derived factors, navigation
# destination/reason codes, and safety state already present in `payload`
# (Phase 8C Part 10). Wording for each safety state matches Part 12 exactly.
# ---------------------------------------------------------------------------

def _title(token: str) -> str:
    return token.replace("_", " ").title()


def _deterministic_explanation(payload: dict) -> MemberExplanation:
    risk = payload["risk"]
    navigation = payload["navigation"]
    safety = payload["safety"]

    tier = risk["tier"]
    factors = risk.get("factors") or []
    increasing = [f["display_name"] for f in factors if f.get("direction") == "INCREASES_RISK"]
    decreasing = [f["display_name"] for f in factors if f.get("direction") == "DECREASES_RISK"]

    risk_explanation = f"This member's modeled risk tier is {tier} (estimated probability {risk['probability']:.1%})."
    if increasing:
        risk_explanation += f" Factors that contributed to a higher estimate: {', '.join(increasing)}."
    if decreasing:
        risk_explanation += f" Factors that contributed to a lower estimate: {', '.join(decreasing)}."

    destination = navigation.get("destination")
    reason_codes = navigation.get("reason_codes") or []
    if destination:
        navigation_explanation = f"The suggested navigation destination is {_title(destination)}."
        if reason_codes:
            navigation_explanation += f" Reasons: {', '.join(_title(rc) for rc in reason_codes)}."
    else:
        navigation_explanation = "No proactive navigation destination applies for this member."

    state = safety["state"]
    safety_explanation = _SAFETY_STATE_TEMPLATES.get(state, "The current safety state could not be determined.")

    summary = (
        f"{_title(tier)} modeled risk tier; suggested navigation: "
        f"{_title(destination) if destination else 'none'}; safety state: {_title(state)}."
    )

    return MemberExplanation(
        summary=summary,
        risk_explanation=risk_explanation,
        navigation_explanation=navigation_explanation,
        safety_explanation=safety_explanation,
        disclaimer=safety_policy.BASE_DISCLAIMER,
        explanation_source=ExplanationSource.DETERMINISTIC_FALLBACK,
        model_used=None,
        generation_time_ms=None,
    )


# ---------------------------------------------------------------------------
# GenAI (Ollama / qwen3:8b) path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an explanation-writing assistant for a care-navigation support tool. "
    "You will be given ONLY a small structured JSON summary of a decision that has "
    "ALREADY been made by separate deterministic systems: a risk model, a navigation "
    "rule engine, and a safety policy engine. Your ONLY job is to translate that "
    "already-decided structured result into brief, plain-English sentences.\n\n"
    "You MUST NOT: state a different risk probability or risk tier than the one given; "
    "invent a different navigation destination; state or imply a different safety state "
    "than the one given; diagnose any condition; recommend medications, dosages, or "
    "treatment; invent symptoms, diagnoses, or risk factors not present in the input "
    "JSON; claim any factor CAUSES a future ED visit -- describe factors only as "
    "contributing to the model's own estimate; tell the reader to avoid, skip, or delay "
    "emergency or ED care; or reveal your internal reasoning process.\n\n"
    "Use ONLY the facts given in the input JSON. If information is not in the input "
    "JSON, do not state it. Respond with ONLY the requested JSON object -- summary, "
    "risk_explanation, navigation_explanation, safety_explanation, disclaimer -- each "
    "value a short plain-English string of 1-2 sentences."
)


def _response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "risk_explanation": {"type": "string"},
            "navigation_explanation": {"type": "string"},
            "safety_explanation": {"type": "string"},
            "disclaimer": {"type": "string"},
        },
        "required": ["summary", "risk_explanation", "navigation_explanation", "safety_explanation", "disclaimer"],
    }


def _call_ollama(config: GenAIConfig, payload: dict) -> dict | None:
    """Returns the parsed JSON object the model produced, or None on ANY
    failure (connection refused, timeout, non-2xx, malformed JSON, wrong
    shape). Never raises."""
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        "think": False,
        "format": _response_schema(),
        "stream": False,
    }
    try:
        response = httpx.post(f"{config.base_url}/api/chat", json=body, timeout=config.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        parsed = json.loads(content)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _genai_prohibited_violation(text: str) -> bool:
    normalized = text.lower()
    stripped = _NEGATED_CLINICAL_CLAIM_RE.sub(" ", normalized)
    return any(s in stripped for s in GENAI_PROHIBITED_SUBSTRINGS)


def _tier_consistency_violation(text: str, actual_tier: str) -> bool:
    """Scoped to the summary + risk_explanation fields only -- the two
    places a tier claim can legitimately appear."""
    normalized = text.lower()
    for tier, phrases in _TIER_PHRASES.items():
        if tier == actual_tier:
            continue
        if any(p in normalized for p in phrases):
            return True
    return False


def _destination_consistency_violation(text: str, actual_destination: str | None, reason_codes: list[str]) -> bool:
    """Scoped to the summary + navigation_explanation fields only. A
    destination-adjacent phrase that is faithfully paraphrasing a reason
    code actually present in `reason_codes` (e.g. "urgent care" while
    explaining URGENT_CARE_ACCESS_ADVANTAGE as context for a
    CARE_MANAGEMENT destination) is NOT a violation -- only a phrase with
    no such justification is treated as an unsupported destination claim."""
    normalized = text.lower()
    justified: set[str] = set()
    for code in reason_codes:
        justified.update(_REASON_CODE_JUSTIFIED_DESTINATION_PHRASES.get(code, ()))
    for destination, phrases in _DESTINATION_PHRASES.items():
        if destination == actual_destination:
            continue
        for phrase in phrases:
            if phrase in normalized and phrase not in justified:
                return True
    return False


def _safety_state_consistency_violation(text: str, actual_state: str) -> bool:
    """Scoped to the summary + safety_explanation fields only."""
    normalized = text.lower()
    for state, phrases in _SAFETY_STATE_PHRASES.items():
        if state == actual_state:
            continue
        if any(p in normalized for p in phrases):
            return True
    return False


def _validate_genai_output(raw: dict, payload: dict) -> MemberExplanation | None:
    """Returns a MemberExplanation built from `raw` if it passes every
    structural, policy, and consistency check; otherwise None (the caller
    falls back to the deterministic explanation). Never raises.

    Note: `raw["disclaimer"]` is checked only for presence/type (required
    by the schema) and is otherwise IGNORED here -- the model's own
    disclaimer text is never surfaced (see below), so it is deliberately
    excluded from the policy/consistency text that follows; validating
    text nobody ever sees would only create false-positive fallbacks."""
    for key in _REQUIRED_KEYS:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        if len(value) > _MAX_FIELD_CHARS:
            return None
    if not isinstance(raw.get("disclaimer"), str):
        return None

    combined = " ".join(raw[key] for key in _REQUIRED_KEYS)
    if len(combined) > _MAX_COMBINED_CHARS:
        return None

    if safety_policy.check_text(combined):
        return None
    if _genai_prohibited_violation(combined):
        return None

    actual_tier = payload["risk"]["tier"]
    actual_destination = payload["navigation"].get("destination")
    actual_state = payload["safety"]["state"]
    reason_codes = payload["navigation"].get("reason_codes") or []

    # Scoped to each field's OWN dedicated sentence only -- deliberately
    # excluding `summary`, which legitimately synthesizes language from
    # ALL of risk/navigation/safety at once (e.g. "...despite telehealth
    # availability reducing risk" is a correct restatement of a risk
    # FACTOR, not a destination claim, but would otherwise collide with
    # the TELEHEALTH destination phrase if summary were included here).
    if _tier_consistency_violation(raw["risk_explanation"], actual_tier):
        return None
    if _destination_consistency_violation(raw["navigation_explanation"], actual_destination, reason_codes):
        return None
    if _safety_state_consistency_violation(raw["safety_explanation"], actual_state):
        return None

    return MemberExplanation(
        summary=raw["summary"].strip(),
        risk_explanation=raw["risk_explanation"].strip(),
        navigation_explanation=raw["navigation_explanation"].strip(),
        safety_explanation=raw["safety_explanation"].strip(),
        # Never the model's own disclaimer text -- always the same
        # centrally governed disclaimer the Safety & Policy Agent uses,
        # so GenAI can never weaken or omit it.
        disclaimer=safety_policy.BASE_DISCLAIMER,
        explanation_source=ExplanationSource.GENAI,
    )


def generate_explanation(payload: dict, config: GenAIConfig | None = None) -> MemberExplanation:
    """Top-level entry point. `payload` must already be the minimal,
    allow-listed structure validated by backend/main.py's ExplainRequest
    Pydantic model -- this function does not see raw CSVs or full member
    history. Never raises; always returns a MemberExplanation, GenAI or
    deterministic-fallback."""
    config = config or load_config()
    if not config.enabled:
        return _deterministic_explanation(payload)

    start = time.monotonic()
    try:
        raw = _call_ollama(config, payload)
        if raw is None:
            return _deterministic_explanation(payload)
        result = _validate_genai_output(raw, payload)
        if result is None:
            return _deterministic_explanation(payload)
        elapsed_ms = (time.monotonic() - start) * 1000
        return dataclasses.replace(result, model_used=config.model, generation_time_ms=round(elapsed_ms, 1))
    except Exception:
        return _deterministic_explanation(payload)
