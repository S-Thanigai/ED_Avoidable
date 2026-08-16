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
    - Phase 8D: the model MUST echo three structured enum fields
      (risk_tier, navigation_destination, safety_state) verbatim from its
      input, and every one of them is exact-match checked against the
      authoritative decision -- this is the PRIMARY guarantee, not free
      text. A Phase 8C health check found that free-text-only consistency
      checks (rejecting known-wrong phrases) could still be bypassed by a
      vague-but-wrong reassurance that names no OTHER state's phrase at
      all (e.g. "everything looks fine" for an actual OVERRIDE); the
      structured echo closes that gap structurally, and the free-text
      checks below now run as a SECOND, independent layer on top of it,
      not instead of it.
    - safety_policy.check_text() (the SAME centralized prohibited-phrase
      list the Safety & Policy Agent itself enforces) is run against
      every generated sentence
    - an additional GenAI-specific prohibited-content check
      (_genai_prohibited_violation) blocks diagnosis/medication/dosage/
      causal-claim language
    - a tier/destination/safety-state free-text CONSISTENCY check
      (_tier_consistency_violation / _destination_consistency_violation /
      _navigation_self_contradiction_violation /
      _safety_state_consistency_violation) rejects output that names a
      DIFFERENT risk tier, navigation destination, or safety state than
      the one actually supplied in `payload`, contradicts the CORRECT
      destination (e.g. calling a real CARE_MANAGEMENT destination
      "no action needed"), or -- for safety text specifically -- uses a
      reassurance phrasing that's never acceptable, or omits the required
      positive meaning for OVERRIDE/CAUTION (Phase 8D Part 2: rejecting
      wrong phrases is not enough; the correct meaning must be present)
    - the returned `disclaimer` field is NEVER the model's own text --
      it is always overwritten with safety_policy.BASE_DISCLAIMER, so the
      model cannot weaken or omit the mandatory disclaimer
    - "think": false suppresses qwen3's chain-of-thought in the raw
      Ollama response, and only the structured content fields are ever
      read out of that response -- no internal reasoning field is ever
      surfaced to a caller
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

# Phase 8D: the model must ECHO these three structured fields verbatim from
# its input -- exact-match validated against the authoritative decision
# below -- rather than being trusted to convey tier/destination/state
# semantics through free text alone. "NONE" is this module's own sentinel
# for a null navigation.destination (which occurs only when safety.state is
# OVERRIDE -- see contracts.py's FinalNavigationView docstring); it is not
# a NavigationDestination enum value, just a JSON-safe way to require an
# exact echo of "no destination" instead of leaving the field unconstrained.
_RISK_TIER_VALUES = ("LOW", "MODERATE", "HIGH")
_NAVIGATION_DESTINATION_VALUES = (
    "PRIMARY_CARE", "URGENT_CARE", "TELEHEALTH", "CARE_MANAGEMENT", "NO_PROACTIVE_NAVIGATION", "NONE",
)
_SAFETY_STATE_VALUES = ("CLEAR", "CAUTION", "OVERRIDE")

_REQUIRED_TEXT_KEYS = ("summary", "risk_explanation", "navigation_explanation", "safety_explanation")
_REQUIRED_ENUM_KEYS = ("risk_tier", "navigation_destination", "safety_state")
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

# Phase 8D Part 3 -- exact-phrase matching alone is too weak: "significant
# and elevated risk" for an actual LOW tier names no OTHER tier's exact
# phrase, so _TIER_PHRASES alone would miss it (a demonstrated real gap).
# These proximity regexes catch a severity WORD appearing near "risk"
# (either order, within a short window) that contradicts the actual tier,
# without banning the word outright -- "elevated" is fine describing an
# unrelated factor, only "elevated risk"/"risk...elevated" is flagged. Not
# an exhaustive synonym list by design (Part 3: "do not overfit to one
# phrase") -- covers the common English ways to overstate/understate a
# tier; anything genuinely ambiguous is deliberately left to fall through
# to the exact-match structured `risk_tier` field, which is the primary
# guarantee.
_ELEVATED_RISK_RE = re.compile(
    r"\b(elevated|significant|substantial|considerable|serious|severe|heightened|high|higher)\b"
    r"[^.]{0,30}\brisk\b"
    r"|\brisk\b[^.]{0,30}\b(elevated|significant|substantial|considerable|serious|severe|heightened|high|higher)\b"
)
_MINIMIZING_RISK_RE = re.compile(
    r"\b(minimal|negligible|insignificant|little to no|very low|low|lower)\b"
    r"[^.]{0,30}\brisk\b"
    r"|\brisk\b[^.]{0,30}\b(minimal|negligible|insignificant|little to no|very low|low|lower)\b"
)
_EXTREME_RISK_RE = re.compile(r"\bvery\s+(low|high)\b[^.]{0,30}\brisk\b|\brisk\b[^.]{0,30}\bvery\s+(low|high)\b")

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

# Phase 8D Part 2 -- POSITIVE safety consistency: rejecting known-wrong-state
# phrases is not enough on its own (a vague-but-wrong reassurance like
# "everything looks fine" names no OTHER state's phrase at all, so it would
# slip past _SAFETY_STATE_PHRASES alone -- this was a demonstrated,
# real gap found in the Phase 8C health check). Two more layers:
#   1. a fixed list of reassurance/all-clear phrasings that are NEVER
#      acceptable in a safety_explanation, regardless of actual state --
#      even a genuinely CLEAR state must never be described this way
#      (Phase 8C Part 12: "never say patient is safe / no emergency exists").
#   2. for OVERRIDE and CAUTION specifically, the explanation must contain
#      at least one phrase that actually carries the REQUIRED positive
#      meaning for that state (an override trigger exists / current
#      information is incomplete) -- silence is a violation, not just
#      contradiction.
_SAFETY_REASSURANCE_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "everything looks fine", "everything is fine", "everything's fine",
    "no need to worry", "nothing to worry about", "no reason to worry",
    "no reason to go to the hospital", "no reason to seek emergency",
    "no reason to seek care", "nothing urgent", "nothing urgent here",
    "routine care is enough", "no emergency concern", "no emergency concerns",
    "you are safe", "you're safe", "patient is safe", "member is safe",
    "no further action is needed", "no further action needed",
    "no action needed", "no action is needed", "all good", "all clear here",
    "no immediate concern", "no immediate concerns", "no concern", "no concerns",
    "confirmed safe", "safe to proceed", "nothing further required",
    "not an emergency", "this is not an emergency",
)

_SAFETY_STATE_REQUIRED_ANY_PHRASES: dict[str, tuple[str, ...]] = {
    "OVERRIDE": (
        "override", "triggered", "trigger", "high-acuity", "high acuity",
        "safety rule", "should not be delayed", "do not delay", "not be delayed",
        "emergency care", "call 911", "emergency department",
    ),
    "CAUTION": (
        "incomplete", "absent", "not confirm", "cannot confirm", "unconfirmed",
        "unknown", "not fully known", "not available", "partial", "not confirmed",
    ),
    # CLEAR has no required-positive-phrase list: the correct wording is a
    # narrow, specific claim ("complete information did not trigger a
    # configured rule") that this module's own deterministic template
    # already models exactly -- requiring a fixed phrase set here would
    # just re-implement _SAFETY_STATE_PHRASES["CLEAR"], which already runs.
}

# Phase 8D Part 4 -- a navigation_explanation for a REAL destination
# (anything except NO_PROACTIVE_NAVIGATION / null) must never claim no
# action is needed -- that directly contradicts having a destination at
# all. Scoped separately from _DESTINATION_PHRASES (which catches naming a
# DIFFERENT destination) because this catches self-contradiction about the
# CORRECT destination.
_NAVIGATION_NO_ACTION_PHRASES: tuple[str, ...] = (
    "no action needed", "no action is needed", "no follow-up needed",
    "no follow-up is needed", "nothing further required", "nothing further is required",
    "no further action", "no further action is needed", "no referral needed",
    "no referral is needed", "not necessary to follow up",
)

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
    "rule engine, and a safety policy engine.\n\n"
    "PART 1 -- REQUIRED STRUCTURED ECHO: your response MUST include three fields that "
    "EXACTLY copy values already given to you in the input JSON, with no rewording or "
    "inference: \"risk_tier\" (copy risk.tier exactly: LOW, MODERATE, or HIGH), "
    "\"navigation_destination\" (copy navigation.destination exactly -- one of "
    "PRIMARY_CARE, URGENT_CARE, TELEHEALTH, CARE_MANAGEMENT, NO_PROACTIVE_NAVIGATION -- or "
    "the literal string \"NONE\" if navigation.destination is null), and \"safety_state\" "
    "(copy safety.state exactly: CLEAR, CAUTION, or OVERRIDE). Copy these verbatim; do "
    "not guess or change them.\n\n"
    "PART 2 -- PLAIN-ENGLISH EXPLANATION: write summary, risk_explanation, "
    "navigation_explanation, and safety_explanation as short (1-2 sentence) plain-English "
    "strings that are fully consistent with the structured values above.\n\n"
    "You MUST NOT: state a different risk probability or risk tier than the one given, "
    "including via a synonym (never call a LOW tier \"elevated\"/\"significant\", and "
    "never call a HIGH tier \"minimal\"/\"low\"); invent a different navigation "
    "destination, or describe a real navigation destination as needing no action or no "
    "follow-up; state or imply a different safety state than the one given.\n\n"
    "Safety-state wording is especially important: for an OVERRIDE safety state, NEVER "
    "use reassuring language such as \"everything looks fine\", \"no need to worry\", "
    "\"you are safe\", or \"nothing urgent\" -- instead clearly state that a configured "
    "high-acuity safety trigger exists and that emergency care should not be delayed. For "
    "a CAUTION safety state, clearly state that current safety information is incomplete "
    "or unknown, and NEVER imply the situation is confirmed safe. For a CLEAR safety "
    "state, say only that complete supplied information did not trigger a configured "
    "safety rule -- NEVER say \"the patient is safe\" or \"no emergency exists\".\n\n"
    "You also MUST NOT: diagnose any condition; recommend medications, dosages, or "
    "treatment; invent symptoms, diagnoses, or risk factors not present in the input "
    "JSON; claim any factor CAUSES a future ED visit -- describe factors only as "
    "contributing to the model's own estimate; tell the reader to avoid, skip, or delay "
    "emergency or ED care; or reveal your internal reasoning process.\n\n"
    "Use ONLY the facts given in the input JSON. If information is not in the input "
    "JSON, do not state it. Respond with ONLY the requested JSON object."
)


def _response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "risk_tier": {"type": "string", "enum": list(_RISK_TIER_VALUES)},
            "navigation_destination": {"type": "string", "enum": list(_NAVIGATION_DESTINATION_VALUES)},
            "safety_state": {"type": "string", "enum": list(_SAFETY_STATE_VALUES)},
            "summary": {"type": "string"},
            "risk_explanation": {"type": "string"},
            "navigation_explanation": {"type": "string"},
            "safety_explanation": {"type": "string"},
        },
        "required": [
            "risk_tier", "navigation_destination", "safety_state",
            "summary", "risk_explanation", "navigation_explanation", "safety_explanation",
        ],
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
    """Scoped to risk_explanation only. Two layers: (1) exact "X risk"
    phrase belonging to a DIFFERENT tier (existing), and (2) Phase 8D --
    a severity word placed near "risk" that contradicts the actual tier
    even via synonym (e.g. "significant and elevated risk" for an actual
    LOW tier). Layer 2 is deliberately conservative/proximity-scoped, not
    an exhaustive synonym ban, per "do not overfit to one phrase, if
    uncertain fall back" -- a false positive here just costs one fallback
    to deterministic text, never a wrong decision."""
    normalized = text.lower()
    for tier, phrases in _TIER_PHRASES.items():
        if tier == actual_tier:
            continue
        if any(p in normalized for p in phrases):
            return True

    if actual_tier == "LOW" and _ELEVATED_RISK_RE.search(normalized):
        return True
    if actual_tier == "HIGH" and _MINIMIZING_RISK_RE.search(normalized):
        return True
    if actual_tier == "MODERATE" and _EXTREME_RISK_RE.search(normalized):
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


def _navigation_self_contradiction_violation(text: str, actual_destination: str | None) -> bool:
    """Phase 8D Part 4 -- scoped to navigation_explanation only. Catches
    describing a REAL destination (anything except null/
    NO_PROACTIVE_NAVIGATION) as needing no action -- a self-contradiction
    about the CORRECT destination that _destination_consistency_violation
    (which only checks for naming a DIFFERENT destination) cannot catch."""
    if not actual_destination or actual_destination == "NO_PROACTIVE_NAVIGATION":
        return False
    normalized = text.lower()
    return any(p in normalized for p in _NAVIGATION_NO_ACTION_PHRASES)


def _safety_state_consistency_violation(text: str, actual_state: str) -> bool:
    """Scoped to safety_explanation only. Phase 8D Part 2 -- three layers:
    (1) a reassurance/all-clear phrasing that is NEVER acceptable for ANY
    state (this alone closes the demonstrated real gap: a vague-but-wrong
    "everything looks fine" for an actual OVERRIDE named no OTHER state's
    phrase, so layer 2 below would have missed it); (2) an exact phrase
    belonging to a DIFFERENT state (existing); (3) for OVERRIDE/CAUTION,
    the text must POSITIVELY carry the required meaning for that state,
    not just avoid contradicting it -- silence on the override trigger, or
    on the incompleteness of current information, is itself a violation."""
    normalized = text.lower()

    if any(p in normalized for p in _SAFETY_REASSURANCE_FORBIDDEN_PHRASES):
        return True

    for state, phrases in _SAFETY_STATE_PHRASES.items():
        if state == actual_state:
            continue
        if any(p in normalized for p in phrases):
            return True

    required_any = _SAFETY_STATE_REQUIRED_ANY_PHRASES.get(actual_state)
    if required_any and not any(p in normalized for p in required_any):
        return True

    return False


def _validate_genai_output(raw: dict, payload: dict) -> MemberExplanation | None:
    """Returns a MemberExplanation built from `raw` if it passes every
    structural, echo, policy, and consistency check; otherwise None (the
    caller falls back to the deterministic explanation). Never raises.

    Phase 8D: validation now has two independent layers, deliberately not
    just one. Layer 1 (structured echo) is the PRIMARY guarantee -- three
    enum fields the model must copy verbatim from its input, exact-match
    checked against the authoritative decision, never inferred from free
    text ("do not trust free text to carry decision semantics"). Layer 2
    (free-text consistency) is defense in depth: even if the structured
    fields are correct, the free-text sentences are independently checked
    so a model that gets the enum right but writes contradictory prose
    (e.g. correct `safety_state: "OVERRIDE"` alongside a
    safety_explanation that says "everything looks fine") is still
    rejected. Either layer failing alone is enough to reject."""
    for key in _REQUIRED_TEXT_KEYS:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        if len(value) > _MAX_FIELD_CHARS:
            return None
    for key in _REQUIRED_ENUM_KEYS:
        if not isinstance(raw.get(key), str):
            return None

    combined = " ".join(raw[key] for key in _REQUIRED_TEXT_KEYS)
    if len(combined) > _MAX_COMBINED_CHARS:
        return None

    if safety_policy.check_text(combined):
        return None
    if _genai_prohibited_violation(combined):
        return None

    actual_tier = payload["risk"]["tier"]
    actual_destination = payload["navigation"].get("destination")
    actual_destination_echo = actual_destination if actual_destination else "NONE"
    actual_state = payload["safety"]["state"]
    reason_codes = payload["navigation"].get("reason_codes") or []

    # --- Layer 1: structured enum echo, exact match, primary guarantee ---
    if raw["risk_tier"] != actual_tier:
        return None
    if raw["navigation_destination"] != actual_destination_echo:
        return None
    if raw["safety_state"] != actual_state:
        return None

    # --- Layer 2: free-text consistency, scoped to each field's OWN
    # dedicated sentence only -- deliberately excluding `summary`, which
    # legitimately synthesizes language from ALL of risk/navigation/safety
    # at once (e.g. "...despite telehealth availability reducing risk" is
    # a correct restatement of a risk FACTOR, not a destination claim, but
    # would otherwise collide with the TELEHEALTH destination phrase if
    # summary were included here). ---
    if _tier_consistency_violation(raw["risk_explanation"], actual_tier):
        return None
    if _destination_consistency_violation(raw["navigation_explanation"], actual_destination, reason_codes):
        return None
    if _navigation_self_contradiction_violation(raw["navigation_explanation"], actual_destination):
        return None
    if _safety_state_consistency_violation(raw["safety_explanation"], actual_state):
        return None

    return MemberExplanation(
        summary=raw["summary"].strip(),
        risk_explanation=raw["risk_explanation"].strip(),
        navigation_explanation=raw["navigation_explanation"].strip(),
        safety_explanation=raw["safety_explanation"].strip(),
        # Never the model's own disclaimer text (not even requested
        # anymore, see _response_schema) -- always the same centrally
        # governed disclaimer the Safety & Policy Agent uses, so GenAI can
        # never weaken or omit it.
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
