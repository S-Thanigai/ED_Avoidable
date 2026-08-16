"""
Phase 8D -- CRITICAL FIXES + REGRESSION HARDENING, Part 15 GENAI tests
(numbered 1-11 in the Phase 8D spec).

Covers the hardened backend/agents/genai_explanation.py: the structured
risk_tier/navigation_destination/safety_state echo (exact-match validated
against the authoritative decision) and the POSITIVE safety-consistency
check added in response to the Phase 8C health check's CRITICAL finding
(a vague-but-wrong reassurance for an actual OVERRIDE state could pass the
old phrase-only-rejection check because it named no OTHER state's exact
phrase).

All tests here monkeypatch `_call_ollama` directly -- no live Ollama
required, CI-safe.
"""
import json

import pytest

import genai_explanation
from contracts import ExplanationSource

OVERRIDE_PAYLOAD = {
    "risk": {"probability": 0.05, "tier": "LOW", "model_version": "uc07-risk-synthetic-v1", "factors": []},
    "navigation": {"destination": None, "reason_codes": []},
    "safety": {"state": "OVERRIDE", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
    "synthetic_model": True,
}

CAUTION_PAYLOAD = {
    "risk": {"probability": 0.15, "tier": "MODERATE", "model_version": "uc07-risk-synthetic-v1", "factors": []},
    "navigation": {"destination": "CARE_MANAGEMENT", "reason_codes": ["ELEVATED_FUTURE_RISK"]},
    "safety": {"state": "CAUTION", "context_completeness": "ABSENT", "context_source": "NOT_AVAILABLE"},
    "synthetic_model": True,
}

LOW_PAYLOAD = {
    "risk": {"probability": 0.03, "tier": "LOW", "model_version": "uc07-risk-synthetic-v1", "factors": []},
    "navigation": {"destination": "PRIMARY_CARE", "reason_codes": ["OUTPATIENT_CONTINUITY_OPPORTUNITY"]},
    "safety": {"state": "CLEAR", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
    "synthetic_model": True,
}

HIGH_PAYLOAD = {
    "risk": {"probability": 0.35, "tier": "HIGH", "model_version": "uc07-risk-synthetic-v1", "factors": []},
    "navigation": {"destination": "CARE_MANAGEMENT", "reason_codes": ["ELEVATED_FUTURE_RISK"]},
    "safety": {"state": "CLEAR", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
    "synthetic_model": True,
}

NO_NAV_PAYLOAD = {
    "risk": {"probability": 0.04, "tier": "LOW", "model_version": "uc07-risk-synthetic-v1", "factors": []},
    "navigation": {"destination": "NO_PROACTIVE_NAVIGATION", "reason_codes": ["NO_OPPORTUNITY_IDENTIFIED"]},
    "safety": {"state": "CLEAR", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
    "synthetic_model": True,
}


def _enabled_config(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    monkeypatch.setenv("GENAI_TIMEOUT_SECONDS", "5")
    return genai_explanation.load_config()


def _good_text(payload: dict) -> dict:
    """Correct structured echo + safe text for the given payload, as a
    baseline good raw response to mutate per test."""
    tier = payload["risk"]["tier"]
    destination = payload["navigation"].get("destination")
    state = payload["safety"]["state"]
    safety_text = {
        "CLEAR": "Complete supplied information did not trigger a configured safety rule.",
        "CAUTION": "Current safety information is incomplete, so the situation cannot be confirmed.",
        "OVERRIDE": "A configured high-acuity safety trigger exists; emergency care should not be delayed.",
    }[state]
    return {
        "risk_tier": tier,
        "navigation_destination": destination if destination else "NONE",
        "safety_state": state,
        "summary": f"{tier.title()} modeled risk; safety state {state.title()}.",
        "risk_explanation": f"This member's modeled risk tier is {tier.lower()}.",
        "navigation_explanation": (
            f"The suggested navigation destination is {destination.replace('_', ' ').title()}."
            if destination and destination != "NO_PROACTIVE_NAVIGATION"
            else "No proactive navigation destination applies for this member."
        ),
        "safety_explanation": safety_text,
    }


# ---- 1. actual OVERRIDE + "everything looks fine" -> fallback ----

def test_1_override_everything_looks_fine_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(OVERRIDE_PAYLOAD)
    raw["safety_explanation"] = "Everything looks fine, no need to worry."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 2. actual OVERRIDE + wrong structured safety_state -> fallback ----

def test_2_override_wrong_structured_safety_state_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(OVERRIDE_PAYLOAD)
    raw["safety_state"] = "CLEAR"  # structured field disagrees with actual OVERRIDE
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 3. actual CAUTION + "patient is safe" -> fallback ----

def test_3_caution_patient_is_safe_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(CAUTION_PAYLOAD)
    raw["safety_explanation"] = "The patient is safe based on available information."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(CAUTION_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 4. actual LOW + structured HIGH -> fallback ----

def test_4_actual_low_structured_high_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(LOW_PAYLOAD)
    raw["risk_tier"] = "HIGH"
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(LOW_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 5. actual HIGH + structured LOW -> fallback ----

def test_5_actual_high_structured_low_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(HIGH_PAYLOAD)
    raw["risk_tier"] = "LOW"
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(HIGH_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_5b_actual_low_synonym_elevated_risk_text_falls_back(monkeypatch):
    """The exact adversarial example from the health check: 'significant
    and elevated risk' for an actual LOW tier, with an otherwise-correct
    structured echo -- proves the free-text synonym check works
    independently of the structured layer."""
    config = _enabled_config(monkeypatch)
    raw = _good_text(LOW_PAYLOAD)
    raw["risk_explanation"] = "This member is at significant and elevated risk of an ED visit soon."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(LOW_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 6. actual CARE_MANAGEMENT + structured NO_PROACTIVE_NAVIGATION -> fallback ----

def test_6_actual_care_management_structured_no_proactive_nav_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(CAUTION_PAYLOAD)  # actual destination is CARE_MANAGEMENT
    raw["navigation_destination"] = "NO_PROACTIVE_NAVIGATION"
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(CAUTION_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_6b_care_management_described_as_no_action_needed_falls_back(monkeypatch):
    """The exact adversarial example from the health check: a real
    CARE_MANAGEMENT destination silently downgraded to 'no action
    needed' in the free text, with a CORRECT structured echo."""
    config = _enabled_config(monkeypatch)
    raw = _good_text(CAUTION_PAYLOAD)
    raw["navigation_explanation"] = "No further action or referral is needed for this member at this time."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(CAUTION_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 7. actual NO_PROACTIVE_NAVIGATION + structured CARE_MANAGEMENT -> fallback ----

def test_7_actual_no_proactive_nav_structured_care_management_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(NO_NAV_PAYLOAD)
    raw["navigation_destination"] = "CARE_MANAGEMENT"
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(NO_NAV_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_7b_no_proactive_nav_described_as_care_management_recommended_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    raw = _good_text(NO_NAV_PAYLOAD)
    raw["navigation_explanation"] = "Care Management is recommended for this member going forward."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: raw)
    result = genai_explanation.generate_explanation(NO_NAV_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 8. correct structured values + safe text -> GENAI accepted ----

@pytest.mark.parametrize("payload", [OVERRIDE_PAYLOAD, CAUTION_PAYLOAD, LOW_PAYLOAD, HIGH_PAYLOAD, NO_NAV_PAYLOAD])
def test_8_correct_structured_and_safe_text_is_accepted(monkeypatch, payload):
    config = _enabled_config(monkeypatch)
    raw = _good_text(payload)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, p: raw)
    result = genai_explanation.generate_explanation(payload, config)
    assert result.explanation_source == ExplanationSource.GENAI


# ---- 9. malformed JSON -> fallback ----

def test_9_malformed_json_content_falls_back(monkeypatch):
    import httpx

    config = _enabled_config(monkeypatch)

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{not valid json"}}

    monkeypatch.setattr(genai_explanation.httpx, "post", lambda *a, **k: _FakeResponse())
    result = genai_explanation.generate_explanation(LOW_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 10. prohibited paraphrase -> fallback ----

@pytest.mark.parametrize("phrase", [
    "Everything looks fine",
    "No need to worry",
    "No reason to go to the hospital",
    "Nothing urgent here",
    "Routine care is enough",
    "No emergency concern",
    "You are safe",
    "No further action is needed",
])
def test_10_prohibited_paraphrase_falls_back(monkeypatch, phrase):
    config = _enabled_config(monkeypatch)
    for payload in (OVERRIDE_PAYLOAD, CAUTION_PAYLOAD):
        raw = _good_text(payload)
        raw["safety_explanation"] = f"{phrase}."
        monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, p, raw=raw: raw)
        result = genai_explanation.generate_explanation(payload, config)
        assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK, (
            f"phrase {phrase!r} should have been rejected for state {payload['safety']['state']}"
        )


# ---- 11. decision values unchanged after GenAI failure ----

def test_11_decision_values_unchanged_after_genai_failure(monkeypatch):
    config = _enabled_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, p: None)  # simulate total failure
    original = json.loads(json.dumps(OVERRIDE_PAYLOAD))
    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    # generate_explanation must never mutate its input payload
    assert OVERRIDE_PAYLOAD == original
    # and the deterministic fallback must describe the ACTUAL decision
    assert "high-acuity" in result.safety_explanation.lower() or "override" in result.safety_explanation.lower()
