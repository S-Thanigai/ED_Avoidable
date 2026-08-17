"""
GroqCloud primary-provider integration tests.

Covers backend/agents/genai_explanation.py's provider chain: Groq
(primary) -> Ollama (secondary fallback) -> deterministic explanation
(final fallback). None of these tests require a live Groq or Ollama
instance -- `_call_groq` / `_call_ollama` are monkeypatched directly, same
convention as tests/test_genai_explanation.py. Groq output goes through
the exact same `_validate_genai_output` gate as Ollama output (see
tests/test_phase8d_genai_hardening.py and
tests/test_genai_explanation_authority.py for the shared validation/
authority regression suite -- this file only adds Groq-specific routing
coverage, it does not re-implement those checks).
"""
import json

import pytest

import genai_explanation
from contracts import ExplanationSource

GOOD_PAYLOAD = {
    "risk": {
        "probability": 0.274,
        "tier": "HIGH",
        "model_version": "uc07-risk-synthetic-v1",
        "factors": [
            {"display_name": "Recent ED utilization", "direction": "INCREASES_RISK"},
            {"display_name": "Telehealth available", "direction": "DECREASES_RISK"},
        ],
    },
    "navigation": {"destination": "CARE_MANAGEMENT", "reason_codes": ["ELEVATED_FUTURE_RISK"]},
    "safety": {"state": "CAUTION", "context_completeness": "ABSENT", "context_source": "NOT_AVAILABLE"},
    "synthetic_model": True,
}

GOOD_RAW = {
    "risk_tier": "HIGH",
    "navigation_destination": "CARE_MANAGEMENT",
    "safety_state": "CAUTION",
    "summary": "High modeled risk; suggested navigation is Care Management.",
    "risk_explanation": "This member's modeled risk tier is high, driven mainly by recent ED utilization.",
    "navigation_explanation": "The suggested navigation destination is care management, given elevated future risk.",
    "safety_explanation": "Current safety information is absent, so the system cannot confirm a fully clear state.",
}

OVERRIDE_PAYLOAD = {
    "risk": {"probability": 0.05, "tier": "LOW", "model_version": "uc07-risk-synthetic-v1", "factors": []},
    "navigation": {"destination": None, "reason_codes": []},
    "safety": {"state": "OVERRIDE", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
    "synthetic_model": True,
}

OVERRIDE_GOOD_RAW = {
    "risk_tier": "LOW",
    "navigation_destination": "NONE",
    "safety_state": "OVERRIDE",
    "summary": "A safety override was triggered for this encounter.",
    "risk_explanation": "This member's modeled risk tier is low.",
    "navigation_explanation": "No proactive navigation applies.",
    "safety_explanation": "A safety override was triggered based on the supplied current information.",
}


def _groq_config(monkeypatch, **overrides):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    monkeypatch.setenv("GENAI_PROVIDER", overrides.get("provider", "groq"))
    monkeypatch.setenv("GROQ_API_KEY", overrides.get("api_key", "test-key-not-real"))
    monkeypatch.setenv("GROQ_MODEL", overrides.get("model", "llama-3.3-70b-versatile"))
    monkeypatch.setenv("OLLAMA_MODEL", overrides.get("ollama_model", "qwen3:8b"))
    monkeypatch.setenv("GENAI_TIMEOUT_SECONDS", str(overrides.get("timeout", 5)))
    return genai_explanation.load_config()


# ---- 1 & 2: Groq success + Groq primary ----

def test_groq_success_returns_valid_ai_explanation(monkeypatch):
    config = _groq_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: dict(GOOD_RAW))

    def _ollama_should_not_be_called(cfg, payload):
        raise AssertionError("Ollama must not be attempted when Groq succeeds")

    monkeypatch.setattr(genai_explanation, "_call_ollama", _ollama_should_not_be_called)

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == "llama-3.3-70b-versatile"
    assert result.generation_time_ms is not None


def test_groq_is_primary_when_provider_is_groq(monkeypatch):
    config = _groq_config(monkeypatch, provider="groq")
    assert config.provider == "groq"
    calls = []
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: (calls.append("groq"), dict(GOOD_RAW))[1])
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert calls == ["groq"]
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == config.groq_model


# ---- 3 & 4: Groq timeout / connection failure -> Ollama fallback ----

def test_groq_timeout_falls_back_to_ollama(monkeypatch):
    config = _groq_config(monkeypatch)

    def _groq_times_out(cfg, payload):
        return None  # _call_groq never raises; a timeout surfaces as None, same as _call_ollama's contract

    monkeypatch.setattr(genai_explanation, "_call_groq", _groq_times_out)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == config.model  # Ollama's model, proving Ollama (not Groq) produced this


def test_groq_connection_failure_falls_back_to_ollama(monkeypatch):
    config = _groq_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: None)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == config.model


# ---- 5: Groq malformed output -> fallback ----

def test_groq_malformed_output_falls_back(monkeypatch):
    config = _groq_config(monkeypatch)

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "{not valid json"}}]}

    monkeypatch.setattr(genai_explanation.httpx, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    # malformed Groq content -> _call_groq returns None -> Ollama fallback succeeds
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == config.model


# ---- 6: Groq safety-validation rejection -> fallback ----

def test_groq_safety_validation_rejection_falls_back_to_ollama(monkeypatch):
    config = _groq_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw["safety_explanation"] = "Everything is fine, no concerns at all."
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: bad_raw)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == config.model  # confirms Groq's bad output was rejected, Ollama's was used


# ---- 7: Groq OVERRIDE contradiction is NEVER displayed ----

def test_groq_override_contradiction_never_displayed(monkeypatch):
    config = _groq_config(monkeypatch)
    bad_raw = {
        "risk_tier": "LOW", "navigation_destination": "NONE", "safety_state": "OVERRIDE",
        "summary": "Everything is fine, no concerns.",
        "risk_explanation": "ok",
        "navigation_explanation": "ok",
        "safety_explanation": "The safety state is clear; proceed as normal.",
    }
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: bad_raw)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: None)

    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    assert "high-acuity/emergency safety rule" in result.safety_explanation
    assert "everything is fine" not in result.summary.lower()


def test_groq_override_correct_text_is_accepted(monkeypatch):
    config = _groq_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: dict(OVERRIDE_GOOD_RAW))
    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert "override" in result.safety_explanation.lower()


# ---- 8: Groq incorrect risk tier is NEVER displayed ----

def test_groq_incorrect_risk_tier_never_displayed(monkeypatch):
    config = _groq_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw["risk_tier"] = "LOW"  # actual tier in GOOD_PAYLOAD is HIGH -- structured echo mismatch
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: bad_raw)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: None)

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    assert "HIGH" in result.risk_explanation


# ---- 9: Groq incorrect navigation is NEVER displayed ----

def test_groq_incorrect_navigation_never_displayed(monkeypatch):
    config = _groq_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw["navigation_destination"] = "URGENT_CARE"  # actual destination is CARE_MANAGEMENT
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: bad_raw)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: None)

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    assert "Care Management" in result.navigation_explanation


# ---- 10: Groq AND Ollama failure -> deterministic fallback ----

def test_groq_and_ollama_failure_falls_back_to_deterministic(monkeypatch):
    config = _groq_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: None)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: None)

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    assert result.model_used is None


# ---- 11: missing GROQ_API_KEY does not crash, falls through cleanly ----

def test_missing_groq_api_key_does_not_crash_and_skips_groq(monkeypatch):
    # Uses the REAL _call_groq (not monkeypatched) to prove its own
    # internal missing-key guard is what skips it -- no network call is
    # even attempted, and the endpoint never crashes or raises.
    config = _groq_config(monkeypatch, api_key="")
    assert config.groq_api_key == ""
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == config.model


def test_call_groq_returns_none_immediately_with_no_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")

    def _post_should_not_be_called(*args, **kwargs):
        raise AssertionError("no network call should be attempted without an API key")

    monkeypatch.setattr(genai_explanation.httpx, "post", _post_should_not_be_called)
    config = genai_explanation.load_config()
    assert genai_explanation._call_groq(config, GOOD_PAYLOAD) is None


# ---- 12: GENAI_ENABLED=false still produces deterministic behavior ----

def test_genai_disabled_produces_deterministic_behavior_even_with_groq_configured(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "false")
    monkeypatch.setenv("GENAI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "some-real-looking-key")
    config = genai_explanation.load_config()

    def _should_not_be_called(cfg, payload):
        raise AssertionError("no provider should be attempted when GENAI_ENABLED=false")

    monkeypatch.setattr(genai_explanation, "_call_groq", _should_not_be_called)
    monkeypatch.setattr(genai_explanation, "_call_ollama", _should_not_be_called)

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 13: API key is never present in logs/output ----

def test_api_key_never_appears_in_result_or_module_source(monkeypatch):
    config = _groq_config(monkeypatch, api_key="sk-super-secret-value-12345")
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: dict(GOOD_RAW))
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)

    import dataclasses
    result_repr = repr(dataclasses.asdict(result))
    assert "sk-super-secret-value-12345" not in result_repr

    import inspect
    source = inspect.getsource(genai_explanation)
    assert "print(" not in source
    assert "import logging" not in source  # this module itself never logs -- see test_genai_privacy.py


def test_on_event_callback_never_receives_api_key_or_payload(monkeypatch):
    config = _groq_config(monkeypatch, api_key="sk-super-secret-value-12345")
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: dict(GOOD_RAW))

    events = []
    genai_explanation.generate_explanation(GOOD_PAYLOAD, config, on_event=events.append)
    assert events  # at least one event was emitted
    for event in events:
        assert isinstance(event, str)
        assert "sk-super-secret-value-12345" not in event
        assert "risk" not in event and "navigation" not in event and "safety" not in event


def test_on_event_receives_expected_groq_then_ollama_sequence(monkeypatch):
    config = _groq_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: None)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))

    events = []
    genai_explanation.generate_explanation(GOOD_PAYLOAD, config, on_event=events.append)
    assert events == ["groq_attempted", "groq_failed", "ollama_attempted", "ollama_succeeded"]


def test_on_event_receives_deterministic_fallback_event(monkeypatch):
    config = _groq_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_groq", lambda cfg, payload: None)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: None)

    events = []
    genai_explanation.generate_explanation(GOOD_PAYLOAD, config, on_event=events.append)
    assert events[-1] == "deterministic_fallback_used"


# ---- provider="ollama" explicitly skips Groq entirely ----

def test_provider_ollama_skips_groq_entirely(monkeypatch):
    config = _groq_config(monkeypatch, provider="ollama")
    assert config.provider == "ollama"

    def _groq_should_not_be_called(cfg, payload):
        raise AssertionError("Groq must not be attempted when GENAI_PROVIDER=ollama")

    monkeypatch.setattr(genai_explanation, "_call_groq", _groq_should_not_be_called)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))

    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == config.model


# ---- /uc07/decide is completely untouched by this integration ----

def test_uc07_decide_endpoint_source_has_no_groq_reference():
    import inspect
    import main
    source = inspect.getsource(main.uc07_decide_endpoint)
    assert "groq" not in source.lower()
    assert "genai_explanation" not in source
