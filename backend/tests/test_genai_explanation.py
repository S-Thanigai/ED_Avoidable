"""
Phase 8C Part 17 -- GENAI tests (numbered 9-18 in the spec).

Covers backend/agents/genai_explanation.py. Deliberately does NOT require a
live Ollama instance for the suite to pass: httpx is monkeypatched to
simulate success, unavailability, timeout, malformed output, etc, so these
tests are deterministic and CI-safe. (Live verification against a real,
running Ollama + qwen3:8b is a separate, manual step -- Phase 8C Part 18 --
not part of the automated suite.)
"""
import json

import httpx
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


def _enabled_config(monkeypatch, **overrides):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    monkeypatch.setenv("OLLAMA_BASE_URL", overrides.get("base_url", "http://localhost:11434"))
    monkeypatch.setenv("OLLAMA_MODEL", overrides.get("model", "qwen3:8b"))
    monkeypatch.setenv("GENAI_TIMEOUT_SECONDS", str(overrides.get("timeout", 5)))
    return genai_explanation.load_config()


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json_body = json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json_body


# ---- 9. success path ----

def test_9_success_path_returns_genai_source(monkeypatch):
    config = _enabled_config(monkeypatch)
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: dict(GOOD_RAW))
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert result.model_used == "qwen3:8b"
    assert result.generation_time_ms is not None
    # the model is never even asked for a disclaimer field anymore -- the
    # centrally governed one is always used
    import safety_policy
    assert result.disclaimer == safety_policy.BASE_DISCLAIMER


# ---- 10. Ollama unavailable (connection refused) ----

def test_10_ollama_unavailable_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)

    def _raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(genai_explanation.httpx, "post", _raise_connect_error)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 11. timeout ----

def test_11_timeout_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)

    def _raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(genai_explanation.httpx, "post", _raise_timeout)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 12. malformed JSON in the model's content field ----

def test_12_malformed_json_content_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)

    def _fake_post(url, json, timeout):
        return _FakeResponse(200, {"message": {"content": "{not valid json"}})

    monkeypatch.setattr(genai_explanation.httpx, "post", _fake_post)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 13. empty response ----

def test_13_empty_response_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)

    def _fake_post(url, json, timeout):
        return _FakeResponse(200, {"message": {"content": ""}})

    monkeypatch.setattr(genai_explanation.httpx, "post", _fake_post)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_13_missing_message_key_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)

    def _fake_post(url, json, timeout):
        return _FakeResponse(200, {"unexpected": "shape"})

    monkeypatch.setattr(genai_explanation.httpx, "post", _fake_post)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 14. prohibited language ----

@pytest.mark.parametrize("field", ["risk_explanation", "navigation_explanation", "safety_explanation", "summary"])
def test_14_prohibited_language_falls_back(monkeypatch, field):
    config = _enabled_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw[field] = "You should avoid the ED and skip the er entirely."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_14_diagnosis_claim_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw["safety_explanation"] = "This confirms the member is diagnosed with a chronic condition."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_14_negated_diagnosis_language_is_allowed(monkeypatch):
    # "does not diagnose" is the SAFE, correct disclaimer framing -- must
    # not be treated as a violation (regression guard for the false
    # positive found during live testing).
    config = _enabled_config(monkeypatch)
    ok_raw = dict(GOOD_RAW)
    ok_raw["safety_explanation"] = (
        "Current information is incomplete; this tool does not diagnose any condition, it only summarizes the safety state."
    )
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: ok_raw)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI


# ---- 15. unsupported / inconsistent output (wrong tier, destination, safety state) ----

def test_15_wrong_tier_claim_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw["risk_explanation"] = "This member has a low risk of future ED use."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_15_wrong_destination_claim_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw["navigation_explanation"] = "We recommend urgent care for this member."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_15_wrong_safety_state_claim_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    bad_raw = dict(GOOD_RAW)
    bad_raw["safety_explanation"] = "The safety state is clear, no override is needed."
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_15_reason_code_justified_destination_mention_is_allowed(monkeypatch):
    # A faithful paraphrase of a reason code that legitimately mentions a
    # different destination's terminology as CONTEXT (not a recommendation
    # change) must not be treated as a violation.
    config = _enabled_config(monkeypatch)
    payload = dict(GOOD_PAYLOAD)
    payload["navigation"] = {"destination": "CARE_MANAGEMENT", "reason_codes": ["URGENT_CARE_ACCESS_ADVANTAGE"]}
    ok_raw = dict(GOOD_RAW)
    ok_raw["navigation_explanation"] = (
        "The suggested navigation destination is care management, noting an urgent care access advantage as context."
    )
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: ok_raw)
    result = genai_explanation.generate_explanation(payload, config)
    assert result.explanation_source == ExplanationSource.GENAI


# ---- 16. GenAI disabled ----

def test_16_genai_disabled_never_calls_ollama(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "false")
    config = genai_explanation.load_config()
    assert config.enabled is False

    called = {"count": 0}

    def _spy(*args, **kwargs):
        called["count"] += 1
        return None

    monkeypatch.setattr(genai_explanation, "_call_ollama", _spy)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    assert called["count"] == 0


# ---- 17. wrong-shaped / unexpected model response ----

def test_17_response_missing_required_key_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    incomplete_raw = {k: v for k, v in GOOD_RAW.items() if k != "safety_explanation"}
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: incomplete_raw)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_17_response_with_wrong_field_types_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)
    wrong_types = dict(GOOD_RAW)
    wrong_types["summary"] = 12345  # not a string
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: wrong_types)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_17_response_that_is_not_a_json_object_falls_back(monkeypatch):
    config = _enabled_config(monkeypatch)

    def _fake_post(url, json, timeout):
        return _FakeResponse(200, {"message": {"content": json_module_dumps_list()}})

    def json_module_dumps_list():
        return json.dumps(["not", "an", "object"])

    monkeypatch.setattr(genai_explanation.httpx, "post", _fake_post)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 18. deterministic fallback generator itself ----

def test_18_deterministic_fallback_never_calls_ollama_when_used_directly():
    result = genai_explanation._deterministic_explanation(GOOD_PAYLOAD)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    assert "HIGH" in result.risk_explanation
    assert "Care Management" in result.navigation_explanation
    assert result.safety_explanation == (
        "Current safety information is absent or incomplete, so the system cannot confirm the safety-rule state fully."
    )
    assert result.model_used is None
    assert result.generation_time_ms is None


def test_18_deterministic_fallback_covers_all_three_safety_states():
    for state, expected_substring in [
        ("CLEAR", "no configured safety override was triggered"),
        ("CAUTION", "cannot confirm the safety-rule state fully"),
        ("OVERRIDE", "triggered a configured high-acuity/emergency safety rule"),
    ]:
        payload = json.loads(json.dumps(GOOD_PAYLOAD))  # deep copy
        payload["safety"]["state"] = state
        result = genai_explanation._deterministic_explanation(payload)
        assert expected_substring in result.safety_explanation


def test_18_deterministic_fallback_handles_no_navigation_destination():
    payload = json.loads(json.dumps(GOOD_PAYLOAD))
    payload["navigation"] = {"destination": None, "reason_codes": []}
    result = genai_explanation._deterministic_explanation(payload)
    assert "No proactive navigation destination" in result.navigation_explanation


def test_18_deterministic_fallback_uses_the_centralized_disclaimer():
    import safety_policy

    result = genai_explanation._deterministic_explanation(GOOD_PAYLOAD)
    assert result.disclaimer == safety_policy.BASE_DISCLAIMER


def test_generate_explanation_never_raises_on_unexpected_exception(monkeypatch):
    config = _enabled_config(monkeypatch)

    def _boom(cfg, payload):
        raise RuntimeError("something totally unexpected")

    monkeypatch.setattr(genai_explanation, "_call_ollama", _boom)
    result = genai_explanation.generate_explanation(GOOD_PAYLOAD, config)  # must not raise
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
