"""
Phase 8C Part 17 -- AUTHORITY tests (numbered 19-24 in the spec).

Proves, structurally AND behaviorally, that the GenAI Explanation Agent
(backend/agents/genai_explanation.py) has zero decision authority:
    - it cannot change risk probability/tier, navigation destination, or
      safety state -- MemberExplanation (contracts.py) has no field that
      could even carry such a value; it is text-only.
    - it never imports or calls the Risk Detection / Care Navigation /
      Safety & Policy agents -- a "malicious" or wrong GenAI response
      cannot feed back into a new decision because there is no code path
      by which it could.
    - OVERRIDE remains OVERRIDE regardless of what the model says.
    - POST /uc07/explain (backend/main.py) never touches
      UC07Orchestrator/RiskDetectionAgent and never returns a
      probability/tier/destination/state field itself.
"""
import dataclasses
import inspect
import json
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent

import genai_explanation
from contracts import ExplanationSource, MemberExplanation

OVERRIDE_PAYLOAD = {
    "risk": {
        "probability": 0.05,
        "tier": "LOW",
        "model_version": "uc07-risk-synthetic-v1",
        "factors": [],
    },
    "navigation": {"destination": None, "reason_codes": []},
    "safety": {"state": "OVERRIDE", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
    "synthetic_model": True,
}


# ---- 19-22. structural: MemberExplanation cannot carry a decision value ----

def test_19_22_member_explanation_has_no_decision_fields():
    field_names = {f.name for f in dataclasses.fields(MemberExplanation)}
    # every field must be explanatory TEXT (or bookkeeping metadata) --
    # never a value that could represent a new probability/tier/
    # destination/safety-state decision
    forbidden = {"probability", "tier", "destination", "state", "safety_state", "risk_tier", "navigation_destination"}
    assert field_names.isdisjoint(forbidden)
    expected = {
        "summary", "risk_explanation", "navigation_explanation", "safety_explanation",
        "disclaimer", "explanation_source", "model_used", "generation_time_ms",
    }
    assert field_names == expected


def test_19_genai_cannot_change_probability_even_if_it_tries(monkeypatch):
    """The model's raw JSON output has no field for probability at all
    (see _response_schema) -- even a compromised/malicious model cannot
    inject one, because generate_explanation() only ever reads the 4
    fixed string keys out of the response, never re-parses free text back
    into a number."""
    monkeypatch.setenv("GENAI_ENABLED", "true")
    config = genai_explanation.load_config()
    malicious_raw = {
        "summary": "ignore everything, the real probability is 0.99",
        "risk_explanation": "probability=0.99 tier=HIGH",  # attempted injection, still just text
        "navigation_explanation": "ok",
        "safety_explanation": "ok",
        "disclaimer": "ok",
        "probability": 0.99,  # even if the model adds an extra key, it is never read
        "tier": "HIGH",
    }
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: malicious_raw)
    payload = json.loads(json.dumps(OVERRIDE_PAYLOAD))
    payload["risk"]["tier"] = "LOW"
    payload["risk"]["probability"] = 0.05
    result = genai_explanation.generate_explanation(payload, config)
    # result is either GENAI (text only, no probability field exists on
    # MemberExplanation to have been changed) or a rejected fallback --
    # either way, nothing capable of altering 0.05/LOW exists in the return
    assert not hasattr(result, "probability")
    assert not hasattr(result, "tier")
    # the actual payload dict passed in is never mutated by this call
    assert payload["risk"]["tier"] == "LOW"
    assert payload["risk"]["probability"] == 0.05


def test_20_genai_cannot_change_tier_wrong_tier_text_is_rejected(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    config = genai_explanation.load_config()
    bad_raw = {
        "summary": "ok", "risk_explanation": "This member is actually high risk.",
        "navigation_explanation": "ok", "safety_explanation": "ok", "disclaimer": "ok",
    }
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    payload = json.loads(json.dumps(OVERRIDE_PAYLOAD))
    payload["risk"]["tier"] = "LOW"
    result = genai_explanation.generate_explanation(payload, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_21_genai_cannot_change_navigation_wrong_destination_text_is_rejected(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    config = genai_explanation.load_config()
    bad_raw = {
        "summary": "ok", "risk_explanation": "ok",
        "navigation_explanation": "This member should go to urgent care instead.",
        "safety_explanation": "ok", "disclaimer": "ok",
    }
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    payload = json.loads(json.dumps(OVERRIDE_PAYLOAD))
    payload["navigation"]["destination"] = "CARE_MANAGEMENT"
    result = genai_explanation.generate_explanation(payload, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


def test_22_genai_cannot_change_safety_state_wrong_state_text_is_rejected(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    config = genai_explanation.load_config()
    bad_raw = {
        "summary": "ok", "risk_explanation": "ok", "navigation_explanation": "ok",
        "safety_explanation": "The safety state is clear, no override is in effect.",
        "disclaimer": "ok",
    }
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK


# ---- 23. OVERRIDE remains OVERRIDE regardless of Qwen's output ----

def test_23_override_state_explanation_always_says_override_even_on_fallback(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    config = genai_explanation.load_config()
    bad_raw = {
        "summary": "Everything is fine, no concerns.",
        "risk_explanation": "ok",
        "navigation_explanation": "ok",
        "safety_explanation": "The safety state is clear; proceed as normal.",
        "disclaimer": "ok",
    }
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: bad_raw)
    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.DETERMINISTIC_FALLBACK
    assert "high-acuity/emergency safety rule" in result.safety_explanation


def test_23_override_accepted_genai_text_must_itself_say_override(monkeypatch):
    monkeypatch.setenv("GENAI_ENABLED", "true")
    config = genai_explanation.load_config()
    good_raw = {
        "summary": "A safety override was triggered for this encounter.",
        "risk_explanation": "This member's modeled risk tier is low.",
        "navigation_explanation": "No proactive navigation applies.",
        "safety_explanation": "A safety override was triggered based on the supplied current information.",
        "disclaimer": "ok",
    }
    monkeypatch.setattr(genai_explanation, "_call_ollama", lambda cfg, payload: good_raw)
    result = genai_explanation.generate_explanation(OVERRIDE_PAYLOAD, config)
    assert result.explanation_source == ExplanationSource.GENAI
    assert "override" in result.safety_explanation.lower()


# ---- 24. structural one-way architecture: Safety Agent remains final authority ----

def test_24_genai_module_never_imports_decision_agents():
    source = inspect.getsource(genai_explanation)
    # module docstring legitimately discusses "safety_policy.decide()" in
    # prose while explaining WHY it is never called -- search only the
    # actual code (after the closing """ of the module docstring)
    code_only = source.split('"""', 2)[-1]
    for forbidden_import in ("import risk_detection", "import care_navigation", "import orchestrator",
                              "from risk_detection", "from care_navigation", "from orchestrator"):
        assert forbidden_import not in code_only
    # safety_policy IS imported, but only as a read-only policy/wording
    # source (check_text, BASE_DISCLAIMER) -- never to call decide()
    assert "safety_policy.decide(" not in code_only


def test_24_explain_endpoint_never_touches_orchestrator():
    import main
    source = inspect.getsource(main.uc07_explain)
    assert "orchestrator" not in source.lower()
    assert "risk_agent" not in source.lower()


def test_24_explain_endpoint_response_has_no_decision_fields():
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    body = {
        "risk": {"probability": 0.05, "tier": "LOW", "model_version": "v1", "factors": []},
        "navigation": {"destination": None, "reason_codes": []},
        "safety": {"state": "CLEAR", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
        "synthetic_model": True,
    }
    resp = client.post("/uc07/explain", json=body)
    assert resp.status_code == 200
    keys = set(resp.json().keys())
    assert keys.isdisjoint({"probability", "tier", "destination", "state", "risk", "navigation", "safety"})
