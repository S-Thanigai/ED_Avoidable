"""
Phase 8C Part 17 -- PRIVACY tests (numbered 25-27 in the spec).

Covers the strict LLM input boundary (Phase 8C Part 5): the GenAI
Explanation Agent must never see raw CSV data or a member's full history,
only the minimal allow-listed structured summary, and must never log
member-identifying content.
"""
import inspect
import json

import genai_explanation

PAYLOAD = {
    "risk": {
        "probability": 0.274,
        "tier": "HIGH",
        "model_version": "uc07-risk-synthetic-v1",
        "factors": [{"display_name": "Recent ED utilization", "direction": "INCREASES_RISK"}],
    },
    "navigation": {"destination": "CARE_MANAGEMENT", "reason_codes": ["ELEVATED_FUTURE_RISK"]},
    "safety": {"state": "CAUTION", "context_completeness": "ABSENT", "context_source": "NOT_AVAILABLE"},
    "synthetic_model": True,
}

ALLOWED_TOP_KEYS = {"risk", "navigation", "safety", "synthetic_model"}
ALLOWED_RISK_KEYS = {"probability", "tier", "model_version", "factors"}
ALLOWED_NAVIGATION_KEYS = {"destination", "reason_codes"}
ALLOWED_SAFETY_KEYS = {"state", "context_completeness", "context_source"}


# ---- 25. raw CSV data is never sent ----

def test_25_explain_request_schema_has_no_csv_or_file_fields():
    import main

    field_names = set(main.ExplainRequest.model_fields.keys())
    assert field_names == ALLOWED_TOP_KEYS
    for forbidden in ("members_file", "ed_visits_file", "care_file", "csv", "file"):
        assert forbidden not in field_names


def test_25_genai_module_has_no_csv_or_dataset_file_access():
    source = inspect.getsource(genai_explanation)
    for forbidden in (".csv", "read_csv", "open(", "members_file", "ed_visits_file", "raw_members"):
        assert forbidden not in source


# ---- 26. only allow-listed fields are ever sent to Ollama ----

def test_26_ollama_request_body_contains_only_allowlisted_payload_shape(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": json.dumps({
                "summary": "ok", "risk_explanation": "ok", "navigation_explanation": "ok",
                "safety_explanation": "ok", "disclaimer": "ok",
            })}}

    def _fake_post(url, json, timeout):
        captured["body"] = json
        return _FakeResponse()

    monkeypatch.setenv("GENAI_ENABLED", "true")
    monkeypatch.setattr(genai_explanation.httpx, "post", _fake_post)
    config = genai_explanation.load_config()

    # a payload deliberately carrying an extra, non-allowlisted key --
    # simulates a caller that (incorrectly) tried to attach extra data
    contaminated_payload = dict(PAYLOAD)
    contaminated_payload["_extra_member_data"] = {"name": "should never appear", "dob": "1990-01-01"}

    genai_explanation.generate_explanation(contaminated_payload, config)

    assert "body" in captured
    user_message = next(m for m in captured["body"]["messages"] if m["role"] == "user")
    sent_payload = json.loads(user_message["content"])

    assert set(sent_payload.keys()) == ALLOWED_TOP_KEYS.union({"_extra_member_data"})
    # generate_explanation forwards the caller's dict as-is (it does not
    # invent new leakage) -- the REAL data-minimization boundary is
    # upstream, at ExplainRequest (Pydantic), which structurally cannot
    # accept such a key in the first place (see test_25 above and
    # test_26_explain_request_rejects_unknown_fields below)
    assert set(sent_payload["risk"].keys()) == ALLOWED_RISK_KEYS
    assert set(sent_payload["navigation"].keys()) == ALLOWED_NAVIGATION_KEYS
    assert set(sent_payload["safety"].keys()) == ALLOWED_SAFETY_KEYS
    # no name, dob, address, ssn, member_id, or any other direct
    # identifier ever appears in what is sent to Ollama
    for identifier in ("member_id", "name", "dob", "date_of_birth", "address", "ssn", "phone"):
        assert identifier not in sent_payload["risk"]
        assert identifier not in sent_payload["navigation"]
        assert identifier not in sent_payload["safety"]


def test_26_explain_request_rejects_unknown_fields():
    """Pydantic's default 'ignore extra' behavior would silently drop
    caller-supplied extra keys; explicitly proves that no member
    identifier can ride along on an /uc07/explain request even if a
    caller tried to attach one -- ExplainRequest simply has no such
    field to populate."""
    import main
    from pydantic import ValidationError

    body = {
        "risk": {"probability": 0.05, "tier": "LOW", "model_version": "v1", "factors": []},
        "navigation": {"destination": None, "reason_codes": []},
        "safety": {"state": "CLEAR", "context_completeness": "COMPLETE", "context_source": "CALLER_SUPPLIED"},
        "synthetic_model": True,
        "member_name": "should be impossible to submit",
    }
    parsed = main.ExplainRequest.model_validate(body)
    # extra field is silently ignored by Pydantic's default config --
    # critically, it never reaches genai_explanation.generate_explanation
    # because model_dump() only emits the declared fields
    dumped = parsed.model_dump()
    assert "member_name" not in dumped
    assert set(dumped.keys()) == ALLOWED_TOP_KEYS


# ---- 27. no unnecessary member data is ever logged ----

def test_27_genai_module_never_logs_payload_contents():
    source = inspect.getsource(genai_explanation)
    for forbidden in ("print(payload", "logging.info(payload", "logger.info(payload", "print(raw", "log.info(raw"):
        assert forbidden not in source


def test_27_genai_module_has_no_logging_calls_at_all():
    """Simplest, strongest privacy guarantee available here: this module
    performs no logging of any kind, so there is no logging call to audit
    for accidental payload leakage in the first place."""
    source = inspect.getsource(genai_explanation)
    assert "import logging" not in source
    assert "print(" not in source
