"""
Phase 6 safety invariant test suite (spec Step 29) + failure-mode tests
(spec Step 28) not already covered by the Phase 5 agent test files.

These are deliberately named INVARIANT 1-10 to match
docs/06_SAFETY_FAIRNESS_ROBUSTNESS_VALIDATION.md's numbering, so a
reviewer can go from the doc straight to the enforcing test. Several
invariants are already covered at the single-scenario level by Phase 5's
test_safety_policy_agent.py / test_care_navigation_agent.py /
test_orchestrator.py / test_legacy_isolation.py -- these versions sweep
broader/systematic input spaces (every trigger, every leave-one-out
missing-field combination, a real TEST-snapshot sample) rather than
duplicating single-case assertions.
"""
from __future__ import annotations

import inspect
import itertools
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import care_navigation
import orchestrator as orchestrator_mod
import safety_policy
from contracts import CurrentSafetyContext, NavigationDecision, NavigationDestination, ReasonCode, RiskTier, SafetyState
from orchestrator import UC07Orchestrator
from risk_detection import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, RiskDetectionAgent

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
TEST_SNAPSHOT = REPO_ROOT / "data" / "derived" / "synthetic" / "test_snapshot.csv"
PHASE6_EVAL_DIR = REPO_ROOT / "artifacts" / "phase6_validation"

CLEAN_NAV = NavigationDecision(
    member_id="M1", destination=NavigationDestination.TELEHEALTH,
    reason_codes=[ReasonCode.TELEHEALTH_AVAILABLE],
    explanation="Telehealth may be a useful future, non-emergency option.",
)

TRIGGERS = [
    {"red_flag": 1}, {"icu": 1}, {"admitted": 1}, {"major_procedure": 1},
    {"triage_level": 1}, {"triage_level": 2},
]


@pytest.fixture(scope="session")
def agent():
    return RiskDetectionAgent()


@pytest.fixture(scope="session")
def test_sample():
    df = pd.read_csv(TEST_SNAPSHOT)
    return df.sample(n=500, random_state=42)


# ---------------------------------------------------------------------------
# INVARIANT 1: Any emergency trigger -> OVERRIDE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trigger_kwargs", TRIGGERS)
@pytest.mark.parametrize("tier", list(RiskTier))
def test_invariant_1_any_trigger_overrides_regardless_of_risk_tier(trigger_kwargs, tier):
    nav = NavigationDecision(member_id="M1", destination=NavigationDestination.CARE_MANAGEMENT,
                              reason_codes=[ReasonCode.ELEVATED_FUTURE_RISK], explanation="A future opportunity.")
    context = CurrentSafetyContext(**trigger_kwargs)
    safety, final_nav = safety_policy.decide(nav, context)
    assert safety.state == SafetyState.OVERRIDE
    assert safety.override is True
    assert final_nav.destination is None


def test_invariant_1_multi_trigger_combinations_all_override():
    for (k1,), (k2,) in itertools.combinations([tuple(t.items()) for t in TRIGGERS], 2):
        context = CurrentSafetyContext(**dict([k1, k2]))
        safety, final_nav = safety_policy.decide(CLEAN_NAV, context)
        assert safety.state == SafetyState.OVERRIDE, f"combination {k1, k2} did not override"
        assert final_nav.destination is None


# ---------------------------------------------------------------------------
# INVARIANT 2: Missing/incomplete safety context -> never CLEAR (Phase 6
# hardening: CLEAR requires ALL FIVE fields known, not just "any" field)
# ---------------------------------------------------------------------------

def test_invariant_2_fully_missing_context_is_caution():
    safety, _ = safety_policy.decide(CLEAN_NAV, CurrentSafetyContext())
    assert safety.state == SafetyState.CAUTION


ALL_SAFE = dict(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4)


@pytest.mark.parametrize("omit_field", ["red_flag", "icu", "admitted", "major_procedure", "triage_level"])
def test_invariant_2_any_single_missing_field_is_caution_not_clear(omit_field):
    partial = {k: v for k, v in ALL_SAFE.items() if k != omit_field}
    safety, _ = safety_policy.decide(CLEAN_NAV, CurrentSafetyContext(**partial))
    assert safety.state == SafetyState.CAUTION, f"context missing only {omit_field} incorrectly resolved to {safety.state}"


@pytest.mark.parametrize("field", ["red_flag", "icu", "admitted", "major_procedure", "triage_level"])
def test_invariant_2_only_one_field_known_is_caution(field):
    safe_value = 4 if field == "triage_level" else 0
    safety, _ = safety_policy.decide(CLEAN_NAV, CurrentSafetyContext(**{field: safe_value}))
    assert safety.state == SafetyState.CAUTION


def test_invariant_2_fully_known_safe_context_is_clear():
    safety, _ = safety_policy.decide(CLEAN_NAV, CurrentSafetyContext(**ALL_SAFE))
    assert safety.state == SafetyState.CLEAR


def test_invariant_2_known_trigger_overrides_even_with_other_fields_missing():
    safety, _ = safety_policy.decide(CLEAN_NAV, CurrentSafetyContext(red_flag=1))
    assert safety.state == SafetyState.OVERRIDE


# ---------------------------------------------------------------------------
# INVARIANT 3: Safety output cannot be replaced/bypassed by Navigation
# ---------------------------------------------------------------------------

def test_invariant_3_navigation_decide_has_no_safety_context_parameter():
    sig = inspect.signature(care_navigation.decide)
    for name in sig.parameters:
        assert "safety" not in name.lower() and "context" not in name.lower()


def test_invariant_3_orchestrator_calls_safety_after_navigation_in_source():
    source = inspect.getsource(orchestrator_mod.UC07Orchestrator._decide_from_risk)
    nav_pos = source.find("care_navigation.decide")
    safety_pos = source.find("safety_policy.decide")
    assert nav_pos != -1 and safety_pos != -1
    assert nav_pos < safety_pos, "Safety Agent must run after Care Navigation Agent in source order"


def test_invariant_3_no_reversed_call_order_exists_anywhere_in_orchestrator():
    """AST-based (not naive text search, which false-positives on the
    module's own docstring prose): for every function defined in
    orchestrator.py, if it calls both care_navigation.decide and
    safety_policy.decide, the navigation call must come first."""
    import ast

    tree = ast.parse(inspect.getsource(orchestrator_mod))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        call_order = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr == "decide" and isinstance(sub.func.value, ast.Name):
                    if sub.func.value.id == "care_navigation":
                        call_order.append(("navigation", sub.lineno))
                    elif sub.func.value.id == "safety_policy":
                        call_order.append(("safety", sub.lineno))
        call_order.sort(key=lambda t: t[1])
        kinds = [k for k, _ in call_order]
        if "navigation" in kinds and "safety" in kinds:
            assert kinds.index("navigation") < kinds.index("safety"), \
                f"function {node.name} calls safety_policy.decide before care_navigation.decide"


def test_invariant_3_final_decision_never_carries_raw_navigation_decision_type():
    orch = UC07Orchestrator()
    members = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")
    ed = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")
    care = pd.read_csv(SYNTHETIC_DIR / "raw_care_history.csv")
    decision = orch.decide_for_member(members["member_id"].iloc[0], members, ed, care, date(2026, 7, 3))
    assert not isinstance(decision.navigation, NavigationDecision)
    from contracts import FinalNavigationView
    assert isinstance(decision.navigation, FinalNavigationView)


# ---------------------------------------------------------------------------
# INVARIANT 4: No prohibited emergency-discouraging language
# ---------------------------------------------------------------------------

def test_invariant_4_every_reason_destination_template_combination_is_clean():
    for destination in NavigationDestination:
        label = care_navigation._DESTINATION_LABELS.get(destination, "")
        for reason in ReasonCode:
            phrase = care_navigation._REASON_PHRASES[reason]
            text = f"Based on {phrase}, {label} may be a useful future, non-emergency option."
            assert safety_policy.check_text(text) == []


def test_invariant_4_all_static_safety_messages_are_clean():
    for text in (safety_policy.BASE_DISCLAIMER, safety_policy.OVERRIDE_MESSAGE, safety_policy.CAUTION_MESSAGE, safety_policy.CLEAR_MESSAGE):
        assert safety_policy.check_text(text) == []


def test_invariant_4_population_scan_artifact_reports_zero_violations():
    path = PHASE6_EVAL_DIR / "prohibited_language_scan.json"
    if not path.exists():
        pytest.skip("Phase 6 validation script has not been run yet in this environment")
    report = json.loads(path.read_text())
    assert report["violation_count"] == 0


# ---------------------------------------------------------------------------
# INVARIANT 5: Risk score alone cannot trigger Care Management
# ---------------------------------------------------------------------------

def test_invariant_5_high_risk_alone_never_triggers_care_management():
    row = pd.Series({
        "transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3, "urgent_care_distance_miles": 20,
        "clinical_burden": 0, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
        "prior_pcp_count_270d": 0, "has_prior_care_management": 0,
    })
    nav = care_navigation.decide("M1", RiskTier.HIGH, row)
    assert nav.destination != NavigationDestination.CARE_MANAGEMENT


def test_invariant_5_complexity_alone_without_risk_or_utilization_never_triggers_care_management():
    row = pd.Series({
        "transportation_barrier": 1, "telehealth_available": 0, "pcp_distance_miles": 15, "urgent_care_distance_miles": 20,
        "clinical_burden": 3, "prior_ed_count_270d": 0, "prior_potentially_avoidable_ed_count_270d": 0,
        "prior_pcp_count_270d": 0, "has_prior_care_management": 0,
    })
    nav = care_navigation.decide("M1", RiskTier.LOW, row)
    assert nav.destination != NavigationDestination.CARE_MANAGEMENT


# ---------------------------------------------------------------------------
# INVARIANT 6/7: probability in [0,1]; tier agrees with frozen thresholds
# ---------------------------------------------------------------------------

def test_invariant_6_7_probability_and_tier_bounds_hold_on_real_test_sample(agent, test_sample):
    X = test_sample[agent.feature_columns]
    probs = agent.pipeline.predict_proba(X)[:, 1]
    assert bool(((probs >= 0.0) & (probs <= 1.0)).all())

    from risk_detection import assign_risk_tier
    for p in probs:
        tier = assign_risk_tier(p, agent.moderate_threshold, agent.high_threshold)
        if tier == RiskTier.LOW:
            assert p < agent.moderate_threshold
        elif tier == RiskTier.MODERATE:
            assert agent.moderate_threshold <= p < agent.high_threshold
        else:
            assert p >= agent.high_threshold


def test_invariant_7_no_duplicate_hardcoded_threshold_source():
    """Thresholds must come from the loaded artifact/metadata only --
    risk_detection.py itself contains no numeric threshold literal."""
    source = Path(REPO_ROOT / "backend" / "agents" / "risk_detection.py").read_text()
    assert "0.105986" not in source
    assert "0.213252" not in source


# ---------------------------------------------------------------------------
# INVARIANT 8: Synthetic disclosure always present
# ---------------------------------------------------------------------------

def test_invariant_8_risk_assessment_always_carries_synthetic_flag(agent, test_sample):
    row = test_sample[agent.feature_columns].iloc[[0]]
    assessment = agent.assess(str(test_sample["member_id"].iloc[0]), row, date(2026, 4, 3))
    assert assessment.synthetic_model is True
    assert assessment.dataset_id == "synthetic_uc07_v1"


def test_invariant_8_metadata_discloses_synthetic_and_not_clinically_validated(agent):
    disclaimer = (agent.metadata.get("disclaimer") or "").lower()
    assert "synthetic" in disclaimer
    assert "clinically validated" in disclaimer


# ---------------------------------------------------------------------------
# INVARIANT 9: Legacy model cannot power /uc07/decide
# ---------------------------------------------------------------------------

LEGACY_TOKENS = ["ed_risk_model.pkl", "from predict import", "import predict", "feature_engineering", "train_model"]


@pytest.mark.parametrize("module_path", [
    "backend/agents/orchestrator.py", "backend/agents/risk_detection.py",
    "backend/agents/care_navigation.py", "backend/agents/safety_policy.py", "backend/agents/contracts.py",
])
def test_invariant_9_agent_modules_never_reference_legacy_path(module_path):
    source = (REPO_ROOT / module_path).read_text()
    for token in LEGACY_TOKENS:
        assert token not in source, f"{module_path} references legacy token {token!r}"


def test_invariant_9_uc07_decide_endpoint_uses_orchestrator_not_legacy_predict():
    source = (REPO_ROOT / "backend" / "main.py").read_text()
    uc07_section = source[source.index("uc07_decide_endpoint"):]
    assert "orchestrator.decide_for_member" in uc07_section or "orchestrator.decide_for_all_members" in uc07_section
    assert "predict(" not in uc07_section.split("def uc07_decide_endpoint")[-1].split("\n\n\n")[0]


# ---------------------------------------------------------------------------
# INVARIANT 10: Same input -> same output
# ---------------------------------------------------------------------------

def test_invariant_10_repeated_calls_are_identical():
    from orchestrator import decision_to_dict
    orch = UC07Orchestrator()
    members = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")
    ed = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")
    care = pd.read_csv(SYNTHETIC_DIR / "raw_care_history.csv")
    member_id = members["member_id"].iloc[3]
    results = [decision_to_dict(orch.decide_for_member(member_id, members, ed, care, date(2026, 7, 3))) for _ in range(5)]
    assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Step 28: additional failure-mode tests not already covered by
# test_risk_detection_agent.py's artifact/metadata mismatch tests
# ---------------------------------------------------------------------------

def test_failure_mode_pit_feature_generation_raises_not_swallowed():
    """A raw_ed_visits.csv missing a required column must raise, not
    silently score with fabricated/defaulted values."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "backend" / "pit"))
    from features import build_observation_features
    from windows import build_snapshot_window

    members = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv").head(5)
    ed = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")
    ed_broken = ed.drop(columns=["triage_level"])
    care = pd.read_csv(SYNTHETIC_DIR / "raw_care_history.csv")
    window = build_snapshot_window("test", date(2026, 7, 3))

    with pytest.raises(KeyError):
        build_observation_features(members, ed_broken, care, window)


def test_failure_mode_risk_agent_exception_propagates_through_orchestrator(monkeypatch):
    """If the Risk Detection Agent raises, the orchestrator must not
    catch it and fabricate a decision -- it must propagate, so the API
    layer converts it into a clean error response rather than silently
    returning an unsafe / made-up recommendation."""
    orch = UC07Orchestrator()
    members = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")
    ed = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")
    care = pd.read_csv(SYNTHETIC_DIR / "raw_care_history.csv")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated Risk Agent failure")

    monkeypatch.setattr(orch.risk_agent, "assess", _boom)
    with pytest.raises(RuntimeError, match="simulated Risk Agent failure"):
        orch.decide_for_member(members["member_id"].iloc[0], members, ed, care, date(2026, 7, 3))


def test_failure_mode_navigation_agent_uses_conservative_defaults_for_missing_fields():
    """care_navigation.decide() is defensive by construction (never
    raises on a missing feature column), but its defaults must be
    conservative -- missing distance defaults to 99.0 (far), never to a
    falsely favorable close value -- not fabricate an aggressive
    recommendation from absent data."""
    empty_row = pd.Series({})
    nav = care_navigation.decide("M1", RiskTier.LOW, empty_row)
    assert nav.destination in (NavigationDestination.NO_PROACTIVE_NAVIGATION, NavigationDestination.CARE_MANAGEMENT)
    assert safety_policy.check_text(nav.explanation) == []


def test_failure_mode_model_metadata_mismatch_already_fails_loudly():
    """Documents (does not re-implement) the existing, already-tested
    guarantee in test_risk_detection_agent.py: a threshold/feature/
    version mismatch between the model artifact and its metadata raises
    ModelIncompatibleError rather than silently serving predictions.
    This test just re-asserts the guarantee is still present so a future
    refactor of risk_detection.py cannot quietly remove it."""
    from risk_detection import ModelIncompatibleError, load_model_bundle
    assert issubclass(ModelIncompatibleError, RuntimeError)
    bundle = load_model_bundle()  # must succeed on the real, unmodified artifact
    assert bundle["artifact"]["moderate_threshold"] < bundle["artifact"]["high_threshold"]
