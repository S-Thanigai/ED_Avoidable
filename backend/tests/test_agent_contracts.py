"""Tests for backend/agents/contracts.py."""
import dataclasses

import pytest

from contracts import (
    CurrentSafetyContext,
    FinalNavigationView,
    FinalUC07Decision,
    NavigationDecision,
    NavigationDestination,
    ReasonCode,
    RiskAssessment,
    RiskTier,
    SafetyDecision,
    SafetyState,
)


def test_current_safety_context_provided_false_when_all_none():
    assert CurrentSafetyContext().provided is False
    assert CurrentSafetyContext(red_flag=None, icu=None, admitted=None, major_procedure=None, triage_level=None).provided is False


@pytest.mark.parametrize("kwargs", [
    {"red_flag": 0}, {"red_flag": 1}, {"icu": 0}, {"admitted": 0},
    {"major_procedure": 0}, {"triage_level": 3},
])
def test_current_safety_context_provided_true_when_any_field_set(kwargs):
    assert CurrentSafetyContext(**kwargs).provided is True


def test_enums_have_expected_members():
    assert {t.value for t in RiskTier} == {"LOW", "MODERATE", "HIGH"}
    assert {d.value for d in NavigationDestination} == {
        "PRIMARY_CARE", "URGENT_CARE", "TELEHEALTH", "CARE_MANAGEMENT", "NO_PROACTIVE_NAVIGATION",
    }
    assert {s.value for s in SafetyState} == {"CLEAR", "CAUTION", "OVERRIDE"}
    expected_reason_codes = {
        "ELEVATED_FUTURE_RISK", "REPEATED_LOWER_ACUITY_HISTORY", "TRANSPORTATION_BARRIER",
        "LIMITED_PCP_ACCESS", "TELEHEALTH_AVAILABLE", "CHRONIC_COMPLEXITY", "PRIOR_CM_ENGAGEMENT",
        "OUTPATIENT_CONTINUITY_OPPORTUNITY",
    }
    assert expected_reason_codes.issubset({r.value for r in ReasonCode})


def test_contracts_are_frozen():
    risk = RiskAssessment(
        member_id="M1", probability=0.1, tier=RiskTier.LOW, contributing_factors=[],
        model_version="v", dataset_id="d", synthetic_model=True,
        index_date=__import__("datetime").date(2026, 1, 1),
        moderate_threshold=0.1, high_threshold=0.2,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        risk.probability = 0.9

    nav = NavigationDecision(member_id="M1", destination=NavigationDestination.PRIMARY_CARE, reason_codes=[], explanation="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        nav.destination = NavigationDestination.TELEHEALTH

    safety = SafetyDecision(state=SafetyState.CLEAR, override=False, message="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        safety.state = SafetyState.OVERRIDE

    context = CurrentSafetyContext()
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.red_flag = 1


def test_final_uc07_decision_shape():
    risk = RiskAssessment(
        member_id="M1", probability=0.5, tier=RiskTier.HIGH, contributing_factors=["x"],
        model_version="uc07-risk-synthetic-v1", dataset_id="synthetic_uc07_v1", synthetic_model=True,
        index_date=__import__("datetime").date(2026, 1, 1), moderate_threshold=0.1, high_threshold=0.2,
    )
    nav_view = FinalNavigationView(destination=NavigationDestination.TELEHEALTH, reason_codes=[ReasonCode.TELEHEALTH_AVAILABLE], explanation="x")
    safety = SafetyDecision(state=SafetyState.CLEAR, override=False, message="x")
    decision = FinalUC07Decision(member_id="M1", risk=risk, navigation=nav_view, safety=safety, disclaimer="d")

    assert decision.member_id == "M1"
    assert isinstance(decision.navigation, FinalNavigationView)
    assert not isinstance(decision.navigation, NavigationDecision)  # never the pre-safety-review type
