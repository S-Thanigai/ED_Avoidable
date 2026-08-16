"""
Tests for backend/agents/care_navigation.py (AGENT 2).

Proves all 4 named destinations plus NO_PROACTIVE_NAVIGATION are
reachable through legitimate, crafted feature rows, that this agent has
no authority to mark anything CLEAR/CAUTION/OVERRIDE (structural check),
and that its explanations never contain prohibited language.
"""
import inspect

import pandas as pd
import pytest

import care_navigation
import safety_policy
from contracts import NavigationDestination, RiskTier

BASE_ROW = {
    "transportation_barrier": 0, "telehealth_available": 0,
    "pcp_distance_miles": 3.0, "urgent_care_distance_miles": 3.0,
    "clinical_burden": 0, "prior_ed_count_270d": 0,
    "prior_potentially_avoidable_ed_count_270d": 0,
    "prior_pcp_count_270d": 0, "has_prior_care_management": 0,
}


def _row(**overrides):
    data = {**BASE_ROW, **overrides}
    return pd.Series(data)


# ---- structural: this agent has no safety authority ----

def test_decide_signature_has_no_safety_context_parameter():
    sig = inspect.signature(care_navigation.decide)
    for name in sig.parameters:
        assert "safety" not in name.lower() and "context" not in name.lower(), (
            f"care_navigation.decide must never accept a safety/context parameter (found {name})"
        )


def test_care_navigation_module_never_imports_safety_module():
    """The module docstring legitimately explains, in prose, that this
    agent must not use CurrentSafetyContext -- check for an actual import
    statement, not any textual mention of the name."""
    source = inspect.getsource(care_navigation)
    assert "import safety_policy" not in source
    assert "from safety_policy" not in source
    assert "from contracts import" in source  # sanity: it does import contracts
    assert "CurrentSafetyContext" not in source.split('"""', 2)[-1]  # not referenced outside the module docstring


# ---- each destination reachable ----

def test_care_management_reachable():
    row = _row(clinical_burden=3, transportation_barrier=1, prior_potentially_avoidable_ed_count_270d=2)
    decision = care_navigation.decide("M1", RiskTier.HIGH, row)
    assert decision.destination == NavigationDestination.CARE_MANAGEMENT
    assert len(decision.reason_codes) > 0


def test_care_management_not_triggered_by_high_risk_alone():
    """HIGH risk with NO complexity/access/repeated-utilization signal
    must NOT route to Care Management (Phase 5 spec: never CM on risk alone)."""
    row = _row()  # no barriers, no burden, no history
    decision = care_navigation.decide("M1", RiskTier.HIGH, row)
    assert decision.destination != NavigationDestination.CARE_MANAGEMENT


def test_telehealth_reachable():
    # urgent_care_distance > 10 trips the telehealth access-barrier check
    # without tripping Care Management's complexity signal (which only
    # looks at transportation_barrier / pcp_distance / prior CM), so this
    # row cleanly isolates the TELEHEALTH branch.
    row = _row(telehealth_available=1, pcp_distance_miles=3.0, urgent_care_distance_miles=15.0)
    decision = care_navigation.decide("M1", RiskTier.MODERATE, row)
    assert decision.destination == NavigationDestination.TELEHEALTH


def test_urgent_care_reachable():
    # pcp_distance and urgent_distance both <= 10 (so CM's distance-based
    # complexity signal never trips), urgent strictly closer than pcp.
    row = _row(pcp_distance_miles=8.0, urgent_care_distance_miles=3.0, prior_ed_count_270d=2)
    decision = care_navigation.decide("M1", RiskTier.MODERATE, row)
    assert decision.destination == NavigationDestination.URGENT_CARE


def test_primary_care_reachable():
    row = _row(pcp_distance_miles=4.0, urgent_care_distance_miles=4.0, prior_pcp_count_270d=2, prior_ed_count_270d=1)
    decision = care_navigation.decide("M1", RiskTier.MODERATE, row)
    assert decision.destination == NavigationDestination.PRIMARY_CARE


def test_no_proactive_navigation_reachable():
    row = _row()  # LOW risk, no barriers, no history
    decision = care_navigation.decide("M1", RiskTier.LOW, row)
    assert decision.destination == NavigationDestination.NO_PROACTIVE_NAVIGATION
    assert decision.reason_codes  # still has a reason code explaining "no opportunity"


# ---- language safety (defense in depth -- Safety Agent double-checks this too) ----

@pytest.mark.parametrize("destination_case", [
    ("care_management", dict(clinical_burden=3, transportation_barrier=1, prior_potentially_avoidable_ed_count_270d=2), RiskTier.HIGH),
    ("telehealth", dict(telehealth_available=1, pcp_distance_miles=3.0, urgent_care_distance_miles=15.0), RiskTier.MODERATE),
    ("urgent_care", dict(pcp_distance_miles=8.0, urgent_care_distance_miles=3.0, prior_ed_count_270d=2), RiskTier.MODERATE),
    ("primary_care", dict(pcp_distance_miles=4.0, prior_pcp_count_270d=2, prior_ed_count_270d=1), RiskTier.MODERATE),
    ("no_navigation", {}, RiskTier.LOW),
])
def test_explanations_never_contain_prohibited_language(destination_case):
    _, overrides, tier = destination_case
    row = _row(**overrides)
    decision = care_navigation.decide("M1", tier, row)
    assert safety_policy.check_text(decision.explanation) == []


def test_reason_codes_are_reason_code_enum_members():
    row = _row(clinical_burden=3, transportation_barrier=1, prior_potentially_avoidable_ed_count_270d=2)
    decision = care_navigation.decide("M1", RiskTier.HIGH, row)
    from contracts import ReasonCode
    for code in decision.reason_codes:
        assert isinstance(code, ReasonCode)


def test_navigation_decision_deterministic():
    row = _row(telehealth_available=1, transportation_barrier=1)
    d1 = care_navigation.decide("M1", RiskTier.MODERATE, row)
    d2 = care_navigation.decide("M1", RiskTier.MODERATE, row)
    assert d1.destination == d2.destination
    assert d1.reason_codes == d2.reason_codes
    assert d1.explanation == d2.explanation
