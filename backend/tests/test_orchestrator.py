"""
Tests for backend/agents/orchestrator.py -- proves the fixed agent call
order, that the Safety & Policy Agent is always final and non-bypassable,
that risk/navigation agents have no authority outside their own contract,
determinism, and correct serialization.
"""
import inspect
from datetime import date

import pandas as pd
import pytest

import care_navigation
import safety_policy
from contracts import CurrentSafetyContext, FinalNavigationView, NavigationDecision, RiskTier, SafetyState
from orchestrator import UC07Orchestrator, decision_to_dict

SYNTHETIC_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"


@pytest.fixture(scope="module")
def synthetic_data():
    members = pd.read_csv(SYNTHETIC_DIR / "raw_members.csv")
    ed = pd.read_csv(SYNTHETIC_DIR / "raw_ed_visits.csv")
    care = pd.read_csv(SYNTHETIC_DIR / "raw_care_history.csv")
    return members, ed, care


@pytest.fixture(scope="module")
def orchestrator():
    return UC07Orchestrator()


INDEX_DATE = date(2026, 7, 3)


# ---- basic end-to-end shape ----

def test_decide_for_member_returns_final_decision(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    member_id = members["member_id"].iloc[0]
    decision = orchestrator.decide_for_member(member_id, members, ed, care, INDEX_DATE)
    assert decision.member_id == member_id
    assert isinstance(decision.navigation, FinalNavigationView)
    assert decision.disclaimer


def test_decide_for_all_members_returns_one_per_member(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    small_members = members.head(10)
    decisions = orchestrator.decide_for_all_members(small_members, ed, care, INDEX_DATE)
    assert len(decisions) == 10
    assert {d.member_id for d in decisions} == set(small_members["member_id"])


def test_unknown_member_raises_keyerror(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    with pytest.raises(KeyError):
        orchestrator.decide_for_member("NOT_A_REAL_MEMBER", members, ed, care, INDEX_DATE)


# ---- call order / non-bypassable safety authority ----

def test_orchestrator_source_calls_safety_after_navigation():
    """Risk assessment happens in _decide_from_features /
    decide_for_all_members (single vs. batch entry points); both hand off
    to _decide_from_risk, which is the single place Navigation and Safety
    are called -- verify their order there."""
    source = inspect.getsource(UC07Orchestrator._decide_from_risk)
    nav_idx = source.index("care_navigation.decide(")
    safety_idx = source.index("safety_policy.decide(")
    assert nav_idx < safety_idx, "Safety & Policy Agent must be called after Care Navigation in source order"


def test_every_return_path_goes_through_safety_agent():
    """_decide_from_risk has exactly one return statement, and it
    constructs FinalUC07Decision from the SafetyDecision/FinalNavigationView
    pair -- there is no early-return path that could skip safety review."""
    source = inspect.getsource(UC07Orchestrator._decide_from_risk)
    assert source.count("return FinalUC07Decision(") == 1
    assert source.count("safety_policy.decide(") == 1


def test_single_and_batch_entry_points_both_delegate_to_decide_from_risk():
    """Both the single-member and population-batch code paths must funnel
    through the one function that calls Navigation then Safety -- proving
    there is no separate, unreviewed decision path for bulk scoring."""
    single_source = inspect.getsource(UC07Orchestrator._decide_from_features)
    batch_source = inspect.getsource(UC07Orchestrator.decide_for_all_members)
    assert "self._decide_from_risk(" in single_source
    assert "self._decide_from_risk(" in batch_source
    assert "care_navigation.decide(" not in single_source
    assert "care_navigation.decide(" not in batch_source
    assert "safety_policy.decide(" not in single_source
    assert "safety_policy.decide(" not in batch_source


def test_final_decision_navigation_is_never_the_raw_navigation_decision_type(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    member_id = members["member_id"].iloc[0]
    decision = orchestrator.decide_for_member(member_id, members, ed, care, INDEX_DATE, CurrentSafetyContext(red_flag=1))
    assert isinstance(decision.navigation, FinalNavigationView)
    assert not isinstance(decision.navigation, NavigationDecision)
    assert decision.navigation.destination is None  # OVERRIDE suppressed it
    assert decision.safety.state == SafetyState.OVERRIDE


def test_override_current_context_always_wins_regardless_of_risk_or_navigation(orchestrator, synthetic_data):
    """A member whose historical features would otherwise drive a strong
    navigation opportunity must still have destination suppressed to None
    when current safety context triggers OVERRIDE -- proving current
    context, not historical risk, has final say."""
    members, ed, care = synthetic_data
    for member_id in members["member_id"].head(5):
        decision = orchestrator.decide_for_member(
            member_id, members, ed, care, INDEX_DATE, CurrentSafetyContext(icu=1),
        )
        assert decision.safety.state == SafetyState.OVERRIDE
        assert decision.navigation.destination is None


# ---- agent authority separation ----

def test_risk_agent_alone_cannot_produce_navigation(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    member_id = members["member_id"].iloc[0]
    features = __import__("orchestrator").build_point_in_time_features(members, ed, care, INDEX_DATE)
    row = features.loc[features["member_id"] == member_id]
    risk = orchestrator.risk_agent.assess(member_id, row, INDEX_DATE)
    assert not hasattr(risk, "destination")
    assert not hasattr(risk, "navigation")


def test_navigation_agent_alone_cannot_mark_clear_or_override():
    row = pd.Series({"transportation_barrier": 0, "telehealth_available": 0, "pcp_distance_miles": 3.0,
                      "urgent_care_distance_miles": 3.0, "clinical_burden": 0, "prior_ed_count_270d": 0,
                      "prior_potentially_avoidable_ed_count_270d": 0, "prior_pcp_count_270d": 0,
                      "has_prior_care_management": 0})
    decision = care_navigation.decide("M1", RiskTier.LOW, row)
    assert not hasattr(decision, "state")
    assert not hasattr(decision, "override")


def test_safety_agent_can_override_navigation_agent():
    nav = NavigationDecision(
        member_id="M1", destination=__import__("contracts").NavigationDestination.CARE_MANAGEMENT,
        reason_codes=[], explanation="A future Care Management review may be useful.",
    )
    safety, final_nav = safety_policy.decide(nav, CurrentSafetyContext(admitted=1))
    assert safety.state == SafetyState.OVERRIDE
    assert final_nav.destination is None  # navigation's original destination was overridden


# ---- determinism ----

def test_orchestrator_deterministic_repeat(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    member_id = members["member_id"].iloc[2]
    context = CurrentSafetyContext(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4)
    d1 = orchestrator.decide_for_member(member_id, members, ed, care, INDEX_DATE, context)
    d2 = orchestrator.decide_for_member(member_id, members, ed, care, INDEX_DATE, context)
    assert d1.risk.probability == d2.risk.probability
    assert d1.risk.tier == d2.risk.tier
    assert d1.navigation.destination == d2.navigation.destination
    assert d1.navigation.reason_codes == d2.navigation.reason_codes
    assert d1.safety.state == d2.safety.state


# ---- serialization ----

def test_decision_to_dict_is_json_serializable(orchestrator, synthetic_data):
    import json
    members, ed, care = synthetic_data
    member_id = members["member_id"].iloc[0]
    decision = orchestrator.decide_for_member(member_id, members, ed, care, INDEX_DATE)
    payload = decision_to_dict(decision)
    serialized = json.dumps(payload)  # must not raise
    assert json.loads(serialized)["member_id"] == member_id
    assert isinstance(payload["risk"]["tier"], str)
    assert isinstance(payload["safety"]["state"], str)


def test_decision_to_dict_override_has_null_destination(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    member_id = members["member_id"].iloc[0]
    decision = orchestrator.decide_for_member(member_id, members, ed, care, INDEX_DATE, CurrentSafetyContext(major_procedure=1))
    payload = decision_to_dict(decision)
    assert payload["navigation"]["destination"] is None
    assert payload["safety"]["state"] == "OVERRIDE"


# ---- missing per-member context in batch call ----

def test_batch_call_missing_context_for_some_members_defaults_to_caution(orchestrator, synthetic_data):
    members, ed, care = synthetic_data
    small_members = members.head(3)
    ids = list(small_members["member_id"])
    contexts = {ids[0]: CurrentSafetyContext(red_flag=1)}  # only first member has context
    decisions = orchestrator.decide_for_all_members(small_members, ed, care, INDEX_DATE, contexts)
    by_id = {d.member_id: d for d in decisions}
    assert by_id[ids[0]].safety.state == SafetyState.OVERRIDE
    assert by_id[ids[1]].safety.state == SafetyState.CAUTION
    assert by_id[ids[2]].safety.state == SafetyState.CAUTION
