"""
Tests for backend/agents/safety_policy.py (AGENT 3 -- final authority).

Covers CLEAR/CAUTION/OVERRIDE, every individual override condition
(red_flag / icu / admitted / major_procedure / triage_level in {1,2}),
missing-current-context -> CAUTION (never CLEAR), the centralized
prohibited-phrase policy (every required phrase + case/variant handling),
and that OVERRIDE always suppresses the navigation destination.
"""
import pytest

import safety_policy
from contracts import CurrentSafetyContext, NavigationDecision, NavigationDestination, ReasonCode, SafetyState

CLEAN_NAV = NavigationDecision(
    member_id="M1", destination=NavigationDestination.TELEHEALTH,
    reason_codes=[ReasonCode.TELEHEALTH_AVAILABLE],
    explanation="Telehealth may be a useful future, non-emergency option.",
)


# ---- CLEAR ----

def test_clear_when_context_provided_and_no_override_condition():
    context = CurrentSafetyContext(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4)
    safety, nav_view = safety_policy.decide(CLEAN_NAV, context)
    assert safety.state == SafetyState.CLEAR
    assert safety.override is False
    assert nav_view.destination == NavigationDestination.TELEHEALTH


def test_partial_context_with_no_override_signal_is_caution_not_clear():
    """Phase 6 hardening: CLEAR requires ALL FIVE current-safety fields to
    be explicitly known, not just one. A caller who supplies only
    triage_level=4 has NOT established that red_flag/icu/admitted/
    major_procedure are also safe -- those remain genuinely unknown, so
    this must be CAUTION, never CLEAR."""
    context = CurrentSafetyContext(triage_level=4)  # only this field set, no exclusion
    safety, _ = safety_policy.decide(CLEAN_NAV, context)
    assert safety.state == SafetyState.CAUTION


# ---- CAUTION: missing current context never becomes CLEAR ----

def test_missing_current_context_produces_caution_not_clear():
    context = CurrentSafetyContext()  # nothing supplied
    safety, nav_view = safety_policy.decide(CLEAN_NAV, context)
    assert safety.state == SafetyState.CAUTION
    assert safety.override is False
    assert "non-emergency" in safety.message.lower()
    # navigation is still shown under CAUTION, just framed as non-emergency-only
    assert nav_view.destination == NavigationDestination.TELEHEALTH


def test_historical_absence_of_red_flags_does_not_imply_clear():
    """This test's name documents the specific principle from the spec:
    the agent never infers safety from what ISN'T in the request."""
    context = CurrentSafetyContext()
    safety, _ = safety_policy.decide(CLEAN_NAV, context)
    assert safety.state != SafetyState.CLEAR


# ---- OVERRIDE: every individual trigger condition ----

@pytest.mark.parametrize("context_kwargs", [
    {"red_flag": 1, "icu": 0, "admitted": 0, "major_procedure": 0, "triage_level": 4},
    {"red_flag": 0, "icu": 1, "admitted": 0, "major_procedure": 0, "triage_level": 4},
    {"red_flag": 0, "icu": 0, "admitted": 1, "major_procedure": 0, "triage_level": 4},
    {"red_flag": 0, "icu": 0, "admitted": 0, "major_procedure": 1, "triage_level": 4},
    {"red_flag": 0, "icu": 0, "admitted": 0, "major_procedure": 0, "triage_level": 1},
    {"red_flag": 0, "icu": 0, "admitted": 0, "major_procedure": 0, "triage_level": 2},
])
def test_each_override_condition_triggers_override(context_kwargs):
    context = CurrentSafetyContext(**context_kwargs)
    safety, nav_view = safety_policy.decide(CLEAN_NAV, context)
    assert safety.state == SafetyState.OVERRIDE
    assert safety.override is True
    assert nav_view.destination is None
    assert nav_view.reason_codes == []


def test_override_message_is_approved_safety_language():
    context = CurrentSafetyContext(red_flag=1)
    safety, nav_view = safety_policy.decide(CLEAN_NAV, context)
    assert "should not be delayed" in safety.message
    assert "911" in safety.message
    assert safety_policy.check_text(safety.message) == []
    assert safety_policy.check_text(nav_view.explanation) == []


def test_triage_3_alone_does_not_trigger_override():
    context = CurrentSafetyContext(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=3)
    safety, _ = safety_policy.decide(CLEAN_NAV, context)
    assert safety.state == SafetyState.CLEAR


# ---- centralized prohibited-language policy ----

@pytest.mark.parametrize("phrase", safety_policy.PROHIBITED_PHRASES)
def test_every_prohibited_phrase_is_detected(phrase):
    # Some entries are legitimate substrings of others (e.g. "not an
    # emergency" inside "this is not an emergency") -- text containing one
    # may correctly match more than one policy entry, so assert membership,
    # not exact-list equality.
    assert phrase in safety_policy.check_text(f"Some text. {phrase} Some more text.")


@pytest.mark.parametrize("variant", [
    "AVOID THE ER",
    "Avoid The ER",
    "avoid   the   er",  # extra whitespace
    "You Don't Need Emergency Care right now",
    "This Is Not An Emergency situation",
])
def test_case_and_whitespace_variants_detected(variant):
    assert len(safety_policy.check_text(variant)) > 0


def test_clean_text_passes_policy():
    assert safety_policy.check_text("Telehealth may be a useful future, non-emergency option.") == []


def test_blocked_navigation_text_is_replaced_not_passed_through():
    bad_nav = NavigationDecision(
        member_id="M1", destination=NavigationDestination.URGENT_CARE, reason_codes=[],
        explanation="You don't need emergency care, so avoid the ER and use urgent care instead.",
    )
    context = CurrentSafetyContext(red_flag=0, icu=0, admitted=0, major_procedure=0, triage_level=4)
    safety, nav_view = safety_policy.decide(bad_nav, context)
    assert safety.state == SafetyState.CLEAR  # clinical state unaffected by language issue
    assert len(safety.blocked_phrases) > 0
    assert safety_policy.check_text(nav_view.explanation) == []
    assert nav_view.explanation != bad_nav.explanation  # text was replaced, not passed through
    assert nav_view.destination == NavigationDestination.URGENT_CARE  # destination itself is untouched


def test_blocked_language_also_replaced_under_caution():
    bad_nav = NavigationDecision(
        member_id="M1", destination=NavigationDestination.URGENT_CARE, reason_codes=[],
        explanation="This is not an emergency, avoid the ED.",
    )
    safety, nav_view = safety_policy.decide(bad_nav, CurrentSafetyContext())
    assert safety.state == SafetyState.CAUTION
    assert safety_policy.check_text(nav_view.explanation) == []


# ---- final response passes policy across all destinations (Step: adversarial testing) ----

@pytest.mark.parametrize("destination", list(NavigationDestination))
def test_final_navigation_view_passes_policy_for_every_destination(destination):
    nav = NavigationDecision(
        member_id="M1", destination=destination, reason_codes=[],
        explanation=f"A future, non-emergency opportunity for {destination.value} was identified.",
    )
    for context in (CurrentSafetyContext(), CurrentSafetyContext(red_flag=0, triage_level=4), CurrentSafetyContext(red_flag=1)):
        safety, nav_view = safety_policy.decide(nav, context)
        assert safety_policy.check_text(nav_view.explanation) == []
        assert safety_policy.check_text(safety.message) == []
