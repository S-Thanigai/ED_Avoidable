"""
safety_policy.py
-----------------
AGENT 3 -- Safety & Policy Agent. FINAL, non-bypassable authority.

Responsibility: review the Care Navigation Agent's decision against
CURRENT safety context (never historical data) and a centralized
prohibited-language policy, and produce the safety-reviewed view that is
the only thing ever returned to a caller. No model probability, risk
tier, navigation destination, frontend, or API caller can bypass this
agent or downgrade its decision after the fact.

Historical absence of red flags does NOT prove the current situation is
safe: if the caller does not supply enough current-encounter information
to establish CLEAR, this agent uses CAUTION -- it never assumes safety
from missing information (docs/05_MULTI_AGENT_SYSTEM.md section 15).

This agent does not, and cannot, predict whether a real-time situation is
a true emergency from a future-utilization risk model. It only classifies
what the system is allowed to say, given what current information (if
any) was supplied.
"""
from __future__ import annotations

import re

from contracts import (
    ContextCompleteness,
    CurrentSafetyContext,
    FinalNavigationView,
    NavigationDecision,
    SafetyDecision,
    SafetyState,
)

BASE_DISCLAIMER = (
    "For care navigation only -- never a reason to delay care. This tool flags patterns of "
    "potentially avoidable ED use and suggests a lower-acuity next step for FUTURE, "
    "non-emergency needs. It does not diagnose, triage, or determine medical necessity, and no "
    "output here should ever be read as discouraging emergency care. If you or a member may be "
    "experiencing a medical emergency, call 911 or go to the nearest emergency department "
    "immediately."
)

OVERRIDE_MESSAGE = (
    "Emergency care should not be delayed when emergency symptoms or high-acuity conditions are "
    "present. If you or this member may be experiencing a medical emergency, call 911 or go to "
    "the nearest emergency department immediately. No navigation alternative is offered for this "
    "encounter."
)

CAUTION_MESSAGE = (
    "Current clinical/safety context for this encounter was not provided, so this system cannot "
    "confirm the current situation is non-emergency. For non-emergency situations only: if you "
    "are experiencing a medical emergency, call 911 or go to the nearest emergency department "
    "immediately."
)

CLEAR_MESSAGE = (
    "This is a navigation suggestion for future, non-emergency needs only. It never overrides "
    "your judgment to seek emergency care."
)

HIGH_ACUITY_TRIAGE_LEVELS = {1, 2}

# Centralized, case-insensitive prohibited-phrase policy. Matched against
# normalized (lowercased, whitespace-collapsed) text. Every entry here is
# a phrase the final user-visible recommendation must never contain.
PROHIBITED_PHRASES: list[str] = [
    "avoid the er",
    "avoid the ed",
    "avoid the emergency room",
    "avoid the emergency department",
    "avoid emergency care",
    "don't go to the er",
    "do not go to the er",
    "don't go to the ed",
    "do not go to the ed",
    "don't go to the emergency room",
    "do not go to the emergency room",
    "don't go to the emergency department",
    "do not go to the emergency department",
    "you don't need emergency care",
    "you do not need emergency care",
    "not an emergency",
    "this is not an emergency",
    "unnecessary emergency visit",
    "inappropriate emergency visit",
    "the ed is unnecessary",
    "the er is unnecessary",
    "the ed is inappropriate",
    "the er is inappropriate",
    "instead of the er",
    "instead of the ed",
    "instead of emergency care",
    "urgent care instead of the ed",
    "urgent care instead of the er",
    "skip the er",
    "skip the ed",
    "no need for emergency",
    "not necessary to go to the er",
    "not necessary to go to the ed",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def check_text(text: str) -> list[str]:
    """Return every prohibited phrase found in `text` (case-insensitive,
    whitespace-normalized). Empty list == the text passes the policy."""
    normalized = _normalize(text)
    return [phrase for phrase in PROHIBITED_PHRASES if phrase in normalized]


def _current_context_triggers_override(context: CurrentSafetyContext) -> bool:
    if context.red_flag == 1 or context.icu == 1 or context.admitted == 1 or context.major_procedure == 1:
        return True
    if context.triage_level is not None and context.triage_level in HIGH_ACUITY_TRIAGE_LEVELS:
        return True
    return False


def _determine_state(context: CurrentSafetyContext) -> SafetyState:
    """Phase 7 formalizes the Phase 6 fix's completeness check as the
    named ContextCompleteness state on CurrentSafetyContext itself
    (docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md section 17) -- the rule
    is unchanged: a known override trigger always wins first (regardless
    of completeness, so a single known trigger field still fires
    OVERRIDE even with every other field missing); only a COMPLETE
    context with no trigger reaches CLEAR; PARTIAL and ABSENT both
    resolve to CAUTION."""
    if _current_context_triggers_override(context):
        return SafetyState.OVERRIDE
    if context.completeness != ContextCompleteness.COMPLETE:
        return SafetyState.CAUTION
    return SafetyState.CLEAR


def decide(navigation: NavigationDecision, context: CurrentSafetyContext) -> tuple[SafetyDecision, FinalNavigationView]:
    """
    Returns (SafetyDecision, FinalNavigationView) -- the FinalNavigationView
    is the ONLY navigation-shaped object ever allowed to leave the
    orchestrator; NavigationDecision itself must never be serialized
    directly to a caller.
    """
    state = _determine_state(context)
    blocked = check_text(navigation.explanation)
    completeness = context.completeness
    source = context.source

    if state == SafetyState.OVERRIDE:
        safety = SafetyDecision(state=state, override=True, message=OVERRIDE_MESSAGE, blocked_phrases=blocked,
                                 context_completeness=completeness, context_source=source)
        final_navigation = FinalNavigationView(destination=None, reason_codes=[], explanation=OVERRIDE_MESSAGE)
        return safety, final_navigation

    # CLEAR or CAUTION: navigation may be shown, but only after the
    # language policy check -- a blocked phrase is replaced, never passed
    # through, regardless of state.
    if blocked:
        explanation = (
            "A future, non-emergency navigation opportunity was identified for this member. "
            "(Original explanation text was withheld because it did not pass the safety language "
            "policy.)"
        )
    else:
        explanation = navigation.explanation

    message = CAUTION_MESSAGE if state == SafetyState.CAUTION else CLEAR_MESSAGE
    safety = SafetyDecision(state=state, override=False, message=message, blocked_phrases=blocked,
                             context_completeness=completeness, context_source=source)
    final_navigation = FinalNavigationView(
        destination=navigation.destination,
        reason_codes=navigation.reason_codes,
        explanation=explanation,
    )
    return safety, final_navigation
