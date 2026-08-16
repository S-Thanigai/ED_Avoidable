"""
contracts.py
------------
Shared typed contracts for the Phase 5 UC07 multi-agent decision system.

These dataclasses are the ONLY way data crosses an agent boundary in this
project. Each agent takes a narrowly-typed input and returns one of these
structures -- no agent reaches into another agent's internals, and no
agent returns a free-form dict as its primary output. This is what makes
the authority boundaries in docs/05_MULTI_AGENT_SYSTEM.md enforceable in
code, not just in documentation.

"Agent" in this project means a bounded software component with a
defined responsibility, input contract, decision logic, output contract,
and authority boundary -- deterministic and auditable except for the
trained risk model's probability output. No LLM or external AI service
is used anywhere in this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class RiskTier(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class NavigationDestination(str, Enum):
    PRIMARY_CARE = "PRIMARY_CARE"
    URGENT_CARE = "URGENT_CARE"
    TELEHEALTH = "TELEHEALTH"
    CARE_MANAGEMENT = "CARE_MANAGEMENT"
    NO_PROACTIVE_NAVIGATION = "NO_PROACTIVE_NAVIGATION"


class SafetyState(str, Enum):
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    OVERRIDE = "OVERRIDE"


class ContextCompleteness(str, Enum):
    """Phase 7 formalization of the Phase 6 safety fix's completeness
    check (docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md section 17):
    COMPLETE -- all five current-safety fields are explicitly known.
    PARTIAL  -- at least one, but not all five, fields are known.
    ABSENT   -- no field was supplied at all.
    Only COMPLETE, with no override trigger present, can ever produce
    CLEAR; PARTIAL and ABSENT both resolve to CAUTION (unless a known
    field already triggers OVERRIDE, which is checked first and wins
    regardless of completeness)."""
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    ABSENT = "ABSENT"


class ContextSource(str, Enum):
    """Audit-only provenance of a CurrentSafetyContext -- where it came
    from, NOT a verification claim. Only CALLER_SUPPLIED and
    NOT_AVAILABLE are ever produced by this system today; SYSTEM_DERIVED
    is reserved for a future capability (e.g. a real-time feed) and must
    never be asserted where none exists."""
    CALLER_SUPPLIED = "CALLER_SUPPLIED"
    SYSTEM_DERIVED = "SYSTEM_DERIVED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class FactorDirection(str, Enum):
    INCREASES_RISK = "INCREASES_RISK"
    DECREASES_RISK = "DECREASES_RISK"


class ExplanationMethod(str, Enum):
    """Phase 8C: which method actually produced a member's
    explanation_factors. SHAP_LINEAR (shap.LinearExplainer over the
    frozen Logistic Regression pipeline, correlation-aware) is primary;
    LINEAR_CONTRIBUTION (exact coefficient x standardized-value
    decomposition) is the deterministic fallback used only if SHAP
    computation fails for any reason -- see
    backend/agents/model_explainability.py and
    docs/08C_GENAI_EXPLAINABILITY.md section 3."""
    SHAP_LINEAR = "SHAP_LINEAR"
    LINEAR_CONTRIBUTION = "LINEAR_CONTRIBUTION"


class ReasonCode(str, Enum):
    ELEVATED_FUTURE_RISK = "ELEVATED_FUTURE_RISK"
    REPEATED_LOWER_ACUITY_HISTORY = "REPEATED_LOWER_ACUITY_HISTORY"
    TRANSPORTATION_BARRIER = "TRANSPORTATION_BARRIER"
    LIMITED_PCP_ACCESS = "LIMITED_PCP_ACCESS"
    TELEHEALTH_AVAILABLE = "TELEHEALTH_AVAILABLE"
    CHRONIC_COMPLEXITY = "CHRONIC_COMPLEXITY"
    PRIOR_CM_ENGAGEMENT = "PRIOR_CM_ENGAGEMENT"
    OUTPATIENT_CONTINUITY_OPPORTUNITY = "OUTPATIENT_CONTINUITY_OPPORTUNITY"
    URGENT_CARE_ACCESS_ADVANTAGE = "URGENT_CARE_ACCESS_ADVANTAGE"
    NO_OPPORTUNITY_IDENTIFIED = "NO_OPPORTUNITY_IDENTIFIED"


# ---------------------------------------------------------------------------
# Agent 1 output: Risk Detection Agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExplanationFactor:
    """One structured, member-level model-explanation factor (Phase 8C).
    NEVER a causal claim -- `direction` describes the sign of this
    feature's contribution to the model's own log-odds, not a real-world
    cause. `contribution` is the raw signed value from whichever
    `explanation_method` produced it; the two methods are NOT on
    identical scales in general (see docs/08C section 3-4), so this
    field should only be compared within one factor list, never across
    two different explanation_method values."""
    feature: str
    display_name: str
    direction: FactorDirection
    contribution: float
    explanation_method: ExplanationMethod


@dataclass(frozen=True)
class RiskAssessment:
    member_id: str
    probability: float
    tier: RiskTier
    contributing_factors: list[str]
    model_version: str
    dataset_id: str
    synthetic_model: bool
    index_date: date
    moderate_threshold: float
    high_threshold: float
    # Phase 8C: structured model explainability. Defaults to an empty
    # list / LINEAR_CONTRIBUTION so existing test construction of
    # RiskAssessment(...) without these fields keeps working unchanged.
    explanation_factors: list[ExplanationFactor] = field(default_factory=list)
    explanation_method: ExplanationMethod = ExplanationMethod.LINEAR_CONTRIBUTION
    explanation_causal: bool = False


# ---------------------------------------------------------------------------
# Current, live safety context -- the ONLY input the Safety & Policy Agent
# uses to decide OVERRIDE vs. CAUTION vs. CLEAR. Deliberately separate from
# any historical feature: historical absence of red flags does not prove
# the CURRENT situation is safe (docs/05_MULTI_AGENT_SYSTEM.md section 15).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurrentSafetyContext:
    red_flag: int | None = None
    icu: int | None = None
    admitted: int | None = None
    major_procedure: int | None = None
    triage_level: int | None = None

    @property
    def provided(self) -> bool:
        """True if the caller supplied ANY current safety signal at all.
        If False, the Safety Agent must never assume CLEAR."""
        return any(
            v is not None
            for v in (self.red_flag, self.icu, self.admitted, self.major_procedure, self.triage_level)
        )

    @property
    def completeness(self) -> ContextCompleteness:
        """Deterministic COMPLETE/PARTIAL/ABSENT classification (Phase 7,
        formalizing the Phase 6 fix). A field that is None is UNKNOWN,
        never treated as 0/false -- this property counts only fields the
        caller explicitly supplied a value for."""
        fields = (self.red_flag, self.icu, self.admitted, self.major_procedure, self.triage_level)
        known = sum(v is not None for v in fields)
        if known == len(fields):
            return ContextCompleteness.COMPLETE
        if known == 0:
            return ContextCompleteness.ABSENT
        return ContextCompleteness.PARTIAL

    @property
    def source(self) -> ContextSource:
        """Audit-only provenance, computed (not stored) so it can never
        drift from the fields actually present. Every CurrentSafetyContext
        in this system today is either caller-supplied (via
        POST /uc07/decide's current_safety_context field) or not supplied
        at all -- this system never claims SYSTEM_DERIVED provenance,
        which would imply a verification capability that does not exist."""
        return ContextSource.CALLER_SUPPLIED if self.provided else ContextSource.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Agent 2 output: Care Navigation Agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NavigationDecision:
    member_id: str
    destination: NavigationDestination
    reason_codes: list[ReasonCode]
    explanation: str


# ---------------------------------------------------------------------------
# Agent 3 output: Safety & Policy Agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyDecision:
    state: SafetyState
    override: bool
    message: str
    blocked_phrases: list[str] = field(default_factory=list)
    # Phase 7 audit metadata -- what the Safety Agent actually saw, not a
    # verification claim. Defaults describe "no context was involved",
    # matching a caller who never supplied a CurrentSafetyContext at all.
    context_completeness: ContextCompleteness = ContextCompleteness.ABSENT
    context_source: ContextSource = ContextSource.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Orchestrator output: the ONLY structure the API is allowed to return to a
# caller. Its `navigation` block is the safety-reviewed final view -- never
# the Care Navigation Agent's raw, pre-safety-review output.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinalNavigationView:
    destination: NavigationDestination | None
    reason_codes: list[ReasonCode]
    explanation: str


@dataclass(frozen=True)
class FinalUC07Decision:
    member_id: str
    risk: RiskAssessment
    navigation: FinalNavigationView
    safety: SafetyDecision
    disclaimer: str


# ---------------------------------------------------------------------------
# Phase 8C -- AGENT 4 (GenAI Explanation Agent) output. NOT a decision
# contract: nothing in this system ever reads a MemberExplanation back into
# the Risk/Navigation/Safety agents (see backend/agents/genai_explanation.py
# and docs/08C_GENAI_EXPLAINABILITY.md section 8 -- "one-way architecture").
# This dataclass exists purely to carry already-decided facts translated
# into natural language, plus an honest record of which mechanism produced
# that language.
# ---------------------------------------------------------------------------

class ExplanationSource(str, Enum):
    GENAI = "GENAI"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


@dataclass(frozen=True)
class MemberExplanation:
    summary: str
    risk_explanation: str
    navigation_explanation: str
    safety_explanation: str
    disclaimer: str
    explanation_source: ExplanationSource
    model_used: str | None = None
    generation_time_ms: float | None = None
