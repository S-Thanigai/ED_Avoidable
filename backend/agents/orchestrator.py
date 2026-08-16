"""
orchestrator.py
----------------
UC07 Orchestrator -- the single authoritative entry point for producing a
FinalUC07Decision. Enforces the fixed call order:

    member/history data
        -> point-in-time feature reconstruction (backend/pit, unmodified)
        -> Risk Detection Agent
        -> Care Navigation Agent
        -> Safety & Policy Agent   (ALWAYS LAST)
        -> FinalUC07Decision

The Safety & Policy Agent's output is never optional and never skipped:
every code path through `decide_for_member` calls `safety_policy.decide()`
exactly once, after navigation, before returning. There is no code path
in this module that returns a NavigationDecision (the Care Navigation
Agent's raw output) directly to a caller -- only the safety-reviewed
FinalNavigationView, embedded in FinalUC07Decision, ever leaves this
module.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pit"))

import care_navigation
import safety_policy
from contracts import CurrentSafetyContext, FinalUC07Decision
from risk_detection import RiskDetectionAgent
from windows import build_snapshot_window
from features import build_observation_features


def build_point_in_time_features(members: pd.DataFrame, ed: pd.DataFrame, care: pd.DataFrame, index_date: date) -> pd.DataFrame:
    """Reuses backend/pit/features.py + windows.py unmodified -- the
    SAME leakage-safe point-in-time logic used for training. No
    synthetic/original-specific branch, no re-implementation."""
    window = build_snapshot_window("live", index_date)
    return build_observation_features(members, ed, care, window)


class UC07Orchestrator:
    def __init__(self, risk_agent: RiskDetectionAgent | None = None):
        self.risk_agent = risk_agent or RiskDetectionAgent()

    def decide_for_member(
        self,
        member_id: str,
        members: pd.DataFrame,
        ed: pd.DataFrame,
        care: pd.DataFrame,
        index_date: date,
        current_context: CurrentSafetyContext | None = None,
    ) -> FinalUC07Decision:
        current_context = current_context or CurrentSafetyContext()
        features = build_point_in_time_features(members, ed, care, index_date)
        return self._decide_from_features(member_id, features, current_context)

    def decide_for_all_members(
        self,
        members: pd.DataFrame,
        ed: pd.DataFrame,
        care: pd.DataFrame,
        index_date: date,
        current_contexts: dict[str, CurrentSafetyContext] | None = None,
    ) -> list[FinalUC07Decision]:
        """Computes point-in-time features once for the whole uploaded
        population, scores risk for the whole population in one
        vectorized pass (RiskDetectionAgent.assess_batch -- see its
        docstring: produces identical results to per-member assess()),
        then runs Navigation -> Safety per member. `current_contexts`
        maps member_id -> CurrentSafetyContext; a member with no entry is
        treated as having NO current safety context supplied (CAUTION),
        never as CLEAR."""
        current_contexts = current_contexts or {}
        features = build_point_in_time_features(members, ed, care, index_date)
        member_ids = features["member_id"].tolist()
        index_date_value = pd.Timestamp(features["index_date"].iloc[0]).date() if len(features) else index_date

        risk_by_member = self.risk_agent.assess_batch(member_ids, features, index_date_value)

        decisions = []
        for i, member_id in enumerate(member_ids):
            context = current_contexts.get(member_id, CurrentSafetyContext())
            decisions.append(self._decide_from_risk(member_id, risk_by_member[member_id], features.iloc[i], context))
        return decisions

    def _decide_from_features(self, member_id: str, features: pd.DataFrame, current_context: CurrentSafetyContext) -> FinalUC07Decision:
        member_row = features.loc[features["member_id"] == member_id]
        if member_row.empty:
            raise KeyError(f"member_id {member_id!r} not found in the provided member data")
        # features.py (backend/pit, unmodified) always inserts an
        # `index_date` column -- read it back rather than threading a
        # second copy of the value through this call chain.
        index_date = pd.Timestamp(member_row["index_date"].iloc[0]).date()

        # ---- 1. Risk Detection Agent ----
        risk = self.risk_agent.assess(member_id, member_row, index_date)

        return self._decide_from_risk(member_id, risk, member_row.iloc[0], current_context)

    def _decide_from_risk(self, member_id: str, risk, feature_row: pd.Series, current_context: CurrentSafetyContext) -> FinalUC07Decision:
        # ---- 2. Care Navigation Agent (never sees current_context) ----
        navigation = care_navigation.decide(member_id, risk.tier, feature_row)

        # ---- 3. Safety & Policy Agent -- ALWAYS LAST, ALWAYS RUNS ----
        safety, final_navigation = safety_policy.decide(navigation, current_context)

        return FinalUC07Decision(
            member_id=member_id,
            risk=risk,
            navigation=final_navigation,
            safety=safety,
            disclaimer=safety_policy.BASE_DISCLAIMER,
        )


def decision_to_dict(decision: FinalUC07Decision) -> dict:
    """JSON-safe serialization of a FinalUC07Decision for the API layer.
    Enum members become their string value; dates become ISO strings."""
    payload = asdict(decision)

    payload["risk"]["tier"] = decision.risk.tier.value
    payload["risk"]["index_date"] = decision.risk.index_date.isoformat()
    # Phase 8C: structured, authoritative model explainability. `asdict`
    # deep-copies nested dataclasses but leaves Enum members as Enum
    # instances (not JSON-safe) -- fix those up explicitly rather than
    # relying on a custom JSON encoder, so this stays an explicit,
    # auditable transform like every other field above.
    payload["risk"]["explanation_method"] = decision.risk.explanation_method.value
    for i, factor in enumerate(decision.risk.explanation_factors):
        payload["risk"]["explanation_factors"][i]["direction"] = factor.direction.value
        payload["risk"]["explanation_factors"][i]["explanation_method"] = factor.explanation_method.value

    payload["navigation"]["destination"] = (
        decision.navigation.destination.value if decision.navigation.destination else None
    )
    payload["navigation"]["reason_codes"] = [r.value for r in decision.navigation.reason_codes]

    payload["safety"]["state"] = decision.safety.state.value
    payload["safety"]["context_completeness"] = decision.safety.context_completeness.value
    payload["safety"]["context_source"] = decision.safety.context_source.value

    return payload
