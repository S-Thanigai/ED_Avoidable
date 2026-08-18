"""
decision_codec.py
------------------
Two-way conversion between `AnalysisResult` ORM rows and the exact JSON
shape `backend/agents/orchestrator.py`'s `decision_to_dict()` already
produces (== `FinalUC07Decision` on the frontend, see
frontend/src/uc07/types.ts). This module computes nothing about risk,
navigation, or safety -- it only stores/reconstructs an
already-decided value, field for field.
"""
from __future__ import annotations

import json

from .models import AnalysisResult


def decision_dict_to_analysis_result_kwargs(population_id: int, decision: dict) -> dict:
    """`decision` is the output of orchestrator.decision_to_dict() for one
    member. Returns kwargs suitable for `AnalysisResult(**kwargs)`."""
    risk = decision["risk"]
    navigation = decision["navigation"]
    safety = decision["safety"]
    return {
        "population_id": population_id,
        "member_id": decision["member_id"],
        "probability": risk["probability"],
        "tier": risk["tier"],
        "contributing_factors_json": json.dumps(risk["contributing_factors"]),
        "model_version": risk["model_version"],
        "dataset_id": risk["dataset_id"],
        "synthetic_model": risk["synthetic_model"],
        "index_date": risk["index_date"],
        "moderate_threshold": risk["moderate_threshold"],
        "high_threshold": risk["high_threshold"],
        "explanation_factors_json": json.dumps(risk["explanation_factors"]),
        "explanation_method": risk["explanation_method"],
        "explanation_causal": risk["explanation_causal"],
        "navigation_destination": navigation["destination"],
        "navigation_reason_codes_json": json.dumps(navigation["reason_codes"]),
        "navigation_explanation": navigation["explanation"],
        "safety_state": safety["state"],
        "safety_override": safety["override"],
        "safety_message": safety["message"],
        "safety_blocked_phrases_json": json.dumps(safety["blocked_phrases"]),
        "safety_context_completeness": safety["context_completeness"],
        "safety_context_source": safety["context_source"],
        "disclaimer": decision["disclaimer"],
    }


def analysis_result_to_decision_dict(row: AnalysisResult) -> dict:
    """Reconstructs the exact FinalUC07Decision-shaped dict from a stored
    row -- what GET /populations/{id}/members and .../members/{member_id}
    return, so the frontend can reuse its existing decision-rendering
    components unchanged."""
    index_date = row.index_date.isoformat() if hasattr(row.index_date, "isoformat") else row.index_date
    return {
        "member_id": row.member_id,
        "risk": {
            "member_id": row.member_id,
            "probability": row.probability,
            "tier": row.tier,
            "contributing_factors": json.loads(row.contributing_factors_json),
            "model_version": row.model_version,
            "dataset_id": row.dataset_id,
            "synthetic_model": row.synthetic_model,
            "index_date": index_date,
            "moderate_threshold": row.moderate_threshold,
            "high_threshold": row.high_threshold,
            "explanation_factors": json.loads(row.explanation_factors_json),
            "explanation_method": row.explanation_method,
            "explanation_causal": row.explanation_causal,
        },
        "navigation": {
            "destination": row.navigation_destination,
            "reason_codes": json.loads(row.navigation_reason_codes_json),
            "explanation": row.navigation_explanation,
        },
        "safety": {
            "state": row.safety_state,
            "override": row.safety_override,
            "message": row.safety_message,
            "blocked_phrases": json.loads(row.safety_blocked_phrases_json),
            "context_completeness": row.safety_context_completeness,
            "context_source": row.safety_context_source,
        },
        "disclaimer": row.disclaimer,
    }
