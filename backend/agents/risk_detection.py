"""
risk_detection.py
------------------
AGENT 1 -- Risk Detection Agent.

Responsibility: estimate the probability that a member will have >=1
future potentially avoidable ED encounter within the next 90 days
(future_potentially_avoidable_ed_90d), using the trained
uc07-risk-synthetic-v1 model, and assign a risk tier from that model's
own frozen VALIDATION-derived thresholds.

SYNTHETIC DATA MODEL -- DEMONSTRATION ONLY. See
docs/04D_SYNTHETIC_MODEL_DEVELOPMENT.md.

This agent MUST NOT:
    - determine whether the current situation is an emergency
    - recommend PCP / Urgent Care / Telehealth / Care Management
    - tell anyone not to use the ED
    - diagnose a condition
Enforced by construction: this module never imports anything from
care_navigation.py or safety_policy.py, and RiskAssessment (contracts.py)
has no navigation- or safety-shaped field.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from contracts import RiskAssessment, RiskTier
import model_explainability

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "backend" / "models" / "uc07_risk_synthetic_v1_model.joblib"
DEFAULT_METADATA_PATH = REPO_ROOT / "backend" / "models" / "uc07_risk_synthetic_v1_model_metadata.json"


class ModelIncompatibleError(RuntimeError):
    """Raised when the model artifact and its metadata disagree, or the
    artifact is otherwise unsafe to serve predictions from. The agent
    fails loudly and safely rather than guessing."""


# Human-readable, non-causal templates for the top contributing features.
# Matched by substring against the (already point-in-time-safe) feature
# name -- never exposes a raw coefficient or claims causation.
_FEATURE_DESCRIPTIONS: list[tuple[str, str]] = [
    ("transportation_barrier", "Transportation access barriers"),
    ("urgent_care_distance", "Greater distance to urgent care"),
    ("pcp_distance", "Greater distance to primary care"),
    ("telehealth_available", "Telehealth availability"),
    ("potentially_avoidable_ed", "Recent potentially avoidable ED utilization"),
    ("protected_ed", "Recent high-acuity ED history"),
    ("uncertain_ed", "Recent ambiguous-acuity ED history"),
    ("ed_count", "Recent overall ED utilization"),
    ("has_prior_ed", "History of ED utilization"),
    ("days_since_prior_ed", "Recency of ED utilization"),
    ("care_management", "Care Management engagement history"),
    ("pcp_count", "Primary care utilization history"),
    ("urgent_care_count", "Urgent care utilization history"),
    ("telehealth_count", "Telehealth utilization history"),
    ("clinical_burden", "Chronic condition burden"),
    ("num_chronic_conditions", "Chronic condition burden"),
    ("access_burden", "Overall access burden"),
    ("velocity", "Recent utilization trend"),
    ("age", "Member age"),
    ("gender", "Member demographic profile"),
]


def _describe_feature(feature_name: str) -> str:
    for token, description in _FEATURE_DESCRIPTIONS:
        if token in feature_name:
            return description
    return "A historical utilization or access pattern"


def load_model_bundle(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> dict:
    """Load the artifact + metadata and verify they agree before trusting
    either. Fails safely (raises ModelIncompatibleError) rather than
    silently serving predictions from a mismatched pair."""
    if not artifact_path.exists():
        raise ModelIncompatibleError(f"Model artifact not found at {artifact_path}")
    if not metadata_path.exists():
        raise ModelIncompatibleError(f"Model metadata not found at {metadata_path}")

    artifact = joblib.load(artifact_path)
    metadata = json.loads(metadata_path.read_text())

    required_artifact_keys = {"pipeline", "feature_columns", "model_version", "dataset_id", "synthetic",
                               "moderate_threshold", "high_threshold"}
    missing = required_artifact_keys - set(artifact.keys())
    if missing:
        raise ModelIncompatibleError(f"Model artifact missing required keys: {sorted(missing)}")

    if artifact["feature_columns"] != metadata.get("feature_list"):
        raise ModelIncompatibleError(
            "Model artifact feature_columns do not match metadata feature_list -- "
            "refusing to serve predictions from a potentially inconsistent model."
        )
    if artifact["model_version"] != metadata.get("model_version"):
        raise ModelIncompatibleError("Model artifact/metadata model_version mismatch.")
    if artifact["moderate_threshold"] != metadata.get("moderate_threshold") or \
       artifact["high_threshold"] != metadata.get("high_threshold"):
        raise ModelIncompatibleError("Model artifact/metadata threshold mismatch.")
    if not (0.0 <= artifact["moderate_threshold"] < artifact["high_threshold"] <= 1.0):
        raise ModelIncompatibleError(
            f"Invalid thresholds: moderate={artifact['moderate_threshold']} high={artifact['high_threshold']}"
        )

    return {"artifact": artifact, "metadata": metadata}


def assign_risk_tier(probability: float, moderate_threshold: float, high_threshold: float) -> RiskTier:
    if probability >= high_threshold:
        return RiskTier.HIGH
    if probability >= moderate_threshold:
        return RiskTier.MODERATE
    return RiskTier.LOW


def _factors_from_contributions(contributions: np.ndarray, feature_names: list[str], max_factors: int = 3) -> list[str]:
    order = np.argsort(np.abs(contributions))[::-1]
    factors: list[str] = []
    seen: set[str] = set()
    for idx in order:
        if len(factors) >= max_factors:
            break
        if abs(contributions[idx]) < 1e-9:
            continue
        direction = "contributed to elevated risk" if contributions[idx] > 0 else "contributed to lower risk"
        description = _describe_feature(feature_names[idx])
        sentence = f"{description} {direction}."
        if sentence in seen:
            continue  # multiple correlated windows of the same signal shouldn't repeat identical text
        seen.add(sentence)
        factors.append(sentence)
    return factors


def _top_contributing_factors(pipeline, row: pd.DataFrame, max_factors: int = 3) -> list[str]:
    """Retained for the legacy plain-sentence `contributing_factors` field.
    Deliberately still uses the deterministic linear-contribution
    decomposition (not SHAP) for this sentence-generation purpose -- it is
    unrelated to the Phase 8C structured `explanation_factors` field, which
    is computed separately via model_explainability.explain_row()."""
    contributions, feature_names = model_explainability.linear_contributions(pipeline, row)
    if contributions is None:
        return []
    return _factors_from_contributions(contributions[0], feature_names, max_factors)


class RiskDetectionAgent:
    """AGENT 1. Stateless except for the loaded model bundle (cheap to
    construct once and reuse across requests)."""

    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT_PATH, metadata_path: Path = DEFAULT_METADATA_PATH):
        bundle = load_model_bundle(artifact_path, metadata_path)
        self.artifact = bundle["artifact"]
        self.metadata = bundle["metadata"]
        self.pipeline = self.artifact["pipeline"]
        self.feature_columns: list[str] = self.artifact["feature_columns"]
        self.moderate_threshold: float = self.artifact["moderate_threshold"]
        self.high_threshold: float = self.artifact["high_threshold"]
        self.model_version: str = self.artifact["model_version"]
        self.dataset_id: str = self.artifact["dataset_id"]
        self.synthetic_model: bool = self.artifact["synthetic"]
        # Cache key for model_explainability's module-level SHAP explainer
        # cache -- keyed by artifact path so a distinct model on disk never
        # reuses another model's cached explainer.
        self._explainability_cache_key: str = str(artifact_path)

    def validate_feature_schema(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ModelIncompatibleError(f"Input is missing required model features: {missing}")

    def assess(self, member_id: str, features_row: pd.DataFrame, index_date: date) -> RiskAssessment:
        """features_row: a single-row DataFrame containing at least
        self.feature_columns, already built by the point-in-time feature
        pipeline (backend/pit/features.py) for this member as of
        index_date. This agent never computes features itself -- it only
        consumes an already-leakage-safe feature row."""
        self.validate_feature_schema(features_row)
        X = features_row[self.feature_columns]

        probability = float(self.pipeline.predict_proba(X)[:, 1][0])
        tier = assign_risk_tier(probability, self.moderate_threshold, self.high_threshold)
        factors = _top_contributing_factors(self.pipeline, X)
        explanation_factors, explanation_method = model_explainability.explain_row(
            self.pipeline, self.feature_columns, X, self._explainability_cache_key
        )

        return RiskAssessment(
            member_id=member_id,
            probability=round(probability, 6),
            tier=tier,
            contributing_factors=factors,
            model_version=self.model_version,
            dataset_id=self.dataset_id,
            synthetic_model=self.synthetic_model,
            index_date=index_date,
            moderate_threshold=self.moderate_threshold,
            high_threshold=self.high_threshold,
            explanation_factors=explanation_factors,
            explanation_method=explanation_method,
            explanation_causal=False,
        )

    def assess_batch(self, member_ids: list[str], features: pd.DataFrame, index_date: date) -> dict[str, RiskAssessment]:
        """Vectorized equivalent of calling assess() once per member --
        one predict_proba call and one contribution matrix for the whole
        population, instead of one per row. Produces identical results to
        calling assess() member-by-member (covered by a dedicated
        equivalence test); exists purely so scoring a full census stays
        fast."""
        self.validate_feature_schema(features)
        X = features[self.feature_columns]

        probabilities = self.pipeline.predict_proba(X)[:, 1]
        contributions, feature_names = model_explainability.linear_contributions(self.pipeline, X)
        explanation_factors_batch, explanation_method = model_explainability.explain_batch(
            self.pipeline, self.feature_columns, X, self._explainability_cache_key
        )

        results: dict[str, RiskAssessment] = {}
        for i, member_id in enumerate(member_ids):
            probability = float(probabilities[i])
            tier = assign_risk_tier(probability, self.moderate_threshold, self.high_threshold)
            if contributions is not None:
                factors = _factors_from_contributions(contributions[i], feature_names)
            else:
                factors = []
            results[member_id] = RiskAssessment(
                member_id=member_id,
                probability=round(probability, 6),
                tier=tier,
                contributing_factors=factors,
                model_version=self.model_version,
                dataset_id=self.dataset_id,
                synthetic_model=self.synthetic_model,
                index_date=index_date,
                moderate_threshold=self.moderate_threshold,
                high_threshold=self.high_threshold,
                explanation_factors=explanation_factors_batch[i],
                explanation_method=explanation_method,
                explanation_causal=False,
            )
        return results
