"""
Phase 8C Part 17 -- MODEL EXPLANATION tests (numbered 1-8 in the spec).

Covers backend/agents/model_explainability.py and the
RiskAssessment.explanation_factors / explanation_method / explanation_causal
fields it populates via backend/agents/risk_detection.py.
"""
from datetime import date

import math
import numpy as np
import pandas as pd
import pytest

import model_explainability
import risk_detection
from contracts import ExplanationMethod, FactorDirection
from feature_spec import load_snapshot_xy
from risk_detection import RiskDetectionAgent

SYNTHETIC_DERIVED_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent.parent / "data" / "derived" / "synthetic"
SYNTHETIC_TEST_CSV = SYNTHETIC_DERIVED_DIR / "test_snapshot.csv"
SYNTHETIC_MANIFEST = SYNTHETIC_DERIVED_DIR / "feature_manifest.json"

INDEX_DATE = date(2026, 4, 3)


@pytest.fixture(scope="module")
def agent():
    return RiskDetectionAgent()


@pytest.fixture(scope="module")
def sample_rows():
    X, y, member_ids = load_snapshot_xy(SYNTHETIC_TEST_CSV, manifest_path=SYNTHETIC_MANIFEST)
    return X.head(10).reset_index(drop=True), member_ids.head(10).reset_index(drop=True)


# ---- 1. generated from the actual model (not hardcoded/templated) ----

def test_1_explanation_factors_generated_from_real_model(agent, sample_rows):
    X, member_ids = sample_rows
    result = agent.assess(member_ids.iloc[0], X.iloc[[0]], INDEX_DATE)
    assert len(result.explanation_factors) > 0
    feature_names = {f.feature for f in result.explanation_factors}
    # every explained feature must be one of the real model's own input
    # columns -- never an invented or unrelated name
    assert feature_names.issubset(set(agent.feature_columns))


# ---- 2. finite values ----

def test_2_contribution_values_are_finite(agent, sample_rows):
    X, member_ids = sample_rows
    for i in range(len(X)):
        result = agent.assess(member_ids.iloc[i], X.iloc[[i]], INDEX_DATE)
        for factor in result.explanation_factors:
            assert math.isfinite(factor.contribution)


# ---- 3. direction matches the sign of the contribution ----

def test_3_direction_matches_contribution_sign(agent, sample_rows):
    X, member_ids = sample_rows
    for i in range(len(X)):
        result = agent.assess(member_ids.iloc[i], X.iloc[[i]], INDEX_DATE)
        for factor in result.explanation_factors:
            if factor.contribution > 0:
                assert factor.direction == FactorDirection.INCREASES_RISK
            elif factor.contribution < 0:
                assert factor.direction == FactorDirection.DECREASES_RISK


# ---- 4. max factor limits: <=3 increasing, <=2 decreasing ----

def test_4_max_factor_limits_respected(agent, sample_rows):
    X, member_ids = sample_rows
    for i in range(len(X)):
        result = agent.assess(member_ids.iloc[i], X.iloc[[i]], INDEX_DATE)
        increasing = [f for f in result.explanation_factors if f.direction == FactorDirection.INCREASES_RISK]
        decreasing = [f for f in result.explanation_factors if f.direction == FactorDirection.DECREASES_RISK]
        assert len(increasing) <= 3
        assert len(decreasing) <= 2
        assert len(result.explanation_factors) <= 5


# ---- 5. no leakage / no features outside the frozen, leakage-safe set ----

def test_5_no_features_outside_frozen_model_feature_set(agent, sample_rows):
    X, member_ids = sample_rows
    result = agent.assess(member_ids.iloc[0], X.iloc[[0]], INDEX_DATE)
    for factor in result.explanation_factors:
        assert factor.feature in agent.feature_columns
    # explicitly: no target/outcome column can ever appear as a factor
    assert not any("future_potentially_avoidable_ed" in f.feature for f in result.explanation_factors)


# ---- 6. correct method/model-version/causal metadata ----

def test_6_explanation_metadata_correct(agent, sample_rows):
    X, member_ids = sample_rows
    result = agent.assess(member_ids.iloc[0], X.iloc[[0]], INDEX_DATE)
    assert result.explanation_method in (ExplanationMethod.SHAP_LINEAR, ExplanationMethod.LINEAR_CONTRIBUTION)
    assert result.model_version == agent.model_version
    assert result.explanation_causal is False
    for factor in result.explanation_factors:
        assert factor.explanation_method == result.explanation_method


# ---- 7. deterministic ----

def test_7_explanation_is_deterministic(agent, sample_rows):
    X, member_ids = sample_rows
    r1 = agent.assess(member_ids.iloc[0], X.iloc[[0]], INDEX_DATE)
    r2 = agent.assess(member_ids.iloc[0], X.iloc[[0]], INDEX_DATE)
    assert r1.explanation_method == r2.explanation_method
    assert [(f.feature, f.direction, f.contribution) for f in r1.explanation_factors] == \
           [(f.feature, f.direction, f.contribution) for f in r2.explanation_factors]


# ---- 8. no causal wording anywhere in the structured output ----

_CAUSAL_WORDS = ("cause", "causes", "caused", "because", "results in", "leads to", "will happen", "proves")


def test_8_no_causal_wording_in_display_names_or_feature_names(agent, sample_rows):
    X, member_ids = sample_rows
    result = agent.assess(member_ids.iloc[0], X.iloc[[0]], INDEX_DATE)
    for factor in result.explanation_factors:
        normalized = f"{factor.display_name} {factor.feature}".lower()
        for word in _CAUSAL_WORDS:
            assert word not in normalized


def test_8_direction_enum_has_no_causal_values():
    # FactorDirection must only ever express sign, never causation
    assert {d.value for d in FactorDirection} == {"INCREASES_RISK", "DECREASES_RISK"}


# ---- SHAP vs linear-contribution: primary path + safe fallback ----

def test_shap_is_the_primary_method_when_available(agent, sample_rows):
    X, member_ids = sample_rows
    result = agent.assess(member_ids.iloc[0], X.iloc[[0]], INDEX_DATE)
    assert result.explanation_method == ExplanationMethod.SHAP_LINEAR


def test_falls_back_to_linear_contribution_if_shap_unavailable(monkeypatch, agent, sample_rows):
    X, member_ids = sample_rows
    monkeypatch.setattr(model_explainability, "shap_contributions", lambda *a, **k: (None, None))
    factors, method = model_explainability.explain_row(
        agent.pipeline, agent.feature_columns, X.iloc[[0]], "forced-fallback-key"
    )
    assert method == ExplanationMethod.LINEAR_CONTRIBUTION
    assert all(f.explanation_method == ExplanationMethod.LINEAR_CONTRIBUTION for f in factors)


def test_explain_row_never_raises_even_if_shap_and_linear_both_fail(monkeypatch, agent, sample_rows):
    X, member_ids = sample_rows
    monkeypatch.setattr(model_explainability, "shap_contributions", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(model_explainability, "linear_contributions", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    factors, method = model_explainability.explain_row(
        agent.pipeline, agent.feature_columns, X.iloc[[0]], "forced-double-failure-key"
    )
    assert factors == []
    assert method == ExplanationMethod.LINEAR_CONTRIBUTION  # safe structured fallback, never a fake value


def test_explain_batch_shares_one_method_across_the_whole_batch(agent, sample_rows):
    X, member_ids = sample_rows
    factors_list, method = model_explainability.explain_batch(
        agent.pipeline, agent.feature_columns, X, agent._explainability_cache_key
    )
    assert len(factors_list) == len(X)
    for factors in factors_list:
        for factor in factors:
            assert factor.explanation_method == method


def test_assess_batch_matches_assess_for_explanation_factors(agent, sample_rows):
    X, member_ids = sample_rows
    ids = member_ids.tolist()
    batch_results = agent.assess_batch(ids, X, INDEX_DATE)
    for i, member_id in enumerate(ids):
        single = agent.assess(member_id, X.iloc[[i]], INDEX_DATE)
        batch = batch_results[member_id]
        assert single.explanation_method == batch.explanation_method
        assert [(f.feature, f.direction, f.contribution) for f in single.explanation_factors] == \
               [(f.feature, f.direction, f.contribution) for f in batch.explanation_factors]


def test_humanize_feature_name_is_readable():
    assert model_explainability.humanize_feature_name("prior_potentially_avoidable_ed_count_270d") == \
        "Prior potentially avoidable ED count (270 days)"
    assert model_explainability.humanize_feature_name("pcp_distance_miles") == "PCP distance miles"
