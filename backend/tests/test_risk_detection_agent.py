"""
Tests for backend/agents/risk_detection.py (AGENT 1).

Covers model/metadata loading + compatibility checks, frozen-threshold
boundary behavior, real end-to-end scoring against the actual
uc07-risk-synthetic-v1 artifact, determinism, and structural proof that
this agent has no navigation/safety authority.
"""
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import risk_detection
from contracts import RiskTier
from feature_spec import load_snapshot_xy
from risk_detection import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_METADATA_PATH,
    ModelIncompatibleError,
    RiskDetectionAgent,
    assign_risk_tier,
    load_model_bundle,
)

SYNTHETIC_DERIVED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "derived" / "synthetic"
SYNTHETIC_TEST_CSV = SYNTHETIC_DERIVED_DIR / "test_snapshot.csv"
SYNTHETIC_MANIFEST = SYNTHETIC_DERIVED_DIR / "feature_manifest.json"


# ---- model/metadata loading + compatibility ----

def test_load_model_bundle_succeeds_with_real_artifact():
    bundle = load_model_bundle()
    assert bundle["artifact"]["model_version"] == "uc07-risk-synthetic-v1"
    assert bundle["artifact"]["dataset_id"] == "synthetic_uc07_v1"
    assert bundle["artifact"]["synthetic"] is True


def test_load_model_bundle_missing_artifact_raises(tmp_path):
    with pytest.raises(ModelIncompatibleError):
        load_model_bundle(artifact_path=tmp_path / "nonexistent.joblib", metadata_path=DEFAULT_METADATA_PATH)


def test_load_model_bundle_missing_metadata_raises(tmp_path):
    with pytest.raises(ModelIncompatibleError):
        load_model_bundle(artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=tmp_path / "nonexistent.json")


def test_load_model_bundle_detects_feature_mismatch(tmp_path):
    bad_metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    bad_metadata["feature_list"] = ["only_one_feature"]
    bad_path = tmp_path / "bad_metadata.json"
    bad_path.write_text(json.dumps(bad_metadata))
    with pytest.raises(ModelIncompatibleError, match="feature_columns"):
        load_model_bundle(artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=bad_path)


def test_load_model_bundle_detects_threshold_mismatch(tmp_path):
    bad_metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    bad_metadata["moderate_threshold"] = 0.999
    bad_path = tmp_path / "bad_metadata.json"
    bad_path.write_text(json.dumps(bad_metadata))
    with pytest.raises(ModelIncompatibleError, match="threshold"):
        load_model_bundle(artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=bad_path)


def test_load_model_bundle_detects_invalid_version_mismatch(tmp_path):
    bad_metadata = json.loads(DEFAULT_METADATA_PATH.read_text())
    bad_metadata["model_version"] = "some-other-version"
    bad_path = tmp_path / "bad_metadata.json"
    bad_path.write_text(json.dumps(bad_metadata))
    with pytest.raises(ModelIncompatibleError, match="model_version"):
        load_model_bundle(artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=bad_path)


# ---- frozen threshold boundary behavior ----

def test_assign_risk_tier_boundaries():
    moderate, high = 0.105986, 0.213252
    assert assign_risk_tier(0.0, moderate, high) == RiskTier.LOW
    assert assign_risk_tier(moderate - 1e-9, moderate, high) == RiskTier.LOW
    assert assign_risk_tier(moderate, moderate, high) == RiskTier.MODERATE  # inclusive lower bound
    assert assign_risk_tier((moderate + high) / 2, moderate, high) == RiskTier.MODERATE
    assert assign_risk_tier(high - 1e-9, moderate, high) == RiskTier.MODERATE
    assert assign_risk_tier(high, moderate, high) == RiskTier.HIGH  # inclusive lower bound
    assert assign_risk_tier(1.0, moderate, high) == RiskTier.HIGH


def test_frozen_thresholds_match_spec():
    agent = RiskDetectionAgent()
    assert agent.moderate_threshold == pytest.approx(0.105986)
    assert agent.high_threshold == pytest.approx(0.213252)


# ---- end-to-end scoring against the real artifact ----

@pytest.fixture(scope="module")
def agent():
    return RiskDetectionAgent()


@pytest.fixture(scope="module")
def real_test_row():
    X, y, member_ids = load_snapshot_xy(SYNTHETIC_TEST_CSV, manifest_path=SYNTHETIC_MANIFEST)
    return X.iloc[[0]], member_ids.iloc[0]


def test_assess_returns_valid_probability_and_tier(agent, real_test_row):
    row, member_id = real_test_row
    from datetime import date
    result = agent.assess(member_id, row, date(2026, 4, 3))
    assert 0.0 <= result.probability <= 1.0
    assert result.tier in (RiskTier.LOW, RiskTier.MODERATE, RiskTier.HIGH)
    assert result.model_version == "uc07-risk-synthetic-v1"
    assert result.dataset_id == "synthetic_uc07_v1"
    assert result.synthetic_model is True


def test_assess_contributing_factors_are_readable_not_raw_feature_names(agent, real_test_row):
    row, member_id = real_test_row
    from datetime import date
    result = agent.assess(member_id, row, date(2026, 4, 3))
    assert len(result.contributing_factors) <= 3
    for factor in result.contributing_factors:
        assert isinstance(factor, str)
        # must not leak a raw column name like "prior_ed_count_270d"
        assert "_270d" not in factor and "_90d" not in factor and "_30d" not in factor and "_180d" not in factor
        assert factor.endswith(".")


def test_assess_is_deterministic(agent, real_test_row):
    row, member_id = real_test_row
    from datetime import date
    r1 = agent.assess(member_id, row, date(2026, 4, 3))
    r2 = agent.assess(member_id, row, date(2026, 4, 3))
    assert r1.probability == r2.probability
    assert r1.tier == r2.tier
    assert r1.contributing_factors == r2.contributing_factors


def test_validate_feature_schema_raises_on_missing_columns(agent):
    bad_row = pd.DataFrame({"age": [50]})
    with pytest.raises(ModelIncompatibleError):
        agent.validate_feature_schema(bad_row)


# ---- vectorized batch scoring must match per-member scoring exactly ----

def test_assess_batch_matches_per_member_assess(agent):
    from datetime import date
    X, y, member_ids = load_snapshot_xy(SYNTHETIC_TEST_CSV, manifest_path=SYNTHETIC_MANIFEST)
    sample = X.head(25).copy()
    sample_ids = member_ids.head(25).tolist()
    index_date = date(2026, 4, 3)

    batch_results = agent.assess_batch(sample_ids, sample, index_date)

    for i, member_id in enumerate(sample_ids):
        single = agent.assess(member_id, sample.iloc[[i]], index_date)
        batch = batch_results[member_id]
        assert single.probability == batch.probability
        assert single.tier == batch.tier
        assert single.contributing_factors == batch.contributing_factors


# ---- structural: no navigation/safety authority ----

def test_risk_detection_module_never_imports_navigation_or_safety():
    source = inspect.getsource(risk_detection)
    assert "import care_navigation" not in source
    assert "import safety_policy" not in source
    assert "NavigationDecision" not in source
    assert "SafetyDecision" not in source


def test_risk_assessment_has_no_navigation_or_safety_fields():
    import dataclasses
    from contracts import RiskAssessment
    field_names = {f.name for f in dataclasses.fields(RiskAssessment)}
    assert "destination" not in field_names
    assert "safety_state" not in field_names
    assert "navigation" not in field_names
