"""
Phase 4 automated model tests (spec Step 21). Uses a session-scoped
fixture that runs the real training pipeline once (backend/modeling/
train.py::main()) against the real, frozen Phase 3 snapshots -- this is
the only way to honestly verify "re-running evaluation does not mutate
snapshot files" and "snapshot SHA-256 hashes unchanged" (items 14-15).
"""
import inspect
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import train as train_mod
from feature_spec import (
    IDENTIFIER_COLUMN,
    METADATA_COLUMN,
    TARGET_COLUMN,
    load_manifest,
    load_model_feature_columns,
    load_snapshot_xy,
)
from risk_tiers import HIGH, LOW, MODERATE, assign_risk_tier, assign_risk_tiers, validate_thresholds


@pytest.fixture(scope="session")
def training_result():
    return train_mod.main()


@pytest.fixture(scope="session")
def loaded_artifact(training_result):
    return joblib.load(training_result["artifact_path"])


@pytest.fixture(scope="session")
def loaded_metadata(training_result):
    return json.loads(training_result["metadata_path"].read_text())


# ---- 1. Training pipeline uses manifest model features only ----

def test_feature_columns_come_from_manifest_model_candidates():
    manifest = load_manifest()
    expected = [f["feature_name"] for f in manifest["features"] if f["model_candidate"]]
    actual = load_model_feature_columns()
    assert actual == expected
    assert len(actual) == 59


# ---- 2/3/4. target / member_id / index_date never in X ----

def test_target_member_id_index_date_never_in_feature_columns():
    feature_columns = load_model_feature_columns()
    assert TARGET_COLUMN not in feature_columns
    assert IDENTIFIER_COLUMN not in feature_columns
    assert METADATA_COLUMN not in feature_columns


def test_load_snapshot_xy_excludes_identifier_metadata_target(training_result):
    X, y, member_ids = load_snapshot_xy(train_mod.TRAIN_CSV)
    assert IDENTIFIER_COLUMN not in X.columns
    assert METADATA_COLUMN not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(member_ids) == len(X)


# ---- 5. Training and inference feature order match ----

def test_feature_order_matches_between_snapshots_and_artifact(loaded_artifact):
    X_train, _, _ = load_snapshot_xy(train_mod.TRAIN_CSV)
    X_test, _, _ = load_snapshot_xy(train_mod.TEST_CSV)
    assert list(X_train.columns) == loaded_artifact["feature_columns"]
    assert list(X_test.columns) == loaded_artifact["feature_columns"]


# ---- 6/7. Model can serialize/deserialize and produces valid probabilities ----

def test_artifact_round_trips_and_produces_valid_probabilities(training_result, loaded_artifact):
    X_test, y_test, _ = load_snapshot_xy(train_mod.TEST_CSV)
    pipeline = loaded_artifact["pipeline"]
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert len(proba) == len(X_test)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert not np.isnan(proba).any()


# ---- 8. Thresholds satisfy 0 <= moderate < high <= 1 ----

def test_thresholds_are_valid(loaded_artifact):
    moderate = loaded_artifact["moderate_threshold"]
    high = loaded_artifact["high_threshold"]
    validate_thresholds(moderate, high)  # raises if invalid
    assert 0.0 <= moderate < high <= 1.0


def test_validate_thresholds_rejects_invalid_pairs():
    with pytest.raises(ValueError):
        validate_thresholds(0.5, 0.5)
    with pytest.raises(ValueError):
        validate_thresholds(0.7, 0.3)
    with pytest.raises(ValueError):
        validate_thresholds(-0.1, 0.5)
    with pytest.raises(ValueError):
        validate_thresholds(0.5, 1.1)


# ---- 9. Risk-tier function maps probabilities correctly ----

def test_risk_tier_scalar_mapping():
    assert assign_risk_tier(0.01, 0.1, 0.2) == LOW
    assert assign_risk_tier(0.1, 0.1, 0.2) == MODERATE  # inclusive lower bound
    assert assign_risk_tier(0.15, 0.1, 0.2) == MODERATE
    assert assign_risk_tier(0.2, 0.1, 0.2) == HIGH  # inclusive lower bound
    assert assign_risk_tier(0.99, 0.1, 0.2) == HIGH


def test_risk_tier_vectorized_matches_scalar():
    probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
    moderate, high = 0.1, 0.2
    vectorized = list(assign_risk_tiers(probs, moderate, high))
    scalar = [assign_risk_tier(p, moderate, high) for p in probs]
    assert vectorized == scalar


# ---- 10. Model artifact metadata matches feature manifest ----

def test_metadata_feature_list_matches_manifest(loaded_metadata):
    manifest_features = load_model_feature_columns()
    assert loaded_metadata["feature_list"] == manifest_features
    assert loaded_metadata["feature_count"] == len(manifest_features)


def test_artifact_feature_columns_match_metadata(loaded_artifact, loaded_metadata):
    assert loaded_artifact["feature_columns"] == loaded_metadata["feature_list"]


# ---- 11. Model artifact does not depend on legacy frequent_ED_user ----

def test_artifact_target_is_the_phase3_target(loaded_artifact, loaded_metadata):
    assert loaded_artifact["target"] == TARGET_COLUMN
    assert loaded_metadata["target"] == TARGET_COLUMN
    assert loaded_artifact["target"] != "frequent_ED_user"


def test_modeling_source_never_references_legacy_target():
    from pathlib import Path
    modeling_dir = Path(__file__).resolve().parent.parent / "modeling"
    hits = {}
    for path in modeling_dir.glob("*.py"):
        text = path.read_text()
        if "frequent_ED_user" in text or 'ED_visits_365d"] >= 2' in text:
            hits[path.name] = True
    assert hits == {}, f"legacy target referenced in backend/modeling source: {hits}"


# ---- 12. Legacy ed_risk_model.pkl is untouched ----

def test_legacy_model_artifact_untouched(training_result):
    from pathlib import Path
    legacy_path = Path(train_mod.REPO_ROOT) / "backend" / "ed_risk_model.pkl"
    assert legacy_path.exists()
    # Untouched means: still loadable in its original legacy dict shape.
    legacy = joblib.load(legacy_path)
    assert isinstance(legacy, dict)
    assert legacy.get("target") == "frequent_ED_user"


# ---- 13. TEST is not referenced during candidate selection code ----

def test_select_model_on_validation_has_no_test_parameter():
    sig = inspect.signature(train_mod.select_model_on_validation)
    param_names = list(sig.parameters)
    assert not any("test" in name.lower() for name in param_names), param_names


def test_select_model_on_validation_source_has_no_test_identifiers():
    """Static check: the selection function's own source code must not
    reference any TEST-related identifier (X_test/y_test/test_ids/etc).
    Combined with the signature check above, TEST data cannot reach this
    function either by parameter or by closure/global reference."""
    source = inspect.getsource(train_mod.select_model_on_validation)
    forbidden = ["X_test", "y_test", "test_ids", "TEST_CSV", "test_prob", "test_snapshot"]
    hits = [tok for tok in forbidden if tok in source]
    assert hits == [], f"select_model_on_validation source references TEST identifiers: {hits}"


# ---- 14/15. Re-running evaluation does not mutate snapshots; hashes unchanged ----

def test_snapshot_hashes_unchanged_by_training_run(training_result):
    assert training_result["snapshot_hashes_before"] == training_result["snapshot_hashes_after"]


def test_snapshot_hashes_match_phase3_frozen_values(training_result):
    expected = {
        "train": "1b6799904302398d95b478ca2a1e33d0b206fcc1983151b743f01cdbb7a534eb",
        "validation": "a19dad00c4a8329074f7dcba94357506fd004ff7798c82d7d8b7313f13c9b70f",
        "test": "1d4e8b22ede975cdad43379dd0566d38b597e270e8dbd1fcf3a1d85d8989ac1a",
    }
    assert training_result["snapshot_hashes_before"] == expected
    assert training_result["snapshot_hashes_after"] == expected


# ---- extra: reproducibility, tier monotonicity, artifact-vs-metadata thresholds ----

def test_tier_report_prevalence_is_monotonic_on_test(training_result):
    rows = {r["tier"]: r for r in training_result["test_eval"]["test_tier_report"]}
    assert rows[LOW]["observed_prevalence"] <= rows[MODERATE]["observed_prevalence"]
    assert rows[MODERATE]["observed_prevalence"] <= rows[HIGH]["observed_prevalence"]


def test_thresholds_consistent_across_artifact_metadata_and_selection(loaded_artifact, loaded_metadata, training_result):
    assert loaded_artifact["moderate_threshold"] == loaded_metadata["moderate_threshold"]
    assert loaded_artifact["high_threshold"] == loaded_metadata["high_threshold"]
    assert loaded_artifact["moderate_threshold"] == training_result["frozen_spec"]["moderate_threshold"]
    assert loaded_artifact["high_threshold"] == training_result["frozen_spec"]["high_threshold"]


def test_raw_dataset_hashes_unchanged_by_training_run(training_result):
    expected = {
        "raw_members.csv": "b94df89ed042a8feaa1bb46d7939e124fb9f6b03308b11da045412a427b78c46",
        "raw_ed_visits.csv": "f8db1839fb7966c4230c771252a3b935c318d0838de9258dc29de42d042f5d47",
        "raw_care_history.csv": "358d3033faa4e0529aed834cd8847f72d0b5d4ca51fa76748523fab790c81657",
    }
    assert training_result["raw_hashes"] == expected
