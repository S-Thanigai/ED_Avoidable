"""
Phase 4D automated tests (spec Step 25): verifies the synthetic model
training pipeline (backend/modeling/train_synthetic.py) uses only the
synthetic Phase 4C snapshots, never touches the original snapshots or
TEST during selection, and produces a correctly-labeled, valid artifact
that never overwrites uc07-risk-v1.

SYNTHETIC DATA MODEL -- DEMONSTRATION ONLY.
"""
import inspect
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import train_synthetic as ts
from feature_spec import IDENTIFIER_COLUMN, METADATA_COLUMN, TARGET_COLUMN, load_snapshot_xy
from risk_tiers import HIGH, LOW, MODERATE, validate_thresholds


@pytest.fixture(scope="session")
def synthetic_result():
    return ts.main()


@pytest.fixture(scope="session")
def loaded_artifact(synthetic_result):
    return joblib.load(synthetic_result["artifact_path"])


@pytest.fixture(scope="session")
def loaded_metadata(synthetic_result):
    return json.loads(synthetic_result["metadata_path"].read_text())


# ---- 1/2. synthetic model uses only synthetic snapshots; original snapshots not used ----

def test_paths_point_only_at_synthetic_snapshots():
    assert "synthetic" in str(ts.TRAIN_CSV)
    assert "synthetic" in str(ts.VALIDATION_CSV)
    assert "synthetic" in str(ts.TEST_CSV)
    assert ts.TRAIN_CSV.parent == ts.SYNTHETIC_DERIVED_DIR
    assert ts.SYNTHETIC_DERIVED_DIR == ts.REPO_ROOT / "data" / "derived" / "synthetic"


def test_original_snapshot_paths_never_referenced_in_training_source():
    """train_synthetic.py must define its snapshot paths from
    SYNTHETIC_DERIVED_DIR only -- never from the original (unqualified)
    DERIVED_DIR constant train.py/build_snapshots.py use. Checks the
    precise assignment form so "SYNTHETIC_DERIVED_DIR" (which contains
    "DERIVED_DIR" as a substring) doesn't produce a false positive."""
    source = inspect.getsource(ts)
    assert 'TRAIN_CSV = SYNTHETIC_DERIVED_DIR / "train_snapshot.csv"' in source
    assert 'TRAIN_CSV = DERIVED_DIR / "train_snapshot.csv"' not in source
    assert ts.TRAIN_CSV == ts.REPO_ROOT / "data" / "derived" / "synthetic" / "train_snapshot.csv"


def test_training_rows_match_synthetic_snapshot_files(synthetic_result):
    train_df = pd.read_csv(ts.TRAIN_CSV)
    val_df = pd.read_csv(ts.VALIDATION_CSV)
    assert len(train_df) == 10000
    assert len(val_df) == 10000
    assert abs(train_df[TARGET_COLUMN].mean() - 0.1194) < 1e-6
    assert abs(val_df[TARGET_COLUMN].mean() - 0.1174) < 1e-6


# ---- 3/4/5. target/member_id/index_date excluded from X ----

def test_target_member_id_index_date_excluded_from_features(loaded_artifact):
    feature_columns = loaded_artifact["feature_columns"]
    assert TARGET_COLUMN not in feature_columns
    assert IDENTIFIER_COLUMN not in feature_columns
    assert METADATA_COLUMN not in feature_columns


def test_load_snapshot_xy_synthetic_excludes_identifier_metadata_target():
    X, y, member_ids = load_snapshot_xy(ts.TRAIN_CSV, manifest_path=ts.SYNTHETIC_MANIFEST_PATH)
    assert IDENTIFIER_COLUMN not in X.columns
    assert METADATA_COLUMN not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert set(y.unique()).issubset({0, 1})


# ---- 6. feature ordering matches manifest ----

def test_feature_order_matches_manifest(loaded_artifact):
    from feature_spec import load_model_feature_columns
    expected = load_model_feature_columns(manifest_path=ts.SYNTHETIC_MANIFEST_PATH)
    assert loaded_artifact["feature_columns"] == expected
    assert len(expected) == 59


# ---- 7/8. no legacy target, no diagnosis crosstab ----

def test_no_legacy_target_or_diagnosis_columns():
    train_df = pd.read_csv(ts.TRAIN_CSV)
    assert "frequent_ED_user" not in train_df.columns
    assert not any(c.startswith("diagnosis_") for c in train_df.columns)


# ---- 9. probabilities within [0,1] ----

def test_probabilities_within_valid_range(loaded_artifact):
    X_test, y_test, _ = load_snapshot_xy(ts.TEST_CSV, manifest_path=ts.SYNTHETIC_MANIFEST_PATH)
    proba = loaded_artifact["pipeline"].predict_proba(X_test)[:, 1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert not np.isnan(proba).any()


# ---- 10. moderate_threshold < high_threshold ----

def test_thresholds_valid(loaded_artifact):
    moderate = loaded_artifact["moderate_threshold"]
    high = loaded_artifact["high_threshold"]
    validate_thresholds(moderate, high)
    assert 0.0 <= moderate < high <= 1.0


# ---- 11. LOW/MODERATE/HIGH assignment works ----

def test_risk_tier_assignment(loaded_artifact):
    from risk_tiers import assign_risk_tiers
    moderate, high = loaded_artifact["moderate_threshold"], loaded_artifact["high_threshold"]
    probs = [0.0, moderate, (moderate + high) / 2, high, 1.0]
    tiers = list(assign_risk_tiers(probs, moderate, high))
    assert tiers[0] == LOW
    assert tiers[1] == MODERATE
    assert tiers[3] == HIGH


# ---- 12. model serialization/deserialization works ----

def test_artifact_round_trips(synthetic_result, tmp_path):
    reloaded = joblib.load(synthetic_result["artifact_path"])
    X_test, _, _ = load_snapshot_xy(ts.TEST_CSV, manifest_path=ts.SYNTHETIC_MANIFEST_PATH)
    proba = reloaded["pipeline"].predict_proba(X_test.head(5))[:, 1]
    assert len(proba) == 5


# ---- 13/14. metadata synthetic=true, dataset_id correct ----

def test_metadata_synthetic_flag_and_dataset_id(loaded_metadata):
    assert loaded_metadata["synthetic"] is True
    assert loaded_metadata["dataset_id"] == "synthetic_uc07_v1"
    assert loaded_metadata["model_version"] == "uc07-risk-synthetic-v1"
    assert "synthetic data" in loaded_metadata["disclaimer"].lower()


def test_artifact_carries_synthetic_labels(loaded_artifact):
    assert loaded_artifact["dataset_id"] == "synthetic_uc07_v1"
    assert loaded_artifact["synthetic"] is True
    assert loaded_artifact["model_version"] == "uc07-risk-synthetic-v1"


# ---- 15/16/17/18. immutability of everything else ----

def test_original_v1_model_unchanged(synthetic_result):
    original = joblib.load(ts.REPO_ROOT / "backend" / "models" / "uc07_risk_v1_model.joblib")
    assert original["model_version"] == "uc07-risk-v1"
    assert original["target"] == TARGET_COLUMN


def test_original_v1_artifact_path_never_written_by_synthetic_script():
    source = inspect.getsource(ts)
    assert "uc07_risk_v1_model.joblib" not in source or "uc07_risk_synthetic_v1_model.joblib" in source
    # the synthetic script must write ONLY the synthetic-named artifact
    assert 'uc07_risk_synthetic_v1_model.joblib' in source


def test_raw_original_datasets_unchanged():
    expected = {
        "raw_members.csv": "b94df89ed042a8feaa1bb46d7939e124fb9f6b03308b11da045412a427b78c46",
        "raw_ed_visits.csv": "f8db1839fb7966c4230c771252a3b935c318d0838de9258dc29de42d042f5d47",
        "raw_care_history.csv": "358d3033faa4e0529aed834cd8847f72d0b5d4ca51fa76748523fab790c81657",
    }
    for name, expected_hash in expected.items():
        actual = ts.sha256_file(ts.REPO_ROOT / name)
        assert actual == expected_hash


def test_raw_synthetic_datasets_unchanged(synthetic_result):
    expected = {
        "raw_members.csv": "00cb4023eb20876fd9b9cd2b3b3e283c8e6681f1452a6c3e9cbfda37f0bd2373",
        "raw_ed_visits.csv": "bb3c9505a836b8c70813aa2fdd62f628bd871f657fa1dfca1330799d27ce88c0",
        "raw_care_history.csv": "20fdcb836f6abbbd1b1b70d7c1f7cd2279f5c519251322944b0ca7109a66db1a",
    }
    assert synthetic_result["raw_hashes_before"] == expected
    assert synthetic_result["raw_hashes_after"] == expected


def test_synthetic_snapshots_unchanged_by_training(synthetic_result):
    expected = {
        "train": "4a8b79cd779a15448117301574c3100a683b2e5547f01ed43469afb34f3ad50c",
        "validation": "afc328b3de95f5237d55c276235b7edd19e5081b122c81995e8a84cb50b05c56",
        "test": "5657522789d1ccb8dc884209843cd4a4ca283892f39494e058ceb4431e76d7d8",
    }
    assert synthetic_result["snapshot_hashes_before"] == expected
    assert synthetic_result["snapshot_hashes_after"] == expected


# ---- 19. TEST cannot be used by candidate-selection code ----

def test_select_model_on_validation_reused_has_no_test_parameter():
    """train_synthetic.py calls v1_train.select_model_on_validation()
    unmodified -- reconfirm here that the reused function still has no
    TEST-named parameter (regression guard shared with Phase 4's own test)."""
    sig = inspect.signature(ts.v1_train.select_model_on_validation)
    assert not any("test" in name.lower() for name in sig.parameters)


def test_train_synthetic_main_loads_test_only_after_freeze():
    source = inspect.getsource(ts.main)
    freeze_idx = source.index("frozen_at_utc")
    test_load_idx = source.index("load_snapshot_xy(TEST_CSV")
    assert freeze_idx < test_load_idx, "TEST must be loaded only after the frozen spec is written"


# ---- 20. final artifact can score a valid synthetic snapshot row ----

def test_artifact_scores_a_real_synthetic_row(loaded_artifact):
    X_test, y_test, member_ids = load_snapshot_xy(ts.TEST_CSV, manifest_path=ts.SYNTHETIC_MANIFEST_PATH)
    row = X_test.iloc[[0]]
    proba = loaded_artifact["pipeline"].predict_proba(row)[:, 1]
    assert len(proba) == 1
    assert 0.0 <= proba[0] <= 1.0


# ---- extra: reports and tier monotonicity ----

def test_evaluation_reports_written(synthetic_result):
    for name in (
        "candidate_metrics.csv", "calibration_comparison.csv", "threshold_analysis.csv",
        "validation_risk_tiers.csv", "test_risk_tiers.csv", "subgroup_metrics.csv",
        "global_feature_importance.csv", "original_vs_synthetic_comparison.csv",
        "frozen_model_selection.json", "final_test_results.json",
    ):
        assert (ts.EVAL_DIR / name).exists(), f"missing report: {name}"


def test_test_tier_prevalence_monotonic(synthetic_result):
    rows = {r["tier"]: r for r in synthetic_result["test_eval"]["test_tier_report"]}
    assert rows[LOW]["observed_prevalence"] <= rows[MODERATE]["observed_prevalence"]
    assert rows[MODERATE]["observed_prevalence"] <= rows[HIGH]["observed_prevalence"]


def test_performance_below_suspicious_ceiling(synthetic_result):
    roc_auc = synthetic_result["test_eval"]["test_rank_metrics"]["roc_auc"]
    assert roc_auc < 0.85, f"TEST ROC-AUC {roc_auc} exceeds the suspicious-performance ceiling; audit required"
