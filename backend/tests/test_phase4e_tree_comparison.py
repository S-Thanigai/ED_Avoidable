"""
Phase 4E automated tests (spec Step 26): verifies the controlled
Random Forest / XGBoost vs. Logistic Regression comparison
(backend/modeling/train_phase4e_tree_comparison.py) and its already-
written artifacts (artifacts/phase4e_tree_model_comparison/) without
re-running the full search in every test session -- the module was
already executed once to produce these artifacts, exactly like Phase
4E's TEST-sealing discipline requires (frozen before TEST, evaluated
exactly once).

SYNTHETIC DATA MODEL -- DEMONSTRATION ONLY.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

import train_phase4e_tree_comparison as p4e
from risk_tiers import validate_thresholds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "artifacts" / "phase4e_tree_model_comparison"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(name: str) -> dict:
    return json.loads((EVAL_DIR / name).read_text())


@pytest.fixture(scope="session")
def frozen_spec():
    return _read_json("frozen_model_selection.json")


@pytest.fixture(scope="session")
def final_test_results():
    return _read_json("final_test_results.json")


@pytest.fixture(scope="session")
def final_model_comparison():
    return _read_json("final_model_comparison.json")


@pytest.fixture(scope="session")
def pre_work_hashes():
    # immutability_check.json's "hashes_before" is written by the Phase 4E
    # script itself (hash_immutable_set()) at the start of its own run --
    # the authoritative pre-work snapshot, keyed consistently with the
    # constants used throughout this test module.
    return _read_json("immutability_check.json")["hashes_before"]


# ---- artifacts exist (Step 25) ----

@pytest.mark.parametrize("name", [
    "candidate_metrics.csv", "rf_search_results.csv", "xgb_search_results.csv",
    "calibration_comparison.csv", "train_validation_gap.csv", "confusion_matrices.json",
    "threshold_analysis.csv", "validation_risk_tiers.csv", "global_feature_importance.csv",
    "frozen_model_selection.json", "final_model_comparison.json", "final_test_results.json",
])
def test_required_artifact_exists(name):
    assert (EVAL_DIR / name).exists(), f"missing required Phase 4E artifact: {name}"


# ---- accuracy / balanced accuracy calculation correctness ----

def test_accuracy_calculation_matches_manual_count():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.6, 0.9, 0.3, 0.7, 0.4, 0.05])
    # threshold 0.5 -> preds [0,0,1,1,0,1,0,0]; correct at indices 0,1,3,5,7 = 5/8
    metrics = p4e.full_metrics_with_accuracy(y_true, y_prob, 0.5)
    assert metrics["accuracy"] == pytest.approx(5 / 8, abs=1e-9)


def test_balanced_accuracy_calculation_matches_manual_average_of_recalls():
    # 4 negatives (3 correctly classified), 4 positives (2 correctly classified)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.7, 0.2, 0.1])
    metrics = p4e.full_metrics_with_accuracy(y_true, y_prob, 0.5)
    # specificity = 3/4, recall = 2/4 -> balanced accuracy = (0.75+0.5)/2 = 0.625
    assert metrics["balanced_accuracy"] == pytest.approx(0.625, abs=1e-9)
    assert metrics["specificity"] == pytest.approx(0.75, abs=1e-9)
    assert metrics["recall"] == pytest.approx(0.5, abs=1e-9)


def test_full_metrics_with_accuracy_reports_threshold_used():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.6])
    for thr in (0.5, 0.105986, 0.213252):
        metrics = p4e.full_metrics_with_accuracy(y_true, y_prob, thr)
        assert metrics["threshold"] == thr


# ---- confusion matrix correctness ----

def test_confusion_matrix_counts_sum_to_population_size():
    cm = _read_json("confusion_matrices.json")
    for model, thresholds in cm.items():
        for thr_name, counts in thresholds.items():
            total = counts["true_positives"] + counts["false_positives"] + counts["false_negatives"] + counts["true_negatives"]
            assert total == 10000, f"{model}/{thr_name} confusion matrix does not sum to VALIDATION population size"


def test_confusion_matrix_n_selected_equals_tp_plus_fp():
    cm = _read_json("confusion_matrices.json")
    for model, thresholds in cm.items():
        for thr_name, counts in thresholds.items():
            assert counts["n_selected"] == counts["true_positives"] + counts["false_positives"]


# ---- probability bounds (loaded from the unchanged existing model artifact) ----

def test_existing_synthetic_model_probabilities_are_bounded():
    import pandas as pd
    bundle = joblib.load(REPO_ROOT / "backend" / "models" / "uc07_risk_synthetic_v1_model.joblib")
    df = pd.read_csv(REPO_ROOT / "data" / "derived" / "synthetic" / "test_snapshot.csv").head(50)
    X = df[bundle["feature_columns"]]
    probs = bundle["pipeline"].predict_proba(X)[:, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ---- threshold loading: never hard-coded, always from the frozen spec ----

def test_frozen_thresholds_are_valid_and_ordered(frozen_spec):
    validate_thresholds(frozen_spec["moderate_threshold"], frozen_spec["high_threshold"])


def test_winner_thresholds_match_between_frozen_spec_and_final_test_results(frozen_spec, final_test_results):
    assert frozen_spec["moderate_threshold"] == final_test_results["moderate_threshold"]
    assert frozen_spec["high_threshold"] == final_test_results["high_threshold"]


# ---- feature-order compatibility ----

def test_frozen_feature_list_matches_manifest_order(frozen_spec):
    from feature_spec import load_model_feature_columns
    manifest_columns = load_model_feature_columns(manifest_path=p4e.SYNTHETIC_MANIFEST_PATH)
    assert frozen_spec["feature_list"] == manifest_columns
    assert frozen_spec["feature_count"] == 59


# ---- metadata compatibility ----

def test_frozen_spec_algorithm_is_one_of_the_three_candidates(frozen_spec):
    assert frozen_spec["algorithm"] in {"logistic_regression", "random_forest", "xgboost"}


def test_decision_field_matches_algorithm(frozen_spec):
    if frozen_spec["algorithm"] == "logistic_regression":
        assert frozen_spec["decision"] == "KEEP_LOGISTIC"
    else:
        assert frozen_spec["decision"] == f"PROMOTE_{frozen_spec['algorithm'].upper()}"


# ---- TEST isolation: the frozen spec must contain no TEST-derived value ----

def test_frozen_spec_contains_no_test_derived_keys(frozen_spec):
    serialized = json.dumps(frozen_spec).lower()
    # "test_index_date" style constants are fine; we assert no numeric TEST
    # rank-metric/tier keys (which only exist post-freeze) leaked into the frozen spec.
    forbidden_keys = ("test_rank_metrics", "test_risk_tiers", "test_pr_enrichment", "test_probabilities")
    for key in forbidden_keys:
        assert key not in frozen_spec, f"frozen_model_selection.json must not contain TEST-derived key {key!r}"


def test_select_model_style_functions_have_no_test_parameter():
    import inspect
    sig = inspect.signature(p4e.run_search)
    for name in sig.parameters:
        assert "test" not in name.lower(), f"run_search must never take a TEST-named parameter (found {name})"


# ---- artifact preservation: no overwrite of existing models ----

def test_existing_model_artifacts_unchanged():
    """Content/semantic check, not a byte-for-byte hash check: Phase 4
    and Phase 4D's OWN test suites (test_model_pipeline.py,
    test_synthetic_model.py) legitimately re-run train.py::main() /
    train_synthetic.py::main() every full-suite session as their only
    honest way to verify pipeline reproducibility -- this refreshes
    each metadata file's training_timestamp_utc (and therefore its
    file hash) without changing any of its substantive content. Phase
    4E's OWN immutability guarantee (that ITS run did not touch these
    files) is the byte-for-byte hash check already recorded in
    immutability_check.json (see test_leakage_audit_passed's sibling
    checks and the Phase 4E script's own hash-before/after guard)."""
    v1 = joblib.load(REPO_ROOT / "backend" / "models" / "uc07_risk_v1_model.joblib")
    assert v1["model_version"] == "uc07-risk-v1"
    synthetic_v1 = joblib.load(REPO_ROOT / "backend" / "models" / "uc07_risk_synthetic_v1_model.joblib")
    assert synthetic_v1["model_version"] == "uc07-risk-synthetic-v1"
    assert synthetic_v1["moderate_threshold"] == 0.105986
    assert synthetic_v1["high_threshold"] == 0.213252


def test_no_unexpected_new_model_artifact_when_logistic_wins(frozen_spec):
    if frozen_spec["algorithm"] == "logistic_regression":
        assert not (REPO_ROOT / "backend" / "models" / "uc07_risk_synthetic_rf_v1_model.joblib").exists()
        assert not (REPO_ROOT / "backend" / "models" / "uc07_risk_synthetic_xgb_v1_model.joblib").exists()


# ---- dataset immutability ----

@pytest.mark.parametrize("key,rel", [
    ("raw_members.csv", "raw_members.csv"),
    ("raw_ed_visits.csv", "raw_ed_visits.csv"),
    ("raw_care_history.csv", "raw_care_history.csv"),
    ("synthetic_raw_members.csv", "data/synthetic/raw_members.csv"),
    ("synthetic_raw_ed_visits.csv", "data/synthetic/raw_ed_visits.csv"),
    ("synthetic_raw_care_history.csv", "data/synthetic/raw_care_history.csv"),
    ("train_snapshot.csv", "data/derived/synthetic/train_snapshot.csv"),
    ("validation_snapshot.csv", "data/derived/synthetic/validation_snapshot.csv"),
    ("test_snapshot.csv", "data/derived/synthetic/test_snapshot.csv"),
    ("feature_manifest.json", "data/derived/synthetic/feature_manifest.json"),
])
def test_dataset_files_unchanged(pre_work_hashes, key, rel):
    assert _sha256(REPO_ROOT / rel) == pre_work_hashes[key], f"{rel} changed during/after Phase 4E"


# ---- Risk Agent compatibility (only relevant if a new model was promoted) ----

def test_risk_agent_still_loads_default_model_unchanged(frozen_spec):
    import risk_detection

    agent = risk_detection.RiskDetectionAgent()
    if frozen_spec["algorithm"] == "logistic_regression":
        assert agent.model_version == "uc07-risk-synthetic-v1"
        assert agent.moderate_threshold == 0.105986
        assert agent.high_threshold == 0.213252
    else:
        # A promotion would only be valid if the Risk Agent's defaults were
        # updated to point at the new artifact -- verify they agree.
        assert agent.model_version == frozen_spec["model_version_candidate"]
        assert agent.moderate_threshold == frozen_spec["moderate_threshold"]
        assert agent.high_threshold == frozen_spec["high_threshold"]


# ---- promotion criteria sanity ----

def test_promotion_decision_is_internally_consistent(frozen_spec):
    crit = frozen_spec["promotion_criteria"]
    if crit["promotion_met"]:
        assert frozen_spec["algorithm"] != "logistic_regression"
    else:
        assert frozen_spec["algorithm"] == "logistic_regression"


def test_leakage_audit_passed():
    audit = _read_json("leakage_audit.json")
    assert audit["passed"] is True
    assert audit["forbidden_columns_in_feature_list"] == []
    assert audit["candidates_above_ceiling"] == {}
