"""
train_synthetic.py
-------------------
Phase 4D orchestration: trains and evaluates the UC07 risk model against
the Phase 4C SYNTHETIC snapshots.

SYNTHETIC DATA MODEL -- DEMONSTRATION ONLY. See
docs/04D_SYNTHETIC_MODEL_DEVELOPMENT.md. Results here describe whether
the UC07 point-in-time pipeline can learn stronger prospective patterns
when the input data contains them -- they are NOT clinical validation.

This script deliberately does NOT duplicate any modeling logic. It reuses,
unmodified, the exact candidate-model builders, hyperparameter grids,
`select_model_on_validation()`, `evaluate_frozen_model_on_test()`,
`extract_global_feature_importance()`, `run_subgroup_checks()`, and
`write_reports()` from train.py (Phase 4's original-data orchestration) --
every one of those functions is already parameterized on the X/y/paths
passed in, with no hardcoded reference to which dataset produced them.
Only the synthetic-specific paths, naming (`uc07-risk-synthetic-v1`),
and metadata fields required by Step 23 are new in this file.

Run: python backend/modeling/train_synthetic.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as metrics_mod
import risk_tiers as risk_tiers_mod
import train as v1_train
from feature_spec import TARGET_COLUMN, load_model_feature_columns, load_snapshot_xy, split_numeric_categorical

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DERIVED_DIR = REPO_ROOT / "data" / "derived" / "synthetic"
SYNTHETIC_MANIFEST_PATH = SYNTHETIC_DERIVED_DIR / "feature_manifest.json"
TRAIN_CSV = SYNTHETIC_DERIVED_DIR / "train_snapshot.csv"
VALIDATION_CSV = SYNTHETIC_DERIVED_DIR / "validation_snapshot.csv"
TEST_CSV = SYNTHETIC_DERIVED_DIR / "test_snapshot.csv"

SYNTHETIC_RAW = {
    "raw_members.csv": REPO_ROOT / "data" / "synthetic" / "raw_members.csv",
    "raw_ed_visits.csv": REPO_ROOT / "data" / "synthetic" / "raw_ed_visits.csv",
    "raw_care_history.csv": REPO_ROOT / "data" / "synthetic" / "raw_care_history.csv",
}

MODELS_DIR = REPO_ROOT / "backend" / "models"
EVAL_DIR = REPO_ROOT / "artifacts" / "synthetic_model_evaluation"

MODEL_VERSION = "uc07-risk-synthetic-v1"
DATASET_ID = "synthetic_uc07_v1"
INTENDED_USE = "demonstration / UC07 navigation prototype"
SYNTHETIC_DISCLAIMER = "This model was trained and evaluated on synthetic data and must not be interpreted as clinically validated."

# Original v1 (Phase 4) TEST results, reused for the comparison report --
# not recomputed here (v1 is not retrained or re-evaluated in this phase).
ORIGINAL_V1_TEST_METRICS = {
    "roc_auc": 0.574721, "pr_auc": 0.111091, "brier_score": 0.082183,
    "prevalence": 0.0908,
}
ORIGINAL_V1_HIGH_TIER_LIFT = 1.3469
ORIGINAL_V1_ALGORITHM = "logistic_regression"


def sha256_file(path: Path) -> str:
    return v1_train.sha256_file(path)


def load_synthetic_snapshots():
    X_train, y_train, _ = load_snapshot_xy(TRAIN_CSV, manifest_path=SYNTHETIC_MANIFEST_PATH)
    X_val, y_val, val_ids = load_snapshot_xy(VALIDATION_CSV, manifest_path=SYNTHETIC_MANIFEST_PATH)
    return X_train, y_train, X_val, y_val, val_ids


def build_synthetic_metadata(
    winner, best_params_by_family, moderate_threshold, high_threshold,
    feature_columns, numeric_features, categorical_features,
    train_prevalence, validation_prevalence, test_prevalence,
    validation_final_metrics, test_eval: dict,
    validation_tier_report, test_tier_report,
    subgroup_rows, raw_hashes: dict, snapshot_hashes: dict,
    hgb_supports_class_weight: bool,
) -> dict:
    test_pr_enrichment = round(test_eval["test_rank_metrics"]["pr_auc"] / test_prevalence, 4) if test_prevalence else None
    return {
        "model_name": "UC07 Avoidable ED Risk Model (Synthetic Demonstration)",
        "model_version": MODEL_VERSION,
        "dataset_id": DATASET_ID,
        "synthetic": True,
        "intended_use": INTENDED_USE,
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "target": TARGET_COLUMN,
        "target_definition": (
            "1 if the member has >=1 ED encounter classified POTENTIALLY_AVOIDABLE "
            "in the 90-day outcome window after index_date, else 0."
        ),
        "prediction_horizon_days": 90,
        "observation_window_days": 270,
        "algorithm": winner["candidate"],
        "hyperparameters": best_params_by_family.get(winner["candidate"]),
        "calibration_method": winner["calibration_method"],
        "preprocessing": {
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot(handle_unknown=ignore)",
            "scaling": "StandardScaler (logistic_regression only)" if winner["candidate"] == "logistic_regression" else "none (tree-based, scale invariant)",
        },
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "risk_tier_semantics": {
            "LOW": "Lower predicted future potentially-avoidable-ED risk; no proactive escalation by risk alone. NOT a medical acuity or emergency-severity judgment.",
            "MODERATE": "Meaningful navigation opportunity; suitable for light-touch navigation or review. NOT a diagnosis.",
            "HIGH": "Strongest predicted navigation/prioritization opportunity. NOT permission to avoid or delay emergency care.",
        },
        "feature_list": feature_columns,
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "categorical_features": categorical_features,
        "train_index_date": "2025-10-05",
        "validation_index_date": "2026-01-03",
        "test_index_date": "2026-04-03",
        "train_prevalence": train_prevalence,
        "validation_prevalence": validation_prevalence,
        "test_prevalence": test_prevalence,
        "validation_metrics": validation_final_metrics,
        "validation_pr_enrichment": round(validation_final_metrics["pr_auc"] / validation_prevalence, 4) if validation_prevalence else None,
        "validation_risk_tiers": validation_tier_report,
        "final_test_metrics": test_eval["test_rank_metrics"],
        "final_test_pr_enrichment": test_pr_enrichment,
        "final_test_confusion_at_moderate_threshold": test_eval["test_moderate_confusion"],
        "final_test_confusion_at_high_threshold": test_eval["test_high_confusion"],
        "test_risk_tiers": test_tier_report,
        "validation_vs_test": test_eval["validation_vs_test"],
        "subgroup_sanity_findings": subgroup_rows,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_synthetic_dataset_sha256": raw_hashes,
        "synthetic_snapshot_sha256": snapshot_hashes,
        "feature_manifest_reference": str(SYNTHETIC_MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "hgb_class_weight_supported_by_installed_sklearn": hgb_supports_class_weight,
        "random_seed": v1_train.RANDOM_STATE,
        "comparison_to_original_v1": {
            "note": (
                "If this synthetic model performs substantially better than uc07-risk-v1, "
                "this demonstrates that the existing pipeline can learn stronger prospective "
                "relationships when those relationships exist in the input data. "
                "It does NOT demonstrate real-world clinical validity."
            ),
            "original_v1_test_metrics": ORIGINAL_V1_TEST_METRICS,
            "original_v1_algorithm": ORIGINAL_V1_ALGORITHM,
            "original_v1_high_tier_lift": ORIGINAL_V1_HIGH_TIER_LIFT,
        },
        "known_limitations": [
            "SYNTHETIC DATA MODEL -- DEMONSTRATION ONLY. Trained and evaluated entirely on the "
            "explicitly synthetic data/synthetic/ dataset trio; contains no real member, "
            "encounter, or care data.",
            "Same single ~18-month data era and 3-snapshot temporal design as the original "
            "experiment (Phase 2/3 design, unchanged).",
            "Subgroup sanity check is initial only, not the full Phase 6 audit.",
            "Risk tiers are risk-only; Care Management/navigation routing logic remains Phase 5 scope.",
            "Any relationships this model learns reflect the synthetic data generator's internal "
            "construction, not measured real-world clinical relationships.",
        ],
    }


def main() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_paths = {"train": TRAIN_CSV, "validation": VALIDATION_CSV, "test": TEST_CSV}
    snapshot_hashes_before = {name: sha256_file(p) for name, p in snapshot_paths.items()}
    raw_hashes = {name: sha256_file(p) for name, p in SYNTHETIC_RAW.items()}

    feature_columns = load_model_feature_columns(manifest_path=SYNTHETIC_MANIFEST_PATH)
    if len(feature_columns) != 59:
        raise SystemExit(f"CRITICAL: expected 59 synthetic model-candidate features, got {len(feature_columns)}")

    X_train, y_train, X_val, y_val, val_ids = load_synthetic_snapshots()
    numeric_features, categorical_features = split_numeric_categorical(X_train, feature_columns)

    print(f"Loaded SYNTHETIC TRAIN={len(X_train)} VALIDATION={len(X_val)} rows; "
          f"{len(feature_columns)} features ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    print(f"TRAIN prevalence: {y_train.mean():.4%}  VALIDATION prevalence: {y_val.mean():.4%}")

    # ---- Selection happens on TRAIN + VALIDATION only (reused unmodified from train.py) ----
    selection = v1_train.select_model_on_validation(X_train, y_train, X_val, y_val, numeric_features, categorical_features)

    final_estimator = selection["final_estimator"]
    moderate_threshold = selection["moderate_threshold"]
    high_threshold = selection["high_threshold"]

    final_val_prob = final_estimator.predict_proba(X_val)[:, 1]
    validation_scored = pd.DataFrame({
        "member_id": val_ids,
        "predicted_probability": final_val_prob,
        "risk_tier": risk_tiers_mod.assign_risk_tiers(final_val_prob, moderate_threshold, high_threshold),
    })

    frozen_spec = {
        "model_version_candidate": MODEL_VERSION,
        "dataset_id": DATASET_ID,
        "synthetic": True,
        "algorithm": selection["winner"]["candidate"],
        "calibration_method": selection["winner"]["calibration_method"],
        "hyperparameters": selection["best_params_by_family"].get(selection["winner"]["candidate"]),
        "preprocessing": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot(handle_unknown=ignore)",
        },
        "feature_list": feature_columns,
        "feature_count": len(feature_columns),
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "validation_metrics_at_freeze": selection["validation_final_metrics"],
        "tie_break_applied": selection["winner"]["tie_break_applied"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Frozen BEFORE any TEST data is loaded. Must not change based on TEST results. SYNTHETIC DATA -- DEMONSTRATION ONLY.",
    }
    (EVAL_DIR / "frozen_model_selection.json").write_text(json.dumps(frozen_spec, indent=2, default=str))
    print(f"FROZEN before TEST: algorithm={frozen_spec['algorithm']} calibration={frozen_spec['calibration_method']} "
          f"moderate_threshold={moderate_threshold:.4f} high_threshold={high_threshold:.4f}")

    # ---- Only now does TEST enter scope ----
    X_test, y_test, test_ids = load_snapshot_xy(TEST_CSV, manifest_path=SYNTHETIC_MANIFEST_PATH)

    test_eval = v1_train.evaluate_frozen_model_on_test(
        final_estimator, X_test, y_test, moderate_threshold, high_threshold, selection["validation_final_metrics"],
    )
    test_prob = test_eval["test_probabilities"]
    test_scored = pd.DataFrame({
        "member_id": test_ids,
        "predicted_probability": test_prob,
        "risk_tier": risk_tiers_mod.assign_risk_tiers(test_prob, moderate_threshold, high_threshold),
    })

    subgroup_rows = v1_train.run_subgroup_checks(X_test, y_test, test_prob, moderate_threshold)
    feature_importance_rows = v1_train.extract_global_feature_importance(final_estimator)

    v1_train.write_reports(
        EVAL_DIR,
        selection["candidate_rows"], selection["hyperparam_search_log"], selection["calibration_rows"],
        selection["threshold_rows"], selection["validation_tier_report"], test_eval["test_tier_report"],
        test_eval["test_calibration_bins"], subgroup_rows, feature_importance_rows,
        frozen_spec, validation_scored, test_scored,
    )
    # write_reports() names the frozen-spec file final_model_selection.json inside EVAL_DIR;
    # Step 24 asks for frozen_model_selection.json specifically -- write both names for compliance
    # without special-casing write_reports() (it is reused unmodified from train.py).
    (EVAL_DIR / "frozen_model_selection.json").write_text(json.dumps(frozen_spec, indent=2, default=str))

    artifact = v1_train.build_model_artifact(
        final_estimator, feature_columns, numeric_features, categorical_features,
        moderate_threshold, high_threshold, selection["winner"], selection["best_params_by_family"],
    )
    artifact["model_version"] = MODEL_VERSION
    artifact["dataset_id"] = DATASET_ID
    artifact["synthetic"] = True
    artifact["intended_use"] = INTENDED_USE
    artifact_path = MODELS_DIR / "uc07_risk_synthetic_v1_model.joblib"
    joblib.dump(artifact, artifact_path)

    snapshot_hashes_after = {name: sha256_file(p) for name, p in snapshot_paths.items()}
    if snapshot_hashes_after != snapshot_hashes_before:
        raise SystemExit(f"CRITICAL: synthetic snapshot hashes changed during training run. "
                          f"before={snapshot_hashes_before} after={snapshot_hashes_after}")
    raw_hashes_after = {name: sha256_file(p) for name, p in SYNTHETIC_RAW.items()}
    if raw_hashes_after != raw_hashes:
        raise SystemExit("CRITICAL: synthetic raw dataset hashes changed during training run.")

    metadata = build_synthetic_metadata(
        selection["winner"], selection["best_params_by_family"], moderate_threshold, high_threshold,
        feature_columns, numeric_features, categorical_features,
        round(float(y_train.mean()), 6), round(float(y_val.mean()), 6), round(float(y_test.mean()), 6),
        selection["validation_final_metrics"], test_eval,
        selection["validation_tier_report"], test_eval["test_tier_report"],
        subgroup_rows, raw_hashes, snapshot_hashes_after, selection["hgb_supports_class_weight"],
    )
    metadata_path = MODELS_DIR / "uc07_risk_synthetic_v1_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    # original-vs-synthetic comparison report (Step 24 / Step 27)
    comparison_rows = [
        {
            "model": "uc07-risk-v1", "dataset_type": "original", "algorithm": ORIGINAL_V1_ALGORITHM,
            "test_prevalence": ORIGINAL_V1_TEST_METRICS["prevalence"],
            "test_roc_auc": ORIGINAL_V1_TEST_METRICS["roc_auc"],
            "test_pr_auc": ORIGINAL_V1_TEST_METRICS["pr_auc"],
            "test_pr_enrichment": round(ORIGINAL_V1_TEST_METRICS["pr_auc"] / ORIGINAL_V1_TEST_METRICS["prevalence"], 4),
            "test_brier": ORIGINAL_V1_TEST_METRICS["brier_score"],
            "test_high_tier_lift": ORIGINAL_V1_HIGH_TIER_LIFT,
        },
        {
            "model": "uc07-risk-synthetic-v1", "dataset_type": "synthetic", "algorithm": selection["winner"]["candidate"],
            "test_prevalence": round(float(y_test.mean()), 6),
            "test_roc_auc": test_eval["test_rank_metrics"]["roc_auc"],
            "test_pr_auc": test_eval["test_rank_metrics"]["pr_auc"],
            "test_pr_enrichment": metadata["final_test_pr_enrichment"],
            "test_brier": test_eval["test_rank_metrics"]["brier_score"],
            "test_high_tier_lift": next(r["lift_vs_overall_prevalence"] for r in test_eval["test_tier_report"] if r["tier"] == "HIGH"),
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(EVAL_DIR / "original_vs_synthetic_comparison.csv", index=False)
    (EVAL_DIR / "final_test_results.json").write_text(json.dumps({
        "test_rank_metrics": test_eval["test_rank_metrics"],
        "test_pr_enrichment": metadata["final_test_pr_enrichment"],
        "test_moderate_confusion": test_eval["test_moderate_confusion"],
        "test_high_confusion": test_eval["test_high_confusion"],
        "test_tier_report": test_eval["test_tier_report"],
        "validation_vs_test": test_eval["validation_vs_test"],
    }, indent=2, default=str))

    return {
        "selection": selection, "frozen_spec": frozen_spec, "test_eval": test_eval,
        "subgroup_rows": subgroup_rows, "feature_importance_rows": feature_importance_rows,
        "artifact_path": artifact_path, "metadata_path": metadata_path, "metadata": metadata,
        "snapshot_hashes_before": snapshot_hashes_before, "snapshot_hashes_after": snapshot_hashes_after,
        "raw_hashes_before": raw_hashes, "raw_hashes_after": raw_hashes_after,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_test": X_test,
    }


if __name__ == "__main__":
    result = main()
    print("\n=== Phase 4D synthetic training run complete ===")
    print(f"Artifact: {result['artifact_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Final TEST PR-AUC: {result['test_eval']['test_rank_metrics']['pr_auc']:.4f} "
          f"ROC-AUC: {result['test_eval']['test_rank_metrics']['roc_auc']:.4f} "
          f"Brier: {result['test_eval']['test_rank_metrics']['brier_score']:.4f}")
