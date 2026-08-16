"""
train_phase4e_tree_comparison.py
----------------------------------
Phase 4E -- Controlled Model Optimization.

Question: can Random Forest or XGBoost meaningfully outperform the
current Logistic Regression synthetic model (uc07-risk-synthetic-v1)
on the SAME frozen, leakage-safe 59-feature synthetic snapshots, without
leakage, overfitting, or poor calibration?

This script does NOT duplicate Phase 4/4D modeling logic. It reuses,
unmodified: feature_spec.load_snapshot_xy / split_numeric_categorical,
preprocessing.build_preprocessor / build_scaled_preprocessor,
metrics.rank_metrics / threshold_confusion_counts / calibration_bins,
risk_tiers.tier_report / assign_risk_tiers, and train.py's sha256_file,
compare_calibration_methods, and extract_global_feature_importance.

The only things new here are: Accuracy/Balanced-Accuracy reporting
(deliberately NOT part of metrics.py -- see that module's docstring --
because Phase 4/4D's rule is "never use Accuracy as a selection metric";
Phase 4E is explicitly told to report it as a descriptive metric
alongside everything else, so it lives only in this Phase 4E script),
Random Forest / XGBoost search grids, and the Phase 4E artifact set.

TRAIN = fitting only. VALIDATION = search, comparison, calibration,
threshold selection, promotion decision. TEST = sealed until
frozen_model_selection.json exists, then read exactly once.

Run: python backend/modeling/train_phase4e_tree_comparison.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as metrics_mod
import risk_tiers as risk_tiers_mod
import train as v1_train
from feature_spec import TARGET_COLUMN, load_model_feature_columns, load_snapshot_xy, split_numeric_categorical
from preprocessing import build_preprocessor, build_scaled_preprocessor

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
EVAL_DIR = REPO_ROOT / "artifacts" / "phase4e_tree_model_comparison"

EXISTING_MODEL_ARTIFACT = MODELS_DIR / "uc07_risk_synthetic_v1_model.joblib"
EXISTING_MODEL_METADATA = MODELS_DIR / "uc07_risk_synthetic_v1_model_metadata.json"

RANDOM_STATE = 42
CURRENT_MODERATE_THRESHOLD = 0.105986
CURRENT_HIGH_THRESHOLD = 0.213252
STANDARD_THRESHOLD = 0.50
ROC_AUC_SUSPICIOUS_CEILING = 0.85
PROMOTION_ROC_AUC_DELTA = 0.02

warnings.filterwarnings("ignore", category=UserWarning)


def sha256_file(path: Path) -> str:
    return v1_train.sha256_file(path)


def hash_immutable_set() -> dict:
    files = {
        "raw_members.csv": REPO_ROOT / "raw_members.csv",
        "raw_ed_visits.csv": REPO_ROOT / "raw_ed_visits.csv",
        "raw_care_history.csv": REPO_ROOT / "raw_care_history.csv",
        "synthetic_raw_members.csv": SYNTHETIC_RAW["raw_members.csv"],
        "synthetic_raw_ed_visits.csv": SYNTHETIC_RAW["raw_ed_visits.csv"],
        "synthetic_raw_care_history.csv": SYNTHETIC_RAW["raw_care_history.csv"],
        "train_snapshot.csv": TRAIN_CSV,
        "validation_snapshot.csv": VALIDATION_CSV,
        "test_snapshot.csv": TEST_CSV,
        "feature_manifest.json": SYNTHETIC_MANIFEST_PATH,
        "uc07_risk_v1_model.joblib": MODELS_DIR / "uc07_risk_v1_model.joblib",
        "uc07_risk_synthetic_v1_model.joblib": EXISTING_MODEL_ARTIFACT,
        "uc07_risk_v1_model_metadata.json": MODELS_DIR / "uc07_risk_v1_model_metadata.json",
        "uc07_risk_synthetic_v1_model_metadata.json": EXISTING_MODEL_METADATA,
    }
    return {name: sha256_file(p) for name, p in files.items()}


# ---------------------------------------------------------------------------
# Accuracy / Balanced Accuracy (Phase 4E only -- see module docstring)
# ---------------------------------------------------------------------------

def full_metrics_with_accuracy(y_true, y_prob, threshold: float) -> dict:
    """metrics_mod.full_metrics_at_threshold() plus Accuracy and Balanced
    Accuracy, computed here (not in metrics.py) because Phase 4/4D
    deliberately keep Accuracy out of the shared selection-metric module.
    Phase 4E must report it descriptively at every threshold anyway."""
    base = metrics_mod.full_metrics_at_threshold(y_true, y_prob, threshold)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    base["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)
    base["balanced_accuracy"] = round(float(balanced_accuracy_score(y_true, y_pred)), 6)
    return base


def multi_threshold_profile(y_true, y_prob, label: str) -> list[dict]:
    rows = []
    for thr_name, thr in (
        ("0.50", STANDARD_THRESHOLD),
        ("MODERATE", CURRENT_MODERATE_THRESHOLD),
        ("HIGH", CURRENT_HIGH_THRESHOLD),
    ):
        row = {"model": label, "threshold_name": thr_name, **full_metrics_with_accuracy(y_true, y_prob, thr)}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Candidate builders
# ---------------------------------------------------------------------------

def _pipeline(preprocessor, model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_logistic_reference(numeric_features, categorical_features) -> Pipeline:
    """Reproduces the frozen uc07-risk-synthetic-v1 hyperparameters exactly
    (C=0.01, class_weight=None) -- Step 1/3, not a new search."""
    pre = build_scaled_preprocessor(numeric_features, categorical_features)
    model = LogisticRegression(C=0.01, class_weight=None, penalty="l2", solver="lbfgs",
                                max_iter=2000, random_state=RANDOM_STATE)
    return _pipeline(pre, model)


def random_forest_grid() -> list[dict]:
    """Curated, computationally sensible combinations -- not a full
    cartesian product (would be hundreds of fits)."""
    return [
        {"n_estimators": 300, "max_depth": 8, "min_samples_split": 2, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": None},
        {"n_estimators": 300, "max_depth": 8, "min_samples_split": 2, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": 8, "min_samples_split": 10, "min_samples_leaf": 20, "max_features": 0.5, "class_weight": None},
        {"n_estimators": 300, "max_depth": 16, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
        {"n_estimators": 300, "max_depth": 16, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": 16, "min_samples_split": 20, "min_samples_leaf": 20, "max_features": 0.5, "class_weight": None},
        {"n_estimators": 600, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
        {"n_estimators": 600, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 600, "max_depth": 8, "min_samples_split": 2, "min_samples_leaf": 5, "max_features": 0.5, "class_weight": None},
        {"n_estimators": 600, "max_depth": 12, "min_samples_split": 20, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": None},
        {"n_estimators": 600, "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 10, "max_features": 0.5, "class_weight": "balanced"},
        {"n_estimators": 400, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": None},
        {"n_estimators": 400, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 400, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 0.5, "class_weight": None},
        {"n_estimators": 400, "max_depth": 20, "min_samples_split": 20, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 20, "min_samples_leaf": 20, "max_features": 0.5, "class_weight": "balanced"},
    ]


def build_random_forest_candidate(numeric_features, categorical_features, params: dict) -> Pipeline:
    pre = build_preprocessor(numeric_features, categorical_features)
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return _pipeline(pre, model)


TRAIN_PREVALENCE_FOR_SCALE_POS_WEIGHT = None  # set in main() from TRAIN y only


def xgboost_grid(scale_pos_weight_value: float) -> list[dict]:
    """Curated combinations. scale_pos_weight is evaluated as an
    optional axis, computed only from TRAIN prevalence (never
    VALIDATION/TEST) -- Step 5."""
    spw_options = (None, scale_pos_weight_value)
    combos = [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": None},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": None},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": scale_pos_weight_value},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": scale_pos_weight_value},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": None},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 1.0, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": scale_pos_weight_value},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": scale_pos_weight_value},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.02, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": None},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.02, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.02, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 10, "scale_pos_weight": None},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": scale_pos_weight_value},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": None},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 500, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5, "scale_pos_weight": None},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 10, "scale_pos_weight": scale_pos_weight_value},
    ]
    return combos


def build_xgboost_candidate(numeric_features, categorical_features, params: dict) -> Pipeline:
    pre = build_preprocessor(numeric_features, categorical_features)
    kwargs = dict(
        objective="binary:logistic",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        tree_method="hist",
        eval_metric="logloss",
        importance_type="gain",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if params.get("scale_pos_weight") is not None:
        kwargs["scale_pos_weight"] = params["scale_pos_weight"]
    model = XGBClassifier(**kwargs)
    return _pipeline(pre, model)


def build_hgb_reference(numeric_features, categorical_features) -> Pipeline:
    """Reuses the Phase 4D HistGradientBoosting winner
    (artifacts/synthetic_model_evaluation/candidate_metrics.csv) as a
    reference, refit here (single fit, cheap) so its metrics are computed
    through the exact same Phase 4E helper functions as every other
    candidate -- Step 6."""
    pre = build_preprocessor(numeric_features, categorical_features)
    model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_STATE,
    )
    return _pipeline(pre, model)


# ---------------------------------------------------------------------------
# Search (VALIDATION only, PR-AUC primary / Brier secondary -- same rule as train.py)
# ---------------------------------------------------------------------------

def run_search(family: str, grid: list[dict], build_fn, numeric_features, categorical_features,
                X_train, y_train, X_val, y_val) -> tuple[list[dict], dict, Pipeline]:
    search_log = []
    best = None
    for params in grid:
        pipeline = build_fn(numeric_features, categorical_features, params)
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        rm = metrics_mod.rank_metrics(y_val, val_prob)
        row = {"family": family, "params": json.dumps(params, default=str), **rm}
        search_log.append(row)
        key = (rm["pr_auc"], -rm["brier_score"])
        if best is None or key > (best[0], -best[1]["brier_score"]):
            best = (rm["pr_auc"], rm, params, pipeline)
    _, _, best_params, best_pipeline = best
    return search_log, best_params, best_pipeline


# ---------------------------------------------------------------------------
# XGBoost gain-based feature importance
# ---------------------------------------------------------------------------

def extract_xgb_feature_importance(pipeline: Pipeline) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    importances = model.feature_importances_  # importance_type="gain" set at construction
    rows = [{"feature_name": name, "importance": float(value), "importance_type": "xgboost_gain"}
            for name, value in zip(feature_names, importances)]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Overfitting check
# ---------------------------------------------------------------------------

def train_val_gap_row(label: str, pipeline: Pipeline, X_train, y_train, X_val, y_val) -> dict:
    train_prob = pipeline.predict_proba(X_train)[:, 1]
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    train_pred = (train_prob >= STANDARD_THRESHOLD).astype(int)
    val_pred = (val_prob >= STANDARD_THRESHOLD).astype(int)
    train_rm = metrics_mod.rank_metrics(y_train, train_prob)
    val_rm = metrics_mod.rank_metrics(y_val, val_prob)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    roc_gap = train_rm["roc_auc"] - val_rm["roc_auc"]
    pr_gap = train_rm["pr_auc"] - val_rm["pr_auc"]
    acc_gap = train_acc - val_acc
    flagged = (train_rm["roc_auc"] > 0.97 and roc_gap > 0.05) or roc_gap > 0.08
    return {
        "model": label,
        "train_accuracy_0.50": round(float(train_acc), 6), "validation_accuracy_0.50": round(float(val_acc), 6), "accuracy_gap": round(float(acc_gap), 6),
        "train_roc_auc": train_rm["roc_auc"], "validation_roc_auc": val_rm["roc_auc"], "roc_auc_gap": round(float(roc_gap), 6),
        "train_pr_auc": train_rm["pr_auc"], "validation_pr_auc": val_rm["pr_auc"], "pr_auc_gap": round(float(pr_gap), 6),
        "overfitting_flag": bool(flagged),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4E: Controlled Model Optimization ===")
    hashes_before = hash_immutable_set()

    feature_columns = load_model_feature_columns(manifest_path=SYNTHETIC_MANIFEST_PATH)
    if len(feature_columns) != 59:
        raise SystemExit(f"CRITICAL: expected 59 approved features, got {len(feature_columns)}")

    X_train, y_train, _ = load_snapshot_xy(TRAIN_CSV, manifest_path=SYNTHETIC_MANIFEST_PATH)
    X_val, y_val, val_ids = load_snapshot_xy(VALIDATION_CSV, manifest_path=SYNTHETIC_MANIFEST_PATH)
    numeric_features, categorical_features = split_numeric_categorical(X_train, feature_columns)

    print(f"TRAIN n={len(X_train)} prevalence={y_train.mean():.4%}  VALIDATION n={len(X_val)} prevalence={y_val.mean():.4%}")
    print(f"Features: {len(feature_columns)} ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")

    # ---- Leakage audit (Step 12, part 1): approved feature list only ----
    forbidden_tokens = ["member_id", TARGET_COLUMN, "red_flag", "icu", "admitted", "major_procedure", "triage_level", "index_date"]
    leaked = [c for c in feature_columns if c in forbidden_tokens]
    leakage_audit_pre = {"forbidden_columns_in_feature_list": leaked, "passed": len(leaked) == 0}
    if leaked:
        raise SystemExit(f"CRITICAL: leakage audit failed -- forbidden columns present: {leaked}")

    # =========================================================================
    # STEP 1/3 -- Logistic reference (reproduce frozen uc07-risk-synthetic-v1)
    # =========================================================================
    print("\n--- Reproducing Logistic Regression reference ---")
    lr_pipeline = build_logistic_reference(numeric_features, categorical_features)
    lr_pipeline.fit(X_train, y_train)
    lr_val_prob = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_rank = metrics_mod.rank_metrics(y_val, lr_val_prob)

    existing_metadata = json.loads(EXISTING_MODEL_METADATA.read_text())
    expected = existing_metadata["validation_metrics"]
    reproduction_deltas = {
        "roc_auc_delta": round(lr_rank["roc_auc"] - expected["roc_auc"], 6),
        "pr_auc_delta": round(lr_rank["pr_auc"] - expected["pr_auc"], 6),
        "brier_delta": round(lr_rank["brier_score"] - expected["brier_score"], 6),
    }
    TOLERANCE = 0.003
    reproduction_ok = all(abs(v) < TOLERANCE for v in reproduction_deltas.values())
    print(f"LR VALIDATION reproduction: roc_auc={lr_rank['roc_auc']} (expected {expected['roc_auc']}), "
          f"pr_auc={lr_rank['pr_auc']} (expected {expected['pr_auc']}), brier={lr_rank['brier_score']} (expected {expected['brier_score']})")
    if not reproduction_ok:
        raise SystemExit(f"CRITICAL: Logistic baseline reproduction disagrees with frozen metadata beyond tolerance: {reproduction_deltas}")
    print(f"Reproduction within tolerance ({TOLERANCE}): PASS")

    lr_threshold_profile = multi_threshold_profile(y_val, lr_val_prob, "logistic_regression")

    # =========================================================================
    # STEP 4 -- Random Forest search
    # =========================================================================
    print("\n--- Random Forest search ---")
    rf_log, rf_best_params, rf_pipeline = run_search(
        "RandomForest", random_forest_grid(), build_random_forest_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    rf_val_prob = rf_pipeline.predict_proba(X_val)[:, 1]
    rf_rank = metrics_mod.rank_metrics(y_val, rf_val_prob)
    print(f"RF best: {rf_best_params} -> ROC-AUC={rf_rank['roc_auc']} PR-AUC={rf_rank['pr_auc']} Brier={rf_rank['brier_score']}")
    rf_threshold_profile = multi_threshold_profile(y_val, rf_val_prob, "random_forest")

    # =========================================================================
    # STEP 5 -- XGBoost search
    # =========================================================================
    print("\n--- XGBoost search ---")
    scale_pos_weight_value = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    xgb_log, xgb_best_params, xgb_pipeline = run_search(
        "XGBoost", xgboost_grid(scale_pos_weight_value), build_xgboost_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    xgb_val_prob = xgb_pipeline.predict_proba(X_val)[:, 1]
    xgb_rank = metrics_mod.rank_metrics(y_val, xgb_val_prob)
    print(f"XGB best: {xgb_best_params} -> ROC-AUC={xgb_rank['roc_auc']} PR-AUC={xgb_rank['pr_auc']} Brier={xgb_rank['brier_score']}")
    xgb_threshold_profile = multi_threshold_profile(y_val, xgb_val_prob, "xgboost")

    # =========================================================================
    # STEP 6 -- HistGradientBoosting reference
    # =========================================================================
    print("\n--- HistGradientBoosting reference (Phase 4D winner combo, refit) ---")
    hgb_pipeline = build_hgb_reference(numeric_features, categorical_features)
    hgb_pipeline.fit(X_train, y_train)
    hgb_val_prob = hgb_pipeline.predict_proba(X_val)[:, 1]
    hgb_rank = metrics_mod.rank_metrics(y_val, hgb_val_prob)
    print(f"HGB reference -> ROC-AUC={hgb_rank['roc_auc']} PR-AUC={hgb_rank['pr_auc']} Brier={hgb_rank['brier_score']}")
    hgb_threshold_profile = multi_threshold_profile(y_val, hgb_val_prob, "hist_gradient_boosting")

    # =========================================================================
    # STEP 12 -- Leakage audit, part 2: suspicious-performance ceiling
    # =========================================================================
    all_roc = {"logistic_regression": lr_rank["roc_auc"], "random_forest": rf_rank["roc_auc"],
               "xgboost": xgb_rank["roc_auc"], "hist_gradient_boosting": hgb_rank["roc_auc"]}
    suspicious = {k: v for k, v in all_roc.items() if v > ROC_AUC_SUSPICIOUS_CEILING}
    leakage_audit = {**leakage_audit_pre, "roc_auc_by_candidate": all_roc,
                      "suspicious_performance_ceiling": ROC_AUC_SUSPICIOUS_CEILING,
                      "candidates_above_ceiling": suspicious, "passed": len(suspicious) == 0}
    print(f"\nLeakage audit: {'PASS' if leakage_audit['passed'] else 'FAIL -- investigate ' + str(suspicious)}")
    if suspicious:
        raise SystemExit(f"CRITICAL: leakage audit tripped the 0.85 ROC-AUC ceiling: {suspicious}. Investigate before proceeding.")

    # =========================================================================
    # STEP 8 -- Confusion matrices at 0.50 / MODERATE / HIGH for every candidate
    # =========================================================================
    confusion_matrices = {}
    for label, prob in (("logistic_regression", lr_val_prob), ("random_forest", rf_val_prob),
                         ("xgboost", xgb_val_prob), ("hist_gradient_boosting", hgb_val_prob)):
        confusion_matrices[label] = {
            "0.50": metrics_mod.threshold_confusion_counts(y_val, prob, STANDARD_THRESHOLD),
            "MODERATE": metrics_mod.threshold_confusion_counts(y_val, prob, CURRENT_MODERATE_THRESHOLD),
            "HIGH": metrics_mod.threshold_confusion_counts(y_val, prob, CURRENT_HIGH_THRESHOLD),
        }

    # =========================================================================
    # STEP 9 -- Calibration comparison for leading RF/XGBoost candidates
    # =========================================================================
    print("\n--- Calibration comparison (RF, XGBoost) ---")
    calibration_rows = []
    calibration_estimators = {}
    for cand_name, build_fn, best_params in (
        ("random_forest", build_random_forest_candidate, rf_best_params),
        ("xgboost", build_xgboost_candidate, xgb_best_params),
    ):
        cal_results = v1_train.compare_calibration_methods(
            build_fn, best_params, numeric_features, categorical_features, X_train, y_train, X_val, y_val,
        )
        calibration_estimators[cand_name] = cal_results
        for method, payload in cal_results.items():
            calibration_rows.append({"candidate": cand_name, "calibration_method": method, **payload["metrics"]})
            print(f"  {cand_name} / {method}: PR-AUC={payload['metrics']['pr_auc']} Brier={payload['metrics']['brier_score']}")

    # =========================================================================
    # STEP 10 -- Overfitting: TRAIN vs VALIDATION
    # =========================================================================
    train_val_gaps = [
        train_val_gap_row("logistic_regression", lr_pipeline, X_train, y_train, X_val, y_val),
        train_val_gap_row("random_forest", rf_pipeline, X_train, y_train, X_val, y_val),
        train_val_gap_row("xgboost", xgb_pipeline, X_train, y_train, X_val, y_val),
        train_val_gap_row("hist_gradient_boosting", hgb_pipeline, X_train, y_train, X_val, y_val),
    ]
    print("\n--- TRAIN vs VALIDATION gaps ---")
    for row in train_val_gaps:
        flag = " FLAGGED" if row["overfitting_flag"] else ""
        print(f"  {row['model']}: TRAIN ROC-AUC={row['train_roc_auc']} VAL ROC-AUC={row['validation_roc_auc']} gap={row['roc_auc_gap']}{flag}")

    # =========================================================================
    # STEP 11 -- Feature importance
    # =========================================================================
    rf_importance = v1_train.extract_global_feature_importance(rf_pipeline)
    xgb_importance = extract_xgb_feature_importance(xgb_pipeline)

    # =========================================================================
    # STEP 13/14 -- Comparison table + selection
    # =========================================================================
    candidate_summary = [
        {"model": "logistic_regression", **lr_rank, "hyperparameters": json.dumps({"C": 0.01, "class_weight": None})},
        {"model": "random_forest", **rf_rank, "hyperparameters": json.dumps(rf_best_params, default=str)},
        {"model": "xgboost", **xgb_rank, "hyperparameters": json.dumps(xgb_best_params, default=str)},
        {"model": "hist_gradient_boosting", **hgb_rank,
         "hyperparameters": json.dumps({"max_iter": 100, "max_depth": 3, "learning_rate": 0.05, "class_weight": "balanced"})},
    ]
    for row in candidate_summary:
        gap = next(g for g in train_val_gaps if g["model"] == row["model"])
        row["train_roc_auc"] = gap["train_roc_auc"]
        row["validation_roc_auc"] = gap["validation_roc_auc"]
        row["train_pr_auc"] = gap["train_pr_auc"]
        row["validation_pr_auc"] = gap["validation_pr_auc"]
        row["overfitting_flag"] = gap["overfitting_flag"]
        row["pr_enrichment"] = round(row["pr_auc"] / row["prevalence"], 4) if row["prevalence"] else None
        conf_50 = confusion_matrices[row["model"]]["0.50"]
        row.update({f"{k}_@0.50": v for k, v in conf_50.items() if k not in ("threshold", "n_selected", "pct_selected")})
        row["accuracy_@0.50"] = round(float(accuracy_score(y_val, (
            (lr_val_prob if row["model"] == "logistic_regression" else
             rf_val_prob if row["model"] == "random_forest" else
             xgb_val_prob if row["model"] == "xgboost" else hgb_val_prob) >= STANDARD_THRESHOLD).astype(int))), 6)
        row["balanced_accuracy_@0.50"] = round(float(balanced_accuracy_score(y_val, (
            (lr_val_prob if row["model"] == "logistic_regression" else
             rf_val_prob if row["model"] == "random_forest" else
             xgb_val_prob if row["model"] == "xgboost" else hgb_val_prob) >= STANDARD_THRESHOLD).astype(int))), 6)

    lr_row = next(r for r in candidate_summary if r["model"] == "logistic_regression")
    rf_row = next(r for r in candidate_summary if r["model"] == "random_forest")
    xgb_row = next(r for r in candidate_summary if r["model"] == "xgboost")

    # Best non-LR candidate by PR-AUC (primary), Brier (secondary tie-break)
    tree_candidates = sorted([rf_row, xgb_row], key=lambda r: (-r["pr_auc"], r["brier_score"]))
    best_tree = tree_candidates[0]

    roc_auc_delta = best_tree["roc_auc"] - lr_row["roc_auc"]
    pr_auc_delta = best_tree["pr_auc"] - lr_row["pr_auc"]
    brier_delta = best_tree["brier_score"] - lr_row["brier_score"]
    calibration_ok = not best_tree["overfitting_flag"]

    promotion_signal = (roc_auc_delta >= PROMOTION_ROC_AUC_DELTA) or (pr_auc_delta > 0.02)
    promotion_met = promotion_signal and calibration_ok and (brier_delta < 0.02)

    if promotion_met and best_tree is rf_row:
        winner_name = "random_forest"
    elif promotion_met and best_tree is xgb_row:
        winner_name = "xgboost"
    else:
        winner_name = "logistic_regression"

    print(f"\n--- SELECTION ---\nBest tree candidate: {best_tree['model']} "
          f"(ROC-AUC delta={roc_auc_delta:+.4f}, PR-AUC delta={pr_auc_delta:+.4f}, Brier delta={brier_delta:+.4f}, "
          f"overfitting_flag={best_tree['overfitting_flag']})")
    print(f"Promotion criteria met: {promotion_met}")
    print(f"WINNER: {winner_name}")

    winner_pipeline = {"logistic_regression": lr_pipeline, "random_forest": rf_pipeline, "xgboost": xgb_pipeline}[winner_name]
    winner_val_prob = {"logistic_regression": lr_val_prob, "random_forest": rf_val_prob, "xgboost": xgb_val_prob}[winner_name]
    winner_hyperparams = {"logistic_regression": {"C": 0.01, "class_weight": None},
                           "random_forest": rf_best_params, "xgboost": xgb_best_params}[winner_name]

    # =========================================================================
    # STEP 15/16 -- Threshold + risk-tier selection for the winner (VALIDATION only)
    # =========================================================================
    if winner_name == "logistic_regression":
        moderate_threshold, high_threshold = CURRENT_MODERATE_THRESHOLD, CURRENT_HIGH_THRESHOLD
        threshold_note = "KEEP LOGISTIC: reuses the existing frozen uc07-risk-synthetic-v1 thresholds unchanged."
    else:
        high_threshold = float(np.percentile(winner_val_prob, 90))
        moderate_threshold = float(np.percentile(winner_val_prob, 65))
        if moderate_threshold >= high_threshold:
            moderate_threshold = max(0.0, high_threshold - 0.01)
        risk_tiers_mod.validate_thresholds(moderate_threshold, high_threshold)
        threshold_note = f"NEW thresholds derived from {winner_name}'s own VALIDATION score distribution (65th/90th percentile), not reused from Logistic."

    winner_threshold_metrics = {
        "0.50": full_metrics_with_accuracy(y_val, winner_val_prob, STANDARD_THRESHOLD),
        "MODERATE": full_metrics_with_accuracy(y_val, winner_val_prob, moderate_threshold),
        "HIGH": full_metrics_with_accuracy(y_val, winner_val_prob, high_threshold),
    }
    validation_risk_tiers = risk_tiers_mod.tier_report(y_val, winner_val_prob, moderate_threshold, high_threshold)
    print(f"\nWinner thresholds: MODERATE={moderate_threshold:.6f} HIGH={high_threshold:.6f} ({threshold_note})")

    # =========================================================================
    # STEP 17 -- FREEZE before TEST
    # =========================================================================
    frozen_spec = {
        "phase": "4E",
        "model_version_candidate": "uc07-risk-synthetic-v1" if winner_name == "logistic_regression" else f"uc07-risk-synthetic-{'rf' if winner_name == 'random_forest' else 'xgb'}-v1",
        "dataset_id": "synthetic_uc07_v1",
        "synthetic": True,
        "decision": "KEEP_LOGISTIC" if winner_name == "logistic_regression" else f"PROMOTE_{winner_name.upper()}",
        "algorithm": winner_name,
        "hyperparameters": winner_hyperparams,
        "calibration_method": "uncalibrated",
        "preprocessing": {
            "numeric_features": numeric_features, "categorical_features": categorical_features,
            "numeric_imputation": "median", "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot(handle_unknown=ignore)",
            "scaling": "StandardScaler (logistic_regression only)" if winner_name == "logistic_regression" else "none (tree-based, scale invariant)",
        },
        "feature_list": feature_columns,
        "feature_count": len(feature_columns),
        "moderate_threshold": round(moderate_threshold, 6),
        "high_threshold": round(high_threshold, 6),
        "threshold_note": threshold_note,
        "validation_metrics_at_freeze": metrics_mod.rank_metrics(y_val, winner_val_prob),
        "promotion_criteria": {
            "roc_auc_delta_vs_logistic": round(roc_auc_delta, 6),
            "pr_auc_delta_vs_logistic": round(pr_auc_delta, 6),
            "brier_delta_vs_logistic": round(brier_delta, 6),
            "overfitting_flag": best_tree["overfitting_flag"],
            "promotion_signal": promotion_signal,
            "promotion_met": promotion_met,
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Frozen BEFORE any TEST data is loaded in this Phase 4E run. Must not change based on TEST results. SYNTHETIC DATA -- DEMONSTRATION ONLY.",
    }
    (EVAL_DIR / "frozen_model_selection.json").write_text(json.dumps(frozen_spec, indent=2, default=str))
    print(f"\nFROZEN before TEST -> {EVAL_DIR / 'frozen_model_selection.json'}")

    # =========================================================================
    # STEP 18 -- FINAL TEST (exactly once)
    # =========================================================================
    print("\n--- Loading TEST (sealed until now) ---")
    X_test, y_test, test_ids = load_snapshot_xy(TEST_CSV, manifest_path=SYNTHETIC_MANIFEST_PATH)
    test_prob = winner_pipeline.predict_proba(X_test)[:, 1]

    test_rank = metrics_mod.rank_metrics(y_test, test_prob)
    test_metrics_by_threshold = {
        "0.50": full_metrics_with_accuracy(y_test, test_prob, STANDARD_THRESHOLD),
        "MODERATE": full_metrics_with_accuracy(y_test, test_prob, moderate_threshold),
        "HIGH": full_metrics_with_accuracy(y_test, test_prob, high_threshold),
    }
    test_risk_tiers = risk_tiers_mod.tier_report(y_test, test_prob, moderate_threshold, high_threshold)
    test_pr_enrichment = round(test_rank["pr_auc"] / test_rank["prevalence"], 4) if test_rank["prevalence"] else None

    final_test_results = {
        "winner": winner_name,
        "test_rank_metrics": test_rank,
        "test_pr_enrichment": test_pr_enrichment,
        "test_metrics_by_threshold": test_metrics_by_threshold,
        "test_risk_tiers": test_risk_tiers,
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
    }
    (EVAL_DIR / "final_test_results.json").write_text(json.dumps(final_test_results, indent=2, default=str))
    print(f"TEST: ROC-AUC={test_rank['roc_auc']} PR-AUC={test_rank['pr_auc']} Brier={test_rank['brier_score']} "
          f"PR-enrichment={test_pr_enrichment}")

    # =========================================================================
    # STEP 24 -- Subgroup sanity check (transportation_barrier especially)
    # =========================================================================
    subgroup_rows = v1_train.run_subgroup_checks(X_test, y_test, test_prob, moderate_threshold)
    transport_1 = next((r for r in subgroup_rows if r["subgroup"] == "transportation_barrier_1"), None)
    print(f"\ntransportation_barrier=1 subgroup: {transport_1}")

    # =========================================================================
    # Write all Phase 4E artifacts (Step 25)
    # =========================================================================
    pd.DataFrame(candidate_summary).to_csv(EVAL_DIR / "candidate_metrics.csv", index=False)
    pd.DataFrame(rf_log).to_csv(EVAL_DIR / "rf_search_results.csv", index=False)
    pd.DataFrame(xgb_log).to_csv(EVAL_DIR / "xgb_search_results.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(EVAL_DIR / "calibration_comparison.csv", index=False)
    pd.DataFrame(train_val_gaps).to_csv(EVAL_DIR / "train_validation_gap.csv", index=False)
    (EVAL_DIR / "confusion_matrices.json").write_text(json.dumps(confusion_matrices, indent=2, default=str))
    threshold_analysis_rows = lr_threshold_profile + rf_threshold_profile + xgb_threshold_profile + hgb_threshold_profile
    pd.DataFrame(threshold_analysis_rows).to_csv(EVAL_DIR / "threshold_analysis.csv", index=False)
    pd.DataFrame(validation_risk_tiers).to_csv(EVAL_DIR / "validation_risk_tiers.csv", index=False)
    pd.DataFrame(rf_importance[:25]).to_csv(EVAL_DIR / "global_feature_importance_rf.csv", index=False)
    pd.DataFrame(xgb_importance[:25]).to_csv(EVAL_DIR / "global_feature_importance_xgb.csv", index=False)
    pd.concat([pd.DataFrame(rf_importance[:10]).assign(model="random_forest"),
               pd.DataFrame(xgb_importance[:10]).assign(model="xgboost")]).to_csv(EVAL_DIR / "global_feature_importance.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(EVAL_DIR / "subgroup_metrics.csv", index=False)
    (EVAL_DIR / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2, default=str))

    final_model_comparison = {
        "candidate_summary": candidate_summary,
        "winner": winner_name,
        "promotion_criteria": frozen_spec["promotion_criteria"],
        "threshold_note": threshold_note,
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "validation_risk_tiers": validation_risk_tiers,
        "test_summary": final_test_results,
        "subgroup_transportation_barrier_1": transport_1,
    }
    (EVAL_DIR / "final_model_comparison.json").write_text(json.dumps(final_model_comparison, indent=2, default=str))

    # =========================================================================
    # STEP 20/21 -- Model artifact + metadata (only if promoted)
    # =========================================================================
    artifact_path = None
    metadata_path = None
    if winner_name != "logistic_regression":
        suffix = "rf" if winner_name == "random_forest" else "xgb"
        model_version = f"uc07-risk-synthetic-{suffix}-v1"
        artifact = {
            "pipeline": winner_pipeline,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target": TARGET_COLUMN,
            "model_version": model_version,
            "dataset_id": "synthetic_uc07_v1",
            "synthetic": True,
            "intended_use": "demonstration / UC07 navigation prototype",
            "algorithm": winner_name,
            "calibration_method": "uncalibrated",
            "hyperparameters": winner_hyperparams,
            "moderate_threshold": round(moderate_threshold, 6),
            "high_threshold": round(high_threshold, 6),
        }
        artifact_path = MODELS_DIR / f"uc07_risk_synthetic_{suffix}_v1_model.joblib"
        joblib.dump(artifact, artifact_path)

        metadata = {
            "model_name": f"UC07 Avoidable ED Risk Model (Synthetic Demonstration, {winner_name})",
            "model_version": model_version,
            "dataset_id": "synthetic_uc07_v1",
            "synthetic": True,
            "intended_use": "demonstration / UC07 navigation prototype",
            "disclaimer": "Synthetic-data demonstration model -- not clinically validated.",
            "target": TARGET_COLUMN,
            "target_definition": existing_metadata["target_definition"],
            "prediction_horizon_days": 90,
            "observation_window_days": 270,
            "algorithm": winner_name,
            "hyperparameters": winner_hyperparams,
            "calibration_method": "uncalibrated",
            "feature_list": feature_columns,
            "feature_count": len(feature_columns),
            "moderate_threshold": round(moderate_threshold, 6),
            "high_threshold": round(high_threshold, 6),
            "train_index_date": "2025-10-05", "validation_index_date": "2026-01-03", "test_index_date": "2026-04-03",
            "train_prevalence": round(float(y_train.mean()), 6),
            "validation_prevalence": round(float(y_val.mean()), 6),
            "test_prevalence": round(float(y_test.mean()), 6),
            "validation_metrics": metrics_mod.rank_metrics(y_val, winner_val_prob),
            "validation_metrics_by_threshold": winner_threshold_metrics,
            "validation_risk_tiers": validation_risk_tiers,
            "final_test_metrics": test_rank,
            "final_test_pr_enrichment": test_pr_enrichment,
            "final_test_metrics_by_threshold": test_metrics_by_threshold,
            "test_risk_tiers": test_risk_tiers,
            "subgroup_sanity_findings": subgroup_rows,
            "python_version": sys.version, "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__ if winner_name == "xgboost" else None,
            "pandas_version": pd.__version__, "numpy_version": np.__version__,
            "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_STATE,
            "comparison_to_logistic_synthetic_v1": {
                "logistic_validation_roc_auc": lr_rank["roc_auc"], "logistic_validation_pr_auc": lr_rank["pr_auc"],
                "roc_auc_delta": round(roc_auc_delta, 6), "pr_auc_delta": round(pr_auc_delta, 6),
            },
            "known_limitations": [
                "SYNTHETIC DATA MODEL -- DEMONSTRATION ONLY. Not clinically validated.",
                "Phase 4E controlled comparison against the existing uc07-risk-synthetic-v1 (logistic regression).",
                "Subgroup sanity check is initial only, not the full Phase 6 audit.",
                "Risk tiers are risk-only; Care Management/navigation routing logic remains Phase 5 scope.",
            ],
        }
        metadata_path = MODELS_DIR / f"uc07_risk_synthetic_{suffix}_v1_model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        print(f"\nPromoted model artifact: {artifact_path}")
        print(f"Promoted model metadata: {metadata_path}")

    # =========================================================================
    # STEP 29 -- Final hash check (immutability)
    # =========================================================================
    hashes_after = hash_immutable_set()
    if hashes_after != hashes_before:
        changed = {k: (hashes_before[k], hashes_after[k]) for k in hashes_before if hashes_before[k] != hashes_after[k]}
        raise SystemExit(f"CRITICAL: immutable file hashes changed during Phase 4E: {changed}")
    (EVAL_DIR / "immutability_check.json").write_text(json.dumps(
        {"hashes_before": hashes_before, "hashes_after": hashes_after, "unchanged": True}, indent=2))
    print("\nImmutability check: PASS (all frozen files unchanged)")

    return {
        "winner": winner_name, "frozen_spec": frozen_spec, "final_test_results": final_test_results,
        "candidate_summary": candidate_summary, "artifact_path": artifact_path, "metadata_path": metadata_path,
        "moderate_threshold": moderate_threshold, "high_threshold": high_threshold,
        "validation_risk_tiers": validation_risk_tiers, "test_risk_tiers": test_risk_tiers,
        "subgroup_rows": subgroup_rows, "transport_1": transport_1,
        "rf_best_params": rf_best_params, "xgb_best_params": xgb_best_params,
        "lr_rank": lr_rank, "rf_rank": rf_rank, "xgb_rank": xgb_rank, "hgb_rank": hgb_rank,
        "rf_importance": rf_importance, "xgb_importance": xgb_importance,
        "train_val_gaps": train_val_gaps, "leakage_audit": leakage_audit,
        "confusion_matrices": confusion_matrices,
        "lr_threshold_profile": lr_threshold_profile, "rf_threshold_profile": rf_threshold_profile,
        "xgb_threshold_profile": xgb_threshold_profile, "hgb_threshold_profile": hgb_threshold_profile,
        "winner_threshold_metrics": winner_threshold_metrics,
    }


if __name__ == "__main__":
    result = main()
    print("\n=== Phase 4E complete ===")
    print(f"WINNER: {result['winner']}")
