"""
train.py
--------
Phase 4 orchestration: trains and compares candidate models on TRAIN/
VALIDATION, selects and calibrates a final model using VALIDATION only,
designs risk-tier thresholds using VALIDATION only, freezes the model
specification, evaluates exactly once on TEST, runs a subgroup sanity
check, extracts global feature importance, and writes the versioned
model artifact + metadata + evaluation reports.

Hard rule enforced by control flow (not just documentation): TEST data
is never loaded into scope until AFTER `select_model_on_validation()`
has returned and its result has been written to
artifacts/model_evaluation/final_model_selection.json. The selection
function's signature takes no TEST-related argument at all -- it is
structurally impossible for it to reference TEST.

Run: python backend/modeling/train.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))  # flat sibling imports, matches backend/ convention

import metrics as metrics_mod
import risk_tiers as risk_tiers_mod
from feature_spec import (
    DERIVED_DIR,
    FEATURE_MANIFEST_PATH,
    TARGET_COLUMN,
    load_model_feature_columns,
    load_snapshot_xy,
    split_numeric_categorical,
)
from preprocessing import build_preprocessor, build_scaled_preprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = DERIVED_DIR / "train_snapshot.csv"
VALIDATION_CSV = DERIVED_DIR / "validation_snapshot.csv"
TEST_CSV = DERIVED_DIR / "test_snapshot.csv"

MODELS_DIR = REPO_ROOT / "backend" / "models"
EVAL_DIR = REPO_ROOT / "artifacts" / "model_evaluation"

MODEL_VERSION = "uc07-risk-v1"
RANDOM_STATE = 42
DEFAULT_COMPARISON_THRESHOLD = 0.5  # stated default threshold for candidate_metrics.csv only

warnings.filterwarnings("ignore", category=UserWarning)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Candidate model builders
# ---------------------------------------------------------------------------

def _pipeline(preprocessor, model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_dummy_candidate(numeric_features, categorical_features) -> Pipeline:
    """MODEL A -- naive baseline. strategy='prior' predicts the TRAIN class
    prior for every row, so predict_proba is a constant equal to TRAIN
    prevalence: ROC-AUC ~= 0.5 and PR-AUC ~= prevalence by construction --
    exactly the no-skill baseline every real candidate must beat."""
    pre = build_preprocessor(numeric_features, categorical_features)
    return _pipeline(pre, DummyClassifier(strategy="prior", random_state=RANDOM_STATE))


def logistic_regression_grid() -> list[dict]:
    grid = []
    for C in (0.01, 0.1, 1.0, 10.0):
        for class_weight in (None, "balanced"):
            grid.append({"C": C, "class_weight": class_weight})
    return grid


def build_logistic_regression_candidate(numeric_features, categorical_features, params: dict) -> Pipeline:
    pre = build_scaled_preprocessor(numeric_features, categorical_features)
    model = LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return _pipeline(pre, model)


def random_forest_grid() -> list[dict]:
    grid = []
    for max_depth in (8, 16):
        for min_samples_leaf in (5, 20):
            for class_weight in (None, "balanced"):
                grid.append({"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "class_weight": class_weight})
    return grid


def build_random_forest_candidate(numeric_features, categorical_features, params: dict) -> Pipeline:
    pre = build_preprocessor(numeric_features, categorical_features)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return _pipeline(pre, model)


def _hgb_supports_class_weight() -> bool:
    return "class_weight" in inspect.signature(HistGradientBoostingClassifier.__init__).parameters


def hist_gb_grid() -> list[dict]:
    grid = []
    supports_cw = _hgb_supports_class_weight()
    class_weight_options = (None, "balanced") if supports_cw else (None,)
    for max_iter in (100, 200):
        for max_depth in (3, None):
            for learning_rate in (0.05, 0.1):
                for class_weight in class_weight_options:
                    grid.append({
                        "max_iter": max_iter, "max_depth": max_depth,
                        "learning_rate": learning_rate, "class_weight": class_weight,
                    })
    return grid


def build_hist_gb_candidate(numeric_features, categorical_features, params: dict) -> Pipeline:
    pre = build_preprocessor(numeric_features, categorical_features)
    kwargs = dict(
        max_iter=params["max_iter"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=RANDOM_STATE,
    )
    if params.get("class_weight") is not None and _hgb_supports_class_weight():
        kwargs["class_weight"] = params["class_weight"]
    model = HistGradientBoostingClassifier(**kwargs)
    return _pipeline(pre, model)


# ---------------------------------------------------------------------------
# Hyperparameter search (VALIDATION only)
# ---------------------------------------------------------------------------

def run_hyperparameter_search(
    family: str,
    grid: list[dict],
    build_fn,
    numeric_features, categorical_features,
    X_train, y_train, X_val, y_val,
) -> tuple[list[dict], dict, Pipeline]:
    """Fit every grid combo on TRAIN, score on VALIDATION (PR-AUC primary,
    Brier secondary tie-break). Returns (search_log_rows, best_params, best_fitted_pipeline)."""
    search_log = []
    best = None  # (pr_auc, -brier, params, fitted_pipeline)

    for params in grid:
        pipeline = build_fn(numeric_features, categorical_features, params)
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        rm = metrics_mod.rank_metrics(y_val, val_prob)

        row = {"family": family, **params, **rm}
        search_log.append(row)

        key = (rm["pr_auc"], -rm["brier_score"])
        if best is None or key > (best[0], -best[1]["brier_score"]):
            best = (rm["pr_auc"], rm, params, pipeline)

    _, best_metrics, best_params, best_pipeline = best
    return search_log, best_params, best_pipeline


# ---------------------------------------------------------------------------
# Calibration comparison (fit on TRAIN via CV, evaluated on VALIDATION)
# ---------------------------------------------------------------------------

def compare_calibration_methods(
    build_fn, best_params: dict,
    numeric_features, categorical_features,
    X_train, y_train, X_val, y_val,
) -> dict:
    """For one candidate family's best hyperparameters, compare: no
    calibration, sigmoid (Platt), isotonic -- all calibration mappings
    fit via 5-fold CV on TRAIN only (CalibratedClassifierCV(cv=...)),
    never touching VALIDATION or TEST for fitting. Evaluated on
    VALIDATION. Returns {"uncalibrated": {...}, "sigmoid": {...}, "isotonic": {...}}
    each with rank_metrics + the fitted estimator."""
    results = {}

    uncal_pipeline = build_fn(numeric_features, categorical_features, best_params)
    uncal_pipeline.fit(X_train, y_train)
    uncal_prob = uncal_pipeline.predict_proba(X_val)[:, 1]
    results["uncalibrated"] = {"metrics": metrics_mod.rank_metrics(y_val, uncal_prob), "estimator": uncal_pipeline}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for method in ("sigmoid", "isotonic"):
        base = build_fn(numeric_features, categorical_features, best_params)
        calibrated = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
        calibrated.fit(X_train, y_train)
        prob = calibrated.predict_proba(X_val)[:, 1]
        results[method] = {"metrics": metrics_mod.rank_metrics(y_val, prob), "estimator": calibrated}

    return results


# ---------------------------------------------------------------------------
# Step: select model on VALIDATION only (structurally cannot see TEST)
# ---------------------------------------------------------------------------

def select_model_on_validation(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    numeric_features: list[str], categorical_features: list[str],
) -> dict:
    """
    Runs the full candidate search, calibration comparison, and threshold
    design using ONLY the TRAIN and VALIDATION arguments passed in. There
    is no TEST parameter on this function -- it is structurally impossible
    for this function to reference TEST data (see test_model_pipeline.py's
    test_select_model_on_validation_has_no_test_parameter).
    """
    hyperparam_search_log: list[dict] = []
    candidate_rows: list[dict] = []
    fitted_candidates: dict[str, Pipeline] = {}

    # MODEL A: Dummy baseline
    dummy = build_dummy_candidate(numeric_features, categorical_features)
    dummy.fit(X_train, y_train)
    dummy_val_prob = dummy.predict_proba(X_val)[:, 1]
    candidate_rows.append({"candidate": "dummy_baseline", "family": "Dummy", "hyperparameters": "strategy=prior",
                            **metrics_mod.full_metrics_at_threshold(y_val, dummy_val_prob, DEFAULT_COMPARISON_THRESHOLD)})
    fitted_candidates["dummy_baseline"] = dummy

    # MODEL B: Logistic Regression
    lr_log, lr_best_params, lr_best_pipeline = run_hyperparameter_search(
        "LogisticRegression", logistic_regression_grid(), build_logistic_regression_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    hyperparam_search_log += lr_log
    lr_val_prob = lr_best_pipeline.predict_proba(X_val)[:, 1]
    candidate_rows.append({"candidate": "logistic_regression", "family": "LogisticRegression",
                            "hyperparameters": json.dumps(lr_best_params),
                            **metrics_mod.full_metrics_at_threshold(y_val, lr_val_prob, DEFAULT_COMPARISON_THRESHOLD)})
    fitted_candidates["logistic_regression"] = lr_best_pipeline

    # MODEL C: Random Forest
    rf_log, rf_best_params, rf_best_pipeline = run_hyperparameter_search(
        "RandomForest", random_forest_grid(), build_random_forest_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    hyperparam_search_log += rf_log
    rf_val_prob = rf_best_pipeline.predict_proba(X_val)[:, 1]
    candidate_rows.append({"candidate": "random_forest", "family": "RandomForest",
                            "hyperparameters": json.dumps(rf_best_params),
                            **metrics_mod.full_metrics_at_threshold(y_val, rf_val_prob, DEFAULT_COMPARISON_THRESHOLD)})
    fitted_candidates["random_forest"] = rf_best_pipeline

    # MODEL D: HistGradientBoostingClassifier
    hgb_log, hgb_best_params, hgb_best_pipeline = run_hyperparameter_search(
        "HistGradientBoosting", hist_gb_grid(), build_hist_gb_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    hyperparam_search_log += hgb_log
    hgb_val_prob = hgb_best_pipeline.predict_proba(X_val)[:, 1]
    candidate_rows.append({"candidate": "hist_gradient_boosting", "family": "HistGradientBoosting",
                            "hyperparameters": json.dumps(hgb_best_params),
                            **metrics_mod.full_metrics_at_threshold(y_val, hgb_val_prob, DEFAULT_COMPARISON_THRESHOLD)})
    fitted_candidates["hist_gradient_boosting"] = hgb_best_pipeline

    # ---- Rank real (non-dummy) candidates by PR-AUC to pick the "leading candidates" for calibration ----
    real_candidates = [r for r in candidate_rows if r["candidate"] != "dummy_baseline"]
    real_candidates_sorted = sorted(real_candidates, key=lambda r: r["pr_auc"], reverse=True)
    leading_two = [r["candidate"] for r in real_candidates_sorted[:2]]

    build_fn_by_candidate = {
        "logistic_regression": (build_logistic_regression_candidate, lr_best_params),
        "random_forest": (build_random_forest_candidate, rf_best_params),
        "hist_gradient_boosting": (build_hist_gb_candidate, hgb_best_params),
    }

    calibration_rows = []
    calibration_results_by_candidate = {}
    for cand_name in leading_two:
        build_fn, best_params = build_fn_by_candidate[cand_name]
        cal_results = compare_calibration_methods(
            build_fn, best_params, numeric_features, categorical_features, X_train, y_train, X_val, y_val,
        )
        calibration_results_by_candidate[cand_name] = cal_results
        for method, payload in cal_results.items():
            calibration_rows.append({"candidate": cand_name, "calibration_method": method, **payload["metrics"]})

    # ---- Final selection objective (documented, not just highest ROC-AUC) ----
    # Primary: PR-AUC. Secondary: Brier score (calibration quality). We
    # compare each leading candidate's best calibration variant (by Brier,
    # among variants whose PR-AUC does not regress materially vs
    # uncalibrated) against the others, then apply the interpretability /
    # complexity tie-break only if candidates are close.
    scored_options = []
    for cand_name in leading_two:
        for method, payload in calibration_results_by_candidate[cand_name].items():
            m = payload["metrics"]
            scored_options.append({
                "candidate": cand_name, "calibration_method": method,
                "pr_auc": m["pr_auc"], "brier_score": m["brier_score"], "roc_auc": m["roc_auc"],
                "estimator": payload["estimator"],
            })

    # Sort primarily by PR-AUC (descending), secondarily by Brier (ascending, lower=better)
    scored_options_sorted = sorted(scored_options, key=lambda o: (-o["pr_auc"], o["brier_score"]))
    winner = scored_options_sorted[0]

    # Interpretability/complexity tie-break: if the simplest available
    # option (logistic_regression, uncalibrated) is within a small PR-AUC
    # margin of the winner, prefer it per the spec's "if two models
    # perform very similarly, prefer the simpler model" instruction.
    PR_AUC_TIE_MARGIN = 0.01
    lr_uncal = next((o for o in scored_options if o["candidate"] == "logistic_regression" and o["calibration_method"] == "uncalibrated"), None)
    tie_break_applied = False
    if lr_uncal is not None and lr_uncal is not winner and (winner["pr_auc"] - lr_uncal["pr_auc"]) <= PR_AUC_TIE_MARGIN:
        winner = lr_uncal
        tie_break_applied = True

    final_estimator = winner["estimator"]
    final_val_prob = final_estimator.predict_proba(X_val)[:, 1]

    # ---- Threshold search on VALIDATION ----
    thresholds = [round(t, 3) for t in np.arange(0.02, 0.96, 0.01)]
    threshold_rows = metrics_mod.threshold_sweep(y_val, final_val_prob, thresholds)

    val_probs_sorted = np.sort(final_val_prob)
    high_threshold = float(np.percentile(final_val_prob, 90))
    moderate_threshold = float(np.percentile(final_val_prob, 65))
    if moderate_threshold >= high_threshold:
        moderate_threshold = max(0.0, high_threshold - 0.01)
    risk_tiers_mod.validate_thresholds(moderate_threshold, high_threshold)

    validation_tier_report = risk_tiers_mod.tier_report(y_val, final_val_prob, moderate_threshold, high_threshold)

    return {
        "candidate_rows": candidate_rows,
        "hyperparam_search_log": hyperparam_search_log,
        "calibration_rows": calibration_rows,
        "leading_two_candidates": leading_two,
        "winner": {
            "candidate": winner["candidate"],
            "calibration_method": winner["calibration_method"],
            "pr_auc": winner["pr_auc"],
            "brier_score": winner["brier_score"],
            "roc_auc": winner["roc_auc"],
            "tie_break_applied": tie_break_applied,
        },
        "best_params_by_family": {
            "logistic_regression": lr_best_params,
            "random_forest": rf_best_params,
            "hist_gradient_boosting": hgb_best_params,
        },
        "final_estimator": final_estimator,
        "threshold_rows": threshold_rows,
        "moderate_threshold": round(moderate_threshold, 6),
        "high_threshold": round(high_threshold, 6),
        "validation_tier_report": validation_tier_report,
        "validation_final_metrics": metrics_mod.rank_metrics(y_val, final_val_prob),
        "hgb_supports_class_weight": _hgb_supports_class_weight(),
    }


def _assert_no_test_parameter():
    sig = inspect.signature(select_model_on_validation)
    for name in sig.parameters:
        assert "test" not in name.lower(), f"select_model_on_validation must never take a TEST-named parameter (found {name})"


_assert_no_test_parameter()


# ---------------------------------------------------------------------------
# Feature importance (technical interpretability, not member-level)
# ---------------------------------------------------------------------------

def _get_fold_pipelines(estimator):
    """Return a list of fitted (preprocessor, model) pairs -- one per CV
    fold if `estimator` is a CalibratedClassifierCV, or a single pair if
    it's a plain fitted Pipeline."""
    if isinstance(estimator, CalibratedClassifierCV):
        pairs = []
        for cc in estimator.calibrated_classifiers_:
            base = cc.estimator
            pairs.append((base.named_steps["preprocessor"], base.named_steps["model"]))
        return pairs
    return [(estimator.named_steps["preprocessor"], estimator.named_steps["model"])]


def extract_global_feature_importance(estimator) -> list[dict]:
    fold_pairs = _get_fold_pipelines(estimator)
    preprocessor, first_model = fold_pairs[0]
    feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]

    if hasattr(first_model, "feature_importances_"):
        stacked = np.vstack([m.feature_importances_ for _, m in fold_pairs])
        importances = stacked.mean(axis=0)
        kind = "impurity_based_feature_importance"
    elif hasattr(first_model, "coef_"):
        stacked = np.vstack([m.coef_[0] for _, m in fold_pairs])
        importances = stacked.mean(axis=0)
        kind = "logistic_regression_signed_coefficient"
    else:
        return []

    rows = [{"feature_name": name, "importance": float(value), "importance_type": kind}
            for name, value in zip(feature_names, importances)]
    rows.sort(key=lambda r: abs(r["importance"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# TEST evaluation (called only after the model spec is frozen)
# ---------------------------------------------------------------------------

def evaluate_frozen_model_on_test(
    final_estimator,
    X_test: pd.DataFrame, y_test: pd.Series,
    moderate_threshold: float, high_threshold: float,
    validation_final_metrics: dict,
) -> dict:
    test_prob = final_estimator.predict_proba(X_test)[:, 1]

    test_rank_metrics = metrics_mod.rank_metrics(y_test, test_prob)
    test_moderate_confusion = metrics_mod.threshold_confusion_counts(y_test, test_prob, moderate_threshold)
    test_high_confusion = metrics_mod.threshold_confusion_counts(y_test, test_prob, high_threshold)
    test_tier_report = risk_tiers_mod.tier_report(y_test, test_prob, moderate_threshold, high_threshold)
    test_calibration_bins = metrics_mod.calibration_bins(y_test, test_prob, n_bins=10)

    pr_auc_delta = test_rank_metrics["pr_auc"] - validation_final_metrics["pr_auc"]
    roc_auc_delta = test_rank_metrics["roc_auc"] - validation_final_metrics["roc_auc"]
    brier_delta = test_rank_metrics["brier_score"] - validation_final_metrics["brier_score"]
    material_degradation = (pr_auc_delta < -0.03) or (brier_delta > 0.01)

    return {
        "test_probabilities": test_prob,
        "test_rank_metrics": test_rank_metrics,
        "test_moderate_confusion": test_moderate_confusion,
        "test_high_confusion": test_high_confusion,
        "test_tier_report": test_tier_report,
        "test_calibration_bins": test_calibration_bins,
        "validation_vs_test": {
            "pr_auc_validation": validation_final_metrics["pr_auc"],
            "pr_auc_test": test_rank_metrics["pr_auc"],
            "pr_auc_delta": round(pr_auc_delta, 6),
            "roc_auc_validation": validation_final_metrics["roc_auc"],
            "roc_auc_test": test_rank_metrics["roc_auc"],
            "roc_auc_delta": round(roc_auc_delta, 6),
            "brier_validation": validation_final_metrics["brier_score"],
            "brier_test": test_rank_metrics["brier_score"],
            "brier_delta": round(brier_delta, 6),
            "material_degradation_flag": bool(material_degradation),
            "degradation_rule": "flag if pr_auc_delta < -0.03 or brier_delta > 0.01 (test worse than validation)",
        },
    }


# ---------------------------------------------------------------------------
# Subgroup sanity check (initial only -- not the full Phase 6 bias analysis)
# ---------------------------------------------------------------------------

MIN_SUBGROUP_N = 200
MIN_SUBGROUP_POSITIVES = 15


def _subgroup_row(name: str, mask: pd.Series, y_true: pd.Series, y_prob: np.ndarray, moderate_threshold: float) -> dict | None:
    n = int(mask.sum())
    if n < MIN_SUBGROUP_N:
        return {"subgroup": name, "n": n, "note": "sample size below reporting threshold", "sufficient_sample": False}

    mask_arr = mask.to_numpy()
    yt = y_true.to_numpy()[mask_arr]
    yp = y_prob[mask_arr]
    positives = int(yt.sum())
    prevalence = positives / n

    row = {
        "subgroup": name, "n": n, "sufficient_sample": True,
        "positive_count": positives, "prevalence": round(prevalence, 6),
        "mean_predicted_probability": round(float(yp.mean()), 6),
    }
    if positives < MIN_SUBGROUP_POSITIVES or positives == n:
        row["note"] = "too few positives (or no negatives) for a stable ROC-AUC/PR-AUC estimate"
        row["roc_auc"] = None
        row["pr_auc"] = None
    else:
        row["roc_auc"] = round(float(_safe_roc_auc(yt, yp)), 6)
        row["pr_auc"] = round(float(_safe_pr_auc(yt, yp)), 6)

    conf = metrics_mod.threshold_confusion_counts(yt, yp, moderate_threshold)
    row["recall_at_moderate_threshold"] = conf["recall"]
    row["precision_at_moderate_threshold"] = conf["precision"]
    return row


def _safe_roc_auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_prob)


def _safe_pr_auc(y_true, y_prob):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_prob)


def run_subgroup_checks(X_test: pd.DataFrame, y_test: pd.Series, test_prob: np.ndarray, moderate_threshold: float) -> list[dict]:
    rows = []
    y_prob = np.asarray(test_prob)

    age_bins = [(0, 35), (35, 50), (50, 65), (65, 80), (80, 200)]
    for lo, hi in age_bins:
        mask = (X_test["age"] >= lo) & (X_test["age"] < hi)
        label = f"age_{lo}_{hi if hi < 200 else 'plus'}"
        r = _subgroup_row(label, mask, y_test, y_prob, moderate_threshold)
        if r:
            rows.append(r)

    if "gender" in X_test.columns:
        for g in sorted(X_test["gender"].dropna().unique()):
            mask = X_test["gender"] == g
            r = _subgroup_row(f"gender_{g}", mask, y_test, y_prob, moderate_threshold)
            if r:
                rows.append(r)

    if "clinical_burden" in X_test.columns:
        burden_bins = [(0, 1), (1, 2), (2, 3), (3, 99)]
        for lo, hi in burden_bins:
            mask = (X_test["clinical_burden"] >= lo) & (X_test["clinical_burden"] < hi)
            label = f"clinical_burden_{lo}_{hi if hi < 99 else 'plus'}"
            r = _subgroup_row(label, mask, y_test, y_prob, moderate_threshold)
            if r:
                rows.append(r)

    if "transportation_barrier" in X_test.columns:
        for v in sorted(X_test["transportation_barrier"].dropna().unique()):
            mask = X_test["transportation_barrier"] == v
            r = _subgroup_row(f"transportation_barrier_{v}", mask, y_test, y_prob, moderate_threshold)
            if r:
                rows.append(r)

    return rows


# ---------------------------------------------------------------------------
# Artifact + metadata + report writers
# ---------------------------------------------------------------------------

def write_reports(
    eval_dir: Path,
    candidate_rows, hyperparam_search_log, calibration_rows,
    threshold_rows, validation_tier_report, test_tier_report,
    test_calibration_bins, subgroup_rows, feature_importance_rows,
    final_model_selection: dict,
    validation_scored: pd.DataFrame, test_scored: pd.DataFrame,
):
    eval_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(candidate_rows).to_csv(eval_dir / "candidate_metrics.csv", index=False)
    pd.DataFrame(hyperparam_search_log).to_csv(eval_dir / "hyperparameter_search.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(eval_dir / "calibration_comparison.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(eval_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame(validation_tier_report).to_csv(eval_dir / "validation_risk_tiers.csv", index=False)
    pd.DataFrame(test_tier_report).to_csv(eval_dir / "test_risk_tiers.csv", index=False)
    pd.DataFrame(test_calibration_bins).to_csv(eval_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(eval_dir / "subgroup_metrics.csv", index=False)
    pd.DataFrame(feature_importance_rows).to_csv(eval_dir / "global_feature_importance.csv", index=False)
    (eval_dir / "final_model_selection.json").write_text(json.dumps(final_model_selection, indent=2, default=str))

    validation_scored.to_csv(eval_dir / "validation_scored_members.csv", index=False)
    test_scored.to_csv(eval_dir / "test_scored_members.csv", index=False)


def build_model_artifact(
    final_estimator, feature_columns, numeric_features, categorical_features,
    moderate_threshold, high_threshold, winner: dict, best_params_by_family: dict,
) -> dict:
    return {
        "pipeline": final_estimator,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET_COLUMN,
        "model_version": MODEL_VERSION,
        "algorithm": winner["candidate"],
        "calibration_method": winner["calibration_method"],
        "hyperparameters": best_params_by_family.get(winner["candidate"]),
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
    }


def build_metadata(
    winner, best_params_by_family, moderate_threshold, high_threshold,
    feature_columns, numeric_features, categorical_features,
    train_prevalence, validation_prevalence, test_prevalence,
    validation_final_metrics, test_eval: dict,
    validation_tier_report, test_tier_report,
    raw_hashes: dict, snapshot_hashes: dict,
    hgb_supports_class_weight: bool,
) -> dict:
    return {
        "model_name": "UC07 Avoidable ED Risk Model",
        "model_version": MODEL_VERSION,
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
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "risk_tier_semantics": {
            "LOW": "Lower predicted likelihood; no proactive escalation by risk alone.",
            "MODERATE": "Meaningful navigation opportunity; suitable for light-touch navigation or review.",
            "HIGH": "Strongest predicted navigation opportunity; candidate for stronger outreach / Care Management review in a later phase.",
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
        "validation_risk_tiers": validation_tier_report,
        "final_test_metrics": test_eval["test_rank_metrics"],
        "final_test_confusion_at_moderate_threshold": test_eval["test_moderate_confusion"],
        "final_test_confusion_at_high_threshold": test_eval["test_high_confusion"],
        "test_risk_tiers": test_tier_report,
        "validation_vs_test": test_eval["validation_vs_test"],
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dataset_sha256": raw_hashes,
        "phase3_snapshot_sha256": snapshot_hashes,
        "feature_manifest_reference": str(FEATURE_MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "hgb_class_weight_supported_by_installed_sklearn": hgb_supports_class_weight,
        "random_seed": RANDOM_STATE,
        "known_limitations": [
            "Trained on a single ~18-month data era reused across 3 non-overlapping temporal snapshots (Phase 2/3 limitation, unchanged here).",
            "~7% positive-label overlap between TRAIN/VALIDATION/TEST members (Phase 2 section 9.4) -- accepted, monitored non-independence, not eliminated.",
            "UNCERTAIN encounters never drive a positive label (Phase 2 section 6.2) -- intentionally conservative, likely undercounts achievable recall.",
            "diagnosis-derived features remain excluded from the feature set (Phase 3 baseline decision) -- their predictive value is unverified.",
            "This is an initial subgroup sanity check only, not a full fairness/bias audit (Phase 6 scope).",
            "Risk tiers are risk-only; Care Management/navigation routing logic is explicitly deferred to Phase 5.",
        ],
    }


def main() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_paths = {"train": TRAIN_CSV, "validation": VALIDATION_CSV, "test": TEST_CSV}
    snapshot_hashes_before = {name: sha256_file(p) for name, p in snapshot_paths.items()}

    raw_paths = {
        "raw_members.csv": REPO_ROOT / "raw_members.csv",
        "raw_ed_visits.csv": REPO_ROOT / "raw_ed_visits.csv",
        "raw_care_history.csv": REPO_ROOT / "raw_care_history.csv",
    }
    raw_hashes = {name: sha256_file(p) for name, p in raw_paths.items()}

    feature_columns = load_model_feature_columns()

    X_train, y_train, _ = load_snapshot_xy(TRAIN_CSV)
    X_val, y_val, val_ids = load_snapshot_xy(VALIDATION_CSV)
    numeric_features, categorical_features = split_numeric_categorical(X_train, feature_columns)

    print(f"Loaded TRAIN={len(X_train)} VALIDATION={len(X_val)} rows; "
          f"{len(feature_columns)} features ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")

    # ---- Selection happens on TRAIN + VALIDATION only ----
    selection = select_model_on_validation(X_train, y_train, X_val, y_val, numeric_features, categorical_features)

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
        "algorithm": selection["winner"]["candidate"],
        "calibration_method": selection["winner"]["calibration_method"],
        "hyperparameters": selection["best_params_by_family"].get(selection["winner"]["candidate"]),
        "preprocessing": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot(handle_unknown=ignore)",
            "scaling": "StandardScaler (logistic_regression only)" if selection["winner"]["candidate"] == "logistic_regression" else "none (tree-based, scale invariant)",
        },
        "feature_list": feature_columns,
        "feature_count": len(feature_columns),
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "model_version_candidate": MODEL_VERSION,
        "validation_metrics_at_freeze": selection["validation_final_metrics"],
        "tie_break_applied": selection["winner"]["tie_break_applied"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "This spec is frozen BEFORE any TEST data is loaded. It must not change based on TEST results.",
    }
    (EVAL_DIR / "final_model_selection.json").write_text(json.dumps(frozen_spec, indent=2, default=str))
    print(f"FROZEN before TEST: algorithm={frozen_spec['algorithm']} calibration={frozen_spec['calibration_method']} "
          f"moderate_threshold={moderate_threshold:.4f} high_threshold={high_threshold:.4f}")

    # ---- Only now does TEST enter scope ----
    X_test, y_test, test_ids = load_snapshot_xy(TEST_CSV)

    test_eval = evaluate_frozen_model_on_test(
        final_estimator, X_test, y_test, moderate_threshold, high_threshold, selection["validation_final_metrics"],
    )
    test_prob = test_eval["test_probabilities"]
    test_scored = pd.DataFrame({
        "member_id": test_ids,
        "predicted_probability": test_prob,
        "risk_tier": risk_tiers_mod.assign_risk_tiers(test_prob, moderate_threshold, high_threshold),
    })

    subgroup_rows = run_subgroup_checks(X_test, y_test, test_prob, moderate_threshold)
    feature_importance_rows = extract_global_feature_importance(final_estimator)

    write_reports(
        EVAL_DIR,
        selection["candidate_rows"], selection["hyperparam_search_log"], selection["calibration_rows"],
        selection["threshold_rows"], selection["validation_tier_report"], test_eval["test_tier_report"],
        test_eval["test_calibration_bins"], subgroup_rows, feature_importance_rows,
        frozen_spec, validation_scored, test_scored,
    )

    artifact = build_model_artifact(
        final_estimator, feature_columns, numeric_features, categorical_features,
        moderate_threshold, high_threshold, selection["winner"], selection["best_params_by_family"],
    )
    artifact_path = MODELS_DIR / f"{MODEL_VERSION.replace('-', '_')}_model.joblib"
    joblib.dump(artifact, artifact_path)

    snapshot_hashes_after = {name: sha256_file(p) for name, p in snapshot_paths.items()}
    if snapshot_hashes_after != snapshot_hashes_before:
        raise SystemExit(f"CRITICAL: Phase 3 snapshot hashes changed during Phase 4 training run. "
                          f"before={snapshot_hashes_before} after={snapshot_hashes_after}")

    metadata = build_metadata(
        selection["winner"], selection["best_params_by_family"], moderate_threshold, high_threshold,
        feature_columns, numeric_features, categorical_features,
        round(float(y_train.mean()), 6), round(float(y_val.mean()), 6), round(float(y_test.mean()), 6),
        selection["validation_final_metrics"], test_eval,
        selection["validation_tier_report"], test_eval["test_tier_report"],
        raw_hashes, snapshot_hashes_after, selection["hgb_supports_class_weight"],
    )
    metadata_path = MODELS_DIR / f"{MODEL_VERSION.replace('-', '_')}_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    return {
        "selection": selection,
        "frozen_spec": frozen_spec,
        "test_eval": test_eval,
        "subgroup_rows": subgroup_rows,
        "feature_importance_rows": feature_importance_rows,
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
        "snapshot_hashes_before": snapshot_hashes_before,
        "snapshot_hashes_after": snapshot_hashes_after,
        "raw_hashes": raw_hashes,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


if __name__ == "__main__":
    result = main()
    print("\n=== Phase 4 training run complete ===")
    print(f"Artifact: {result['artifact_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Final TEST PR-AUC: {result['test_eval']['test_rank_metrics']['pr_auc']:.4f} "
          f"ROC-AUC: {result['test_eval']['test_rank_metrics']['roc_auc']:.4f} "
          f"Brier: {result['test_eval']['test_rank_metrics']['brier_score']:.4f}")
