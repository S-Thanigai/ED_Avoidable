"""
improve.py
----------
Phase 4B orchestration: controlled feature/model improvement experiments
against uc07-risk-v1, using TRAIN for fitting and VALIDATION for every
experimental decision. TEST is loaded only if a v2 candidate passes the
VALIDATION promotion gate, and only inside `evaluate_v2_on_test()`, which
takes no other Phase 4B state as input besides the already-frozen
candidate -- mirroring Phase 4's TEST-isolation pattern.

Run: python backend/modeling/improve.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pit"))

import metrics as metrics_mod
import risk_tiers as risk_tiers_mod
import train as v1_train
from feature_spec import DERIVED_DIR, FEATURE_MANIFEST_PATH, TARGET_COLUMN, load_model_feature_columns, split_numeric_categorical
from preprocessing import build_preprocessor, build_scaled_preprocessor
from windows import build_all_snapshot_windows
import features_v2 as fv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "backend" / "models"
EVAL_DIR = REPO_ROOT / "artifacts" / "model_improvement"
RANDOM_STATE = 42
V1_MODEL_VERSION = "uc07-risk-v1"
V2_MODEL_VERSION = "uc07-risk-v2"

RAW_PATHS = {
    "raw_members.csv": REPO_ROOT / "raw_members.csv",
    "raw_ed_visits.csv": REPO_ROOT / "raw_ed_visits.csv",
    "raw_care_history.csv": REPO_ROOT / "raw_care_history.csv",
}
SNAPSHOT_PATHS = {
    "train": DERIVED_DIR / "train_snapshot.csv",
    "validation": DERIVED_DIR / "validation_snapshot.csv",
    "test": DERIVED_DIR / "test_snapshot.csv",
}

V1_ED_WINDOW_COLS = [
    "prior_ed_count_30d", "prior_ed_count_90d", "prior_ed_count_180d", "prior_ed_count_270d",
    "prior_potentially_avoidable_ed_count_30d", "prior_potentially_avoidable_ed_count_90d",
    "prior_potentially_avoidable_ed_count_180d", "prior_potentially_avoidable_ed_count_270d",
]
V1_OLD_VELOCITY_COLS = ["ed_utilization_velocity_30_over_180", "potentially_avoidable_ed_velocity_90_over_270"]

VELOCITY_GROUP_COLS = [
    "ed_acceleration_30_vs_240", "potentially_avoidable_ed_acceleration_30_vs_240",
    "alternative_care_engagement_trend_90_vs_270",
]
CARE_MIX_CONTINUITY_COLS = [
    "total_outpatient_alternative_visits_270d", "ed_to_outpatient_ratio_270d",
    "ed_share_of_total_utilization_270d", "telehealth_share_270d", "urgent_care_share_270d",
    "pcp_share_270d", "has_recent_outpatient_followup_after_last_ed",
    "days_from_last_ed_to_next_outpatient", "care_setting_diversity_270d",
    "days_since_any_outpatient_contact", "long_gap_without_outpatient_care_flag",
    "recent_outpatient_contact_30d_flag", "repeated_ED_without_recent_PCP_flag",
]
ACCESS_INTERACTION_COLS = [
    "transportation_barrier_x_recent_ed", "pcp_distance_x_recent_ed",
    "urgent_care_distance_x_recent_ed", "telehealth_available_x_recent_ed",
    "chronic_burden_x_access_barrier", "chronic_burden_x_recent_ed",
]
HISTORICAL_ED_PATTERN_EXTRA_COLS = ["avoidable_share_of_prior_ed_270d", "repeat_potentially_avoidable_ed_flag"]
DIAGNOSIS_COLS = [
    "distinct_prior_ed_diagnosis_categories_270d", "most_common_prior_diagnosis_share_270d",
    "prior_diagnosis_diversity_ratio_270d", "repeat_same_diagnosis_flag_270d",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data loading (TRAIN + VALIDATION only in the main flow)
# ---------------------------------------------------------------------------

def load_raw():
    members = pd.read_csv(RAW_PATHS["raw_members.csv"])
    ed = pd.read_csv(RAW_PATHS["raw_ed_visits.csv"])
    care = pd.read_csv(RAW_PATHS["raw_care_history.csv"])
    return members, ed, care


def build_candidate_feature_frames(snapshot_name: str, members, ed, care, windows) -> dict:
    """Build every Phase 4B candidate feature frame for one snapshot
    (train/validation, or test only inside the gated TEST step)."""
    window = windows[snapshot_name]
    v1_df = pd.read_csv(SNAPSHOT_PATHS[snapshot_name])
    member_order = members["member_id"].drop_duplicates().reset_index(drop=True)

    banded = fv2.build_banded_ed_features(ed, member_order, window)
    reduced = fv2.build_reduced_window_ed_features(ed, member_order, window)
    extended = fv2.build_extended_candidate_features(members, ed, care, v1_df, window, include_diagnosis=True)

    return {"v1": v1_df, "banded": banded, "reduced": reduced, "extended": extended}


# ---------------------------------------------------------------------------
# Step 3: window representation experiment
# ---------------------------------------------------------------------------

def _fit_lr_grid(X_train, y_train, X_val, y_val, numeric_features, categorical_features):
    grid = v1_train.logistic_regression_grid()
    log, best_params, best_pipeline = v1_train.run_hyperparameter_search(
        "LogisticRegression", grid, v1_train.build_logistic_regression_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    return best_params, best_pipeline, metrics_mod.rank_metrics(y_val, val_prob), val_prob


def run_window_experiment(train_frames, val_frames, y_train, y_val) -> dict:
    v1_train_df, v1_val_df = train_frames["v1"], val_frames["v1"]

    variants = {}
    variants["baseline_nested"] = (v1_train_df[V1_ED_WINDOW_COLS], v1_val_df[V1_ED_WINDOW_COLS])

    reduced_cols = [c for c in train_frames["reduced"].columns if c != "member_id"]
    variants["reduced_nested_30_90_270"] = (train_frames["reduced"][reduced_cols], val_frames["reduced"][reduced_cols])

    banded_cols = [c for c in train_frames["banded"].columns if c != "member_id"]
    variants["non_overlapping_bands"] = (train_frames["banded"][banded_cols], val_frames["banded"][banded_cols])

    other_v1_cols = [c for c in load_model_feature_columns() if c not in V1_ED_WINDOW_COLS]
    X_train_other = v1_train_df[other_v1_cols]
    X_val_other = v1_val_df[other_v1_cols]

    rows = []
    fitted = {}
    for variant_name, (train_window_cols, val_window_cols) in variants.items():
        X_train = pd.concat([X_train_other.reset_index(drop=True), train_window_cols.reset_index(drop=True)], axis=1)
        X_val = pd.concat([X_val_other.reset_index(drop=True), val_window_cols.reset_index(drop=True)], axis=1)
        numeric, categorical = split_numeric_categorical(X_train, list(X_train.columns))
        best_params, pipeline, rm, val_prob = _fit_lr_grid(X_train, y_train, X_val, y_val, numeric, categorical)
        rows.append({"variant": variant_name, "feature_count": X_train.shape[1], "best_lr_params": json.dumps(best_params), **rm})
        fitted[variant_name] = {"pipeline": pipeline, "X_train": X_train, "X_val": X_val, "val_prob": val_prob}

    winner_name = max(rows, key=lambda r: r["pr_auc"])["variant"]
    return {"rows": rows, "winner": winner_name, "fitted": fitted}


# ---------------------------------------------------------------------------
# Step 10: feature ablation study (Experiments A-G)
# ---------------------------------------------------------------------------

def _assemble_experiment_features(experiment: str, window_winner: str, train_frames, val_frames) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1_train_df, v1_val_df = train_frames["v1"], val_frames["v1"]
    v1_cols = load_model_feature_columns()

    if experiment == "A_v1_baseline":
        return v1_train_df[v1_cols].copy(), v1_val_df[v1_cols].copy()

    base_cols = [c for c in v1_cols if c not in V1_ED_WINDOW_COLS + V1_OLD_VELOCITY_COLS]
    if window_winner == "reduced_nested_30_90_270":
        window_cols = [c for c in train_frames["reduced"].columns if c != "member_id"]
        window_train = train_frames["reduced"][window_cols]
        window_val = val_frames["reduced"][window_cols]
    elif window_winner == "non_overlapping_bands":
        window_cols = [c for c in train_frames["banded"].columns if c != "member_id"]
        window_train = train_frames["banded"][window_cols]
        window_val = val_frames["banded"][window_cols]
    else:
        window_cols = V1_ED_WINDOW_COLS
        window_train = v1_train_df[window_cols]
        window_val = v1_val_df[window_cols]

    X_train = pd.concat([v1_train_df[base_cols].reset_index(drop=True), window_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([v1_val_df[base_cols].reset_index(drop=True), window_val.reset_index(drop=True)], axis=1)
    if experiment == "B_restructured_windows":
        return X_train, X_val

    add_cols_cumulative: list[str] = []
    if experiment in ("C_plus_velocity", "D_plus_care_mix_continuity", "E_plus_access_interactions", "F_plus_historical_ed_extras", "G_plus_diagnosis"):
        add_cols_cumulative += VELOCITY_GROUP_COLS
    if experiment in ("D_plus_care_mix_continuity", "E_plus_access_interactions", "F_plus_historical_ed_extras", "G_plus_diagnosis"):
        add_cols_cumulative += CARE_MIX_CONTINUITY_COLS
    if experiment in ("E_plus_access_interactions", "F_plus_historical_ed_extras", "G_plus_diagnosis"):
        add_cols_cumulative += ACCESS_INTERACTION_COLS
    if experiment in ("F_plus_historical_ed_extras", "G_plus_diagnosis"):
        add_cols_cumulative += HISTORICAL_ED_PATTERN_EXTRA_COLS
    if experiment == "G_plus_diagnosis":
        add_cols_cumulative += DIAGNOSIS_COLS

    X_train = pd.concat([X_train.reset_index(drop=True), train_frames["extended"][add_cols_cumulative].reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_val.reset_index(drop=True), val_frames["extended"][add_cols_cumulative].reset_index(drop=True)], axis=1)
    return X_train, X_val


EXPERIMENTS = [
    "A_v1_baseline", "B_restructured_windows", "C_plus_velocity",
    "D_plus_care_mix_continuity", "E_plus_access_interactions",
    "F_plus_historical_ed_extras", "G_plus_diagnosis",
]


def run_ablation_study(window_winner: str, train_frames, val_frames, y_train, y_val) -> dict:
    rows = []
    fitted = {}
    for experiment in EXPERIMENTS:
        X_train, X_val = _assemble_experiment_features(experiment, window_winner, train_frames, val_frames)
        numeric, categorical = split_numeric_categorical(X_train, list(X_train.columns))
        best_params, pipeline, rm, val_prob = _fit_lr_grid(X_train, y_train, X_val, y_val, numeric, categorical)

        op_threshold = float(np.percentile(val_prob, 90))
        conf = metrics_mod.threshold_confusion_counts(y_val, val_prob, op_threshold)

        rows.append({
            "experiment": experiment, "feature_count": X_train.shape[1],
            "best_lr_params": json.dumps(best_params),
            "operating_threshold_p90": round(op_threshold, 6),
            "recall_at_p90": conf["recall"], "precision_at_p90": conf["precision"],
            **rm,
        })
        fitted[experiment] = {"pipeline": pipeline, "X_train": X_train, "X_val": X_val, "val_prob": val_prob, "numeric": numeric, "categorical": categorical}

    return {"rows": rows, "fitted": fitted}


# ---------------------------------------------------------------------------
# Step 11: model comparison on the winning ablation feature set
# ---------------------------------------------------------------------------

def run_model_comparison(X_train, y_train, X_val, y_val, numeric_features, categorical_features) -> dict:
    rows = []
    fitted = {}

    lr_log, lr_params, lr_pipeline = v1_train.run_hyperparameter_search(
        "LogisticRegression", v1_train.logistic_regression_grid(), v1_train.build_logistic_regression_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    lr_prob = lr_pipeline.predict_proba(X_val)[:, 1]
    rows.append({"algorithm": "logistic_regression", "hyperparameters": json.dumps(lr_params), **metrics_mod.rank_metrics(y_val, lr_prob)})
    fitted["logistic_regression"] = {"pipeline": lr_pipeline, "params": lr_params}

    rf_log, rf_params, rf_pipeline = v1_train.run_hyperparameter_search(
        "RandomForest", v1_train.random_forest_grid(), v1_train.build_random_forest_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    rf_prob = rf_pipeline.predict_proba(X_val)[:, 1]
    rows.append({"algorithm": "random_forest", "hyperparameters": json.dumps(rf_params), **metrics_mod.rank_metrics(y_val, rf_prob)})
    fitted["random_forest"] = {"pipeline": rf_pipeline, "params": rf_params}

    hgb_log, hgb_params, hgb_pipeline = v1_train.run_hyperparameter_search(
        "HistGradientBoosting", v1_train.hist_gb_grid(), v1_train.build_hist_gb_candidate,
        numeric_features, categorical_features, X_train, y_train, X_val, y_val,
    )
    hgb_prob = hgb_pipeline.predict_proba(X_val)[:, 1]
    rows.append({"algorithm": "hist_gradient_boosting", "hyperparameters": json.dumps(hgb_params), **metrics_mod.rank_metrics(y_val, hgb_prob)})
    fitted["hist_gradient_boosting"] = {"pipeline": hgb_pipeline, "params": hgb_params}

    return {"rows": rows, "fitted": fitted, "search_logs": lr_log + rf_log + hgb_log}


# ---------------------------------------------------------------------------
# Step 12: Logistic Regression regularization / stability
# ---------------------------------------------------------------------------

def _build_lr_with_penalty(numeric_features, categorical_features, params: dict):
    pre = build_scaled_preprocessor(numeric_features, categorical_features)
    kwargs = dict(C=params["C"], class_weight=params.get("class_weight"), max_iter=3000, random_state=RANDOM_STATE)
    if params["penalty"] == "l2":
        model = LogisticRegression(penalty="l2", solver="lbfgs", **kwargs)
    elif params["penalty"] == "l1":
        model = LogisticRegression(penalty="l1", solver="saga", **kwargs)
    else:
        model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, **kwargs)
    from sklearn.pipeline import Pipeline
    return Pipeline(steps=[("preprocessor", pre), ("model", model)])


def run_regularization_experiment(X_train, y_train, X_val, y_val, numeric_features, categorical_features, class_weight) -> list[dict]:
    rows = []
    for penalty in ("l2", "l1", "elasticnet"):
        for C in (0.01, 0.1, 1.0):
            params = {"C": C, "penalty": penalty, "class_weight": class_weight}
            pipeline = _build_lr_with_penalty(numeric_features, categorical_features, params)
            pipeline.fit(X_train, y_train)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            model = pipeline.named_steps["model"]
            n_nonzero_coefs = int(np.sum(np.abs(model.coef_[0]) > 1e-6))
            rows.append({
                "penalty": penalty, "C": C, "class_weight": class_weight,
                "n_nonzero_coefficients": n_nonzero_coefs, "n_total_coefficients": len(model.coef_[0]),
                **metrics_mod.rank_metrics(y_val, val_prob),
            })
    return rows


# ---------------------------------------------------------------------------
# Step 15: suspicious performance guardrail
# ---------------------------------------------------------------------------

def check_suspicious_performance(rows: list[dict], roc_auc_ceiling: float = 0.80) -> list[dict]:
    flags = []
    for row in rows:
        if row.get("roc_auc", 0) is not None and row.get("roc_auc", 0) > roc_auc_ceiling:
            flags.append({"row": row, "reason": f"roc_auc {row['roc_auc']} exceeds suspicious ceiling {roc_auc_ceiling}"})
    return flags


# ---------------------------------------------------------------------------
# Master VALIDATION-only selection flow (Steps 3, 9-17) -- NO TEST parameter
# ---------------------------------------------------------------------------

def select_v2_candidate_on_validation(
    members: pd.DataFrame, ed: pd.DataFrame, care: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series,
    windows: dict,
) -> dict:
    """
    Runs the entire Phase 4B experimentation flow using only TRAIN and
    VALIDATION. Structurally cannot reference TEST -- no TEST-named
    parameter exists on this function (verified by
    test_select_v2_candidate_on_validation_has_no_test_parameter).
    """
    train_frames = build_candidate_feature_frames("train", members, ed, care, windows)
    val_frames = build_candidate_feature_frames("validation", members, ed, care, windows)

    # ---- Step 3: window representation experiment ----
    window_experiment = run_window_experiment(train_frames, val_frames, y_train, y_val)
    window_winner = window_experiment["winner"]

    # ---- Step 10: ablation study A-G (diagnosis-free rows: A-F) ----
    ablation = run_ablation_study(window_winner, train_frames, val_frames, y_train, y_val)

    # Evidence-based winner selection: argmax VALIDATION PR-AUC among the
    # diagnosis-free experiments (A-F), NOT a mechanical cascade to the
    # last experiment in the chain -- a feature group that doesn't help
    # (or hurts) must not be kept just because it was tried. A Brier
    # sanity filter excludes any experiment whose calibration broke
    # (e.g. from a class_weight="balanced" grid winner distorting raw
    # probabilities, as seen for HistGradientBoosting in Phase 4) even if
    # its PR-AUC looks fine, so a badly-calibrated experiment can't win by
    # ranking alone.
    non_diagnosis_rows = [r for r in ablation["rows"] if r["experiment"] != "G_plus_diagnosis"]
    baseline_brier = next(r["brier_score"] for r in non_diagnosis_rows if r["experiment"] == "A_v1_baseline")
    eligible_rows = [r for r in non_diagnosis_rows if r["brier_score"] <= baseline_brier * 1.5]
    best_non_diagnosis_row = max(eligible_rows, key=lambda r: r["pr_auc"])
    best_non_diagnosis_experiment = best_non_diagnosis_row["experiment"]

    best_non_diag_fit = ablation["fitted"][best_non_diagnosis_experiment]
    X_train_winner, X_val_winner = best_non_diag_fit["X_train"], best_non_diag_fit["X_val"]

    # ---- Step 9 isolated diagnosis view: (actual PR-AUC winner) vs (winner + diagnosis) ----
    X_train_with_diag = pd.concat([X_train_winner.reset_index(drop=True), train_frames["extended"][DIAGNOSIS_COLS].reset_index(drop=True)], axis=1)
    X_val_with_diag = pd.concat([X_val_winner.reset_index(drop=True), val_frames["extended"][DIAGNOSIS_COLS].reset_index(drop=True)], axis=1)
    numeric_wo, categorical_wo = split_numeric_categorical(X_train_winner, list(X_train_winner.columns))
    numeric_w, categorical_w = split_numeric_categorical(X_train_with_diag, list(X_train_with_diag.columns))
    _, _, rm_without, _ = _fit_lr_grid(X_train_winner, y_train, X_val_winner, y_val, numeric_wo, categorical_wo)
    _, _, rm_with, _ = _fit_lr_grid(X_train_with_diag, y_train, X_val_with_diag, y_val, numeric_w, categorical_w)
    diagnosis_experiment = {
        "base_experiment": best_non_diagnosis_experiment,
        "without": {"feature_count": X_train_winner.shape[1], **rm_without},
        "with": {"feature_count": X_train_with_diag.shape[1], **rm_with},
        "pr_auc_gain": round(rm_with["pr_auc"] - rm_without["pr_auc"], 6),
    }
    diagnosis_kept = diagnosis_experiment["pr_auc_gain"] >= 0.005  # small, real, non-trivial incremental gain required

    if diagnosis_kept:
        best_ablation_experiment = f"{best_non_diagnosis_experiment}+diagnosis"
        X_train_best, X_val_best = X_train_with_diag, X_val_with_diag
        numeric_best, categorical_best = numeric_w, categorical_w
    else:
        best_ablation_experiment = best_non_diagnosis_experiment
        X_train_best, X_val_best = X_train_winner, X_val_winner
        numeric_best, categorical_best = numeric_wo, categorical_wo

    # ---- Step 15 guardrail on ablation + diagnosis + window rows ----
    suspicious = check_suspicious_performance(
        ablation["rows"] + window_experiment["rows"]
        + [{"roc_auc": diagnosis_experiment["without"]["roc_auc"]}, {"roc_auc": diagnosis_experiment["with"]["roc_auc"]}]
    )

    # ---- Step 11: model comparison on the winning feature set ----
    model_comparison = run_model_comparison(X_train_best, y_train, X_val_best, y_val, numeric_best, categorical_best)
    suspicious += check_suspicious_performance(model_comparison["rows"])

    best_model_row = max(model_comparison["rows"], key=lambda r: r["pr_auc"])
    best_algorithm = best_model_row["algorithm"]

    # ---- Step 12: LR regularization stability (only meaningful if LR remains competitive) ----
    lr_class_weight = model_comparison["fitted"]["logistic_regression"]["params"].get("class_weight")
    regularization_rows = run_regularization_experiment(X_train_best, y_train, X_val_best, y_val, numeric_best, categorical_best, lr_class_weight)
    best_regularized_row = max(regularization_rows, key=lambda r: r["pr_auc"])

    # ---- Step 13: calibration comparison for the leading candidate(s) ----
    calibration_rows = []
    candidates_for_calibration = sorted(model_comparison["rows"], key=lambda r: r["pr_auc"], reverse=True)[:2]
    build_fn_map = {
        "logistic_regression": (v1_train.build_logistic_regression_candidate, model_comparison["fitted"]["logistic_regression"]["params"]),
        "random_forest": (v1_train.build_random_forest_candidate, model_comparison["fitted"]["random_forest"]["params"]),
        "hist_gradient_boosting": (v1_train.build_hist_gb_candidate, model_comparison["fitted"]["hist_gradient_boosting"]["params"]),
    }
    calibration_estimators = {}
    for row in candidates_for_calibration:
        algo = row["algorithm"]
        build_fn, params = build_fn_map[algo]
        cal_results = v1_train.compare_calibration_methods(build_fn, params, numeric_best, categorical_best, X_train_best, y_train, X_val_best, y_val)
        calibration_estimators[algo] = cal_results
        for method, payload in cal_results.items():
            calibration_rows.append({"algorithm": algo, "calibration_method": method, **payload["metrics"]})

    # ---- Final V2 winner across algorithm x calibration ----
    scored = []
    for algo, cal_results in calibration_estimators.items():
        for method, payload in cal_results.items():
            m = payload["metrics"]
            scored.append({"algorithm": algo, "calibration_method": method, "pr_auc": m["pr_auc"], "brier_score": m["brier_score"], "roc_auc": m["roc_auc"], "estimator": payload["estimator"]})
    scored_sorted = sorted(scored, key=lambda o: (-o["pr_auc"], o["brier_score"]))
    v2_winner = scored_sorted[0]
    v2_estimator = v2_winner["estimator"]
    v2_val_prob = v2_estimator.predict_proba(X_val_best)[:, 1]

    # ---- Step 16: VALIDATION risk tiers for V2 candidate (own thresholds, not reused from v1) ----
    v2_high_threshold = float(np.percentile(v2_val_prob, 90))
    v2_moderate_threshold = float(np.percentile(v2_val_prob, 65))
    if v2_moderate_threshold >= v2_high_threshold:
        v2_moderate_threshold = max(0.0, v2_high_threshold - 0.01)
    risk_tiers_mod.validate_thresholds(v2_moderate_threshold, v2_high_threshold)
    v2_tier_report = risk_tiers_mod.tier_report(y_val, v2_val_prob, v2_moderate_threshold, v2_high_threshold)

    v1_val_prob = None  # computed by caller from the already-loaded v1 artifact for a like-for-like comparison

    return {
        "window_experiment": window_experiment,
        "ablation": ablation,
        "diagnosis_experiment": diagnosis_experiment,
        "diagnosis_kept": diagnosis_kept,
        "best_ablation_experiment": best_ablation_experiment,
        "model_comparison": model_comparison,
        "best_algorithm": best_algorithm,
        "regularization_rows": regularization_rows,
        "best_regularized_row": best_regularized_row,
        "calibration_rows": calibration_rows,
        "v2_winner": {k: v for k, v in v2_winner.items() if k != "estimator"},
        "v2_estimator": v2_estimator,
        "v2_moderate_threshold": round(v2_moderate_threshold, 6),
        "v2_high_threshold": round(v2_high_threshold, 6),
        "v2_tier_report": v2_tier_report,
        "v2_validation_metrics": metrics_mod.rank_metrics(y_val, v2_val_prob),
        "X_train_best": X_train_best, "X_val_best": X_val_best,
        "numeric_best": numeric_best, "categorical_best": categorical_best,
        "feature_columns_best": list(X_train_best.columns),
        "suspicious_flags": suspicious,
    }


def _assert_no_test_parameter():
    sig = inspect.signature(select_v2_candidate_on_validation)
    for name in sig.parameters:
        assert "test" not in name.lower(), f"select_v2_candidate_on_validation must never take a TEST-named parameter (found {name})"


_assert_no_test_parameter()


# ---------------------------------------------------------------------------
# Step 14: promotion decision (VALIDATION only)
# ---------------------------------------------------------------------------

PR_AUC_PROMOTION_MARGIN = 0.01
BRIER_REGRESSION_TOLERANCE = 0.005
TIER_LIFT_PROMOTION_MARGIN = 0.15  # relative improvement in HIGH-tier lift considered "meaningfully stronger"


def make_promotion_decision(v1_val_metrics: dict, v1_val_tier_report: list[dict], v2_result: dict) -> dict:
    v2_metrics = v2_result["v2_validation_metrics"]
    pr_auc_gain = v2_metrics["pr_auc"] - v1_val_metrics["pr_auc"]
    brier_change = v2_metrics["brier_score"] - v1_val_metrics["brier_score"]  # negative = better

    v1_high = next(r for r in v1_val_tier_report if r["tier"] == "HIGH")
    v2_high = next(r for r in v2_result["v2_tier_report"] if r["tier"] == "HIGH")
    v1_high_lift = v1_high["lift_vs_overall_prevalence"] or 0.0
    v2_high_lift = v2_high["lift_vs_overall_prevalence"] or 0.0
    lift_gain_relative = (v2_high_lift - v1_high_lift) / v1_high_lift if v1_high_lift else 0.0

    v2_tiers = {r["tier"]: r["observed_prevalence"] for r in v2_result["v2_tier_report"]}
    monotonic = (
        v2_tiers["LOW"] is not None and v2_tiers["MODERATE"] is not None and v2_tiers["HIGH"] is not None
        and v2_tiers["LOW"] <= v2_tiers["MODERATE"] <= v2_tiers["HIGH"]
    )

    calibration_preserved = brier_change <= BRIER_REGRESSION_TOLERANCE
    pr_auc_case = pr_auc_gain >= PR_AUC_PROMOTION_MARGIN
    tier_lift_case = lift_gain_relative >= TIER_LIFT_PROMOTION_MARGIN

    promote = bool((pr_auc_case or tier_lift_case) and calibration_preserved and monotonic and not v2_result["suspicious_flags"])

    reasons = []
    reasons.append(f"PR-AUC gain vs V1 VALIDATION: {pr_auc_gain:+.4f} (promotion margin: >= {PR_AUC_PROMOTION_MARGIN})")
    reasons.append(f"Brier change vs V1 VALIDATION: {brier_change:+.4f} (regression tolerance: <= {BRIER_REGRESSION_TOLERANCE})")
    reasons.append(f"HIGH-tier lift: V1={v1_high_lift:.3f}x V2={v2_high_lift:.3f}x (relative gain {lift_gain_relative:+.3f}, margin: >= {TIER_LIFT_PROMOTION_MARGIN})")
    reasons.append(f"V2 tier monotonicity (LOW<=MODERATE<=HIGH observed prevalence): {monotonic}")
    reasons.append(f"Suspicious performance flags: {len(v2_result['suspicious_flags'])}")

    return {
        "promote": promote,
        "pr_auc_gain": round(pr_auc_gain, 6),
        "brier_change": round(brier_change, 6),
        "v1_high_lift": round(v1_high_lift, 4),
        "v2_high_lift": round(v2_high_lift, 4),
        "lift_gain_relative": round(lift_gain_relative, 4),
        "tier_monotonic": monotonic,
        "calibration_preserved": calibration_preserved,
        "pr_auc_case_met": pr_auc_case,
        "tier_lift_case_met": tier_lift_case,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Step 18: gated single TEST evaluation -- only called if promoted
# ---------------------------------------------------------------------------

def evaluate_v2_on_test(frozen_estimator, feature_columns: list[str], moderate_threshold: float, high_threshold: float, members, ed, care, windows) -> dict:
    window = windows["test"]
    v1_test_df = pd.read_csv(SNAPSHOT_PATHS["test"])
    y_test = v1_test_df[TARGET_COLUMN].astype(int)

    test_frames = build_candidate_feature_frames("test", members, ed, care, windows)
    combined = pd.DataFrame({"member_id": test_frames["v1"]["member_id"]})
    for frame_name, frame in test_frames.items():
        combined = combined.merge(frame, on="member_id", how="left", suffixes=("", f"_{frame_name}"))

    missing = [c for c in feature_columns if c not in combined.columns]
    if missing:
        raise KeyError(f"TEST feature frame missing required V2 columns: {missing}")
    X_test = combined[feature_columns]

    test_prob = frozen_estimator.predict_proba(X_test)[:, 1]
    test_rank_metrics = metrics_mod.rank_metrics(y_test, test_prob)
    test_moderate_confusion = metrics_mod.threshold_confusion_counts(y_test, test_prob, moderate_threshold)
    test_high_confusion = metrics_mod.threshold_confusion_counts(y_test, test_prob, high_threshold)
    test_tier_report = risk_tiers_mod.tier_report(y_test, test_prob, moderate_threshold, high_threshold)
    test_calibration_bins = metrics_mod.calibration_bins(y_test, test_prob, n_bins=10)

    return {
        "test_prob": test_prob, "y_test": y_test, "X_test": X_test,
        "test_rank_metrics": test_rank_metrics,
        "test_moderate_confusion": test_moderate_confusion,
        "test_high_confusion": test_high_confusion,
        "test_tier_report": test_tier_report,
        "test_calibration_bins": test_calibration_bins,
    }


# ---------------------------------------------------------------------------
# Report / artifact writers
# ---------------------------------------------------------------------------

def write_experiment_reports(eval_dir: Path, result: dict):
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["window_experiment"]["rows"]).to_csv(eval_dir / "window_experiment.csv", index=False)
    pd.DataFrame(result["ablation"]["rows"]).to_csv(eval_dir / "feature_ablation_results.csv", index=False)

    diag = result["diagnosis_experiment"]
    pd.DataFrame([
        {"variant": "WITHOUT_DIAGNOSIS", **diag["without"]},
        {"variant": "WITH_DIAGNOSIS", **diag["with"]},
    ]).to_csv(eval_dir / "diagnosis_experiment.csv", index=False)

    pd.DataFrame(result["model_comparison"]["rows"]).to_csv(eval_dir / "model_comparison_v2.csv", index=False)
    pd.DataFrame(result["regularization_rows"]).to_csv(eval_dir / "regularization_experiment.csv", index=False)
    pd.DataFrame(result["calibration_rows"]).to_csv(eval_dir / "calibration_comparison_v2.csv", index=False)
    pd.DataFrame(result["v2_tier_report"]).to_csv(eval_dir / "validation_risk_tiers_v2.csv", index=False)


def write_decision_artifact(eval_dir: Path, decision: dict, frozen_spec: dict | None):
    payload = {"decision": decision, "frozen_spec": frozen_spec}
    (eval_dir / "phase4b_decision.json").write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_hashes_before = {name: sha256_file(p) for name, p in SNAPSHOT_PATHS.items()}
    raw_hashes_before = {name: sha256_file(p) for name, p in RAW_PATHS.items()}

    members, ed, care = load_raw()
    windows = build_all_snapshot_windows()

    v1_train_df = pd.read_csv(SNAPSHOT_PATHS["train"])
    v1_val_df = pd.read_csv(SNAPSHOT_PATHS["validation"])
    y_train = v1_train_df[TARGET_COLUMN].astype(int)
    y_val = v1_val_df[TARGET_COLUMN].astype(int)

    v1_artifact = joblib.load(MODELS_DIR / "uc07_risk_v1_model.joblib")
    v1_val_prob = v1_artifact["pipeline"].predict_proba(v1_val_df[v1_artifact["feature_columns"]])[:, 1]
    v1_val_metrics = metrics_mod.rank_metrics(y_val, v1_val_prob)
    v1_val_tier_report = risk_tiers_mod.tier_report(
        y_val, v1_val_prob, v1_artifact["moderate_threshold"], v1_artifact["high_threshold"]
    )

    print(f"V1 VALIDATION: ROC-AUC={v1_val_metrics['roc_auc']:.4f} PR-AUC={v1_val_metrics['pr_auc']:.4f} Brier={v1_val_metrics['brier_score']:.4f}")

    result = select_v2_candidate_on_validation(members, ed, care, y_train, y_val, windows)

    print(f"Best ablation experiment: {result['best_ablation_experiment']} ({len(result['feature_columns_best'])} features)")
    print(f"V2 winner: {result['v2_winner']['algorithm']} / {result['v2_winner']['calibration_method']} "
          f"PR-AUC={result['v2_winner']['pr_auc']:.4f} Brier={result['v2_winner']['brier_score']:.4f}")

    decision = make_promotion_decision(v1_val_metrics, v1_val_tier_report, result)
    print(f"PROMOTION DECISION: {'PROMOTE V2' if decision['promote'] else 'KEEP V1'}")
    for r in decision["reasons"]:
        print(f"  - {r}")

    write_experiment_reports(EVAL_DIR, result)

    frozen_spec = None
    test_eval = None
    subgroup_rows = None
    feature_importance_rows = None
    artifact_path = None
    metadata_path = None

    if decision["promote"]:
        frozen_spec = {
            "algorithm": result["v2_winner"]["algorithm"],
            "calibration_method": result["v2_winner"]["calibration_method"],
            "feature_list": result["feature_columns_best"],
            "feature_count": len(result["feature_columns_best"]),
            "numeric_features": result["numeric_best"],
            "categorical_features": result["categorical_best"],
            "moderate_threshold": result["v2_moderate_threshold"],
            "high_threshold": result["v2_high_threshold"],
            "model_version_candidate": V2_MODEL_VERSION,
            "validation_metrics_at_freeze": result["v2_validation_metrics"],
            "best_ablation_experiment": result["best_ablation_experiment"],
            "diagnosis_kept": result["diagnosis_kept"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Frozen BEFORE any TEST data is loaded. Must not change based on TEST results.",
        }
        write_decision_artifact(EVAL_DIR, decision, frozen_spec)

        test_eval = evaluate_v2_on_test(
            result["v2_estimator"], result["feature_columns_best"],
            result["v2_moderate_threshold"], result["v2_high_threshold"],
            members, ed, care, windows,
        )
        print(f"V2 TEST: ROC-AUC={test_eval['test_rank_metrics']['roc_auc']:.4f} "
              f"PR-AUC={test_eval['test_rank_metrics']['pr_auc']:.4f} "
              f"Brier={test_eval['test_rank_metrics']['brier_score']:.4f}")

        v1_metadata = json.loads((MODELS_DIR / "uc07_risk_v1_model_metadata.json").read_text())
        pd.DataFrame([
            {"model": "v1", "split": "test", **v1_metadata["final_test_metrics"]},
            {"model": "v2", "split": "test", **test_eval["test_rank_metrics"]},
        ]).to_csv(EVAL_DIR / "test_comparison_v1_v2.csv", index=False)
        pd.DataFrame(test_eval["test_tier_report"]).to_csv(EVAL_DIR / "test_risk_tiers_v2.csv", index=False)

        subgroup_rows = v1_train.run_subgroup_checks(test_eval["X_test"], test_eval["y_test"], test_eval["test_prob"], result["v2_moderate_threshold"])
        pd.DataFrame(subgroup_rows).to_csv(EVAL_DIR / "subgroup_sanity_v2.csv", index=False)

        feature_importance_rows = v1_train.extract_global_feature_importance(result["v2_estimator"])
        pd.DataFrame(feature_importance_rows).to_csv(EVAL_DIR / "global_feature_importance_v2.csv", index=False)

        v2_artifact = {
            "pipeline": result["v2_estimator"],
            "feature_columns": result["feature_columns_best"],
            "numeric_features": result["numeric_best"],
            "categorical_features": result["categorical_best"],
            "target": TARGET_COLUMN,
            "model_version": V2_MODEL_VERSION,
            "parent_model_version": V1_MODEL_VERSION,
            "algorithm": result["v2_winner"]["algorithm"],
            "calibration_method": result["v2_winner"]["calibration_method"],
            "moderate_threshold": result["v2_moderate_threshold"],
            "high_threshold": result["v2_high_threshold"],
        }
        artifact_path = MODELS_DIR / "uc07_risk_v2_model.joblib"
        joblib.dump(v2_artifact, artifact_path)

        metadata = {
            "model_name": "UC07 Avoidable ED Risk Model",
            "model_version": V2_MODEL_VERSION,
            "parent_model_version": V1_MODEL_VERSION,
            "reason_for_new_version": "Phase 4B controlled feature/model improvement demonstrated meaningful, stable VALIDATION improvement over v1.",
            "feature_changes": {
                "best_ablation_experiment": result["best_ablation_experiment"],
                "window_representation": result["window_experiment"]["winner"],
                "diagnosis_features_kept": result["diagnosis_kept"],
                "feature_count_v1": 59, "feature_count_v2": len(result["feature_columns_best"]),
            },
            "target": TARGET_COLUMN,
            "algorithm": result["v2_winner"]["algorithm"],
            "hyperparameters": frozen_spec,
            "calibration_method": result["v2_winner"]["calibration_method"],
            "moderate_threshold": result["v2_moderate_threshold"],
            "high_threshold": result["v2_high_threshold"],
            "feature_list": result["feature_columns_best"],
            "feature_count": len(result["feature_columns_best"]),
            "validation_metrics": result["v2_validation_metrics"],
            "test_metrics": test_eval["test_rank_metrics"],
            "validation_risk_tiers": result["v2_tier_report"],
            "test_risk_tiers": test_eval["test_tier_report"],
            "raw_dataset_sha256": raw_hashes_before,
            "phase3_snapshot_sha256": snapshot_hashes_before,
            "feature_manifest_reference": str(FEATURE_MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "python_version": sys.version, "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__, "numpy_version": np.__version__,
            "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "known_limitations": [
                "Same single ~18-month data era and 3-snapshot design as v1 (Phase 2/3 limitation, unchanged).",
                "Subgroup sanity check is initial only, not the full Phase 6 audit.",
                "Risk tiers are risk-only; Care Management/navigation routing remains Phase 5 scope.",
            ],
        }
        metadata_path = MODELS_DIR / "uc07_risk_v2_model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    else:
        write_decision_artifact(EVAL_DIR, decision, None)

    snapshot_hashes_after = {name: sha256_file(p) for name, p in SNAPSHOT_PATHS.items()}
    raw_hashes_after = {name: sha256_file(p) for name, p in RAW_PATHS.items()}
    if snapshot_hashes_after != snapshot_hashes_before or raw_hashes_after != raw_hashes_before:
        raise SystemExit("CRITICAL: raw or snapshot hashes changed during Phase 4B run.")

    return {
        "v1_val_metrics": v1_val_metrics, "v1_val_tier_report": v1_val_tier_report,
        "result": result, "decision": decision, "frozen_spec": frozen_spec,
        "test_eval": test_eval, "subgroup_rows": subgroup_rows,
        "feature_importance_rows": feature_importance_rows,
        "artifact_path": artifact_path, "metadata_path": metadata_path,
        "snapshot_hashes_before": snapshot_hashes_before, "snapshot_hashes_after": snapshot_hashes_after,
        "raw_hashes_before": raw_hashes_before, "raw_hashes_after": raw_hashes_after,
    }


if __name__ == "__main__":
    out = main()
    print("\n=== Phase 4B run complete ===")
    print(f"Decision: {'PROMOTE V2' if out['decision']['promote'] else 'KEEP V1'}")
