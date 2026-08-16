"""
predict.py
----------
Loads the saved model and runs predictions on a feature DataFrame.
Returns a result DataFrame with member_id, risk_probability, risk_score,
risk_category.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# Default path (same directory as this file)
MODEL_PATH = Path(__file__).parent / "ed_risk_model.pkl"


def _assign_risk_category(prob: float) -> str:
    """Map a probability to a Low / Medium / High category."""
    if prob >= 0.60:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    else:
        return "Low"


def _build_alternative_care_recommendation(row: pd.Series) -> tuple[str, str]:
    """Recommend the most practical lower-acuity care option for a patient."""
    telehealth_available = int(row.get("telehealth_available", 0) or 0)
    transportation_barrier = int(row.get("transportation_barrier", 0) or 0)
    pcp_distance = float(row.get("pcp_distance_miles", 99) or 99)
    urgent_distance = float(row.get("urgent_care_distance_miles", 99) or 99)

    if telehealth_available and (transportation_barrier == 1 or pcp_distance > 10 or urgent_distance > 10):
        return (
            "Telehealth",
            "Telehealth is available and removes travel barriers for same-day follow-up and triage.",
        )
    if pcp_distance <= 5 and transportation_barrier == 0:
        return (
            "PCP",
            "Primary care is close and accessible, making it the best lower-acuity option before the ER.",
        )
    if urgent_distance <= 5:
        return (
            "Urgent Care",
            "Urgent care is geographically closer and appropriate for non-emergency needs instead of the ER.",
        )
    if pcp_distance <= 10:
        return (
            "PCP",
            "PCP is within a reasonable distance and better matches chronic-care management needs.",
        )
    if telehealth_available:
        return (
            "Telehealth",
            "Telehealth is the lowest-friction option when travel distance and access barriers are limiting.",
        )
    return (
        "Urgent Care",
        "No telehealth option is available and urgent care is the closest practical alternative to the ER.",
    )


def _compute_shap_explanations(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Return per-patient SHAP explanations and a compact summary for the dashboard."""
    if X.empty:
        return pd.DataFrame(
            columns=[
                "shap_feature_1",
                "shap_value_1",
                "shap_feature_2",
                "shap_value_2",
                "shap_feature_3",
                "shap_value_3",
                "shap_explanation_summary",
                "shap_explanation_json",
                "top_positive_factors",
                "top_negative_factors",
            ]
        )

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    rows = []
    for i in range(len(X)):
        row_values = shap_values[i]
        top_idx = np.argsort(np.abs(row_values))[::-1][:3]

        top_features = []
        top_scores = []
        feature_details = []

        for idx in top_idx:
            feat = feature_names[idx]
            score = float(row_values[idx])
            top_features.append(feat)
            top_scores.append(round(score, 4))
            direction = "increases" if score > 0 else "reduces"
            feature_details.append({
                "feature": feat,
                "contribution": round(score, 4),
                "impact": direction,
            })

        summary = "; ".join(
            [f"{feat} {('increases' if score > 0 else 'reduces')} risk ({score:.4f})" for feat, score in zip(top_features, top_scores)]
        ) if top_features else "No strong SHAP drivers were detected."

        pos_list = []
        neg_list = []
        for idx in range(len(feature_names)):
            feat = feature_names[idx]
            val = float(row_values[idx])
            if val > 0.0001:
                pos_list.append({"feature": feat, "shap_value": round(val, 4)})
            elif val < -0.0001:
                neg_list.append({"feature": feat, "shap_value": round(val, 4)})

        pos_list = sorted(pos_list, key=lambda x: abs(x["shap_value"]), reverse=True)[:3]
        neg_list = sorted(neg_list, key=lambda x: abs(x["shap_value"]), reverse=True)[:3]

        rows.append(
            {
                "shap_feature_1": top_features[0] if len(top_features) > 0 else "",
                "shap_value_1": round(float(top_scores[0]), 4) if len(top_scores) > 0 else 0.0,
                "shap_feature_2": top_features[1] if len(top_features) > 1 else "",
                "shap_value_2": round(float(top_scores[1]), 4) if len(top_scores) > 1 else 0.0,
                "shap_feature_3": top_features[2] if len(top_features) > 2 else "",
                "shap_value_3": round(float(top_scores[2]), 4) if len(top_scores) > 2 else 0.0,
                "shap_explanation_summary": summary,
                "shap_explanation_json": json.dumps(feature_details),
                "top_positive_factors": json.dumps(pos_list),
                "top_negative_factors": json.dumps(neg_list),
            }
        )

    return pd.DataFrame(rows)


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the joblib model package from disk."""
    pkg = joblib.load(model_path)
    # Support both plain pipeline and the dict wrapper used in the docx
    if isinstance(pkg, dict):
        return pkg["model"]
    return pkg


def explain_member(
    member_id: str,
    X: pd.DataFrame,
    member_ids: list[str],
    model_path: str | Path = MODEL_PATH,
) -> dict:
    """
    Compute the SHAP explanation for a single member on demand. TreeExplainer
    over the full RandomForest costs ~70ms/row (measured on this model), so
    batch-computing it for every row in a large upload is what made bulk
    prediction take minutes; scoped to one row it's near-instant.
    """
    try:
        idx = member_ids.index(member_id)
    except ValueError as exc:
        raise KeyError(f"member_id '{member_id}' not found in this dataset") from exc

    pipeline = load_model(model_path)
    shap_row = _compute_shap_explanations(pipeline, X.iloc[[idx]])
    return shap_row.iloc[0].to_dict()


def predict(
    X: pd.DataFrame,
    member_ids: list[str],
    members_raw: pd.DataFrame,
    ed_visits_raw: pd.DataFrame | None = None,
    care_raw: pd.DataFrame | None = None,
    model_path: str | Path = MODEL_PATH,
    include_shap: bool = True,
) -> pd.DataFrame:
    """
    Run inference and build the output DataFrame in the same shape as the
    unseen patient risk results file.

    include_shap=False skips the expensive per-row SHAP explanation pass
    (the dominant cost on large uploads) — use it for bulk/interactive
    scoring and fetch per-patient explanations on demand via explain_member().
    """
    pipeline = load_model(model_path)

    # ----- Predict -----
    proba = pipeline.predict_proba(X)[:, 1]

    # Risk outputs
    risk_score = np.clip(np.round(proba * 100, 2), 1, 100)
    risk_category = [_assign_risk_category(p) for p in proba]
    predicted_frequent_ed = (proba >= 0.50).astype(int)

    # Build the base output with the final file layout
    result = pd.DataFrame(
        {
            "member_id": member_ids,
            "risk_probability": np.round(proba, 4),
            "risk_score": risk_score,
            "predicted_frequent_ED": predicted_frequent_ed,
            "risk_category": risk_category,
        }
    )

    demo_cols = [
        "member_id", "age", "gender", "diabetes", "copd", "hypertension",
        "chf", "asthma", "ckd", "transportation_barrier",
        "telehealth_available", "pcp_distance_miles", "urgent_care_distance_miles",
    ]
    demo = members_raw[demo_cols].drop_duplicates("member_id")
    result = result.merge(demo, on="member_id", how="left")

    if ed_visits_raw is not None and not ed_visits_raw.empty:
        ed = ed_visits_raw.copy()
        ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="coerce")
        reference_date = ed["visit_date"].max()
        last_ed = (
            ed.groupby("member_id")["visit_date"].max().reset_index(name="last_ED_date")
        )
        last_ed["days_since_last_ED"] = (reference_date - last_ed["last_ED_date"]).dt.days
        result = result.merge(last_ed[["member_id", "days_since_last_ED"]], on="member_id", how="left")
    else:
        result["days_since_last_ED"] = None

    if care_raw is not None and not care_raw.empty:
        care = care_raw.copy()
        care["visit_date"] = pd.to_datetime(care["visit_date"], errors="coerce")
        reference_date = care["visit_date"].max()
        last_care = (
            care.groupby("member_id")["visit_date"].max().reset_index(name="last_care_date")
        )
        last_care["days_since_last_care"] = (reference_date - last_care["last_care_date"]).dt.days

        care_counts = pd.crosstab(care["member_id"], care["care_type"]).reset_index()
        care_counts.columns = ["member_id"] + [
            "care_" + str(col).replace(" ", "_") for col in care_counts.columns[1:]
        ]
        for col in ["care_PCP", "care_Urgent_Care", "care_Telehealth", "care_Care_Management"]:
            if col not in care_counts.columns:
                care_counts[col] = 0

        care_counts["alternative_care_visits"] = (
            care_counts.get("care_Urgent_Care", 0) + care_counts.get("care_Telehealth", 0)
        )
        care_counts["total_non_ED_care"] = (
            care_counts.get("care_PCP", 0)
            + care_counts.get("care_Urgent_Care", 0)
            + care_counts.get("care_Telehealth", 0)
            + care_counts.get("care_Care_Management", 0)
        )

        result = result.merge(
            care_counts[["member_id", "alternative_care_visits", "total_non_ED_care"]],
            on="member_id",
            how="left",
        )
        result = result.merge(last_care[["member_id", "days_since_last_care"]], on="member_id", how="left")
    else:
        result["days_since_last_care"] = None
        result["alternative_care_visits"] = 0
        result["total_non_ED_care"] = 0

    result["access_burden"] = (
        result["transportation_barrier"].fillna(0)
        + (result["pcp_distance_miles"].fillna(0) > 10).astype(int)
        + (result["urgent_care_distance_miles"].fillna(0) > 10).astype(int)
    )
    clinical_cols = ["diabetes", "copd", "hypertension", "chf", "asthma", "ckd"]
    result["clinical_burden"] = result[clinical_cols].fillna(0).sum(axis=1)

    # Calculate safety guardrails
    safety_members = set()
    if ed_visits_raw is not None and not ed_visits_raw.empty:
        # High acuity indicators: red_flag, icu, major_procedure, admitted, or triage_level <= 2
        high_acuity = ed_visits_raw[
            (ed_visits_raw["red_flag"] == 1) |
            (ed_visits_raw["icu"] == 1) |
            (ed_visits_raw["major_procedure"] == 1) |
            (ed_visits_raw["admitted"] == 1) |
            (ed_visits_raw["triage_level"].isin([1, 2]))
        ]
        safety_members = set(high_acuity["member_id"].unique())

    result["safety_guardrail_flag"] = result["member_id"].apply(lambda mid: mid in safety_members)
    result["safety_guardrail_message"] = result["safety_guardrail_flag"].apply(
        lambda flag: "Clinical/Emergency care indicators present — DO NOT discourage ED. This patient's risk score reflects utilization pattern only, not a recommendation to avoid the ED." if flag else None
    )

    # Calculate navigation opportunities
    def _build_navigation_opportunities(row):
        if row.get("safety_guardrail_flag", False):
            return json.dumps([])

        opps = []
        if int(row.get("telehealth_available", 0) or 0) == 1:
            opps.append("Telehealth access opportunity")
        if int(row.get("transportation_barrier", 0) or 0) == 1:
            opps.append("Transportation-support resource opportunity")

        pcp_dist = float(row.get("pcp_distance_miles", 0) or 0)
        urgent_dist = float(row.get("urgent_care_distance_miles", 0) or 0)
        if pcp_dist > 5:
            opps.append("PCP follow-up/navigation opportunity")
        if urgent_dist > 5:
            opps.append("Urgent Care navigation opportunity")
        return json.dumps(opps)

    result["navigation_opportunities"] = result.apply(_build_navigation_opportunities, axis=1)

    columns = [
        "member_id",
        "risk_probability",
        "risk_score",
        "predicted_frequent_ED",
        "risk_category",
        "age",
        "gender",
        "diabetes",
        "copd",
        "hypertension",
        "chf",
        "asthma",
        "ckd",
        "transportation_barrier",
        "telehealth_available",
        "pcp_distance_miles",
        "urgent_care_distance_miles",
        "days_since_last_ED",
        "days_since_last_care",
        "alternative_care_visits",
        "total_non_ED_care",
        "access_burden",
        "clinical_burden",
        "recommended_alternative_care",
        "alternative_care_reason",
        "safety_guardrail_flag",
        "safety_guardrail_message",
        "navigation_opportunities",
    ]

    if include_shap:
        shap_cols = _compute_shap_explanations(pipeline, X)
        result = pd.concat([result.reset_index(drop=True), shap_cols.reset_index(drop=True)], axis=1)
        columns += [
            "shap_feature_1",
            "shap_value_1",
            "shap_feature_2",
            "shap_value_2",
            "shap_feature_3",
            "shap_value_3",
            "shap_explanation_summary",
            "shap_explanation_json",
            "top_positive_factors",
            "top_negative_factors",
        ]
    else:
        result["shap_feature_1"] = ""
        result["shap_value_1"] = 0.0
        result["shap_feature_2"] = ""
        result["shap_value_2"] = 0.0
        result["shap_feature_3"] = ""
        result["shap_value_3"] = 0.0
        result["shap_explanation_summary"] = "SHAP explanations not requested."
        result["shap_explanation_json"] = "[]"
        result["top_positive_factors"] = "[]"
        result["top_negative_factors"] = "[]"
        columns += [
            "shap_feature_1",
            "shap_value_1",
            "shap_feature_2",
            "shap_value_2",
            "shap_feature_3",
            "shap_value_3",
            "shap_explanation_summary",
            "shap_explanation_json",
            "top_positive_factors",
            "top_negative_factors",
        ]

    result[["recommended_alternative_care", "alternative_care_reason"]] = result.apply(
        lambda row: pd.Series(_build_alternative_care_recommendation(row)), axis=1
    )

    return result[columns]
