"""
feature_engineering.py
-----------------------
Exact replication of the AVOIDABLE.docx ML pipeline feature extraction steps.
Accepts three DataFrames (members, ed_visits, care_history) and returns a
feature DataFrame ready for model inference (member_id is preserved separately).
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Expected care types – ensures columns always exist even if absent in data
# ---------------------------------------------------------------------------
EXPECTED_CARE = [
    "care_PCP",
    "care_Urgent_Care",
    "care_Telehealth",
    "care_Care_Management",
]

# ---------------------------------------------------------------------------
# Columns dropped before model prediction (leakage + identifiers)
# ---------------------------------------------------------------------------
DROP_BEFORE_MODEL = [
    "frequent_ED_user",       # target (won't be present in inference data)
    "member_id",
    "last_ED_date",
    "last_care_date",
    # Target-leakage features
    "ED_visits_365d",
    "ED_visits_180d",
    "ED_visits_90d",
    "ED_visits_30d",
    "ED_to_alternative_ratio",
    # Direct ED outcome vars (dropped in docx step 22)
    "ED_admissions",
    "ED_ICU_visits",
    "ED_major_procedures",
    "ED_red_flags",
    "ED_total_cost",
    "ED_avg_cost",
]


def extract_features(
    members: pd.DataFrame,
    ed: pd.DataFrame,
    care: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the complete feature engineering pipeline.

    Parameters
    ----------
    members  : DataFrame from raw_members.csv  (or a similar unseen file)
    ed       : DataFrame from raw_ed_visits.csv
    care     : DataFrame from raw_care_history.csv

    Returns
    -------
    X            : Feature DataFrame (no member_id, no leakage columns)
    member_ids   : List of member_id strings in the same row order as X
    """

    # ------------------------------------------------------------------
    # Step 1 – Parse dates
    # ------------------------------------------------------------------
    ed = ed.copy()
    care = care.copy()

    ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="coerce")
    care["visit_date"] = pd.to_datetime(care["visit_date"], errors="coerce")

    # ------------------------------------------------------------------
    # Step 2 – Reference date (max ED visit date)
    # ------------------------------------------------------------------
    reference_date = ed["visit_date"].max()

    # ------------------------------------------------------------------
    # Step 3 – Days since reference for each ED visit
    # ------------------------------------------------------------------
    ed["days_since_reference"] = (reference_date - ed["visit_date"]).dt.days

    # ------------------------------------------------------------------
    # Steps 4-7 – ED visit counts at 4 time windows
    # ------------------------------------------------------------------
    def _ed_window(days: int, col_name: str) -> pd.DataFrame:
        return (
            ed[
                (ed["days_since_reference"] >= 0)
                & (ed["days_since_reference"] <= days)
            ]
            .groupby("member_id")
            .size()
            .reset_index(name=col_name)
        )

    ed_365 = _ed_window(365, "ED_visits_365d")
    ed_180 = _ed_window(180, "ED_visits_180d")
    ed_90  = _ed_window(90,  "ED_visits_90d")
    ed_30  = _ed_window(30,  "ED_visits_30d")

    # ------------------------------------------------------------------
    # Step 8 – Last ED visit / days since
    # ------------------------------------------------------------------
    last_ed = (
        ed.groupby("member_id")["visit_date"]
        .max()
        .reset_index(name="last_ED_date")
    )
    last_ed["days_since_last_ED"] = (
        reference_date - last_ed["last_ED_date"]
    ).dt.days

    # ------------------------------------------------------------------
    # Step 9 – Diagnosis crosstab features
    # ------------------------------------------------------------------
    diagnosis_features = pd.crosstab(ed["member_id"], ed["diagnosis"])
    diagnosis_features.columns = [
        "diagnosis_"
        + str(col).replace(" ", "_").replace("-", "_").replace("/", "_")
        for col in diagnosis_features.columns
    ]
    diagnosis_features = diagnosis_features.reset_index()

    # ------------------------------------------------------------------
    # Step 10 – ED severity features
    # ------------------------------------------------------------------
    ed_summary = (
        ed.groupby("member_id")
        .agg(
            ED_admissions=("admitted", "sum"),
            ED_ICU_visits=("icu", "sum"),
            ED_major_procedures=("major_procedure", "sum"),
            ED_red_flags=("red_flag", "sum"),
            ED_total_cost=("cost", "sum"),
            ED_avg_cost=("cost", "mean"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Step 11 – Care history crosstab
    # ------------------------------------------------------------------
    care_counts = pd.crosstab(care["member_id"], care["care_type"])
    care_counts.columns = [
        "care_" + str(col).replace(" ", "_") for col in care_counts.columns
    ]
    care_counts = care_counts.reset_index()

    # Ensure all expected care columns exist
    for col in EXPECTED_CARE:
        if col not in care_counts.columns:
            care_counts[col] = 0

    # ------------------------------------------------------------------
    # Step 12 – Last care visit / days since
    # ------------------------------------------------------------------
    last_care = (
        care.groupby("member_id")["visit_date"]
        .max()
        .reset_index(name="last_care_date")
    )
    last_care["days_since_last_care"] = (
        reference_date - last_care["last_care_date"]
    ).dt.days

    # ------------------------------------------------------------------
    # Step 13 – Merge all tables onto members
    # ------------------------------------------------------------------
    features = members.copy()
    for table in [
        ed_365, ed_180, ed_90, ed_30,
        last_ed, diagnosis_features, ed_summary,
        care_counts, last_care,
    ]:
        features = features.merge(table, on="member_id", how="left")

    # ------------------------------------------------------------------
    # Step 14 – Fill missing utilisation features with 0
    # ------------------------------------------------------------------
    for col in features.columns:
        if (
            col.startswith("ED_")
            or col.startswith("care_")
            or col.startswith("diagnosis_")
        ):
            features[col] = features[col].fillna(0)

    # ------------------------------------------------------------------
    # Step 15 – Fill distance NaNs with median
    # ------------------------------------------------------------------
    for col in ["pcp_distance_miles", "urgent_care_distance_miles"]:
        if col in features.columns:
            features[col] = features[col].fillna(features[col].median())

    # ------------------------------------------------------------------
    # Step 16 – Derived features
    # ------------------------------------------------------------------
    features["alternative_care_visits"] = (
        features["care_Urgent_Care"] + features["care_Telehealth"]
    )

    features["total_non_ED_care"] = (
        features["care_PCP"]
        + features["care_Urgent_Care"]
        + features["care_Telehealth"]
        + features["care_Care_Management"]
    )

    features["ED_to_alternative_ratio"] = features["ED_visits_365d"] / (
        features["alternative_care_visits"] + 1
    )

    features["access_burden"] = (
        features["transportation_barrier"]
        + (features["pcp_distance_miles"] > 10).astype(int)
        + (features["urgent_care_distance_miles"] > 10).astype(int)
    )

    clinical_columns = ["diabetes", "copd", "hypertension", "chf", "asthma", "ckd"]
    existing_clinical = [c for c in clinical_columns if c in features.columns]
    features["clinical_burden"] = features[existing_clinical].sum(axis=1)

    # ------------------------------------------------------------------
    # Step 17 – Preserve member_ids then drop non-feature columns
    # ------------------------------------------------------------------
    member_ids = features["member_id"].tolist()

    drop_cols = [c for c in DROP_BEFORE_MODEL if c in features.columns]
    X = features.drop(columns=drop_cols, errors="ignore")

    return X, member_ids
