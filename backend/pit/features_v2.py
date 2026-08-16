"""
features_v2.py
---------------
Phase 4B candidate feature engineering. ADDITIVE ONLY: this module never
imports from or modifies `features.py` (the frozen Phase 3 v1 feature
module), `windows.py`'s window definitions, `target.py`, or the three
approved index dates. It reuses the same `windows.py` boundary functions
and `encounter_classification.py` classifier that `features.py` already
uses, so every feature here inherits the identical point-in-time
guarantee: computed strictly from
`observation_start <= event_date < index_date`. Nothing from the outcome
window (`event_date >= index_date`) ever reaches this module's output.

Two families of candidate features are produced:

1. `build_banded_ed_features()` -- an ALTERNATIVE representation of ED
   utilization to the existing nested-cumulative windows (30/90/180/270d):
   non-overlapping bands (0-30d, 31-90d, 91-180d, 181-270d). Evaluated
   against the nested representation in the Phase 4B window experiment
   (docs/04B_MODEL_IMPROVEMENT.md section 7) -- NOT assumed superior.

2. `build_extended_candidate_features()` -- new engineered groups
   (velocity, care-setting mix, continuity/engagement, access x
   utilization interactions, historical-ED-pattern extras, and a
   controlled historical-diagnosis experiment) evaluated via ablation
   (section 13) before any of them are accepted into a v2 candidate
   feature set.

Every feature here belongs to one of the Phase 4B feature groups (A-K)
documented in docs/04B_MODEL_IMPROVEMENT.md section 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from encounter_classification import (
    POTENTIALLY_AVOIDABLE,
    classify_ed_encounters,
)
from windows import SnapshotWindow, in_observation_window

BAND_EDGES_DAYS = [(0, 30), (31, 90), (91, 180), (181, 270)]
ALTERNATIVE_CARE_TYPES = {
    "PCP": "pcp",
    "Urgent Care": "urgent_care",
    "Telehealth": "telehealth",
    "Care Management": "care_management",
}

RECENT_FOLLOWUP_WINDOW_DAYS = 30  # Step 5's "recent_outpatient_followup_after_prior_ed"


# ---------------------------------------------------------------------------
# Group D-alt: non-overlapping band ED utilization (Step 3)
# ---------------------------------------------------------------------------

def build_banded_ed_features(ed: pd.DataFrame, member_order: pd.Series, window: SnapshotWindow) -> pd.DataFrame:
    """Non-overlapping-band alternative to prior_ed_count_{30,90,180,270}d
    and prior_potentially_avoidable_ed_count_{30,90,180,270}d. Each band
    counts events with `index_date - hi_days <= visit_date < index_date - lo_days + 1day`
    i.e. days-before-index in [lo, hi], disjoint across bands by
    construction (so, unlike the nested windows, these 4 values always
    sum exactly to the 270d total)."""
    ed = ed.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="raise")
    prior = ed.loc[in_observation_window(ed["visit_date"], window)].copy()
    prior["encounter_state"] = classify_ed_encounters(prior)
    prior["days_before_index"] = (window.index_date - prior["visit_date"]).dt.days

    out = pd.DataFrame({"member_id": member_order})

    for lo, hi in BAND_EDGES_DAYS:
        band_mask_all = (prior["days_before_index"] >= lo) & (prior["days_before_index"] <= hi)
        counts_all = prior.loc[band_mask_all].groupby("member_id").size().reindex(member_order).fillna(0).astype(int)
        out[f"ed_count_band_{lo}_{hi}d"] = counts_all.values

        band_mask_avoid = band_mask_all & (prior["encounter_state"] == POTENTIALLY_AVOIDABLE)
        counts_avoid = prior.loc[band_mask_avoid].groupby("member_id").size().reindex(member_order).fillna(0).astype(int)
        out[f"potentially_avoidable_ed_count_band_{lo}_{hi}d"] = counts_avoid.values

    return out


def build_reduced_window_ed_features(ed: pd.DataFrame, member_order: pd.Series, window: SnapshotWindow) -> pd.DataFrame:
    """Reduced nested-window alternative: drops the 180d window for total
    and potentially-avoidable ED counts (Step 1 diagnostics found 180d-270d
    the single most redundant nested pair, r=0.81, vs. 30d-270d at r=0.33
    -- i.e. 180d carries the least independent information of the four
    windows, not 30d as might be naively assumed). Keeps 30d/90d/270d."""
    ed = ed.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="raise")
    prior = ed.loc[in_observation_window(ed["visit_date"], window)].copy()
    prior["encounter_state"] = classify_ed_encounters(prior)

    out = pd.DataFrame({"member_id": member_order})
    for days in (30, 90, 270):
        cutoff = window.index_date - pd.Timedelta(days=days)
        subset = prior.loc[prior["visit_date"] >= cutoff]
        counts = subset.groupby("member_id").size().reindex(member_order).fillna(0).astype(int)
        out[f"prior_ed_count_{days}d"] = counts.values

        avoid_subset = subset.loc[subset["encounter_state"] == POTENTIALLY_AVOIDABLE]
        avoid_counts = avoid_subset.groupby("member_id").size().reindex(member_order).fillna(0).astype(int)
        out[f"prior_potentially_avoidable_ed_count_{days}d"] = avoid_counts.values

    return out


# ---------------------------------------------------------------------------
# Extended candidate features: velocity, care-mix, continuity, interactions,
# historical-ED-pattern extras, controlled diagnosis (Steps 4-9)
# ---------------------------------------------------------------------------

def build_extended_candidate_features(
    members: pd.DataFrame,
    ed: pd.DataFrame,
    care: pd.DataFrame,
    v1_features: pd.DataFrame,
    window: SnapshotWindow,
    include_diagnosis: bool = True,
) -> pd.DataFrame:
    """
    v1_features: the already-built, frozen v1 feature frame for this
    snapshot (loaded from the frozen train/validation/test_snapshot.csv),
    used here ONLY to read already-point-in-time-safe v1 columns
    (prior_*_count_*, days_since_prior_*, has_prior_*) as building blocks
    for ratios/interactions -- no new raw-data access is needed for those
    parts, avoiding recomputing what's already correct and frozen.
    """
    member_order = members["member_id"].drop_duplicates().reset_index(drop=True)
    v1 = v1_features.set_index("member_id").reindex(member_order).reset_index()

    out = pd.DataFrame({"member_id": member_order})

    # ---- Group I: utilization velocity / trend (Step 4) ----
    # "recent 30d activity vs. the remaining 240 days of the 270d observation
    # window" -- denominator is the REMAINDER (270d total minus the 30d
    # count already inside it), floored at 1 to avoid unstable division by
    # a near-zero remainder.
    out["ed_acceleration_30_vs_240"] = v1["prior_ed_count_30d"] / (v1["prior_ed_count_270d"] - v1["prior_ed_count_30d"]).clip(lower=1)
    out["potentially_avoidable_ed_acceleration_30_vs_240"] = (
        v1["prior_potentially_avoidable_ed_count_30d"]
        / (v1["prior_potentially_avoidable_ed_count_270d"] - v1["prior_potentially_avoidable_ed_count_30d"]).clip(lower=1)
    )
    total_alt_care_90d = v1["prior_pcp_count_90d"] + v1["prior_urgent_care_count_90d"] + v1["prior_telehealth_count_90d"] + v1["prior_care_management_count_90d"]
    total_alt_care_270d = v1["prior_pcp_count_270d"] + v1["prior_urgent_care_count_270d"] + v1["prior_telehealth_count_270d"] + v1["prior_care_management_count_270d"]
    out["alternative_care_engagement_trend_90_vs_270"] = total_alt_care_90d / (total_alt_care_270d - total_alt_care_90d).clip(lower=1)

    # ---- Group G: care-setting mix (Step 5) ----
    out["total_outpatient_alternative_visits_270d"] = total_alt_care_270d
    out["ed_to_outpatient_ratio_270d"] = v1["prior_ed_count_270d"] / total_alt_care_270d.clip(lower=1)
    out["ed_share_of_total_utilization_270d"] = v1["prior_ed_count_270d"] / (v1["prior_ed_count_270d"] + total_alt_care_270d).clip(lower=1)
    out["telehealth_share_270d"] = v1["prior_telehealth_count_270d"] / total_alt_care_270d.clip(lower=1)
    out["urgent_care_share_270d"] = v1["prior_urgent_care_count_270d"] / total_alt_care_270d.clip(lower=1)
    out["pcp_share_270d"] = v1["prior_pcp_count_270d"] / total_alt_care_270d.clip(lower=1)

    # recent_outpatient_followup_after_prior_ed: did the member have ANY
    # outpatient (PCP/UC/Telehealth/CM) visit within
    # RECENT_FOLLOWUP_WINDOW_DAYS after their most recent prior ED visit,
    # strictly before index_date? Both the ED visit and the follow-up
    # visit are drawn only from the observation window (event_date < index_date).
    followup_flag, followup_days = _recent_outpatient_followup(ed, care, member_order, window)
    out["has_recent_outpatient_followup_after_last_ed"] = followup_flag
    out["days_from_last_ed_to_next_outpatient"] = followup_days

    # ---- Group J: care continuity / engagement (Step 6) ----
    has_any = [v1["has_prior_pcp"], v1["has_prior_urgent_care"], v1["has_prior_telehealth"], v1["has_prior_care_management"]]
    out["care_setting_diversity_270d"] = sum(h.astype(int) for h in has_any)

    days_since_cols = ["days_since_prior_pcp", "days_since_prior_urgent_care", "days_since_prior_telehealth", "days_since_prior_care_management"]
    days_since_any_outpatient = v1[days_since_cols].min(axis=1, skipna=True)
    out["days_since_any_outpatient_contact"] = days_since_any_outpatient
    out["long_gap_without_outpatient_care_flag"] = (days_since_any_outpatient.isna() | (days_since_any_outpatient > 180)).astype(int)

    care_count_30d_cols = ["prior_pcp_count_30d", "prior_urgent_care_count_30d", "prior_telehealth_count_30d", "prior_care_management_count_30d"]
    out["recent_outpatient_contact_30d_flag"] = (v1[care_count_30d_cols].sum(axis=1) > 0).astype(int)

    out["repeated_ED_without_recent_PCP_flag"] = ((v1["prior_ed_count_270d"] >= 2) & (v1["prior_pcp_count_270d"] == 0)).astype(int)

    # ---- Group C x D: access x utilization interactions (Step 7) ----
    # Rationale for each: an access barrier or long distance combined with
    # RECENT ED use is a more specific "access-driven utilization" pattern
    # than either variable alone; chronic burden combined with access
    # barriers or recent ED use similarly represents a compounding-risk
    # pattern business logic supports investigating.
    out["transportation_barrier_x_recent_ed"] = v1["transportation_barrier"] * v1["prior_ed_count_30d"]
    out["pcp_distance_x_recent_ed"] = v1["pcp_distance_miles"] * v1["prior_ed_count_30d"]
    out["urgent_care_distance_x_recent_ed"] = v1["urgent_care_distance_miles"] * v1["prior_ed_count_30d"]
    out["telehealth_available_x_recent_ed"] = v1["telehealth_available"] * v1["prior_ed_count_30d"]
    out["chronic_burden_x_access_barrier"] = v1["clinical_burden"] * v1["access_burden"]
    out["chronic_burden_x_recent_ed"] = v1["clinical_burden"] * v1["prior_ed_count_30d"]

    # ---- Group E: historical ED pattern extras (Step 8) ----
    out["avoidable_share_of_prior_ed_270d"] = v1["prior_potentially_avoidable_ed_count_270d"] / v1["prior_ed_count_270d"].clip(lower=1)
    out["repeat_potentially_avoidable_ed_flag"] = (v1["prior_potentially_avoidable_ed_count_270d"] >= 2).astype(int)

    # ---- Group K: controlled historical diagnosis (Step 9) ----
    if include_diagnosis:
        diag_features = _build_controlled_diagnosis_features(ed, member_order, window, v1["prior_ed_count_270d"])
        out = out.merge(diag_features, on="member_id", how="left")

    return out


def _recent_outpatient_followup(ed: pd.DataFrame, care: pd.DataFrame, member_order: pd.Series, window: SnapshotWindow) -> tuple[np.ndarray, np.ndarray]:
    ed = ed.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="raise")
    prior_ed = ed.loc[in_observation_window(ed["visit_date"], window)]
    last_ed_by_member = prior_ed.groupby("member_id")["visit_date"].max()

    care = care.copy()
    care["visit_date"] = pd.to_datetime(care["visit_date"], errors="raise")
    prior_care = care.loc[in_observation_window(care["visit_date"], window)]

    flags = np.zeros(len(member_order), dtype=int)
    days = np.full(len(member_order), np.nan)

    care_by_member = {mid: grp["visit_date"].sort_values() for mid, grp in prior_care.groupby("member_id")}

    for i, member_id in enumerate(member_order):
        last_ed_date = last_ed_by_member.get(member_id)
        if last_ed_date is None:
            continue
        care_dates = care_by_member.get(member_id)
        if care_dates is None or care_dates.empty:
            continue
        # follow-up visits: strictly after the last ED visit, strictly
        # before index_date (already guaranteed by in_observation_window
        # having filtered prior_care to < index_date).
        followups = care_dates[care_dates > last_ed_date]
        if followups.empty:
            continue
        earliest_followup = followups.min()
        gap_days = (earliest_followup - last_ed_date).days
        days[i] = gap_days
        if gap_days <= RECENT_FOLLOWUP_WINDOW_DAYS:
            flags[i] = 1

    return flags, days


def _build_controlled_diagnosis_features(ed: pd.DataFrame, member_order: pd.Series, window: SnapshotWindow, prior_ed_count_270d: pd.Series) -> pd.DataFrame:
    """Compressed, volume-normalized diagnosis representations -- NOT a
    per-category crosstab (that was the Phase 1 leakage/reconstruction
    pattern). Uses ONLY observation-window ED encounters. Every count-like
    quantity here is normalized by the member's own prior_ed_count_270d
    (floored at 1) specifically so it captures diagnosis SPREAD/
    CONCENTRATION rather than merely re-encoding ED volume."""
    ed = ed.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="raise")
    prior = ed.loc[in_observation_window(ed["visit_date"], window)]

    out = pd.DataFrame({"member_id": member_order})

    distinct_categories = prior.groupby("member_id")["diagnosis"].nunique().reindex(member_order).fillna(0).astype(int)
    out["distinct_prior_ed_diagnosis_categories_270d"] = distinct_categories.values

    # Vectorized "most frequent diagnosis category count per member" --
    # deliberately avoids DataFrameGroupBy.apply(custom_fn): on an EMPTY
    # `prior` frame, pandas calls the function once on a dummy probe slice
    # to infer the output shape, which can silently produce a malformed
    # (non-scalar) result. groupby(["member_id","diagnosis"]).size() then
    # taking the per-member max is both safe on empty input and faster.
    if prior.empty:
        most_common_count = pd.Series(0, index=member_order)
    else:
        per_member_diag_counts = prior.groupby(["member_id", "diagnosis"], observed=True).size()
        most_common_count = per_member_diag_counts.groupby(level="member_id").max().reindex(member_order).fillna(0).astype(int)

    denom = prior_ed_count_270d.clip(lower=1).reset_index(drop=True)
    out["most_common_prior_diagnosis_share_270d"] = most_common_count.values / denom.values
    out["prior_diagnosis_diversity_ratio_270d"] = distinct_categories.values / denom.values
    out["repeat_same_diagnosis_flag_270d"] = (most_common_count.values >= 2).astype(int)

    return out
