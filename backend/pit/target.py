"""
target.py
---------
Point-in-time member-level target construction for UC07 (Phase 3).

    future_potentially_avoidable_ed_90d = 1  if the member has >= 1 ED
        encounter classified POTENTIALLY_AVOIDABLE inside the snapshot's
        90-day OUTCOME window, else 0.

UNCERTAIN and PROTECTED_OR_HIGH_ACUITY outcome-window encounters never
create a positive label on their own (docs/02_UC07_AND_DATA_DESIGN.md
section 6.2; docs/DECISION_LOG.md item 12). A member may have protected
and/or uncertain encounters in the outcome window and still be labeled 0.

This module intentionally does NOT reuse, import, or reference the legacy
`frequent_ED_user` (ED_visits_365d >= 2) target from backend/train_model.py.
Only ED encounters strictly inside the outcome window
(index_date <= visit_date < outcome_end) ever contribute to the label --
no observation-window (past) encounter can influence it.
"""
from __future__ import annotations

import pandas as pd

from encounter_classification import POTENTIALLY_AVOIDABLE, classify_ed_encounters
from windows import SnapshotWindow, in_outcome_window

TARGET_COLUMN = "future_potentially_avoidable_ed_90d"


def build_member_target(
    members: pd.DataFrame,
    ed: pd.DataFrame,
    window: SnapshotWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the binary member-level target for one snapshot.

    Returns
    -------
    target_df : one row per member in `members`, columns
        [member_id, TARGET_COLUMN]. Safe to merge into the feature
        snapshot -- contains no raw outcome-window encounter fields.
    outcome_detail_df : the classified outcome-window ED encounters
        themselves, for reporting/validation ONLY. Must never be merged
        into a model feature frame (it contains label-only fields).
    """
    if "member_id" not in members.columns:
        raise KeyError("members frame must contain member_id")

    ed = ed.copy()
    ed["visit_date"] = pd.to_datetime(ed["visit_date"], errors="raise")

    outcome_mask = in_outcome_window(ed["visit_date"], window)
    outcome = ed.loc[outcome_mask].copy()
    outcome["encounter_state"] = classify_ed_encounters(outcome)

    positive_members = set(
        outcome.loc[outcome["encounter_state"] == POTENTIALLY_AVOIDABLE, "member_id"]
    )

    target_df = members[["member_id"]].drop_duplicates().copy()
    target_df[TARGET_COLUMN] = target_df["member_id"].isin(positive_members).astype(int)

    detail_cols = [c for c in ["member_id", "visit_id", "visit_date", "encounter_state"] if c in outcome.columns]
    outcome_detail_df = outcome[detail_cols].reset_index(drop=True)

    return target_df.reset_index(drop=True), outcome_detail_df
