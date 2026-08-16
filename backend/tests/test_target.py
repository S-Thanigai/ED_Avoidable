"""Tests for backend/pit/target.py, using small synthetic in-memory data
(never the raw datasets) to cover edge cases precisely."""
import pandas as pd

from target import TARGET_COLUMN, build_member_target
from windows import build_snapshot_window

WINDOW = build_snapshot_window("train", "2025-10-05")


def _members(member_ids):
    return pd.DataFrame({"member_id": member_ids})


def _ed_row(member_id, visit_date, triage_level, red_flag=0, admitted=0, icu=0, major_procedure=0, visit_id=None):
    return {
        "visit_id": visit_id or f"V-{member_id}-{visit_date}",
        "member_id": member_id,
        "visit_date": visit_date,
        "triage_level": triage_level,
        "red_flag": red_flag,
        "admitted": admitted,
        "icu": icu,
        "major_procedure": major_procedure,
    }


def test_member_with_potentially_avoidable_outcome_encounter_is_positive():
    members = _members(["M1"])
    ed = pd.DataFrame([_ed_row("M1", "2025-11-01", triage_level=5)])  # inside outcome window
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 1


def test_member_with_only_uncertain_outcome_encounter_is_negative():
    members = _members(["M1"])
    ed = pd.DataFrame([_ed_row("M1", "2025-11-01", triage_level=3)])  # UNCERTAIN
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 0


def test_member_with_only_protected_outcome_encounter_is_negative():
    members = _members(["M1"])
    ed = pd.DataFrame([_ed_row("M1", "2025-11-01", triage_level=5, red_flag=1)])  # PROTECTED
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 0


def test_member_with_protected_and_avoidable_encounters_is_positive():
    """A member can have both a protected and an avoidable encounter in the
    same outcome window; the presence of >=1 avoidable encounter still
    makes them positive -- other encounters don't cancel it out."""
    members = _members(["M1"])
    ed = pd.DataFrame([
        _ed_row("M1", "2025-11-01", triage_level=5, red_flag=1, visit_id="V1"),
        _ed_row("M1", "2025-11-15", triage_level=4, visit_id="V2"),
    ])
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 1


def test_member_with_no_ed_encounters_is_negative():
    members = _members(["M1"])
    ed = pd.DataFrame(columns=["visit_id", "member_id", "visit_date", "triage_level", "red_flag", "admitted", "icu", "major_procedure"])
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 0


def test_observation_window_encounter_does_not_affect_target():
    """A POTENTIALLY_AVOIDABLE encounter strictly BEFORE index_date (i.e.
    in the observation window) must never make the target positive --
    only outcome-window encounters count."""
    members = _members(["M1"])
    ed = pd.DataFrame([_ed_row("M1", "2025-09-01", triage_level=5)])  # before index_date 2025-10-05
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 0


def test_encounter_exactly_at_index_date_counts_as_outcome():
    members = _members(["M1"])
    ed = pd.DataFrame([_ed_row("M1", str(WINDOW.index_date.date()), triage_level=5)])
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 1


def test_encounter_on_outcome_end_date_is_excluded():
    """outcome_end is exclusive -- a visit exactly on that date is not
    counted (it belongs to the NEXT day, outside the 90-day window)."""
    members = _members(["M1"])
    ed = pd.DataFrame([_ed_row("M1", str(WINDOW.outcome_end.date()), triage_level=5)])
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert target_df.loc[target_df["member_id"] == "M1", TARGET_COLUMN].iloc[0] == 0


def test_every_member_gets_a_label():
    members = _members(["M1", "M2", "M3"])
    ed = pd.DataFrame([_ed_row("M1", "2025-11-01", triage_level=5)])
    target_df, _ = build_member_target(members, ed, WINDOW)
    assert set(target_df["member_id"]) == {"M1", "M2", "M3"}
    assert target_df[TARGET_COLUMN].isin([0, 1]).all()
    assert target_df[TARGET_COLUMN].isna().sum() == 0


def test_outcome_detail_never_used_as_features_contains_label_only_fields():
    """The detail frame legitimately contains label-only fields (that's
    its whole purpose for reporting) -- this test just documents that it
    is a SEPARATE object from target_df, which contains none of them."""
    members = _members(["M1"])
    ed = pd.DataFrame([_ed_row("M1", "2025-11-01", triage_level=5)])
    target_df, detail_df = build_member_target(members, ed, WINDOW)
    assert "triage_level" not in target_df.columns
    assert "encounter_state" in detail_df.columns
