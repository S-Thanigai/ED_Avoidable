"""
safety_context_csv.py
------------------------
Phase 8B -- optional batch `current_safety_context.csv` upload for
POST /uc07/decide. Lets a caller supply CURRENT safety context for many
members at once (member_id, red_flag, icu, admitted, major_procedure,
triage_level) instead of hand-typing the JSON `current_safety_context`
form field member-by-member.

This module owns ONLY parsing/validation of the CSV into
CurrentSafetyContext objects. It reuses
backend/agents/safety_context_schema.py's SafetyContextEntry for
per-row value validation (0/1, 1-5, finite, no silent coercion) rather
than duplicating that logic -- a blank cell (pandas NaN) is treated
exactly like an omitted JSON key: the field stays None/unknown, never 0.

Never used for anything but the Safety & Policy Agent's CLEAR/CAUTION/
OVERRIDE determination. Never a model feature, never joined into the
PIT feature pipeline, never influences the risk score.
"""
from __future__ import annotations

import pandas as pd
from pydantic import ValidationError

from contracts import CurrentSafetyContext
from safety_context_schema import SafetyContextEntry

REQUIRED_COLUMN = "member_id"
OPTIONAL_SAFETY_COLUMNS = ("red_flag", "icu", "admitted", "major_procedure", "triage_level")
ALLOWED_COLUMNS = {REQUIRED_COLUMN, *OPTIONAL_SAFETY_COLUMNS}


class SafetyContextCsvValidationError(ValueError):
    """Every problem found in the optional safety-context CSV, not just
    the first -- lets a caller fix a bad upload in one pass."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


def parse_safety_context_csv(df: pd.DataFrame, known_member_ids: set[str]) -> dict[str, CurrentSafetyContext]:
    """Validates and parses an optional current_safety_context.csv
    (already read into a DataFrame). Joins to the member population
    ONLY by `member_id` (never row order/position).

    Raises SafetyContextCsvValidationError (never silently coerces or
    drops a problem row) if:
      - `member_id` column is missing, or any row's member_id is blank
      - a column outside {member_id, red_flag, icu, admitted,
        major_procedure, triage_level} is present
      - any member_id appears more than once (ambiguous -- rejected
        outright, never silently resolved by picking a row)
      - any member_id is not present in the already-validated
        members_file population
      - any red_flag/icu/admitted/major_procedure/triage_level value is
        not blank and not a valid 0/1 (or 1-5 for triage_level) --
        delegated to SafetyContextEntry, the same validator the JSON
        current_safety_context path uses.
    A blank cell for an optional column means UNKNOWN for that member,
    exactly like an omitted JSON key -- never coerced to 0/false.
    """
    issues: list[str] = []

    if REQUIRED_COLUMN not in df.columns:
        raise SafetyContextCsvValidationError([f"{REQUIRED_COLUMN} column is required"])

    extra_columns = sorted(set(df.columns) - ALLOWED_COLUMNS)
    if extra_columns:
        issues.append(f"unrecognized column(s): {extra_columns}")

    if df[REQUIRED_COLUMN].isna().any():
        blank_rows = df.index[df[REQUIRED_COLUMN].isna()].tolist()
        issues.append(f"member_id is blank on row(s): {blank_rows}")

    member_ids = df[REQUIRED_COLUMN].dropna().astype(str)
    duplicate_ids = sorted(member_ids[member_ids.duplicated(keep=False)].unique().tolist())
    if duplicate_ids:
        issues.append(
            f"duplicate member_id row(s) -- ambiguous, cannot be safely resolved: {duplicate_ids}"
        )

    unknown_ids = sorted(set(member_ids.unique()) - known_member_ids)
    if unknown_ids:
        issues.append(f"member_id(s) not found in members_file: {unknown_ids}")

    if issues:
        raise SafetyContextCsvValidationError(issues)

    contexts: dict[str, CurrentSafetyContext] = {}
    for idx, row in df.iterrows():
        member_id = str(row[REQUIRED_COLUMN])
        entry_kwargs = {}
        for field_name in OPTIONAL_SAFETY_COLUMNS:
            if field_name not in df.columns:
                continue
            value = row[field_name]
            if pd.isna(value):
                continue  # blank -- stays unknown, never coerced to 0
            entry_kwargs[field_name] = value

        try:
            validated = SafetyContextEntry(**entry_kwargs)
        except ValidationError as exc:
            issues.append(f"row {idx} (member_id {member_id!r}): {exc.errors()}")
            continue

        contexts[member_id] = CurrentSafetyContext(
            red_flag=validated.red_flag,
            icu=validated.icu,
            admitted=validated.admitted,
            major_procedure=validated.major_procedure,
            triage_level=validated.triage_level,
        )

    if issues:
        raise SafetyContextCsvValidationError(issues)

    return contexts
