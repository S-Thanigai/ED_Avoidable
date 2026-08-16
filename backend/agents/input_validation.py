"""
input_validation.py
---------------------
Phase 7 -- static value-level validation and safe normalization for the
raw member/ED-visit CSV uploads that feed the authoritative UC07 path
(Phase 3 PIT pipeline + Phase 5 orchestrator). Runs BEFORE point-in-time
feature reconstruction and BEFORE the Risk Detection Agent ever sees a
row -- rejects impossible inputs at the boundary rather than letting
them silently flow into scoring.

Column PRESENCE was already checked by backend/main.py's
_validate_columns(); this module checks VALUE plausibility (Phase 6
Section 28's documented gap) and is deliberately kept out of main.py
(business logic belongs here, not in the FastAPI route handlers) and
out of backend/pit/ (the frozen, unmodified Phase 3 PIT pipeline never
needs to re-validate what already passed through here).

Never silently coerces an invalid value into a valid-looking default:
every problem found is collected and raised together in one
MemberDataValidationError / EdVisitDataValidationError, so a caller can
fix a bad upload in one pass.
"""
from __future__ import annotations

import math

import pandas as pd

# Generous, documented engineering bounds -- NOT narrowed to this
# project's synthetic sample's observed ranges (age 18-90, distances
# 0.2-30mi), which would be an invented restriction the schema does not
# actually require. These reject clear data-entry errors (negative
# values, six-figure distances, negative ages) without imposing
# unrealistic clinical limits.
AGE_MIN = 0
AGE_MAX = 120
DISTANCE_MIN = 0.0
DISTANCE_MAX = 500.0

CHRONIC_FLAG_COLUMNS = ["diabetes", "copd", "hypertension", "chf", "asthma", "ckd"]
BINARY_MEMBER_FIELDS = CHRONIC_FLAG_COLUMNS + ["transportation_barrier", "telehealth_available"]
DISTANCE_FIELDS = ["pcp_distance_miles", "urgent_care_distance_miles"]

ED_BINARY_FIELDS = ["admitted", "icu", "major_procedure", "red_flag"]
TRIAGE_VALID_VALUES = (1, 2, 3, 4, 5)


class MemberDataValidationError(ValueError):
    """Every problem found in raw_members.csv-shaped data, not just the
    first -- lets a caller fix a bad upload in one pass."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


class EdVisitDataValidationError(ValueError):
    """Every problem found in raw_ed_visits.csv-shaped data's
    safety-relevant fields."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


def _is_finite_number(value) -> bool:
    if isinstance(value, bool):
        return True  # bool is a valid finite 0/1-shaped value; range-checked separately
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _validate_binary_series(series: pd.Series, name: str, issues: list[str]) -> None:
    for idx, value in series.items():
        if _is_missing(value):
            issues.append(f"{name}: row {idx} is missing (NaN) -- binary fields must be 0 or 1, never blank")
            continue
        if not _is_finite_number(value):
            issues.append(f"{name}: row {idx} has a non-finite or non-numeric value {value!r}")
            continue
        if float(value) not in (0.0, 1.0):
            issues.append(f"{name}: row {idx} has invalid value {value!r} (must be 0 or 1)")


def validate_and_normalize_members_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validates raw_members.csv-shaped data. Returns a normalized COPY:
    `num_chronic_conditions` is safely derived from the six chronic-flag
    columns when the caller omits it (Phase 7 Step 12 -- prefer deriving
    a genuinely-derivable field over requiring a redundant client-supplied
    value); when the caller DOES supply it, it is validated for
    consistency against the flags rather than silently trusted or
    silently overwritten. Raises MemberDataValidationError listing every
    problem found if anything is invalid."""
    issues: list[str] = []
    out = df.copy()

    if "age" not in out.columns:
        issues.append("age column is required")
    else:
        for idx, value in out["age"].items():
            if _is_missing(value):
                issues.append(f"age: row {idx} is missing (NaN)")
                continue
            if not _is_finite_number(value):
                issues.append(f"age: row {idx} has a non-finite or non-numeric value {value!r}")
                continue
            f = float(value)
            if f != int(f):
                issues.append(f"age: row {idx} value {value!r} is not a whole number")
                continue
            if not (AGE_MIN <= f <= AGE_MAX):
                issues.append(f"age: row {idx} value {value!r} is outside the supported range [{AGE_MIN}, {AGE_MAX}]")

    for field_name in BINARY_MEMBER_FIELDS:
        if field_name not in out.columns:
            issues.append(f"{field_name} column is required")
            continue
        _validate_binary_series(out[field_name], field_name, issues)

    for field_name in DISTANCE_FIELDS:
        if field_name not in out.columns:
            issues.append(f"{field_name} column is required")
            continue
        for idx, value in out[field_name].items():
            if _is_missing(value):
                issues.append(f"{field_name}: row {idx} is missing (NaN)")
                continue
            if not _is_finite_number(value):
                issues.append(f"{field_name}: row {idx} has a non-finite or non-numeric value {value!r}")
                continue
            f = float(value)
            if not (DISTANCE_MIN <= f <= DISTANCE_MAX):
                issues.append(f"{field_name}: row {idx} value {value!r} is outside the supported range [{DISTANCE_MIN}, {DISTANCE_MAX}]")

    have_all_flags = all(c in out.columns for c in CHRONIC_FLAG_COLUMNS)
    flags_are_valid_so_far = have_all_flags and not any(f"{c}:" in issue for c in CHRONIC_FLAG_COLUMNS for issue in issues)

    if "num_chronic_conditions" not in out.columns:
        if flags_are_valid_so_far:
            out["num_chronic_conditions"] = out[CHRONIC_FLAG_COLUMNS].sum(axis=1).astype(int)
        elif have_all_flags:
            pass  # flag columns present but invalid; already recorded above -- do not compound with a derived value
        else:
            issues.append(
                "num_chronic_conditions is missing and cannot be safely derived "
                "(one or more chronic-flag columns are also missing)"
            )
    else:
        for idx, value in out["num_chronic_conditions"].items():
            if _is_missing(value):
                issues.append(f"num_chronic_conditions: row {idx} is missing (NaN)")
                continue
            if not _is_finite_number(value):
                issues.append(f"num_chronic_conditions: row {idx} has a non-finite or non-numeric value {value!r}")
                continue
            f = float(value)
            if f != int(f) or f < 0:
                issues.append(f"num_chronic_conditions: row {idx} value {value!r} must be a non-negative whole number")
        if flags_are_valid_so_far and not any("num_chronic_conditions:" in issue for issue in issues):
            expected = out[CHRONIC_FLAG_COLUMNS].sum(axis=1)
            mismatched = out.index[out["num_chronic_conditions"].astype(float) != expected.astype(float)]
            for idx in mismatched:
                issues.append(
                    f"num_chronic_conditions: row {idx} value {out.loc[idx, 'num_chronic_conditions']!r} "
                    f"does not equal the sum of diabetes+copd+hypertension+chf+asthma+ckd "
                    f"({int(expected.loc[idx])}) for that row"
                )

    if issues:
        raise MemberDataValidationError(issues)
    return out


def validate_ed_visits_df(df: pd.DataFrame) -> None:
    """Validates raw_ed_visits.csv-shaped data's safety-relevant fields.

    triage_level is validated HERE across every row (not just rows that
    happen to fall inside some snapshot's observation window) -- closing
    the exact gap documented in docs/06_SAFETY_FAIRNESS_ROBUSTNESS_VALIDATION.md
    section 28: previously, an invalid triage_level outside every
    snapshot's observation window would never reach
    backend/pit/encounter_classification.py's own validation and would
    silently pass through unrejected."""
    issues: list[str] = []

    for field_name in ED_BINARY_FIELDS:
        if field_name not in df.columns:
            issues.append(f"{field_name} column is required")
            continue
        _validate_binary_series(df[field_name], field_name, issues)

    if "triage_level" not in df.columns:
        issues.append("triage_level column is required")
    else:
        for idx, value in df["triage_level"].items():
            if _is_missing(value):
                issues.append(f"triage_level: row {idx} is missing (NaN)")
                continue
            if not _is_finite_number(value):
                issues.append(f"triage_level: row {idx} has a non-finite or non-numeric value {value!r}")
                continue
            f = float(value)
            if f != int(f) or int(f) not in TRIAGE_VALID_VALUES:
                issues.append(f"triage_level: row {idx} value {value!r} must be one of {list(TRIAGE_VALID_VALUES)}")

    if issues:
        raise EdVisitDataValidationError(issues)
