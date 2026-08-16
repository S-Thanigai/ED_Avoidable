"""
encounter_classification.py
----------------------------
Deterministic, safety-first classification of historical ED encounters.

Implements the Phase 2 design (docs/02_UC07_AND_DATA_DESIGN.md section 4,
"Historical Avoidability Definition") and the locked Phase 3 spec.

encounter_state is one of:
    POTENTIALLY_AVOIDABLE     -- triage 4 or 5, no safety exclusion
    PROTECTED_OR_HIGH_ACUITY  -- any safety exclusion tripped (absolute
                                  precedence over triage level)
    UNCERTAIN                 -- triage 3, no safety exclusion

Safety exclusions (red_flag, icu, admitted, major_procedure, triage_level
in {1, 2}) are evaluated FIRST and unconditionally override any
triage-level-based avoidability logic.

`diagnosis` is intentionally NEVER used here -- Phase 1/2 verified it
carries no measurable acuity signal in this dataset (near-identical
triage/red_flag/admitted/icu/major_procedure rates across every diagnosis
category), so encoding a diagnosis-based rule would be an invented
assumption, not a data-supported pattern.

This module never labels an encounter "unnecessary", "inappropriate", or
"definitely avoidable" -- only the three approved state names above.
"""
from __future__ import annotations

import pandas as pd

POTENTIALLY_AVOIDABLE = "POTENTIALLY_AVOIDABLE"
PROTECTED_OR_HIGH_ACUITY = "PROTECTED_OR_HIGH_ACUITY"
UNCERTAIN = "UNCERTAIN"

ENCOUNTER_STATES = (POTENTIALLY_AVOIDABLE, PROTECTED_OR_HIGH_ACUITY, UNCERTAIN)

# Safety-exclusion triage levels -- unconditionally PROTECTED_OR_HIGH_ACUITY.
HIGH_ACUITY_TRIAGE_LEVELS = {1, 2}
# Candidate potentially-avoidable triage levels (only reached if no
# safety exclusion applies).
LOWER_ACUITY_TRIAGE_LEVELS = {4, 5}
# Ambiguous triage level -- never forced positive.
AMBIGUOUS_TRIAGE_LEVELS = {3}

REQUIRED_COLUMNS = ["triage_level", "red_flag", "admitted", "icu", "major_procedure"]


def classify_ed_encounter(triage_level, red_flag, admitted, icu, major_procedure) -> str:
    """
    Classify a single historical ED encounter. Deterministic. Safety
    exclusions take absolute precedence over triage-level-based
    avoidability logic (docs/02_UC07_AND_DATA_DESIGN.md section 4.3).
    """
    triage_level = int(triage_level)

    if (
        int(red_flag) == 1
        or int(icu) == 1
        or int(admitted) == 1
        or int(major_procedure) == 1
        or triage_level in HIGH_ACUITY_TRIAGE_LEVELS
    ):
        return PROTECTED_OR_HIGH_ACUITY

    if triage_level in LOWER_ACUITY_TRIAGE_LEVELS:
        return POTENTIALLY_AVOIDABLE

    if triage_level in AMBIGUOUS_TRIAGE_LEVELS:
        return UNCERTAIN

    raise ValueError(f"Unrecognized triage_level: {triage_level!r} (expected an integer 1-5)")


def classify_ed_encounters(ed: pd.DataFrame) -> pd.Series:
    """
    Vectorized version of classify_ed_encounter(). `ed` must contain the
    columns in REQUIRED_COLUMNS. Returns a pd.Series of encounter_state
    strings aligned to `ed`'s index. Logic is kept in exact lockstep with
    the scalar function above (both implement the same safety-precedence
    rule) and is covered by tests asserting agreement between the two.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in ed.columns]
    if missing:
        raise KeyError(f"classify_ed_encounters is missing required columns: {missing}")

    triage = ed["triage_level"].astype(int)

    protected = (
        (ed["red_flag"].astype(int) == 1)
        | (ed["icu"].astype(int) == 1)
        | (ed["admitted"].astype(int) == 1)
        | (ed["major_procedure"].astype(int) == 1)
        | triage.isin(HIGH_ACUITY_TRIAGE_LEVELS)
    )
    potentially_avoidable = (~protected) & triage.isin(LOWER_ACUITY_TRIAGE_LEVELS)
    uncertain = (~protected) & triage.isin(AMBIGUOUS_TRIAGE_LEVELS)

    state = pd.Series(pd.NA, index=ed.index, dtype="object")
    state[protected] = PROTECTED_OR_HIGH_ACUITY
    state[potentially_avoidable] = POTENTIALLY_AVOIDABLE
    state[uncertain] = UNCERTAIN

    unclassified = state.isna()
    if unclassified.any():
        bad_values = sorted(triage.loc[unclassified].unique().tolist())
        raise ValueError(
            f"Encounters with unrecognized triage_level could not be classified: {bad_values}"
        )

    return state
