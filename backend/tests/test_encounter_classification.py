"""Tests for backend/pit/encounter_classification.py."""
import pandas as pd
import pytest

from encounter_classification import (
    POTENTIALLY_AVOIDABLE,
    PROTECTED_OR_HIGH_ACUITY,
    UNCERTAIN,
    classify_ed_encounter,
    classify_ed_encounters,
)


# The 9 mandated Phase 3 test cases: (triage_level, red_flag, admitted, icu, major_procedure, expected)
MANDATED_CASES = [
    (5, 0, 0, 0, 0, POTENTIALLY_AVOIDABLE),   # 1. triage 5, no exclusion
    (4, 0, 0, 0, 0, POTENTIALLY_AVOIDABLE),   # 2. triage 4, no exclusion
    (3, 0, 0, 0, 0, UNCERTAIN),               # 3. triage 3, no exclusion
    (2, 0, 0, 0, 0, PROTECTED_OR_HIGH_ACUITY),  # 4. triage 2
    (1, 0, 0, 0, 0, PROTECTED_OR_HIGH_ACUITY),  # 5. triage 1
    (5, 0, 1, 0, 0, PROTECTED_OR_HIGH_ACUITY),  # 6. triage 5 + admitted
    (5, 1, 0, 0, 0, PROTECTED_OR_HIGH_ACUITY),  # 7. triage 5 + red_flag
    (4, 0, 0, 1, 0, PROTECTED_OR_HIGH_ACUITY),  # 8. triage 4 + ICU
    (4, 0, 0, 0, 1, PROTECTED_OR_HIGH_ACUITY),  # 9. triage 4 + major_procedure
]


@pytest.mark.parametrize("triage_level,red_flag,admitted,icu,major_procedure,expected", MANDATED_CASES)
def test_mandated_cases_scalar(triage_level, red_flag, admitted, icu, major_procedure, expected):
    assert classify_ed_encounter(triage_level, red_flag, admitted, icu, major_procedure) == expected


def test_mandated_cases_vectorized():
    df = pd.DataFrame(
        MANDATED_CASES,
        columns=["triage_level", "red_flag", "admitted", "icu", "major_procedure", "expected"],
    )
    result = classify_ed_encounters(df)
    assert list(result) == list(df["expected"])


def test_safety_exclusion_overrides_low_acuity_triage():
    """A safety exclusion at the lowest-acuity triage level (5) must still
    win -- exclusions have absolute precedence regardless of triage."""
    assert classify_ed_encounter(5, 1, 0, 0, 0) == PROTECTED_OR_HIGH_ACUITY
    assert classify_ed_encounter(5, 0, 0, 1, 0) == PROTECTED_OR_HIGH_ACUITY
    assert classify_ed_encounter(5, 0, 0, 0, 1) == PROTECTED_OR_HIGH_ACUITY


def test_multiple_exclusions_still_protected():
    assert classify_ed_encounter(1, 1, 1, 1, 1) == PROTECTED_OR_HIGH_ACUITY


def test_never_returns_forbidden_terminology():
    forbidden = {"unnecessary", "inappropriate", "definitely_avoidable", "avoidable"}
    for case in MANDATED_CASES:
        result = classify_ed_encounter(*case[:5])
        assert result in (POTENTIALLY_AVOIDABLE, PROTECTED_OR_HIGH_ACUITY, UNCERTAIN)
        assert result.lower() not in forbidden


def test_diagnosis_not_used_in_classification():
    """classify_ed_encounter has no diagnosis parameter at all -- this test
    documents that omission is intentional (Phase 2 section 4.5)."""
    import inspect

    params = inspect.signature(classify_ed_encounter).parameters
    assert "diagnosis" not in params


def test_vectorized_matches_scalar_on_random_grid():
    """Exhaustively check every triage_level x exclusion-flag combination:
    vectorized and scalar implementations must agree everywhere."""
    rows = []
    for triage_level in range(1, 6):
        for red_flag in (0, 1):
            for admitted in (0, 1):
                for icu in (0, 1):
                    for major_procedure in (0, 1):
                        rows.append((triage_level, red_flag, admitted, icu, major_procedure))
    df = pd.DataFrame(rows, columns=["triage_level", "red_flag", "admitted", "icu", "major_procedure"])
    vectorized = classify_ed_encounters(df)
    scalar = [classify_ed_encounter(*row) for row in rows]
    assert list(vectorized) == scalar


def test_unrecognized_triage_level_raises():
    with pytest.raises(ValueError):
        classify_ed_encounter(9, 0, 0, 0, 0)


def test_missing_columns_raises_keyerror():
    df = pd.DataFrame({"triage_level": [4]})
    with pytest.raises(KeyError):
        classify_ed_encounters(df)
