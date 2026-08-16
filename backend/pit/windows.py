"""
windows.py
----------
Point-in-time snapshot boundary definitions for the UC07 Phase 3 pipeline.

Convention (docs/02_UC07_AND_DATA_DESIGN.md section 9; locked in the
Phase 3 spec):

    OBSERVATION window (history, half-open):
        index_date - OBSERVATION_WINDOW_DAYS  <=  event_date  <  index_date

    OUTCOME window (future, half-open):
        index_date  <=  event_date  <  index_date + OUTCOME_WINDOW_DAYS

Both intervals share `index_date` as their common boundary -- inclusive on
the outcome side, exclusive on the observation side -- so they are
disjoint by construction: no event date can ever satisfy both conditions.

Index dates are FIXED, approved calendar dates (docs/DECISION_LOG.md item
15). They are never derived from the dataset's own maximum ED visit date
or any other dataset-dependent value -- that global-max-date pattern is
exactly the Phase 1 leakage bug this design replaces.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

OBSERVATION_WINDOW_DAYS = 270
OUTCOME_WINDOW_DAYS = 90

# Approved Phase 2 snapshot index dates (docs/DECISION_LOG.md item 15).
# These are literal, fixed calendar dates -- not computed from any raw
# dataset column.
TRAIN_INDEX_DATE = pd.Timestamp("2025-10-05")
VALIDATION_INDEX_DATE = pd.Timestamp("2026-01-03")
TEST_INDEX_DATE = pd.Timestamp("2026-04-03")

SNAPSHOT_INDEX_DATES = {
    "train": TRAIN_INDEX_DATE,
    "validation": VALIDATION_INDEX_DATE,
    "test": TEST_INDEX_DATE,
}


@dataclass(frozen=True)
class SnapshotWindow:
    name: str
    index_date: pd.Timestamp
    observation_start: pd.Timestamp   # inclusive
    observation_end: pd.Timestamp     # == index_date, exclusive upper bound
    outcome_start: pd.Timestamp       # == index_date, inclusive lower bound
    outcome_end: pd.Timestamp         # exclusive upper bound


def build_snapshot_window(
    name: str,
    index_date,
    observation_days: int = OBSERVATION_WINDOW_DAYS,
    outcome_days: int = OUTCOME_WINDOW_DAYS,
) -> SnapshotWindow:
    """Build a SnapshotWindow from an explicit index date. Proper datetime
    types throughout -- never raw string comparison."""
    index_date = pd.Timestamp(index_date)
    return SnapshotWindow(
        name=name,
        index_date=index_date,
        observation_start=index_date - pd.Timedelta(days=observation_days),
        observation_end=index_date,
        outcome_start=index_date,
        outcome_end=index_date + pd.Timedelta(days=outcome_days),
    )


def build_all_snapshot_windows() -> dict[str, SnapshotWindow]:
    """Build the three approved (train/validation/test) snapshot windows."""
    return {
        name: build_snapshot_window(name, index_date)
        for name, index_date in SNAPSHOT_INDEX_DATES.items()
    }


def in_observation_window(event_date: pd.Series, window: SnapshotWindow) -> pd.Series:
    """Boolean mask: observation_start <= event_date < index_date."""
    return (event_date >= window.observation_start) & (event_date < window.observation_end)


def in_outcome_window(event_date: pd.Series, window: SnapshotWindow) -> pd.Series:
    """Boolean mask: index_date <= event_date < outcome_end."""
    return (event_date >= window.outcome_start) & (event_date < window.outcome_end)
