"""Tests for backend/pit/windows.py."""
import pandas as pd

from windows import (
    OBSERVATION_WINDOW_DAYS,
    OUTCOME_WINDOW_DAYS,
    TEST_INDEX_DATE,
    TRAIN_INDEX_DATE,
    VALIDATION_INDEX_DATE,
    build_all_snapshot_windows,
    build_snapshot_window,
    in_observation_window,
    in_outcome_window,
)


def test_approved_index_dates():
    assert TRAIN_INDEX_DATE == pd.Timestamp("2025-10-05")
    assert VALIDATION_INDEX_DATE == pd.Timestamp("2026-01-03")
    assert TEST_INDEX_DATE == pd.Timestamp("2026-04-03")


def test_window_days_are_locked():
    assert OBSERVATION_WINDOW_DAYS == 270
    assert OUTCOME_WINDOW_DAYS == 90


def test_boundary_arithmetic():
    w = build_snapshot_window("train", "2025-10-05")
    assert w.index_date == pd.Timestamp("2025-10-05")
    assert w.observation_start == pd.Timestamp("2025-10-05") - pd.Timedelta(days=270)
    assert w.observation_end == w.index_date
    assert w.outcome_start == w.index_date
    assert w.outcome_end == pd.Timestamp("2025-10-05") + pd.Timedelta(days=90)


def test_observation_and_outcome_use_datetime_types():
    w = build_snapshot_window("train", "2025-10-05")
    for field in (w.index_date, w.observation_start, w.observation_end, w.outcome_start, w.outcome_end):
        assert isinstance(field, pd.Timestamp)


def test_observation_window_half_open_boundaries():
    w = build_snapshot_window("train", "2025-10-05")
    dates = pd.to_datetime([
        w.observation_start - pd.Timedelta(days=1),  # just before -> excluded
        w.observation_start,                          # exactly at start -> included
        w.index_date - pd.Timedelta(days=1),           # day before index -> included
        w.index_date,                                  # exactly index_date -> excluded from observation
    ])
    mask = in_observation_window(pd.Series(dates), w)
    assert list(mask) == [False, True, True, False]


def test_outcome_window_half_open_boundaries():
    w = build_snapshot_window("train", "2025-10-05")
    dates = pd.to_datetime([
        w.index_date - pd.Timedelta(days=1),   # day before index -> excluded from outcome
        w.index_date,                          # exactly index_date -> included
        w.outcome_end - pd.Timedelta(days=1),  # day before outcome_end -> included
        w.outcome_end,                         # exactly outcome_end -> excluded
    ])
    mask = in_outcome_window(pd.Series(dates), w)
    assert list(mask) == [False, True, True, False]


def test_observation_and_outcome_never_overlap_for_any_date():
    """No event date can ever satisfy both in_observation_window and
    in_outcome_window for the same snapshot -- swept across a wide date
    range around the index date."""
    w = build_snapshot_window("train", "2025-10-05")
    dates = pd.date_range(w.observation_start - pd.Timedelta(days=10), w.outcome_end + pd.Timedelta(days=10), freq="D")
    obs = in_observation_window(pd.Series(dates), w)
    out = in_outcome_window(pd.Series(dates), w)
    assert not (obs & out).any()


def test_build_all_snapshot_windows_returns_three_approved_snapshots():
    windows = build_all_snapshot_windows()
    assert set(windows) == {"train", "validation", "test"}
    assert windows["train"].index_date == TRAIN_INDEX_DATE
    assert windows["validation"].index_date == VALIDATION_INDEX_DATE
    assert windows["test"].index_date == TEST_INDEX_DATE


def test_snapshots_have_non_overlapping_outcome_windows():
    windows = build_all_snapshot_windows()
    train, val, test = windows["train"], windows["validation"], windows["test"]
    assert train.outcome_end <= val.outcome_start
    assert val.outcome_end <= test.outcome_start
