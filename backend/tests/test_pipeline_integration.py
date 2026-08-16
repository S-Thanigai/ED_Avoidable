"""
End-to-end integration test: runs the real Phase 3 pipeline against the
actual (immutable) raw CSVs and checks the result. This is the one test
file that touches the real data on disk -- it never writes to the raw
files, only reads them, and independently verifies their SHA-256 hashes
are unchanged before and after the run.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import build_snapshots as bs
from target import TARGET_COLUMN

RAW_FILES = [bs.RAW_MEMBERS, bs.RAW_ED, bs.RAW_CARE]

# Hashes recorded at the start of the Phase 3 implementation session
# (before any code in this pipeline existed). If these ever fail to match
# the live files, the raw datasets have been modified and that is a
# hard failure -- see docs/03_ML_DATA_PIPELINE.md section 20.
EXPECTED_RAW_HASHES = {
    "raw_members.csv": "b94df89ed042a8feaa1bb46d7939e124fb9f6b03308b11da045412a427b78c46",
    "raw_ed_visits.csv": "f8db1839fb7966c4230c771252a3b935c318d0838de9258dc29de42d042f5d47",
    "raw_care_history.csv": "358d3033faa4e0529aed834cd8847f72d0b5d4ca51fa76748523fab790c81657",
}


def test_raw_files_exist_at_expected_locations():
    for path in RAW_FILES:
        assert path.exists(), f"expected immutable raw dataset at {path}"


def test_raw_dataset_hashes_match_recorded_values():
    for path in RAW_FILES:
        actual = bs.sha256_file(path)
        assert actual == EXPECTED_RAW_HASHES[path.name], f"{path.name} hash changed -- raw dataset was modified"


@pytest.fixture(scope="module")
def pipeline_result():
    return bs.main()


def test_raw_hashes_unchanged_before_and_after_pipeline_run(pipeline_result):
    assert pipeline_result["raw_hashes_before"] == pipeline_result["raw_hashes_after"]
    assert pipeline_result["raw_hashes_before"] == EXPECTED_RAW_HASHES


def test_validation_report_all_passed(pipeline_result):
    assert pipeline_result["validation_report"]["all_passed"] is True


def test_three_snapshots_produced_with_all_members(pipeline_result):
    members = pd.read_csv(bs.RAW_MEMBERS)
    for name in ("train", "validation", "test"):
        snap = pipeline_result["snapshots"][name]
        assert len(snap) == len(members)
        assert set(snap["member_id"]) == set(members["member_id"])


def test_schema_identical_across_snapshots(pipeline_result):
    cols = {name: list(df.columns) for name, df in pipeline_result["snapshots"].items()}
    assert cols["train"] == cols["validation"] == cols["test"]


def test_target_column_present_and_binary(pipeline_result):
    for name, snap in pipeline_result["snapshots"].items():
        assert TARGET_COLUMN in snap.columns
        assert set(snap[TARGET_COLUMN].unique()).issubset({0, 1})


def test_legacy_target_and_diagnosis_columns_absent(pipeline_result):
    for name, snap in pipeline_result["snapshots"].items():
        assert "frequent_ED_user" not in snap.columns
        assert not any(c.startswith("diagnosis_") for c in snap.columns)


def test_identifier_metadata_target_not_in_feature_columns(pipeline_result):
    for name, snap in pipeline_result["snapshots"].items():
        feature_cols = [c for c in snap.columns if c not in {"member_id", "index_date", TARGET_COLUMN}]
        assert "member_id" not in feature_cols
        assert "index_date" not in feature_cols
        assert TARGET_COLUMN not in feature_cols


def test_prevalence_in_plausible_range(pipeline_result):
    """Phase 2 measured ~9% prevalence at these exact snapshots. This is
    not forced to equal 9% -- just checked as a broad sanity range."""
    for name, snap in pipeline_result["snapshots"].items():
        prevalence = snap[TARGET_COLUMN].mean()
        assert 0.03 < prevalence < 0.20, f"{name} prevalence {prevalence:.4f} outside plausible range"


def test_derived_files_written(pipeline_result):
    for name in ("train", "validation", "test"):
        assert (bs.DERIVED_DIR / f"{name}_snapshot.csv").exists()
    assert (bs.DERIVED_DIR / "feature_manifest.json").exists()
    assert (bs.DERIVED_DIR / "snapshot_metadata.json").exists()
    assert (bs.DERIVED_DIR / "validation_report.json").exists()


def test_feature_manifest_is_valid_json_and_covers_every_column(pipeline_result):
    manifest = json.loads((bs.DERIVED_DIR / "feature_manifest.json").read_text())
    manifest_names = {f["feature_name"] for f in manifest["features"]}
    snapshot_cols = set(pipeline_result["snapshots"]["train"].columns)
    assert manifest_names == snapshot_cols
    assert "future_potentially_avoidable_ed_90d" in manifest["target_columns"]
    assert "member_id" in manifest["identifier_columns"]


def test_no_unclassified_manifest_entries(pipeline_result):
    manifest = json.loads((bs.DERIVED_DIR / "feature_manifest.json").read_text())
    unclassified = [f["feature_name"] for f in manifest["features"] if f["category"] == "unclassified"]
    assert unclassified == [], f"manifest has unclassified columns: {unclassified}"


def test_snapshot_metadata_contains_required_fields(pipeline_result):
    metadata = json.loads((bs.DERIVED_DIR / "snapshot_metadata.json").read_text())
    for key in (
        "phase", "generated_at_utc", "raw_dataset_sha256",
        "observation_window_days", "outcome_window_days", "snapshot_index_dates",
        "target_column", "target_definition", "encounter_state_definition",
        "snapshots", "member_overlap",
    ):
        assert key in metadata, f"missing metadata key: {key}"
    assert metadata["observation_window_days"] == 270
    assert metadata["outcome_window_days"] == 90
    assert metadata["snapshot_index_dates"]["train"] == "2025-10-05"
    assert metadata["snapshot_index_dates"]["validation"] == "2026-01-03"
    assert metadata["snapshot_index_dates"]["test"] == "2026-04-03"


def test_written_csvs_round_trip_row_counts(pipeline_result):
    for name in ("train", "validation", "test"):
        on_disk = pd.read_csv(bs.DERIVED_DIR / f"{name}_snapshot.csv")
        assert len(on_disk) == len(pipeline_result["snapshots"][name])
