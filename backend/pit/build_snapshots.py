"""
build_snapshots.py
-------------------
Phase 3 orchestration: builds the TRAIN/VALIDATION/TEST point-in-time
snapshot datasets from a members/ED/care-history CSV trio.

Usage (run from anywhere; paths are resolved relative to this file):
    python backend/pit/build_snapshots.py

Reads the raw CSVs READ-ONLY. Writes ONLY to the requested output
directory. Never modifies its input CSVs -- this is enforced by never
opening those paths for writing anywhere in this package, and
independently verified via SHA-256 hash comparison before and after
every run (see docs/03_ML_DATA_PIPELINE.md section 20).

Data-source aware (Phase 4C, docs/04C_SYNTHETIC_DATA_EXPERIMENT.md): the
SAME feature/target/window logic in `features.py`/`target.py`/`windows.py`
is reused unchanged for any input trio -- there is exactly one point-in-time
implementation, never a second copy for a different dataset. `main()`'s
defaults reproduce the original, pre-Phase-4C behavior exactly (original
raw CSVs at the repo root -> data/derived/, dataset_id="original"); every
other data source is an explicit override, never a hard-coded branch.

This script does NOT train, retrain, or evaluate any predictive model.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # flat sibling imports, matches backend/ convention

from encounter_classification import POTENTIALLY_AVOIDABLE, PROTECTED_OR_HIGH_ACUITY, UNCERTAIN, classify_ed_encounters
from features import build_observation_features
from manifest import build_feature_manifest
from target import TARGET_COLUMN, build_member_target
from validation import run_all_checks
from windows import OBSERVATION_WINDOW_DAYS, OUTCOME_WINDOW_DAYS, build_all_snapshot_windows

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_MEMBERS = REPO_ROOT / "raw_members.csv"
RAW_ED = REPO_ROOT / "raw_ed_visits.csv"
RAW_CARE = REPO_ROOT / "raw_care_history.csv"
DERIVED_DIR = REPO_ROOT / "data" / "derived"

SYNTHETIC_RAW_MEMBERS = REPO_ROOT / "data" / "synthetic" / "raw_members.csv"
SYNTHETIC_RAW_ED = REPO_ROOT / "data" / "synthetic" / "raw_ed_visits.csv"
SYNTHETIC_RAW_CARE = REPO_ROOT / "data" / "synthetic" / "raw_care_history.csv"
SYNTHETIC_DERIVED_DIR = REPO_ROOT / "data" / "derived" / "synthetic"

IDENTIFIER_COLUMNS = {"member_id"}
METADATA_COLUMNS = {"index_date"}
TARGET_COLUMNS = {TARGET_COLUMN}

PHASE_VERSION = "phase3-v1"
DEFAULT_DATASET_ID = "original"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(members_path: Path = RAW_MEMBERS, ed_path: Path = RAW_ED, care_path: Path = RAW_CARE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    members = pd.read_csv(members_path)
    ed = pd.read_csv(ed_path)
    care = pd.read_csv(care_path)
    return members, ed, care


def build_snapshot(window, members, ed, care):
    features = build_observation_features(members, ed, care, window)
    target_df, outcome_detail = build_member_target(members, ed, window)
    snapshot = features.merge(target_df, on="member_id", how="left")
    return snapshot, outcome_detail


def _outcome_state_counts(outcome_detail: pd.DataFrame) -> dict:
    counts = outcome_detail["encounter_state"].value_counts().to_dict() if not outcome_detail.empty else {}
    return {
        "total_outcome_ed_encounters": int(len(outcome_detail)),
        "potentially_avoidable": int(counts.get(POTENTIALLY_AVOIDABLE, 0)),
        "protected_or_high_acuity": int(counts.get(PROTECTED_OR_HIGH_ACUITY, 0)),
        "uncertain": int(counts.get(UNCERTAIN, 0)),
    }


def _member_overlap(snapshots: dict[str, pd.DataFrame]) -> dict:
    ids = {name: set(df["member_id"]) for name, df in snapshots.items()}
    names = list(ids)
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise[f"{a}_and_{b}"] = len(ids[a] & ids[b])
    all_three = len(ids[names[0]] & ids[names[1]] & ids[names[2]]) if len(names) == 3 else None

    positive_ids = {
        name: set(df.loc[df[TARGET_COLUMN] == 1, "member_id"]) for name, df in snapshots.items()
    }
    pairwise_positive = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise_positive[f"{a}_and_{b}"] = len(positive_ids[a] & positive_ids[b])

    return {
        "member_id_overlap_all_snapshots": pairwise,
        "member_id_overlap_all_three": all_three,
        "positive_label_overlap_pairwise": pairwise_positive,
        "note": (
            "Members legitimately appear in more than one snapshot because all "
            "three snapshots draw from the same population at different, "
            "non-overlapping points in time. This is NOT outcome leakage -- "
            "each snapshot's features and target are computed strictly from "
            "that snapshot's own window. See docs/02_UC07_AND_DATA_DESIGN.md "
            "section 9.4."
        ),
    }


def build_snapshot_metadata(
    snapshots, outcome_details, windows, raw_hashes, validation_report,
    dataset_id: str = DEFAULT_DATASET_ID, synthetic: bool = False,
) -> dict:
    per_snapshot = {}
    for name, snapshot in snapshots.items():
        window = windows[name]
        feature_cols = [c for c in snapshot.columns if c not in IDENTIFIER_COLUMNS | METADATA_COLUMNS | TARGET_COLUMNS]
        positives = int(snapshot[TARGET_COLUMN].sum())
        n = int(len(snapshot))
        per_snapshot[name] = {
            "index_date": str(window.index_date.date()),
            "observation_start": str(window.observation_start.date()),
            "observation_end_exclusive": str(window.observation_end.date()),
            "outcome_start_inclusive": str(window.outcome_start.date()),
            "outcome_end_exclusive": str(window.outcome_end.date()),
            "row_count": n,
            "positive_count": positives,
            "negative_count": n - positives,
            "prevalence": round(positives / n, 6) if n else None,
            "feature_count": len(feature_cols),
            **_outcome_state_counts(outcome_details[name]),
        }

    return {
        "phase": "Phase 3 -- Point-in-Time ML Data Pipeline",
        "version": PHASE_VERSION,
        "dataset_id": dataset_id,
        "synthetic": synthetic,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dataset_sha256": raw_hashes,
        "observation_window_days": OBSERVATION_WINDOW_DAYS,
        "outcome_window_days": OUTCOME_WINDOW_DAYS,
        "boundary_semantics": {
            "observation": "index_date - observation_window_days <= event_date < index_date",
            "outcome": "index_date <= event_date < index_date + outcome_window_days",
        },
        "snapshot_index_dates": {name: str(w.index_date.date()) for name, w in windows.items()},
        "target_column": TARGET_COLUMN,
        "target_definition": (
            "1 if the member has >=1 ED encounter classified POTENTIALLY_AVOIDABLE "
            "in the snapshot's 90-day outcome window, else 0. UNCERTAIN and "
            "PROTECTED_OR_HIGH_ACUITY outcome-window encounters never create a "
            "positive label on their own."
        ),
        "encounter_state_definition": {
            "PROTECTED_OR_HIGH_ACUITY": "red_flag==1 OR icu==1 OR admitted==1 OR major_procedure==1 OR triage_level in {1,2} (absolute precedence)",
            "POTENTIALLY_AVOIDABLE": "no safety exclusion AND triage_level in {4,5}",
            "UNCERTAIN": "no safety exclusion AND triage_level == 3",
        },
        "snapshots": per_snapshot,
        "member_overlap": _member_overlap(snapshots),
        "validation_report_summary": {"all_passed": validation_report["all_passed"]},
        "code_version": {
            "python": sys.version,
            "pandas": pd.__version__,
        },
    }


def main(
    members_path: Path = RAW_MEMBERS,
    ed_path: Path = RAW_ED,
    care_path: Path = RAW_CARE,
    output_dir: Path = DERIVED_DIR,
    dataset_id: str = DEFAULT_DATASET_ID,
    synthetic: bool = False,
) -> dict:
    """
    Build TRAIN/VALIDATION/TEST point-in-time snapshots from one
    members/ED/care-history CSV trio.

    Calling with no arguments reproduces the exact original Phase 3
    behavior (original raw CSVs at the repo root -> data/derived/,
    dataset_id="original", synthetic=False) -- every other data source is
    an explicit override via these parameters, never a separate code path.
    """
    members_path, ed_path, care_path, output_dir = Path(members_path), Path(ed_path), Path(care_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_hashes = {
        members_path.name: sha256_file(members_path),
        ed_path.name: sha256_file(ed_path),
        care_path.name: sha256_file(care_path),
    }

    members, ed, care = load_raw(members_path, ed_path, care_path)
    windows = build_all_snapshot_windows()

    snapshots = {}
    outcome_details = {}
    for name, window in windows.items():
        snapshot, detail = build_snapshot(window, members, ed, care)
        snapshots[name] = snapshot
        outcome_details[name] = detail

    validation_report = run_all_checks(
        snapshots, windows, members, ed, care,
        identifier_cols=IDENTIFIER_COLUMNS,
        metadata_cols=METADATA_COLUMNS,
        target_cols=TARGET_COLUMNS,
    )

    if not validation_report["all_passed"]:
        report_path = output_dir / "FAILED_validation_report.json"
        report_path.write_text(json.dumps(validation_report, indent=2, default=str))
        raise SystemExit(
            f"Phase 3 leakage/quality validation FAILED for dataset_id={dataset_id!r}. "
            f"See {report_path}. No snapshot files were written."
        )

    for name, snapshot in snapshots.items():
        snapshot.to_csv(output_dir / f"{name}_snapshot.csv", index=False)

    manifest = build_feature_manifest(snapshots["train"])
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    metadata = build_snapshot_metadata(snapshots, outcome_details, windows, raw_hashes, validation_report, dataset_id=dataset_id, synthetic=synthetic)
    (output_dir / "snapshot_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    validation_report_path = output_dir / "validation_report.json"
    validation_report_path.write_text(json.dumps(validation_report, indent=2, default=str))

    post_hashes = {
        members_path.name: sha256_file(members_path),
        ed_path.name: sha256_file(ed_path),
        care_path.name: sha256_file(care_path),
    }
    if post_hashes != raw_hashes:
        raise SystemExit(
            f"CRITICAL: raw dataset SHA-256 hashes changed during the build run (dataset_id={dataset_id!r}). "
            f"before={raw_hashes} after={post_hashes}"
        )

    return {
        "dataset_id": dataset_id,
        "synthetic": synthetic,
        "output_dir": output_dir,
        "snapshots": snapshots,
        "outcome_details": outcome_details,
        "windows": windows,
        "validation_report": validation_report,
        "metadata": metadata,
        "raw_hashes_before": raw_hashes,
        "raw_hashes_after": post_hashes,
    }


if __name__ == "__main__":
    result = main()
    print(f"PASS -- wrote {len(result['snapshots'])} snapshots to {result['output_dir']} (dataset_id={result['dataset_id']!r})")
    for name, snap in result["snapshots"].items():
        pos = int(snap[TARGET_COLUMN].sum())
        print(f"  {name}: rows={len(snap)} positives={pos} prevalence={pos/len(snap):.4%}")
