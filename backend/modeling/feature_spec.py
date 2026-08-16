"""
feature_spec.py
----------------
The single authoritative source of "which columns are model features" for
Phase 4. Every training and inference code path must derive its feature
list from here -- never hard-code a second, possibly-drifting feature list
elsewhere (Phase 4 spec, Step 1).

The feature list is loaded directly from data/derived/feature_manifest.json
(the Phase 3 artifact): only entries with model_candidate == true are used.
member_id, index_date, the target column, and any non-model-candidate
column (e.g. a future audit/fairness-only field) are excluded by
construction, not by a second parallel exclusion list.

Numeric vs. categorical is determined from the actual snapshot dataframe's
dtypes at load time (object dtype => categorical), not guessed from the
manifest's free-text "category" field -- this keeps preprocessing correct
even if the manifest's descriptive categories evolve.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED_DIR = REPO_ROOT / "data" / "derived"
FEATURE_MANIFEST_PATH = DERIVED_DIR / "feature_manifest.json"

TARGET_COLUMN = "future_potentially_avoidable_ed_90d"
IDENTIFIER_COLUMN = "member_id"
METADATA_COLUMN = "index_date"


def load_manifest(manifest_path: Path = FEATURE_MANIFEST_PATH) -> dict:
    return json.loads(Path(manifest_path).read_text())


def load_model_feature_columns(manifest_path: Path = FEATURE_MANIFEST_PATH) -> list[str]:
    """Ordered list of every feature_name with model_candidate == true,
    in the order it appears in the manifest (which matches the snapshot
    CSVs' own column order)."""
    manifest = load_manifest(manifest_path)
    return [f["feature_name"] for f in manifest["features"] if f["model_candidate"]]


def split_numeric_categorical(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Split `feature_columns` into (numeric, categorical) based on the
    actual dtypes present in `df` (a loaded snapshot dataframe).

    Uses pandas.api.types.is_numeric_dtype rather than a bare
    `dtype == object` check -- pandas 3.x can infer plain string columns
    (e.g. gender's "M"/"F") as its newer StringDtype rather than classic
    `object`, which a naive `== object` check would silently miss and
    misclassify as numeric."""
    categorical = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in feature_columns if c not in categorical]
    return numeric, categorical


def load_snapshot_xy(
    csv_path: Path,
    manifest_path: Path = FEATURE_MANIFEST_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load a Phase 3 snapshot CSV and split it into (X, y, member_ids) using
    ONLY the manifest's model-candidate feature columns, in manifest order.

    Never includes member_id, index_date, or the target column in X.
    """
    df = pd.read_csv(csv_path)
    feature_columns = load_model_feature_columns(manifest_path)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Snapshot {csv_path} is missing manifest model-candidate columns: {missing}")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Snapshot {csv_path} is missing the target column {TARGET_COLUMN}")

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)
    member_ids = df[IDENTIFIER_COLUMN]
    return X, y, member_ids
