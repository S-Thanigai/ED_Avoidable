"""
Step 14 (Legacy Pipeline Isolation): the new backend/pit/ package must
never COMPUTE or IMPORT the legacy `frequent_ED_user` (ED_visits_365d >= 2)
target, and must never derive a snapshot's index_date from the dataset's
own global max date.

Note: `validation.py` legitimately contains the string "frequent_ED_user"
as a constant it checks OTHER data FOR THE ABSENCE of (LEGACY_TARGET_COLUMN),
and `target.py`'s module docstring legitimately documents, in prose, that
it does not reuse that target. Neither of those is "relying on the old
target logic" -- so these checks look for the actual dangerous patterns
(a live assignment/computation of the legacy column, or the legacy
formula itself), not mere textual mentions.
"""
from pathlib import Path

PIT_DIR = Path(__file__).resolve().parent.parent / "pit"
PIT_SOURCE_FILES = sorted(PIT_DIR.glob("*.py"))

# The exact legacy formula from backend/train_model.py:
#   features["frequent_ED_user"] = (features["ED_visits_365d"] >= 2).astype(int)
LEGACY_FORMULA_FRAGMENTS = [
    'ED_visits_365d"] >= 2',
    "ED_visits_365d'] >= 2",
]

# A live assignment that would CREATE a frequent_ED_user column/variable
# (as opposed to referencing the string as a detection constant).
LEGACY_TARGET_ASSIGNMENT_FRAGMENTS = [
    'frequent_ED_user"] =',
    "frequent_ED_user'] =",
    "frequent_ED_user =",
]


def test_legacy_target_formula_never_computed():
    hits = {}
    for path in PIT_SOURCE_FILES:
        text = path.read_text()
        found = [frag for frag in LEGACY_FORMULA_FRAGMENTS if frag in text]
        if found:
            hits[path.name] = found
    assert hits == {}, f"legacy target formula (ED_visits_365d >= 2) found in backend/pit source: {hits}"


def test_legacy_target_column_never_assigned():
    """`frequent_ED_user` may appear as a detection-only string constant
    (validation.py's LEGACY_TARGET_COLUMN, used to check its ABSENCE from
    snapshots) or in documentation prose (target.py's docstring) -- but no
    file may ever assign/compute a column or variable by that name."""
    hits = {}
    for path in PIT_SOURCE_FILES:
        text = path.read_text()
        found = [frag for frag in LEGACY_TARGET_ASSIGNMENT_FRAGMENTS if frag in text]
        if found:
            hits[path.name] = found
    assert hits == {}, f"legacy target column assignment found in backend/pit source: {hits}"


def test_pit_package_does_not_import_legacy_backend_modules():
    """backend/pit must not import backend/feature_engineering.py,
    backend/train_model.py, or backend/predict.py -- the legacy pipeline
    and the new pipeline must stay isolated from each other."""
    forbidden_imports = ["feature_engineering", "train_model", "predict"]
    hits = {}
    for path in PIT_SOURCE_FILES:
        text = path.read_text()
        found = [mod for mod in forbidden_imports if f"import {mod}" in text or f"from {mod}" in text]
        if found:
            hits[path.name] = found
    assert hits == {}, f"backend/pit imports legacy backend modules: {hits}"


def test_windows_module_defines_no_dataset_dependent_index_date():
    """windows.py (the sole owner of index-date definitions) must contain
    zero `.max()` calls of any kind -- it should never need to inspect a
    dataset to know what the index dates are. (features.py legitimately
    uses per-member, already-observation-window-filtered `.max()` calls
    for recency calculations -- that is a different, safe operation,
    covered instead by the runtime `check_no_global_max_date_index` /
    reconciliation checks in test_validation.py and
    test_pipeline_integration.py.)"""
    source = (PIT_DIR / "windows.py").read_text()
    assert ".max()" not in source


def test_snapshot_index_dates_are_fixed_literals_not_derived():
    """windows.py must define the three approved index dates as literal
    pd.Timestamp(...) constants, not as any expression derived from a
    dataframe."""
    source = (PIT_DIR / "windows.py").read_text()
    assert 'pd.Timestamp("2025-10-05")' in source
    assert 'pd.Timestamp("2026-01-03")' in source
    assert 'pd.Timestamp("2026-04-03")' in source
