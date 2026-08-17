# UC07 — Point-in-Time ML Data Pipeline (Phase 3)

**Implementation date:** 2026-08-15
**Phase:** 3 — Point-in-Time ML Data Foundation (implementation, no model training)
**Builds on:** `docs/01_PROJECT_BASELINE.md` (Phase 1), `docs/02_UC07_AND_DATA_DESIGN.md`
(Phase 2), `docs/DECISION_LOG.md`

No predictive model was trained or retrained in this phase. No multi-agent
code was written. No frontend or FastAPI behavior changed. No Docker/Azure
work was done. This phase implements exactly the point-in-time data
foundation approved in Phase 2: encounter classification, target
construction, observation-window feature engineering, three fixed
temporal snapshots, and their automated validation.

---

## 1. Executive Summary

This phase implements the Phase 2 design as working, tested code, without
training a model. A new `backend/pit/` package provides pure, testable
functions for: classifying historical ED encounters
(`encounter_classification.py`), defining the three approved snapshot
windows (`windows.py`), building the point-in-time member-level target
(`target.py`), building leakage-safe observation-window features
(`features.py`), running automated leakage/quality checks
(`validation.py`), documenting every output column
(`manifest.py`), and orchestrating the whole run
(`build_snapshots.py`). Running it against the three immutable raw CSVs
produces three derived snapshot datasets
(`data/derived/{train,validation,test}_snapshot.csv`), a feature manifest,
and snapshot metadata — all backed by 86 automated tests (17 of which run
against the real raw data) and a 17-check automated validation report that
passed on every snapshot.

Raw dataset SHA-256 hashes were verified identical before and after every
step of this phase.

---

## 2. Why the Old Pipeline Was Unsafe for UC07

Per `docs/01_PROJECT_BASELINE.md` (Phase 1) and
`docs/02_UC07_AND_DATA_DESIGN.md` (Phase 2): the legacy
`backend/train_model.py` target (`frequent_ED_user = ED_visits_365d >= 2`)
measured raw ED frequency, not avoidability; its two dominant features
(`days_since_last_ED` at 40.8% importance and an unwindowed `diagnosis_*`
crosstab at 36.3%) were computed relative to a single global reference
date shared by the label itself, producing confirmed indirect leakage; and
its train/test split was random, not temporal. This phase replaces all
three problems with code, without touching the legacy files at all (see
§21).

---

## 3. New Target Implementation

`backend/pit/target.py::build_member_target(members, ed, window)` builds
the member-level target:

```
future_potentially_avoidable_ed_90d = 1  if the member has >= 1 ED
    encounter classified POTENTIALLY_AVOIDABLE inside the snapshot's
    outcome window (index_date <= visit_date < index_date + 90d)
    else 0.
```

`UNCERTAIN` and `PROTECTED_OR_HIGH_ACUITY` outcome-window encounters never
create a positive label on their own — a member can have protected and/or
uncertain encounters in the same outcome window and still be labeled `0`
if none of their encounters clear the `POTENTIALLY_AVOIDABLE` bar (tested
explicitly in `backend/tests/test_target.py`). A member with *both* a
protected and an avoidable encounter in the same window is still
positive — the presence of ≥1 avoidable encounter is what matters; other
encounters in the same window don't cancel it out.

The function returns two separate objects: `target_df` (safe to merge into
model features — contains only `member_id` and the target column) and
`outcome_detail_df` (the classified outcome-window encounters themselves,
for reporting/validation only — it legitimately contains label-only
fields like `encounter_state`, and is never merged into a feature frame).

`backend/train_model.py`'s legacy `frequent_ED_user` target is not
imported, referenced computationally, or relied upon anywhere in this
module (verified in `backend/tests/test_legacy_isolation.py`).

---

## 4. Encounter Classification Implementation

`backend/pit/encounter_classification.py` provides both a scalar
`classify_ed_encounter(triage_level, red_flag, admitted, icu,
major_procedure)` and a vectorized `classify_ed_encounters(ed_dataframe)`,
kept in exact lockstep (tested by exhaustively comparing both over every
`triage_level × red_flag × admitted × icu × major_procedure` combination —
`test_vectorized_matches_scalar_on_random_grid`). `diagnosis` has no
parameter in either function — it is structurally impossible to pass it
in, not just avoided by convention (verified by
`test_diagnosis_not_used_in_classification`, which inspects the function
signature).

## 5. Safety Precedence

```
IF red_flag == 1 OR icu == 1 OR admitted == 1 OR major_procedure == 1
   OR triage_level IN {1, 2}
    -> PROTECTED_OR_HIGH_ACUITY          (absolute precedence)
ELSE IF triage_level IN {4, 5}
    -> POTENTIALLY_AVOIDABLE
ELSE  (triage_level == 3)
    -> UNCERTAIN
```

All 9 mandated Phase 3 test cases pass, including the 4 cases specifically
designed to prove exclusions override low-acuity triage levels (triage 5 +
admitted, triage 5 + red_flag, triage 4 + ICU, triage 4 + major_procedure —
all correctly classify `PROTECTED_OR_HIGH_ACUITY` despite their
lower-acuity triage level). See
`backend/tests/test_encounter_classification.py`.

---

## 6. Temporal Architecture

Three fixed, non-overlapping snapshots, exactly as locked in Phase 2:

| Snapshot | Index date | Observation start | Outcome end (exclusive) |
|---|---|---|---|
| TRAIN | 2025-10-05 | 2025-01-08 | 2026-01-03 |
| VALIDATION | 2026-01-03 | 2025-04-08 | 2026-04-03 |
| TEST | 2026-04-03 | 2025-07-07 | 2026-07-02 |

Index dates are literal `pd.Timestamp(...)` constants in
`backend/pit/windows.py` — never derived from any dataset column
(verified statically in `test_snapshot_index_dates_are_fixed_literals_not_derived`
and behaviorally at runtime by `check_no_global_max_date_index`, which
confirms no snapshot's index date equals the dataset's actual global max
ED visit date).

---

## 7. Observation/Outcome Boundaries

Locked convention, implemented exactly:

```
OBSERVATION (history, half-open):
    index_date - 270 days  <=  event_date  <  index_date

OUTCOME (future, half-open):
    index_date  <=  event_date  <  index_date + 90 days
```

Both windows share `index_date` as their common boundary — included in
outcome, excluded from observation — so they are disjoint by construction.
This is verified three ways: (1) unit tests sweeping a full calendar range
around the index date and asserting no date ever satisfies both masks
(`test_observation_and_outcome_never_overlap_for_any_date`); (2) an
explicit runtime check against the real raw ED data in every pipeline run
(`check_no_observation_outcome_overlap`, part of the 17-check validation
report); (3) explicit boundary-day tests (visit exactly on `index_date`
counts as outcome; visit exactly on `outcome_end` is excluded;
`test_encounter_exactly_at_index_date_counts_as_outcome` /
`test_encounter_on_outcome_end_date_is_excluded`).

All date handling uses `pd.Timestamp`/`pd.Timedelta` throughout — no raw
string date comparison anywhere in `backend/pit/`.

---

## 8. Point-in-Time Feature Engineering

`backend/pit/features.py::build_observation_features(members, ed, care,
window)` builds one row per member containing only static member fields
and longitudinal features computed strictly from
`observation_start <= event_date < index_date` for that snapshot. No
per-encounter severity field (`triage_level`, `red_flag`, `admitted`,
`icu`, `major_procedure`, `diagnosis`, `cost`) of any encounter — past or
future — ever appears in the output; only aggregated *counts and recency*
of prior encounters do.

**Missing-value representation:** every recency feature
(`days_since_prior_*`) uses an explicit `NaN` when no qualifying event
falls within the 270-day observation window, paired with a companion
`has_prior_*` 0/1 flag — not a sentinel/magic number, which a model could
otherwise misinterpret as a legitimate large value. This was a
Phase-3-level implementation decision (recorded in
`docs/DECISION_LOG.md`) that does not alter any approved Phase 2
methodology.

**Recency is capped at the observation window**, not looked up
indefinitely into the past: if a member's last ED visit was more than 270
days before their snapshot's index date, `days_since_prior_ed` is `NaN`
(not "no prior event ever," just "none within this snapshot's defined
observation window") — consistent with Step 5's instruction that every
longitudinal feature must be computed strictly from the observation
window.

---

## 9. Included Feature Groups

62 columns total per snapshot (2 identifier/metadata + 58 features + 1
target — see `data/derived/feature_manifest.json` for the authoritative,
per-column breakdown):

| Group | Count | Examples |
|---|---:|---|
| Identifier / metadata | 2 | `member_id`, `index_date` |
| Demographic | 2 | `age`, `gender` |
| Chronic condition | 8 | `diabetes`…`ckd`, `num_chronic_conditions`, `clinical_burden` |
| Access | 5 | `transportation_barrier`, `telehealth_available`, `pcp_distance_miles`, `urgent_care_distance_miles`, `access_burden` |
| Prior ED utilization | 12 | `prior_ed_count_{30,90,180,270}d`, `prior_potentially_avoidable_ed_count_{30,90,180,270}d`, `prior_protected_ed_count_{90,270}d`, `prior_uncertain_ed_count_{90,270}d` |
| Prior ED recency | 6 | `days_since_prior_ed` / `has_prior_ed`, and the same pair for `potentially_avoidable_ed` and `protected_ed` |
| Velocity | 2 | `ed_utilization_velocity_30_over_180`, `potentially_avoidable_ed_velocity_90_over_270` |
| Prior care utilization | 16 | `prior_{pcp,urgent_care,telehealth,care_management}_count_{30,90,180,270}d` |
| Prior care recency | 8 | `days_since_prior_{pcp,urgent_care,telehealth,care_management}` + matching `has_prior_*` flags |
| Target | 1 | `future_potentially_avoidable_ed_90d` |

**Judgment call on subcount windows:** total and potentially-avoidable ED
counts get the full 30/90/180/270-day ladder (the primary
utilization/navigation signals); protected and uncertain ED subcounts are
provided at 90d/270d only, to avoid a proliferation of near-collinear,
low-count features for states that are not themselves the target signal
— documented per-feature in the manifest.

**Velocity formulas** (denominators floored at 1 to avoid divide-by-zero):
```
ed_utilization_velocity_30_over_180        = prior_ed_count_30d / max(prior_ed_count_180d, 1)
potentially_avoidable_ed_velocity_90_over_270 = prior_potentially_avoidable_ed_count_90d / max(prior_potentially_avoidable_ed_count_270d, 1)
```

---

## 10. Excluded Feature Groups

- **All outcome-window encounter fields** — `triage_level`, `red_flag`,
  `admitted`, `icu`, `major_procedure`, `diagnosis`, `cost` — for the
  encounter(s) being predicted. These are label-only by design (Phase 2
  §12) and structurally cannot appear in `build_observation_features`'s
  output (it never reads the outcome window at all).
- **`diagnosis_*` crosstab features** — see §11.
- **Legacy `frequent_ED_user`** and any `ED_visits_{30,90,180,365}d`-named
  columns — never computed anywhere in `backend/pit/`.
- **`num_chronic_conditions` vs. the 6 individual chronic flags vs.
  `clinical_burden`** — all three are retained (per Phase 2 approval,
  which explicitly kept this redundancy rather than treating it as
  leakage), but flagged in the manifest description as a known
  multicollinearity note for future modeling.

---

## 11. Why Diagnosis Predictors Were Excluded

Phase 1 found the legacy model's unwindowed `diagnosis_*` crosstab
leaking ED-utilization volume (~36% of that model's feature importance,
because it counted *all-time* diagnosis-tagged ED visits regardless of
any time window). Phase 2 additionally verified, by direct crosstab
inspection, that `diagnosis` category carries no measurable acuity signal
in this dataset — near-identical `triage_level`/`red_flag`/`admitted`/
`icu`/`major_procedure` rates across all 14 categories. Per the Phase 3
spec, diagnosis-derived predictors are excluded from this baseline feature
set by default; the manifest documents diagnosis as *"a candidate for
future controlled experimentation, excluded from the Phase 3 baseline
feature set to reduce leakage/reconstruction risk"* — not implemented,
not present anywhere in `data/derived/*.csv` (verified by
`test_no_diagnosis_columns_produced` and the runtime
`diagnosis_crosstab_absent` check on every real snapshot).

---

## 12. Leakage Controls

Two layers, both automated and both run against the real raw data on
every pipeline execution:

1. **Structural** — `build_observation_features` and `build_member_target`
   physically cannot read outside their respective window
   (`in_observation_window` / `in_outcome_window` masks are applied before
   any aggregation), so there is no code path by which a future field
   could reach a feature.
2. **Verification** — `backend/pit/validation.py::run_all_checks` runs 17
   checks per snapshot (structural boundary checks, forbidden-column
   checks, identifier/metadata/target-exclusion checks, legacy-target and
   diagnosis-crosstab absence checks, a global-max-date regression check,
   an *independent reconciliation* that recomputes
   `prior_ed_count_270d`, `prior_ed_count_30d`,
   `prior_potentially_avoidable_ed_count_270d`, and the target column
   directly from raw data using only the window-boundary constants — not
   by calling back into `features.py`/`target.py` — and asserts exact
   equality against what the pipeline actually produced) plus data-quality
   checks (duplicate/missing member IDs, infinite values, negative counts,
   negative recency, cross-snapshot schema consistency). **All 17 checks
   passed on all three real snapshots** (`data/derived/validation_report.json`).
   `build_snapshots.py` refuses to write any snapshot file if any check
   fails.

The reconciliation checks in particular were verified to actually catch
bugs, not just always pass: `test_reconciliation_fails_on_corrupted_count_column`
and `test_reconciliation_fails_on_corrupted_target` inject a deliberate
bug into a snapshot and confirm the check reports failure.

---

## 13. Snapshot Generation

`backend/pit/build_snapshots.py::main()`:
1. Computes SHA-256 hashes of the three raw CSVs (before).
2. Loads them read-only.
3. Builds all three snapshot windows, then features + target for each.
4. Runs the full 17-check validation report; aborts (writing only a
   `FAILED_validation_report.json`, no snapshot CSVs) if anything fails.
5. Writes `train_snapshot.csv`, `validation_snapshot.csv`,
   `test_snapshot.csv`, `feature_manifest.json`, `snapshot_metadata.json`,
   and `validation_report.json` to `data/derived/`.
6. Recomputes the three raw CSV hashes (after) and hard-fails if they
   changed during the run.

Run directly via `python backend/pit/build_snapshots.py`, or indirectly
through the `pipeline_result` fixture in
`backend/tests/test_pipeline_integration.py`. Both were executed for this
phase; results agree.

---

## 14. Snapshot Statistics

| Snapshot | Rows | Positives | Prevalence | Feature count | Outcome-window ED encounters (total / avoidable / protected / uncertain) |
|---|---:|---:|---:|---:|---|
| TRAIN | 10,000 | 904 | 9.04% | 59 | 2,238 / 950 / 649 / 639 |
| VALIDATION | 10,000 | 958 | 9.58% | 59 | 2,325 / 997 / 652 / 676 |
| TEST | 10,000 | 908 | 9.08% | 59 | 2,275 / 947 / 645 / 683 |

(Feature count = 62 total columns minus `member_id`, `index_date`, and the
target column.)

---

## 15. Target Prevalence — Consistency with Phase 2

Phase 2's exploratory estimate (computed informally, before this
implementation) was 9.07% / 9.57% / 9.09% for train/validation/test. The
actual, code-verified figures are 9.04% / 9.58% / 9.08% — consistent to
within a handful of members per snapshot, **not forced to match exactly**.

The small delta is fully explained, not a bug: Phase 2's exploratory
script used an approximate `(index_date, outcome_end]` window (excluding
the index date, including the day after `outcome_end`), while the locked
Phase 3 specification's convention is `[index_date, outcome_end)`
(including the index date, excluding `outcome_end`) — a one-day shift at
each boundary. Direct inspection shows 19–23 ED visits fall on each of the
four relevant boundary dates (2025-10-05, 2026-01-03, 2026-04-03,
2026-07-02), which fully accounts for the 1–4 member difference in
positive counts per snapshot between the Phase 2 estimate and this
implementation's exact figures. This implementation follows the Phase 3
spec's boundary convention exactly, as required.

---

## 16. Feature Manifest

`data/derived/feature_manifest.json` documents every one of the 62
columns: `feature_name`, `source_dataset`, `category`, `temporal_window`,
`static_or_longitudinal`, `model_candidate`, `leakage_status`, and a
human-readable `description`. It separately lists `identifier_columns`
(`member_id`), `metadata_columns` (`index_date`), `target_columns`
(`future_potentially_avoidable_ed_90d`), `fairness_audit_columns`
(`gender`), and the two `excluded_feature_groups` (diagnosis crosstabs;
outcome-window encounter fields) with their exclusion rationale. Zero
columns fell through to the manifest's `"unclassified"` catch-all category
(verified by `test_no_unclassified_manifest_entries` against the real
generated manifest).

---

## 17. Missing-Value Behavior

- **Static member fields**: none missing (Phase 1 confirmed 0 missing
  values in `raw_members.csv`); no imputation performed in this phase.
- **Prior utilization counts** (`prior_*_count_*d`): `0` when no
  qualifying event exists in the window — a true, meaningful zero, not a
  missing-value placeholder.
- **Recency features** (`days_since_prior_*`): explicit `NaN` when no
  qualifying event falls within the observation window, always paired
  with a `has_prior_*` 0/1 flag (§8). No sentinel numeric value (e.g.
  `-1`, `9999`) is ever used.
- **Velocity features**: never `NaN`/infinite — denominators are floored
  at 1 by construction (verified by `check_no_infinite_values`/
  `check_no_negative_counts` passing on every real snapshot).

---

## 18. Member Overlap

Every member appears in **all three** snapshots by design — this phase
scores the same 10,000-member population at three different points in
time, so member-ID overlap is trivially 100% across every pair and across
all three simultaneously (`data/derived/snapshot_metadata.json`:
`member_overlap.member_id_overlap_all_snapshots`). This is expected and
was not removed, per the Phase 3 spec's explicit instruction not to strip
members merely for appearing in multiple snapshots.

The informative overlap metric is among **positive-labeled** members
only: TRAIN∩VALIDATION = 66, VALIDATION∩TEST = 67, TRAIN∩TEST = 69 (out of
~900–960 positives per snapshot, ≈7%) — closely matching Phase 2's
exploratory estimate of 66–72. This is documented, per Phase 2 §9.4, as an
accepted, monitored non-independence between splits, not outcome leakage
(each snapshot's label is still computed strictly from its own
non-overlapping outcome window).

---

## 19. Automated Tests

86 tests across 7 files in `backend/tests/`, all passing:

| File | Focus | Count |
|---|---|---:|
| `test_encounter_classification.py` | All 9 mandated cases + safety-precedence + exhaustive scalar/vectorized agreement + diagnosis-exclusion-by-signature | 16 |
| `test_windows.py` | Boundary arithmetic, half-open semantics, datetime typing, disjointness sweep, approved index dates | 9 |
| `test_target.py` | Positive/negative label construction, mixed protected+avoidable, observation-window isolation, boundary-day edge cases | 10 |
| `test_features.py` | Window filtering, recency-vs-index-date (not global max), explicit NaN+flag, care-history type mapping, velocity formula, no diagnosis/forbidden columns | 12 |
| `test_validation.py` | Every check exercised on both clean and intentionally-corrupted data, including reconciliation bug-injection | 21 |
| `test_pipeline_integration.py` | Full real-data run: hash immutability, schema/target/prevalence, manifest completeness, metadata completeness | 15 |
| `test_legacy_isolation.py` | No legacy formula/column computed, no import of legacy backend modules, index dates are fixed literals | 4 |

`pytest` was added to `.venv` as a dev/test dependency (explicitly
authorized by the Phase 3 spec: *"If no test framework exists, use pytest
unless introducing it creates a serious compatibility problem"*) — no
production/ML library was added.

---

## 20. Raw Dataset Hash Verification

| File | SHA-256 |
|---|---|
| `raw_members.csv` | `b94df89ed042a8feaa1bb46d7939e124fb9f6b03308b11da045412a427b78c46` |
| `raw_ed_visits.csv` | `f8db1839fb7966c4230c771252a3b935c318d0838de9258dc29de42d042f5d47` |
| `raw_care_history.csv` | `358d3033faa4e0529aed834cd8847f72d0b5d4ca51fa76748523fab790c81657` |

Verified identical: (1) before any Phase 3 work began; (2) automatically,
inside every `build_snapshots.py` run, immediately after writing derived
files (hard-fails the run if changed); (3) independently in
`test_pipeline_integration.py::test_raw_dataset_hashes_match_recorded_values`
and `test_raw_hashes_unchanged_before_and_after_pipeline_run`; (4) by a
final standalone verification after all implementation work completed.
**Result: PASS — all three hashes identical throughout.**

---

## 21. Legacy Pipeline Isolation

`backend/feature_engineering.py`, `backend/train_model.py`,
`backend/predict.py`, `backend/main.py`, and `backend/ed_risk_model.pkl`
were **not read for modification, not edited, and not deleted** in this
phase (only re-read for reference during Phase 1). `backend/pit/` imports
none of them (verified by
`test_pit_package_does_not_import_legacy_backend_modules`), computes
neither the legacy formula (`ED_visits_365d >= 2`) nor a
`frequent_ED_user` column anywhere (verified by
`test_legacy_target_formula_never_computed` /
`test_legacy_target_column_never_assigned`), and the two pipelines share
no code path. The legacy model remains exactly as it was at the end of
Phase 1 — untouched, unretrained, still loadable — clearly superseded in
documentation but not deleted, per the Phase 3 spec.

---

## 22. Known Limitations

1. **Positive-label overlap across snapshots (~7%)** is accepted, not
   eliminated (Phase 2 §9.4/§24; §18 above) — a fully member-disjoint
   design remains a candidate future refinement.
2. **`UNCERTAIN` encounters never drive a positive label** (Phase 2 §6.2)
   — intentionally conservative, likely undercounts achievable recall;
   unchanged in this phase, as Phase 2 methodology is locked.
3. **Diagnosis excluded from this baseline** (§11) — its value as a
   properly-windowed feature remains unverified; a documented candidate
   for controlled future experimentation, not resolved here.
4. **Cold-start members** — a member with zero prior ED/care history
   before their snapshot's index date receives `0` counts and `NaN`
   recency + `has_prior_*=0` flags; their real-world risk is unvalidated
   by this pipeline (no model exists yet to check against).
5. **`gender` remains a candidate predictor**, flagged in the manifest's
   `fairness_audit_columns` for subgroup validation once a model exists —
   not resolved in this data-only phase.
6. **`num_chronic_conditions` / individual chronic flags / `clinical_burden`
   redundancy** is retained per Phase 2 approval and flagged, not removed.
7. Only one 547-day data era exists in the source datasets, so all three
   snapshots draw from the same historical period rather than genuinely
   independent years (Phase 2 §24, carried forward unchanged).

---

## 23. Phase 4 Readiness Assessment

The data foundation required to train a model safely now exists and is
independently verified: three schema-consistent, leakage-checked,
point-in-time snapshot datasets with a documented feature manifest and
metadata. Phase 4 (not started, not authorized by this phase) would need
to: select a modeling algorithm, fit preprocessing on TRAIN only, select
risk-tier thresholds and any Care Management trigger cutoffs on
VALIDATION only, evaluate once on TEST, package the resulting artifact
with the feature manifest and version metadata, and only then proceed to
agent implementation, API/frontend integration, testing expansion,
Docker, and Azure — in that order, per
`docs/02_UC07_AND_DATA_DESIGN.md` §25.

---

## 24. Architecture Diagram

```
                         RAW DATASETS (immutable)
        raw_members.csv   raw_ed_visits.csv   raw_care_history.csv
                    |            |                    |
                    +------------+--------+------------+
                                 |
                     backend/pit/windows.py
              (fixed index_date per snapshot: TRAIN/VAL/TEST)
                                 |
              +------------------+------------------+
              |                                     |
              v                                     v
   270-DAY OBSERVATION WINDOW               90-DAY OUTCOME WINDOW
   observation_start <= date < index_date   index_date <= date < outcome_end
              |                                     |
              v                                     v
  backend/pit/features.py               backend/pit/encounter_classification.py
  build_observation_features()          classify_ed_encounters()
   - static member fields                          |
   - prior ED utilization/recency                   v
   - prior care utilization/recency        POTENTIALLY_AVOIDABLE /
   - velocity                              PROTECTED_OR_HIGH_ACUITY /
   (diagnosis_* excluded, sec 11)          UNCERTAIN
              |                                     |
              |                                     v
              |                        backend/pit/target.py
              |                        build_member_target()
              |                        future_potentially_avoidable_ed_90d
              |                                     |
              +------------------+------------------+
                                 |
                                 v
                    backend/pit/build_snapshots.py
              merge features + target, one row per member
                                 |
                                 v
                backend/pit/validation.py :: run_all_checks()
             17 leakage/quality checks -- MUST all pass to proceed
                                 |
                                 v
                          SNAPSHOT DATASET
        data/derived/{train,validation,test}_snapshot.csv
        data/derived/feature_manifest.json
        data/derived/snapshot_metadata.json
        data/derived/validation_report.json
```

---

## 25. Statement of Scope

**No predictive model was trained, retrained, calibrated, or compared in
this phase.** No risk thresholds were chosen. `backend/ed_risk_model.pkl`
was not modified. No multi-agent code was implemented. No FastAPI route
or frontend behavior changed. No Docker or Azure work was done. The three
raw source datasets were verified byte-identical (SHA-256) before and
after this phase's entire implementation, testing, and pipeline-execution
work.
