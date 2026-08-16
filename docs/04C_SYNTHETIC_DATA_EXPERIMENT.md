# UC07 — Synthetic Dataset Integration & PIT Rebuild (Phase 4C)

**Implementation date:** 2026-08-15
**Phase:** 4C — Synthetic Dataset Integration & PIT Rebuild (data pipeline only; no model training)
**Builds on:** `docs/01_PROJECT_BASELINE.md` … `docs/04B_MODEL_IMPROVEMENT.md`, `docs/DECISION_LOG.md`

**This phase does not train any model.** It safely introduces a second,
explicitly synthetic dataset trio, preserves every original-data
artifact byte-for-byte, and reuses the exact same Phase 2/3 point-in-time
methodology (unchanged target, encounter classification, windows, and
index dates) to build a parallel, clearly-labeled set of synthetic
snapshots. Model retraining against these snapshots is deferred to the
next prompt.

> **Scientific framing, stated once here and never contradicted anywhere
> in this document:** *"The synthetic experiment tests whether the UC07
> pipeline can learn meaningful prospective patterns when such
> relationships exist in the input data."* It does **not** and **cannot**
> demonstrate that the model works in real healthcare settings. Any
> stronger performance observed later on synthetic data must never be
> interpreted as clinical validation.

---

## 1. Why the Synthetic Experiment Was Introduced

Phase 4 and 4B established, through extensive and honest experimentation,
that `uc07-risk-v1` has modest discrimination (TEST ROC-AUC 0.5747,
PR-AUC 0.1111) on the original dataset, and that no amount of additional
leakage-safe feature engineering against that same dataset materially
improved it (Phase 4B's best finding: +0.002 PR-AUC, far short of the
promotion bar). This leaves an open question the original data alone
cannot answer: **is the UC07 point-in-time pipeline itself capable of
recovering strong prospective signal when such signal is actually present
in the input data**, or is there a structural reason (feature design,
modeling approach, target construction) that would suppress it
regardless? A dataset explicitly constructed to contain clearer
input→outcome relationships is the direct way to test that question,
without touching or reinterpreting the real dataset.

---

## 2. Why Original-Data V1 Is Being Preserved

`uc07-risk-v1` and every Phase 3/4/4B artifact built from the original
three datasets represent the project's only evidence about real
(original) data behavior. Nothing about introducing synthetic data
changes that evidence's validity or relevance. Per the non-negotiable
rules for this phase, the original three raw datasets, the original
Phase 3 derived snapshots, and the `uc07-risk-v1` artifact are all
preserved unmodified — verified by SHA-256 hash comparison before and
after every step of this phase (§6, §17).

---

## 3. Original vs. Synthetic Dataset Distinction

| | Original | Synthetic |
|---|---|---|
| Location | `raw_members.csv`, `raw_ed_visits.csv`, `raw_care_history.csv` (repo root) | `data/synthetic/raw_members.csv`, `raw_ed_visits.csv`, `raw_care_history.csv` |
| Status | Immutable, unchanged since Phase 1 | Immutable for this phase (never written to) |
| Derived output | `data/derived/` (live) + `data/derived/original/` (permanent archive, this phase) | `data/derived/synthetic/` only |
| `dataset_id` in metadata | `"original"` | `"synthetic_uc07_v1"` |
| `synthetic` flag in metadata | `false` | `true` |
| Model artifact | `backend/models/uc07_risk_v1_model.joblib` | none yet (next phase) |

## 4. Synthetic Data Is Demonstration Data, Not Real-World Evidence

The synthetic dataset was explicitly constructed for this experiment. It
is schema-compatible with the original dataset (§7) but is **not** a
sample of real members, real encounters, or real care patterns. Every
statistic, prevalence figure, and relationship reported in this document
describes the synthetic file's own internal structure — nothing here is a
claim about actual patient populations or actual healthcare utilization.

---

## 5. Directory Structure

```
raw_members.csv, raw_ed_visits.csv, raw_care_history.csv    <- original, repo root, unchanged
data/synthetic/raw_members.csv, raw_ed_visits.csv, raw_care_history.csv  <- synthetic, unchanged

data/derived/                          <- live original-experiment output (unchanged by this phase)
data/derived/original/                 <- NEW: permanent byte-identical archive of the above
data/derived/synthetic/                <- NEW: synthetic-experiment output ONLY

backend/pit/build_snapshots.py         <- refactored to accept configurable paths (still ONE implementation)
backend/pit/features.py, target.py,    <- UNCHANGED (reused as-is for both datasets)
  windows.py, encounter_classification.py

artifacts/synthetic_experiment/        <- NEW: descriptive comparison + signal-check reports
```

No second copy of the feature/target/window logic exists — `features.py`,
`target.py`, `windows.py`, and `encounter_classification.py` were not
modified at all in this phase; both experiments call the identical
functions.

---

## 6. Dataset Hashes

**Original raw (unchanged, matches every prior phase's recorded value):**

| File | SHA-256 |
|---|---|
| `raw_members.csv` | `b94df89ed042a8feaa1bb46d7939e124fb9f6b03308b11da045412a427b78c46` |
| `raw_ed_visits.csv` | `f8db1839fb7966c4230c771252a3b935c318d0838de9258dc29de42d042f5d47` |
| `raw_care_history.csv` | `358d3033faa4e0529aed834cd8847f72d0b5d4ca51fa76748523fab790c81657` |

**Synthetic raw (recorded at the start of this phase; verified unchanged at the end):**

| File | SHA-256 |
|---|---|
| `data/synthetic/raw_members.csv` | `00cb4023eb20876fd9b9cd2b3b3e283c8e6681f1452a6c3e9cbfda37f0bd2373` |
| `data/synthetic/raw_ed_visits.csv` | `bb3c9505a836b8c70813aa2fdd62f628bd871f657fa1dfca1330799d27ce88c0` |
| `data/synthetic/raw_care_history.csv` | `20fdcb836f6abbbd1b1b70d7c1f7cd2279f5c519251322944b0ca7109a66db1a` |

**Archived original-experiment derived artifacts** (`data/derived/original/`, copied byte-for-byte from the pre-existing `data/derived/`, never regenerated):

| File | SHA-256 |
|---|---|
| `train_snapshot.csv` | `1b6799904302398d95b478ca2a1e33d0b206fcc1983151b743f01cdbb7a534eb` |
| `validation_snapshot.csv` | `a19dad00c4a8329074f7dcba94357506fd004ff7798c82d7d8b7313f13c9b70f` |
| `test_snapshot.csv` | `1d4e8b22ede975cdad43379dd0566d38b597e270e8dbd1fcf3a1d85d8989ac1a` |
| `feature_manifest.json` | `fdd661653638649e3da02087bbdcd95a70f754f69e3218e0155cf8368cbd657a` |
| `snapshot_metadata.json` | `428e31b408334392df66929bcd9594074290b0edd2c9854022b158b986d59c7e` |
| `validation_report.json` | `e783ec7fe1c518daa3d959b6300504e2acbe18bfb12a54f3e395db5b125710e0` |

**Note on the live `data/derived/snapshot_metadata.json`:** this file is
regenerated (with identical content) whenever the existing Phase 3 test
suite re-invokes `build_snapshots.main()` with its default arguments —
that has always been true since Phase 3. This phase's refactor adds two
new, backward-compatible fields (`dataset_id: "original"`,
`synthetic: false`) to that regeneration; the archived copy above is
never regenerated by any code path and is the permanent, byte-for-byte
historical record. `backend/tests/test_synthetic_pipeline.py::
test_original_metadata_core_content_unchanged` verifies every
pre-existing field's *value* (not just the file's hash) is unchanged.

---

## 7. Schema Compatibility

Verified directly (not assumed) before any pipeline code ran:

| Check | Result |
|---|---|
| Column names identical (all 3 files) | PASS |
| Column dtypes identical (all 3 files) | PASS |
| `gender` categories | `{M, F}` in both |
| `triage_level`, `red_flag`, `admitted`, `icu`, `major_procedure` value sets | identical in both |
| `diagnosis` category set (14 categories) | identical in both |
| `care_type` category set (4 categories) | identical in both |
| Duplicate `member_id`/`visit_id`/`care_id` | 0 in both |
| Missing values | 0 in both |
| Orphan foreign keys (ED/care `member_id` not in `members`) | 0 in both |
| `visit_date` range | `2025-01-01` → `2026-07-02` in both |

**Result: full schema compatibility. No incompatibility was found; the
pipeline was not modified to force compatibility.** Row counts and
underlying relationships differ (by design — see §9), which is exactly
what this experiment is meant to explore, not a compatibility problem.

---

## 8. PIT Pipeline Refactor

`backend/pit/build_snapshots.py::main()` now accepts
`members_path`, `ed_path`, `care_path`, `output_dir`, `dataset_id`, and
`synthetic` — all with defaults that exactly reproduce the pre-Phase-4C
behavior (original raw CSVs → `data/derived/`, `dataset_id="original"`,
`synthetic=False`) when called with no arguments. `load_raw()` and
`build_snapshot_metadata()` were extended the same way. No other function
changed — `build_snapshot()`, `_outcome_state_counts()`,
`_member_overlap()`, and every function in `features.py`/`target.py`/
`windows.py`/`encounter_classification.py` are untouched, so both
experiments run through the identical point-in-time implementation.

---

## 9. Synthetic Descriptive Statistics

Full table: `artifacts/synthetic_experiment/original_vs_synthetic_comparison.csv`.

| Statistic | Original | Synthetic |
|---|---:|---:|
| Members | 10,000 | 10,000 |
| ED encounters | 13,798 | 15,108 |
| Care-history encounters | 26,289 | 27,104 |
| Date range | 2025-01-01 → 2026-07-02 | 2025-01-01 → 2026-07-02 |
| Members with ≥1 ED visit | 84.47% | 67.29% |
| ED visits per member-with-ED | 1.633 | 2.245 |
| Mean chronic-condition burden | 0.837 | 0.899 |
| Transportation barrier rate | 14.28% | 16.22% |
| Telehealth available rate | 77.47% | 75.67% |
| Mean PCP distance (miles) | 3.69 | 7.24 |
| Mean urgent-care distance (miles) | 2.69 | 5.71 |
| PCP visits/member | 1.238 | 0.956 |
| Urgent Care visits/member | 0.432 | 0.432 |
| Telehealth visits/member | 0.698 | 0.954 |
| Care Management visits/member | 0.261 | 0.368 |

**Synthetic members are, on average, further from care (roughly double
the mean PCP/urgent-care distance) and split into a smaller ED-using
subgroup that uses the ED more intensively**, with correspondingly higher
telehealth and care-management engagement. This is a materially
different access/utilization "world" than the original data, by
construction.

## Synthetic Encounter Classifications

| State | Original (% of all ED) | Synthetic (% of all ED) |
|---|---:|---:|
| POTENTIALLY_AVOIDABLE | 42.27% (5,832) | **62.66% (9,467)** |
| PROTECTED_OR_HIGH_ACUITY | 28.77% (3,969) | 18.61% (2,811) |
| UNCERTAIN | 28.97% (3,997) | 18.73% (2,830) |

The synthetic ED encounters skew substantially toward lower-triage
(4–5), non-excluded visits — the encounter classification rule itself
(`backend/pit/encounter_classification.py`) was not changed; it simply
classifies a differently-distributed input differently, exactly as
designed.

---

## 10. Synthetic Target Prevalence

| Snapshot | Rows | Positives | Prevalence | Outcome ED encounters (avoidable/protected/uncertain) |
|---|---:|---:|---:|---|
| TRAIN | 10,000 | 1,194 | **11.94%** | 2,332 (1,511 / 405 / 416) |
| VALIDATION | 10,000 | 1,174 | **11.74%** | 2,415 (1,510 / 429 / 476) |
| TEST | 10,000 | 1,307 | **13.07%** | 2,748 (1,753 / 500 / 495) |

Not forced to match the original's ~9% — reported as-is. The synthetic
prevalence is naturally higher (~12–13% vs. ~9%), consistent with §9's
finding that synthetic ED encounters skew more heavily toward
`POTENTIALLY_AVOIDABLE`.

---

## 11. Point-in-Time Snapshot Construction

Identical methodology to Phase 3, unchanged: target
`future_potentially_avoidable_ed_90d`; encounter classification with
safety exclusions taking absolute precedence
(`red_flag==1 OR icu==1 OR admitted==1 OR major_procedure==1 OR
triage_level in {1,2}` → `PROTECTED_OR_HIGH_ACUITY`; else `triage_level
in {4,5}` → `POTENTIALLY_AVOIDABLE`; else `triage_level==3` →
`UNCERTAIN`, never driving a positive label); 270-day observation window,
90-day outcome window; fixed index dates TRAIN=2025-10-05,
VALIDATION=2026-01-03, TEST=2026-04-03. All verified unchanged by reusing
the exact same `backend/pit/windows.py` constants and
`backend/pit/encounter_classification.py` logic — no synthetic-specific
branch exists anywhere in that logic.

---

## 12. Leakage-Validation Results

All 17 automated checks (per snapshot) passed for all three synthetic
snapshots — `data/derived/synthetic/validation_report.json`,
`all_passed: true`, zero failed checks in any of TRAIN/VALIDATION/TEST,
schema consistency confirmed across all three. This includes: target
absent from features, `member_id`/`index_date` excluded from features, no
future/outcome-window events used, correct observation/outcome
boundaries, recency computed relative to each row's own index date (not
a global max date), no unwindowed diagnosis crosstab, `frequent_ED_user`
absent, and an independent reconciliation that recomputes several feature
columns and the target directly from the raw synthetic data and confirms
exact agreement with the pipeline's own output (§12–13 of
`docs/03_ML_DATA_PIPELINE.md`'s methodology, reused unchanged).

---

## 13. Original vs. Synthetic Comparison

See §9–§10 above for the full descriptive tables. Summary: the synthetic
dataset represents a population with materially worse average care
access (further PCP/urgent-care distances, more transportation barriers)
concentrated ED use among a smaller subgroup, and a higher proportion of
lower-acuity ED encounters — producing higher overall target prevalence
(~12–13% vs. ~9%) under the identical, unchanged UC07 definition.

---

## 14. Descriptive Signal Checks

Computed on the synthetic TRAIN snapshot only (descriptive point-in-time
checks; no model trained). Full table:
`artifacts/synthetic_experiment/synthetic_descriptive_signal_checks.csv`.
Overall TRAIN prevalence: 11.94%.

| Signal group | Bucket | n | Observed prevalence | Lift |
|---|---|---:|---:|---:|
| `prior_potentially_avoidable_ed_count_270d` | 0 | 7,358 | 8.81% | 0.74× |
| | 1 | 1,844 | 13.67% | 1.15× |
| | 2 | 429 | 26.11% | 2.19× |
| | 3+ | 369 | 49.32% | **4.13×** |
| `transportation_barrier` | 0 | 8,378 | 8.96% | 0.75× |
| | 1 | 1,622 | 27.31% | 2.29× |
| `has_prior_pcp` | 0 | 6,503 | 12.86% | 1.08× |
| | 1 | 3,497 | 10.24% | 0.86× |
| `has_prior_care_management` | 0 | 8,459 | 12.05% | 1.01× |
| | 1 | 1,541 | 11.36% | 0.95× |
| `telehealth_available` | 0 | 2,433 | 18.70% | 1.57× |
| | 1 | 7,567 | 9.77% | 0.82× |
| `clinical_burden` | 0 | 3,938 | 9.62% | 0.81× |
| | 1 | 3,849 | 11.20% | 0.94× |
| | 2 | 1,608 | 16.36% | 1.37× |
| | 3+ | 605 | 20.00% | 1.68× |
| `pcp_distance_miles` band | 0–5 | 3,782 | 8.36% | 0.70× |
| | 5–10 | 3,907 | 11.82% | 0.99× |
| | 10–20 | 2,122 | 17.67% | 1.48× |
| | 20+ | 189 | 21.69% | 1.82× |
| `has_prior_ed` | 0 | 5,716 | 8.75% | 0.73× |
| | 1 | 4,284 | 16.20% | 1.36× |

**Every relationship is meaningful but non-deterministic** — no bucket
reaches 0% or 100%, even the smallest (n=189, `pcp_distance 20+`, at
21.69%). Chronic burden and PCP-distance both show clean monotonic
dose-response gradients. Prior potentially-avoidable ED history shows the
strongest, cleanest monotonic gradient found in this project (0.74× →
4.13× lift). Telehealth availability shows a large *protective*
association (members without telehealth: 18.70% vs. with: 9.77%).

**Notable cross-dataset finding:** `has_prior_ed` and
`prior_potentially_avoidable_ed_count_270d` are **positively** associated
with future risk in the synthetic data (intuitive direction) — the
**opposite** of the genuine, counter-intuitive *negative* association
Phase 4B found and confirmed in the original data (§4 of
`docs/04B_MODEL_IMPROVEMENT.md`). This is reported as an observed
difference between the two datasets' internal structure, not explained
further here — it is a natural consequence of how each dataset was
constructed, not a pipeline artifact (the same unmodified classification
and target logic produced both results).

**Suspicious deterministic relationships: none found.**

---

## 15. Known Limitations

- The synthetic dataset's internal relationships were not authored by
  this project and are not independently documented beyond what is
  measurable from the data itself — this document reports only what was
  observed, not why the synthetic generator produced it.
- Descriptive signal checks (§14) are univariate; they do not account for
  confounding between signal groups (e.g., `transportation_barrier` and
  `pcp_distance_miles` may be correlated with each other) — a full
  multivariate assessment is deferred to the next phase's model training.
- As with the original dataset, all three synthetic snapshots are drawn
  from a single ~18-month data era reused across three non-overlapping
  temporal windows (same Phase 2/3 design, same limitation).
- No fairness/subgroup audit has been performed on the synthetic data —
  out of scope for a data-pipeline phase.

## 16. Why Stronger Synthetic Performance Must Not Be Interpreted as Clinical Validation

If a future model trained on these synthetic snapshots shows
substantially stronger discrimination than `uc07-risk-v1`, that result
will demonstrate only that **the UC07 point-in-time pipeline can recover
strong prospective signal when the input data contains it** — a
statement about the pipeline's mechanics, not about real emergency
department utilization, real patients, or real clinical relationships.
Synthetic results must always be labeled `dataset_id="synthetic_uc07_v1"`
/ `synthetic=true` wherever they are reported (metadata, future model
artifacts, documentation) and must never be merged with, substituted for,
or presented alongside `uc07-risk-v1`'s original-data results without
that label. This document's title and every section explicitly carry
that distinction forward.

---

## 17. Readiness for Synthetic Model Retraining

The synthetic point-in-time data foundation is complete, isolated, and
independently leakage-verified: three schema-consistent, 59-feature
snapshots with the same target/window/index-date methodology as the
original experiment, zero cross-contamination with original-experiment
artifacts, and a documented descriptive comparison. The next phase can
proceed to train candidate models against
`data/derived/synthetic/{train,validation,test}_snapshot.csv` using the
same TRAIN-fit / VALIDATION-select / TEST-once discipline established in
Phase 4, producing a clearly-labeled synthetic model artifact that is
never confused with `uc07-risk-v1`. **Not started in this phase.**
