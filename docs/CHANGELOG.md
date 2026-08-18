# Changelog

## Phase 1 — Repository Audit / Freeze — 2026-08-15

**Objective:** Perform a complete, read-only audit of the existing UC07
repository (data, feature engineering, target, model, training/validation,
navigation, safety, explainability, FastAPI backend, React frontend,
testing, Docker/Azure readiness, security/privacy) and freeze the current
state in documentation so future phases can proceed safely.

**Files created:**
- `docs/01_PROJECT_BASELINE.md`
- `docs/CHANGELOG.md`
- `docs/DECISION_LOG.md`

**Files modified:** none.

**Functional code changed:** NO
**Data changed:** NO
**Model changed:** NO

**Tests run:** None existed to run (confirmed zero test files in the
repository). Verification for this audit was performed via read-only
inspection scripts run against the three raw CSVs and the existing
`backend/ed_risk_model.pkl` artifact (row/column/dtype/duplicate/missing-value
checks, join/member-overlap checks, target-prevalence recomputation, and
feature-importance/correlation checks) — none of these scripts wrote to any
tracked repository file.

**Findings:** See `docs/01_PROJECT_BASELINE.md` §22–§28 for the full list.
Summary of the most significant, verified findings:
1. The current target (`frequent_ED_user = ED_visits_365d >= 2`) is a
   frequency label, not an avoidability label — it does not align with the
   UC07 objective as stated.
2. Confirmed indirect data leakage: `days_since_last_ED` (40.8% of RF
   feature importance) and the unwindowed `diagnosis_*` crosstab (36.3%
   combined importance) are ED-utilization-derived features that were not
   caught by the existing `DROP_BEFORE_MODEL` leakage guard; together they
   account for ~77% of what the model relies on.
3. Train/test split is random and stratified, not point-in-time/temporal,
   against a single global reference date shared by every member.
4. Care Management exists in the source data (`care_type` category, 2,612
   rows) and as an engineered feature, but is not a reachable output of the
   navigation/recommendation logic.
5. No deterministic safety layer exists — `red_flag`, `triage_level`,
   `admitted`, and `icu` are captured in `raw_ed_visits.csv` but discarded
   before reaching the model, the recommendation logic, or the API
   response. Current safety behavior is entirely static frontend
   disclaimer text (which is itself well-written and appropriately
   cautious).
6. Risk scoring, navigation, explainability, and safety logic are
   intermingled within `backend/predict.py` and `backend/main.py` with no
   architectural separation — no multi-agent or modular structure exists.
7. Zero automated tests, no Dockerfile, and no Azure/CI-CD configuration
   exist anywhere in the repository.

**Remaining risks:**
- Reported model metrics (accuracy/precision/recall/F1/ROC-AUC in
  `train_model.py`) likely overstate true prospective performance given the
  leakage and non-temporal split noted above.
- No calibration, PR-AUC, confusion matrix, cross-validation, or subgroup
  (fairness) validation has ever been performed on this model.
- No model/environment version metadata is stored in the artifact; only
  `scikit-learn` is pinned in `requirements.txt`.
- FastAPI backend has open CORS (`*`), no authentication, no upload
  size/row limits, and reloads the model from disk on every request.
- `frontend_legacy/index.html` remains in the repo as unused dead code
  pointing at a hardcoded local API URL.

**Next phase:** Design (not implementation) of an avoidability-aware target
definition, a leakage remediation plan, a point-in-time validation
strategy, and the risk-detection / care-navigation / safety-policy
architectural split — to be ratified before any code, data, or model
changes are made. See `docs/01_PROJECT_BASELINE.md` §26.

## Phase 2 — UC07 Problem Definition & Architecture Design — 2026-08-15

**Objective:** Produce a technically defensible, fully-specified UC07
design — historical avoidability definition, member-level prediction
target, temporal validation strategy, leakage policy, and a three-agent
(Risk Detection / Care Navigation / Safety & Policy) architecture —
answering every open question raised by the Phase 1 audit, before any
implementation begins.

**Files created:**
- `docs/02_UC07_AND_DATA_DESIGN.md`

**Files modified:**
- `docs/DECISION_LOG.md` (Phase 2 decisions appended)
- `docs/CHANGELOG.md` (this entry)

**Functional code changed:** NO
**Source data changed:** NO
**Model changed:** NO
**Model retrained:** NO

**Key decisions:** see `docs/DECISION_LOG.md` "2026-08-15 — Phase 2 final
decisions" (items 9–22). Summary: a conservative 3-state historical
avoidability label built from `triage_level`/`red_flag`/`admitted`/`icu`/
`major_procedure` (explicitly not `diagnosis`, which was verified to carry
no acuity signal in this dataset); a 90-day forward-looking member-level
target; a 270-day observation window; a 3-snapshot, non-overlapping
temporal train/validation/test split (2025-10-05 / 2026-01-03 / 2026-04-03
index dates); an explicit label-only-vs-safe-historical-feature leakage
policy that resolves both Phase 1 leakage findings without discarding ED
history as a feature source; and a three-agent architecture with the
Safety & Policy Agent holding final, non-bypassable authority and Care
Management made reachable for the first time.

**Read-only analyses performed:** diagnosis×triage_level, diagnosis×
red_flag, diagnosis×admitted, diagnosis×icu, diagnosis×major_procedure,
and triage_level×{admitted,icu,major_procedure} crosstabs; candidate
avoidability-rule prevalence checks; member/ED/care date-range and monthly
volume checks; member-level ED visit-count distribution; chronic-condition
and access-variable prevalence in `raw_members.csv`; care-type coverage in
`raw_care_history.csv`; outcome-window prevalence at 30/60/90/120/180-day
horizons; 3-snapshot temporal-split feasibility and prevalence stability
check, including cross-snapshot member/positive-set overlap. All analyses
were read-only against the three immutable source CSVs; no dataset,
model, or functional file was modified.

**Known risks:** see `docs/02_UC07_AND_DATA_DESIGN.md` §24 "Open Design
Risks" — most significantly: the conservative `UNCERTAIN`-as-negative
labeling choice likely undercounts real recall; `diagnosis`'s predictive
value as a feature (as opposed to a label input) is unverified; ~7–8%
member overlap between temporal splits is an accepted, monitored
non-independence rather than a fully member-disjoint design; the 3
snapshots reuse one 547-day data era rather than independent years.

**Next phase:** Phase 3 implementation, in the sequence given in
`docs/02_UC07_AND_DATA_DESIGN.md` §25 — shared point-in-time feature
module → label construction function → derived TRAIN/VALIDATION/TEST
snapshot datasets → model retraining and threshold selection → agent
refactor of `predict.py`/`main.py` → frontend updates → automated tests →
Docker → Azure. Not started in this phase.

## Phase 3 — Point-in-Time ML Data Pipeline — 2026-08-15

**Objective:** Implement (not train) the point-in-time ML data
foundation approved in Phase 2: historical ED encounter classification,
the 90-day future member-level target, 270-day observation-window
feature engineering, the three fixed TRAIN/VALIDATION/TEST snapshots,
automated leakage prevention, derived snapshot datasets, automated tests,
and data-quality/leakage reports.

**Functional files created:**
- `backend/pit/encounter_classification.py`
- `backend/pit/windows.py`
- `backend/pit/target.py`
- `backend/pit/features.py`
- `backend/pit/validation.py`
- `backend/pit/manifest.py`
- `backend/pit/build_snapshots.py`

**Functional files modified:** none. `backend/feature_engineering.py`,
`backend/train_model.py`, `backend/predict.py`, `backend/main.py`, and
`backend/ed_risk_model.pkl` were not edited.

**Generated derived files** (`data/derived/`, new directory):
- `train_snapshot.csv`, `validation_snapshot.csv`, `test_snapshot.csv`
- `feature_manifest.json`
- `snapshot_metadata.json`
- `validation_report.json`

**Tests created** (`backend/tests/`, new directory, 86 tests, all passing):
`test_encounter_classification.py`, `test_windows.py`, `test_target.py`,
`test_features.py`, `test_validation.py`, `test_pipeline_integration.py`,
`test_legacy_isolation.py`, plus `conftest.py`. `pytest` was added to
`.venv` as a test dependency (explicitly authorized by the Phase 3 spec).

**Raw datasets changed:** NO (SHA-256 verified identical before/after —
see `docs/03_ML_DATA_PIPELINE.md` §20)
**Old model changed:** NO
**New model trained:** NO

**Key implementation decisions:**
- Explicit `NaN` + companion `has_prior_*` 0/1 flag used as the
  missing-value representation for every recency feature, instead of a
  sentinel numeric value (recorded in `docs/DECISION_LOG.md` as a
  Phase 3 implementation decision — does not alter Phase 2 methodology).
- Protected/uncertain ED subcounts limited to 90d/270d windows (rather
  than the full 30/90/180/270d ladder used for total/avoidable counts) to
  avoid low-count, near-collinear features for non-primary signals.
- Recency features are capped at the 270-day observation window (an event
  older than that is treated as "no prior event in this window," not
  looked up indefinitely into the past) — consistent with the Phase 3
  spec's "strictly from the observation window" instruction.
- `backend/pit/` uses flat sibling imports (no package `__init__.py`),
  matching the existing flat-import convention already used throughout
  `backend/` (e.g. `main.py`'s `from feature_engineering import ...`).

**Validation results:** all 17 automated leakage/quality checks passed on
all three real snapshots (`data/derived/validation_report.json`,
`all_passed: true`). Target prevalence: TRAIN 9.04%, VALIDATION 9.58%,
TEST 9.08% — consistent with, though not forced to equal, Phase 2's
9.07%/9.57%/9.09% exploratory estimate; the small delta is fully
explained by an intentional one-day boundary-convention difference
between Phase 2's informal estimate and the Phase 3 spec's exact
`[index_date, outcome_end)` convention (see
`docs/03_ML_DATA_PIPELINE.md` §15).

**Leakage checks:** structural boundary checks, forbidden-column checks,
identifier/metadata/target-exclusion checks, legacy-target and
diagnosis-crosstab absence checks, a global-max-date regression check, and
an independent reconciliation check (recomputes counts/target directly
from raw data using only window-boundary constants and asserts exact
equality against the pipeline's own output) — all passed; the
reconciliation and global-max-date checks were separately proven to
actually catch injected bugs (`test_reconciliation_fails_on_corrupted_*`,
`test_no_global_max_date_index_detects_bug_pattern`).

**Known limitations:** ~7% positive-label overlap across temporal
snapshots (accepted per Phase 2 §9.4); `UNCERTAIN` encounters never drive
a positive label (intentionally conservative, Phase 2 §6.2); diagnosis
predictors remain excluded and their value unverified; cold-start members
receive zero/NaN history features with no model yet to validate against;
`gender` retained pending future subgroup validation. Full list in
`docs/03_ML_DATA_PIPELINE.md` §22.

**Next phase:** Phase 4 — model training and selection, fit on TRAIN
only, threshold selection on VALIDATION only, single evaluation on TEST,
artifact packaging with the feature manifest and version metadata — then
agent implementation, API/frontend integration, expanded testing, Docker,
and Azure, in that order. Not started in this phase.

## Phase 4 — Model Development, Calibration & Risk Tiers — 2026-08-15

**Objective:** Train and select a model for the Phase 3 target
(`future_potentially_avoidable_ed_90d`), fitting on TRAIN only, using
VALIDATION for every model/calibration/threshold decision, evaluating
exactly once on TEST, and producing a versioned artifact, metadata, and
evaluation reports. No agents, no Care Navigation logic, no API/frontend
changes, no Docker/Azure.

**Files created:**
- `backend/modeling/feature_spec.py`, `preprocessing.py`, `metrics.py`,
  `risk_tiers.py`, `train.py`
- `backend/tests/test_model_pipeline.py`
- `docs/04_MODEL_DEVELOPMENT.md`

**Files modified:**
- `backend/requirements.txt` (pinned `joblib==1.5.3`; `scikit-learn==1.7.2`
  was already pinned in Phase 1)
- `backend/tests/conftest.py` (added `backend/modeling` to sys.path)
- `docs/CHANGELOG.md` (this entry), `docs/DECISION_LOG.md`

**Model artifacts created:**
- `backend/models/uc07_risk_v1_model.joblib`
- `backend/models/uc07_risk_v1_model_metadata.json`

**Generated report files** (`artifacts/model_evaluation/`, new directory):
`candidate_metrics.csv`, `hyperparameter_search.csv`,
`calibration_comparison.csv`, `threshold_analysis.csv`,
`validation_risk_tiers.csv`, `test_risk_tiers.csv`, `calibration_bins.csv`,
`subgroup_metrics.csv`, `global_feature_importance.csv`,
`final_model_selection.json`, `validation_scored_members.csv` (member_id +
score + tier only), `test_scored_members.csv` (member_id + score + tier
only).

**Legacy model changed:** NO (`backend/ed_risk_model.pkl` untouched, verified reloadable in its original shape)
**Raw data changed:** NO (SHA-256 verified unchanged before/after)
**Phase 3 snapshot files changed:** NO (SHA-256 verified unchanged before/after, matches the exact hashes recorded at the start of this phase)

**Candidate models evaluated:** Dummy baseline (`strategy=prior`),
Logistic Regression (8 hyperparameter combinations), Random Forest (8
combinations), HistGradientBoostingClassifier (16 combinations) — 32
total combinations, all fit on TRAIN, scored on VALIDATION.

**Selected model:** Logistic Regression, `C=0.01`, `class_weight=None`,
uncalibrated — highest VALIDATION PR-AUC (0.1220) and best Brier (0.0861)
among every candidate/calibration combination evaluated; the
interpretability/complexity tie-break rule was checked but not needed
(this candidate won outright on the primary metric).

**Calibration decision:** compared uncalibrated vs. sigmoid vs. isotonic
(5-fold CV fit on TRAIN) for the two leading candidates
(LogisticRegression, HistGradientBoosting). Retained the simpler
uncalibrated Logistic Regression — calibration did not improve its
PR-AUC or Brier. (Calibration *did* materially fix HistGradientBoosting's
poorly-calibrated raw probabilities under `class_weight="balanced"`,
Brier 0.233 → ~0.086 — documented as a useful illustration even though
that candidate wasn't selected.)

**Threshold decision:** `moderate_threshold = 0.097609` (65th percentile
of VALIDATION scores), `high_threshold = 0.123776` (90th percentile) —
percentile-derived from the model's own VALIDATION score distribution,
not invented constants. Produces monotonically increasing observed
prevalence LOW (8.5%) < MODERATE (11.0%) < HIGH (12.7%) on VALIDATION,
confirmed again on TEST (7.9% < 10.9% < 12.2%).

**Validation metrics:** ROC-AUC 0.5728, PR-AUC 0.1220, Brier 0.0861, log
loss 0.3129 (prevalence 9.58%, no-skill PR-AUC baseline 0.0958).

**Final TEST metrics:** ROC-AUC 0.5747, PR-AUC 0.1111, Brier 0.0822, log
loss 0.3018 (prevalence 9.08%). VALIDATION→TEST deltas (PR-AUC −0.0109,
ROC-AUC +0.0019, Brier −0.0040) did not trip the pre-declared degradation
rule (`pr_auc_delta < -0.03` or `brier_delta > 0.01`) — no material
degradation, no post-TEST tuning performed.

**Automated tests:** 107/107 passing (86 from Phase 3 unchanged + 21 new
Phase 4 tests in `test_model_pipeline.py`, covering manifest-only
features, target/identifier/metadata exclusion from X, feature-order
matching, artifact round-trip + valid-probability output, threshold
validity, risk-tier mapping correctness, artifact/metadata consistency,
legacy-target absence, legacy artifact untouched, TEST-isolation of the
selection function (by signature and by source-code scan), and
snapshot/raw-hash immutability across a real training run).

**Known limitations:** modest overall discrimination (ROC-AUC ≈0.57,
PR-AUC ≈0.11) — an honest consequence of correctly excluding
leakage-risk features, not a bug; one small subgroup
(`clinical_burden 3+`, n=402) showed sub-no-skill ROC-AUC (0.483),
flagged for Phase 6; several top feature coefficients have
multicollinearity-driven counter-intuitive signs; all Phase 2/3 known
limitations (uncertain-as-negative labeling, diagnosis exclusion,
cross-snapshot overlap, single data era, gender fairness review pending)
carry forward unchanged. Full list in `docs/04_MODEL_DEVELOPMENT.md` §27.

**Next phase:** Phase 5 — Care Navigation Agent (PCP/Urgent Care/
Telehealth/Care Management routing using this model's risk tier plus
access/utilization features) and Safety & Policy Agent implementation per
the Phase 2 contracts, then wiring the Risk Detection Agent behind the
existing FastAPI endpoints — `predict.py`/`main.py`/frontend changes
begin only at that point. Not started in this phase.

## Phase 4B — Controlled Model & Feature Improvement — 2026-08-15

**Objective:** Determine, via controlled experimentation using only
TRAIN/VALIDATION, whether uc07-risk-v1's predictive performance can be
meaningfully improved with additional leakage-safe point-in-time
features (restructured time windows, velocity, care-setting mix,
continuity, access interactions, historical-ED-pattern extras, and a
controlled historical-diagnosis representation) — without weakening any
Phase 2–4 leakage control, and without touching TEST unless a candidate
passes a pre-declared VALIDATION promotion gate.

**Files created:**
- `backend/pit/features_v2.py`
- `backend/modeling/improve.py`
- `backend/tests/test_phase4b.py`
- `docs/04B_MODEL_IMPROVEMENT.md`

**Files modified:** none in `backend/pit/features.py`, `windows.py`,
`target.py`, `encounter_classification.py`, or any Phase 4 file — Phase
4B is purely additive.

**Experiments performed:** window representation (baseline nested vs.
reduced nested vs. non-overlapping bands); a 7-step ablation (A–G, adding
one feature group at a time, winner selected by evidence/argmax
VALIDATION PR-AUC, not by mechanical cascade); an isolated
WITHOUT-vs-WITH diagnosis comparison against the actual best-performing
non-diagnosis feature set; a 3-algorithm model comparison
(LogisticRegression/RandomForest/HistGradientBoosting); a 3-penalty (L2/
L1/ElasticNet) regularization sweep; a 2-method (sigmoid/isotonic)
calibration comparison for the two leading candidates; a suspicious
performance guardrail (0.80 ROC-AUC ceiling, never triggered).

**New feature groups:** utilization velocity/trend (3 features),
care-setting mix + continuity/engagement (13 features), access ×
utilization interactions (6 features), historical-ED-pattern extras (2
features), controlled historical diagnosis (4 features, volume-normalized,
observation-window only).

**Diagnosis experiment result:** WITHOUT_DIAGNOSIS VALIDATION PR-AUC
0.12210 vs. WITH_DIAGNOSIS 0.12186 (Δ −0.00024) — **rejected**, no
incremental signal demonstrated.

**Best VALIDATION improvement:** window restructuring alone (drop the
most-redundant 180d window) — PR-AUC 0.12210 vs. v1's 0.12198 (+0.0002,
noise-level). The single best configuration found anywhere in Phase 4B
(ElasticNet, C=0.01, on the restructured-windows feature set) reached
PR-AUC 0.12394 (+0.00196 vs. v1) — reported for transparency, still well
short of the promotion margin.

**TEST accessed:** NO — no candidate passed the VALIDATION promotion
gate (PR-AUC gain ≥0.01 absolute OR HIGH-tier lift relative gain ≥15%,
with calibration preserved); actual gains were +0.0002 PR-AUC / +3.0%
tier lift.

**Final decision:** KEEP UC07-RISK-V1 (Decision B — a valid, non-negative
finding per the Phase 4B spec, not a failure).

**Model promoted:** NO
**Model version:** unchanged — `uc07-risk-v1` remains the production
candidate; no `uc07-risk-v2` artifact was created.

**Automated tests:** 123/123 passing (107 carried forward from Phase 3/4
unchanged + 16 new Phase 4B tests in `test_phase4b.py`, covering
observation-window-only filtering for every new feature, band-boundary
correctness, safe zero-denominator ratio handling, no infinite/negative
values, interaction-feature correctness, ablation feature-order
reproducibility, artifact-serialization mechanics, V1-artifact
untouched, no-fake-V2-artifact-when-not-promoted, and TEST isolation of
the candidate-selection function by both signature and source-code scan).

**Raw datasets changed:** NO (SHA-256 verified unchanged before/after)
**Phase 3 snapshots changed:** NO (SHA-256 verified unchanged before/after)

**Known limitations:** the dataset's genuine prospective signal for this
target appears intrinsically modest (ROC-AUC 0.57–0.59 across every
configuration tried); the counter-intuitive negative association between
prior ED utilization and future avoidable-ED risk (found in Step 1
diagnostics) is real in the data and unexplained, flagged for future
investigation; diagnosis features demonstrably add no signal even under
careful controls. Full list in `docs/04B_MODEL_IMPROVEMENT.md` §22.

**Next phase:** Phase 5 — Care Navigation Agent and Safety & Policy Agent
implementation per the Phase 2 contracts, using `uc07-risk-v1` unchanged
(no v2 exists). Not started in this phase.

## Phase 4C — Synthetic Dataset Integration & PIT Rebuild — 2026-08-15

**Objective:** Safely introduce a second, explicitly synthetic dataset
trio (`data/synthetic/raw_{members,ed_visits,care_history}.csv`),
preserve every original-data experiment artifact byte-for-byte, refactor
the point-in-time pipeline to be data-source configurable (one
implementation, not a duplicate), and rerun it against the synthetic data
using the unchanged Phase 2/3 methodology. No model training in this
phase.

**Files created:**
- `data/derived/original/` (archived copy of the pre-existing `data/derived/*.csv|json`)
- `data/derived/synthetic/{train,validation,test}_snapshot.csv`, `feature_manifest.json`, `snapshot_metadata.json`, `validation_report.json`
- `artifacts/synthetic_experiment/original_vs_synthetic_comparison.csv`
- `artifacts/synthetic_experiment/synthetic_descriptive_signal_checks.csv`
- `backend/tests/test_synthetic_pipeline.py`
- `docs/04C_SYNTHETIC_DATA_EXPERIMENT.md`

**Files modified:**
- `backend/pit/build_snapshots.py` — `main()`, `load_raw()`, and
  `build_snapshot_metadata()` now accept configurable
  `members_path`/`ed_path`/`care_path`/`output_dir`/`dataset_id`/`synthetic`
  parameters, all defaulting to the exact pre-Phase-4C original-dataset
  behavior. `features.py`, `target.py`, `windows.py`, and
  `encounter_classification.py` were **not** modified — both experiments
  reuse the identical point-in-time implementation.

**Original datasets changed:** NO (SHA-256 verified unchanged before/after)
**Synthetic datasets changed:** NO (SHA-256 verified unchanged before/after)
**Original derived artifacts changed:** NO (`data/derived/*.csv` verified
byte-identical to the new `data/derived/original/` archive; metadata's
pre-existing field *values* verified unchanged, with two new
backward-compatible fields — `dataset_id`, `synthetic` — now present on
regeneration)
**V1 model changed:** NO (`backend/models/uc07_risk_v1_model.joblib` reloaded and verified unchanged)

**Synthetic snapshots generated:** TRAIN 10,000 rows / 1,194 positives /
11.94% prevalence; VALIDATION 10,000 / 1,174 / 11.74%; TEST 10,000 /
1,307 / 13.07% — not forced to match the original's ~9%, reported as
naturally produced. 59-feature schema, identical to the original.

**Leakage-test results:** 17/17 automated checks passed on all 3
synthetic snapshots (`data/derived/synthetic/validation_report.json`,
`all_passed: true`); schema consistency confirmed across TRAIN/VALIDATION/TEST.

**Automated-test results:** 140/140 passing (123 carried forward from
Phase 3/4/4B unchanged + 17 new tests in `test_synthetic_pipeline.py`
covering configurable-path operation for both datasets, output-directory
isolation, non-overwrite of original outputs, dataset-identity metadata
correctness for both experiments, feature-schema reproducibility, full
leakage-check pass-through, raw-CSV and V1-artifact immutability, and a
fresh before/after hash guard on a throwaway synthetic run).

**Next step:** synthetic model retraining (candidate models fit on
`data/derived/synthetic/train_snapshot.csv`, selected on VALIDATION,
evaluated once on TEST, artifact clearly labeled
`dataset_id="synthetic_uc07_v1"` and never confused with
`uc07-risk-v1`). Not started in this phase.

## Phase 4D — Synthetic Model Retraining & Selection — 2026-08-15

**Objective:** Train and evaluate the UC07 risk model against the
Phase 4C synthetic snapshots, using the exact Phase 4 methodology and the
unmodified 59-feature baseline (no Phase 4B experimental features), to
determine whether the synthetic data's stronger prospective relationships
produce a genuinely stronger model under identical leakage controls.

**Files created:**
- `backend/modeling/train_synthetic.py` (reuses `train.py`'s candidate
  builders, `select_model_on_validation()`, `evaluate_frozen_model_on_test()`,
  `extract_global_feature_importance()`, `run_subgroup_checks()`, and
  `write_reports()` unmodified — no modeling-logic duplication)
- `backend/tests/test_synthetic_model.py`
- `docs/04D_SYNTHETIC_MODEL_DEVELOPMENT.md`

**Files modified:** none. `train.py` (Phase 4) was not edited.

**Candidate models evaluated:** Dummy (`strategy=prior`), Logistic
Regression (8 combos), Random Forest (8 combos), HistGradientBoosting
(16 combos) — same grids as Phase 4.

**Selected model:** Logistic Regression, `C=0.01`, `class_weight=None`,
uncalibrated — highest VALIDATION PR-AUC (0.3288) and Brier (0.0927)
among every candidate/calibration combination, won outright on the
primary metric.

**Calibration:** sigmoid/isotonic compared for the two leading
candidates (5-fold CV on TRAIN); neither improved the selected model;
uncalibrated retained.

**Thresholds:** `moderate_threshold=0.105986` (65th percentile),
`high_threshold=0.213252` (90th percentile), VALIDATION-derived.

**Validation metrics:** ROC-AUC 0.6993, PR-AUC 0.3288, Brier 0.0927
(prevalence 11.74%).

**Final TEST metrics:** ROC-AUC 0.7048, PR-AUC 0.3366, Brier 0.1023
(prevalence 13.07%); PR enrichment 2.58×; HIGH-tier lift 2.87× —
materially stronger than `uc07-risk-v1`'s TEST result (ROC-AUC 0.5747,
PR-AUC 0.1111, HIGH lift 1.35×). VALIDATION→TEST deltas did not trip the
pre-declared degradation rule (PR-AUC +0.0079, ROC-AUC +0.0055, Brier
+0.0097 — the closest-to-threshold Brier delta of any comparison in this
project, noted but not flagged) — no post-TEST tuning performed. ROC-AUC
stayed below the 0.85 suspicious-performance ceiling at every step; a
leakage/proxy audit (`docs/04D_SYNTHETIC_MODEL_DEVELOPMENT.md` §21) found
no leakage artifact — the improvement is attributed to the synthetic
data's intentionally stronger constructed relationships.

**Model artifact:** `backend/models/uc07_risk_synthetic_v1_model.joblib`
**Metadata artifact:** `backend/models/uc07_risk_synthetic_v1_model_metadata.json`
(includes the required disclaimer: *"This model was trained and
evaluated on synthetic data and must not be interpreted as clinically
validated."*)

**Synthetic labeling:** `model_version="uc07-risk-synthetic-v1"`,
`dataset_id="synthetic_uc07_v1"`, `synthetic=true`,
`intended_use="demonstration / UC07 navigation prototype"` present in
both the artifact and metadata.

**Automated tests:** 164/164 passing (140 carried forward from Phase
3/4/4B/4C unchanged + 24 new tests in `test_synthetic_model.py`, covering
synthetic-only data sourcing, target/identifier/metadata exclusion,
feature-order/manifest matching, legacy-target/diagnosis absence,
probability validity, threshold validity, risk-tier mapping, artifact
round-trip, synthetic metadata labeling, original-artifact and
original/synthetic-raw-dataset/snapshot immutability, TEST isolation of
the reused selection function, and end-to-end scoring of a real synthetic
row).

**Original V1 changed:** NO (reloaded and verified unchanged)
**Original datasets changed:** NO (SHA-256 verified unchanged)
**Synthetic datasets changed:** NO (SHA-256 verified unchanged)
**Phase 4C snapshots changed:** NO (SHA-256 verified unchanged before/after training)

**Next phase:** Phase 5 — Care Navigation Agent and Safety & Policy Agent
implementation. `uc07-risk-synthetic-v1` recommended for the Phase 5
demonstration multi-agent system; `uc07-risk-v1` preserved as the
original-data benchmark. Runtime model selection/configuration is a
Phase 5 architecture concern, not implemented here. Not started in this phase.

## Phase 5 — Multi-Agent UC07 Decision System — 2026-08-16

**What existed before:** a single legacy inference path
(`backend/predict.py` + `backend/feature_engineering.py`, pre-Phase-2
`frequent_ED_user` model) with risk scoring and navigation logic mixed
together and no deterministic safety layer, plus the Phase 4/4D trained
risk models sitting unused by the API.

**What changed:** introduced a properly separated three-agent decision
system (Risk Detection, Care Navigation, Safety & Policy) with explicit
typed contracts and a fixed, enforced orchestration order, wired into
FastAPI via a new endpoint. The Safety & Policy Agent is the final,
non-bypassable authority over every response. The legacy endpoints and
legacy model are untouched.

**Files created:**
- `backend/agents/contracts.py`, `risk_detection.py`, `care_navigation.py`,
  `safety_policy.py`, `orchestrator.py`
- `backend/tests/test_agent_contracts.py`, `test_risk_detection_agent.py`,
  `test_care_navigation_agent.py`, `test_safety_policy_agent.py`,
  `test_orchestrator.py`, `test_uc07_api.py`
- `docs/05_MULTI_AGENT_SYSTEM.md`

**Files modified:**
- `backend/main.py` — added `POST /uc07/decide`, `GET /model-info`,
  extended `GET /health`; every legacy endpoint (`/`, `/dashboard`,
  `/predict`, `/predict-json`, `/explain-member`) unchanged.
- `backend/tests/conftest.py` — added `backend/agents` to `sys.path`
  (same flat-import convention as `backend/pit`/`backend/modeling`).
- `backend/requirements-dev.txt` — added `httpx==0.28.1` (test-only,
  required for FastAPI's `TestClient`).

**Agents introduced:** Risk Detection Agent (loads `uc07-risk-synthetic-v1`,
frozen thresholds `MODERATE=0.105986`/`HIGH=0.213252`, non-causal
top-3 contributing factors), Care Navigation Agent (deterministic rule
tree over 5 destinations: PRIMARY_CARE/URGENT_CARE/TELEHEALTH/
CARE_MANAGEMENT/NO_PROACTIVE_NAVIGATION), Safety & Policy Agent
(CLEAR/CAUTION/OVERRIDE, 6 independent override triggers, centralized
32-phrase prohibited-language policy, missing-context-means-CAUTION).

**API changes:** `POST /uc07/decide` (new), `GET /model-info` (new),
`GET /health` extended with `uc07_model_loaded`/`uc07_model_version`/
`uc07_model_error`. No existing endpoint's request/response shape changed.

**Tests added:** 129 new tests (293 total, up from 164; all previously
passing tests still pass unchanged) covering contracts, each agent in
isolation, orchestration call-order/non-bypass proofs, determinism,
serialization, threshold boundaries, all 11 named safety/navigation
scenarios, adversarial language scanning across the full synthetic
population, and FastAPI-level error handling.

**Safety controls:** centralized `PROHIBITED_PHRASES` policy applied to
every navigation explanation and safety message in every state; OVERRIDE
unconditionally suppresses the navigation destination; missing current
safety context always resolves to CAUTION, never CLEAR; Safety Agent
called exactly once, always last, on every code path (single-member and
batch).

**Model changed:** NO (`uc07-risk-v1` and `uc07-risk-synthetic-v1` both
reloaded and verified unchanged)
**Datasets changed:** NO (original + synthetic raw, Phase 3 + Phase 4C
snapshots all SHA-256 verified unchanged)
**Thresholds changed:** NO (loaded from frozen model metadata, never
hard-coded, never modified)

**Remaining risks:** `current_safety_context` is caller-supplied and
unverified by this system; navigation rule thresholds are engineering
judgment, not outcome-validated; no frontend integration yet; runtime
model selection between v1/synthetic-v1 is not implemented (hard-wired
to synthetic-v1 for this demonstration).

**Next phase:** frontend integration/polish (not started), using the new
`/uc07/decide` endpoint; Docker/Azure packaging remains out of scope
until agent validation is further along.

## Phase 4E — Controlled Model Optimization (Tree Models) — 2026-08-16

**Objective:** Determine, with a dedicated wider search and full
threshold-dependent reporting (Accuracy included, never as a selection
metric), whether Random Forest or XGBoost can meaningfully outperform
the current Logistic Regression `uc07-risk-synthetic-v1` on the exact
same frozen 59-feature synthetic snapshots, before Phase 6.

**Files created:**
- `backend/modeling/train_phase4e_tree_comparison.py` (reuses
  `feature_spec.py`, `preprocessing.py`, `metrics.py`, `risk_tiers.py`,
  and `train.py`'s `sha256_file`/`compare_calibration_methods`/
  `extract_global_feature_importance`/`run_subgroup_checks` unmodified)
- `backend/tests/test_phase4e_tree_comparison.py` (40 new tests)
- `docs/04E_TREE_MODEL_OPTIMIZATION.md`
- `artifacts/phase4e_tree_model_comparison/` (16 report files)

**Files modified:** `backend/requirements.txt` — added `xgboost==3.2.0`
(newly introduced dependency; not previously used anywhere in the
repository). `train.py`, `train_synthetic.py`, `metrics.py`,
`preprocessing.py`, `feature_spec.py`, `risk_tiers.py` were **not** edited.

**Logistic Regression reference:** refit with the frozen hyperparameters
(`C=0.01`, `class_weight=None`); reproduced the frozen VALIDATION metrics
exactly (ROC-AUC 0.699301, PR-AUC 0.32877, Brier 0.09267, tolerance
0.003) — no investigation triggered.

**Random Forest search:** 16 curated combinations. Best (by VALIDATION
PR-AUC): `n_estimators=400, max_depth=6, min_samples_leaf=20,
class_weight="balanced"` — ROC-AUC 0.6897, PR-AUC 0.3176, Brier 0.2001
uncalibrated (~0.093 after sigmoid/isotonic calibration).

**XGBoost search:** 20 curated combinations, including an optional
`scale_pos_weight` axis computed from TRAIN prevalence only. Best:
`n_estimators=400, max_depth=3, learning_rate=0.02, min_child_weight=5,
subsample=0.8, colsample_bytree=0.8` (no class weighting) — ROC-AUC
0.6893, PR-AUC 0.3198, Brier 0.0935.

**HistGradientBoosting reference:** Phase 4D's existing winning
combination, refit once for consistent Phase 4E metrics — ROC-AUC
0.6915, PR-AUC 0.3203, Brier 0.2032 uncalibrated.

**Accuracy / Balanced Accuracy:** reported at 0.50, MODERATE (0.105986),
and HIGH (0.213252) for all four candidates
(`docs/04E_TREE_MODEL_OPTIMIZATION.md` §11–13) — never used for
selection. Notably, Random Forest's and HistGradientBoosting's winning
`class_weight="balanced"` combinations select 100% of the population at
Logistic's MODERATE/HIGH thresholds (0 true negatives) — Accuracy
collapses to VALIDATION prevalence (0.1174) there, a concrete
illustration of why Accuracy must always be reported with its threshold.

**Overfitting:** every tree candidate's TRAIN→VALIDATION ROC-AUC gap
(0.054–0.072) is ~9–12× Logistic's (0.006) — below the hard
overfitting flag but a material generalization concern at this dataset
size, and the deciding factor alongside the metric deltas below.

**Leakage audit:** PASS — 0 forbidden columns in the 59-feature list;
no candidate's ROC-AUC (0.689–0.699) approached the 0.85 suspicious-
performance ceiling.

**Promotion decision: KEEP LOGISTIC.** Best tree candidate (XGBoost, by
PR-AUC+Brier tie-break) trailed Logistic on both primary metrics
(ROC-AUC delta -0.0100, PR-AUC delta -0.0089); the promotion signal
(ROC-AUC delta ≥ +0.02, or a clear PR-AUC/lift improvement) was not met.
No new model artifact was created; `uc07-risk-synthetic-v1` and its
thresholds (MODERATE=0.105986, HIGH=0.213252) are unchanged.

**Final TEST (evaluated exactly once, after freeze):** ROC-AUC 0.704752,
PR-AUC 0.336636, Brier 0.102334, PR enrichment 2.58×, HIGH-tier lift
2.87× — identical to `uc07-risk-synthetic-v1`'s existing Phase 4D TEST
result (same frozen pipeline), confirming full consistency.

**Automated tests:** 333/333 passing (293 carried forward unchanged +
40 new tests in `test_phase4e_tree_comparison.py`, covering accuracy/
balanced-accuracy calculation correctness, confusion-matrix arithmetic,
probability bounds, threshold loading, feature-order compatibility,
metadata compatibility, TEST isolation, artifact preservation, dataset
immutability, and Risk Agent compatibility).

**Risk Agent updated:** NO (no model promoted; still loads
`uc07-risk-synthetic-v1` with its existing frozen thresholds).
**Navigation Agent changed:** NO. **Safety Agent changed:** NO.
**Original datasets changed:** NO (SHA-256 verified unchanged).
**Synthetic datasets/snapshots changed:** NO (SHA-256 verified
unchanged directly after the Phase 4E script's own run; the two model
metadata `.json` sidecars are refreshed with a new
`training_timestamp_utc` — content otherwise byte-identical — whenever
the pre-existing `test_model_pipeline.py`/`test_synthetic_model.py`
session fixtures re-run `train.py`/`train_synthetic.py` as part of the
full suite, which is documented, pre-existing behavior unrelated to
Phase 4E; the underlying `.joblib` model artifacts themselves and all
raw/snapshot datasets remain byte-for-byte unchanged).
**Existing models preserved:** YES (both `.joblib` artifacts unchanged).

**Remaining concern carried forward:** `transportation_barrier=1` still
shows ~0.99 recall at MODERATE on TEST — unresolved, deferred to Phase 6
as before.

**Synthetic limitation:** all Phase 4E results describe this synthetic
generator's constructed relationships at n=10,000 per snapshot; not
real-world clinical evidence, and not a general claim about tree models
vs. logistic regression for ED utilization prediction.

**Next phase:** Phase 6 — fairness/bias audit — proceeds against
`uc07-risk-synthetic-v1` unchanged. Frontend integration, Docker, and
Azure packaging remain explicitly out of scope for this phase.

## Phase 6 — Safety, Fairness, Robustness & End-to-End Validation — 2026-08-16

**Objective:** Validate (not develop) whether the complete UC07 multi-agent
system behaves safely, consistently, robustly, and reasonably across
member groups and edge cases. Model frozen throughout — `uc07-risk-synthetic-v1`
was never retrained, tuned, recalibrated, or re-thresholded.

**Files created:**
- `backend/validation/phase6_validation.py` (produces every artifact
  under `artifacts/phase6_validation/`)
- `backend/tests/test_phase6_safety_invariants.py` (56 new tests: 10
  named safety invariants + failure-mode coverage)
- `docs/06_SAFETY_FAIRNESS_ROBUSTNESS_VALIDATION.md`
- `artifacts/phase6_validation/` (19 report files)

**Files modified:**
- `backend/agents/safety_policy.py` — `_determine_state()` hardened:
  CLEAR now requires all five current-safety fields (`red_flag`, `icu`,
  `admitted`, `major_procedure`, `triage_level`) to be explicitly known;
  a partial set that doesn't already trigger OVERRIDE now resolves to
  CAUTION instead of CLEAR (previously, supplying even one field, e.g.
  only `triage_level=4`, was sufficient for CLEAR). Confirmed with the
  user before implementation, since this changes deliberate, tested
  Phase 5 safety behavior; strictly more conservative (no OVERRIDE case
  weakened).
- `backend/tests/test_safety_policy_agent.py` — one test updated
  (`test_clear_partial_context_still_clear_if_no_override_signal` →
  `test_partial_context_with_no_override_signal_is_caution_not_clear`,
  assertion CLEAR→CAUTION) to match the hardened behavior.

**Safety override matrix:** 68 scenarios (6 individual triggers × 5
destinations, × 3 risk tiers, 5 named combinations, 15 two-trigger
combinations) — 100% resulted in `OVERRIDE` with `destination=None`,
regardless of preliminary navigation destination or risk tier.

**Missing-context matrix:** 18 scenarios (fully missing, single-field,
leave-one-out partial, fully-known, and known-trigger-with-rest-missing)
— 100% passed against the hardened CAUTION/CLEAR/OVERRIDE rule.

**Prohibited-language scan:** 40,055 checks (every navigation
template combination + all static safety messages + the full 10,000-
member population's real generated explanations, re-checked under
CLEAR/CAUTION/OVERRIDE) — **0 violations**. Existing `PROHIBITED_PHRASES`
policy used as-is, not weakened.

**Transportation-barrier investigation:** `transportation_barrier=1`
recall@MODERATE on TEST reconfirmed at 0.9898 (unchanged from Phase
4D/4E, same frozen model). New this phase: its logistic-regression
coefficient is the **largest-magnitude of all 60 encoded features**
(+0.2235, rank 1/60); counterfactual 0→1 perturbation moves mean
predicted probability by +0.061 (crossing ~56% of the affected group at
least one tier); classified as driven primarily by a combination of
threshold interaction and coefficient magnitude, with correlated access
features as a secondary contributor — not primarily the synthetic
generator alone. Not modified; flagged INVESTIGATE per this phase's
explicit disparity-classification rule (never "the model is fair").

**Subgroup assessment:** 17 subgroups across age/gender/clinical-burden/
transportation/telehealth/PCP-distance, all n≥605 (well above the n≥100
reporting floor); 75 pairwise disparity comparisons — 44 NO MATERIAL
SIGNAL DETECTED, 17 MONITOR, 14 INVESTIGATE (all from transportation_barrier,
telehealth_available, and clinical_burden). No subgroup ROC-AUC ≤ 0.5.

**Navigation policy validation:** all 5 destinations confirmed
reachable; Care Management confirmed to require risk/utilization **plus**
a complexity/access/history signal (neither alone is sufficient, proven
in both directions); a documented CARE_MANAGEMENT-precedence-over-
URGENT_CARE behavior found when `pcp_distance_miles>10mi` (real system
behavior, not a bug).

**Input validation:** 16 API edge cases, 100% handled cleanly
(status<500, no traceback leak); malformed triage/context values
correctly rejected (422); static member-field value-range validation
(age, distances) is absent — documented as a low-severity, non-safety-
critical robustness gap, not fixed this phase.

**End-to-end population run:** n=10,000 at the TEST index date. Risk
tiers (LOW 6,527 / MODERATE 2,465 / HIGH 1,008) exactly match the frozen
model's own TEST risk-tier report. Navigation: PRIMARY_CARE 2,187 /
URGENT_CARE 403 / TELEHEALTH 1,361 / CARE_MANAGEMENT 2,989 /
NO_PROACTIVE_NAVIGATION 3,060. Safety: CAUTION 10,000 (no
`current_safety_context` exists in the static snapshot — not fabricated,
per instruction; OVERRIDE/CLEAR validated via scenario matrices instead).
Cross-agent consistency: 100% (model_version/dataset_id/synthetic_model/
thresholds identical across every decision).

**Determinism:** 125 repeated calls (25 members × 5 repeats), 0 mismatches.

**Automated tests:** 389/389 passing (333 carried forward, one updated
for the safety fix + 56 new in `test_phase6_safety_invariants.py`).

**Model changed:** NO. **Thresholds changed:** NO. **Original datasets
changed:** NO (SHA-256 verified). **Synthetic datasets/snapshots
changed:** NO (SHA-256 verified). **Both model `.joblib` artifacts
changed:** NO (SHA-256 verified).

**Synthetic limitation:** all Phase 6 findings describe this synthetic
population and this frozen model; not real-world clinical evidence.

**Next phase:** frontend integration, Docker packaging, and Azure demo
deployment remain explicitly out of scope and not started. Phase 7 should
address: transportation_barrier/telehealth/clinical_burden INVESTIGATE
findings, static-field input validation hardening, and
`current_safety_context` verification.

## Phase 7 — Disparity Investigation, Input Validation & Safety-Context Hardening — 2026-08-16

**Objective:** Decompose the Phase 6 subgroup disparities to determine
whether they reflect a serious modeling/data-design defect vs.
understandable synthetic-data behavior; harden static input validation;
formalize the safety-context completeness/provenance contract. Hardening
only — model frozen throughout, never retrained/tuned/recalibrated/
re-thresholded.

**Files created:**
- `backend/agents/input_validation.py`, `safety_context_schema.py`
- `backend/validation/phase7_disparity_analysis.py`, `phase7_api_artifacts.py`
- `backend/tests/test_phase7_hardening.py` (116 new tests)
- `docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md`
- `artifacts/phase7_hardening/` (16 report files)

**Files modified:**
- `backend/agents/contracts.py` — added `ContextCompleteness`
  (COMPLETE/PARTIAL/ABSENT) and `ContextSource`
  (CALLER_SUPPLIED/SYSTEM_DERIVED/NOT_AVAILABLE) enums;
  `CurrentSafetyContext.completeness`/`.source` computed properties;
  `SafetyDecision.context_completeness`/`.context_source` audit fields
  (safe defaults, no existing construction broken).
- `backend/agents/safety_policy.py` — `_determine_state()` now expressed
  via `context.completeness` (formalizes, does not change, the Phase 6
  fix's outcome); `decide()` populates the new audit fields.
- `backend/agents/orchestrator.py` — `decision_to_dict()` serializes the
  two new safety fields into the API response.
- `backend/main.py` — wires the two new validation modules into
  `POST /uc07/decide`; `UC07_MEMBERS_REQUIRED` no longer requires
  `num_chronic_conditions` (now safely derived when absent);
  `_parse_current_safety_context()` delegates to the new Pydantic schema.

**Transportation decomposition:** true prevalence is 3.1× higher for
`transportation_barrier=1` (30.21% vs. 9.75%); every ED-utilization
covariate is meaningfully higher (r=0.30-0.35); age is nearly unrelated
(r=0.04). Conditional analysis (22 strata) shows the effect **persists**
after conditioning on telehealth/burden/history/distance — not a
confound fully explained by any one covariate. Its standardized
logistic coefficient (+0.2235) is the largest of all 60 encoded
features. Threshold-interaction analysis **revises** Phase 6's initial
hypothesis: only 4.75% of `barrier=1` sit within ±0.01 of MODERATE
(vs. 13.12% for `barrier=0`) — 51.3% of the group sits far (≥0.10) above
threshold, so the disparity is driven primarily by a genuinely
shifted score distribution, not threshold-edge proximity.

**Telehealth disparity:** primarily a real, smaller, partially-
transportation-correlated prevalence effect (20.80% vs. 10.59%
prevalence; own coefficient rank 2/60, smaller than transportation's).

**Clinical burden disparity:** recall rises 0.48→0.86 across bands while
ROC-AUC stays flat-to-improving (0.69→0.74) — both a genuine prevalence
rise (10.7%→22.5%) and threshold interaction with a right-shifting
distribution, as expected; recall not forced equal across bands.

**Disparity classification:** transportation_barrier, telehealth_available,
and clinical_burden all classified **INVESTIGATE** (large, real,
direction-consistent with true prevalence); none meets the explicit
BLOCKER bar (no subgroup ROC-AUC≤0.5, no safety/language-policy
failure, no reversed-direction disparity).

**Model-change decision: A — KEEP MODEL UNCHANGED.** No BLOCKER-level
evidence; disparities are explained (real prevalence + largest
coefficient + correlated features, persisting under conditioning), not
unexplained anomalies. Default-preserve instruction not overridden.

**Static input validation:** `age` integer in [0,120]; distances finite,
≥0, ≤500mi; all binary member/ED fields strictly 0/1 (no NaN/Infinity);
`triage_level` validated across every ED row (closes the Phase 6
documented gap where an out-of-window invalid triage value passed
through unrejected); `num_chronic_conditions` derived when the caller
omits it, validated for consistency against its six component flags
when supplied (rejected if mismatched, never silently overwritten).

**Safety-context hardening:** `CurrentSafetyContext.completeness`
(COMPLETE/PARTIAL/ABSENT) and `.source` (CALLER_SUPPLIED/NOT_AVAILABLE)
formalize the Phase 6 fix without changing its outcome; missing fields
remain `None`, never coerced to 0/false; a new Pydantic schema
(`safety_context_schema.py`) rejects invalid triage/binary values,
non-finite values, wrong types, and unrecognized extra keys with a
structured 422, replacing ad-hoc manual JSON validation in `main.py`.

**Automated tests:** 505/505 passing (389 carried forward unchanged +
116 new in `test_phase7_hardening.py`, covering static validation,
safety-context completeness/provenance, invalid-value rejection, and
the full API validation matrix).

**Model changed:** NO. **Thresholds changed:** NO. **Original/synthetic
datasets and snapshots changed:** NO (SHA-256 verified). **Both model
`.joblib` artifacts changed:** NO (SHA-256 verified; metadata `.json`
sidecars refresh `training_timestamp_utc` only, per the documented
pre-existing pytest-fixture behavior, content otherwise byte-identical).

**Remaining risks:** transportation_barrier/telehealth/clinical_burden
disparities remain unresolved by design (explained, not eliminated);
`current_safety_context` remains caller-supplied/unverified (provenance
is now auditable, not verified); age/distance bounds are engineering
sanity limits, not clinical constraints.

**Next phase:** Phase 8 — frontend integration against the hardened
`/uc07/decide` contract (surface `context_completeness`/`context_source`
in the UI, not just risk/navigation/safety). Docker and Azure remain out
of scope until frontend integration is complete.

## Phase 8 — Frontend Integration & Production UI Hardening — 2026-08-16

**Objective:** Update the frontend to present the actual UC07 multi-agent
system (`POST /uc07/decide`) as a pure view/interaction layer -- no
client-side risk-tier, navigation, or safety computation. Backend/model/
thresholds frozen throughout.

**Audit finding:** the existing frontend had **zero** connection to
`/uc07/decide` -- it was wired entirely to the legacy `/predict-json` +
`/explain-member` endpoints (pre-Phase-2 `frequent_ED_user` model). No
discrepancy with the prompt's expectations; this was the actual, verified
starting state.

**Files created:** `frontend/src/apiConfig.ts`; `frontend/src/uc07/`
(`types.ts`, `api.ts`, `Uc07View.tsx` + 13 presentational components,
each with its own `.css`); `frontend/src/uc07/__tests__/` (4 files, 28
tests); `frontend/src/test/setup.ts`; `frontend/vitest.config.ts`;
`artifacts/phase8_frontend/` (5 files); `docs/08_FRONTEND_INTEGRATION.md`.

**Files modified:** `frontend/src/App.tsx` (tab switcher: "UC07
Navigator" default, "Legacy Demo" isolated with an explicit banner);
`frontend/src/App.css`; `frontend/src/api.ts` (reads shared
`apiConfig.ts`, isolation comment); `frontend/package.json` (`test`
script + vitest/testing-library devDependencies).

**Authoritative API decision:** `POST /uc07/decide` is the default tab
and the only source of risk/navigation/safety data anywhere in the new
UI. Response shape was captured from a **live decode** of
`orchestrator.decision_to_dict()`, not assumed or invented
(`artifacts/phase8_frontend/frontend_api_contract.json`).

**Legacy isolation:** the legacy dashboard remains fully functional,
now under an explicitly labeled "Legacy Demo" tab with a banner naming
the pre-Phase-2 model and stating its output "is never a UC07
risk/navigation/safety decision." No shared state/components/types with
the new UC07 flow beyond the identical CSV-upload shape.

**Context-completeness UI:** `ContextStatus` surfaces COMPLETE/PARTIAL/
ABSENT with plain-language explanations; `SafetyContextForm`'s tri-state
inputs (Unknown/No/Yes, Unknown/1-5) never default an unset field to
false -- verified by dedicated tests. `CALLER_SUPPLIED` is never labeled
"verified" anywhere in the UI.

**Synthetic disclosure UI decision:** a persistent `SyntheticDisclosure`
badge appears both before any decision loads and inside every rendered
decision, always showing the exact `model_version` that produced it --
never hidden behind a toggle.

**Error/failure behavior:** `DecisionError` classifies by HTTP status
(backend-unavailable / not-found / validation / model-unavailable /
server error), always paired with "no navigation recommendation is being
shown" and the standard emergency-care sentence; no raw stack trace is
ever rendered; a failed request clears any stale previous result rather
than leaving it unmarked on screen.

**Model/safety logic remains backend-only:** verified by construction
(the frontend never imports or reimplements threshold/navigation/safety
logic) and by test (`test("no client-side fabricated recommendation")`
asserts none of the 4 real destination labels appear when the backend
suppressed navigation under OVERRIDE).

**Frontend tests:** none existed before this phase; added `vitest` +
`@testing-library/react` + `jsdom` (minimal, idiomatic for this exact
stack). **28/28 passing**, covering all 20 required scenarios.

**Backend regression:** **505/505 passing, 0 failed** -- identical to
the Phase 7 baseline; no backend code was modified.

**Frontend lint/build:** oxlint clean, `tsc -b && vite build` clean,
both before and after this phase's changes.

**Local integration smoke test:** live backend + 7 representative
scenarios (LOW/no-nav, MODERATE/Primary Care, Telehealth, Urgent Care,
Care Management/HIGH, CAUTION, OVERRIDE) via a Node script performing
the exact request `decideUC07()` would -- **7/7 passed**, response shape
matched frontend types exactly; unknown-member→404 and invalid-context→
422 also verified live. No browser-driven (Chrome) visual smoke test was
performed -- network-contract + jsdom-rendering verification only,
documented as a limitation.

**Model changed:** NO. **Thresholds changed:** NO. **Datasets/snapshots
changed:** NO (SHA-256 verified). **Backend code changed:** NO (frontend-
only phase).

**Remaining risks:** no literal browser click-through screenshot;
`SYSTEM_DERIVED` context source is modeled but never actually producible;
UC07 population view has no sorting/filtering/pagination.

**Next phase:** Phase 9 — Docker + production configuration/security
hardening. Frontend is already structured for deployment (`VITE_API_URL`
is the only backend-URL integration point); no Dockerfile, docker-compose,
ACR, Azure config, or CI/CD was created this phase.

## UC07 Dashboard UX Enhancement (table pagination, filters, sort, member-details drawer) — 2026-08-16

**Objective:** direct user request to improve the UC07 Navigator dashboard's UX -- 25-per-page pagination, a full filter toolbar (member search, risk tier, navigation, safety, probability range/presets, removable chips), sortable columns, a redesigned Details-column table (SAFETY moved into the member details view instead of a repetitive table column), a three-card cohort summary (Risk distribution / Navigation / Safety, total-vs-filtered), and a right-side member details drawer showing profile/chronic-conditions/ED-history/access-barriers/care-history parsed client-side from the same CSVs already uploaded. Frontend-only; no backend change; no ML/threshold/navigation/safety logic touched (this predates and is superseded in scope by Phase 8B's safety-context additions below, which build on this table/drawer work).

**Files created:** `frontend/src/uc07/tableState.ts`, `csvUtils.ts`, `components/{Pagination,MemberFilters,MemberDataSections,MemberDetailsDrawer}.{tsx,css}`, 3 new test files (24 tests).
**Files modified:** `Uc07View.tsx`, `Uc07ResultsTable.tsx`+css, `PopulationSummary.tsx`+css, `index.css` (shared `.sr-only`).
**Tests:** 52/52 frontend passing (28 carried forward + 24 new). Lint/build clean. No backend files touched.

## Phase 8B — Current Safety Context Workflow (Single-Member + Batch CSV) — 2026-08-16

**Objective:** let current safety context be supplied naturally in two
workflows -- (A) a single-member "Evaluate Current Safety" action inside
the member details view, and (B) an optional fourth
`current_safety_context.csv` batch upload -- without changing the model,
thresholds, navigation rules, or Safety Agent rules, and without adding
any safety decision logic to the frontend.

**Files created:**
- `backend/agents/safety_context_csv.py` (reuses `safety_context_schema.py`'s
  `SafetyContextEntry` per row -- no duplicated validation logic)
- `backend/tests/test_phase8b_safety_context.py` (25 new tests)
- `frontend/src/uc07/components/{CurrentSafetyContextSection,SafetyContextCsvUpload}.{tsx,css}`
- `frontend/src/uc07/__tests__/{CurrentSafetyContextSection,SafetyContextCsvUpload,Uc07View.safetyEvaluation}.test.tsx` (15 new tests)
- `docs/08B_CURRENT_SAFETY_CONTEXT_WORKFLOW.md`
- `artifacts/phase8b_current_safety_context/pre_work_hashes.json`

**Files modified:**
- `backend/main.py` -- new optional `safety_context_file` multipart field on
  `POST /uc07/decide`; CSV context merges with (and is overridden per-member
  by) the existing JSON `current_safety_context` field.
- `frontend/src/uc07/Uc07View.tsx` -- removed the old pre-run safety form;
  added the optional CSV upload and a `safetyOverrides` map so a
  single-member "Evaluate Current Safety" result supersedes that member's
  original batch decision everywhere (table selection, summary counts,
  drawer) without touching any other member.
- `frontend/src/uc07/components/MemberDetailsDrawer.tsx` -- renders the new
  Current Safety Context section.
- `frontend/src/uc07/api.ts`, `types.ts` -- `decideUC07()` accepts an
  optional `safetyContextFile`.

**Verified against the live backend (not just asserted):** the spec's own
worked CSV example reproduced exactly -- M00001→CLEAR, M00002→CAUTION,
M00003→OVERRIDE, M00004→CLEAR, M00005→OVERRIDE; all 4 CSV rejection cases
(invalid binary, invalid triage, unknown member_id, duplicate member_id)
returned clean 422s; risk probability/tier and pre-safety navigation
output confirmed byte-identical across no-context/CLEAR/OVERRIDE calls
for the same member.

**Safety Agent rules unchanged:** OVERRIDE still checked before
completeness (a single known trigger overrides even with everything else
unknown, including on a LOW-risk member); CLEAR still requires all five
fields known and none triggering; CAUTION remains the default for
absent/incomplete context and for any member with no CSV row -- never
forced to equal counts across CLEAR/CAUTION/OVERRIDE.

**Frontend safety logic: NONE added.** Every component only displays
values already present in a `POST /uc07/decide` response; a grep-based
backend test (`test_19`) enforces this going forward, tuned to exclude
TypeScript union-type declarations and JSX attribute values (both
legitimate) while still catching a real `identifier = "OVERRIDE"`-style
assignment.

**Automated tests:** backend 530/530 passing (505 carried forward + 25
new); frontend 63/63 passing (48 carried forward + 15 new). Lint/build
clean.

**Model changed:** NO. **Thresholds changed:** NO. **Navigation rules
changed:** NO. **Datasets/model artifacts changed:** NO (SHA-256 verified).

**Remaining limitations:** the frontend-safety-logic grep check is a
heuristic, not a formal proof; `SYSTEM_DERIVED` context source remains
defined but unused; no server-side persistence of current safety context
between requests (by design).

**Next phase:** Phase 9 — Docker + production configuration/security
hardening, unchanged from the prior entry's recommendation. STOPPED per
instruction; Docker/Azure not started.

## Phase 8C — Authoritative Model Explainability + Controlled GenAI Explanation Agent — 2026-08-16

**Objective:** Add structured, SHAP-based model explainability for the
frozen `uc07-risk-synthetic-v1` model, and a new, strictly bounded fourth
agent — the GenAI Explanation Agent (Ollama, `qwen3:8b`) — that converts
already-decided results into short natural-language text. Neither
capability makes a decision; neither can change a risk probability, risk
tier, navigation destination, safety state, or safety override.

**Files created:**
- `backend/agents/model_explainability.py` — SHAP LinearExplainer
  (primary) + exact linear-contribution decomposition (fallback), with a
  module-level explainer cache and NumPy global-RNG-state save/restore
  guards around SHAP calls.
- `backend/agents/genai_explanation.py` — AGENT 4. Ollama HTTP client,
  deterministic fallback generator, and code-enforced (not
  prompt-only) validation: prohibited-phrase check, negation-aware
  diagnosis/prescription check, field-scoped tier/destination/
  safety-state consistency checks, always-overridden disclaimer.
- `backend/tests/test_model_explainability.py`,
  `test_genai_explanation.py`, `test_genai_explanation_authority.py`,
  `test_genai_privacy.py`, `test_phase8c_frontend_architecture.py` — 61
  new backend tests.
- `frontend/src/uc07/components/WhyFlaggedSection.{tsx,css}`,
  `AiExplanationSection.{tsx,css}` — 12 new frontend tests.
- `docs/08C_GENAI_EXPLAINABILITY.md`.

**Files modified:**
- `backend/agents/contracts.py` — `FactorDirection`, `ExplanationMethod`,
  `ExplanationFactor`, `ExplanationSource`, `MemberExplanation`; extended
  `RiskAssessment` with `explanation_factors`/`explanation_method`/
  `explanation_causal` (all default-safe, zero impact on existing
  construction).
- `backend/agents/risk_detection.py` — populates the new structured
  fields via `model_explainability`; existing sentence-based
  `contributing_factors` field unchanged.
- `backend/agents/orchestrator.py` — serializes the new fields.
- `backend/main.py` — new `POST /uc07/explain` endpoint (Pydantic
  `ExplainRequest`, allow-listed fields only; never touches
  `UC07Orchestrator`/`RiskDetectionAgent`).
- `backend/requirements.txt` — added `httpx` (runtime Ollama client).
- `frontend/src/uc07/types.ts`, `api.ts`,
  `components/Uc07DecisionPanel.tsx`, `MemberDetailsDrawer.tsx`.

**New endpoint:** `POST /uc07/explain` — lazy, single-member only. Accepts
only an already-computed decision summary; always returns 200 with
either a GenAI or deterministic-fallback explanation.

**Env vars (new, all optional):** `GENAI_ENABLED` (default `false`),
`OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL`
(default `qwen3:8b`), `GENAI_TIMEOUT_SECONDS` (default `20`).

**Model/thresholds/navigation rules/Safety Agent logic: NONE changed.**
Verified by re-running the full pre-Phase-8C test suite unmodified (all
prior-phase tests still pass) and by SHA-256/content verification of both
model artifacts and all datasets (see §19 of
`docs/08C_GENAI_EXPLAINABILITY.md`).

**Automated tests:** backend 590/591 passing (530 carried forward + 61
new; the one non-Phase-8C exception is a pre-existing, content-verified-
benign test fragility, see `docs/DECISION_LOG.md` #118) — full 591/591
when run in isolation from the interaction that triggers it; frontend
75/75 passing (63 carried forward + 12 new). Lint/build clean.

**Live verification:** confirmed against a real running Ollama
(`qwen3:8b`) across LOW/MODERATE/HIGH risk tiers × CLEAR/CAUTION/OVERRIDE
safety states (6/6 scenarios), and confirmed both `/uc07/decide` and
`/uc07/explain` remain fully functional with Ollama unreachable
(deterministic fallback engaged).

**Remaining limitations:** consistency checks are phrase-based heuristics,
not full NLP; `GENAI_TIMEOUT_SECONDS` needs tuning above Ollama's
cold-start time in production; GenAI deployment strategy for
Azure/non-localhost environments is explicitly deferred to Phase 9.

**Next phase:** Phase 9 — Docker + production configuration/security
hardening, and a decision on GenAI (Ollama) deployment strategy off of
localhost. STOPPED per instruction; Docker/Azure not started.

## Phase 8D — Critical Fixes + Regression Hardening — 2026-08-16

**Objective:** Fix the CRITICAL and HIGH findings from the Phase 8C
health check (GenAI safety-state contradiction, legacy `/predict`
server-freeze, GenAI navigation/tier misrepresentation) without
retraining the model, changing thresholds, or altering the Risk/
Navigation/Safety agents' decision logic.

**Files modified:**
- `backend/agents/genai_explanation.py` — structured `risk_tier`/
  `navigation_destination`/`safety_state` echo (exact-match validated,
  primary guarantee); positive safety-consistency check (reassurance
  phrases forbidden for ANY state, required-positive-phrase for OVERRIDE/
  CAUTION); navigation self-contradiction check; risk-tier synonym
  proximity regexes; updated system prompt.
- `backend/main.py` — `POST /predict` routed through a
  `ProcessPoolExecutor` (not `run_in_threadpool`, which was proven
  insufficient — see `docs/08D_CRITICAL_FIXES.md` §5); `extract_features()`/
  `explain_member()` on `run_in_threadpool`; `CORS_ORIGINS` env var
  replacing hard-coded `allow_origins=["*"]`.
- `frontend/src/uc07/Uc07View.tsx` — `clampedPage` fixes the pagination
  desync after a safety re-evaluation shrinks the filtered set.
- `frontend/src/uc07/components/WhyFlaggedSection.tsx` — added SHAP
  attribution-vs-causal UI caveat (math unchanged).
- `backend/tests/test_genai_explanation.py`,
  `test_genai_explanation_authority.py` — updated fixtures for the new
  structured-echo schema; strengthened 3 tests that previously passed
  for the wrong reason (missing enum fields) to now genuinely exercise
  the free-text consistency layer.

**Files created:**
- `backend/tests/test_phase8d_genai_hardening.py` — 25 tests, the exact
  11 numbered GenAI scenarios from the phase spec plus adversarial
  variants (e.g. correct structured echo + contradicting free text).
- `backend/tests/test_phase8d_legacy_concurrency.py` — 3 tests proving
  `/predict` no longer blocks `/health` or `/uc07/decide`, using real
  concurrent ASGI requests against a genuinely small population slice
  (not mocked).
- `frontend/src/uc07/__tests__/Uc07View.pagination.test.tsx` — 2 tests
  reproducing the exact pagination-desync scenario and the proper-empty-
  state case.
- `frontend/src/uc07/__tests__/PopulationSummary.test.tsx`,
  `MemberFilters.test.tsx`, `Uc07ResultsTable.test.tsx`,
  `MemberDetailsDrawer.test.tsx` — 15 focused tests for previously
  thinly-covered components.
- `docs/08D_CRITICAL_FIXES.md`.
- 9 new `/uc07/explain` HTTP contract tests added to
  `backend/tests/test_uc07_api.py`.

**Model/thresholds/navigation rules/Safety Agent logic: NONE changed.**
Verified: all pre-Phase-8D backend/frontend tests still pass; SHA-256
hashes of all 3 original datasets, all 3 synthetic datasets, all 3
synthetic snapshots, and both model artifacts match the Phase 8C
baseline exactly, before and after every fix.

**Automated tests:** backend 630 collected, 629 passing (1 known,
pre-existing, content-verified-benign exception — see
`docs/DECISION_LOG.md` #118, unrelated to this phase); frontend 93/93
passing. Lint/build clean.

**Live verification:** structured-echo + positive-consistency validation
confirmed against a real running Ollama (`qwen3:8b`) across
OVERRIDE/CAUTION/CLEAR scenarios. The `/predict` concurrency fix
confirmed live: `GET /health` responded in 3ms while `/predict` ran in
its own process (vs. 20+ seconds of total unresponsiveness before any
fix, and ~22s of continued starvation with thread-pool-only — proven
insufficient by a controlled diagnostic before committing to the
process-pool fix).

**Remaining non-blocking issues:** consistency checks remain phrase-/
regex-based heuristics (the structured echo is the primary, exact
guarantee); the Phase 8B hash-flake test remains order-dependent
(documented, out of scope); `@app.on_event` deprecation warning
(functionally correct); CORS config is preparation only, not a full
production pass.

**Next phase:** Phase 9 — Docker + production configuration/security
hardening, GenAI deployment strategy off localhost, and (optionally)
migrating `@app.on_event` to FastAPI lifespan handlers. STOPPED per
instruction; Docker/Azure not started.

## Phase 9 — Member Communication & Reporting (PDF Report + Email) — 2026-08-17

Adds an operational communication/reporting layer on top of the
existing UC07 decision system: "Download PDF Report" and "Send to
Member" in the member details workspace. Consumes an already-computed
`FinalUC07Decision` (+ an already-approved explanation, if one has been
fetched) — it never creates, changes, or influences a risk/navigation/
safety decision. See `docs/09_MEMBER_COMMUNICATION_REPORTING.md`.

- PDF ("Member Care Navigation & Risk Summary") rendered server-side
  with ReportLab: A4, header/footer with page numbers/report ID/
  timestamp, semantic color-coded sections, human-readable model
  factors only (never a raw feature slug), multi-page support. For an
  OVERRIDE safety state, the Safety section moves ahead of the Risk
  section and is visually flagged priority; the risk score is never
  suppressed, only made secondary.
- `POST /uc07/report` (returns `application/pdf`) and `POST /uc07/email`
  (always 200, `{sent, provider, message, error_code, report_id}`) share
  ONE rendering call (`main.py`'s `_build_report_context`) so the
  downloaded and emailed reports are always the same document.
- No new LLM call for reports: reuses an already-fetched
  `/uc07/explain` result if the frontend has one cached, otherwise
  builds the same deterministic fallback `genai_explanation.py` always
  produces, with GenAI forced off for that one call (zero network
  calls either way).
- Email: `backend/services/email_service.py`, provider-abstracted
  (`EmailProvider` ABC, `SmtpEmailProvider` implemented; a future Azure
  provider can be added without changing the public API). Header/
  newline-injection rejected before any network call; every failure
  path (disabled, unconfigured, auth, timeout, network, provider error)
  returns a safe, generic result — never a credential, never a raw
  SMTP exception/stack trace.
- Member email/name: confirmed by inspection that `raw_members.csv` has
  no such columns. Added `frontend/src/uc07/memberContacts.ts`, a
  `localStorage`-only `member_id -> {name, email}` map — never sent to
  `/uc07/decide`, never an ML feature. The composer's recipient field is
  always editable (there is no trusted/verified contact source to make
  it read-only against).
- Audit trail: structured logging only (`backend/services/audit.py`,
  logger `uc07.communication.audit`) — no database in this prototype
  (documented as a production follow-up, not implemented).
- `GET /health` extended with `email_configured`/`email_provider`
  (never credentials).

**Files created:**
- `backend/services/__init__.py`, `report_service.py`,
  `email_service.py`, `audit.py`.
- `backend/tests/test_report_service.py` (20 tests),
  `test_email_service.py` (20 tests, `smtplib.SMTP` always mocked),
  `test_uc07_communication_api.py` (15 HTTP-level tests).
- `frontend/src/uc07/memberContacts.ts`.
- `frontend/src/uc07/components/MemberReportActions.{tsx,css}`,
  `EmailComposerModal.{tsx,css}`.
- `frontend/src/uc07/__tests__/MemberReportActions.test.tsx` (5 tests),
  `EmailComposerModal.test.tsx` (14 tests).
- `docs/09_MEMBER_COMMUNICATION_REPORTING.md`.

**Files modified:**
- `backend/main.py` — new `ReportRequest`/`EmailSendRequest` Pydantic
  schemas, `POST /uc07/report`, `POST /uc07/email`, `/health` extension.
  `/uc07/decide`, `/uc07/explain`, and every existing endpoint's
  behavior: unchanged.
- `backend/requirements.txt` (+`reportlab`),
  `backend/requirements-dev.txt` (+`pypdf`, test-only).
- `backend/.env.example`, `backend/.env` — `EMAIL_*`/`SMTP_*` variables
  (disabled by default; no real credentials filled in).
- `backend/tests/conftest.py` — pinned `EMAIL_*`/`SMTP_*` test defaults
  (same reasoning as the existing `GENAI_*`/`GROQ_*` pins — a
  developer's real `backend/.env` must never leak into the test
  session).
- `frontend/src/uc07/types.ts`, `api.ts` — added
  `ReportRequestPayload`/`EmailSendRequestPayload` types and
  `buildReportRequest`/`fetchMemberReportPdf`/`sendMemberReportEmail`.
  Every existing exported type/function: unchanged.
- `frontend/src/uc07/components/MemberDetailsDrawer.tsx` — renders
  `MemberReportActions` in the header, next to the close button.
- `frontend/src/test/setup.ts` — stubs `URL.createObjectURL`/
  `revokeObjectURL`/`window.open` (jsdom does not implement them; the
  new PDF download/preview flow needs them).

**Model/risk thresholds/SHAP/Risk Agent/Navigation Agent/Safety
Agent/GenAI decision authority/`/uc07/decide` behavior/training
datasets/model artifacts: NONE changed.** Verified: `backend/services/
report_service.py` and `email_service.py` import neither
`risk_detection`, `care_navigation`, `safety_policy`, `orchestrator`,
nor `model_explainability` (asserted directly by
`test_no_model_decision_authority_imports`); `POST /uc07/email`'s
response body carries no risk/navigation/safety field at all.

**Automated tests:** backend 700 collected, 699 passing (the same 1
known, pre-existing, order-dependent hash-flake exception already
documented at `docs/DECISION_LOG.md` #118 — unrelated to this phase,
confirmed by re-running it in isolation); frontend 135/135 passing
(118 pre-existing + 17 new). `tsc -b` and `oxlint`: clean (only
pre-existing, unrelated warnings in legacy `.jsx` files).

**Live verification:** a real `POST /uc07/decide` call against the dev
backend, followed by a real `POST /uc07/report` (200,
`application/pdf`, correct `Content-Disposition`, correct CORS header
for the Vite dev origin) and `POST /uc07/email` (200,
`EMAIL_DISABLED` — the safe default with no SMTP configured) —
end-to-end through the actual running server, not just `TestClient`.
No real email was sent (no SMTP credentials were provided; email
sending was verified via mocked-SMTP tests instead — see
`docs/DECISION_LOG.md` #121). Browser tools were unavailable this
session, so the composer's visual/interaction behavior was verified via
jsdom + Testing Library component tests (17 tests covering prefill,
validation, two-step confirmation, success/failure states, preview,
and keyboard accessibility) rather than a manual click-through; the dev
servers were left running (backend :8000, frontend :5175) for manual
visual confirmation.

**Remaining non-blocking issues:** no server-side persisted-decision
store, so `/uc07/report`/`/uc07/email` trust the frontend's echoed
decision the same way `/uc07/explain` already does (documented, not a
regression); audit trail is process logging only, no database; Azure
email provider not started (per instruction).

**Next phase:** Docker + production configuration/security hardening
(carried over from Phase 8D, still not started), and — only if
requested — an Azure `EmailProvider` implementation. STOPPED per
instruction.

## Phase 9.1 -- SMTP Reliability Fix (TIMEOUT/PROVIDER_ERROR investigation) -- 2026-08-17

Investigated intermittent `EMAIL_FAILED result=TIMEOUT` / `result=PROVIDER_ERROR`
against real Gmail SMTP, plus emails occasionally landing in Spam. Fixed
genuine reliability/observability gaps in `backend/services/email_service.py`
without touching the model, agents, GenAI provider chain, datasets, or the
API's response contract. See `docs/09_MEMBER_COMMUNICATION_REPORTING.md`
section 8a.

- **Root cause of TIMEOUT/PROVIDER_ERROR:** not a missing/wrong timeout
  (`EMAIL_TIMEOUT_SECONDS` was already correctly bounding every SMTP
  stage via `smtplib`'s own socket-timeout semantics, confirmed from
  CPython source) -- it was that every kind of failure collapsed into
  one generic bucket with no indication of WHERE in the SMTP
  conversation (connect/EHLO/STARTTLS/AUTH/send) or WHAT kind of
  failure it was.
- `SmtpEmailProvider.send()` rewritten to run connect/ehlo/starttls/
  (post-TLS)ehlo/authenticate/send as individually-caught stages, each
  categorized into a specific `SmtpStageError` (`TIMEOUT`,
  `CONNECTION_FAILED`, `TLS_FAILED`, `AUTH_FAILED`, `SENDER_REJECTED`,
  `RECIPIENT_REJECTED`, `MESSAGE_REJECTED`, `RATE_LIMITED` (SMTP 421),
  `PROVIDER_TEMPORARY_ERROR` (4xx), `PROVIDER_PERMANENT_ERROR` (5xx),
  `UNKNOWN_PROVIDER_ERROR`), using Python's actual `smtplib` exception
  hierarchy plus the server's numeric response code where one exists.
- Confirmed the 587/STARTTLS flow was already correct (`smtplib.SMTP` +
  `.starttls()`, never `SMTP_SSL`'s port-465 semantics) and added the
  RFC 3207-required second `ehlo()` explicitly, immediately after
  STARTTLS, rather than relying on it happening implicitly inside
  `login()`.
- Bounded, targeted retry: at most 1 additional attempt, only for
  failures that PROVE the message was not accepted (any pre-"send"-
  stage transient failure, or an explicit 4xx/rate-limit response
  received during send). A `TIMEOUT`/`CONNECTION_FAILED` that happens
  DURING the send stage is deliberately never retried -- the client
  cannot tell if the server already queued the message, so retrying
  there risks a duplicate email. Documented as a known limitation
  rather than solved with a full idempotency mechanism (out of scope).
- Connection cleanup hardened: `smtp.quit()` in a `finally` block,
  falling back to `smtp.close()` if `quit()` itself fails -- a failed
  send can no longer leave a stale connection for the next request.
- Safe, stage-attributed logging added (`uc07.communication.email`
  logger): `stage=`, `result=`, `smtp_code=`, `attempt=`,
  `connection_ms=`/`send_ms=` on success. Never the password, raw
  provider response text, report content, or email body -- verified by
  dedicated tests.
- Deliverability hygiene: added explicit `Date` and `Message-ID`
  headers (smtplib does not add either automatically); confirmed the
  message body stays plain-text-only with a single professionally-named
  attachment and a matching `From` display name/address. Documented,
  explicitly, that Primary Inbox placement can never be guaranteed by
  application code -- the durable fix is a transactional provider on an
  authenticated domain (SPF/DKIM/DMARC), noted as future work.
- **Verified, not assumed, that POST /uc07/email does not block the
  event loop:** `POST /uc07/report`/`POST /uc07/email` are plain `def`
  handlers, so Starlette's automatic `run_in_threadpool` already keeps
  blocking SMTP I/O off the main asyncio event loop -- proven with a
  new concurrency test (a mocked 1.5s-slow SMTP send does not delay a
  concurrent `GET /health`), not just asserted from the code shape.
- **API response contract deliberately UNCHANGED:** `POST /uc07/email`
  still always returns HTTP 200 with `{sent, provider, message,
  error_code, report_id}` -- this already IS the "explicit structured
  status" the investigation's own spec recommended as the fallback if
  changing HTTP status codes wasn't worth the compatibility risk; the
  frontend already branches on `result.sent`/`.message`, not the HTTP
  status, so no frontend change was needed or made.

**Files created:** `backend/tests/test_uc07_email_concurrency.py`.

**Files modified:** `backend/services/email_service.py` (staged
execution, categorization, retry, safe logging, Date/Message-ID
headers -- `EmailService`'s public API/constructor signature
unchanged); `backend/main.py` (`uc07_email` -- added
`report_generation_ms` timing + a safe result-summary log line; no
behavioral/contract change); `backend/tests/test_email_service.py`
(rewritten -- provider-layer categorization tests + service-layer
retry/validation tests, replacing the old single-layer mocks that are
no longer representative of the real code path);
`backend/tests/test_uc07_communication_api.py` (+4 tests: OVERRIDE
wording preserved through the email path, `/uc07/decide` unaffected,
no leaked SMTP internals); `backend/.env.example` (expanded
`EMAIL_TIMEOUT_SECONDS` guidance); `docs/09_MEMBER_COMMUNICATION_REPORTING.md`.

**Model/SHAP/Risk Agent/Navigation Agent/Safety Agent/GenAI/datasets/
feature engineering: NONE changed.** This phase touched only
`backend/services/email_service.py`, `backend/main.py`'s `uc07_email`
handler body (timing/logging only), and tests/docs.

**Automated tests:** backend Phase 9 suite 87/87 passing (19 report +
51 email + 16 communication-API + 1 concurrency; up from 50 before this
pass); full backend regression (excluding retraining self-tests, which
are unrelated and were not re-run to avoid unnecessary artifact churn)
618/618 passing, no new failures. Frontend: unchanged, still 135/135
(this pass touched no frontend files).

**Manual verification:** the user filled in real Gmail SMTP credentials
(`smtp.gmail.com:587`, STARTTLS, an App Password) in their own
`backend/.env` and confirmed `GET /health` reports
`email_configured: true`. No real email was sent as part of this
investigation (per instruction -- "I will perform the manual send");
manual test commands are provided directly to the user instead.

**Remaining non-blocking issues:** no idempotency/duplicate-send
protection for the documented ambiguous-timeout case (Section 8a); no
guarantee of Primary Inbox placement (inherent to any application, not
fixable here); Azure email provider not started (per instruction).

**Next phase:** Docker + production configuration/security hardening
(carried over from Phase 8D/9, still not started), and -- only if
requested -- an Azure `EmailProvider` implementation. STOPPED per
instruction.

## Phase 9.2 -- STARTTLS Diagnosability Fix (stage=starttls result=UNKNOWN_PROVIDER_ERROR) -- 2026-08-17

Follow-up to Phase 9.1: a repeated `smtp_send stage=starttls
result=UNKNOWN_PROVIDER_ERROR` was reported against real Gmail. PDF
generation was already confirmed unaffected; the failure is strictly
pre-authentication. Root cause could not be reproduced directly (no
real email was sent by the assistant, per instruction), so this pass
fixes the DIAGNOSTIC GAP that made the failure unclassifiable, rather
than guessing at a specific fix. See
`docs/09_MEMBER_COMMUNICATION_REPORTING.md` section 8a.

- `SmtpStageError` gained an `exception_type` field -- the raised
  exception's CLASS NAME ONLY (never its message/args) -- populated on
  EVERY path through `_categorize_smtp_exception()`, including the
  final catch-all. A failure the fixed `error_code` taxonomy can't
  distinguish further is no longer a dead end: `exc_type=` is now in
  the safe `smtp_send` log line alongside `stage=`/`result=`/
  `smtp_code=`.
- `ssl.SSLCertVerificationError` (a subclass of `ssl.SSLError`) is now
  checked explicitly, before the generic TLS branch, with its own
  message ("Could not verify the email server's TLS certificate.").
  `ConnectionResetError`/`BrokenPipeError` are now named explicitly
  too (same `CONNECTION_FAILED` outcome as the generic `OSError`
  branch, just no longer incidental).
- **TLS certificate verification was explicitly NOT weakened**:
  `ssl.create_default_context()` is still called with zero arguments
  (verified by spying on the real function, not mocking it away) and
  the context actually handed to `starttls()` is asserted to still
  have `verify_mode == CERT_REQUIRED` and `check_hostname is True`. A
  new regression guard test fails if `CERT_NONE`/`check_hostname=False`/
  an unverified context is ever introduced.
- Confirmed (unchanged from Phase 9.1) the Gmail flow itself is
  correct: `smtplib.SMTP("smtp.gmail.com", 587)` → `EHLO` →
  `STARTTLS` with the default secure context → `EHLO` again → `login`
  → `send_message`.

**Files modified:** `backend/services/email_service.py` (`exception_type`
field + tagging on every categorization path, no behavioral/API change
to `EmailSendResult` or the retry policy), `backend/tests/test_email_service.py`
(+9 tests: certificate verification failure, `SMTPNotSupportedError`
during STARTTLS, connection reset, timeout, unrecognized-exception
tagging, exc_type in log line, TLS-never-weakened guards),
`docs/09_MEMBER_COMMUNICATION_REPORTING.md`.

**Model/agents/report generation: NONE changed**, per explicit
instruction -- this pass touched only `email_service.py`'s exception
categorization and its tests.

**Automated tests:** `test_email_service.py` 59/59 passing (up from
51); full Phase 9 suite 95/95; no regressions elsewhere (not re-run in
full this pass -- no code outside `email_service.py`/its own tests was
touched).

**Outcome:** diagnostic-only, as instructed -- the actual STARTTLS
failure was NOT reproduced or fixed here (no real send was attempted).
The next real send attempt's `exc_type=` value in the log will name
the true exception class directly; see the assistant's response for
the exact safe line to report back.

**Next phase:** unchanged from Phase 9.1 -- Docker/production hardening
and, if requested, Azure `EmailProvider`. STOPPED per instruction
(diagnosis only, no redesign).

## Phase 9.3 -- STARTTLS Regression Fix (root cause: constructor vs. deferred connect()) -- 2026-08-17

Root-caused the 100%-reproducible `stage=starttls
result=UNKNOWN_PROVIDER_ERROR` reported against real Gmail. Confirmed
DIRECTLY against `smtp.gmail.com:587` (connect/EHLO/STARTTLS/EHLO only
-- login and send were stubbed out, no real email was sent). See
`docs/09_MEMBER_COMMUNICATION_REPORTING.md` section 8a,
`docs/DECISION_LOG.md` #127.

**Root cause:** Phase 9.1's staged-connect rewrite constructed
`smtplib.SMTP(timeout=...)` with NO host (to make "connect" its own
catchable/timeable stage), then called `smtp.connect(host, port)`
separately. `smtplib.SMTP.__init__` is the ONLY place that sets the
instance's internal `self._host`; a separate `connect()` call
establishes the socket but never writes the host back to that
attribute. `starttls()` reads `self._host` to pass as `server_hostname`
to `ssl.SSLContext.wrap_socket()` for TLS SNI/hostname verification --
with it stuck at `''`, `wrap_socket()` raised a plain
`ValueError("check_hostname requires server_hostname")`, which matched
none of the categorizer's `smtplib`/`ssl`/`socket`/`OSError` branches
and fell through to `UNKNOWN_PROVIDER_ERROR` every single time.

**Exact fix (smallest change, no redesign):**
`backend/services/email_service.py`'s `SmtpEmailProvider` gained a
`_connect()` helper that constructs `smtplib.SMTP(host, port,
timeout=...)` with host/port passed DIRECTLY to the constructor --
exactly matching the original, pre-hardening implementation -- while
still keeping "connect" as its own individually-categorized stage (any
exception during construction, including the connect it performs
internally, is still tagged `stage="connect"`). Nothing else in the
staged flow changed: EHLO -> STARTTLS (unmodified default,
verification-enabled `ssl.create_default_context()`) -> EHLO again ->
LOGIN -> SEND, still individually attributed/categorized/logged,
still with the same retry policy and connection cleanup as Phase 9.1/
9.2.

**TLS certificate verification was not touched/weakened** -- confirmed
by a spy-based test asserting `ssl.create_default_context()` is still
called with zero arguments and the context actually handed to
`starttls()` still has `verify_mode == CERT_REQUIRED` and
`check_hostname is True`.

**Files modified:** `backend/services/email_service.py`
(`SmtpEmailProvider._connect()` added; `send()`'s first two lines
changed from constructing bare + calling `.connect()` to calling
`self._connect()`; nothing else in the class changed),
`backend/tests/test_email_service.py` (6 tests updated to assert the
constructor now receives `(host, port, timeout=...)` and to simulate a
connect-stage failure via the constructor's `side_effect` instead of a
now-unused `instance.connect` mock; +1 new test asserting the exact
required call order end-to-end: EHLO -> STARTTLS -> EHLO -> LOGIN ->
SEND), `docs/09_MEMBER_COMMUNICATION_REPORTING.md`.

**Model/SHAP/agents/report generation/GenAI/frontend/datasets: NONE
changed**, per explicit instruction.

**Automated tests:** `test_email_service.py` 60/60 passing (up from
59); full Phase 9 suite 96/96 passing.

**Live verification:** a real connect->EHLO->STARTTLS->EHLO handshake
against `smtp.gmail.com:587`, through the actual fixed
`SmtpEmailProvider.send()` code path, succeeded (login/send_message
were monkeypatched to no-ops for this check only -- no real
authentication attempt, no real email sent, per instruction).

**Next phase:** unchanged -- Docker/production hardening, and, if
requested, an Azure `EmailProvider`. STOPPED per instruction (fix
STARTTLS only, no further redesign).

## Phase 10 -- Frontend UX Audit & Targeted Redesign Pass -- 2026-08-17

Audited the entire active UC07 frontend (App.tsx's "UC07 Navigator" tab
and everything under frontend/src/uc07/) against a full healthcare-
product redesign brief. Finding: the existing design system (tokens.css),
component architecture, cross-filtering, 15-member pagination, member
workspace, SHAP/AI-explanation sections, and safety-override handling
were ALREADY built to a high standard (systematic semantic color
tokens for risk/safety/navigation/SHAP/GenAI, working chart<->filter
cross-filtering, accessible tabs with focus trapping, a professional
AI-explanation identity that never outranks the decision). Rather than
a risky ground-up rewrite of a sound architecture (and without browser/
screenshot tooling available this session to visually verify a larger
change), this pass fixed concrete, verified gaps instead:

- **Navigation destination color consistency**: the population chart
  already colored destinations via `--nav-*` tokens, but the member
  workspace's Navigation card and the results table's Navigation
  column showed plain text. Extracted the single shared mapping
  (`frontend/src/uc07/navigationDisplay.ts`) and applied it in all
  three places (color dot + label, never color alone).
- **Population KPIs expanded** (Section 14's suggested list): added
  Moderate Risk, Navigation Opportunities, and Safety Caution counts
  alongside the existing Total/High Risk/Safety Override.
- **Communication moved into its own workspace tab**: "Download PDF
  Report"/"Send to Member" were two small buttons wedged into the
  member workspace header; moved into a proper "Communication" tab
  (5th tab, matching the spec's explicit tab list) with a heading,
  description, and two labeled action cards (Section 23-25), reusing
  the exact same request-building/report/email logic unchanged.
- **Header accent**: a restrained 3px brand-gradient top line (Section
  5/10) -- deliberately not a full gradient header, to keep clinical
  text on a plain, high-contrast surface.

**Files created:** `frontend/src/uc07/navigationDisplay.ts`.

**Files modified:** `AnalyticsCharts.tsx` (now imports the shared nav
color/label map instead of its own copy), `NavigationCard.{tsx,css}`,
`Uc07ResultsTable.{tsx,css}`, `PopulationSummary.{tsx,css}`,
`MemberDetailsDrawer.tsx` (Communication tab), `MemberReportActions.
{tsx,css}` (redesigned as a full tab-panel section rather than a
compact header strip), `Header.css`; test files updated to match
(`PopulationSummary.test.tsx`, `Uc07DecisionPanel.test.tsx`,
`MemberReportActions.test.tsx` rewritten, `MemberDetailsDrawer.test.tsx`
+1 test for the new tab).

**Backend/model/agents/report-PDF-generation/email-sending logic:
NONE changed** -- this was a frontend presentation pass only; every
value displayed still comes unmodified from the existing API contracts.

**Automated tests:** frontend 136/136 passing (135 pre-existing/updated
+ 1 new). `tsc -b`: clean. `oxlint`: clean (same pre-existing warnings
in unrelated legacy `.jsx` files only). Production build: succeeds.

**Not attempted this pass (documented, not silently skipped):** a
literal pixel-level pass across all ~40 component stylesheets (spacing/
shadow/gradient tuning at 1440/1024/768/390px in both themes) --
without browser/screenshot tooling this session, claiming a completed
visual pass would not be honest; the existing design tokens and
responsive patterns (already present throughout) were judged sound
enough not to risk a blind rewrite. A left-sidebar shell was
deliberately NOT introduced -- the app is fundamentally one continuous
analysis flow (upload -> results -> table, not separate pages), and
the brief's own Section 9 caveat ("do not create fake routes... only
where it improves the existing architecture") argues against inventing
one.

**Next phase:** none requested. STOPPED per instruction (no Docker/
Azure/deployment).

## Phase 10.1 -- Frontend Visual Identity Overhaul -- 2026-08-17

Follow-up to Phase 10: the prior pass was judged too conservative
("already mature", mostly unchanged). This pass makes a genuinely
visible transformation rather than another audit, per explicit
instruction, while keeping every backend contract, model/agent/SHAP/
GenAI/PDF/email behavior, and existing test guarantee (15/page
pagination, cross-filtering, AI-explanation caching) untouched.

- **Full palette rewrite** (`tokens.css`, same token NAMES, new
  VALUES): page background shifted from a warm neutral off-white to a
  cool blue-gray; primary blue deepened/saturated to a healthcare azure
  (`--accent`); a new secondary teal token (`--teal`) and a deep navy
  brand token (`--navy`) added; `--nav-urgent-care` corrected from cyan
  to orange (the spec's own stated mapping, previously not applied);
  SHAP increase/decrease shifted to a clearer warm-coral/cool-blue pair;
  a new `--gradient-header` (navy -> blue -> teal) brand sweep added.
  Dark theme mirrored with a navy-black (not neutral-black) elevation
  ladder. Because nearly every component already consumed these tokens
  rather than hard-coded colors, this one file cascades a materially
  different look across the entire application.
- **Header redesigned**: new mark/wordmark ("UC07 Navigator" + "Care
  Management Intelligence" tagline), a 3px navy->blue->teal brand
  accent bar, and a live API-status pill wired to the real `GET
  /health` endpoint (polls every 60s) -- not a decorative static label.
- **Population KPIs**: 6 tiles (was 3), each with its own icon badge
  and category-colored top edge and hover elevation.
- **Chart cards**: each of the 4 analytics cards (Risk/Navigation/
  Safety/Probability) now has a colored icon badge and matching top
  accent so they read as distinct categories, not identical gray boxes.
- **Navigation destination identity**: unified color+label everywhere
  (results table, member workspace NavigationCard, population chart)
  via a new shared `navigationDisplay.ts` module.
- **"Member Prioritization"** section: renamed from "Members", with a
  descriptive subtitle; filter toolbar restyled to pill-shaped
  controls with a search icon and a focus ring.
- **Results table**: sticky header darkened/strengthened, Safety
  OVERRIDE badge gained an explicit warning-triangle icon (icon + label
  + color, not color alone).
- **Member workspace**: widened from ~62vw to ~68vw (clamp 680-1180px)
  per the updated spec; header gained the same brand accent bar as the
  app header.
- **Communication tab**: emoji icons replaced with the app's consistent
  stroke-SVG icon language; action cards gained hover elevation.
- **Email composer**: button flow renamed to match the spec exactly
  (`Review & Send` -> a persistent "Confirm Send" screen showing
  Recipient/Attachment -> `Back`/`Send Email`) -- the confirm screen now
  stays visible through the actual send (previously it silently
  reverted to the pre-confirm layout once sending started).
- **Safety status icon** (member workspace SafetyCard) and Member
  Communication action icons converted from emoji to the same
  stroke-SVG icon language used everywhere else in the app.

**Files created:** none new this pass (navigationDisplay.ts was created
in Phase 10).

**Files modified:** `tokens.css`, `Header.{tsx,css}`,
`PopulationSummary.{tsx,css}`, `AnalyticsCharts.{tsx,css}`,
`MemberFilters.{tsx,css}`, `Uc07ResultsTable.{tsx,css}`,
`MemberDetailsDrawer.css`, `MemberReportActions.{tsx,css}`,
`EmailComposerModal.{tsx,css}`, `SafetyCard.{tsx,css}`; test files
updated to match (`EmailComposerModal.test.tsx` -- button label
renames only, same assertions/coverage).

**Backend/model/agents/SHAP/GenAI/PDF/email logic: NONE changed.**

**Automated tests:** frontend 136/136 passing. `tsc -b`: clean.
`oxlint`: clean (same pre-existing warnings in unrelated legacy files
only). Production build: succeeds.

**Not verified visually this pass either** (no browser/screenshot
tooling available this session) -- changes were designed and reasoned
about via the token cascade and component structure, then verified by
type-check/tests/build only. Documented explicitly rather than
overclaiming a visual QA pass that did not happen.

**Next phase:** none requested. STOPPED per instruction.
