# Decision Log

Decisions recorded here are durable project-level constraints. They are not
implementation choices — implementation decisions are made phase-by-phase
and documented in that phase's section of `docs/CHANGELOG.md`.

## 2026-08-15 — Phase 1 initial decisions

1. **The three source datasets are immutable.** `raw_members.csv`,
   `raw_ed_visits.csv`, and `raw_care_history.csv` must never be modified,
   renamed, overwritten, cleaned in-place, deleted, or regenerated. Derived
   datasets may be created in later phases as new, separate files.
2. **The existing project will be evolved, not rewritten.** Future phases
   build on `backend/` and `frontend/` as they exist today rather than
   starting a parallel implementation.
3. **UC07 must target potentially avoidable ED utilization, not generic
   frequent ED use.** The current `frequent_ED_user` target
   (`ED_visits_365d >= 2`) is documented in
   `docs/01_PROJECT_BASELINE.md` §7 as not meeting this bar and is a primary
   input to Phase 2 design discussion.
4. **Emergency care must never be blocked, discouraged, or gatekept.** Any
   future safety-layer or navigation-logic design must preserve and
   strengthen (never weaken) the existing frontend disclaimer guarantees
   documented in §14 of the baseline.
5. **Future architecture will separate risk detection, care navigation, and
   safety/policy responsibilities.** These are currently intermingled in
   `backend/predict.py` and `backend/main.py`; the separation is a
   Phase 2+ design task, not performed in Phase 1.
6. **Docker is the deployment packaging target.**
7. **Azure is the intended cloud platform.**
8. **Documentation must be updated continuously after each phase.**
   `docs/01_PROJECT_BASELINE.md`, `docs/CHANGELOG.md`, and this file are the
   canonical, living record of project state and decisions.

## 2026-08-15 — Phase 2 final decisions

Full rationale for every decision below is in
`docs/02_UC07_AND_DATA_DESIGN.md`. Only final, selected decisions are
recorded here.

9. **Historical avoidability definition (`potentially_avoidable_ed_event`)
   is built only from `triage_level`, `red_flag`, `admitted`, `icu`, and
   `major_procedure` — never from `diagnosis`.** Verified by direct
   crosstab inspection that `diagnosis` category carries no measurable
   acuity signal in this dataset (near-uniform triage/red_flag/admitted/
   icu/major_procedure rates across all 14 categories); encoding a
   diagnosis-based rule would be an invented assumption, not a
   data-supported pattern.
10. **Safety exclusions have absolute precedence.** `red_flag==1 OR
    icu==1 OR admitted==1 OR major_procedure==1 OR triage_level IN {1,2}`
    unconditionally forces `PROTECTED_OR_HIGH_ACUITY`, evaluated before any
    avoidability logic, with no code path that can override it.
11. **Three encounter-level label states**: `POTENTIALLY_AVOIDABLE`
    (triage 4–5, no exclusion), `PROTECTED_OR_HIGH_ACUITY` (any exclusion),
    `UNCERTAIN` (triage 3, no exclusion). Measured distribution: 42.27% /
    28.76% / 28.97% of historical ED encounters respectively.
12. **`UNCERTAIN` encounters never drive a positive member-level label.**
    A member is labeled positive only if the outcome window contains ≥1
    `POTENTIALLY_AVOIDABLE` encounter; members with only `UNCERTAIN` and/or
    `PROTECTED_OR_HIGH_ACUITY` encounters, or no ED encounter at all, are
    labeled negative. Chosen as the most conservative option that still
    yields a fully-labeled binary target for every member.
13. **Prediction horizon: 90 days.** Chosen over 30/60/120/180 days
    because it is the largest horizon that still permits a genuine 3-way
    non-overlapping temporal split within the dataset's 547-day span
    (180-day would leave only 7 days for observation history — infeasible),
    while producing stable ~9% prevalence across all three snapshots
    (9.07% / 9.57% / 9.09%).
14. **Observation window: 270 days, capped uniformly across all three
    temporal snapshots.** Chosen as the largest window that fits before
    the earliest (TRAIN) snapshot's index date without truncating history.
15. **Index-date strategy: three fixed, non-overlapping snapshots**
    (TRAIN index 2025-10-05, VALIDATION index 2026-01-03, TEST index
    2026-04-03), each with its own 270-day observation window and 90-day
    outcome window, replacing the current single-global-reference-date
    design. For live production inference, a single "now" index date is
    used with the same 270-day observation logic.
16. **Temporal (period-based) split chosen over member-level random
    split**, and over a fully member-disjoint split. Cross-snapshot member
    overlap (~7–8% of positives) is accepted as a documented, monitored
    limitation rather than a blocker, given the 2–3 day implementation
    timeline and the small further reduction in per-split positive counts
    a member-disjoint design would cause.
17. **Label-only variables** (usable only to construct the label for the
    encounter being predicted, never as that row's feature):
    `triage_level`, `red_flag`, `admitted`, `icu`, `major_procedure`,
    `diagnosis`, and `cost`, all restricted to the **outcome-window**
    encounter(s) only. The same field types, aggregated from **prior**
    (observation-window) encounters, are allowed historical predictors.
18. **Allowed historical feature groups** (all computed strictly using
    `visit_date <= index_date`): prior ED/avoidable/high-acuity utilization
    counts and recency, prior alternative-care (PCP/Urgent Care/
    Telehealth/Care Management) counts and recency, chronic-condition
    flags and burden, access/transportation/distance variables, age (and
    gender, conditional on subgroup validation before production use).
19. **Three-agent architecture is final for Phase 3**: Risk Detection
    Agent (leakage-safe features → risk probability/tier only, no
    navigation or safety decisions), Care Navigation Agent (deterministic
    rules → one of PCP/Urgent Care/Telehealth/Care Management, never
    emergency-discouraging language), Safety & Policy Agent (deterministic,
    final authority on every response, `CLEAR`/`CAUTION`/`OVERRIDE` states).
20. **Safety & Policy Agent has final, non-bypassable authority** over
    every response in every failure mode; all failures degrade toward more
    conservative, disclaimer-forward messaging, never toward more
    assertive redirection.
21. **Care Management becomes a reachable recommendation**, triggered when
    a risk/repeated-utilization signal co-occurs with chronic burden,
    access barriers, or prior Care Management engagement — never from a
    single signal in isolation. Exact numeric cutoffs deferred to Phase 3
    validation-set analysis.
22. **System claim boundaries are final** (`docs/02_UC07_AND_DATA_DESIGN.md`
    §22): the system may claim pattern-association risk identification and
    navigation support; it must never claim real-time emergency assessment,
    diagnosis, visit-necessity determination, or utilization prevention.

## 2026-08-15 — Phase 3 implementation-level decisions

These are implementation details discovered/chosen while building the
Phase 2 design into code. None of them changes the approved Phase 2
methodology (target definition, encounter classification, snapshot dates,
window lengths, or leakage policy) — each is a detail Phase 2 left open at
the "how exactly" level. Full rationale in
`docs/03_ML_DATA_PIPELINE.md`.

23. **Missing recency values use explicit `NaN` + a companion `has_prior_*`
    0/1 flag**, not a sentinel numeric value (e.g. `-1`/`9999`). Applies to
    every `days_since_prior_*` feature. Chosen because a sentinel number
    risks being misread by a model as a legitimate large distance; the
    flag makes "no qualifying prior event in this window" unambiguous.
24. **Recency is capped at the 270-day observation window**, not looked up
    indefinitely into a member's full history. An ED visit older than 270
    days before the index date produces `days_since_prior_ed = NaN` +
    `has_prior_ed = 0`, not the actual (larger) day count. This follows
    directly from the Phase 3 spec's instruction that longitudinal
    features must be computed strictly from the observation window.
25. **Protected/uncertain ED subcounts are limited to 90d/270d windows**
    (vs. the full 30/90/180/270d ladder used for total and
    potentially-avoidable ED counts). Chosen to avoid a proliferation of
    low-count, near-collinear features for encounter states that are not
    themselves the primary utilization/navigation signal, per the spec's
    "do not generate meaningless duplicate features" instruction.
26. **`backend/pit/` uses flat sibling imports** (no `__init__.py`
    package), matching the existing flat-import convention already used
    throughout `backend/` (e.g. `main.py`'s
    `from feature_engineering import ...`). Chosen for consistency with
    the existing codebase rather than introducing a new import style.
27. **Derived datasets live in `data/derived/`** at the repository root
    (sibling to the three raw CSVs' directory, in its own subdirectory),
    not under `backend/`. Chosen because the raw CSVs themselves already
    live at the repo root, and keeping derived output clearly separate
    from `backend/` source code avoids any ambiguity with the immutable
    raw files while staying close to them.
28. **`pytest` was added to `.venv`** as a test-only dependency, per the
    Phase 3 spec's explicit authorization ("If no test framework exists,
    use pytest..."). No production/ML library was added.

## 2026-08-15 — Phase 4 final decisions

Full rationale in `docs/04_MODEL_DEVELOPMENT.md`.

29. **Final algorithm: Logistic Regression** (`C=0.01`, `class_weight=None`,
    scikit-learn `lbfgs` solver, uncalibrated). Selected because it had
    the single highest VALIDATION PR-AUC (0.1220) and best Brier (0.0861)
    among all candidates and all calibration variants evaluated — not
    merely "close enough to justify picking the simpler model," it won
    outright on the primary metric.
30. **Final preprocessing:** median imputation for all 58 numeric
    features, `StandardScaler` for Logistic Regression only (tree-based
    candidates are scale-invariant and did not use it), most-frequent
    imputation + one-hot encoding (`handle_unknown="ignore"`) for the one
    categorical feature (`gender`). Fit on TRAIN only.
31. **Imbalance handling:** `class_weight` was tuned (`None` vs.
    `"balanced"`) as a grid parameter rather than assumed; the winning
    configuration used `class_weight=None`. No oversampling/SMOTE used
    anywhere; TEST class distribution was never altered.
32. **Calibration method: none (uncalibrated).** Sigmoid and isotonic
    calibration (5-fold CV fit on TRAIN) were compared for the two
    leading candidates and did not improve the selected model's PR-AUC or
    Brier score, so the simpler uncalibrated probabilities were retained.
33. **Primary model-selection metric: PR-AUC**, with Brier score
    (calibration quality) as the required secondary criterion — ROC-AUC
    was explicitly NOT the deciding metric (the three real candidates'
    ROC-AUC values were statistically indistinguishable from one another).
34. **`moderate_threshold = 0.097609`** (65th percentile of the selected
    model's VALIDATION predicted-probability distribution).
35. **`high_threshold = 0.123776`** (90th percentile of the same
    distribution). Both are data-derived from VALIDATION only, not
    invented constants; both produced monotonically increasing observed
    prevalence (LOW < MODERATE < HIGH) on both VALIDATION and TEST.
36. **Risk-tier semantics are final for Phase 4/5 handoff:** LOW = no
    proactive escalation by risk alone; MODERATE = meaningful navigation
    opportunity, light-touch; HIGH = strongest navigation opportunity,
    candidate for stronger outreach/Care Management review **in Phase 5**
    — this phase assigns tiers only and makes no navigation-routing
    decision.
37. **Final model version: `uc07-risk-v1`** (not `final_model`/
    `best_model_final`/similar — the model is expected to evolve).
    Serialization format: `joblib`, matching the legacy artifact's
    convention, saved as a single dict bundle (pipeline + feature list +
    thresholds + metadata) at
    `backend/models/uc07_risk_v1_model.joblib`, entirely separate from
    (and never overwriting) the legacy `backend/ed_risk_model.pkl`.
38. **Feature-order policy:** the model artifact's `feature_columns` is
    the single frozen ordered list (sourced from
    `data/derived/feature_manifest.json`'s `model_candidate` entries);
    every inference call must present columns in this exact order/set —
    enforced by `backend/modeling/feature_spec.py` being the only place
    that list is ever constructed.
39. **TEST evaluation policy:** TEST is loaded only after the model
    specification is frozen and written to
    `artifacts/model_evaluation/final_model_selection.json`; the
    selection function's signature structurally cannot accept TEST data;
    TEST is evaluated exactly once, with no post-hoc tuning regardless of
    result (a pre-declared degradation rule was checked, not used to
    trigger re-tuning).
40. **Rejected alternatives:** Random Forest and HistGradientBoosting
    were both seriously evaluated (full hyperparameter grids + calibration
    comparison for HGB) but rejected — both had lower VALIDATION PR-AUC
    than Logistic Regression, and HGB's winning raw configuration also had
    a materially worse uncalibrated Brier score (0.233 vs. ~0.086), which
    calibration could fix but which added complexity Logistic Regression
    didn't need. SMOTE/oversampling was considered and rejected per the
    Phase 4 spec's explicit instruction not to alter class balance by
    resampling.
41. **`joblib` pinned to `1.5.3`** in `backend/requirements.txt`
    (previously unpinned) because it is now directly load-bearing for the
    new model artifact's serialization; `scikit-learn` remains pinned at
    `1.7.2` as it was since Phase 1. No other dependency changed.

## 2026-08-15 — Phase 4B final decisions

Full rationale in `docs/04B_MODEL_IMPROVEMENT.md`.

42. **Accepted feature group: restructured ED-count windows only**
    (drop the 180d nested window, keep 30/90/270d — chosen because Step 1
    diagnostics showed 180d↔270d as the single most redundant nested pair,
    r=0.81, vs. 30d↔270d at r=0.33). VALIDATION PR-AUC gain: +0.0001
    (noise-level), retained only because it was the best of the tested
    window representations, not because it materially improved anything.
43. **Rejected feature groups: utilization velocity, care-setting mix,
    care continuity/engagement, access×utilization interactions,
    historical-ED-pattern extras.** None demonstrated real incremental
    VALIDATION PR-AUC beyond the restructured-windows baseline; several
    (access interactions, historical-ED extras) modestly reduced it.
44. **Diagnosis excluded from v1/v2** (reaffirming the Phase 3 baseline
    decision, now with direct evidence rather than a Phase 1 leakage
    concern alone): a controlled, volume-normalized, observation-window-only
    diagnosis representation was tested and produced a −0.00024 PR-AUC
    change vs. the same base feature set without it — no incremental
    signal demonstrated.
45. **Final algorithm remains Logistic Regression** — reconfirmed best
    among LogisticRegression/RandomForest/HistGradientBoosting on the
    restructured-windows feature set, consistent with Phase 4's original
    finding.
46. **Final feature set: `uc07-risk-v1`'s existing 59 features,
    unchanged.** The Phase 4B ablation winner (B, 55 features) was not
    promoted, so it never became a production feature set.
47. **Promotion decision: KEEP UC07-RISK-V1 (Decision B).** No
    `uc07-risk-v2` was created. Applied mechanically against pre-declared
    criteria (PR-AUC gain ≥0.01 absolute OR HIGH-tier lift relative gain
    ≥15%, with calibration preserved): actual results were +0.0002 PR-AUC
    and +3.0% relative tier lift, both short of their respective margins.
48. **Reason:** across every experiment run (window representation,
    7-step feature ablation, model comparison, regularization sweep,
    calibration comparison), the largest VALIDATION PR-AUC gain found
    anywhere was +0.00196 (a single ElasticNet configuration, reported
    for transparency but not adopted) — roughly a fifth of the promotion
    margin. This is treated as direct evidence that the three fixed
    source datasets, under the locked leakage controls, do not currently
    support a materially better model than v1, not as a search failure.
49. **Remaining predictive limitations acknowledged as unresolved:** a
    genuine, counter-intuitive negative univariate association between
    prior ED utilization and future avoidable-ED risk exists in this data
    (found via Step 1 diagnostics, confirmed not to be a multicollinearity
    artifact) and is not explained by this phase; intrinsically modest
    ROC-AUC (0.57–0.59) across every configuration tried suggests limited
    prospective signal in the fixed datasets for this target, independent
    of further feature engineering.

## 2026-08-15 — Phase 4C final decisions

Full rationale in `docs/04C_SYNTHETIC_DATA_EXPERIMENT.md`.

50. **The original three datasets are retained permanently, unchanged,
    as the project's sole real-data evidence base.** They are never
    moved, renamed, or written to; `uc07-risk-v1` and every Phase 3/4/4B
    artifact built from them remain the canonical original-data record.
51. **The synthetic experiment is fully isolated** into
    `data/synthetic/` (input) and `data/derived/synthetic/` (output).
    `data/derived/original/` was created as a permanent, byte-for-byte
    archive of the pre-existing original-data Phase 3 outputs
    specifically so the live `data/derived/` — which the existing Phase 3
    test suite legitimately regenerates on every full test run — can
    never be mistaken for the frozen historical record.
52. **`dataset_id` strategy:** every snapshot-metadata file now carries
    `dataset_id` (`"original"` or `"synthetic_uc07_v1"`) and a boolean
    `synthetic` flag. `"original"` is the default so pre-existing
    zero-argument calls to `build_snapshots.main()` remain behaviorally
    identical; all other data sources are explicit overrides.
53. **The point-in-time pipeline (`backend/pit/build_snapshots.py`) is
    now configurable** (`members_path`, `ed_path`, `care_path`,
    `output_dir`, `dataset_id`, `synthetic`) but remains a single
    implementation — `features.py`, `target.py`, `windows.py`, and
    `encounter_classification.py` were not modified or duplicated for
    the synthetic dataset.
54. **UC07 target, encounter classification, 270-day observation window,
    90-day outcome horizon, and the three fixed index dates (TRAIN
    2025-10-05 / VALIDATION 2026-01-03 / TEST 2026-04-03) are unchanged**
    for the synthetic run — reused exactly as locked in Phase 2/3, not
    re-tuned to produce a particular synthetic result.
55. **Synthetic results must always be labeled synthetic.** Every
    synthetic metadata file, and every future synthetic model artifact,
    must carry `dataset_id="synthetic_uc07_v1"` / `synthetic=true` and
    must never be presented as, merged with, or substituted for
    original-data (`uc07-risk-v1`) results.
56. **No model retraining occurred in Phase 4C.** This phase is data
    pipeline and descriptive-comparison only; synthetic model training is
    explicitly deferred to the next phase.

## 2026-08-15 — Phase 4D final decisions

Full rationale in `docs/04D_SYNTHETIC_MODEL_DEVELOPMENT.md`.

57. **Winning algorithm: Logistic Regression** (`C=0.01`,
    `class_weight=None`, uncalibrated) — same algorithm family as
    `uc07-risk-v1`, won outright on VALIDATION PR-AUC among all four
    candidates and all calibration variants.
58. **Hyperparameters/calibration/thresholds are synthetic-specific, not
    reused from `uc07-risk-v1`.** Thresholds were re-derived fresh from
    this model's own VALIDATION score distribution
    (`moderate_threshold=0.105986`, `high_threshold=0.213252`), per the
    explicit instruction not to reuse v1's thresholds automatically.
59. **Feature list policy: the exact, unmodified Phase 4C 59-feature
    baseline.** No Phase 4B experimental features (diagnosis, velocity,
    care-mix, interactions) were introduced — this was a deliberate
    apples-to-apples comparison against `uc07-risk-v1`'s feature set,
    not an attempt to further maximize synthetic performance.
60. **Synthetic model naming: `uc07-risk-synthetic-v1`**, deliberately
    not `uc07-risk-v2` — the "v2" name is avoided specifically because it
    could imply a direct production successor trained on equivalent
    real/original data, which this is not.
61. **`uc07-risk-v1` (original data) is permanently preserved** as the
    project's sole real-data model and benchmark; it was not retrained,
    modified, or replaced by this phase's work.
62. **Intended use of `uc07-risk-synthetic-v1`: demonstration / UC07
    navigation prototype only** — recorded explicitly in both the model
    artifact and its metadata, alongside the mandatory disclaimer
    ("must not be interpreted as clinically validated").
63. **Final model selected for the Phase 5 demonstration multi-agent
    system: `uc07-risk-synthetic-v1`**, because it produces materially
    more useful risk stratification for showcasing the navigation/safety
    architecture end-to-end. `uc07-risk-v1` remains available and
    preserved as the original-data benchmark; the architecture should
    later support model selection/configuration rather than deleting
    either model. Runtime switching is explicitly not implemented in
    Phase 4D.
64. **Limitations carried into Phase 5 planning:** synthetic results must
    never be presented as real-world evidence; the
    `transportation_barrier=1` subgroup's near-total recall (0.99) at the
    moderate threshold is flagged, unresolved, for Phase 6; VALIDATION→TEST
    Brier stability was the closest-to-threshold of any comparison in this
    project (not flagged, but noted).

## 2026-08-16 — Phase 5 final decisions

Full rationale in `docs/05_MULTI_AGENT_SYSTEM.md`.

65. **Deterministic software agents, not LLM agents.** No LLM or external
    AI API is used anywhere in `backend/agents/`. "Agent" means a bounded
    software component with a defined responsibility, input contract,
    decision logic, output contract, and authority boundary. The system
    is deterministic and auditable except for the trained model's own
    probability output.
66. **Three-agent responsibility split is final**: Risk Detection (risk
    only), Care Navigation (routing only, no safety authority), Safety &
    Policy (final authority, no risk/navigation authority of its own).
67. **Risk Agent has no navigation authority** — enforced by construction
    (no import of navigation/safety modules; `RiskAssessment` has no
    navigation-shaped field), not just by convention.
68. **Navigation Agent has no emergency-clearance authority** — enforced
    by construction (`decide()` has no `CurrentSafetyContext` parameter at
    all; structurally cannot see current safety information).
69. **Safety Agent has final, non-bypassable authority** over every
    response, in every orchestration path (single-member and batch),
    verified by source-order and type-based tests, not documentation alone.
70. **Missing current context = CAUTION, never CLEAR.** Historical
    absence of red flags does not prove the current situation is safe.
71. **Risk score never independently redirects care** — Care Management
    specifically requires a complexity/access/continuity signal in
    addition to elevated risk or repeated utilization; risk alone never
    triggers any specific destination on its own privileged path.
72. **`uc07-risk-synthetic-v1` is used only for this Phase 5
    demonstration system** — every response carries explicit, always-present
    machine-readable `model_version`/`dataset_id`/`synthetic_model`
    identity for auditability, plus a synthetic-data disclaimer available
    via `/model-info`.
73. **`uc07-risk-v1` (original data) is retained, untouched**, as the
    real-data benchmark; Phase 5 does not retrain, replace, or modify
    either model artifact.
74. **Centralized prohibited-language policy** (`safety_policy.
    PROHIBITED_PHRASES`, `check_text()`) is the single source of truth for
    what the system may say — applied to every navigation explanation and
    safety message in every safety state, not just OVERRIDE.
75. **Legacy `predict.py` is left unmodified and not forced to delegate**
    to the new orchestrator — its feature schema is fundamentally
    incompatible with the new point-in-time schema, and forcing
    delegation would silently change its existing contract. The new
    orchestrator (`backend/agents/orchestrator.py`) is the single
    authoritative UC07 decision path going forward; the legacy endpoints
    remain available, unchanged, and documented as pre-Phase-2 legacy.
76. **`httpx` added as a test-only dependency** (required for FastAPI's
    `TestClient`), matching the `pytest` precedent from Phase 3 — no
    production dependency changed.

## 2026-08-16 — Phase 4E final decisions

Full rationale in `docs/04E_TREE_MODEL_OPTIMIZATION.md`.

77. **KEEP LOGISTIC REGRESSION.** Random Forest and XGBoost, given a
    dedicated wider search (16 and 20 curated combinations respectively)
    on the exact same TRAIN/VALIDATION synthetic snapshots, did not
    exceed Logistic Regression on VALIDATION PR-AUC (best tree candidate
    delta -0.0089) or ROC-AUC (delta -0.0100). The pre-declared
    promotion signal (ROC-AUC delta ≥ +0.02, or a clear PR-AUC/HIGH-lift
    improvement) was not met; no model was promoted.
78. **Accuracy/Balanced Accuracy are reported at every threshold for
    every candidate but never used for selection** — consistent with
    `metrics.py`'s existing Phase 4 design decision. They live only in
    the new Phase 4E script, not in the shared `metrics.py`, to avoid
    diluting that module's documented "never a headline metric" policy.
79. **Generalization gap, not just the raw metric delta, decided this.**
    Every tree candidate's TRAIN→VALIDATION ROC-AUC gap (0.054–0.072)
    was ~9–12× Logistic's (0.006) — below the hard overfitting flag, but
    a real, material concern at this dataset's size (10,000 rows/59
    features) that independently argued against promotion even before
    the PR-AUC/ROC-AUC deltas were considered.
80. **`class_weight="balanced"` produced unusable threshold behavior for
    Random Forest and HistGradientBoosting at another model's operating
    points** — their PR-AUC-winning combinations selected 100% of the
    VALIDATION population (0 true negatives) at Logistic's MODERATE/HIGH
    thresholds. Documented as a caution, not treated as disqualifying on
    its own (each model's own newly-derived thresholds would behave
    differently), since Logistic was already ahead on rank metrics.
81. **`xgboost==3.2.0` added to `backend/requirements.txt`** (a genuinely
    new dependency, not present anywhere previously in the repository) —
    placed in the production requirements file, not dev-only, because a
    promoted XGBoost model would need it at inference time; since no
    model was promoted this run, it is currently exercised only by the
    training/comparison script and its tests.
82. **No new model artifact, no threshold change, no Risk Agent change.**
    `uc07-risk-synthetic-v1` and its frozen thresholds
    (`MODERATE=0.105986`, `HIGH=0.213252`) remain exactly as Phase 4D
    left them. Care Navigation Agent, Safety & Policy Agent, and the
    fixed orchestration order are untouched — Phase 4E is scoped to
    model comparison only, not agent redesign.
83. **`transportation_barrier=1`'s ~0.99 recall at MODERATE persists**,
    re-confirmed on the same frozen model/thresholds, still unresolved
    and still explicitly deferred to Phase 6.
84. **Phase 4E's own immutability guarantee is a hash-before/after check
    inside `train_phase4e_tree_comparison.py` itself** (recorded in
    `artifacts/phase4e_tree_model_comparison/immutability_check.json`),
    not a claim that the full pytest suite never touches the model
    metadata files — `test_model_pipeline.py` and `test_synthetic_model.py`
    already legitimately re-run `train.py`/`train_synthetic.py` every
    full-suite session (pre-existing Phase 4/4D behavior), which
    refreshes each metadata file's `training_timestamp_utc` (and hence
    its file hash) without changing any substantive content; the
    underlying `.joblib` artifacts and every raw/snapshot dataset remain
    byte-for-byte unchanged.

## 2026-08-16 — Phase 6 final decisions

Full rationale in `docs/06_SAFETY_FAIRNESS_ROBUSTNESS_VALIDATION.md`.

85. **Model frozen throughout Phase 6, no exceptions.**
    `uc07-risk-synthetic-v1`, its hyperparameters, feature list,
    calibration, and thresholds (`MODERATE=0.105986`, `HIGH=0.213252`)
    were never retrained, tuned, recalibrated, or re-derived. Every
    finding in this phase describes behavior of the existing frozen
    model, not a new one.
86. **Safety CLEAR now requires a COMPLETE current-safety context, not
    just a non-empty one.** Before this phase, `safety_policy.py`
    treated any single supplied field (e.g. only `triage_level=4`, with
    the other four fields unknown) as sufficient for CLEAR. This phase's
    brief explicitly required "insufficient" context to default to
    CAUTION, not just "fully absent" context. Confirmed with the user
    before changing this deliberate, tested Phase 5 behavior (it locks
    two existing test assertions) rather than unilaterally overriding
    it: user chose to tighten. Implementation checks OVERRIDE first
    (unchanged, still fires from a single known trigger field with
    everything else missing), then requires all five fields known before
    CLEAR is reachable; anything else resolves to CAUTION. Strictly more
    conservative — no OVERRIDE path was weakened.
87. **Transportation-barrier's ~0.99 recall at MODERATE is attributed to
    a combination of threshold interaction and coefficient magnitude**
    (its standardized logistic-regression coefficient is the single
    largest of all 60 encoded features), with correlated access features
    as a secondary contributor — not primarily an artifact of the
    synthetic generator alone. This is model-sensitivity analysis, not a
    causal or clinical claim, and the model was not modified in response.
    Classified INVESTIGATE per the phase's explicit disparity rule; this
    project deliberately never states "the model is fair."
88. **`telehealth_available` and `clinical_burden` also produced
    INVESTIGATE-level recall disparities** (Δ=-0.340 and Δ=-0.380
    respectively) — new findings this phase, likely correlated with the
    transportation-barrier finding rather than independent signals, both
    carried forward to Phase 7 rather than acted on here.
89. **CARE_MANAGEMENT pre-empts URGENT_CARE whenever `pcp_distance_miles`
    itself exceeds the 10-mile access-barrier threshold**, even when
    urgent care is closer than PCP — confirmed as genuine, documented
    Care Navigation Agent behavior (both signals share the same 10-mile
    constant), not a defect. URGENT_CARE is therefore only reachable
    when PCP access itself stays within that threshold.
90. **Minimum subgroup reporting size set at n≥100** (below that:
    descriptive-only / unstable, marked not hidden). Every subgroup
    actually evaluated this phase had n≥605.
91. **Static member-level CSV field value-range validation (age,
    distances, binary flags) is a documented gap, not fixed this
    phase.** Column *presence* is validated; column *value plausibility*
    is not. Classified low-severity because it never touches the
    `CurrentSafetyContext`-gated OVERRIDE path and never crashes or
    produces an unsafe output category — deferred to Phase 7 rather than
    expanding this validation phase's scope into new feature-engineering
    code.
92. **No new model artifact, no threshold change.** Only
    `backend/agents/safety_policy.py` was modified in this phase (Item
    86); Care Navigation Agent, Risk Detection Agent, `contracts.py`,
    `orchestrator.py`, and `main.py` are all unchanged.

## 2026-08-16 — Phase 7 final decisions

Full rationale in `docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md`.

93. **KEEP MODEL UNCHANGED (decision A).** No BLOCKER-level evidence
    (defined explicitly: subgroup ROC-AUC≤0.5, a safety/language-policy
    failure, or a disparity direction contradicting true prevalence)
    exists for any of the three investigated disparities. The default
    instruction to preserve the frozen model unless evidence is strong
    was not overridden.
94. **The transportation_barrier disparity is attributed to a
    combination of a real 3.1× prevalence difference, the single
    largest standardized coefficient in the model (rank 1/60), and
    correlated access/utilization-history features — and it persists
    after conditioning on telehealth/burden/history/distance**, meaning
    it is not a confound fully explained away by any one correlated
    covariate. This is model-behavior analysis, not a causal or
    clinical claim.
95. **Phase 7's threshold-interaction analysis revises Phase 6's initial
    hypothesis.** Only 4.75% of `transportation_barrier=1` sits within
    ±0.01 of the MODERATE threshold (vs. 13.12% for `=0`); 51.3% of the
    group sits ≥0.10 above threshold. The disparity is therefore driven
    primarily by a genuinely shifted score distribution, not by members
    clustering barely over a threshold edge. Recorded explicitly as a
    refinement, not silently corrected without note.
96. **`num_chronic_conditions` is now optional in the raw upload and
    safely derived from its six component chronic-flag columns when
    absent; when supplied, it is validated for consistency against
    those flags and rejected (not silently overwritten) if mismatched.**
    Chosen because this field is a genuine, separate model input (not
    purely a derived display value), so silently trusting an
    inconsistent caller-supplied value would let bad data reach the
    model undetected.
97. **Closed a real, Phase-6-documented input-validation gap:**
    `triage_level` is now validated across every ED-visit row via
    `backend/agents/input_validation.py`, not only rows that happen to
    fall inside some snapshot's observation window (previously, an
    invalid triage_level outside every window silently reached scoring
    unrejected).
98. **Safety-context completeness/provenance are now named, typed
    contract elements** (`ContextCompleteness.COMPLETE/PARTIAL/ABSENT`,
    `ContextSource.CALLER_SUPPLIED/SYSTEM_DERIVED/NOT_AVAILABLE`) rather
    than only internal booleans — this formalizes, and does not change,
    the Phase 6 safety-fix outcome (verified: `safety_context_matrix.csv`,
    7/7 passed, including known-override-wins-even-with-partial-context).
    `SYSTEM_DERIVED` is defined but never produced by this system, per
    the explicit instruction not to claim a verification capability that
    does not exist.
99. **Safety-context JSON validation moved from ad-hoc manual checks in
    `main.py` into a dedicated Pydantic schema**
    (`backend/agents/safety_context_schema.py`), because that payload is
    genuinely JSON-shaped and well-suited to declarative validation;
    the outer multipart file-upload signature of `POST /uc07/decide` was
    deliberately left as-is (Pydantic is not idiomatic for multipart
    form/file parameters, and rewriting the working, heavily-tested
    endpoint signature carried real regression risk for no benefit).
100. **Distance/age validation bounds are engineering sanity limits
     (age 0-120, distance 0-500mi), explicitly not narrowed to the
     synthetic sample's observed ranges** (age 18-90, distance 0.2-30mi)
     — narrowing to the observed sample would be an invented restriction
     the schema does not actually require, per the explicit instruction
     not to invent unrealistic clinical restrictions.

## 2026-08-16 — Phase 8 final decisions

Full rationale in `docs/08_FRONTEND_INTEGRATION.md`.

101. **`POST /uc07/decide` is the frontend's sole source of risk/
     navigation/safety data**, presented under a new default "UC07
     Navigator" tab. The pre-existing frontend had zero connection to
     it before this phase -- confirmed by audit, not assumed.
102. **Legacy dashboard kept, not deleted, but explicitly isolated**
     under a "Legacy Demo" tab with a banner naming the pre-Phase-2
     `frequent_ED_user` model and stating its output is never a UC07
     decision. Preserves existing functionality (nothing is lost) while
     making the authority boundary visually unambiguous to any user.
103. **Frontend TypeScript contracts were derived from a live-captured
     backend response, not written from memory or documentation
     alone** (`artifacts/phase8_frontend/frontend_api_contract.json`) --
     avoids the risk of a frontend field that doesn't actually exist in
     the backend contract.
104. **Safety-context tri-state UI (`SafetyContextForm`) never defaults
     an unset field to false.** Selecting "Unknown" deletes the key from
     the outgoing payload rather than sending `0`; a member with no
     field set at all sends no `current_safety_context` entry for that
     member, preserving the exact "absent means ABSENT, never CLEAR"
     contract Phase 6/7 built on the backend.
105. **`CALLER_SUPPLIED` is never rendered as "verified" anywhere in the
     UI.** Only a provenance label ("User/caller supplied"), never a
     confirmation of accuracy -- matches the backend's own
     `ContextSource` docstring, which makes the same distinction.
106. **Safety Card is always rendered above Navigation Card**
     (Risk → Safety → Navigation order in `Uc07DecisionPanel`), and
     OVERRIDE's styling is the single most visually dominant state in
     the app (thickest border, largest text, `role="alert"`) --
     satisfies "safety message must take visual priority over
     navigation" through both DOM order and visual weight, not styling
     alone.
107. **Added `vitest` + `@testing-library/react` + `jsdom` as the
     frontend's first test framework** -- the minimal, idiomatic choice
     for this exact Vite+React+TS stack (near-zero config, not a "huge
     new framework"), chosen because no test infrastructure existed and
     20 specific rendering/safety scenarios were required to be covered.
108. **No literal Chrome-driven browser smoke test was performed.**
     Verification instead combined a live-backend network-contract
     smoke test (Node script issuing the exact request `decideUC07()`
     would) with jsdom-based component rendering tests. Documented
     explicitly as a limitation rather than presented as equivalent to
     a real browser click-through.
109. **No backend code was modified in this phase.** `input_validation.py`,
     `safety_context_schema.py`, `contracts.py`, `safety_policy.py`,
     `orchestrator.py`, and `main.py` (all touched in Phase 7) are
     unchanged; the 505/505 backend test count is identical to the
     Phase 7 baseline, confirming this.

## 2026-08-16 — Phase 8B final decisions

Full rationale in `docs/08B_CURRENT_SAFETY_CONTEXT_WORKFLOW.md`.

110. **Current safety context moves from a pre-run form to a
     post-decision "Evaluate Current Safety" action.** Earlier (Phase
     8), the single-member safety form was filled in BEFORE the initial
     decision request. This phase relocates it into the member details
     view, evaluated only after the baseline (CAUTION) decision is
     already visible -- matching the intended UX ("select member → see
     risk/navigation → then optionally supply current safety info") and
     removing a confusing second, redundant entry point for the same
     data.
111. **The optional batch `safety_context_file` CSV reuses
     `safety_context_schema.py`'s `SafetyContextEntry` Pydantic model
     for per-row validation, rather than writing a second parallel
     validator.** A blank CSV cell and an omitted JSON key now go
     through the exact same "stays None, never becomes 0" code path.
112. **Any duplicate `member_id` in the safety CSV is rejected outright,
     regardless of whether the duplicate rows agree or conflict.**
     Silently picking one row (first, last, or "the safe one") would be
     an implicit judgment call this system should never make quietly --
     consistent with this project's established pattern of rejecting
     ambiguity rather than resolving it silently.
113. **A `member_id` present in both the CSV and the JSON
     `current_safety_context` field resolves to the JSON entry.** The
     JSON path represents a specific, ad-hoc single-member action (the
     drawer's "Evaluate Current Safety"); the CSV represents a bulk
     baseline. The more specific, more recent action wins.
114. **The single-member "Evaluate Current Safety" call explicitly
     passes the ORIGINAL batch response's own `index_date`** (not
     "today," and not omitted) to guarantee the risk score and
     pre-safety navigation output cannot drift between the initial
     batch decision and the re-evaluated one for that member --
     verified directly (`test_21_22_23`), not merely assumed from the
     orchestrator's structure.
115. **Frontend has no independent safety decision logic**, enforced by
     a new grep-based backend test tuned to avoid two classes of false
     positive discovered while writing it: TypeScript `type X = "A" |
     "B"` union declarations, and JSX attribute values like `<option
     value="CLEAR">` (a static dropdown choice, not computed logic) --
     both legitimate uses of the literal strings that a naive substring
     search would have wrongly flagged.
116. **No new backend endpoint was created; `POST /uc07/decide` gained
     one optional field.** Consistent with every prior phase's
     philosophy of extending the single authoritative endpoint rather
     than fragmenting the UC07 API surface.
117. **SHAP LinearExplainer was chosen over the existing bare
     `coefficient x standardized_value` shortcut as the PRIMARY
     explanation method (Phase 8C), with the shortcut retained as a
     deterministic fallback.** Verified empirically, not assumed: on 50
     real TEST-snapshot rows, an INDEPENDENT-masker SHAP computation
     matched the plain shortcut closely (mean abs diff 0.0043), but a
     CORRELATION-AWARE masker (`shap.maskers.Impute`, which properly
     reallocates credit among UC07's genuinely correlated features --
     e.g. `access_burden` vs `transportation_barrier`, r~0.70) diverged
     materially (mean abs diff 0.0185, max 0.377). The two are NOT
     automatically identical; SHAP is the more statistically grounded
     choice for a dataset with real feature correlation, per Phase 8C's
     explicit instruction not to assume equivalence.
118. **A pre-existing, Phase-8B-authored test
     (`test_24_25_26_model_and_dataset_hashes_unchanged`) intermittently
     fails when the full backend suite runs in one process, due to a
     newly-exposed interaction, NOT a retraining or model-modification
     bug.** Root cause: `shap.LinearExplainer(..., seed=...)`
     internally reseeds/perturbs NumPy's global RNG state and does not
     fully restore it (confirmed only partially fixable via
     `np.random.get_state()/set_state()` guards around the explainer
     build and `shap_values()` calls in
     `backend/agents/model_explainability.py` -- a residual, unresolved
     global-state interaction remains, likely at the NumPy `Generator`
     or C-extension level, not the legacy `RandomState` singleton).
     When `test_model_explainability.py` (which exercises the real SHAP
     explainer) runs before `test_model_pipeline.py` in the same
     process, `backend/modeling/train.py::main()`'s re-run of the
     legacy `uc07-risk-v1` training pipeline (itself a pre-existing,
     Phase-4-authored reproducibility self-test, unrelated to Phase 8C)
     produces a BYTE-DIFFERENT but VERIFIED CONTENT-IDENTICAL pickled
     artifact -- coefficients, intercept, `feature_columns`, and both
     thresholds were confirmed bit-for-bit equal across both hash
     states by direct array comparison (`np.array_equal`), element by
     element, not merely by spot-checking a summary statistic. No
     retraining with different data/parameters occurred; no model
     behavior changed. This is the same class of byte-hash fragility
     the codebase's own `test_existing_model_artifacts_unchanged`
     (Phase 4E) already documents and works around with a
     content-based check rather than a byte hash, for exactly this
     reason. Phase 8C's own Part 19 final immutability verification
     therefore also uses content-based comparison (coefficients,
     thresholds, feature list) for the model artifacts, alongside plain
     byte hashes for the CSV datasets (which have no such instability).
     Left as a documented, known limitation rather than modifying the
     Phase 8B test, which is out of Phase 8C's scope. The same benign
     fluctuation was independently confirmed on `uc07-risk-synthetic-v1`
     too (triggered by `test_synthetic_model.py`/
     `test_synthetic_pipeline.py`'s equivalent reproducibility self-test
     of `train_synthetic.py`), with the identical bit-for-bit
     content-equivalence proof. Phase 8C's own final verification
     (`artifacts/phase8c_genai_explainability/immutability_verification.json`)
     re-ran each model's isolated reproducibility test once so its final
     recorded byte hash matches the Part 1 baseline exactly for all 11
     tracked files, datasets included.
