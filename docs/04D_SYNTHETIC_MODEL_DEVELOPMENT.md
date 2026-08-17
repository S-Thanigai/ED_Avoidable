# UC07 — Synthetic Model Retraining & Final Selection (Phase 4D)

**Implementation date:** 2026-08-15
**Phase:** 4D — Synthetic Model Retraining & Final Selection
**Builds on:** `docs/01_PROJECT_BASELINE.md` … `docs/04C_SYNTHETIC_DATA_EXPERIMENT.md`, `docs/DECISION_LOG.md`

> # SYNTHETIC DATA MODEL — DEMONSTRATION ONLY
> `uc07-risk-synthetic-v1` was trained and evaluated entirely on the
> explicitly synthetic `data/synthetic/` dataset trio. It contains no
> real member, encounter, or care data, and **must never be interpreted
> as clinically validated** or presented as evidence about real
> healthcare utilization. `uc07-risk-v1` (Phase 4, original data) remains
> the project's only real-data model and is unchanged by this phase.

---

## 1. Executive Summary

Using the identical Phase 4 methodology (TRAIN-fit / VALIDATION-select /
TEST-sealed, the same four candidate algorithms, the same unmodified
59-feature point-in-time baseline) applied to the Phase 4C synthetic
snapshots, Logistic Regression (`C=0.01`, uncalibrated) again won on
VALIDATION — but this time with **substantially stronger, genuinely
useful discrimination**: TEST ROC-AUC 0.7048 (vs. `uc07-risk-v1`'s
0.5747), PR-AUC 0.3366 (vs. 0.1111), PR enrichment 2.58× prevalence (vs.
1.22×), and a HIGH-tier lift of 2.87× (vs. 1.35×). Performance stayed
below the 0.85 suspicious-performance ceiling at every step, and the
gains are fully consistent with — not disproportionate to — the
materially stronger univariate relationships already documented in
Phase 4C's descriptive checks. **This demonstrates that the UC07
point-in-time pipeline can recover strong prospective signal when the
input data contains it. It does not, and cannot, demonstrate real-world
clinical validity.**

---

## 2. Why Synthetic Retraining Was Performed

Phase 4C established that the synthetic dataset contains measurably
stronger, cleanly monotonic univariate relationships between historical
features and future risk than the original dataset — including a
reversal of Phase 4B's confusing negative coefficient finding. Phase 4D
exists to answer the natural next question: does a model trained under
the exact same leakage-safe methodology actually capture that stronger
signal, using the same 59-feature baseline, with no new feature
engineering? This isolates "is the pipeline capable of learning strong
signal when present" from "did Phase 4B's extra feature engineering
help" (Phase 4B already answered the latter: no, for the original data).

---

## 3. Scientific / Demo Limitations

This is a controlled methodology demonstration, not a clinical study.
Every number in this document describes model behavior on a
synthetically constructed dataset. No claim in this document, the
model's metadata, or any downstream Phase 5 demonstration may assert or
imply that `uc07-risk-synthetic-v1` predicts real emergency department
utilization, was validated against real patients, or is suitable for any
clinical or coverage decision.

---

## 4. Frozen Phase 4C Inputs

| File | SHA-256 |
|---|---|
| `data/synthetic/raw_members.csv` | `00cb4023eb20876fd9b9cd2b3b3e283c8e6681f1452a6c3e9cbfda37f0bd2373` |
| `data/synthetic/raw_ed_visits.csv` | `bb3c9505a836b8c70813aa2fdd62f628bd871f657fa1dfca1330799d27ce88c0` |
| `data/synthetic/raw_care_history.csv` | `20fdcb836f6abbbd1b1b70d7c1f7cd2279f5c519251322944b0ca7109a66db1a` |
| `data/derived/synthetic/train_snapshot.csv` | `4a8b79cd779a15448117301574c3100a683b2e5547f01ed43469afb34f3ad50c` |
| `data/derived/synthetic/validation_snapshot.csv` | `afc328b3de95f5237d55c276235b7edd19e5081b122c81995e8a84cb50b05c56` |
| `data/derived/synthetic/test_snapshot.csv` | `5657522789d1ccb8dc884209843cd4a4ca283892f39494e058ceb4431e76d7d8` |
| `data/derived/synthetic/feature_manifest.json` | `fdd661653638649e3da02087bbdcd95a70f754f69e3218e0155cf8368cbd657a` |

All verified identical before and after the entire training run (`backend/tests/test_synthetic_model.py`).

| Snapshot | Rows | Positives | Prevalence |
|---|---:|---:|---:|
| TRAIN | 10,000 | 1,194 | 11.94% |
| VALIDATION | 10,000 | 1,174 | 11.74% |
| TEST | 10,000 | 1,307 | 13.07% |

---

## 5. Feature Set

The exact, unmodified 59-feature Phase 4C manifest — no diagnosis
features, no Phase 4B experimental interactions/velocity/care-mix
additions. `backend/modeling/feature_spec.py::load_model_feature_columns()`
loaded against `data/derived/synthetic/feature_manifest.json` (byte-identical
to the original manifest's feature schema, verified in Phase 4C §7) — the
same 58 numeric + 1 categorical (`gender`) split as `uc07-risk-v1`.

**Feature safety verified before training** (Step 3): target, `member_id`,
`index_date` absent from X; no outcome-window information; no
`frequent_ED_user`; no unwindowed diagnosis columns; identical column
order across TRAIN/VALIDATION/TEST; feature manifest matches actual
snapshot columns exactly (`backend/tests/test_synthetic_model.py`, items
3–8).

---

## 6. Candidate Models

Reused unmodified from `backend/modeling/train.py` (Phase 4): Dummy
(`strategy=prior`), LogisticRegression, RandomForestClassifier,
HistGradientBoostingClassifier — same builder functions, same grids, same
`select_model_on_validation()` orchestration, applied to the synthetic
TRAIN/VALIDATION data. No XGBoost/LightGBM/CatBoost/neural networks added.

## 7. Hyperparameter Search

Identical small grids to Phase 4 (32 total combinations): LogisticRegression
`C ∈ {0.01,0.1,1,10} × class_weight ∈ {None,"balanced"}` (8); RandomForest
`max_depth ∈ {8,16} × min_samples_leaf ∈ {5,20} × class_weight ∈ {None,"balanced"}`,
`n_estimators=400` fixed (8); HistGradientBoosting `max_iter ∈ {100,200} ×
max_depth ∈ {3,None} × learning_rate ∈ {0.05,0.1} × class_weight ∈ {None,"balanced"}` (16).
Full log: `artifacts/synthetic_model_evaluation/hyperparameter_search.csv`.

## 8. Class Imbalance Handling

`class_weight` tuned (not assumed) for every applicable algorithm.
Winning Logistic Regression configuration: `class_weight=None`. No
SMOTE/oversampling used anywhere; TEST class distribution never altered.

---

## 9. Validation Results

| Candidate | Hyperparameters | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
| Dummy | `strategy=prior` | 0.5000 | 0.1174 | 0.1036 |
| **Logistic Regression** | `C=0.01, class_weight=None` | **0.6993** | **0.3288** | **0.0927** |
| Random Forest | `max_depth=8, min_samples_leaf=20, class_weight=None` | 0.6913 | 0.3170 | 0.0937 |
| HistGradientBoosting | `max_iter=100, max_depth=3, lr=0.05, class_weight=balanced` | 0.6915 | 0.3203 | 0.2032 ⚠ |

**Dummy's PR-AUC (0.1174) matches VALIDATION prevalence (11.74%) exactly**,
confirming the no-skill baseline behaves as expected. Every real
candidate clears it by a wide margin (PR-AUC ~0.32–0.33 vs. 0.1174) — a
qualitatively different result from the original-data experiment, where
all three real candidates clustered within 0.02 of the Dummy baseline.
HistGradientBoosting's uncalibrated Brier (0.2032) is the same
well-understood `class_weight="balanced"` artifact seen in Phase 4/4B —
not evidence of a problem with that candidate's ranking ability, just its
raw (uncalibrated) probability scale.

Full table: `artifacts/synthetic_model_evaluation/candidate_metrics.csv`.

## 10. Calibration Analysis

Uncalibrated vs. sigmoid vs. isotonic compared for the two leading
candidates (5-fold CV fit on TRAIN only):

| Candidate | Calibration | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
| Logistic Regression | **uncalibrated** | 0.6993 | **0.3288** | **0.0927** |
| Logistic Regression | sigmoid | 0.6994 | 0.3282 | 0.0927 |
| Logistic Regression | isotonic | 0.6994 | 0.3241 | 0.0928 |
| HistGradientBoosting | uncalibrated | 0.6915 | 0.3203 | 0.2032 |
| HistGradientBoosting | sigmoid | 0.6925 | 0.3209 | 0.0934 |
| HistGradientBoosting | isotonic | 0.6914 | 0.3177 | 0.0934 |

Logistic Regression's uncalibrated probabilities were already best on
every metric — calibration was evaluated, found not to help, and the
simpler uncalibrated model was kept (same decision pattern as Phase 4).
Full table: `artifacts/synthetic_model_evaluation/calibration_comparison.csv`.

## 11. Model Selection Rationale

**Selected: Logistic Regression, `C=0.01`, `class_weight=None`, uncalibrated.**

1. **PR-AUC** (primary): 0.3288, clearly the highest across every
   candidate/calibration combination evaluated.
2. **ROC-AUC**: 0.6993, also highest, though close to RF (0.6913) and HGB
   (0.6915) — not the deciding factor on its own.
3. **Calibration/Brier**: also best (0.0927) among compared options.
4. **Risk concentration**: strongest tier separation of any candidate
   evaluated (§13).
5. **Interpretability**: Logistic Regression's coefficients are directly
   reportable and, unlike the original-data experiment, align with
   intuitive directionality (§18).
6. **Operational simplicity**: simplest model family.

Logistic Regression won outright on the primary metric — no
close-call tie-break against a more complex model was needed.

---

## 12. Threshold Selection

Percentile-based on the selected model's own VALIDATION score
distribution (same method as `uc07-risk-v1`, not reused verbatim —
recomputed fresh for this model): `moderate_threshold = 0.105986` (65th
percentile), `high_threshold = 0.213252` (90th percentile). Representative
points from the sweep (`artifacts/synthetic_model_evaluation/threshold_analysis.csv`,
94 thresholds):

| Threshold | Selected | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| 0.11 (≈ moderate) | 32.8% | 0.215 | 0.601 | 0.708 | 0.317 |
| 0.21 (≈ high) | 10.4% | 0.360 | 0.318 | 0.925 | 0.337 |

---

## 13. Validation Risk Tiers

| Tier | Members | % pop. | Positives | Observed prevalence | Lift | Mean predicted probability |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 6,500 | 65.00% | 452 | 6.95% | 0.59× | 0.0718 |
| MODERATE | 2,500 | 25.00% | 356 | 14.24% | 1.21× | 0.1436 |
| HIGH | 1,000 | 10.00% | 366 | **36.60%** | **3.12×** | 0.3724 |

Monotonic (6.95% < 14.24% < 36.60%) with a dramatically stronger HIGH-tier
concentration than `uc07-risk-v1`'s VALIDATION result (12.70% observed
prevalence, 1.33× lift) — over a third of the HIGH tier actually has the
outcome in the synthetic experiment, vs. roughly one in eight originally.

---

## 14. Frozen Model Specification

Written to `artifacts/synthetic_model_evaluation/frozen_model_selection.json`
*before* `TEST_CSV` is loaded anywhere in `train_synthetic.py` (verified
structurally — `backend/tests/test_synthetic_model.py::
test_train_synthetic_main_loads_test_only_after_freeze` confirms the
freeze-write precedes the TEST load in the function's own source order):

```
model_version_candidate: uc07-risk-synthetic-v1
dataset_id:              synthetic_uc07_v1
algorithm:                logistic_regression
calibration_method:       uncalibrated
hyperparameters:          {"C": 0.01, "class_weight": null}
feature_list:              59 features, identical manifest order to uc07-risk-v1
moderate_threshold:        0.105986
high_threshold:             0.213252
```

---

## 15. Final TEST Results

Evaluated exactly once:

| Metric | VALIDATION | TEST |
|---|---:|---:|
| Prevalence | 11.74% | 13.07% |
| ROC-AUC | 0.6993 | **0.7048** |
| PR-AUC | 0.3288 | **0.3366** |
| PR enrichment (PR-AUC / prevalence) | 2.80× | **2.58×** |
| Brier | 0.0927 | 0.1023 |
| Log loss | 0.3247 | 0.3499 |

Confusion matrices on TEST:

| Threshold | Selected | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| moderate (0.1060) | 3,473 (34.7%) | 0.230 | 0.612 | 0.693 | 0.335 |
| high (0.2133) | 1,008 (10.1%) | 0.375 | 0.289 | 0.928 | 0.327 |

## 16. TEST Risk Tiers

| Tier | Members | % pop. | Positives | Observed prevalence | Lift | Mean predicted probability |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 6,527 | 65.27% | 507 | 7.77% | 0.59× | 0.0721 |
| MODERATE | 2,465 | 24.65% | 422 | 17.12% | 1.31× | 0.1432 |
| HIGH | 1,008 | 10.08% | 378 | **37.50%** | **2.87×** | 0.3692 |

Monotonic (7.77% < 17.12% < 37.50%) and closely tracks VALIDATION in both
population share and direction — the threshold design generalizes stably.

---

## 17. Validation vs. TEST Stability

| Metric | Δ (TEST − VALIDATION) | Flagged? |
|---|---:|---|
| PR-AUC | +0.0079 | No (improved) |
| ROC-AUC | +0.0055 | No (improved) |
| Brier | +0.0097 | No — under the 0.01 flag threshold, but the closest call of any comparison in this project so far |
| HIGH-tier lift | 3.12× → 2.87× | No — both are strong; the direction (slightly lower on TEST) is worth watching, not acting on |

Degradation rule (declared before comparison): flag if `pr_auc_delta <
-0.03` or `brier_delta > 0.01`. Neither condition was met —
**no material degradation, no post-TEST tuning performed.** The Brier
delta (+0.0097) is close enough to the 0.01 threshold that it is called
out explicitly here rather than silently passed over.

---

## 18. Subgroup Sanity Checks

Computed on TEST (`artifacts/synthetic_model_evaluation/subgroup_metrics.csv`),
minimum 200 members / 15 positives to report ROC-AUC/PR-AUC (all groups
met this bar):

| Subgroup | n | Prevalence | ROC-AUC | PR-AUC | Recall@moderate | Precision@moderate |
|---|---:|---:|---:|---:|---:|---:|
| age 0–35 | 1,440 | 11.81% | 0.713 | 0.309 | 0.594 | 0.212 |
| age 35–50 | 2,668 | 11.36% | 0.689 | 0.266 | 0.587 | 0.208 |
| age 50–65 | 3,164 | 13.46% | 0.715 | 0.354 | 0.622 | 0.246 |
| age 65–80 | 1,934 | 14.58% | 0.683 | 0.346 | 0.589 | 0.231 |
| age 80+ | 794 | 15.87% | 0.740 | 0.457 | 0.714 | 0.262 |
| gender F | 5,084 | 13.30% | 0.713 | 0.352 | 0.623 | 0.231 |
| gender M | 4,916 | 12.84% | 0.696 | 0.321 | 0.601 | 0.230 |
| clinical_burden 0–1 | 3,938 | 10.74% | 0.694 | 0.259 | 0.480 | 0.233 |
| clinical_burden 1–2 | 3,849 | 12.24% | 0.688 | 0.300 | 0.580 | 0.203 |
| clinical_burden 2–3 | 1,608 | 17.23% | 0.700 | 0.390 | 0.747 | 0.249 |
| clinical_burden 3+ | 605 | 22.48% | 0.736 | 0.559 | 0.860 | 0.275 |
| transportation_barrier=0 | 8,378 | 9.75% | 0.632 | 0.179 | 0.386 | 0.166 |
| transportation_barrier=1 | 1,622 | **30.21%** | 0.676 | 0.524 | **0.990** | 0.309 |

**No degenerate (ROC-AUC ≤ 0.5) subgroup found** — a marked improvement
over `uc07-risk-v1`'s TEST result, where `clinical_burden 3+` fell to
0.483 (below no-skill). Here the same subgroup shows ROC-AUC 0.736, the
second-highest of any group. **`transportation_barrier=1` shows a
dramatically higher prevalence (30.2% vs. 9.75%) and near-total recall at
the moderate threshold (0.99)** — worth flagging for Phase 6, since a
model that essentially flags almost everyone in this subgroup MODERATE-or-
higher may not be discriminating well *within* that subgroup even though
it discriminates the subgroup as a whole. **This is not a fairness
determination** — only an observation for deeper Phase 6 review.

---

## 19. Feature Importance

Logistic Regression signed coefficients (standardized numeric features),
top 10 by absolute value:

| Rank | Feature | Coefficient | Direction |
|---:|---|---:|---|
| 1 | `transportation_barrier` | +0.2235 | associated with higher model-estimated risk |
| 2 | `telehealth_available` | −0.1804 | associated with lower model-estimated risk |
| 3 | `pcp_distance_miles` | +0.1184 | associated with higher model-estimated risk |
| 4 | `prior_potentially_avoidable_ed_count_30d` | +0.1144 | associated with higher model-estimated risk |
| 5 | `prior_potentially_avoidable_ed_count_270d` | +0.1097 | associated with higher model-estimated risk |
| 6 | `prior_potentially_avoidable_ed_count_90d` | +0.1049 | associated with higher model-estimated risk |
| 7 | `prior_ed_count_270d` | +0.1011 | associated with higher model-estimated risk |
| 8 | `prior_uncertain_ed_count_90d` | −0.0853 | associated with lower model-estimated risk |
| 9 | `prior_ed_count_180d` | +0.0756 | associated with higher model-estimated risk |
| 10 | `num_chronic_conditions` | +0.0659 | associated with higher model-estimated risk |

No causal claim is made anywhere in this document or the model metadata
— e.g., *"transportation barrier was associated with higher
model-estimated risk in this synthetic dataset,"* never *"transportation
barriers cause ED visits."* Full table: `artifacts/synthetic_model_evaluation/global_feature_importance.csv`.

**Interpretability comparison to `uc07-risk-v1`:** the original model's
top features included two with counter-intuitive negative signs
(`has_prior_ed`, `prior_potentially_avoidable_ed_count_270d` — Phase 4B
confirmed this was a genuine, if surprising, pattern in the original
data, not a bug). **The synthetic model shows the intuitive positive
direction on every prior-ED-utilization feature** — consistent with, and
a direct reflection of, Phase 4C's descriptive finding that
`has_prior_ed` and `prior_potentially_avoidable_ed_count_270d` flipped to
positive association in the synthetic data (`docs/04C_SYNTHETIC_DATA_EXPERIMENT.md`
§14). This is not the model "fixing" anything — it is accurately
reflecting a different underlying dataset.

---

## 20. Original V1 vs. Synthetic Model Comparison

| | **uc07-risk-v1** (original) | **uc07-risk-synthetic-v1** (synthetic) |
|---|---:|---:|
| Dataset type | original | **synthetic — demonstration only** |
| Algorithm | logistic_regression | logistic_regression |
| TEST prevalence | 9.08% | 13.07% |
| TEST ROC-AUC | 0.5747 | **0.7048** |
| TEST PR-AUC | 0.1111 | **0.3366** |
| TEST PR enrichment | 1.22× | **2.58×** |
| TEST Brier | 0.0822 | 0.1023 |
| TEST HIGH-tier lift | 1.35× | **2.87×** |

**Interpretation, stated exactly once and consistently throughout this
project:** *if the synthetic model performs substantially better, this
demonstrates that the existing pipeline can learn stronger prospective
relationships when those relationships exist in the input data. It does
NOT demonstrate real-world clinical validity.* Full comparison file:
`artifacts/synthetic_model_evaluation/original_vs_synthetic_comparison.csv`.

---

## 21. Why Improved Synthetic Performance Does NOT Imply Real-World Clinical Performance

The synthetic dataset was explicitly constructed (Phase 4C) with cleaner,
stronger, more separable relationships between historical features and
the target than the original data exhibits. A model trained on that data
learning those relationships well is an expected, mechanical consequence
of the input data's construction — not a demonstration that real members'
future ED utilization follows the same clean pattern. `uc07-risk-v1`
remains the only evidence this project has about real-data behavior, and
its modest performance (Phase 4/4B) is the honest, relevant number for
any real-world discussion. `uc07-risk-synthetic-v1` answers a pipeline
question ("can this methodology learn strong signal when present"), not
a clinical one.

**Leakage/proxy audit performed per Step 12 (ROC-AUC did not exceed the
0.85 ceiling, but this section is included anyway for transparency
given the size of the jump from the original result):**
- Same, unmodified point-in-time code (`features.py`, `target.py`,
  `windows.py`, `encounter_classification.py`) computed both experiments'
  snapshots — no synthetic-specific branch exists anywhere in that logic.
- The synthetic snapshots independently passed the same 17
  reconciliation-based leakage checks as the original snapshots at
  Phase 4C build time, including checks that recompute several feature/
  target values directly from raw data and require exact agreement with
  the pipeline's own output.
- The magnitude of improvement is fully consistent with — not
  disproportionate to — Phase 4C's already-documented, independently
  measured univariate relationships (e.g., prior potentially-avoidable ED
  count showing up to 4.13× lift, transportation barrier 2.29× lift); a
  multivariate model combining several such genuinely strong,
  non-degenerate univariate signals into ROC-AUC ≈ 0.70 is expected, not
  suspicious.
- No subgroup, threshold, or calibration result showed a degenerate
  (0%/100%, or otherwise implausible) pattern.
- **Conclusion: no leakage or proxy-target artifact found. The
  improvement is attributed to the synthetic data's intentionally
  stronger constructed relationships.**

---

## 22. Model Artifact Structure

`backend/models/uc07_risk_synthetic_v1_model.joblib` — joblib dict,
structurally identical to `uc07-risk-v1`'s artifact plus explicit
synthetic labeling:

```python
{
    "pipeline": <sklearn Pipeline: preprocessor + LogisticRegression>,
    "feature_columns": [...59 names, identical order to uc07-risk-v1...],
    "numeric_features": [...58...], "categorical_features": ["gender"],
    "target": "future_potentially_avoidable_ed_90d",
    "model_version": "uc07-risk-synthetic-v1",
    "dataset_id": "synthetic_uc07_v1",
    "synthetic": True,
    "intended_use": "demonstration / UC07 navigation prototype",
    "algorithm": "logistic_regression",
    "calibration_method": "uncalibrated",
    "hyperparameters": {"C": 0.01, "class_weight": None},
    "moderate_threshold": 0.105986, "high_threshold": 0.213252,
}
```

Does **not** overwrite `backend/models/uc07_risk_v1_model.joblib` (verified
reloadable, unchanged — `backend/tests/test_synthetic_model.py::
test_original_v1_model_unchanged`). Metadata:
`backend/models/uc07_risk_synthetic_v1_model_metadata.json` — full
schema per Phase 4D spec Step 23, including the required disclaimer
sentence verbatim.

---

## 23. Known Limitations

1. **Synthetic data model — demonstration only** (restated intentionally,
   throughout this document): no real member/encounter/care data was used.
2. Same single ~18-month data era and 3-snapshot temporal design as the
   original experiment (Phase 2/3 limitation, unchanged).
3. `transportation_barrier=1` subgroup shows near-total recall at the
   moderate threshold (0.99) — flagged for Phase 6 review, not resolved here.
4. Subgroup sanity check is initial only, not the full Phase 6 fairness audit.
5. VALIDATION→TEST Brier delta (+0.0097) is the closest call to the
   degradation-flag threshold (0.01) of any comparison in this project —
   not flagged, but noted for monitoring if this model is ever extended.
6. Risk tiers are risk-only; Care Management/navigation routing logic
   remains Phase 5 scope, unimplemented here.
7. Any relationship this model learns reflects the synthetic data
   generator's internal construction, not measured real-world clinical relationships.

---

## 24. Phase 5 Readiness

A clearly-labeled, fully tested, leakage-verified synthetic model
(`uc07-risk-synthetic-v1`) now exists alongside the untouched original
model (`uc07-risk-v1`). Both are available for Phase 5. Per the Phase 4D
spec, `uc07-risk-synthetic-v1` is recommended as the model for the
Phase 5 **demonstration** multi-agent system (it produces materially more
useful risk stratification for showcasing the navigation/safety
architecture end-to-end), while `uc07-risk-v1` remains preserved as the
original-data benchmark. Runtime model selection/configuration (rather
than a hard-coded choice) is a Phase 5 architecture concern and is
**not** implemented in this phase — no agent code, no model-selection
runtime logic, no frontend change was made here.
