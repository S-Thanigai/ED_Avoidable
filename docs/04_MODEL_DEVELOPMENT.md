# UC07 — Model Development, Calibration & Risk Tiers (Phase 4)

**Implementation date:** 2026-08-15
**Phase:** 4 — Model Development, Calibration & Risk Tiers (model training; no agents, no API/frontend changes)
**Builds on:** `docs/01_PROJECT_BASELINE.md`, `docs/02_UC07_AND_DATA_DESIGN.md`,
`docs/03_ML_DATA_PIPELINE.md`, `docs/DECISION_LOG.md`

No multi-agent code was written. No Care Navigation logic (PCP/Urgent
Care/Telehealth/Care Management routing) was implemented — this phase
produces LOW/MODERATE/HIGH risk tiers only. `predict.py`, `main.py`, and
the frontend were not modified. No Docker/Azure work was done. The three
raw datasets and the three frozen Phase 3 snapshot CSVs were verified
byte-identical (SHA-256) before and after this phase.

---

## 1. Executive Summary

Four candidate model families (Dummy baseline, Logistic Regression,
Random Forest, HistGradientBoostingClassifier) were trained on TRAIN and
compared on VALIDATION using PR-AUC as the primary metric. All three real
candidates clustered close together (ROC-AUC 0.57–0.57, PR-AUC 0.11–0.12)
— a modest but genuine improvement over the no-skill Dummy baseline
(ROC-AUC 0.50, PR-AUC ≈ prevalence). **Logistic Regression (C=0.01, no
class weighting, uncalibrated) was selected** as the final model: it had
the single best VALIDATION PR-AUC among every candidate/calibration
combination evaluated, so no interpretability/complexity tie-break was
even needed — the simplest, most interpretable candidate also happened to
win outright. Risk tiers were designed on VALIDATION using
percentile-based thresholds on the model's own score distribution
(`moderate_threshold ≈ 0.0976`, 65th percentile; `high_threshold ≈
0.1238`, 90th percentile), producing a monotonically increasing observed
prevalence across LOW → MODERATE → HIGH on both VALIDATION and TEST. The
frozen model was evaluated exactly once on TEST: PR-AUC 0.1111, ROC-AUC
0.5747, Brier 0.0822 — consistent with VALIDATION (no material
degradation). This is an honest, modest result: with diagnosis-derived
and other leakage-risk features correctly excluded (Phase 1–3), the
remaining leakage-safe signal for 90-day forward avoidable-ED risk is
real but weak, and that finding is reported plainly rather than hidden.

---

## 2. Phase 4 Objective

Train and select the best available model for: *probability that a
member will have at least one future potentially avoidable ED encounter
within the next 90 days*, using TRAIN for fitting, VALIDATION for every
selection/calibration/threshold decision, and TEST exactly once at the
end — then package a versioned artifact, metadata, and evaluation
reports. Care Navigation routing and the multi-agent runtime are
explicitly out of scope (Phase 5).

---

## 3. Frozen Phase 3 Inputs

| Snapshot | Index date | Rows | Positives | Prevalence | SHA-256 |
|---|---|---:|---:|---:|---|
| `train_snapshot.csv` | 2025-10-05 | 10,000 | 904 | 9.04% | `1b6799904302398d95b478ca2a1e33d0b206fcc1983151b743f01cdbb7a534eb` |
| `validation_snapshot.csv` | 2026-01-03 | 10,000 | 958 | 9.58% | `a19dad00c4a8329074f7dcba94357506fd004ff7798c82d7d8b7313f13c9b70f` |
| `test_snapshot.csv` | 2026-04-03 | 10,000 | 908 | 9.08% | `1d4e8b22ede975cdad43379dd0566d38b597e270e8dbd1fcf3a1d85d8989ac1a` |

Verified byte-identical (SHA-256) before this phase began and again after
every training run, including the final one (`backend/tests/
test_model_pipeline.py::test_snapshot_hashes_match_phase3_frozen_values`).
Raw dataset hashes (`raw_members.csv`, `raw_ed_visits.csv`,
`raw_care_history.csv`) were independently reverified unchanged in the
same run.

---

## 4. Model Feature Set

`backend/modeling/feature_spec.py::load_model_feature_columns()` is the
single authoritative source: it loads `data/derived/feature_manifest.json`
and returns every `feature_name` with `model_candidate: true`, in
manifest order — **59 features**. `member_id`, `index_date`, the target
column, and any non-model-candidate manifest entry are excluded by this
same mechanism, not by a second hand-maintained list anywhere else in the
codebase (verified by `test_feature_columns_come_from_manifest_model_candidates`
and `test_target_member_id_index_date_never_in_feature_columns`).

Numeric vs. categorical is determined from the loaded snapshot's actual
pandas dtypes (`pandas.api.types.is_numeric_dtype`), not guessed from the
manifest's descriptive `category` field: **58 numeric features, 1
categorical feature (`gender`)**.

> **Implementation note:** an initial dtype check using `dtype == object`
> silently misclassified `gender` as numeric, because the installed
> pandas 3.0.5 infers plain string columns using its newer StringDtype
> rather than classic `object`. Fixed by switching to
> `pandas.api.types.is_numeric_dtype`. Recorded in
> `docs/DECISION_LOG.md` as a Phase 4 implementation decision.

---

## 5. Preprocessing

`backend/modeling/preprocessing.py`, fit on TRAIN only for every candidate
(enforced by control flow — the preprocessor is always constructed fresh
inside each candidate-building function and `.fit()` is only ever called
with `X_train`/`y_train`):

| Feature type | Imputation | Encoding/Scaling |
|---|---|---|
| Numeric (58 features) | median | none (tree-based candidates: scale-invariant) or `StandardScaler` (Logistic Regression only — unscaled wide-range counts/distances would otherwise dominate the L2 penalty) |
| Categorical (`gender`) | most-frequent | `OneHotEncoder(handle_unknown="ignore")` |

Missing-value source: only the Phase 3 `days_since_prior_*` recency
columns carry real `NaN`s (no qualifying prior event within the 270-day
observation window); median imputation is a simple, safe choice given
each such column's paired `has_prior_*` 0/1 flag already carries the
"no prior event" signal explicitly.

---

## 6. Class Imbalance Strategy

Prevalence is ~9%. `class_weight ∈ {None, "balanced"}` was included as a
tuned hyperparameter for Logistic Regression and Random Forest (both
options evaluated in the grid, not assumed), and for
HistGradientBoostingClassifier where the installed scikit-learn (1.7.2)
does support a `class_weight` parameter (checked at runtime via
`inspect.signature`, not assumed — recorded in metadata as
`hgb_class_weight_supported_by_installed_sklearn: true`). **No
oversampling or SMOTE was used**, and TEST's class distribution was never
altered. The winning Logistic Regression configuration used
`class_weight=None` — VALIDATION PR-AUC was highest without balancing for
this dataset; this is reported, not assumed a priori.

---

## 7. Candidate Algorithms

| Model | Family | Purpose |
|---|---|---|
| A | `DummyClassifier(strategy="prior")` | No-skill baseline every real candidate must beat — constant predicted probability equal to TRAIN prevalence |
| B | `LogisticRegression` | Required interpretable baseline |
| C | `RandomForestClassifier` | Tree ensemble, controlled hyperparameter search |
| D | `HistGradientBoostingClassifier` | sklearn's native boosting implementation (no external boosting library added) |

## 8. Candidate Hyperparameters (small, meaningful grids — 32 total combinations)

| Family | Grid | Combinations |
|---|---|---:|
| LogisticRegression | `C ∈ {0.01, 0.1, 1, 10}` × `class_weight ∈ {None, "balanced"}` | 8 |
| RandomForest | `n_estimators=400` (fixed) × `max_depth ∈ {8, 16}` × `min_samples_leaf ∈ {5, 20}` × `class_weight ∈ {None, "balanced"}` | 8 |
| HistGradientBoosting | `max_iter ∈ {100, 200}` × `max_depth ∈ {3, None}` × `learning_rate ∈ {0.05, 0.1}` × `class_weight ∈ {None, "balanced"}` | 16 |

Every combination was fit on TRAIN and scored on VALIDATION (PR-AUC
primary, Brier secondary tie-break) — full log in
`artifacts/model_evaluation/hyperparameter_search.csv`. `random_state=42`
throughout; `RandomForestClassifier(n_jobs=-1)` remains deterministic
under a fixed `random_state` regardless of parallelism.

---

## 9. TRAIN-Only Fitting Policy

Every `ColumnTransformer`/model `.fit()` call in `backend/modeling/train.py`
is invoked with `X_train`/`y_train` only. Calibration mappings
(`CalibratedClassifierCV`) use 5-fold `StratifiedKFold` cross-validation
**within TRAIN only** — VALIDATION and TEST are never used to fit any
imputer, encoder, scaler, model, or calibration mapping.

## 10. VALIDATION Model-Selection Policy

`backend/modeling/train.py::select_model_on_validation(X_train, y_train,
X_val, y_val, numeric_features, categorical_features)` performs the
entire candidate search, calibration comparison, winner selection, and
threshold design. **This function's signature has no TEST-related
parameter at all** — enforced both by a module-level assertion
(`_assert_no_test_parameter()`, runs at import time) and by a dedicated
test (`test_select_model_on_validation_has_no_test_parameter`) plus a
source-code scan confirming the function body references no
TEST-related identifier
(`test_select_model_on_validation_source_has_no_test_identifiers`). TEST
is loaded into the script only *after* this function returns and its
result has already been written to
`artifacts/model_evaluation/final_model_selection.json`.

---

## 11. Candidate VALIDATION Results

Full metric bundle for each of the four finalists (best hyperparameters
per family), confusion matrix at the stated default threshold of 0.5
(this is **not** the final operational threshold — see §14):

| Candidate | Hyperparameters | ROC-AUC | PR-AUC | Brier | Log loss | Precision@0.5 | Recall@0.5 | Specificity@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy baseline | `strategy=prior` | 0.5000 | 0.0958 | 0.0867 | — | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Logistic Regression | `C=0.01, class_weight=None` | 0.5728 | **0.1220** | 0.0861 | 0.3129 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Random Forest | `max_depth=8, min_samples_leaf=5, class_weight=None` | 0.5709 | 0.1148 | 0.0862 | — | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| HistGradientBoosting | `max_iter=100, max_depth=3, learning_rate=0.1, class_weight=balanced` | 0.5706 | 0.1157 | 0.2334 | — | 0.1147 | 0.9958 | 0.0623 | 0.2056 |

(At a 0.5 threshold, three of the four candidates select essentially
nobody — expected at ~9% prevalence with unbalanced/mildly-tuned models;
0.5 is reported only because the spec calls for a stated default, not
because it's operationally meaningful. The HGB grid's *winning* combo
happened to use `class_weight="balanced"`, which shifts its raw
probabilities well above 0.5 for most rows — hence its very different
0.5-threshold profile and its poor uncalibrated Brier score, addressed in
§12.)

**PR-AUC no-skill baseline for context:** VALIDATION prevalence = 0.0958,
matching the Dummy baseline's PR-AUC (0.0958) almost exactly, confirming
the metric behaves as expected under imbalance.

Full hyperparameter search log (32 rows): `artifacts/model_evaluation/hyperparameter_search.csv`.

---

## 12. Calibration Analysis

Applied to the two leading (highest-PR-AUC) non-dummy candidates —
Logistic Regression and HistGradientBoosting — comparing uncalibrated vs.
`CalibratedClassifierCV(method="sigmoid")` vs.
`CalibratedClassifierCV(method="isotonic")`, all fit via 5-fold CV on
TRAIN only, evaluated on VALIDATION:

| Candidate | Calibration | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
| Logistic Regression | uncalibrated | 0.5728 | **0.1220** | **0.0861** |
| Logistic Regression | sigmoid | 0.5712 | 0.1218 | 0.0862 |
| Logistic Regression | isotonic | 0.5715 | 0.1216 | 0.0861 |
| HistGradientBoosting | uncalibrated | 0.5706 | 0.1157 | 0.2334 |
| HistGradientBoosting | sigmoid | 0.5686 | 0.1143 | 0.0863 |
| HistGradientBoosting | isotonic | 0.5715 | 0.1154 | 0.0862 |

**Logistic Regression's uncalibrated probabilities were already the best
option on every metric** — calibration made PR-AUC and Brier marginally
*worse*, not better, so the simpler uncalibrated model was retained (no
material improvement to justify the added complexity — directly
satisfying the spec's "if calibration does not materially improve the
selected model, retain the simpler uncalibrated model" instruction).

**HistGradientBoosting is a genuinely useful illustration of *why*
calibration matters here**: its winning hyperparameter combination used
`class_weight="balanced"`, which pushed its raw `predict_proba` output
well away from the true prevalence — Brier score 0.2334 uncalibrated vs.
~0.086 for every other candidate. Sigmoid/isotonic calibration corrected
this back down to ~0.086 without hurting ranking (PR-AUC), confirming the
calibration step is doing real, necessary work for a class-weighted
booster, even though the ultimately-selected model (uncalibrated LR)
didn't need it.

Full log: `artifacts/model_evaluation/calibration_comparison.csv`.

---

## 13. Final Model-Selection Rationale

**Selected: Logistic Regression, `C=0.01`, `class_weight=None`,
uncalibrated.**

Applying the Phase 4 objective in order:
1. **PR-AUC** (primary): 0.1220, the single highest value across *every*
   candidate/calibration combination evaluated (both leading candidates,
   all three calibration variants each) — not just highest among
   Logistic Regression's own variants.
2. **Calibration quality / Brier**: also the best (0.0861) among the
   compared options — no tension between the two primary criteria here.
3. **Recall at operationally reasonable precision**: assessed via the
   threshold sweep (§14) — comparable in shape to the other candidates at
   the achievable precision levels this feature set supports.
4. **Temporal stability**: confirmed post-hoc on TEST (§18) — PR-AUC/
   ROC-AUC/Brier all close to VALIDATION, no material degradation.
5. **Interpretability**: Logistic Regression's signed coefficients are
   directly reportable (§20) — the most interpretable of the four
   candidates by construction.
6. **Model complexity**: the simplest model family.

Because Logistic Regression already won on the primary metric outright,
**the spec's interpretability/complexity tie-break rule was evaluated but
not needed to change the outcome** (`tie_break_applied: false` in
`artifacts/model_evaluation/final_model_selection.json`) — the winner
would have been chosen on PR-AUC alone, and happens to also be the
simplest, most interpretable option. This is not a case of settling for a
weaker-but-simpler model; the simple model is genuinely the best-measured
one here. **ROC-AUC was not the deciding metric** — Random Forest's and
HistGradientBoosting's ROC-AUC values were statistically indistinguishable
from Logistic Regression's (0.570–0.573 across all three), so ROC-AUC
alone would not have discriminated between them; PR-AUC and Brier did.

---

## 14. Threshold Design

Thresholds were derived from the **selected model's own VALIDATION score
distribution** (`numpy.percentile`), not invented round numbers:

- `high_threshold = 90th percentile of VALIDATION predicted probabilities ≈ 0.1238` — the top ~10% of the population by predicted risk.
- `moderate_threshold = 65th percentile ≈ 0.0976` — the next ~25% of the population (from the 65th to 90th percentile).

Representative points from the sweep
(`artifacts/model_evaluation/threshold_analysis.csv`, 94 thresholds from
0.02 to 0.95 in steps of 0.01) bracketing the two chosen values, computed
on TEST for illustration of the operating characteristics near each cut:

| Threshold | % selected | Precision | Recall | Specificity | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.10 (≈ moderate) | 31.7% | 0.116 | 0.385 | 0.690 | 0.179 |
| 0.12 (≈ high) | 12.3% | 0.131 | 0.167 | 0.882 | 0.147 |

## 15. LOW/MODERATE/HIGH Definitions

| Tier | Threshold rule | Operational meaning |
|---|---|---|
| LOW | `probability < moderate_threshold` | Lower predicted likelihood; no proactive escalation by risk alone. |
| MODERATE | `moderate_threshold <= probability < high_threshold` | Meaningful navigation opportunity; suitable for light-touch navigation or review. |
| HIGH | `probability >= high_threshold` | Strongest predicted navigation opportunity; candidate for stronger outreach / Care Management review **in a later phase** — this phase assigns the tier only, it does not decide what to do about it. |

`backend/modeling/risk_tiers.py::assign_risk_tier[s]()` implements this;
`validate_thresholds()` enforces `0 <= moderate_threshold < high_threshold <= 1`
at every call site (artifact load, tier assignment, tests).

---

## 16. Validation Risk-Tier Analysis

| Tier | Members | % pop. | Positives | Observed prevalence | Lift vs. overall | Mean predicted probability |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 6,500 | 65.00% | 555 | 8.54% | 0.89× | 0.0755 |
| MODERATE | 2,500 | 25.00% | 276 | 11.04% | 1.15× | 0.1085 |
| HIGH | 1,000 | 10.00% | 127 | 12.70% | 1.33× | 0.1427 |

Monotonic as required: 8.54% < 11.04% < 12.70%.

---

## 17. Frozen Model Specification Before TEST

Written to `artifacts/model_evaluation/final_model_selection.json`
*before* `TEST_CSV` is loaded anywhere in `train.py`:

```
algorithm:              logistic_regression
calibration_method:     uncalibrated
hyperparameters:        {"C": 0.01, "class_weight": null}
preprocessing:           median-impute numeric, StandardScaler (LR only),
                         most-frequent-impute + one-hot categorical (gender)
feature_list:            59 features, manifest order (frozen)
moderate_threshold:      0.097609
high_threshold:          0.123776
model_version_candidate: uc07-risk-v1
```

Nothing in this specification was changed after TEST was evaluated.

---

## 18. Final TEST Performance

Evaluated exactly once:

| Metric | VALIDATION | TEST |
|---|---:|---:|
| ROC-AUC | 0.5728 | 0.5747 |
| PR-AUC | 0.1220 | 0.1111 |
| Brier score | 0.0861 | 0.0822 |
| Log loss | 0.3129 | 0.3018 |

Confusion matrices on TEST:

| Threshold | Selected | TP | FP | FN | TN | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| moderate (0.0976) | 3,474 (34.7%) | 392 | 3,082 | 516 | 6,010 | 0.1128 | 0.4317 | 0.6610 | 0.1789 |
| high (0.1238) | 973 (9.7%) | 119 | 854 | 789 | 8,238 | 0.1223 | 0.1311 | 0.9061 | 0.1265 |

## 19. TEST Risk-Tier Analysis

| Tier | Members | % pop. | Positives | Observed prevalence | Lift vs. overall | Mean predicted probability |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 6,526 | 65.26% | 516 | 7.91% | 0.87× | 0.0752 |
| MODERATE | 2,501 | 25.01% | 273 | 10.92% | 1.20× | 0.1084 |
| HIGH | 973 | 9.73% | 119 | 12.23% | 1.35× | 0.1437 |

Monotonic (7.91% < 10.92% < 12.23%), and closely tracks the VALIDATION
tier table in both population share and prevalence — the threshold design
generalizes stably to an unseen snapshot.

## 20. Calibration on TEST

10-bin calibration curve computed on TEST without any recalibration
(`artifacts/model_evaluation/calibration_bins.csv`); Brier score 0.0822
(TEST) vs. 0.0861 (VALIDATION) — calibration held up slightly *better* on
TEST than VALIDATION, not worse.

## 21. VALIDATION vs. TEST Comparison

| Metric | Δ (TEST − VALIDATION) | Flagged? |
|---|---:|---|
| PR-AUC | −0.0109 | No |
| ROC-AUC | +0.0019 | No |
| Brier | −0.0040 (better) | No |

Degradation rule (declared before comparison, applied mechanically):
flag if `pr_auc_delta < -0.03` or `brier_delta > 0.01`. Neither condition
was met — **no material degradation**. No tuning occurred after this
comparison was computed.

---

## 22. Initial Subgroup Sanity Checks

Computed on TEST for the final frozen model (minimum 200 members and 15
positives per group to report ROC-AUC/PR-AUC; all groups below met the
size bar). This is an initial sanity pass, **not** the full Phase 6
bias/fairness audit.

| Subgroup | n | Prevalence | ROC-AUC | PR-AUC | Recall@moderate | Precision@moderate |
|---|---:|---:|---:|---:|---:|---:|
| age 0–35 | 521 | 8.83% | 0.583 | 0.108 | 0.500 | 0.106 |
| age 35–50 | 1,975 | 8.51% | 0.611 | 0.116 | 0.524 | 0.119 |
| age 50–65 | 3,480 | 8.88% | 0.573 | 0.109 | 0.434 | 0.112 |
| age 65–80 | 2,838 | 9.06% | 0.561 | 0.108 | 0.393 | 0.107 |
| age 80+ | 1,186 | 10.79% | 0.563 | 0.136 | 0.359 | 0.122 |
| gender F | 5,049 | 9.03% | 0.588 | 0.115 | 0.393 | 0.117 |
| gender M | 4,951 | 9.13% | 0.562 | 0.110 | 0.471 | 0.110 |
| clinical_burden 0–1 | 4,047 | 7.24% | 0.570 | 0.085 | 0.229 | 0.086 |
| clinical_burden 1–2 | 3,983 | 9.67% | 0.560 | 0.114 | 0.468 | 0.117 |
| clinical_burden 2–3 | 1,568 | 11.80% | 0.562 | 0.140 | 0.611 | 0.133 |
| clinical_burden 3+ | 402 | 11.19% | **0.483** | 0.120 | 0.711 | 0.103 |
| transportation_barrier=0 | 8,572 | 8.98% | 0.581 | 0.113 | 0.391 | 0.114 |
| transportation_barrier=1 | 1,428 | 9.66% | 0.542 | 0.110 | 0.659 | 0.109 |

**Observations flagged for deeper Phase 6 review (not conclusions):**
- `clinical_burden 3+` shows ROC-AUC of 0.483 — *below* the 0.5 no-skill
  line, on a group of 402 members. This is the one group where the model
  may not be ranking meaningfully better than chance; worth targeted
  investigation, though the sample is the smallest reported group and
  could be noise.
- ROC-AUC is somewhat higher for `gender=F` (0.588) than `gender=M`
  (0.562), and for younger age bands than older ones. All differences are
  modest relative to the model's overall weak discrimination and are not
  interpreted here as evidence of unfairness — only as candidates for the
  dedicated Phase 6 analysis.
- This model is **not** declared fair by this check; it is explicitly
  scoped as a sanity pass only.

Full table: `artifacts/model_evaluation/subgroup_metrics.csv`.

---

## 23. Feature Importance

Logistic Regression signed coefficients (post-preprocessing, standardized
numeric features), top 10 by absolute value:

| Rank | Feature | Coefficient | Direction |
|---:|---|---:|---|
| 1 | `has_prior_ed` | −0.2484 | reduces predicted risk |
| 2 | `transportation_barrier` | +0.1302 | increases predicted risk |
| 3 | `days_since_prior_care_management` | +0.0878 | increases predicted risk |
| 4 | `prior_potentially_avoidable_ed_count_270d` | −0.0816 | reduces predicted risk |
| 5 | `prior_uncertain_ed_count_270d` | +0.0750 | increases predicted risk |
| 6 | `prior_urgent_care_count_180d` | +0.0646 | increases predicted risk |
| 7 | `prior_telehealth_count_270d` | +0.0638 | increases predicted risk |
| 8 | `prior_potentially_avoidable_ed_count_30d` | +0.0634 | increases predicted risk |
| 9 | `ckd` | +0.0617 | increases predicted risk |
| 10 | `age` | −0.0610 | reduces predicted risk |

**Caution on interpretation:** several prior-ED-utilization count
features have *negative* coefficients (e.g. `has_prior_ed`,
`prior_potentially_avoidable_ed_count_270d`), which is counter to a naive
"more prior ED use → more future risk" intuition. With ~15 mutually
correlated prior-utilization features feeding one L2-penalized linear
model on a modest positive-class sample (904 TRAIN positives), individual
coefficient signs are influenced by multicollinearity and are **not**
individually reliable causal statements — this is a documented risk to
flag for Phase 5's explanation-language design (Phase 2 §21's "do not
imply causation" rule applies directly here). The overall ranking
(PR-AUC/Brier) is more trustworthy than any single coefficient's sign.

Full table (all 59 post-encoding features): `artifacts/model_evaluation/global_feature_importance.csv`.

---

## 24. Model Artifact Structure

`backend/models/uc07_risk_v1_model.joblib` — a single joblib-serialized
Python dict (matches the legacy artifact's dict-wrapper convention):

```python
{
    "pipeline": <sklearn Pipeline: preprocessor + LogisticRegression>,
    "feature_columns": [...59 names, frozen order...],
    "numeric_features": [...58 names...],
    "categorical_features": ["gender"],
    "target": "future_potentially_avoidable_ed_90d",
    "model_version": "uc07-risk-v1",
    "algorithm": "logistic_regression",
    "calibration_method": "uncalibrated",
    "hyperparameters": {"C": 0.01, "class_weight": None},
    "moderate_threshold": 0.097609,
    "high_threshold": 0.123776,
}
```

Does **not** overwrite `backend/ed_risk_model.pkl` (legacy artifact,
untouched — verified by
`test_legacy_model_artifact_untouched`, which reloads it and confirms its
original `target: "frequent_ED_user"` shape is unchanged).

## 25. Model Metadata

`backend/models/uc07_risk_v1_model_metadata.json` — see §11–§21 above for
its content; full schema matches the Phase 4 spec's Step 18 checklist
(model name/version, target + definition, horizon/observation window,
algorithm + hyperparameters, calibration method, both thresholds, full
feature list + count, all three index dates, all three prevalences,
validation metrics, final test metrics, both risk-tier tables,
Python/sklearn/pandas/numpy versions, training timestamp, raw dataset
hashes, snapshot hashes, feature manifest reference, and known
limitations).

---

## 26. Reproducibility / Versioning

- `RANDOM_STATE = 42` used for every stochastic component: `DummyClassifier`,
  `LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`,
  and the `StratifiedKFold(shuffle=True, random_state=42)` used for
  calibration CV.
- Re-running `python backend/modeling/train.py` twice produced
  byte-identical VALIDATION/TEST metrics both times (PR-AUC 0.1111,
  ROC-AUC 0.5747, Brier 0.0822 on TEST, verified during this phase's own
  development).
- Model version: **`uc07-risk-v1`** — deliberately not named
  `final_model`/`best_model_final`/similar, per the spec's explicit
  instruction; the model is expected to evolve in later phases.
- `scikit-learn==1.7.2` and `joblib==1.5.3` are now both pinned in
  `backend/requirements.txt` (joblib was previously unpinned; pinned in
  this phase since it is now directly load-bearing for the new artifact's
  serialization, and per the spec's specific caution about
  scikit-learn-version-driven artifact incompatibility). No other
  dependency was changed. `pandas`, `numpy`, `fastapi`, `uvicorn`, etc.
  remain unpinned, unchanged from Phase 1–3.

---

## 27. Known Limitations

1. **Weak overall discrimination** (ROC-AUC ≈ 0.57, PR-AUC ≈ 0.11 vs. a
   0.096 no-skill baseline) — a direct, honest consequence of correctly
   excluding leakage-risk features (diagnosis crosstabs, same-window ED
   counts) per Phase 1–3. This is not a modeling bug; it is the genuine
   ceiling of the current leakage-safe feature set on this dataset.
2. **`clinical_burden 3+` subgroup ROC-AUC (0.483) is below no-skill** on
   a modest sample (402 members) — flagged for Phase 6, not resolved here.
3. **Several top feature coefficients have counter-intuitive signs**
   (§23) due to multicollinearity among ~15 correlated prior-utilization
   features — individual coefficients should not be read causally.
4. Carries forward, unchanged, every Phase 2/3 known limitation:
   `UNCERTAIN`-as-negative labeling, diagnosis exclusion, ~7%
   cross-snapshot positive-member overlap, single 547-day data era, and
   `gender`'s pending fairness review.
5. **Risk tiers were derived from a single VALIDATION snapshot's score
   distribution** (percentile-based) — they have not been stress-tested
   against a different time period or population shift; Phase 5/6 should
   monitor tier stability as new data becomes available.

---

## 28. Phase 5 Readiness Assessment

A versioned, tested, leakage-verified risk model with calibrated-enough
probabilities and validated, monotonic risk tiers now exists. Phase 5 can
proceed to implement the Care Navigation Agent (routing among PCP/Urgent
Care/Telehealth/Care Management using this model's `risk_tier` output plus
the already-approved access/utilization feature groups) and the Safety &
Policy Agent (per the Phase 2 contracts), then wire both together with
the Risk Detection Agent behind the existing FastAPI endpoints — updating
`predict.py`/`main.py` and the frontend only at that point, not before.
Given the model's modest discrimination, Phase 5's Care Navigation and
Safety design should treat the risk score as one input among several
(access, utilization pattern, chronic burden), not as a sole
determinant — consistent with the Phase 2 architecture's original intent.

---

## Architecture Diagram

```
                    TRAIN (10,000 rows, 904 positives)
                                 |
                                 v
              Fit Candidate Models (Dummy / LR / RF / HGB)
              32 hyperparameter combinations, all fit on TRAIN only
                                 |
                                 v
                 VALIDATION (10,000 rows, 958 positives)
                                 |
                 +-------------------------------+
                 |     model selection (PR-AUC)    |
                 |     calibration selection        |
                 |     (5-fold CV fit on TRAIN)       |
                 |     threshold selection              |
                 |     (65th / 90th percentile)            |
                 +-------------------------------+
                                 |
                                 v
                        FREEZE MODEL
        (final_model_selection.json written BEFORE TEST loads)
                                 |
                                 v
                   TEST ONCE (10,000 rows, 908 positives)
                                 |
                                 v
                       UC07 RISK MODEL v1
        backend/models/uc07_risk_v1_model.joblib + metadata.json
```
