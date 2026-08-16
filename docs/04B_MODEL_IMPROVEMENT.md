# UC07 — Controlled Model & Feature Improvement (Phase 4B)

**Implementation date:** 2026-08-15
**Phase:** 4B — Controlled Model & Feature Improvement (experimentation against VALIDATION only; one gated TEST check)
**Builds on:** `docs/01_PROJECT_BASELINE.md` … `docs/04_MODEL_DEVELOPMENT.md`, `docs/DECISION_LOG.md`

**Final decision: KEEP UC07-RISK-V1.** No `uc07-risk-v2` artifact was
created. This is reported as a legitimate, evidence-based finding, not a
failure — the fixed three datasets, under the leakage controls locked in
Phases 2–4, do not support a materially better model than v1 using
additional point-in-time feature engineering.

---

## 1. Executive Summary

Phase 4B diagnosed why `uc07-risk-v1` (ROC-AUC 0.5747, PR-AUC 0.1111 on
TEST) is weak, then ran a controlled series of feature-engineering
experiments — restructured time windows, utilization velocity, care-setting
mix, care continuity, access×utilization interactions, historical-ED-pattern
extras, and a controlled historical-diagnosis representation — comparing
each against `uc07-risk-v1` on VALIDATION only. The single best
configuration found (a window restructuring that drops the most redundant
nested window) produced a **VALIDATION PR-AUC gain of +0.0002** — noise,
not signal — and every feature group added *after* that made VALIDATION
performance flat or measurably worse. Diagnosis features, tested under
strict point-in-time and volume-normalized controls, produced no
incremental signal (`-0.0002` PR-AUC vs. the same base feature set). No
suspicious/inflated performance was observed at any point (no ROC-AUC
approached the 0.80 guardrail). TEST was never opened, because no
candidate passed the pre-declared VALIDATION promotion gate — this is the
correct, by-design outcome for Decision B.

---

## 2. Why Phase 4B Was Performed

`uc07-risk-v1`'s TEST performance (ROC-AUC 0.5747, PR-AUC 0.1111) is
modest. Phase 4 attributed this to the strict, correct exclusion of
leakage-risk features rather than a modeling defect, but did not
systematically test whether *additional, still-leakage-safe*
point-in-time engineering could close some of that gap. Phase 4B exists
to answer that question rigorously, under the same non-negotiable
constraints (immutable raw data, unchanged target/windows/index dates,
sealed TEST) established in Phases 2–4.

---

## 3. V1 Limitations (Restated)

From `docs/04_MODEL_DEVELOPMENT.md` §27 and confirmed again here: modest
overall discrimination, a `clinical_burden 3+` subgroup with sub-no-skill
TEST ROC-AUC, and several top LR coefficients with counter-intuitive
signs. Phase 4B's Step 1 diagnosis (§4 below) explains the last point
directly.

---

## 4. Feature Diagnostics (Step 1)

Computed against TRAIN only (`artifacts/model_improvement/feature_diagnostics.csv`,
`high_correlation_pairs.csv`).

**Near-zero-variance / sparse features:** many 30-day-window counts are
>90% zero (`prior_care_management_count_30d` 98.7% zero,
`prior_urgent_care_count_30d` 97.5%, `prior_potentially_avoidable_ed_count_30d`
96.9%). These are extremely sparse given ~9% baseline event prevalence
and low per-member ED/care frequency — plausible sources of noise rather
than reliable signal.

**18 feature pairs with |r| ≥ 0.8** (full list in
`high_correlation_pairs.csv`), including:
- `num_chronic_conditions` vs. `clinical_burden`: r = 1.000 (exactly
  redundant, already flagged in Phase 1/3).
- Each `has_prior_X` flag vs. its own `prior_X_count_270d`: r = 0.83–0.97
  across ED, PCP, Urgent Care, Telehealth, and Care Management — these
  flags turn out to be highly redundant with the corresponding count, not
  primarily with the recency field they were designed to disambiguate
  (correlation of `has_prior_ed` with the *imputed* `days_since_prior_ed`
  is only 0.05).
- `transportation_barrier` vs. `access_burden`: r = 0.87 (expected —
  `access_burden` is partly built from `transportation_barrier`).

**Nested-window redundancy is uneven, not uniform:** for
`prior_ed_count_{30,90,180,270}d`, the 180d↔270d pair is the most
redundant (r = 0.81), while 30d↔270d is nearly independent (r = 0.33).
This directly informed the "reduced windows" design in §7 — dropping 180d
(the most redundant window), not 30d (the least redundant), which a naive
guess might have targeted instead.

### Coefficient sign investigation (`has_prior_ed`, `prior_potentially_avoidable_ed_count_270d`)

Both carry negative Logistic Regression coefficients in v1 — the
Phase 4B spec required determining *why*, without inventing a clinical
explanation. Directly checked against TRAIN:

| has_prior_ed | Observed future-positive rate |
|---|---:|
| 0 (no prior ED) | 10.85% |
| 1 (has prior ED) | 7.36% |

| prior_potentially_avoidable_ed_count_270d | Observed future-positive rate | n |
|---:|---:|---:|
| 0 | 9.62% | 7,451 |
| 1 | 7.59% | 2,279 |
| 2 | 5.74% | 244 |
| 3+ | 0.00% | 26 |

**Finding: this is a genuine univariate pattern in the data, not a
multicollinearity/suppression artifact.** The negative direction is
already present before any model is fit, across bins with substantial
sample size (n=7,451 and n=2,279) — only the most extreme bin (3+, n=26)
is small enough that its exact 0.00% should be read as noisy. The
high correlation between `has_prior_ed` and `prior_ed_count_270d` (r=0.83,
§4) does affect *how* this shared signal gets distributed across several
correlated coefficients under L2 regularization, but it does not
manufacture the negative *direction* — that direction is real,
univariate, and present in the raw counts. No causal interpretation is
offered for *why* this pattern exists in the source data; it is reported
as observed only.

---

## 5. Multicollinearity Findings

Summarized in §4. In addition: `ed_utilization_velocity_30_over_180` and
`potentially_avoidable_ed_velocity_90_over_270` (v1's existing velocity
features) each correlate strongly with their own numerator count
(r = 0.95 and 0.96 respectively) — largely because both ratios' numerators
are usually small integers and the *270d* denominator dominates the
ratio's variance far less than the numerator does. This motivated
Phase 4B's replacement velocity formulas (§9), which use a *remainder*
denominator (`recent_count / (total_count - recent_count)`) specifically
to reduce this numerator-dominance effect — though as shown in §13, this
did not translate into a measurable VALIDATION improvement either.

---

## 6. Feature-Group Design

| Group | Contents |
|---|---|
| A. Static/demographic | `age`, `gender` |
| B. Chronic burden | 6 condition flags, `num_chronic_conditions`, `clinical_burden` |
| C. Access barriers | `transportation_barrier`, `telehealth_available`, `pcp_distance_miles`, `urgent_care_distance_miles`, `access_burden` |
| D. ED utilization | `prior_ed_count_*` (nested, reduced, or banded — §7) |
| E. Historical potentially avoidable ED | `prior_potentially_avoidable_ed_count_*`, recency, `avoidable_share_of_prior_ed_270d`, `repeat_potentially_avoidable_ed_flag` |
| F. High-acuity/uncertain ED history | `prior_protected_ed_count_*`, `prior_uncertain_ed_count_*`, recency |
| G. Outpatient/alternative-care engagement | `prior_{pcp,urgent_care,telehealth,care_management}_count_*`, recency, care-setting mix ratios |
| H. Recency | every `days_since_prior_*` / `has_prior_*` pair |
| I. Utilization velocity/trend | `ed_acceleration_30_vs_240`, `potentially_avoidable_ed_acceleration_30_vs_240`, `alternative_care_engagement_trend_90_vs_270` |
| J. Cross-setting care patterns | `care_setting_diversity_270d`, `days_since_any_outpatient_contact`, `long_gap_without_outpatient_care_flag`, `recent_outpatient_contact_30d_flag`, `repeated_ED_without_recent_PCP_flag`, `has_recent_outpatient_followup_after_last_ed`, `days_from_last_ed_to_next_outpatient` |
| K. Optional historical diagnosis | `distinct_prior_ed_diagnosis_categories_270d`, `most_common_prior_diagnosis_share_270d`, `prior_diagnosis_diversity_ratio_270d`, `repeat_same_diagnosis_flag_270d` |

Implemented in `backend/pit/features_v2.py` — additive only; never
modifies `backend/pit/features.py`, `windows.py`, `target.py`, or
`encounter_classification.py`.

---

## 7. Temporal-Window Experiments (Step 3)

Three representations of `prior_ed_count_*` /
`prior_potentially_avoidable_ed_count_*` compared, everything else held
identical, LR-tuned on each, VALIDATION-scored
(`artifacts/model_improvement/window_experiment.csv`):

| Variant | Feature count | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| Baseline (nested 30/90/180/270d) | 59 | 0.5728 | 0.12198 | 0.08614 |
| **Reduced nested (30/90/270d, drop 180d)** | 57 | 0.5731 | **0.12209** | 0.08614 |
| Non-overlapping bands (0-30/31-90/91-180/181-270) | 59 | 0.5717 | 0.12177 | 0.08616 |

**Reduced windows wins by 0.00012 PR-AUC** — within noise, but it is the
best of the three and uses 2 fewer features, so it was carried forward as
the Phase 4B window representation. Banded (non-overlapping) counts
performed *worse* than the nested cumulative representation, contrary to
what might be assumed — cumulative windows apparently carry the same or
more usable signal for this linear model than disjoint bands do.

---

## 8. Velocity Features (Step 4)

Formulas (`backend/pit/features_v2.py`), denominators floored at 1:

```
ed_acceleration_30_vs_240 = prior_ed_count_30d / max(prior_ed_count_270d - prior_ed_count_30d, 1)
potentially_avoidable_ed_acceleration_30_vs_240 = prior_potentially_avoidable_ed_count_30d / max(prior_potentially_avoidable_ed_count_270d - prior_potentially_avoidable_ed_count_30d, 1)
alternative_care_engagement_trend_90_vs_270 = total_alt_care_90d / max(total_alt_care_270d - total_alt_care_90d, 1)
```

`care_management`/`pcp`/`telehealth` "engagement recency" concepts named
in the spec are already covered by v1's existing `days_since_prior_pcp`
etc. and were not duplicated.

---

## 9. Care-Setting Features (Step 5)

`total_outpatient_alternative_visits_270d`, `ed_to_outpatient_ratio_270d`,
`ed_share_of_total_utilization_270d`, `telehealth_share_270d`,
`urgent_care_share_270d`, `pcp_share_270d` (all ratios floored-denominator
safe), plus `has_recent_outpatient_followup_after_last_ed` /
`days_from_last_ed_to_next_outpatient` — computed by finding, for each
member, the earliest care-history visit strictly *after* their most
recent prior ED visit and strictly *before* `index_date` (both events
drawn only from the observation window, so nothing after `index_date` is
ever used, per the spec's explicit warning).

---

## 10. Access Interactions (Step 7)

Six interactions, each with an explicit rationale (not an exhaustive
pairwise sweep): `transportation_barrier × prior_ed_count_30d`,
`pcp_distance_miles × prior_ed_count_30d`,
`urgent_care_distance_miles × prior_ed_count_30d`,
`telehealth_available × prior_ed_count_30d` (does telehealth availability
relate differently to recent ED use?), `clinical_burden × access_burden`,
`clinical_burden × prior_ed_count_30d` (compounding-burden patterns).

---

## 11. Historical ED-Pattern Features (Step 8)

`avoidable_share_of_prior_ed_270d` (what fraction of prior ED use was
potentially avoidable) and `repeat_potentially_avoidable_ed_flag`
(`prior_potentially_avoidable_ed_count_270d >= 2`). Everything else this
step names already exists in v1.

---

## 12. Diagnosis Experiment (Step 9)

Controlled, compressed, volume-normalized representations only (never a
per-category crosstab): `distinct_prior_ed_diagnosis_categories_270d`,
`most_common_prior_diagnosis_share_270d` (= most-frequent-category count
/ `prior_ed_count_270d`, floored at 1), `prior_diagnosis_diversity_ratio_270d`
(= distinct categories / `prior_ed_count_270d`), `repeat_same_diagnosis_flag_270d`.
Every feature is normalized by the member's own ED volume specifically so
it captures *spread/concentration*, not merely re-encoding volume.

**WITHOUT_DIAGNOSIS vs. WITH_DIAGNOSIS**, controlling for everything else
(base = the actual best-performing non-diagnosis experiment, §13, not a
fixed assumption of "the full stack"):

| Variant | Feature count | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| WITHOUT_DIAGNOSIS | 55 | 0.5730 | 0.12210 | 0.08614 |
| WITH_DIAGNOSIS | 59 | 0.5738 | 0.12186 | 0.08615 |

**PR-AUC change: −0.00024.** Diagnosis added no incremental VALIDATION
signal beyond ED utilization counts (in fact very slightly negative) —
**excluded from the Phase 4B candidate feature set**, consistent with the
required-proof-before-inclusion rule.

---

## 13. Ablation Study (Step 10)

All experiments fit Logistic Regression (small grid, matching Phase 4's
methodology) on TRAIN, scored on VALIDATION
(`artifacts/model_improvement/feature_ablation_results.csv`):

| Experiment | Features | ROC-AUC | PR-AUC | Brier | Recall@P90 | Precision@P90 |
|---|---:|---:|---:|---:|---:|---:|
| A — V1 baseline | 59 | 0.5728 | 0.12198 | 0.08614 | 0.1326 | 0.127 |
| **B — restructured windows** | **55** | **0.5730** | **0.12210** | **0.08614** | 0.1347 | 0.129 |
| C — B + velocity | 58 | 0.5727 | 0.12198 | 0.08614 | 0.1326 | 0.127 |
| D — C + care-mix/continuity | 71 | 0.5751 | 0.12163 | **0.24201** ⚠ | 0.1367 | 0.131 |
| E — D + access interactions | 77 | 0.5712 | 0.11842 | 0.08646 | 0.1399 | 0.134 |
| F — E + historical ED extras | 79 | 0.5726 | 0.11862 | 0.08645 | 0.1409 | 0.135 |
| G — F + diagnosis | 83 | 0.5722 | 0.11780 | 0.08627 | 0.1336 | 0.128 |

**B is the actual PR-AUC winner** — selected by evidence (argmax
VALIDATION PR-AUC among A–F, with a Brier sanity filter), not by
mechanically cascading to the last experiment tried. Every group added
after C leaves PR-AUC flat-to-worse; **D's Brier score explodes to
0.242** because its winning hyperparameter grid combination happened to
select `class_weight="balanced"` — the same well-understood effect
observed for HistGradientBoosting in Phase 4 (§14 of
`docs/04_MODEL_DEVELOPMENT.md`): class weighting shifts raw probabilities
away from true prevalence without necessarily helping ranking. This is a
calibration artifact of that specific grid point, not evidence the
care-mix/continuity features themselves are harmful — but it is also not
evidence they help, since D's uncalibrated PR-AUC (0.12163) is still
below B's (0.12210).

**Conclusion: none of Groups I (velocity), G/J (care-mix/continuity), the
access interactions, the historical-ED-pattern extras, or diagnosis (K)
produced a real VALIDATION improvement over the restructured-windows
baseline (B).** The full ablation ladder is retained in the report for
transparency even though most of it did not help.

---

## 14. Model Comparison (Step 11)

On B's 55-feature set (`artifacts/model_improvement/model_comparison_v2.csv`):

| Algorithm | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| **Logistic Regression** | 0.5730 | **0.12210** | 0.08614 |
| Random Forest | 0.5722 | 0.11572 | 0.08616 |
| HistGradientBoosting | 0.5703 | 0.11567 | 0.23374 (uncalibrated) |

Logistic Regression remains the best-performing algorithm after feature
restructuring — the same conclusion as Phase 4.

## Regularization / LR Stability (Step 12)

L2 (matching v1's approach), L1, and ElasticNet compared across
`C ∈ {0.01, 0.1, 1.0}` (`artifacts/model_improvement/regularization_experiment.csv`).
Most informative rows:

| Penalty | C | Non-zero coefs | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| L2 (v1-style) | 0.01 | 56/56 | 0.5730 | 0.12210 | 0.08614 |
| L1 | 0.10 | 31/56 | 0.5775 | 0.12327 | 0.08607 |
| **ElasticNet** | **0.01** | **8/56** | **0.5885** | **0.12394** | 0.08607 |

**For full transparency:** the single best VALIDATION configuration found
anywhere in Phase 4B was ElasticNet (`C=0.01`, `l1_ratio=0.5`) on the
restructured-windows feature set — PR-AUC 0.12394 vs. v1's 0.12198, a
**+0.00196 absolute gain**. This is reported honestly even though it does
**not** cross the pre-declared promotion margin (≥0.01 absolute, §16) —
it is roughly a fifth of the required threshold. ElasticNet's aggressive
sparsification (only 8 of 56 coefficients non-zero) is a genuinely
interesting stability finding — it suggests most of the 56 features carry
redundant or near-zero information for this target — but it does not, on
its own, constitute the "materially higher PR-AUC" the promotion
criteria require.

## Calibration (Step 13)

Uncalibrated vs. sigmoid vs. isotonic compared for the two leading
candidates (`artifacts/model_improvement/calibration_comparison_v2.csv`),
5-fold CV fit on TRAIN only:

| Algorithm | Calibration | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
| Logistic Regression | uncalibrated | 0.5730 | 0.12210 | 0.08614 |
| Logistic Regression | **isotonic** | 0.5738 | **0.12214** | **0.08612** |
| Random Forest | uncalibrated | 0.5722 | 0.11572 | 0.08616 |
| Random Forest | sigmoid | 0.5706 | 0.11520 | 0.08625 |

Logistic Regression + isotonic is a razor-thin (+0.00004 PR-AUC) winner
over uncalibrated — essentially indistinguishable, retained as the
Phase 4B "V2 winner" purely for completeness of the pipeline, not because
it represents a meaningful improvement.

---

## 15. Leakage Audit (Step 15 Guardrail)

No experiment or calibration variant produced ROC-AUC above the 0.80
suspicious-performance ceiling (highest observed: 0.5885, the ElasticNet
row above) — **zero suspicious-performance flags raised**, so no
temporal/leakage audit was triggered. All new features were built via
`backend/pit/features_v2.py`, which reuses the same
`in_observation_window()` boundary function and `classify_ed_encounters()`
classifier as the frozen `features.py`, and is covered by dedicated tests
(`backend/tests/test_phase4b.py`) proving observation-window-only
filtering, correct band boundaries, safe ratio denominators, and absence
of any unwindowed diagnosis crosstab.

---

## 16. V1 vs. V2 VALIDATION Comparison

| Metric | V1 | V2 candidate (B, LR+isotonic) | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.5728 | 0.5738 | +0.0010 |
| PR-AUC | 0.1220 | 0.1221 | +0.0001 |
| Brier | 0.0861 | 0.0861 | −0.0000 |
| HIGH-tier lift | 1.326× | 1.365× | +0.030 relative |

**Promotion gate (`docs/04_MODEL_DEVELOPMENT.md`-style criteria, applied
mechanically and reported honestly):**

- PR-AUC gain ≥ 0.01 absolute: **not met** (+0.0002)
- HIGH-tier lift relative gain ≥ 0.15: **not met** (+0.030)
- Calibration preserved (Brier not materially worse): met
- Tier monotonicity preserved: met
- No suspicious performance: met

Because neither the PR-AUC case nor the tier-lift case was met, **the
promotion gate does not pass**, regardless of the other criteria being
satisfied.

---

## 17. Final Frozen Candidate

A frozen specification was written to
`artifacts/model_improvement/phase4b_decision.json` *before* the
promotion decision was finalized, exactly as the spec requires ("before
looking at TEST, lock the spec") — reproduced here:

```
algorithm:            logistic_regression
calibration_method:   isotonic
feature_list:         55 features (B: restructured windows, no velocity/
                       care-mix/interactions/diagnosis additions)
moderate_threshold:   (VALIDATION-derived, 65th percentile)
high_threshold:        (VALIDATION-derived, 90th percentile)
model_version_candidate: uc07-risk-v2
```

Because the promotion gate did not pass, **this frozen spec was never
used to build a production artifact** — no `uc07_risk_v2_model.joblib`
was created, consistent with the spec's explicit instruction not to
fabricate a v2 artifact when promotion criteria are not met.

---

## 18. V1 vs. V2 TEST Comparison

**TEST was not opened in Phase 4B.** Per the sealed-TEST policy, TEST is
only evaluated once a candidate passes the VALIDATION promotion gate
(§16). Since no candidate passed, `evaluate_v2_on_test()` was never
called — verified structurally (the candidate-selection function has no
TEST-named parameter, and its source contains no TEST-related identifier
— `backend/tests/test_phase4b.py::test_select_v2_candidate_has_no_test_parameter`
and `test_select_v2_candidate_source_has_no_test_identifiers`) and
confirmed by the absence of `test_comparison_v1_v2.csv`,
`test_risk_tiers_v2.csv`, `subgroup_sanity_v2.csv`, and
`global_feature_importance_v2.csv` in
`artifacts/model_improvement/` (all four are only ever written inside the
promoted-only code path). V1's existing TEST performance
(`docs/04_MODEL_DEVELOPMENT.md`: ROC-AUC 0.5747, PR-AUC 0.1111, Brier
0.0822, HIGH lift 1.35×) remains the only TEST evidence for this use
case.

---

## 19. Subgroup Sanity Check

Not performed in Phase 4B — the spec scopes this step to "a promoted V2
candidate," and no candidate was promoted. V1's existing initial subgroup
findings (`docs/04_MODEL_DEVELOPMENT.md` §22, including the
`clinical_burden 3+` sub-no-skill ROC-AUC) stand unchanged and remain
flagged for Phase 6.

---

## 20. Interpretability Comparison (Step 21)

The winning ablation experiment's (B) Logistic Regression coefficients
were inspected specifically to test whether window restructuring resolves
V1's confusing coefficient signs:

| Rank | Feature | V1 coefficient | B (restructured windows) coefficient |
|---:|---|---:|---:|
| 1 | `has_prior_ed` | −0.2484 | −0.2397 |
| 4 | `prior_potentially_avoidable_ed_count_270d` | −0.0816 | −0.0727 |

**No material change.** Both signs persist, nearly unchanged in
magnitude, in the restructured-windows feature set. This is consistent
with §4's finding that the negative signs reflect a genuine univariate
data pattern, not a windowing-induced multicollinearity artifact —
restructuring the windows was never expected to fix a pattern that isn't
caused by window structure, and it didn't. No causal statement is made
about *why* the pattern exists.

---

## 21. Final Decision

# DECISION B: KEEP UC07-RISK-V1

**Reason:** Across a window-representation experiment, a 7-step feature
ablation covering 6 new feature groups (velocity, care-setting mix, care
continuity, access interactions, historical-ED-pattern extras, and
controlled diagnosis), a 3-algorithm model comparison, a 3-penalty
regularization sweep, and a calibration comparison — all conducted on
TRAIN/VALIDATION only — the best VALIDATION PR-AUC improvement found
anywhere was +0.0002 (the promoted-comparison candidate) to +0.0020 (the
single best ElasticNet configuration, reported for transparency but not
adopted as the candidate). Both are far short of the ≥0.01 absolute
promotion margin, and the HIGH-tier lift gain (+3.0% relative) is far
short of the ≥15% margin. No feature group beyond the marginal window
restructuring showed real incremental VALIDATION signal; several (care-mix/
continuity, access interactions, diagnosis) modestly *hurt* PR-AUC. This
is not a failure of search effort — it is direct evidence that, under the
leakage controls locked in Phases 2–4, **the three fixed source datasets
do not currently support a materially better 90-day avoidable-ED risk
model than `uc07-risk-v1`.**

Per the spec's explicit instruction, this is treated as a valid,
non-negative scientific finding, not a Phase 4B failure.

---

## 22. Honest Assessment of Remaining Predictive Limitations

- The dataset's genuine prospective signal for this specific target (90-day
  forward potentially-avoidable ED risk, leakage-safe features only)
  appears to be intrinsically modest — ROC-AUC in the 0.57–0.59 range
  across every configuration tried, never approaching values that would
  indicate strong prospective separability.
- The counter-intuitive negative association between prior ED utilization
  and future avoidable-ED risk (§4) is real in this data and worth
  further investigation in a future phase (e.g., whether it reflects a
  genuine care-seeking/avoidance dynamic, a synthetic-data generation
  artifact, or something else) — but that investigation is out of scope
  for a feature-engineering phase and is flagged, not resolved, here.
- Sparse 30-day-window features (§4) may be contributing more noise than
  signal; a future phase could test dropping them explicitly (this
  overlaps with, but is distinct from, the window experiment already run).
- Diagnosis, even under careful volume-normalized, point-in-time controls,
  adds no signal in this dataset — this is now demonstrated, not merely
  assumed.

---

## 23. Implications for the Multi-Agent Architecture

The Phase 2 architecture already anticipated a model with "modest
discrimination" (`docs/02_UC07_AND_DATA_DESIGN.md`) and designed the Risk
Detection Agent's output as *one input among several* to Care Navigation,
not a sole determinant. Phase 4B's finding reinforces that this was the
correct design choice: Phase 5's Care Navigation Agent should continue to
weight access/utilization/chronic-burden signals directly (as already
planned) rather than leaning more heavily on the risk score than
Phase 2 intended, since Phase 4B found no additional feature engineering
meaningfully sharpens that score.

---

## 24. Phase 5 Recommendation

Proceed to Phase 5 using **`uc07-risk-v1`** exactly as produced in
Phase 4 (no change). Implement the Care Navigation Agent and Safety &
Policy Agent per the Phase 2 contracts, using v1's `risk_tier` output
alongside — not instead of — the access/utilization/chronic-burden
features already available. Phase 4B's negative results should be
retained as documentation (this file) rather than repeated; future model
improvement attempts should focus on data acquisition (new signal sources
beyond the three fixed datasets) rather than further feature
re-engineering of the same three datasets, since this phase's controlled
search suggests that avenue is largely exhausted.
