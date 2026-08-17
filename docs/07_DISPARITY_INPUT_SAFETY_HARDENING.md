# Phase 7 — Disparity Investigation, Input Validation & Safety-Context Hardening

**Date:** 2026-08-16
**Status:** Complete. Model frozen and unchanged. One safety-context contract formalized (no behavior regression). New input-validation and safety-context-schema modules added.

---

## 1. Objective

Answer three questions without touching the frozen model: (1) are the Phase 6 subgroup disparities understandable consequences of the synthetic data/model, or evidence of a serious design defect? (2) can invalid/malformed inputs be rejected reliably? (3) can the system represent current safety context distinguishing KNOWN / UNKNOWN / INCOMPLETE / HIGH-ACUITY-OVERRIDE without falsely assuming safety? This is a hardening phase, not model development.

## 2. Frozen model

`uc07-risk-synthetic-v1`, Logistic Regression, verified unchanged from `backend/models/uc07_risk_synthetic_v1_model_metadata.json`: TEST ROC-AUC 0.704752, PR-AUC 0.336636, Brier 0.102334, HIGH lift 2.87×, thresholds MODERATE=0.105986/HIGH=0.213252 — matched the brief exactly. Never retrained, tuned, recalibrated, or re-thresholded in this phase (Section 34).

## 3. Phase 6 carry-forward findings (reproduced)

Verified from `artifacts/phase6_validation/`: 389/389 tests passed; `transportation_barrier=1` recall@MODERATE=0.9898, FPR=0.9594, MODERATE+=96.86%; telehealth and clinical-burden disparities present; no static-field validation existed; `current_safety_context` was caller-supplied/unverified; the Phase 6 safety-context completeness fix (partial context must not produce CLEAR) was in place. All confirmed exactly as claimed — no discrepancy.

## 4. Transportation decomposition

`artifacts/phase7_hardening/transportation_decomposition.csv`. `transportation_barrier=1` (n=1,622) vs. `=0` (n=8,378):

| Covariate | barrier=0 | barrier=1 | Δ | corr. with barrier |
|---|---|---|---|---|
| Target prevalence | 9.75% | 30.21% | +20.46pp | — |
| Age (mean) | 53.38 | 55.43 | +2.05 | 0.044 |
| Clinical burden (mean) | 0.868 | 1.062 | +0.195 | 0.078 |
| Telehealth available (rate) | 78.03% | 63.50% | −14.5pp | −0.125 |
| PCP distance (mi) | 6.62 | 10.46 | +3.84 | 0.300 |
| Urgent care distance (mi) | 5.21 | 8.29 | +3.08 | 0.296 |
| Prior ED count (270d) | 0.548 | 1.562 | +1.014 | 0.323 |
| Prior potentially-avoidable ED count (270d) | 0.292 | 1.271 | +0.979 | 0.353 |
| Prior PCP count (270d) | 0.471 | 0.316 | −0.155 | −0.085 |
| Prior urgent care count (270d) | 0.190 | 0.258 | +0.068 | 0.055 |
| Prior telehealth count (270d) | 0.455 | 0.376 | −0.079 | −0.043 |
| Prior Care Management count (270d) | 0.161 | 0.245 | +0.084 | 0.073 |

The single largest, most decision-relevant fact: **the true target prevalence itself is 3.1× higher** for `transportation_barrier=1` (30.21% vs. 9.75%) in this synthetic population. Every ED-utilization-history covariate is also meaningfully higher (correlations 0.30-0.35), age is barely different (correlation 0.04), and outpatient-continuity covariates (PCP/telehealth prior use) are slightly lower. Not a causal claim — this describes the synthetic generator's constructed associations.

## 5. Conditional transportation analysis

`artifacts/phase7_hardening/transportation_conditional_analysis.csv` (22 strata). Critical finding: **the transportation effect remains dominant in every stratum tested**, not fully explained away by any single covariate:

| Conditioning | barrier=0 recall | barrier=1 recall |
|---|---|---|
| × telehealth_available=0 | 0.667 | 1.000 |
| × telehealth_available=1 | 0.244 | 0.981 |
| × clinical_burden=0 | 0.243 | 0.964 |
| × clinical_burden=3+ | 0.698 | 1.000 |
| × prior_avoidable_ed_history=0 | 0.215 | 0.972 |
| × prior_avoidable_ed_history=1 | 0.735 | 0.997 |
| × pcp_distance_band=0-5mi | 0.279 | 1.000 (n=49, small) |
| × pcp_distance_band=10mi+ | 0.579 | 0.991 |

Within-stratum recall for `barrier=1` never drops below ~0.93 (except the small n=49 cell, still 1.00) across any conditioning variable, while `barrier=0`'s recall varies widely (0.21–0.70) depending on the same covariates. **Conclusion: the disparity is not a confound fully explained by any one correlated covariate — it persists after conditioning, consistent with a direct, large model effect (Section 6) compounding with the real prevalence difference (Section 4).**

## 6. Logistic contribution analysis

`artifacts/phase7_hardening/feature_contribution_summary.csv` / `feature_contribution_overlap.json`. `transportation_barrier`'s standardized coefficient (+0.2235) is the **largest-magnitude of all 60 encoded features** (rank 1/60); `telehealth_available` is rank 2 (−0.180); `pcp_distance_miles` is rank 3 (+0.118). Per-row contribution to the log-odds ranges from −0.098 (barrier=0) to +0.508 (barrier=1) — the entire distribution shifts, not just its mean. Contribution overlap (correlation between `transportation_barrier`'s per-row contribution and other features'): `access_burden` 0.698 (by construction — partly derived from `transportation_barrier`), `prior_potentially_avoidable_ed_count_270d` 0.353, `pcp_distance_miles` 0.300, `urgent_care_distance_miles` 0.296 — moderate, not near-collinear. Coefficient magnitude is not causality; it describes the fitted linear model's behavior on this data.

## 7. Threshold interaction

`artifacts/phase7_hardening/transportation_threshold_analysis.csv` — **this section revises Phase 6's initial characterization with more precise evidence.** Phase 6 attributed the disparity partly to "threshold interaction"; Phase 7's direct measurement shows the opposite is closer to true:

| | barrier=0 | barrier=1 |
|---|---|---|
| % within ±0.01 of MODERATE | 13.12% | 4.75% |
| % within ±0.01 of HIGH | 0.73% | 7.89% |
| n barely over MODERATE (<0.02 above) | 823 (9.8%) | 114 (7.0%) |
| n far over MODERATE (≥0.10 above) | 251 (3.0%) | 832 (**51.3%**) |
| Median probability | 0.078 | 0.209 |
| p90 | 0.136 | 0.521 |

**Barrier=1 members are not clustered at the threshold edge — the large majority (51.3%) sit far above MODERATE, not barely over it.** This means the disparity is driven predominantly by the shifted score distribution itself (Sections 4/6: real prevalence difference + largest coefficient + correlated features), with threshold-edge proximity playing a comparatively minor role. `barrier=0`'s distribution, by contrast, sits mostly below MODERATE (median 0.078 < 0.106) with a meaningful "barely under" population, so a modest score shift would cross more of *that* group — an asymmetry worth noting but not evidence of a defect (thresholds are not being changed).

## 8. Telehealth disparity

`artifacts/phase7_hardening/telehealth_disparity_analysis.csv`:

| | `telehealth_available=0` | `telehealth_available=1` |
|---|---|---|
| n | 2,433 | 7,567 |
| Prevalence | 20.80% | 10.59% |
| Mean probability | 0.1863 | 0.0981 |
| Recall@MODERATE | 0.8202 | 0.4806 |
| Precision | 0.2555 | 0.2082 |
| FPR | 0.6274 | 0.2164 |
| FNR | 0.1798 | 0.5194 |
| ROC-AUC | 0.7004 | 0.6725 |
| PR-AUC | 0.4581 | 0.2461 |
| MODERATE+ rate | 66.75% | 24.44% |
| HIGH rate | 23.92% | 5.63% |

Stratified by `transportation_barrier`: recall is 0.667 (barrier=0/telehealth=0) vs. 0.244 (barrier=0/telehealth=1) vs. 1.000 (barrier=1/telehealth=0) vs. 0.981 (barrier=1/telehealth=1) — telehealth's own effect is real but **smaller than transportation's**, and the two features are negatively correlated (r=−0.125: members with a transportation barrier are less likely to also have telehealth). **Conclusion: primarily a genuine prevalence difference (2.8pp reversed direction — telehealth=0 group has the higher true risk) plus a real, smaller, partially-overlapping correlation with transportation/access factors — not a distinct, unexplained direct model effect, and not purely threshold interaction** (its own coefficient, rank 2/60, is meaningfully smaller than transportation's).

## 9. Clinical burden disparity

`artifacts/phase7_hardening/clinical_burden_analysis.csv`:

| Burden | n | Prevalence | Recall | Precision | FPR | FNR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| 0 | 3,938 | 10.74% | 0.4799 | 0.2333 | 0.1898 | 0.5201 | 0.6941 | 0.2594 |
| 1 | 3,849 | 12.24% | 0.5796 | 0.2027 | 0.3179 | 0.4204 | 0.6882 | 0.2995 |
| 2 | 1,608 | 17.23% | 0.7473 | 0.2491 | 0.4688 | 0.2527 | 0.6998 | 0.3900 |
| 3+ | 605 | 22.48% | 0.8603 | 0.2753 | 0.6567 | 0.1397 | 0.7358 | 0.5590 |

Recall rises monotonically with burden (0.48→0.86); **ROC-AUC stays essentially flat-to-slightly-improving (0.69→0.74) across bands** — discrimination quality is not degrading for high-burden members. **Conclusion: rising recall reflects both a genuine rise in true prevalence (10.7%→22.5%) and the fixed absolute MODERATE threshold interacting with a correspondingly right-shifted score distribution — both factors together, as anticipated, not one alone.** Per instruction, recall is not forced equal across burden groups.

## 10. Disparity classification

`artifacts/phase7_hardening/disparity_decisions.json`, using the explicit criteria in `classify_disparity_issue()` (Phase7 script): BLOCKER is reserved for behavior making the demo unsafe/misleading (subgroup ROC-AUC≤0.5, a safety/language-policy failure, or a disparity whose *direction* contradicts true prevalence). None of the three issues meet that bar.

| Issue | Recall Δ | Prevalence Δ | Classification |
|---|---|---|---|
| `transportation_barrier` (1 vs 0) | +0.604 | +0.205 | **INVESTIGATE** |
| `telehealth_available` (0 vs 1) | +0.340 | +0.102 | **INVESTIGATE** |
| `clinical_burden` (3+ vs 0) | +0.380 | +0.117 | **INVESTIGATE** |

All three are large, real, and directionally consistent with true prevalence (never "the model favors the wrong group"). None qualifies as EXPECTED_SYNTHETIC_SIGNAL outright (magnitude too large to wave off) nor BLOCKER (no unsafe/misleading behavior). Differences are described as differences — this document does not call the model "biased" or "fair."

## 11. Final model-change recommendation

**A. KEEP MODEL UNCHANGED.** `artifacts/phase7_hardening/model_change_recommendation.json`: no BLOCKER-level finding exists; every disparity's direction matches its subgroup's true target prevalence (the model discriminating between genuinely higher- and lower-risk synthetic groups, which is the model working as designed); the transportation_barrier effect, while large, is explained by a documented combination of a real 3.1× prevalence difference, the single-largest model coefficient, and correlated access/history features — not an unexplained anomaly, and it persists (rather than vanishes) under conditioning (Section 5), meaning it's a real property of this synthetic data/model pairing, not a confound artifact to "fix" by dropping a feature. Per the default instruction ("preserve the frozen model unless evidence is strong"), the evidence here does not clear that bar. No data regeneration performed or recommended by this decision.

## 12. Static input validation

New module `backend/agents/input_validation.py`. `age`: integer, `[0, 120]` (a generous real-world bound, not narrowed to the synthetic sample's observed 18-90, which would be an invented restriction). Distances (`pcp_distance_miles`, `urgent_care_distance_miles`): finite, `>= 0`, `<= 500` miles (generous sanity ceiling; observed synthetic range is 0.2-30mi). Every problem found is collected and raised together (`MemberDataValidationError`), not just the first.

## 13. Binary validation

`diabetes`, `copd`, `hypertension`, `chf`, `asthma`, `ckd`, `transportation_barrier`, `telehealth_available` (members) and `admitted`, `icu`, `major_procedure`, `red_flag` (ED visits): must be exactly `0` or `1` — finite, numeric, no NaN/Infinity/other value accepted. `triage_level` (ED visits): integer `1-5`, validated across **every** row, closing the Phase 6-documented gap where an invalid triage value outside every snapshot's observation window silently passed through unrejected (`test_invalid_triage_level_rejected_regardless_of_observation_window`).

## 14. Distance validation

See Section 12. Boundary-tested at 0, 500 (pass) and −0.01, 500.01 (reject); NaN/Infinity rejected explicitly.

## 15. Chronic-count consistency

Verified from actual data: `num_chronic_conditions` exactly equals `diabetes+copd+hypertension+chf+asthma+ckd` for all 10,000 synthetic members (checked directly). Chosen behavior (Step 12): if the caller supplies `num_chronic_conditions`, it is validated for consistency against the six flags and rejected if mismatched (never silently overwritten); if the caller **omits** it, it is safely **derived** from the flags rather than required as a redundant client-supplied value — `backend/main.py`'s `UC07_MEMBERS_REQUIRED` no longer lists it. This is a genuine field (fed to the model separately from the derived `clinical_burden`), so consistency, not silent trust, was the right default for the supplied case.

## 16. Safety-context contract

`backend/agents/contracts.py`: `CurrentSafetyContext` gained a `completeness` property (`ContextCompleteness.COMPLETE/PARTIAL/ABSENT`, computed from which of the 5 fields are non-`None`) and a `source` property (`ContextSource.CALLER_SUPPLIED/NOT_AVAILABLE`, computed from `.provided`). A field the caller omits stays `None` — never coerced to `0`/false — verified explicitly (`test_missing_field_is_none_not_zero`).

## 17. Context completeness

Formalizes (does not change) the Phase 6 fix. `backend/agents/safety_policy.py`'s `_determine_state()`: if any known field triggers OVERRIDE → OVERRIDE (checked first, regardless of completeness — a single known trigger field still overrides with everything else missing); else if `completeness == COMPLETE` → CLEAR; else (PARTIAL or ABSENT) → CAUTION. Verified: `safety_context_matrix.csv`, 7/7 passed.

## 18. Safety-context provenance

`SafetyDecision` gained `context_completeness` and `context_source` audit fields (defaults describe "no context involved," so existing test construction without these kwargs remains valid), populated by `safety_policy.decide()` and serialized in `POST /uc07/decide`'s JSON response (`orchestrator.decision_to_dict`). Only `CALLER_SUPPLIED`/`NOT_AVAILABLE` are ever produced; `SYSTEM_DERIVED` is defined but never asserted (Section 15's instruction: never claim verification that doesn't exist).

## 19. Invalid safety inputs

New `backend/agents/safety_context_schema.py` (Pydantic v2, `extra="forbid"`): rejects `triage_level` outside `{1..5}`, binary fields outside `{0,1}`, non-finite (NaN/±Infinity) values, wrong types, and unrecognized extra keys (e.g. a `triage_level` typo) — all via 422 with a structured error list, never coerced to a safe-looking default. JSON booleans (`true`/`false`) map cleanly to `1`/`0` (a reasonable, unambiguous representation, not "garbage").

## 20. API validation

`artifacts/phase7_hardening/api_validation_results.csv`, 12 cases, **100% passed**: valid complete / no-context / partial-context / override-context requests all succeed (200); invalid age, negative distance, invalid binary flags, invalid safety-context triage, inconsistent chronic count, missing required field all correctly 422; unknown member 404; extra CSV columns tolerated (200, documented policy — column-level extras are non-safety-relevant, unlike JSON-schema extras which are rejected). No response leaked a traceback.

## 21. Feature-schema validation

Unchanged from Phase 5: `RiskDetectionAgent.validate_feature_schema()` checks required columns are present before scoring; `load_model_bundle()` cross-validates the artifact's `feature_columns` against metadata at construction. Phase 7 adds validation one layer earlier (raw CSV values, Section 12-15), so malformed data is rejected before it ever reaches feature reconstruction, not just before scoring.

## 22. Failure-mode hardening

`artifacts/phase7_hardening/failure_mode_results.csv`, 9 modes, **100% conservative** (fail loudly/clearly, never fabricate a recommendation): member/safety-context data invalid → 422; feature generation incomplete → `KeyError` propagates; model metadata incompatible/missing/threshold-invalid → `ModelIncompatibleError` → 503; Navigation Agent given malformed input → defensive conservative defaults (never raises, never fabricates a favorable destination); Safety Agent → pure deterministic logic, language policy always applied; Risk Agent exception → propagates uncaught to a clean error response.

## 23. E2E regression

Full suite re-run after every change (Sections 12-19): **505/505 passing** (389 carried forward unchanged + 116 new). Representative LOW/MODERATE/HIGH tiers, all 5 navigation destinations, and CLEAR/CAUTION/OVERRIDE states all re-exercised via the unchanged Phase 5/6 test files plus this phase's own API matrix — no legitimate flow broken by the new validation layer.

## 24. Bugs found

1. (Carried from Phase 6, already fixed there) Safety-context partial-completeness gap — re-verified fixed and now formalized as named states.
2. An invalid `triage_level` outside every snapshot's observation window previously passed through the API unrejected (Phase 6 Section 28's documented gap) — now closed (Section 13).
3. `num_chronic_conditions` had no consistency check against its six component flags, despite being fed to the model as a separate feature from the derived `clinical_burden` — a caller could silently supply an internally-inconsistent value.

## 25. Bugs fixed

Item 2 and 3 above, via `backend/agents/input_validation.py` (new), wired into `backend/main.py`'s `uc07_decide_endpoint`. No production code outside `backend/agents/input_validation.py`, `backend/agents/safety_context_schema.py`, `backend/agents/contracts.py`, `backend/agents/safety_policy.py`, `backend/agents/orchestrator.py` (one-line serialization addition), and `backend/main.py` (wiring + required-column list change) was touched. `backend/pit/`, `backend/agents/care_navigation.py`, `backend/agents/risk_detection.py`, and the model artifacts are unmodified.

## 26. Remaining limitations

- `transportation_barrier`, `telehealth_available`, and `clinical_burden` disparities remain INVESTIGATE-classified — explained, not eliminated; Phase 8+ should decide whether a real-data replacement or a fairness-aware modeling approach is warranted before any real clinical use (explicitly out of scope for a synthetic demo).
- `current_safety_context` is still entirely caller-supplied; Phase 7 makes *incomplete* context safer and its provenance auditable, but does not add any verification capability (`SYSTEM_DERIVED` remains unimplemented by design).
- Distance/age bounds (500mi / 0-120) are engineering sanity limits, not clinically validated constraints.
- Runtime model selection (`uc07-risk-v1` vs. `uc07-risk-synthetic-v1`) remains unimplemented.

## 27. Readiness for frontend

Backend/API contract is hardened: structured validation errors, no silent coercion, formalized safety-context contract, all prior safety invariants intact. Frontend integration (Phase 8) can proceed against this hardened `/uc07/decide` contract; the new `safety.context_completeness` / `safety.context_source` response fields are additive (no existing field removed or renamed) and should be surfaced audibly, not hidden, in any UI built against this API.

## 28. File-by-file changes

**Created:**
- `backend/agents/input_validation.py`
- `backend/agents/safety_context_schema.py`
- `backend/validation/phase7_disparity_analysis.py`
- `backend/validation/phase7_api_artifacts.py`
- `backend/tests/test_phase7_hardening.py` (116 new tests)
- `docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md` (this file)
- `artifacts/phase7_hardening/` (16 report files)

**Modified:**
- `backend/agents/contracts.py` — added `ContextCompleteness`, `ContextSource` enums; `CurrentSafetyContext.completeness`/`.source` properties; `SafetyDecision.context_completeness`/`.context_source` fields (both with safe defaults, no existing construction broken)
- `backend/agents/safety_policy.py` — `_determine_state()` now uses `context.completeness` (formalized, same outcome as the Phase 6 fix); `decide()` populates the new audit fields
- `backend/agents/orchestrator.py` — `decision_to_dict()` serializes the two new safety fields
- `backend/main.py` — imports and wires `input_validation.py`/`safety_context_schema.py`; `UC07_MEMBERS_REQUIRED` no longer requires `num_chronic_conditions`; `_parse_current_safety_context()` now delegates schema validation to Pydantic

**Not modified:** the model artifacts/metadata (content), all raw and derived datasets, `backend/pit/*`, `backend/agents/care_navigation.py`, `backend/agents/risk_detection.py`, `backend/modeling/*`.
