# Phase 6 — Safety, Fairness, Robustness & End-to-End Multi-Agent Validation

**Date:** 2026-08-16
**Status:** Complete. Model frozen (`uc07-risk-synthetic-v1`, unchanged). One safety-hardening code fix applied and documented (Section 37-38).

---

## 1. Phase objective

Answer: *"Does the complete UC07 multi-agent system behave safely, consistently, robustly, and reasonably across different member groups and edge cases?"* This is a validation phase, not model development — the frozen model was never retrained, tuned, or recalibrated. Six areas: safety validation, subgroup/fairness assessment, transportation-barrier investigation, robustness/edge-case testing, navigation-policy validation, end-to-end system validation.

## 2. Frozen model identity

`uc07-risk-synthetic-v1`, Logistic Regression, `dataset_id=synthetic_uc07_v1`, verified from `backend/models/uc07_risk_synthetic_v1_model_metadata.json`: TEST ROC-AUC 0.704752, PR-AUC 0.336636, Brier 0.102334, Accuracy@0.50 0.876, HIGH-tier lift 2.87× — all matched the brief's expected values exactly. Phase 4E (`docs/04E_TREE_MODEL_OPTIMIZATION.md`) tested Logistic Regression vs. Random Forest vs. XGBoost vs. HistGradientBoosting and retained Logistic Regression; not repeated here.

## 3. Frozen thresholds

`moderate_threshold=0.105986`, `high_threshold=0.213252`, loaded from the model artifact/metadata by `RiskDetectionAgent`, confirmed identical to the brief's expected values. `test_invariant_7_no_duplicate_hardcoded_threshold_source` (new) asserts `risk_detection.py` itself contains no numeric threshold literal.

## 4. Multi-agent architecture (verified from code)

Risk Detection Agent (`backend/agents/risk_detection.py`) → Care Navigation Agent (`care_navigation.py`) → Safety & Policy Agent (`safety_policy.py`, final, non-bypassable) → `FinalUC07Decision`, enforced by `orchestrator.py`. Confirmed: Navigation Agent's `decide()` takes no `CurrentSafetyContext` parameter at all (structurally cannot see current safety information); Risk Agent never imports navigation/safety modules; Safety Agent's `decide()` is called exactly once, always last, on every code path (single-member and batch).

## 5. Validation methodology

All validation reuses the existing, unmodified agents/orchestrator/model against the frozen synthetic data. New code lives in `backend/validation/phase6_validation.py` (produces every artifact under `artifacts/phase6_validation/`) and `backend/tests/test_phase6_safety_invariants.py` (10 named invariants + 4 new failure-mode tests). SHA-256 of every raw dataset, snapshot, and model `.joblib` was captured before any work and re-verified identical afterward (Section 33-34).

## 6. Safety principles

The system must never discourage, delay, block, deny, or gatekeep genuine emergency care. This is enforced by construction, not convention: OVERRIDE unconditionally suppresses the navigation destination (`FinalNavigationView(destination=None, ...)`), missing current-safety-context is conservative (never CLEAR), and a centralized `PROHIBITED_PHRASES` policy is applied to every navigation explanation and every safety message in every state.

## 7. Safety override matrix

`artifacts/phase6_validation/safety_override_matrix.csv` — **68 rows, 100% passed**. Each of the 6 individual triggers (`red_flag=1`, `icu=1`, `admitted=1`, `major_procedure=1`, `triage_level=1`, `triage_level=2`) tested against all 5 reachable navigation destinations (30 rows) and against LOW/MODERATE/HIGH risk tiers on a fixed feature row (18 rows); the 5 named combination scenarios from the brief (`red_flag + HIGH risk`, `ICU + telehealth available`, `admitted + strong PCP access`, `major procedure + LOW risk`, `triage 1 + care-management eligibility`); and all C(6,2)=15 two-trigger combinations. Every single row: `safety_state=OVERRIDE`, `final_destination=None`, regardless of preliminary destination or risk tier.

## 8. Missing-context behavior (includes a safety-hardening fix — see Section 37)

`artifacts/phase6_validation/missing_safety_context_tests.csv` — **18 rows, 100% passed** after the fix in Section 37. Fully-missing context → CAUTION. Each of the 5 fields supplied alone (safe value, others unknown) → CAUTION. Each "all-safe-except-one-missing" leave-one-out combination (5 rows) → CAUTION. Fully-known, all-safe context → CLEAR. `red_flag=1` with everything else missing → OVERRIDE (matches the brief's explicit example). Every other single-trigger-with-rest-missing case → OVERRIDE. A mixed case (`red_flag=0` known-safe + `triage_level=1` known-override) → OVERRIDE. A mixed case (3 known-safe + 2 missing) → CAUTION.

## 9. Safety precedence

Proven three ways: (1) `care_navigation.decide()`'s signature has no `CurrentSafetyContext`-shaped parameter at all (`test_invariant_3_navigation_decide_has_no_safety_context_parameter`); (2) an AST walk of every function in `orchestrator.py` confirms every `care_navigation.decide()` call precedes any `safety_policy.decide()` call within the same function, so no reversed "Safety → Navigation" path exists anywhere in the module (`test_invariant_3_no_reversed_call_order_exists_anywhere_in_orchestrator`); (3) the orchestrator's return type is always `FinalNavigationView` (the safety-reviewed view), never the raw `NavigationDecision` (`test_invariant_3_final_decision_never_carries_raw_navigation_decision_type`). Also carried forward unchanged from Phase 5: `test_orchestrator.py`'s source-order and non-bypass tests.

## 10. Prohibited-language validation

`artifacts/phase6_validation/prohibited_language_scan.json` — **40,055 checks, 0 violations**. Scanned: every possible (destination × reason-code) template combination the Navigation Agent's own phrase dictionaries can generate (55 combinations, exhaustive — not sample-dependent), all 4 static safety/disclaimer message constants, and the full synthetic population (10,000 members) scored at their real model-derived tier, with every generated `NavigationDecision.explanation` re-checked after passing through the Safety Agent under CLEAR/CAUTION/OVERRIDE contexts (4 checks per member). The existing `PROHIBITED_PHRASES` policy (`safety_policy.py`) was used as-is and not weakened.

## 11. Transportation-barrier investigation

`artifacts/phase6_validation/transportation_barrier_analysis.json` (full metrics) + `subgroup_metrics.csv`. Using the frozen model on the sealed TEST snapshot:

| Metric (@MODERATE) | `transportation_barrier=0` | `transportation_barrier=1` |
|---|---|---|
| n | 8,378 | 1,622 |
| Prevalence | 9.75% | 30.21% |
| Mean probability | 0.0905 | 0.2699 |
| Median probability | 0.0784 | 0.2086 |
| LOW % | 77.30% | 3.14% |
| MODERATE % | 15.98% | 51.62% |
| HIGH % | 2.66% | 48.40% |
| Accuracy | 0.7507 | 0.3274 |
| Balanced Accuracy | 0.5878 | 0.5152 |
| Precision | 0.1656 | 0.3087 |
| Recall | **0.3856** | **0.9898** |
| Specificity | 0.7901 | 0.0406 |
| F1 | 0.2317 | 0.4706 |
| ROC-AUC | 0.6322 | 0.6765 |
| PR-AUC | 0.1794 | 0.5242 |
| Brier | 0.0855 | 0.1893 |
| FPR | 0.2099 | 0.9594 |
| FNR | 0.6144 | 0.0102 |

`transportation_barrier=1`'s MODERATE-threshold recall is **0.989796** — identical to the Phase 4D/4E finding, confirmed unchanged on the same frozen model (no model modification could have changed it; this is a re-confirmation, not a new result).

**Coefficient/contribution:** `transportation_barrier`'s standardized logistic-regression coefficient is **+0.2235**, the **largest-magnitude coefficient of all 60 encoded features** (rank 1 of 60) — larger than `telehealth_available` (−0.180), `pcp_distance_miles` (+0.118), and every ED-utilization-history feature.

**Correlation:** `transportation_barrier` correlates most strongly with `access_burden` (r=0.698, by construction — `access_burden` is partly derived from `transportation_barrier`), then with prior-ED-utilization features (`prior_potentially_avoidable_ed_count_270d` r=0.353, `prior_ed_count_270d` r=0.323) and distance features (`pcp_distance_miles` r=0.300, `urgent_care_distance_miles` r=0.296).

**Determination (not causal):** the evidence points to **a combination of B (threshold interaction) and C (model coefficient magnitude), with D (correlation with other access features) as a contributor** — not primarily A (synthetic generative relationship alone). Reasoning: (1) the coefficient is the single largest in the model, so a member with `transportation_barrier=1` receives a large, fixed upward push to their log-odds regardless of any other feature; (2) `transportation_barrier=1`'s group prevalence (30.2%) sits well above the MODERATE threshold's population-wide selection point, so a large fraction of the group crosses MODERATE almost independent of their other features — a direct threshold-interaction effect; (3) the correlation with other access/utilization features (D) compounds but does not dominate the effect (correlations are moderate, 0.30-0.35, not near-collinear). We explicitly do not claim this reflects a real causal clinical relationship — see Section 31.

## 12. Transportation counterfactual sensitivity

`artifacts/phase6_validation/transportation_counterfactual_analysis.csv` — **model sensitivity / counterfactual feature perturbation, NOT clinical causality**, computed on in-memory copies of the TEST features only (`test_snapshot.csv` on disk is byte-for-byte unchanged, verified in Section 34).

| Direction | n affected | Mean Δp | Median Δp | Max |Δp| | LOW→MOD | MOD→HIGH | HIGH→lower | MOD→LOW |
|---|---|---|---|---|---|---|---|---|
| 0→1 | 8,378 | +0.0612 | +0.0565 | 0.1504 | 4,669 | 790 | 0 | 0 |
| 1→0 | 1,622 | −0.0877 | −0.0819 | 0.1504 | 0 | 0 | 451 | 530 |

Flipping `transportation_barrier` alone moves the mean predicted probability by ~0.06-0.09 and crosses roughly 56% of the `0→1` group at least one tier up (4,669+790 of 8,378) — consistent with, and quantitatively explaining, the large coefficient found in Section 11: this is the single largest average one-feature perturbation effect of any tested in this phase (compare Section 13, where the next-largest, `telehealth_available` flip, moves probability by ~0.02 on average).

## 13. Other access-feature sensitivity

`artifacts/phase6_validation/access_feature_sensitivity.csv`:

| Perturbation | Mean Δp | Max |Δp| | Tier crossings |
|---|---|---|---|
| `telehealth_available` flip 0↔1 | 0.0194 | 0.1048 | 38.60% |
| `pcp_distance_miles` +1mi (tiny) | 0.0024 | 0.0063 | 1.85% |
| `pcp_distance_miles` +5mi | 0.0123 | 0.0314 | 9.63% |
| `pcp_distance_miles` +10mi (large) | 0.0256 | 0.0627 | 21.45% |
| `urgent_care_distance_miles` +1mi (tiny) | 0.0013 | 0.0035 | 1.08% |
| `urgent_care_distance_miles` +5mi | 0.0067 | 0.0174 | 5.18% |
| `urgent_care_distance_miles` +10mi (large) | 0.0137 | 0.0349 | 10.66% |

No cliff-edge sensitivity: effect size scales roughly monotonically and proportionally with perturbation magnitude for both distance features (a 1-mile change moves ~1.9% of the population across a tier; a 10-mile change moves ~21%) — a tiny, plausible data-entry-level change (1 mile) does not produce an outsized tier jump relative to a large change (10 miles).

## 14. Age subgroup results

4 bands (18-34, 35-49, 50-64, 65+), all n≥1,440 (sufficient). ROC-AUC range 0.689-0.715 (no material spread), recall@MODERATE range 0.587-0.627 (small spread, all pairwise deltas classified NO MATERIAL SIGNAL DETECTED, largest |Δ|=0.040). See `subgroup_metrics.csv` / `subgroup_disparity_summary.csv`.

## 15. Gender subgroup results

`gender_F` (n=5,084) vs. `gender_M` (n=4,916): recall@MODERATE 0.6228 vs. 0.6006 (Δ=+0.022, NO MATERIAL SIGNAL DETECTED); ROC-AUC 0.713 vs. 0.696; PR-AUC 0.352 vs. 0.321. No material disparity detected on this dimension.

## 16. Clinical-burden subgroup results

4 bands (0, 1, 2, 3+), all n≥605 (sufficient, smallest group flagged as the least-stable of the "sufficient" set but still above the n≥100 floor). Recall@MODERATE rises monotonically with burden: 0.480 → 0.580 → 0.747 → 0.860. `clinical_burden_0` vs. `clinical_burden_3_plus`: Δ=−0.380, **INVESTIGATE**. This tracks the intended design — the model is more sensitive for members with more chronic conditions — but the magnitude is large enough to record explicitly rather than wave off.

## 17. Transportation subgroup results

See Section 11 in full. Summary classification: recall@MODERATE Δ=+0.604 (`transportation_barrier_1` vs. `_0`), **INVESTIGATE** — the largest disparity found in this phase on any dimension.

## 18. Telehealth subgroup results

`telehealth_available_1` (n=7,567) vs. `_0` (n=2,433): recall@MODERATE 0.4806 vs. 0.8202 (Δ=−0.340, **INVESTIGATE** — members *without* telehealth access have *higher* recall at MODERATE, the inverse direction of the transportation-barrier finding). `telehealth_available` and `transportation_barrier` are correlated in the synthetic generator (members with worse access more often lack telehealth too), so these two findings likely share a common root rather than being independent signals — see Section 19.

## 19. Fairness interpretation

Per the brief's explicit instruction, this phase does **not** conclude "the model is fair" or "the model is unbiased" anywhere. Findings are classified only as NO MATERIAL SIGNAL DETECTED / MONITOR / INVESTIGATE, using an explicit, documented rule: `|Δ| ≥ 0.30` → INVESTIGATE, `0.10 ≤ |Δ| < 0.30` → MONITOR, `|Δ| < 0.10` → NO MATERIAL SIGNAL DETECTED, applied to recall/FPR/FNR/selection-rate/Brier differences (`classify_disparity()` in `phase6_validation.py`). Full table: `artifacts/phase6_validation/subgroup_disparity_summary.csv` (75 pairwise comparisons: 44 NO MATERIAL SIGNAL DETECTED, 17 MONITOR, 14 INVESTIGATE — all 14 INVESTIGATE rows come from the `transportation_barrier`, `telehealth_available`, and `clinical_burden` dimensions; every age/gender comparison is NO MATERIAL SIGNAL DETECTED). This is a descriptive triage rule, not a statistical-significance test, and is not used to modify the model.

## 20. Small-group limitations

Minimum subgroup size: **n < 100 → descriptive only / unstable**, marked with `sufficient_sample=False` and an explicit note in `subgroup_metrics.csv` rather than hidden. Every subgroup actually evaluated in this phase had n≥605 (well above the floor); the `pcp_distance_band`/age/burden/access dimensions were deliberately chosen at a granularity that keeps every cell well-populated at n=10,000 TEST rows. No subgroup was excluded from reporting.

## 21. Navigation policy validation

`artifacts/phase6_validation/navigation_policy_matrix.csv` — 24 scenarios. All 5 destinations confirmed reachable with a documented `why_selected` rationale for each; every observed destination matches its documented expectation after two scenario-construction corrections (Section 39).

## 22. Care Management validation

Confirmed: HIGH risk alone (no complexity/access/history) → **PRIMARY_CARE, not CARE_MANAGEMENT**. HIGH + chronic complexity / transportation barrier / prior CM engagement → CARE_MANAGEMENT. MODERATE + repeated utilization + access barrier → CARE_MANAGEMENT. LOW + complexity only (no elevated risk, no repeated utilization) → **NO_PROACTIVE_NAVIGATION, not CARE_MANAGEMENT** — proving the actual repository rule symmetrically: CARE_MANAGEMENT requires **both** (elevated risk OR repeated utilization) **and** a complexity/access/continuity signal; neither side alone is sufficient. Enforced going forward by `INVARIANT 5` (`test_invariant_5_high_risk_alone_never_triggers_care_management`, `test_invariant_5_complexity_alone_without_risk_or_utilization_never_triggers_care_management`).

## 23. Telehealth validation

Available + transportation barrier → TELEHEALTH. Unavailable + transportation barrier → falls through to NO_PROACTIVE_NAVIGATION (no other opportunity signal present). Available + no access barrier (both distances ≤10mi) → NOT selected, falls through. HIGH risk + telehealth available + transportation barrier → **CARE_MANAGEMENT, not TELEHEALTH** (CARE_MANAGEMENT is checked first in priority order and its complexity signal is also satisfied). Emergency override + telehealth available (conflict scenario, Section 27) → destination suppressed to `None`; OVERRIDE defeats telehealth every time, confirmed programmatically.

## 24. Urgent Care validation

Urgent meaningfully closer than PCP, **with PCP itself inside the 10-mile access-barrier threshold** → URGENT_CARE. Urgent farther than PCP → falls to PRIMARY_CARE. Both nearby → URGENT_CARE. **Both distant (PCP itself >10mi)** → **CARE_MANAGEMENT, not URGENT_CARE** — a genuine, documented property of the deterministic rule tree (Section 39): once `pcp_distance_miles` itself exceeds 10 miles, CARE_MANAGEMENT's `LIMITED_PCP_ACCESS` complexity signal is satisfied and, checked first in priority order, pre-empts a plain URGENT_CARE suggestion even when urgent care is closer. URGENT_CARE is therefore only reachable when PCP access itself stays within the access-barrier threshold. Missing distances default to 99.0 (far), never a falsely favorable value; with no other opportunity signal this correctly falls to NO_PROACTIVE_NAVIGATION rather than fabricating a recommendation. Safety overrides win in every case tested (Section 7/27).

## 25. PCP validation

Good access + recent engagement → PRIMARY_CARE. No prior engagement (no continuity opportunity) → NOT PRIMARY_CARE, falls to NO_PROACTIVE_NAVIGATION. Telehealth unavailable + urgent care not advantageous + MODERATE risk + close access + continuity → PRIMARY_CARE, consistent with policy.

## 26. No-proactive-navigation validation

Confirmed genuinely reachable: LOW risk, no history, distant access, no complexity → NO_PROACTIVE_NAVIGATION. The system does not force every member into an intervention — of the full 10,000-member population run (Section 32), **3,060 members (30.6%)** received NO_PROACTIVE_NAVIGATION.

## 27. Conflict/adversarial scenarios

`artifacts/phase6_validation/conflict_scenarios.csv` — all 7 scenarios from the brief produced deterministic, single-valued outcomes (no exceptions, no ambiguity). Notably: "urgent care closer but emergency safety trigger present" → preliminary destination CARE_MANAGEMENT, but `safety_state=OVERRIDE` and `final_destination=None` — safety wins even when the pre-safety navigation call already looked reasonable.

## 28. Input validation

`artifacts/phase6_validation/input_validation_results.csv` — 16 cases via `TestClient` against `/uc07/decide`, **100% handled with status<500** (no unhandled crashes, no raw traceback leaked, every response JSON-parseable). Findings by category:

- **Correctly rejected (4xx):** unrecognized `triage_level=9` on an in-observation-window ED visit (422, via `classify_ed_encounters`'s existing `ValueError`→422 path), missing required column (422), malformed/out-of-range `current_safety_context` fields (422, pre-existing Phase 5 validation), unknown `member_id` (404), empty upload (400).
- **Accepted without rejection (200):** out-of-range static member-level fields — negative `pcp_distance_miles`, extreme `urgent_care_distance_miles`, `transportation_barrier=7`, `NaN`/negative/extreme `age`, an extra unexpected column. These flow through to feature scoring without a value-range check (only column *presence* is validated for the members/ED/care CSVs). **This is a documented, non-safety-critical robustness gap, not fixed in this phase**: it never touches the `CurrentSafetyContext`-gated OVERRIDE path (which *is* fully validated), never crashes, and never produces an unsafe *category* of output (worst case is a model prediction extrapolated from an implausible input value) — see Section 40.

## 29. Probability/tier invariants

`artifacts/phase6_validation/probability_tier_invariants.json`, all 10,000 TEST rows: `probability_bounds_ok=true`, `low/moderate/high_tier_invariant_ok=true` (every tier assignment agrees exactly with the frozen thresholds), `thresholds_loaded_from_metadata_match=true`. `INVARIANT 7` additionally confirms no threshold value is hard-coded a second time anywhere in `risk_detection.py`.

## 30. Explanation validation

`artifacts/phase6_validation/explanation_validation.json`, 200 members checked: max contributing-factor count observed = 3 (the documented `max_factors=3` cap, respected in every case), **zero** causal-wording violations ("will cause", "guaranteed to", etc. — the actual templates use "contributed to elevated/lower risk"), **zero** target/identifier/leakage-token violations. Factor sign/direction correspondence with actual model behavior was additionally confirmed indirectly via Section 12/13's sensitivity analysis: `transportation_barrier`'s positive coefficient (Section 11) is consistent with the positive mean probability shift observed when perturbing it 0→1.

## 31. Synthetic-data disclosure

`artifacts/phase6_validation/synthetic_disclosure_check.json`: `synthetic_flag_true=true`, disclaimer present and mentions both "synthetic" and "clinically validated" — semantically equivalent to the brief's required concept ("Synthetic-data demonstration model — not clinically validated"), exposed via `/model-info` and every `RiskAssessment.synthetic_model`/`dataset_id` field. No API/metadata surface implies real clinical validation anywhere audited.

## 32. End-to-end population validation

`artifacts/phase6_validation/population_decision_summary.json`, full synthetic population (n=10,000) at the TEST index date (2026-04-03):

- **Risk tiers:** LOW 6,527 / MODERATE 2,465 / HIGH 1,008 — identical to the frozen model's own TEST risk-tier report (a strong internal-consistency check: the live orchestrator's point-in-time feature reconstruction reproduces the exact same tier assignments as the frozen `test_snapshot.csv`).
- **Navigation destinations:** PRIMARY_CARE 2,187 / URGENT_CARE 403 / TELEHEALTH 1,361 / CARE_MANAGEMENT 2,989 / NO_PROACTIVE_NAVIGATION 3,060 / suppressed-by-OVERRIDE 0.
- **Safety states:** CAUTION 10,000 / CLEAR 0 / OVERRIDE 0 — expected and intentional: the static synthetic snapshot has no `current_safety_context` field, and per the brief's explicit instruction this was never fabricated. Every decision correctly and conservatively resolves to CAUTION. OVERRIDE/CLEAR are validated exhaustively via the scenario-based matrices in Sections 7-8 instead.

## 33. API validation

`GET /health`, `GET /model-info`, `POST /uc07/decide` all exercised via `TestClient` (existing `test_uc07_api.py`, 16 tests, plus this phase's `input_validation_results.csv`, 16 more cases). Confirmed: successful response schema (probability, tier, navigation destination, safety state, explanation, model identity all present), invalid-request handling (422/404/400 as appropriate), unknown-member 404, synthetic disclosure surfaced via `/model-info`, and that no legacy endpoint (`/predict`, `/predict-json`, `/explain-member`) became authoritative for UC07 — they remain wired to the pre-Phase-2 `ed_risk_model.pkl` exactly as before, unmodified.

## 34. Cross-agent consistency

Verified across all 10,000 population-run decisions: `model_version`, `dataset_id`, `synthetic_model`, `moderate_threshold`, `high_threshold` each take **exactly one distinct value** across the entire population (`cross_agent_consistency.all_consistent=true` in `population_decision_summary.json`); 0 probability-bounds violations; 0 cases of an OVERRIDE state with a non-null destination.

## 35. Legacy isolation

Confirmed via source-grep (`INVARIANT 9`, `backend/tests/test_legacy_isolation.py` unchanged): none of `orchestrator.py`, `risk_detection.py`, `care_navigation.py`, `safety_policy.py`, `contracts.py` reference `ed_risk_model.pkl`, `predict.py`, or `feature_engineering.py`; `/uc07/decide` calls only `orchestrator.decide_for_member`/`decide_for_all_members`. The legacy `frequent_ED_user` formula is never computed inside `backend/pit/`. `ed_risk_model.pkl` cannot become the authoritative UC07 path.

## 36. Determinism

`artifacts/phase6_validation/determinism_check.json`: 25 members × 5 repeated identical calls each = 125 calls, **0 mismatches**. `INVARIANT 10` extends this as a standing regression test.

## 37. Failure modes

| Failure | Behavior | Evidence |
|---|---|---|
| Model artifact missing | `ModelIncompatibleError` raised at construction, never silently served | `test_risk_detection_agent.py` (Phase 5, unchanged) |
| Metadata missing | `ModelIncompatibleError` raised | `test_risk_detection_agent.py` (Phase 5, unchanged) |
| Artifact/metadata mismatch (feature list, version, thresholds) | `ModelIncompatibleError` raised | `test_risk_detection_agent.py` (Phase 5, unchanged); re-asserted by this phase's `test_failure_mode_model_metadata_mismatch_already_fails_loudly` |
| Invalid threshold ordering | `ModelIncompatibleError` raised | `test_risk_detection_agent.py` (Phase 5, unchanged) |
| PIT feature generation fails (e.g. a required raw column missing) | `KeyError` raised, propagates, never silently scores with fabricated data | New: `test_failure_mode_pit_feature_generation_raises_not_swallowed` |
| Risk Agent raises mid-orchestration | Propagates uncaught through `orchestrator.py` to the API layer's `try/except`, which returns a clean error status — never fabricates a decision | New: `test_failure_mode_risk_agent_exception_propagates_through_orchestrator` |
| Navigation Agent given a malformed/empty feature row | Never raises (defensive `_get()` helper with conservative defaults — e.g. missing distance defaults to 99.0 = "far", never a falsely favorable close value); still passes the language policy | New: `test_failure_mode_navigation_agent_uses_conservative_defaults_for_missing_fields` |
| Safety Agent given a well-typed but adversarial `NavigationDecision` | Pure deterministic logic over dataclass fields; blocked language replaced, never passed through, in every state | Phase 5 `test_safety_policy_agent.py` (unchanged) |

No failure path in this table silently invents a recommendation when required safety information or a system component is unavailable.

## 38. Bugs found

1. **Safety-context completeness gap (safety-relevant, fixed — Section 39).** The Safety & Policy Agent's CLEAR determination only required that *some* current-safety field be supplied, not *all five*. A caller supplying only `triage_level=4` (with `red_flag`/`icu`/`admitted`/`major_procedure` left unknown) received CLEAR — the same verdict as a caller who supplied a fully-known, all-safe context. This directly conflicted with this phase's explicit conservative-default requirement.
2. Three of this phase's own `navigation_policy_matrix.csv` test-scenario feature rows were mis-constructed (not a system bug — a validation-script bug, fixed during authoring, Section 39) because they didn't account for `pcp_distance_miles > 10` / `urgent_care_distance_miles > 10` also feeding CARE_MANAGEMENT's complexity signal.

## 39. Bugs fixed

**Fix 1 — `backend/agents/safety_policy.py`, `_determine_state()`:** CLEAR now requires all five current-safety fields (`red_flag`, `icu`, `admitted`, `major_procedure`, `triage_level`) to be explicitly known (non-`None`); any partial set that does not already trigger OVERRIDE now resolves to CAUTION instead of CLEAR. Strictly more conservative — no OVERRIDE case was weakened (the override check still runs first and unconditionally), only some previously-CLEAR partial-context cases now correctly resolve to CAUTION. Two existing Phase 5 tests were updated: `test_clear_partial_context_still_clear_if_no_override_signal` → renamed `test_partial_context_with_no_override_signal_is_caution_not_clear`, assertion changed from CLEAR to CAUTION (`backend/tests/test_safety_policy_agent.py`); `test_triage_3_alone_does_not_trigger_override` required no change (its scenario already supplies all five fields). All other existing tests referencing `CurrentSafetyContext` were audited and either already supply a complete context or supply only an OVERRIDE-triggering field (unaffected by the completeness check, since override is checked first). Full regression suite re-run after the fix: 333/333 passing (no other regressions). This decision was confirmed with the user before implementation given it changes previously-deliberate, tested Phase 5 safety behavior.

**Fix 2 — this phase's own `navigation_policy_matrix()` scenario construction** (Section 38, item 2): corrected the three affected feature rows and the "UC: both distant" scenario's documented expectation to reflect the system's actual (correct) CARE_MANAGEMENT-precedence behavior (Section 24) rather than an incorrect assumption. No production code was touched for this fix.

## 40. Remaining risks

- `transportation_barrier=1`'s near-total recall (0.9898) at MODERATE remains **INVESTIGATE**-classified and unresolved — carried forward from Phase 4D/4E, now with a fuller root-cause picture (Section 11) but still not something this validation-only phase changes.
- `telehealth_available` and `clinical_burden` also produced INVESTIGATE-level recall disparities (Sections 16, 18); likely correlated with the transportation-barrier finding rather than independent.
- Static member-level CSV fields (age, distances, `transportation_barrier`) have no value-range validation at the API layer (Section 28) — low severity (never touches the OVERRIDE path, never crashes), but a candidate for Phase 7 hardening.
- `current_safety_context` remains entirely caller-supplied and unverified by the system itself (unchanged from Phase 5) — this phase's Section 8 hardening makes *incomplete* context safer, but a caller could still supply a fully-populated, all-safe context that does not reflect reality.
- Runtime model selection between `uc07-risk-v1` and `uc07-risk-synthetic-v1` remains unimplemented (unchanged from Phase 5).

## 41. Production-readiness assessment

Not clinically production-ready — synthetic demonstration data only. See the terminal summary for the itemized PASS/CONDITIONAL/FAIL classification across ML demo, safety architecture, and multi-agent integration readiness, plus explicit NO/NOT-ESTABLISHED answers for frontend, Docker, Azure, and clinical deployment.

## 42. File-by-file changes

**Created:**
- `backend/validation/phase6_validation.py`
- `backend/tests/test_phase6_safety_invariants.py` (56 new tests)
- `docs/06_SAFETY_FAIRNESS_ROBUSTNESS_VALIDATION.md` (this file)
- `artifacts/phase6_validation/` (safety_override_matrix.csv, missing_safety_context_tests.csv, prohibited_language_scan.json, transportation_barrier_analysis.json, transportation_counterfactual_analysis.csv, access_feature_sensitivity.csv, subgroup_metrics.csv, subgroup_disparity_summary.csv, navigation_policy_matrix.csv, conflict_scenarios.csv, input_validation_results.csv, probability_tier_invariants.json, explanation_validation.json, synthetic_disclosure_check.json, population_decision_summary.json, determinism_check.json, immutability_check.json, pre_work_hashes.json, phase6_validation_summary.json)

**Modified:**
- `backend/agents/safety_policy.py` — `_determine_state()` hardened (Section 39, Fix 1)
- `backend/tests/test_safety_policy_agent.py` — one test updated to match the hardened behavior

**Not modified:** the model artifacts/metadata, all raw and derived datasets, `care_navigation.py`, `orchestrator.py`, `contracts.py`, `risk_detection.py`, `main.py`, `backend/pit/*`, `backend/modeling/*`.
