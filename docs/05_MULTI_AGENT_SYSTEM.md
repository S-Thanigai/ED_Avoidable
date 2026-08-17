# UC07 — Multi-Agent Decision System (Phase 5)

**Implementation date:** 2026-08-16
**Phase:** 5 — Multi-Agent UC07 Decision System (backend/agent architecture; no frontend redesign, no Docker/Azure)
**Builds on:** `docs/01_PROJECT_BASELINE.md` … `docs/04D_SYNTHETIC_MODEL_DEVELOPMENT.md`, `docs/DECISION_LOG.md`

> **SYNTHETIC DATA MODEL — DEMONSTRATION ONLY.** The Phase 5 demonstration
> system uses `uc07-risk-synthetic-v1` (TEST ROC-AUC 0.7048, PR-AUC
> 0.3366, prevalence 13.07%, HIGH-tier lift 2.87×, thresholds
> `MODERATE=0.105986` / `HIGH=0.213252`). It must never be presented as
> clinically validated. `uc07-risk-v1` (original data) remains preserved,
> untouched, and available as the real-data benchmark.

---

## 1. Purpose

Transform the single ML classifier inference call into a properly
separated three-agent decision system that enforces, in code (not just
documentation), the responsibility boundaries designed in Phase 2:
risk estimation, care navigation, and safety policy are three distinct
concerns with three distinct authorities, and the safety layer has final,
non-bypassable say over every response.

## 2. Why Multi-Agent Separation Is Used

A single function that predicts risk *and* decides where to route a
member *and* decides what's safe to say conflates three different kinds
of judgment with three different failure modes. Phase 1 found exactly
this conflation in the legacy `predict.py` (risk scoring and navigation
logic mixed in one module, no deterministic safety layer at all). Phase 5
separates these so that: a bug or limitation in risk scoring cannot
silently change what the system is allowed to say about safety; a
navigation-rule change cannot accidentally acquire the authority to
declare a situation clear; and the safety layer can be audited,
tested, and reasoned about in complete isolation from the other two.

## 3. Definition of "Agent" in This Project

**An agent is a bounded software component with a defined
responsibility, input contract, decision logic, output contract, and
authority boundary.** No LLM, no external AI API (OpenAI, Anthropic,
Azure OpenAI, or otherwise), and no autonomous-agent framework is used
anywhere in `backend/agents/`. The system is deterministic and auditable
end-to-end, with exactly one exception: the trained Logistic Regression
model's probability output (Risk Detection Agent), which is itself
deterministic given fixed inputs but is a statistical estimate rather
than a hand-written rule.

---

## 4. Architecture

```
member/history data (members.csv-shaped, ed_visits.csv-shaped, care_history.csv-shaped)
        |
        v
backend/pit/features.py + windows.py  (UNMODIFIED, reused as-is)
point-in-time feature reconstruction as of index_date
        |
        v
+-------------------------------------------------------------+
|  AGENT 1 -- RISK DETECTION AGENT (backend/agents/risk_detection.py) |
|  loads uc07-risk-synthetic-v1, validates schema, predicts,   |
|  assigns tier from frozen thresholds, extracts factors        |
+-------------------------------------------------------------+
        | RiskAssessment (probability, tier, factors, model identity)
        v
+-------------------------------------------------------------+
|  AGENT 2 -- CARE NAVIGATION AGENT (backend/agents/care_navigation.py) |
|  deterministic rule tree: risk tier + utilization/access/     |
|  chronic-burden signals -> destination + reason codes          |
+-------------------------------------------------------------+
        | NavigationDecision (pre-safety-review)
        v
+-------------------------------------------------------------+
|  AGENT 3 -- SAFETY & POLICY AGENT (backend/agents/safety_policy.py) |
|  reviews CURRENT safety context + prohibited-language policy;  |
|  FINAL, non-bypassable authority                                |
+-------------------------------------------------------------+
        | SafetyDecision + FinalNavigationView
        v
FinalUC07Decision  (backend/agents/orchestrator.py assembles this --
                     the ONLY structure ever returned to a caller)
```

`backend/agents/orchestrator.py::UC07Orchestrator` is the single
authoritative entry point. No other module constructs a
`FinalUC07Decision`, and no code path returns a `NavigationDecision`
(Agent 2's pre-safety-review output) directly to a caller.

---

## 5. Risk Detection Agent

`backend/agents/risk_detection.py::RiskDetectionAgent`. Loads
`backend/models/uc07_risk_synthetic_v1_model.joblib` +
`_metadata.json`, cross-validates that the artifact's
`feature_columns`/`model_version`/thresholds agree exactly with the
metadata (`load_model_bundle()`), and **fails safely**
(`ModelIncompatibleError`, mapped to HTTP 503) if they don't — it never
silently serves predictions from a possibly-inconsistent pair.

**Output (`RiskAssessment`):** `probability`, `tier`, `contributing_factors`
(≤3 human-readable, non-causal strings), `model_version`, `dataset_id`,
`synthetic_model`, `index_date`, and the exact thresholds used.

**Structurally cannot** determine emergency status, recommend a care
destination, or diagnose — enforced by: this module never imports
`care_navigation` or `safety_policy`; `RiskAssessment` has no
navigation- or safety-shaped field (`test_risk_detection_agent.py::
test_risk_detection_module_never_imports_navigation_or_safety`,
`test_risk_assessment_has_no_navigation_or_safety_fields`).

**Point-in-time features are never computed by this agent** — it only
consumes an already-leakage-safe feature row built by
`backend/pit/features.py` (unmodified, same module used for training),
reused via `orchestrator.py::build_point_in_time_features()`.

### Frozen thresholds

Loaded from the model artifact/metadata, never hard-coded:
`moderate_threshold = 0.105986`, `high_threshold = 0.213252` — exactly
`uc07-risk-synthetic-v1`'s VALIDATION-derived thresholds from Phase 4D,
unchanged.

```
probability < 0.105986              -> LOW
0.105986 <= probability < 0.213252  -> MODERATE
probability >= 0.213252             -> HIGH
```

### Risk explanations

For Logistic Regression, `contribution_i = coefficient_i ×
standardized_value_i` is exact for the model's log-odds — a fully
deterministic, no-SHAP, no-LLM explanation basis. Top-3 (deduplicated)
contributions are mapped through a curated, non-causal phrase table
(e.g. *"Recent potentially avoidable ED utilization contributed to
elevated risk."*), never a raw feature name or coefficient value. A
vectorized `assess_batch()` produces byte-identical results to calling
`assess()` once per member (proven by
`test_assess_batch_matches_per_member_assess`) — added because scoring a
full 10,000-member census one row at a time took over two minutes; the
batched version takes ~4.5 seconds.

---

## 6. Care Navigation Agent

`backend/agents/care_navigation.py::decide()`. **Pure function** — takes
`member_id`, `risk_tier`, and a point-in-time feature row; returns a
`NavigationDecision`. **Structurally cannot** see current safety context:
`decide()` has no such parameter at all (verified by
`test_decide_signature_has_no_safety_context_parameter`), so it is not
merely policy but architecturally impossible for this agent to base a
decision on real-time safety information.

**Allowed destinations:** `PRIMARY_CARE`, `URGENT_CARE`, `TELEHEALTH`,
`CARE_MANAGEMENT`, `NO_PROACTIVE_NAVIGATION`. All five are reachable
through legitimate, crafted inputs (unit tests) and confirmed reachable
across the real synthetic population (API-level test scanning ≥3 distinct
destinations in a 10,000-member batch response).

### Deterministic rule tree (priority order, highest first)

1. **CARE_MANAGEMENT** — `(elevated risk [MODERATE/HIGH] OR repeated
   lower-acuity utilization [≥2 prior ED or ≥2 prior potentially-avoidable
   ED visits]) AND (chronic complexity [clinical_burden≥2] OR
   transportation barrier OR PCP distance >10mi OR prior CM engagement)`.
   **Never** triggered by risk alone — a complexity/access/continuity
   signal is always required in addition (`test_care_management_not_triggered_by_high_risk_alone`).
2. **TELEHEALTH** — telehealth available AND an access barrier
   (transportation barrier, or PCP/urgent-care distance >10mi) makes it
   useful.
3. **URGENT_CARE** — urgent care access materially better than PCP access,
   AND some navigation opportunity exists (elevated risk or repeated
   utilization).
4. **PRIMARY_CARE** — PCP reasonably accessible (≤10mi) AND a
   continuity/follow-up opportunity exists (prior PCP contact or recent ED use).
5. **NO_PROACTIVE_NAVIGATION** — fallback when none of the above apply
   (typically LOW risk with no meaningful access/utilization signal).

### Navigation reason codes

`ELEVATED_FUTURE_RISK`, `REPEATED_LOWER_ACUITY_HISTORY`,
`TRANSPORTATION_BARRIER`, `LIMITED_PCP_ACCESS`, `TELEHEALTH_AVAILABLE`,
`CHRONIC_COMPLEXITY`, `PRIOR_CM_ENGAGEMENT`,
`OUTPATIENT_CONTINUITY_OPPORTUNITY`, `URGENT_CARE_ACCESS_ADVANTAGE`,
`NO_OPPORTUNITY_IDENTIFIED`. Each destination's human-readable
explanation is deterministically templated from its reason codes — never
LLM-generated.

### Prohibited language (enforced twice — belt and suspenders)

This agent's templates are hand-written to avoid emergency-discouraging
language by construction, and every explanation is independently
re-checked by the Safety & Policy Agent's centralized policy before
anything reaches a caller (§9). Neither layer trusts the other alone.

---

## 7. Safety & Policy Agent

`backend/agents/safety_policy.py::decide()`. **Final, non-bypassable
authority.** Takes the `NavigationDecision` plus a `CurrentSafetyContext`
and returns `(SafetyDecision, FinalNavigationView)` — the
`FinalNavigationView` is the *only* navigation-shaped object ever allowed
to leave the orchestrator.

### Safety states

| State | Trigger | Effect |
|---|---|---|
| **OVERRIDE** | Current context provided AND `red_flag==1` OR `icu==1` OR `admitted==1` OR `major_procedure==1` OR `triage_level in {1,2}` | `destination=None`, `reason_codes=[]`, explanation replaced with the approved override message. No navigation alternative is ever presented as an alternative to emergency care. |
| **CAUTION** | Current context **not** provided at all | Navigation may still be shown, always framed for **non-emergency situations only**, with an explicit statement that the current situation's safety cannot be confirmed. |
| **CLEAR** | Current context provided AND no override condition met | Navigation shown with the standard "future, non-emergency needs only" framing. |

**Missing current context is never treated as evidence of safety** —
`CurrentSafetyContext.provided` is `False` only when every field is
`None`; in that case the state is always `CAUTION`, never `CLEAR`
(`test_missing_current_context_produces_caution_not_clear`,
`test_historical_absence_of_red_flags_does_not_imply_clear`). Historical
data (encounter history used for risk scoring) is never substituted for
current context — the two are structurally separate inputs to two
different agents.

**This agent does not, and cannot, predict whether a real-time situation
is a true emergency from the future-utilization risk model.** It only
classifies what the system is allowed to say, given whatever current
information (if any) was supplied — this is the explicit historical-vs-
real-time distinction the Phase 5 spec requires.

### Override language

> *"Emergency care should not be delayed when emergency symptoms or
> high-acuity conditions are present. If you or this member may be
> experiencing a medical emergency, call 911 or go to the nearest
> emergency department immediately. No navigation alternative is offered
> for this encounter."*

### Centralized prohibited-language policy

`safety_policy.PROHIBITED_PHRASES` — 32 case-insensitive,
whitespace-normalized phrases including every one explicitly required by
the spec (`avoid the er`/`ed`, `don't`/`do not go to the er`/`ed`, `you
don't`/`do not need emergency care`, `not an emergency`, `unnecessary
emergency visit`, `inappropriate emergency visit`) plus close variants
(`instead of the er`/`ed`, `skip the er`/`ed`, `the er/ed is
unnecessary`/`inappropriate`, `no need for emergency`, etc.).
`check_text()` runs against **every** navigation explanation and safety
message before it can leave `decide()` — a blocked phrase is replaced
with a neutral, policy-compliant sentence, never passed through, in every
safety state (`test_blocked_navigation_text_is_replaced_not_passed_through`,
`test_blocked_language_also_replaced_under_caution`).

---

## 8. Agent Contracts

`backend/agents/contracts.py` — the only way data crosses an agent
boundary. All frozen dataclasses + enums:

| Type | Produced by | Consumed by |
|---|---|---|
| `RiskAssessment` | Risk Detection Agent | Care Navigation Agent (tier only), Orchestrator |
| `CurrentSafetyContext` | API caller | Safety & Policy Agent **only** |
| `NavigationDecision` | Care Navigation Agent | Safety & Policy Agent **only** |
| `SafetyDecision` | Safety & Policy Agent | Orchestrator |
| `FinalNavigationView` | Safety & Policy Agent | Orchestrator, API |
| `FinalUC07Decision` | Orchestrator | API / caller |

All dataclasses are `frozen=True` — mutation after construction raises
`FrozenInstanceError` (`test_contracts_are_frozen`), so an agent cannot
even accidentally patch another agent's output in place.

---

## 9. Authority Boundaries

| Boundary | How it's enforced | Test |
|---|---|---|
| Risk Agent cannot navigate | No import of `care_navigation`/`safety_policy`; `RiskAssessment` has no navigation field | `test_risk_detection_module_never_imports_navigation_or_safety` |
| Navigation Agent cannot mark CLEAR/OVERRIDE | `decide()` signature has no context parameter; `NavigationDecision` has no `state`/`override` field | `test_decide_signature_has_no_safety_context_parameter`, `test_navigation_agent_alone_cannot_mark_clear_or_override` |
| Safety Agent can override Navigation | `decide()` always returns a `FinalNavigationView` that may null out the destination | `test_safety_agent_can_override_navigation_agent` |
| Orchestrator always calls Safety last | Source-order assertion on `_decide_from_risk` | `test_orchestrator_source_calls_safety_after_navigation` |
| No code path returns raw `NavigationDecision` | `FinalUC07Decision.navigation` is typed `FinalNavigationView`; isinstance checks confirm the type, not just the shape | `test_final_decision_navigation_is_never_the_raw_navigation_decision_type` |
| Current context, not historical risk, has final say | A member whose history would drive strong navigation is still fully suppressed under OVERRIDE | `test_override_current_context_always_wins_regardless_of_risk_or_navigation` |

---

## 10. Orchestration Sequence

`UC07Orchestrator.decide_for_member()` (single) and
`.decide_for_all_members()` (batch, vectorized risk scoring) both funnel
through the single `_decide_from_risk()` method that calls
`care_navigation.decide()` then `safety_policy.decide()`, in that fixed
order, exactly once, every time (`test_single_and_batch_entry_points_both_delegate_to_decide_from_risk`,
`test_every_return_path_goes_through_safety_agent`).

```
member/history CSVs + index_date + (optional) current_safety_context
    -> build_point_in_time_features()          [backend/pit, unmodified]
    -> RiskDetectionAgent.assess() / assess_batch()
    -> care_navigation.decide()
    -> safety_policy.decide()                  [ALWAYS LAST]
    -> FinalUC07Decision
```

---

## 11. Risk Tiers

LOW/MODERATE/HIGH, from `uc07-risk-synthetic-v1`'s frozen thresholds
(§5). Operationally: LOW = no proactive escalation by risk alone;
MODERATE = meaningful navigation opportunity; HIGH = strongest
prioritization opportunity. **These are navigation-risk tiers only** —
never medical acuity, emergency severity, a diagnosis, or permission to
avoid emergency care. That distinction is enforced by the Safety Agent's
independent authority (§7) over what any tier is allowed to produce.

## 12. Navigation Rules

See §6 for the full deterministic rule tree and priority order.

## 13. Care Management Rules

Care Management requires an elevated-risk-or-repeated-utilization signal
**plus** a complexity/access/continuity signal — never risk alone
(§6, rule 1). This directly implements the Phase 2 design principle
(`docs/02_UC07_AND_DATA_DESIGN.md` §20) and makes Care Management
reachable for the first time in this project (Phase 1 found it
unreachable in the legacy `predict.py`).

## 14. Safety Override Rules

See §7. Six independent trigger conditions (`red_flag`, `icu`,
`admitted`, `major_procedure`, `triage_level∈{1,2}` — evaluated as two
separate cases for triage 1 and triage 2), each individually proven to
trigger `OVERRIDE` (`test_each_override_condition_triggers_override`,
parametrized over all six).

## 15. Missing-Current-Context Behavior

See §7. Formalized as: `CurrentSafetyContext.provided == False` (i.e. no
field was supplied at all) → `SafetyState.CAUTION`, unconditionally,
regardless of risk tier or navigation destination. A caller supplying
*some* fields (e.g. only `triage_level`) is treated as having provided
context — the state then depends on whether any supplied field trips an
override condition (`test_clear_partial_context_still_clear_if_no_override_signal`).

## 16. Prohibited Language

See §7. `safety_policy.PROHIBITED_PHRASES` is the single source of truth;
`check_text()` is applied to every navigation explanation and safety
message. Verified against all 32 required phrases individually, several
case/whitespace variants, and — as a whole-system adversarial test — every
explanation/message text produced by scoring the entire 10,000-member
synthetic population (`test_full_population_response_contains_no_prohibited_language`).

---

## 17. API Integration

`backend/main.py` gained new endpoints; **no existing endpoint's
behavior changed.**

| Endpoint | Status | Notes |
|---|---|---|
| `GET /` | unchanged | |
| `GET /dashboard` | unchanged | |
| `POST /predict` | unchanged | legacy `ed_risk_model.pkl` (`frequent_ED_user`), pre-Phase-2 |
| `POST /predict-json` | unchanged | same legacy model |
| `POST /explain-member` | unchanged | same legacy model |
| `GET /health` | **extended** | adds `uc07_model_loaded`, `uc07_model_version`, `uc07_model_error`; existing `status`/`model_loaded` fields unchanged |
| `GET /model-info` | **new** | non-sensitive UC07 model identity/config |
| `POST /uc07/decide` | **new** | the orchestrator-backed decision endpoint |

**Business logic lives in `backend/agents/`, never in `main.py`** — the
endpoint only validates input, parses `current_safety_context`/`index_date`,
calls `UC07Orchestrator`, and serializes the result
(`decision_to_dict()`). No navigation rule, threshold, or safety phrase
is duplicated in `main.py`.

### `POST /uc07/decide`

- **Input:** `members_file`, `ed_visits_file` (must include
  `triage_level`), `care_file` (same upload pattern as the legacy
  endpoints); optional `member_id` (single-member mode), `index_date`
  (ISO date, defaults to today), `current_safety_context` (JSON object
  keyed by `member_id`, e.g. `{"M00001": {"red_flag": 1, "triage_level": 2}}`
  — a member absent from this object gets `CAUTION`, never `CLEAR`).
- **Output:** `model_version`, `dataset_id`, `synthetic_model`,
  `index_date`, `count`, `decisions: [FinalUC07Decision, ...]`.

---

## 18. Model Identity

Every `/uc07/decide` response and every `RiskAssessment` carries
`model_version="uc07-risk-synthetic-v1"`, `dataset_id="synthetic_uc07_v1"`,
`synthetic_model=true` — machine-readable, always present, regardless of
safety state (even `OVERRIDE` responses still carry the risk block with
this identity). `GET /model-info` exposes the same identity plus
thresholds/target/horizon for auditability, with **zero member-level
data**.

## 19. Synthetic-Model Disclosure

The metadata field `agent.metadata["disclaimer"]` (surfaced via
`/model-info`) carries the exact required sentence: *"This model was
trained and evaluated on synthetic data and must not be interpreted as
clinically validated."* This is available for demo disclosure without
forcing every UI surface to display it — per the spec, the information
must be *available*, not necessarily shouted on every card.

---

## 20. Error Handling

| Condition | Response |
|---|---|
| Unknown `member_id` | 404, clean message, no traceback |
| Missing required CSV column (e.g. `triage_level`) | 422, names the missing column(s) |
| Invalid `triage_level` value (not 1–5) | 422 |
| Invalid binary flag (not 0/1) | 422 |
| Malformed `current_safety_context` JSON | 422 |
| Invalid `index_date` | 422 |
| Missing/incompatible model artifact | 503, `ModelIncompatibleError` message, no traceback |
| Any other inference failure | 500, generic message, no traceback |

No response body contains a raw Python traceback or file path
(`test_uc07_decide_unknown_member_returns_404` /
`test_uc07_decide_malformed_context_json_returns_422` assert
`"Traceback" not in resp.text`).

---

## 21. Testing

293 tests total (164 carried forward from Phases 3/4/4B/4C/4D, unchanged
+ 129 new Phase 5 tests) across 6 new files: `test_agent_contracts.py`,
`test_risk_detection_agent.py`, `test_care_navigation_agent.py`,
`test_safety_policy_agent.py`, `test_orchestrator.py`, `test_uc07_api.py`.
`pytest` (Phase 3) and `httpx` (Phase 5, required for FastAPI's
`TestClient`) are the only test-only dependencies, pinned in
`backend/requirements-dev.txt`.

## 22. Safety Test Matrix

| Case | Setup | Expected |
|---|---|---|
| 1 | HIGH risk + no current safety data | CAUTION, navigation shown as non-emergency guidance only |
| 2 | HIGH risk + `red_flag=1` | OVERRIDE |
| 3 | LOW risk + `triage_level=1` | OVERRIDE |
| 4 | HIGH risk + `admitted=1` | OVERRIDE |
| 5 | MODERATE risk + `icu=1` | OVERRIDE |
| 6 | HIGH risk + `major_procedure=1` | OVERRIDE |
| 7 | LOW risk, no navigation opportunity | NO_PROACTIVE_NAVIGATION |
| 8 | Elevated risk + transportation barrier + telehealth available | TELEHEALTH |
| 9 | Elevated risk + complexity + qualifying CM signal | CARE_MANAGEMENT |
| 10 | Urgent care access materially better than PCP + appropriate context | URGENT_CARE |
| 11 | PCP continuity opportunity | PRIMARY_CARE |

All 11 cases are covered by named or directly-equivalent tests across
`test_safety_policy_agent.py` and `test_care_navigation_agent.py`. All
five destinations plus `NO_PROACTIVE_NAVIGATION` are proven reachable
both via crafted unit tests and via a real 10,000-member population scan.

---

## 23. Known Limitations

1. **Synthetic model demonstration only** (restated intentionally) — no
   real-world evidentiary value; carried forward from Phase 4D.
2. **Navigation rule thresholds** (10-mile access barrier, burden≥2,
   repeated-utilization≥2) are engineering judgment calls consistent with
   Phase 2's design principles, not independently validated against
   outcomes.
3. **`current_safety_context` is caller-supplied and unverified** — the
   Safety Agent trusts whatever current-encounter data the API caller
   provides; this system has no independent way to confirm it (a Phase 6+
   concern if a real clinical integration is ever pursued).
4. **CAUTION-state framing is fixed, non-tiered** — the same caution
   language is used regardless of risk tier; a future phase could explore
   tier-sensitive caution framing if operationally useful.
5. **No frontend surfaces any of this yet** — Phase 5 is backend/agent
   architecture only, by design; a dedicated frontend integration phase
   follows.
6. **Runtime model selection (v1 vs. synthetic-v1) is not implemented** —
   the orchestrator is hard-wired to `uc07-risk-synthetic-v1` for this
   demonstration; making the model configurable is future architecture
   work, not done here.

---

## 24. File-by-File Changes

**Created:**
- `backend/agents/contracts.py`
- `backend/agents/risk_detection.py`
- `backend/agents/care_navigation.py`
- `backend/agents/safety_policy.py`
- `backend/agents/orchestrator.py`
- `backend/tests/test_agent_contracts.py`
- `backend/tests/test_risk_detection_agent.py`
- `backend/tests/test_care_navigation_agent.py`
- `backend/tests/test_safety_policy_agent.py`
- `backend/tests/test_orchestrator.py`
- `backend/tests/test_uc07_api.py`
- `docs/05_MULTI_AGENT_SYSTEM.md`

**Modified:**
- `backend/main.py` — added `/uc07/decide`, `/model-info`, extended
  `/health`; all legacy endpoints unchanged.
- `backend/tests/conftest.py` — added `backend/agents` to `sys.path`.
- `backend/requirements-dev.txt` — added `httpx==0.28.1` (required for
  FastAPI's `TestClient`; test-only, no production dependency changed).

**Not modified:** `backend/predict.py`, `backend/feature_engineering.py`,
`backend/train_model.py`, `backend/ed_risk_model.pkl`, `backend/pit/*`,
`backend/modeling/*`, any raw or synthetic dataset, any Phase 3/4C
snapshot, either model artifact.

## 25. Legacy `predict.py` Handling

`backend/predict.py` was inspected and left unmodified. Its inference
path (RandomForest on the legacy 37-feature `frequent_ED_user` schema)
is fundamentally incompatible with the new 59-feature point-in-time
schema and cannot meaningfully "delegate" to the new orchestrator without
either breaking its existing contract or silently changing what the
legacy endpoints return — both of which the spec prohibits. Instead:
**`backend/agents/orchestrator.py` is the single authoritative UC07
decision path going forward**, documented as such in `main.py`'s
module-level comment at the Phase 5 integration point; the legacy
`/predict*` endpoints remain available, unchanged, and clearly understood
(per Phase 1/3 documentation already on record) as the pre-Phase-2
legacy system, not the UC07-compliant one.

## 26. How to Run Phase 5 Locally

```bash
# from the repo root, with the project virtualenv active
uvicorn main:app --app-dir backend --reload --port 8001

# health / model identity
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/model-info

# a single-member decision
curl -X POST http://127.0.0.1:8001/uc07/decide \
  -F "members_file=@data/synthetic/raw_members.csv" \
  -F "ed_visits_file=@data/synthetic/raw_ed_visits.csv" \
  -F "care_file=@data/synthetic/raw_care_history.csv" \
  -F "member_id=M00001" \
  -F "index_date=2026-07-03"
```

Run the full test suite: `pytest backend/tests -q` (requires
`pip install -r backend/requirements-dev.txt`).
