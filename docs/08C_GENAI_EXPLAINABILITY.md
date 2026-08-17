# Phase 8C — Authoritative Model Explainability + Controlled GenAI Explanation Agent

**Date:** 2026-08-16
**Status:** Complete. STOPPED per instruction before any Docker/Azure work.

This document covers everything added in Phase 8C: structured, SHAP-based
model explainability for the frozen `uc07-risk-synthetic-v1` model, and a
new, strictly bounded fourth agent — the GenAI Explanation Agent — that
turns already-decided results into short natural-language text using a
locally hosted Ollama model (`qwen3:8b`). Neither capability makes a
decision. Both are additive and cannot change a risk probability, risk
tier, navigation destination, safety state, or safety override.

---

## 1. Why explainability was added

Phases 1–8B built a fully deterministic (model-probability aside)
multi-agent decision pipeline, but exposed only a coarse, sentence-based
`contributing_factors` list (`backend/agents/risk_detection.py`,
pre-Phase-8C) and no way for a case worker to ask "why," in plain
language, without reading raw JSON. Phase 8C adds two additive layers on
top of the existing, unmodified decision pipeline:

1. **Authoritative model explainability** — a structured,
   per-member breakdown of which features pushed the model's own
   estimate up or down, with an honest record of which mathematical
   method produced it.
2. **A controlled GenAI explanation agent** — a strictly bounded
   fourth logical component that converts (1), plus the already-decided
   navigation and safety results, into a short natural-language summary.
   It has zero decision authority.

## 2. Method selected: SHAP LinearExplainer (primary), linear contribution (fallback)

`backend/agents/model_explainability.py` tries
`shap.LinearExplainer(model, shap.maskers.Impute(background), nsamples=1000, seed=42)`
first. If SHAP construction or evaluation raises **any** exception (import
error, missing background snapshot, incompatible model, etc.), it falls
back to the exact, deterministic `coefficient_i * standardized_value_i`
decomposition (`ExplanationMethod.LINEAR_CONTRIBUTION`) — mathematically
exact for this model's log-odds, never a fabricated value. Every
`RiskAssessment.explanation_method` field records which one actually ran
for that member; nothing is ever silently mislabeled.

Both methods are computed **eagerly** for the whole batch in
`RiskDetectionAgent.assess()`/`assess_batch()` — unlike the GenAI agent,
SHAP is cheap enough (see §14) that there is no reason to make it lazy.

## 3. Why SHAP is not automatically identical to the linear-contribution shortcut

This was verified empirically before writing any code, per this phase's
explicit instruction not to assume equivalence. On 50 real TEST-snapshot
rows:

- **Independence-masker SHAP** (feature-independence assumption) matched
  the bare `coefficient × standardized_value` shortcut closely — mean
  absolute difference 0.0043. This is expected: `StandardScaler` produces
  exactly zero-mean features on the training distribution, which is
  exactly the condition under which the shortcut and an
  independence-assuming SHAP computation coincide.
- **Correlation-aware masker SHAP** (`shap.maskers.Impute`, which
  reallocates credit among genuinely correlated features) diverged
  materially — mean absolute difference 0.0185, max 0.377 (roughly
  4×/13× larger). UC07's 59-feature set has real correlation structure
  (e.g. `access_burden` is partly derived from `transportation_barrier`
  and the distance features, r≈0.70; the `prior_ed_count_{30,90,180,270}d`
  windows are nested and strongly correlated with each other).

The two methods are **not** interchangeable in general. SHAP with a
correlation-aware masker is the more statistically grounded choice for a
dataset with real feature correlation, which is why it is primary here.
See `docs/DECISION_LOG.md` #117 for the full record.

## 4. Why SHAP, not the legacy TreeExplainer

The pre-existing `/explain-member` endpoint (`backend/predict.py`) uses
`shap.TreeExplainer` against the **legacy, pre-Phase-2 RandomForest**
model (`ed_risk_model.pkl`) — a completely separate model, dataset, and
target from the UC07 pipeline (see `docs/05_MULTI_AGENT_SYSTEM.md` §17).
`TreeExplainer` does not apply to `uc07-risk-synthetic-v1`, a
`LogisticRegression` estimator; `LinearExplainer` is the SHAP explainer
designed for exactly this case. The two explainability systems remain
fully separate; Phase 8C touches only the UC07 path.

## 5. What is exposed per member

`RiskAssessment.explanation_factors`: up to 3 `INCREASES_RISK` +
2 `DECREASES_RISK` factors (fewer if the model doesn't produce that many
non-negligible ones), each:

```json
{
  "feature": "access_burden",
  "display_name": "Access burden",
  "direction": "DECREASES_RISK",
  "contribution": -0.126950,
  "explanation_method": "SHAP_LINEAR"
}
```

Plus, on `RiskAssessment` itself: `explanation_method` (`"SHAP_LINEAR"` or
`"LINEAR_CONTRIBUTION"`) and `explanation_causal` (always `false`).

**Never causal.** No factor is ever described as causing anything; the
system only ever says a factor "contributed to the model's estimate."
Enforced structurally (`FactorDirection` has exactly two non-causal
values) and by test (`backend/tests/test_model_explainability.py`,
items 1–8).

## 6. What the GenAI Explanation Agent is

`backend/agents/genai_explanation.py` is the fourth logical component in
the pipeline — **not** a decision-making agent. It converts an
already-computed, already-safety-reviewed decision summary into 1–2
sentences each for a risk explanation, a navigation explanation, and a
safety explanation, plus a short overall summary. It runs after the
Safety & Policy Agent, which remains the final decision authority (§8).

Configured via:

| Env var | Default | Meaning |
|---|---|---|
| `GENAI_ENABLED` | `false` | Master on/off switch. When `false`, every call returns the deterministic fallback immediately — Ollama is never contacted. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama's HTTP API base URL. |
| `OLLAMA_MODEL` | `qwen3:8b` | Model name passed to Ollama. |
| `GENAI_TIMEOUT_SECONDS` | `20.0` | Request timeout; on expiry, falls back (never raises). |

## 7. What Qwen may and must never do

**May:** explain the risk result in plain English; translate structured
factors into plain language; explain the navigation decision; explain the
safety state (including what "incomplete information" means); explain
uncertainty/missing context; produce concise wording.

**Must never:** calculate or modify a risk probability/tier; choose or
change a navigation destination; modify or override a
CLEAR/CAUTION/OVERRIDE safety state; diagnose a condition; determine
emergency status; recommend treatment or medication; invent
symptoms/diagnoses/risk factors/SHAP values not in its input; claim a
causal relationship; discourage ED/emergency care in any wording; expose
hidden chain-of-thought or raw prompts.

**These restrictions are enforced by application code, not merely by the
system prompt** (`genai_explanation.py`):

1. `"think": false` in the Ollama request suppresses qwen3's
   chain-of-thought from ever reaching the response at all.
2. `"format": <json_schema>` forces structured output; anything that
   isn't valid JSON matching the 5-key schema is rejected.
3. `safety_policy.check_text()` — the **same centralized prohibited-phrase
   list the Safety & Policy Agent itself enforces** — runs against every
   generated sentence.
4. A GenAI-specific prohibited-content check
   (`_genai_prohibited_violation`, negation-aware — "does not diagnose" is
   correctly allowed, "you are diagnosed with…" is not) blocks
   diagnosis/prescription/dosage/causal-claim language.
5. Field-scoped consistency checks reject output that names a **different**
   risk tier, navigation destination, or safety state than the one
   actually supplied — scoped to each field's own sentence (not `summary`,
   which legitimately synthesizes language from all three at once) and
   reason-code-aware (a faithful paraphrase of a reason code that's
   actually present, e.g. mentioning "urgent care access advantage" as
   context for a `CARE_MANAGEMENT` destination, is not treated as a
   destination-change attempt).
6. The returned `disclaimer` is **never** the model's own text — it is
   always overwritten with the same `safety_policy.BASE_DISCLAIMER` the
   rest of the app uses.

Any failure at any of these checks — or any network/timeout/HTTP/parse
failure — falls through to the deterministic fallback. Covered by
`backend/tests/test_genai_explanation.py` (items 9–18) and
`test_genai_explanation_authority.py` (items 19–24).

## 8. One-way architecture: Safety & Policy Agent remains final authority

```
Point-in-Time Features → Risk Detection Agent → Model Explainability
  → Care Navigation Agent → Safety & Policy Agent → Final structured decision
  → GenAI Explanation Agent → Human-readable explanation
```

The GenAI Explanation Agent receives the FINAL decision and produces text
only. There is no code path anywhere that feeds a GenAI response back
into the Risk Detection, Care Navigation, or Safety & Policy agents:

- `genai_explanation.py` never imports `risk_detection.py`,
  `care_navigation.py`, `orchestrator.py`, or calls
  `safety_policy.decide()` — only `safety_policy.check_text()` and
  `BASE_DISCLAIMER` are reused, as a read-only policy/wording source.
- `MemberExplanation` (`contracts.py`) has no field that could carry a
  probability/tier/destination/state value — it is text-only.
- `POST /uc07/explain` (`backend/main.py`) never touches
  `UC07Orchestrator` or `RiskDetectionAgent`.

Regression tests proving this one-way architecture:
`backend/tests/test_genai_explanation_authority.py`.

## 9. Structured input/output boundary and contract

**Input** (`POST /uc07/explain`'s `ExplainRequest`, Pydantic-validated —
built entirely from a `FinalUC07Decision` the frontend already has):

```json
{
  "risk": {"probability": 0.274, "tier": "HIGH", "model_version": "uc07-risk-synthetic-v1",
            "factors": [{"display_name": "...", "direction": "INCREASES_RISK"}]},
  "navigation": {"destination": "CARE_MANAGEMENT", "reason_codes": ["ELEVATED_FUTURE_RISK"]},
  "safety": {"state": "CAUTION", "context_completeness": "ABSENT", "context_source": "NOT_AVAILABLE"},
  "synthetic_model": true
}
```

This deliberately differs from this phase's illustrative example payload
in one respect: `safety.reason_codes` was **omitted**, not fabricated —
`SafetyDecision` (`contracts.py`) has no such field. `context_completeness`
/`context_source`, which do exist, are included instead, since they let
the model correctly explain uncertainty/missing context (a permitted
responsibility).

**Output** (`MemberExplanationResponse`):

```json
{
  "summary": "...", "risk_explanation": "...", "navigation_explanation": "...",
  "safety_explanation": "...", "disclaimer": "...",
  "explanation_source": "GENAI",
  "model_used": "qwen3:8b",
  "generation_time_ms": 11842.3
}
```

## 10. Fallback architecture

`_deterministic_explanation()` builds a MemberExplanation using **only**
the risk tier, structured `explanation_factors`, navigation
destination/reason codes, and safety state already present in the
request — zero external dependencies, cannot fail, cannot time out.
`explanation_source` is always `"GENAI"` or `"DETERMINISTIC_FALLBACK"`,
so a caller always knows which one produced what it's looking at.

`generate_explanation()` never raises. `POST /uc07/decide` and
`POST /uc07/explain` are fully decoupled endpoints — GenAI failure can
only affect the fields `/uc07/explain` itself returns; it is structurally
incapable of affecting `/uc07/decide`.

## 11. Safety-state wording (verbatim, per this phase's spec)

- **CLEAR:** "Complete current safety context was supplied and no
  configured safety override was triggered." (never "the patient is
  safe" / "no emergency")
- **CAUTION:** "Current safety information is absent or incomplete, so
  the system cannot confirm the safety-rule state fully." (never
  "probably safe")
- **OVERRIDE:** "Supplied current information triggered a configured
  high-acuity/emergency safety rule." (never a diagnosis)

These are the deterministic fallback's exact templates
(`_SAFETY_STATE_TEMPLATES`); GenAI output making the same claim in its
own words is accepted, but any GenAI output claiming a **different**
state's wording is rejected by the consistency check in §7.

## 12. Security & privacy protections

- **Minimization:** only the allow-listed `ExplainRequest` fields are ever
  sent to Ollama — no raw CSVs, no full member history, no member_id, no
  name/DOB/address.
- **No direct Ollama exposure:** the flow is strictly
  Frontend → FastAPI → Explanation Agent → Ollama; the frontend has no
  Ollama URL/port anywhere (`backend/tests/test_phase8c_frontend_architecture.py`
  asserts this).
- **No logging of payload contents:** `genai_explanation.py` performs no
  logging of any kind (no `import logging`, no `print`).
- **Sanitized errors:** any Ollama/network failure is caught and reduced
  to the deterministic fallback — no raw exception detail is ever
  returned to a caller.
- Covered by `backend/tests/test_genai_privacy.py` (items 25–27).

## 13. Frontend behavior

Two new sections in the member details drawer
(`frontend/src/uc07/components/`):

- **`WhyFlaggedSection`** ("WHY THIS MEMBER WAS FLAGGED") — renders
  `explanation_factors` with ↑/↓ indicators and an
  "Explanation method: …" footnote reflecting the actual method used.
  Purely additive alongside the existing plain-sentence
  `contributing_factors` in `RiskCard`.
- **`AiExplanationSection`** ("AI EXPLANATION") — fetches
  `POST /uc07/explain` once when the drawer opens for a member (and
  again if that member's safety state changes via "Evaluate Current
  Safety"), shows a loading state, then renders the four explanation
  fields plus a source line: *"Source: AI-generated explanation using
  Qwen3 8B."* for `GENAI`, or *"Explanation source: Deterministic system
  explanation."* for the fallback. A backend/Ollama failure surfaces a
  clean error message without breaking the rest of the drawer.

No frontend component computes, infers, or overrides any decision or
explanation value — both sections only render what the backend returned.
Enforced by the same grep-based pattern used since Phase 8B
(`test_frontend_never_assigns_an_explanation_method_or_source_literal`).
No hidden reasoning/thinking field exists anywhere in the response for the
frontend to accidentally render.

## 14. Lazy/on-demand generation

Population-wide prediction (`decide_for_all_members`) computes SHAP
explanations for every member eagerly (cheap — see §15) but **never**
calls the GenAI agent. `explainMember` (`frontend/src/uc07/api.ts`) is
only ever invoked from `AiExplanationSection`, i.e. only for the one
member a user has actually opened — verified structurally
(`test_decide_uc07_batch_path_never_calls_explain_member`).

## 15. Performance

Measured on this machine, this session:

- **SHAP explainer build:** ~5–11s, one-time per process, dominated by
  the fixed `nsamples=1000` masker-estimation cost. Cached at module
  level (`model_explainability._explainer_cache`), never rebuilt per
  request or per `RiskDetectionAgent` instance.
- **SHAP per-row cost after warm-up:** sub-millisecond; the full
  10,000-member synthetic population: ~13ms of pure SHAP compute.
- **Full 10,000-member `POST /uc07/decide`** (SHAP warm): ~10s
  end-to-end, including CSV parsing, feature reconstruction, and JSON
  serialization of 10,000 decisions × 5 factors each.
- **Ollama (`qwen3:8b`) generation:** cold start ~30s (includes model
  load into memory), warm ~7–20s per call. `GENAI_TIMEOUT_SECONDS` must
  be set generously (≥30s) to avoid unnecessary fallbacks on a cold
  Ollama process.
- **Deterministic fallback:** sub-millisecond, always.

GenAI's per-call latency is why it must stay lazy/on-demand (§14) —
10,000 sequential ~10s Ollama calls would make a population upload take
hours; SHAP's sub-13ms population cost has no such constraint.

## 16. Tests

40 automated tests across 5 new backend files plus 2 new frontend files,
covering: MODEL EXPLANATION (8), GENAI (10), AUTHORITY (6), PRIVACY (6),
FRONTEND (5 structural + component tests), REGRESSION (full existing
suite re-run). See `backend/tests/test_model_explainability.py`,
`test_genai_explanation.py`, `test_genai_explanation_authority.py`,
`test_genai_privacy.py`, `test_phase8c_frontend_architecture.py`, and
`frontend/src/uc07/__tests__/WhyFlaggedSection.test.tsx` /
`AiExplanationSection.test.tsx`. GenAI tests use monkeypatched HTTP calls
so the automated suite passes without a live Ollama instance; separate,
manual live verification (§17) confirmed real end-to-end behavior against
a running `qwen3:8b`.

## 17. Limitations

- The tier/destination/safety-state consistency checks (§7) are
  deliberately narrow, phrase-based heuristics — not full NLP — chosen to
  minimize false-positive fallbacks on legitimate output (e.g. a risk
  factor legitimately named "Telehealth availability" must not be
  mistaken for a destination claim). A sufficiently unusual phrasing
  could theoretically slip past; the deterministic fallback and the
  authoritative decision fields are unaffected either way.
- A pre-existing, unrelated pickle-serialization non-determinism in
  `backend/modeling/train.py`/`train_synthetic.py` causes their own
  reproducibility self-tests to occasionally leave a byte-different but
  content-identical model artifact on disk across full-suite pytest runs
  — see `docs/DECISION_LOG.md` #118. Not a Phase 8C issue; documented for
  future phases.
- `GENAI_TIMEOUT_SECONDS` needs to be set higher than Ollama's default
  cold-start time in production, or every first call after an idle period
  will fall back even though Ollama would have succeeded given more time.

## 18. Azure / Docker implications (not solved here)

This phase deliberately does **not** solve deployment. `OLLAMA_BASE_URL`
is fully configurable via environment variable specifically so it is
never hard-wired to `localhost` — but a working local
`http://localhost:11434` (this developer's laptop) is not something
Azure can assume exists. Phase 9 must decide the GenAI deployment
strategy (e.g. a containerized Ollama sidecar, a managed inference
endpoint, or shipping with `GENAI_ENABLED=false` initially) before any
production rollout. Nothing in Phase 8C requires that decision to be made
now: with `GENAI_ENABLED=false` (the default), the entire app — including
`POST /uc07/explain`, which still returns a deterministic explanation —
functions with zero dependency on Ollama being reachable anywhere.
