# Phase 8D — Critical Fixes + Regression Hardening

**Date:** 2026-08-16
**Status:** Complete. STOPPED per instruction before any Docker/Azure work.

This phase fixes the CRITICAL and HIGH findings from the Phase 8C health
check without retraining the model, changing thresholds, or altering the
Risk/Navigation/Safety agents' decision logic. Every fix here is either
(a) a hardening of the GenAI Explanation Agent's validation contract, or
(b) an execution-context fix (how/where code runs), never a change to
what a decision-making agent decides.

## 1. GenAI safety-state contradiction bug

**Bug:** the GenAI Explanation Agent's old validation only rejected text
containing a *known-wrong* state's exact phrase. A vague-but-wrong
reassurance ("Everything looks fine, no need to worry") for an actual
`safety.state = OVERRIDE` named no *other* state's phrase, so it passed
validation and was served to the user labeled `explanation_source: GENAI`
— directly contradicting an active emergency override. The underlying
decision (`safety.state`, `navigation.destination`, etc.) was never
actually wrong; only the AI-generated *text describing* it could be.

**Root cause:** free text was trusted to carry decision semantics, and
the rejection logic was a *deny-list* (block known-bad phrasings) instead
of also *requiring* the correct meaning to be present.

**Fix — structured echo (`backend/agents/genai_explanation.py`):** the
Ollama response schema now requires three additional fields —
`risk_tier`, `navigation_destination`, `safety_state` — that the model
must copy **verbatim** from its input. After the response comes back,
the backend exact-match-checks all three against the authoritative
decision (`payload["risk"]["tier"]`, etc.) *before* looking at any free
text at all. `navigation_destination` uses the sentinel string `"NONE"`
for a null `navigation.destination` (which occurs only when
`safety.state == "OVERRIDE"`), so the echo is always well-defined. Any
mismatch → immediate deterministic fallback, no free-text check needed.

**Fix — positive safety consistency:** rejecting known-wrong phrases is
no longer sufficient on its own. Three new layers, all in
`_safety_state_consistency_violation`:

1. `_SAFETY_REASSURANCE_FORBIDDEN_PHRASES` — ~25 reassurance/all-clear
   phrasings ("everything looks fine", "you are safe", "no further
   action is needed", "not an emergency", …) that are **never**
   acceptable in `safety_explanation`, regardless of the actual state
   (even a genuine CLEAR must never be phrased this way — Phase 8C Part
   12 already required this; this closes the enforcement gap).
2. The existing "names a different state's exact phrase" check.
3. `_SAFETY_STATE_REQUIRED_ANY_PHRASES` — for OVERRIDE and CAUTION
   specifically, the text must contain *at least one* phrase that
   positively carries the required meaning (an override/trigger word, or
   an incomplete/unknown-information word). Silence on the required
   meaning is now itself a violation, not just contradiction of it. CLEAR
   has no required-phrase list — its correct wording is a narrow, exact
   claim the deterministic template already models precisely.

**Verified live** against a running Ollama (`qwen3:8b`): the model
reliably echoes the structured fields and produces text satisfying the
positive-consistency checks, including for OVERRIDE: *"An override
safety state is active, indicating that emergency care should not be
delayed due to a configured high-acuity safety trigger."*

## 2. Navigation-language hardening

Two new, independent checks, both scoped to `navigation_explanation`:

- `navigation_destination` structured echo (see above) catches any
  destination substitution outright.
- `_navigation_self_contradiction_violation` — new. For any REAL
  destination (anything except null/`NO_PROACTIVE_NAVIGATION`), text
  containing "no action needed" / "no follow-up needed" / "nothing
  further required" / etc. is rejected. This is a *self*-contradiction
  check (the CORRECT destination described as needing no action),
  distinct from `_destination_consistency_violation` (which catches
  naming a *different* destination) — the two together close the exact
  adversarial example from the health check: a real `CARE_MANAGEMENT`
  destination silently downgraded to "No further action or referral is
  needed for this member at this time."

## 3. Risk-tier language hardening

The structured `risk_tier` echo (exact match) is now the PRIMARY
guarantee. On top of it, `_tier_consistency_violation` adds proximity
regexes (`_ELEVATED_RISK_RE`, `_MINIMIZING_RISK_RE`, `_EXTREME_RISK_RE`)
that catch a severity word placed near "risk" that contradicts the
actual tier via synonym — e.g. "significant and elevated risk" for an
actual LOW tier, or "minimal risk" for an actual HIGH tier — without
banning those words outright (they remain fine describing an unrelated
factor). Deliberately not an exhaustive synonym list ("do not overfit to
one phrase, if uncertain fall back" — a false positive here only costs
one fallback to deterministic text, never a wrong decision).

## 4. Deterministic fallback remains mandatory and unchanged

`_deterministic_explanation()` was not modified — it never depended on
free text or the new structured fields, and its Part 12-exact wording for
CLEAR/CAUTION/OVERRIDE is exactly what the new `_SAFETY_STATE_REQUIRED_ANY_PHRASES`
checks are modeled on. `generate_explanation()` still never raises;
malformed JSON, missing/wrong-type fields, timeout, Ollama unreachable,
prohibited content, and now structured-echo mismatches ALL fall through
to it. `POST /uc07/decide` was not touched and remains fully decoupled
from `POST /uc07/explain` — GenAI failure (of any kind, old or new) can
only affect the fields `/uc07/explain` returns.

## 5. Legacy `POST /predict` server-freeze fix

**Bug (confirmed empirically, not assumed):** wrapping `predict()` in
`starlette.concurrency.run_in_threadpool` — the textbook, usually-
sufficient FastAPI fix, and this phase's own first attempt — was
**proven insufficient** by a controlled test: an `asyncio` task ticking
every 0.5s, run concurrently with `await run_in_threadpool(predict, ...)`
on real data, was starved for the *entire* ~22 second duration of a
300-row `predict()` call. Root cause: `predict.py`'s per-row legacy SHAP
`TreeExplainer` loop holds the GIL near-continuously. CPython threads
share ONE global interpreter lock — a CPU-bound thread that doesn't
yield it (via I/O or a GIL-releasing C call) can starve every other
thread in the *process*, including the one running the asyncio event
loop, regardless of which OS thread it happens to run on. A thread pool
cannot fix that.

**Fix (`backend/main.py`):** `predict()` is now routed through a small,
lazily-created `ProcessPoolExecutor(max_workers=1)` instead of a thread
pool — a separate OS process has its own, independent GIL, immune to
this contention. `extract_features()` (confirmed NOT to exhibit this
problem — it is vectorized pandas/numpy work that releases the GIL
normally) and `explain_member()` (a much smaller, ~70ms-per-call
blocking window) remain on `run_in_threadpool`, which is correctly
sufficient for both. Model behavior, output format, and SHAP logic are
completely unchanged — only *where* `predict()` executes changed.

**Verified live:** with the process-pool fix, `GET /health` responded in
3ms while `/predict` was actively running in its own process (vs. 20+
seconds of complete unresponsiveness before any fix, and ~22s of
asyncio-level starvation with thread-pool-only). `POST /uc07/decide`
also stayed fully responsive throughout. `/predict`'s own output
(correctness, column shape, SHAP fields) is unchanged — confirmed via a
real 300-row round trip.

Regression tests: `backend/tests/test_phase8d_legacy_concurrency.py`
makes real concurrent ASGI requests (httpx `AsyncClient` +
`ASGITransport`, no live server needed) against the actual app with a
genuinely small population slice, so the suite stays fast while still
exercising the real `predict()`/SHAP code path, not a mock.

## 6. Frontend pagination desync fix

**Bug:** `Uc07View.tsx` computed `paged = sorted.slice((page - 1) *
PAGE_SIZE, page * PAGE_SIZE)` using the raw `page` state. Evaluating a
member's Current Safety Context can change their `safety.state` such
that they no longer match an ACTIVE filter, shrinking `sorted` while
`page` stays wherever the user was. If the user was on a later page that
no longer exists, `sorted.slice(...)` returned `[]` while
`sorted.length` was still `> 0`, so neither a valid page NOR the "no
members match" empty state (keyed off `sorted.length === 0`) rendered —
the table just went blank with no explanation. `Pagination.tsx` has its
own internal `clampedPage` for *display* purposes (so the control looked
fine) but never reported that clamp back up to the parent.

**Fix (`Uc07View.tsx`):** a `clampedPage` value is now computed directly
from `page` and `sorted.length` and used for BOTH the actual page slicing
(`paged`) and the `<Pagination>` prop, so a blank page is never rendered
even for a single transient cycle. A companion `useEffect` commits the
clamped value back into `page` state, so a later "Next"/"Previous" click
starts from the corrected position, not the stale one.

Regression tests: `frontend/src/uc07/__tests__/Uc07View.pagination.test.tsx`
— reproduces the exact scenario (26 CAUTION members, filtered to safety=CAUTION,
navigate to page 2, evaluate the sole member on page 2 to CLEAR, verify
the table clamps to a valid non-blank page 1) and separately verifies the
true-zero-matches case still shows the proper empty state.

## 7. `/uc07/explain` HTTP contract tests

Added to `backend/tests/test_uc07_api.py`: valid request, malformed JSON
body, missing top-level field, missing nested field, invalid
`risk_tier`/`navigation_destination`/`safety_state` enum values, invalid
probability range, no body, null `navigation.destination` (a legitimately
valid case for OVERRIDE), and a dedicated stack-trace-leakage check across
several malformed bodies. All resolve via Pydantic's automatic 422
responses or the endpoint's own 200/fallback path — no new exception
handling was needed in `main.py`.

## 8. SHAP UI wording

No math changed. `WhyFlaggedSection.tsx` now shows an explicit caveat
under the factor list: *"Model contribution values are attribution
signals and may reflect correlated features; they should not be
interpreted as causal effects."*

## 9. CORS environment configuration (Phase 9 preparation)

`backend/main.py`'s hard-coded `allow_origins=["*"]` is now
`CORS_ORIGINS` (comma-separated env var), defaulting to
`http://localhost:5173,http://127.0.0.1:5173` (the two local Vite dev
server addresses) so `npm run dev` keeps working with zero configuration.
No Azure URL is hard-coded anywhere — Phase 9 sets `CORS_ORIGINS` at
deploy time. This is explicitly NOT a full production security pass (no
auth/credential changes); it only removes the literal wildcard.

## 10. Legacy isolation (confirmed, not re-implemented)

`backend/tests/test_legacy_isolation.py` and
`test_phase8c_frontend_architecture.py` (both pre-existing) already
confirm the UC07 frontend never references `/predict`, `/predict-json`,
or `/explain-member`; re-confirmed by direct grep during this phase
(zero matches). No new isolation work was needed — the Phase 8D
`/predict` fix changes only *where* legacy code executes, never what it
returns or who can reach it.

## 11. Direct frontend test coverage added

Four new, focused test files (not exhaustive suites, per instruction):
`PopulationSummary.test.tsx`, `MemberFilters.test.tsx`,
`Uc07ResultsTable.test.tsx`, `MemberDetailsDrawer.test.tsx` — 15 tests
total, prioritizing the paths actually touched by this phase's fixes
(tally correctness, filter-chip removal, sort toggling, drawer
composition including the new `AiExplanationSection`).

## 12. Model/data immutability

Verified before AND after all fixes: all 3 original datasets, all 3
synthetic datasets, all 3 synthetic snapshots, and both model artifacts
(`uc07-risk-v1`, `uc07-risk-synthetic-v1`) hash-match the Phase 8C
end-of-phase baseline exactly
(`artifacts/phase8d_critical_fixes/pre_work_hashes.json` /
`post_work_hashes.json`). No retraining, no threshold changes, no agent
decision-logic changes anywhere in this phase.

## 13. Remaining, non-blocking issues (not fixed here, per scope)

- The tier/destination/safety-state consistency checks remain
  phrase-/regex-based heuristics, not full NLP — a sufficiently unusual
  phrasing could theoretically still slip past the free-text layer.
  The structured-echo layer (§1) is the primary guarantee and is exact,
  so this residual risk is now a defense-in-depth gap, not a
  decision-authority gap.
- `test_24_25_26_model_and_dataset_hashes_unchanged` (Phase 8B, pre-
  existing) remains intermittently order-dependent for the reason
  documented in `docs/DECISION_LOG.md` #118 — not touched, out of scope.
- `@app.on_event("shutdown")` (used for the new process-pool cleanup) is
  deprecated in favor of FastAPI lifespan handlers; functionally correct,
  just triggers a deprecation warning. Not migrated, to keep this phase's
  diff minimal.
- CORS configuration (§9) is preparation only, not a full production
  security pass.
