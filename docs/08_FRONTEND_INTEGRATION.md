# Phase 8 — Frontend Integration & Production UI Hardening

**Date:** 2026-08-16
**Status:** Complete. Backend/model/thresholds unchanged. New UC07 frontend flow added; legacy flow preserved and isolated.

---

## 1. Objective

Update the frontend so it reflects the actual multi-agent UC07 system (Risk Detection → Care Navigation → Safety & Policy → `FinalUC07Decision`) via `POST /uc07/decide`, as a pure view/interaction layer — never computing a risk tier, navigation destination, or safety state client-side.

## 2. Existing frontend audit

React 19 + TypeScript + Vite (`frontend/`), oxlint for linting, no test framework present before this phase. Entirely wired to the **legacy** `/predict-json` + `/explain-member` endpoints (pre-Phase-2 `frequent_ED_user` model, `backend/predict.py`) — `frontend/src/api.ts`, `types.ts` (`PatientRow`, `ShapExplanation`, `RiskCategory: "Low"|"Medium"|"High"`), and every component (`UploadPanel`, `PatientTable`, `PatientDetailPanel`, `StatCards`, `RiskDistributionChart`, `Pills`) modeled the legacy contract exclusively. **Zero connection to `/uc07/decide` existed anywhere in the frontend before this phase.** API base URL already environment-driven (`VITE_API_URL`, default `http://127.0.0.1:8001`, matches `.env.example` and backend docs) — no hard-coded production/Azure URL found. A light/dark CSS-token theming system (`tokens.css`) with `--status-good/warning/critical` tokens was already in place and reused directly for CLEAR/CAUTION/OVERRIDE.

## 3. Authoritative API

`POST /uc07/decide` (multipart: `members_file`, `ed_visits_file`, `care_file`, optional `member_id`, `index_date`, `current_safety_context`), `GET /health`, `GET /model-info` — all verified directly from `backend/main.py`. Response shape captured from a **live decode** of `orchestrator.decision_to_dict()` (not assumed): `artifacts/phase8_frontend/frontend_api_contract.json`. Legacy `/predict`, `/predict-json`, `/explain-member` remain registered and functional, untouched.

## 4. Frontend architecture

New `frontend/src/uc07/` module, isolated from the legacy code: `types.ts`, `api.ts`, `Uc07View.tsx`, `components/*`. `App.tsx` now renders a tab switcher — **"UC07 Navigator" (default tab)** using the new module, and **"Legacy Demo"** (explicitly labeled, banner naming the pre-Phase-2 model) using the untouched original component tree. No component exceeds a single responsibility; `Uc07View` orchestrates, `Uc07DecisionPanel` composes, everything else is presentational.

## 5. API client

`frontend/src/uc07/api.ts`: `decideUC07(params)`, `getModelInfo()`, `getHealth()`, plus `buildSafetyContextPayload()`. All UC07 network calls are centralized here — no component calls `fetch` directly. `UC07ApiError` carries the HTTP status (or `null` for a network failure) so `DecisionError` can classify it without re-parsing. Shared `frontend/src/apiConfig.ts` (`API_BASE_URL`) is the single source read by both the legacy and UC07 clients — no duplicated/divergent base-URL logic, no Azure URL embedded.

## 6. Type contracts

`frontend/src/uc07/types.ts` models `RiskAssessment`, `FinalNavigationView`, `SafetyDecision`, `FinalUC07Decision`, `UC07DecideResponse`, `CurrentSafetyContextInput`, `ModelInfoResponse`, `HealthResponse` — every field matches the live-captured backend contract exactly; no field was invented. `RiskTier`/`NavigationDestination`/`SafetyState`/`ContextCompleteness`/`ContextSource`/`ReasonCode` are string-literal unions mirroring `backend/agents/contracts.py`'s enums verbatim (cross-checked against the source file, not from memory).

## 7. Risk presentation

`RiskCard`: probability as `27.4%`-style (1 decimal, never raw float), tier badge (Low/Moderate/High, colored via the good/warning/critical tokens), top contributing factors list, model-version/index-date footnote. Eyebrow text reads **"Estimated risk of future potentially avoidable ED utilization"** / probability labeled **"predicted 90-day navigation risk"** — never "emergency risk," "chance you should avoid the ER," "accuracy," or "confidence."

## 8. Navigation presentation

`NavigationCard` translates the 5 backend enum values to human labels (`PRIMARY_CARE`→"Primary Care", …, `NO_PROACTIVE_NAVIGATION`→"No proactive navigation recommended") and renders the backend's own `explanation` string verbatim — no LLM-generated or fabricated rationale. Reason codes are shown in a collapsible "Why this recommendation?" section, also enum-label-translated.

## 9. Safety presentation

`SafetyCard` renders CLEAR/CAUTION/OVERRIDE with a **2px border, colored background, icon + text label (never color alone)**, and the backend's own `message` text. Placed **above** `NavigationCard` in `Uc07DecisionPanel`'s render order (Risk → Safety → Navigation) so the Safety Agent's output always sits with higher visual priority than Navigation's, independent of state.

## 10. Override behavior

When `safety.state === "OVERRIDE"`, `NavigationCard` renders **"Navigation logic suppressed by safety override"** instead of any destination — never "use this instead," never a fabricated destination. `OVERRIDE`'s `SafetyCard` styling is deliberately the most visually dominant state in the whole app (largest text, thickest border, `role="alert"`, box-shadow) — verified in `Uc07DecisionPanel.test.tsx`'s OVERRIDE test, which also asserts none of the 4 real destination labels leak into the DOM.

## 11. CAUTION behavior

`SafetyCard--caution` uses the warning token (not the good/green token) and the backend's own message ("current clinical/safety context... was not provided, so this system cannot confirm the current situation is non-emergency"). Verified: no test or component string contains "no emergency detected" or any phrase implying confirmed safety.

## 12. Context completeness

`ContextStatus` surfaces COMPLETE→"Complete", PARTIAL→"Partial", ABSENT→"Not available", each with a one-line explanation of what that means (e.g. PARTIAL: "the remaining fields are unknown, not assumed safe"). Raw enum values remain available in the underlying data (not stripped), just not the primary label shown.

## 13. Context source

`SOURCE_LABEL`: `CALLER_SUPPLIED`→"User/caller supplied", `NOT_AVAILABLE`→"Not available", `SYSTEM_DERIVED`→"System derived" (defined, never currently produced by the backend). **`CALLER_SUPPLIED` is never rendered as "verified"** anywhere in the codebase — checked by `test_never_labels_caller_supplied_as_verified` equivalent assertion in the test suite.

## 14. Safety disclaimer

`Uc07DisclaimerBanner` — persistent, always visible independent of whether a decision has loaded, wording aligned with `backend/agents/safety_policy.py`'s `BASE_DISCLAIMER` and Phase 5-7 docs: "For care navigation only — never a reason to delay care... Emergency care should never be delayed when emergency symptoms or high-acuity conditions are present." No new medical instructions added.

## 15. Synthetic disclosure

`SyntheticDisclosure` — a small persistent badge, "Demo model trained on synthetic data; not clinically validated." — shown both before any decision loads (next to the "About this model" toggle) and inside every rendered decision panel (with the specific `model_version` that produced it). Never hidden behind a details toggle.

## 16. Model information

`ModelInfo` — collapsible "About this model" panel backed by `GET /model-info`: version, algorithm, dataset, synthetic flag, prediction horizon, observation window, feature count, intended use, target definition, disclaimer. Deliberately kept out of the primary decision flow.

## 17. Input handling

The UC07 flow reuses the existing `UploadPanel` component unchanged (same 3-CSV drag/drop UX as the legacy flow — same required file shape). Frontend performs **no** value-range validation duplicate of the backend's `backend/agents/input_validation.py` (age/distance/binary/chronic-count checks) — any invalid upload is rejected by the backend and surfaced via `DecisionError`, exactly as designed in Phase 7.

## 18. Safety-context input

`SafetyContextForm`: `red_flag`, `icu`, `admitted`, `major_procedure` (tri-state select: Unknown/No/Yes), `triage_level` (Unknown/1-5). **A field left on "Unknown" is never sent as `0`** — `buildSafetyContextPayload()`/`setBinary()`/`setTriage()` `delete` the key rather than defaulting it, verified by 4 dedicated tests (`api.test.ts`, `SafetyContextForm.test.tsx`). If no field is set at all, no `current_safety_context` entry is sent for that member — matching the backend's "absent means ABSENT, never CLEAR" contract. Labeled for care-management/clinical staff entering structured encounter data, not member self-report (matches this repo's established "care management outreach" audience, per `Header.tsx`'s existing subtitle).

## 19. Loading states

`DecisionLoading` (accessible `role="status"`, spinner) shown during the decision request; the "Get UC07 decision" button disables while `loading` is true, preventing duplicate submissions. `ModelInfo` has its own independent loading state for `GET /model-info`.

## 20. Error states

`DecisionError` classifies by HTTP status: `null`→"Backend unavailable", `404`→"Member not found", `422`→"Invalid request data", `503`→"UC07 model unavailable", `5xx`→"Backend error" — always paired with "Decision unavailable — no navigation recommendation is being shown" and the standard emergency-care sentence. Never renders a raw stack trace (backend already returns clean `detail` strings; `parseErrorDetail()` falls back safely if the body isn't JSON). On any failure, `Uc07View` clears the previous result before showing the error rather than leaving a stale, unmarked decision on screen (Step 23's "safer UX" choice).

## 21. Legacy isolation

Legacy calls (`runPrediction`, `explainMember` in `frontend/src/api.ts`) are untouched but explicitly commented as isolated from the UC07 flow; the "Legacy Demo" tab carries a banner naming the pre-Phase-2 model and stating its output "is never a UC07 risk/navigation/safety decision." No shared state, no shared components, no shared types between the two flows (only the identical `UploadFiles` CSV-upload shape is intentionally reused, since both flows genuinely accept the same 3-file upload).

## 22. Explainability

The legacy SHAP view (`PatientDetailPanel`) is untouched and remains scoped to the legacy model only — never wired to `uc07-risk-synthetic-v1`. The UC07 flow uses **only** the Risk Detection Agent's own `contributing_factors` (already-validated, non-causal text from `backend/agents/risk_detection.py`), rendered as-is in `RiskCard` — no new explanation logic, no LLM.

## 23. Dashboard updates

New `PopulationSummary` (UC07-only, does not touch the legacy `StatCards`): LOW/MODERATE/HIGH counts, navigation-destination counts, safety-state counts, computed purely as a tally of the batch's own `FinalUC07Decision[]`. Explicit caveat text: "describe the model/agent output for this population, not a clinical outcome or diagnosis."

## 24. Accessibility

Semantic HTML (`<fieldset>`/`<legend>` for the safety-context form, `<table>`/`<caption>`/`scope="col"` for results, native `<select>`/`<label>` wrapping for tri-state inputs), `role="status"`/`aria-live="polite"` on loading, `role="alert"` on errors and OVERRIDE, `aria-expanded` on collapsible toggles, every status communicated by icon + text + color together (never color alone — `RiskCard`, `SafetyCard`, `Uc07ResultsTable` all pair an icon or bold label with the color).

## 25. Responsive behavior

Reused the existing app's responsive grid/flex patterns (`auto-fit`/`minmax` grids in `ModelInfo`, `SafetyContextForm`; `overflow-x: auto` wrapper on `Uc07ResultsTable`; the detail drawer collapses to `min(480px, 100%)` width). No branding redesign performed.

## 26. Frontend tests

No test framework existed before this phase — added `vitest` + `@testing-library/react` + `jsdom` (the minimal, idiomatic choice for this exact Vite+React+TS stack, not a "huge new framework"). **28 tests, 4 files, 100% passing**, covering all 20 required scenarios (`artifacts/phase8_frontend/frontend_test_summary.json`) plus the "missing ≠ false" safety-context contract explicitly.

## 27. Backend regression

Full backend suite re-run after all frontend work: **505/505 passing, 0 failed** — identical to the Phase 7 baseline (no backend code was modified in this phase).

## 28. Production build

`npm run lint` (oxlint): **PASS, 0 issues**. `npm run build` (`tsc -b && vite build`): **PASS** (229 kB JS / 29 kB CSS, gzipped 70 kB / 5 kB). Both clean before and after this phase's changes (`artifacts/phase8_frontend/frontend_build_summary.json`).

## 29. Local integration smoke test

Backend started locally (`uvicorn main:app --app-dir backend --port 8001`); 7 representative scenarios exercised via a Node script performing the **exact same multipart request** `decideUC07()` would (`artifacts/phase8_frontend/integration_smoke_results.json`): LOW/no-nav, MODERATE/Primary Care, Telehealth, Urgent Care, Care Management (HIGH), CAUTION (no context), OVERRIDE (`red_flag=1`) — **7/7 passed**, every response's shape matched the frontend TypeScript types exactly (no missing field). Also verified unknown-member→404 and invalid-safety-context→422 directly against the live server. No backend decision was fabricated or assumed — every result came from the real orchestrator. **Limitation:** this verifies the network contract and (separately, via jsdom) component rendering logic; it does not include a literal browser screenshot/Chrome-driven click-through, which was out of scope for the available tooling in this session.

## 30. Known limitations

- No literal browser-driven (Chrome) visual smoke test was performed — network-contract + jsdom-rendering verification only (Section 29).
- The UC07 flow's population view has no sorting/filtering/pagination (out of scope — Phase 8 asked for a decision-accurate summary, not a full analytics table).
- `SYSTEM_DERIVED` context source is modeled in the frontend but can never actually occur (backend never produces it) — correct today, worth revisiting only if a real-time context feed is ever added.
- Frontend does not cache `/model-info` across tab switches (refetches each time the panel opens) — a minor inefficiency, not a correctness issue.

## 31. File-by-file changes

**Created:**
- `frontend/src/apiConfig.ts`
- `frontend/src/uc07/types.ts`, `api.ts`, `Uc07View.tsx`, `Uc07View.css`
- `frontend/src/uc07/components/{RiskCard,NavigationCard,SafetyCard,ContextStatus,ModelInfo,DecisionError,DecisionLoading,SyntheticDisclosure,Uc07DisclaimerBanner,SafetyContextForm,Uc07DecisionPanel,PopulationSummary,Uc07ResultsTable}.{tsx,css}`
- `frontend/src/uc07/__tests__/{fixtures.ts,Uc07DecisionPanel.test.tsx,DecisionStates.test.tsx,api.test.ts,SafetyContextForm.test.tsx}`
- `frontend/src/test/setup.ts`
- `frontend/vitest.config.ts`
- `artifacts/phase8_frontend/` (5 files)
- `docs/08_FRONTEND_INTEGRATION.md` (this file)

**Modified:**
- `frontend/src/App.tsx` — tab switcher (UC07 Navigator default, Legacy Demo isolated)
- `frontend/src/App.css` — tab-nav/legacy-banner styles
- `frontend/src/api.ts` — reads shared `apiConfig.ts`, isolation comment added
- `frontend/package.json` — `test` script, new devDependencies (vitest, @testing-library/*, jsdom)

**Not modified:** `frontend/src/types.ts`, every legacy component (`UploadPanel` reused as-is, `PatientDetailPanel`, `PatientTable`, `StatCards`, `RiskDistributionChart`, `Pills`, `DisclaimerBanner`, `Header`, `EmptyState`, `ErrorBanner`), `frontend/src/tokens.css`, any backend file, the model artifacts, thresholds, or datasets.

## 32. Phase 9 readiness

Backend and frontend both build/test cleanly; the frontend now has a real, tested, isolated authoritative UC07 flow. No Dockerfile, docker-compose, ACR, Azure config, or CI/CD was created (explicitly out of scope). Phase 9 should package both apps for container deployment and inject the real backend URL via environment configuration (the frontend is already structured for this — `VITE_API_URL` is the only integration point).
