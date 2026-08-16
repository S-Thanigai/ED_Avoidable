# Phase 8B — Current Safety Context Workflow (Single-Member + Batch CSV)

**Date:** 2026-08-16
**Status:** Complete. Model, thresholds, navigation rules, and Safety Agent rules unchanged. No frontend safety logic added.

---

## 1. Why current safety context is separate from historical datasets

`raw_members.csv` / `raw_ed_visits.csv` / `raw_care_history.csv` describe the past (up to 270 days before an index date) and are used to predict a **future** outcome: `future_potentially_avoidable_ed_90d`. "Current safety context" describes something categorically different — what is true about this member's situation **right now, at the moment a navigation recommendation would be shown**. Conflating the two would let a stale historical record stand in for present-moment safety information, which is exactly the failure mode this system is built to prevent (docs/05_MULTI_AGENT_SYSTEM.md section 15, carried through every phase since).

## 2. Why it is NOT an ML training dataset

`current_safety_context` (JSON) and the new optional `safety_context_file` (CSV) are consumed **only** by `backend/agents/safety_policy.py`'s `decide()` — never by `backend/pit/features.py`, never by the Logistic Regression pipeline, never written to any snapshot. Verified directly: `test_21_22_23_risk_and_navigation_unchanged_by_safety_context` asserts `risk.probability`, `risk.tier`, and the Navigation Agent's own destination/reason_codes/explanation are byte-identical across three calls to the same member with no context, safe-complete context, and an OVERRIDE-triggering context — only `safety` differs.

## 3. Single-member workflow (Part A)

**Where:** a new "Current Safety Context" section inside the member details view (`MemberDetailsDrawer` for a batch table row, or inline for a direct single-member request) — not a pre-run form. Workflow: get the member's decision first (baseline, CAUTION by default) → open details → optionally fill in current fields → **Evaluate Current Safety** → the SAME `POST /uc07/decide`, scoped to just this member, re-run with the new context and the ORIGINAL batch's own `index_date` (so risk/navigation cannot drift) → the returned decision replaces the displayed one for that member only.

**Fields:** Red flag / ICU / Admitted / Major procedure (each: Yes / No / Unknown), Triage level (Unknown / 1-5). Selecting "Unknown" removes the key from the outgoing payload entirely — it is never sent as `0`/false (`frontend/src/uc07/components/SafetyContextForm.tsx`, unchanged from Phase 7/8, reused here).

## 4. Batch workflow (Part B)

A new **optional fourth upload**, `current_safety_context.csv`, alongside the three required historical files. The UI groups them explicitly:

```
HISTORICAL DATA (Required)
  1. Members CSV
  2. ED Visits CSV
  3. Care History CSV

CURRENT SAFETY CONTEXT (Optional)
  4. current_safety_context.csv
```

with the helper text: *"Optional current-encounter information used only by the Safety Agent. It does not affect ML risk prediction."*

## 5. CSV schema

```
member_id,red_flag,icu,admitted,major_procedure,triage_level
M00001,0,0,0,0,4
M00002,,,,,
M00003,1,0,0,0,2
```

`member_id` required; the five safety columns are each optional and may be blank per-cell (blank = unknown for that field, never coerced to 0). No other column is accepted. Verified against the spec's own worked example exactly (`test_12_mixed_clear_caution_override_population`): M00001→CLEAR, M00002→CAUTION, M00003→OVERRIDE, M00004→CLEAR, M00005→OVERRIDE.

## 6. CLEAR definition

All five current-safety fields explicitly known, and none trigger an OVERRIDE condition. Unchanged from Phase 6/7: `backend/agents/safety_policy.py`'s `_determine_state()` still requires `context.completeness == ContextCompleteness.COMPLETE`.

## 7. CAUTION definition

Current safety information is absent or incomplete (any of the five fields unknown), **and** no known field already triggers OVERRIDE. This is the conservative default and remains the outcome for any member with no row in the optional CSV and no JSON override — never CLEAR.

## 8. OVERRIDE definition

Any of `red_flag == 1`, `icu == 1`, `admitted == 1`, `major_procedure == 1`, or `triage_level in {1, 2}` — checked **before** completeness, so a single known trigger field overrides even with every other field unknown (`test_10_low_risk_member_plus_emergency_context_is_override` and the six `test_4_to_9_each_override_trigger` cases confirm this for every trigger, including on a LOW-risk member).

## 9. Safety precedence

Unchanged. `safety_policy.decide()` is called exactly once, always last, by `orchestrator.py`, for both the single-member and batch code paths — the CSV/JSON context path added in this phase only changes what `CurrentSafetyContext` object reaches that same, unmodified final call. Verified: `test_20_safety_agent_remains_final_authority_override_suppresses_navigation`.

## 10. Context completeness

`ContextCompleteness.COMPLETE / PARTIAL / ABSENT` (Phase 7, unchanged) — computed identically regardless of whether the context came from the JSON field, the new CSV, or the drawer's ad-hoc evaluate call, because all three converge on the same `CurrentSafetyContext` dataclass before reaching `safety_policy.py`.

## 11. Context provenance

`ContextSource.CALLER_SUPPLIED` for both the single-member evaluate action and the batch CSV upload — both are, factually, information the caller supplied, not something the system verified. `NOT_AVAILABLE` when nothing was supplied. `SYSTEM_DERIVED` remains defined but is never asserted by any code path added in this phase (no verification capability exists).

## 12. API changes

`POST /uc07/decide` gained one new optional multipart field: `safety_context_file` (an UploadFile). New module `backend/agents/safety_context_csv.py`, `parse_safety_context_csv(df, known_member_ids)`:
- reuses `backend/agents/safety_context_schema.py`'s `SafetyContextEntry` Pydantic model per row (no duplicated 0/1/1-5/finite validation logic)
- joins **only** by `member_id` (never row position)
- rejects (422, `SafetyContextCsvValidationError`, listing every issue found): missing `member_id` column, blank `member_id`, any unrecognized column, any duplicate `member_id` (always rejected as ambiguous — never silently resolved), any `member_id` not present in the uploaded `members_file`, and any invalid field value
- if a `member_id` appears in **both** the CSV and the JSON `current_safety_context` field, the JSON entry wins for that member (the more specific, ad-hoc single-member override) — `contexts = {**csv_contexts, **json_contexts}` in `backend/main.py`

No other endpoint changed. `backend/agents/orchestrator.py`, `risk_detection.py`, `care_navigation.py`, and `safety_policy.py`'s decision logic are **unmodified** — this phase only changed how a `dict[str, CurrentSafetyContext]` gets built before reaching the existing, unmodified `decide_for_member`/`decide_for_all_members` calls.

## 13. Frontend changes

New: `CurrentSafetyContextSection.tsx` (status/completeness/source display + editable form + "Evaluate Current Safety", used in both the drawer and the single-member inline view), `SafetyContextCsvUpload.tsx` (the optional 4th file, clearly labeled and separated from the 3 required files). Modified: `Uc07View.tsx` (removed the old pre-run safety form; added the CSV upload, a `safetyOverrides` map so an evaluated member's fresh decision supersedes the original batch result everywhere — table selection, population summary counts, drawer — without affecting any other member), `MemberDetailsDrawer.tsx` (renders the new section), `uc07/api.ts` / `uc07/types.ts` (added `safetyContextFile` to `decideUC07()`). No component computes a safety state — every one only displays what `POST /uc07/decide` returned (Section 19 test, below).

## 14. Validation rules

| Input | Rule | Result if violated |
|---|---|---|
| `red_flag`/`icu`/`admitted`/`major_procedure` | blank, or exactly 0 or 1 | 422 |
| `triage_level` | blank, or exactly 1-5 | 422 |
| `member_id` | required, non-blank, present in `members_file` | 422 |
| duplicate `member_id` rows | never allowed | 422 |
| unrecognized column | never allowed | 422 |

Blank is always "unknown," never "0/false" — enforced by both the Pydantic schema (JSON path, Phase 7) and the CSV parser (this phase), which delegates to the same schema per row.

## 15. Tests

**Backend** (`backend/tests/test_phase8b_safety_context.py`, 25 tests): all 11 single-member scenarios, all 7 batch-CSV scenarios (mixed population, missing-row CAUTION, blank-values CAUTION, 4 rejection cases + an extra unrecognized-column case), a JSON-overrides-CSV precedence test, a grep-based frontend-has-no-safety-logic check, Safety-Agent-final-authority, risk/navigation-unchanged, model/dataset hash-unchanged, and zero-prohibited-language. **Frontend** (`CurrentSafetyContextSection.test.tsx`, `SafetyContextCsvUpload.test.tsx`, `Uc07View.safetyEvaluation.test.tsx`, 15 new tests): CLEAR/CAUTION/OVERRIDE rendering, the evaluate round-trip (mocked `decideUC07`, asserts the exact payload sent and that only the evaluated member changes), error handling, optional-CSV upload/removal, mixed-state summary counts without forced equality, and Escape-key drawer close.

## 16. Limitations

- The frontend's "no independent safety decision logic" check (backend test 19) is a heuristic grep, not a formal proof — it is deliberately conservative (JSX attributes and TypeScript union-type declarations are excluded) and could in principle miss a sufficiently obfuscated violation; architectural review remains the primary safeguard.
- Dark/light theme and full responsive/keyboard behavior for the new components were verified by reusing the same CSS custom-property tokens and semantic-HTML patterns already validated in Phase 8, not re-verified pixel-by-pixel in this phase.
- The batch CSV and the single-member evaluate action are independent, session-local UI actions; there is no server-side persistence of a "current safety context" between requests (by design — it describes a single moment, not a stored fact).

## 17. Example demo workflow

```bash
curl -X POST http://127.0.0.1:8001/uc07/decide \
  -F "members_file=@data/synthetic/raw_members.csv" \
  -F "ed_visits_file=@data/synthetic/raw_ed_visits.csv" \
  -F "care_file=@data/synthetic/raw_care_history.csv" \
  -F "safety_context_file=@current_safety_context.csv"
```
produces a population with a genuine mixture of CLEAR/CAUTION/OVERRIDE, driven entirely by whichever member_ids appear in the CSV and what their fields say — verified against the live backend during this phase's work (not merely asserted): M00001→CLEAR, M00002→CAUTION, M00003→OVERRIDE, exactly matching the spec's own worked example.
