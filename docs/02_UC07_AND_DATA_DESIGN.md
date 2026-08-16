# UC07 — Problem Definition & Multi-Agent Architecture Design (Phase 2)

**Design date:** 2026-08-15
**Phase:** 2 — UC07 Problem Definition & Architecture Design (design/documentation only)
**Builds on:** `docs/01_PROJECT_BASELINE.md` (Phase 1 audit findings)

All figures in this document were computed by direct, read-only inspection
of the three immutable source datasets (`raw_members.csv`,
`raw_ed_visits.csv`, `raw_care_history.csv`) as they exist today. No
dataset, model, or functional code was modified to produce this design.

---

## 1. Executive Summary

Phase 1 established that the current model predicts raw ED **frequency**
(`ED_visits_365d >= 2`) using features that substantially leak that same
frequency signal (`days_since_last_ED` at 40.8% importance, unwindowed
`diagnosis_*` crosstabs at 36.3%), validated with a random (non-temporal)
split, and never reaches Care Management or a deterministic safety layer.

This document replaces that design with:

- A conservative, three-state **historical avoidability classification**
  per ED encounter (`POTENTIALLY_AVOIDABLE` / `PROTECTED_OR_HIGH_ACUITY` /
  `UNCERTAIN`) built only from `triage_level`, `red_flag`, `admitted`,
  `icu`, and `major_procedure` — **not** `diagnosis`, because direct
  inspection shows `diagnosis` category has no measurable relationship to
  acuity in this dataset (see §4.5).
- A member-level, **forward-looking** target: probability of ≥1
  `POTENTIALLY_AVOIDABLE` ED encounter in a **90-day outcome window**
  following a fixed index date — a horizon chosen from measured prevalence
  stability and observation-window feasibility, not convention (§8).
- A **point-in-time observation/index/outcome window structure** with a
  3-snapshot (train/validation/test) **temporal** split that fits inside
  the dataset's actual 547-day span (§9–§11).
- An explicit **leakage policy** separating label-only variables from
  historical predictors, resolving the two Phase 1 leakage findings without
  discarding ED history as a feature source entirely (§12).
- A **three-agent architecture** — Risk Detection, Care Navigation, and
  Safety & Policy — with the Safety & Policy Agent holding final,
  deterministic authority over every response (§20–§22), and a reachable
  Care Management recommendation (§24).

This is a design document. No target was created, no features were
implemented, no model was retrained, and no agent code was written.

---

## 2. Why the Current Frequency Target Is Inadequate

`frequent_ED_user = ED_visits_365d >= 2` measures only how many times a
member visited the ED, with no regard to why. The three datasets already
carry the fields needed to distinguish lower-acuity from clinically urgent
encounters (`triage_level`, `red_flag`, `admitted`, `icu`,
`major_procedure`), but target construction ignores all of them. A member
with two genuinely emergent visits (e.g., two cardiac events) receives the
identical label to a member with two low-acuity visits a PCP could have
handled. This directly contradicts UC07's objective, which is about
**avoidability**, not volume. Phase 1 also showed the two features the
model relies on most heavily are themselves near-restatements of ED
visit history, which compounds the problem: the model is not learning risk
factors for avoidable utilization, it is substantially re-deriving whether
someone has used the ED recently.

---

## 3. Dataset Constraints (Recap, Verified in This Phase)

| Dataset | Rows | Unique members | Date range | Notes |
|---|---|---|---|---|
| `raw_members.csv` | 10,000 | 10,000 | n/a (static) | anchor for all joins |
| `raw_ed_visits.csv` | 13,798 | 8,447 | 2025-01-01 → 2026-07-02 | 1–8 visits/member, mean 1.63 |
| `raw_care_history.csv` | 26,289 | 9,160 | 2025-01-01 → 2026-07-02 | PCP 12,378 / Telehealth 6,978 / Urgent Care 4,321 / Care Management 2,612 |

**Total usable span: 547 days (~18 months), with only 2 trailing days of
July 2026 data** (44 ED rows / 104 care rows on 2026-07-01/02) — the last
calendar day in the data is treated as the fixed dataset boundary, not as a
representative full month. Monthly volumes are otherwise stable
(~700–850 ED visits/month, ~1,400–1,500 care visits/month) with no
detected seasonality anomaly that would bias window selection.

No source column was added, removed, or renamed to produce any figure in
this document.

---

## 4. Historical Avoidability Definition

### 4.1 Design principle

The label must never assert that a specific visit was clinically
unnecessary. It represents only: *"this encounter's recorded
characteristics are consistent with a lower-acuity / potentially avoidable
utilization pattern."* Uncertain cases are excluded from the positive
definition rather than guessed at (§4.4, §6).

### 4.2 Variables considered, and why

| Variable | Values actually observed | Used in label? | Rationale |
|---|---|---|---|
| `triage_level` | 1 (most acute) – 5 (least acute), all 5 levels present | **Yes** | Standard acuity proxy; directly recorded at the encounter, not inferred |
| `red_flag` | 0/1, 6.67% positive rate | **Yes** | Explicit clinical safety marker in the source data — treated as an absolute exclusion signal |
| `admitted` | 0/1, 7.1% positive rate overall | **Yes** | Admission is strong evidence the visit required inpatient-level care, i.e., not avoidable |
| `icu` | 0/1, 0.5% positive rate overall | **Yes** | Strongest possible severity signal present in the data |
| `major_procedure` | 0/1, 3.6% positive rate overall | **Yes** | A procedure performed in the ED indicates a clinical need that a lower-acuity setting could not have met |
| `diagnosis` | 14 categories (`Other`, `UTI`, `Chest pain`, `Minor injury`, `Respiratory infection`, `Fever`, `Abdominal pain`, `Migraine`, `Diabetes-related symptoms`, `Dehydration`, `COPD exacerbation`, `Asthma exacerbation`, `Back pain`, `Hypertension`) | **No — deliberately excluded** | See §4.5. Empirically uncorrelated with acuity in this dataset; using it would encode a false clinical assumption (e.g., that "Chest pain" is inherently avoidable, or that "Minor injury" never needs a procedure) |
| `cost` | $120–$11,422.50 | **No** (sanity-check only) | Cost is a downstream consequence of care delivered, not a cause; used only to sanity-check the label after the fact (§4.6), never as a label input |

### 4.3 Rule order (safety exclusions take precedence — Step 3 requirement)

```
FOR EACH ED ENCOUNTER:

  IF red_flag == 1
     OR icu == 1
     OR admitted == 1
     OR major_procedure == 1
     OR triage_level IN {1, 2}
                                            → PROTECTED_OR_HIGH_ACUITY
                                              (never labeled avoidable,
                                               regardless of any other field)

  ELSE IF triage_level IN {4, 5}
                                            → POTENTIALLY_AVOIDABLE
                                              (candidate)

  ELSE  (triage_level == 3, no exclusion tripped)
                                            → UNCERTAIN
```

Any single exclusion condition is sufficient to force
`PROTECTED_OR_HIGH_ACUITY`, regardless of triage level — this is an
unconditional veto, evaluated first, exactly as required by Step 3.

### 4.4 Observed distribution (13,798 ED encounters, current data)

| State | Count | % of visits | Definition |
|---|---:|---:|---|
| `PROTECTED_OR_HIGH_ACUITY` | 3,969 | 28.76% | any exclusion condition true |
| `POTENTIALLY_AVOIDABLE` | 5,832 | 42.27% | no exclusion, triage 4 or 5 |
| `UNCERTAIN` | 3,997 | 28.97% | no exclusion, triage 3 |

This is intentionally conservative: triage level 3 ("urgent" on most
5-level triage scales) is treated as ambiguous rather than assumed
avoidable, even though it is the plurality single triage level. A visit
only becomes a positive `POTENTIALLY_AVOIDABLE` label if it clears triage
4–5 *and* every safety exclusion.

### 4.5 Why `diagnosis` was excluded from label construction (verified, not assumed)

Cross-tabulating `diagnosis` against every severity field shows diagnosis
category carries almost no acuity signal in this dataset:

- `triage_level` distribution is nearly proportional across all 14
  diagnosis categories (no diagnosis is concentrated at level 1–2 or 4–5).
- `red_flag` rate by diagnosis ranges narrowly (COPD exacerbation 12.7%,
  Dehydration 4.1%; most categories cluster near the 6.67% overall rate).
- `admitted` rate by diagnosis ranges 5.0–7.5% for every category, with no
  category standing out.
- `icu` rate is under 1% for every single diagnosis category, including
  "Minor injury" and "Migraine."
- `major_procedure` rate ranges 3.0–5.7% across all categories, again with
  no separation.

Given this, encoding a rule like "Chest pain → not avoidable" or "UTI →
avoidable" would be an assumption invented by this design, not a pattern
supported by the data — which the brief explicitly prohibits. `diagnosis`
is therefore used only as a **feature** (via strictly-prior, point-in-time
counts, §12) and never as a **label** input.

### 4.6 Sanity check (not a label input)

Mean `cost` for encounters meeting the `POTENTIALLY_AVOIDABLE` criteria is
$855.96 vs. $1,033.45 for excluded (`PROTECTED_OR_HIGH_ACUITY` +
`UNCERTAIN`) encounters — directionally consistent with lower-acuity care
being less costly, which supports (without proving) that the rule is
pointed in a clinically sensible direction.

---

## 5. Safety Exclusion Layer

Formalized as §4.3 above. Key properties:

- **Precedence:** safety exclusions are evaluated first and unconditionally
  override any acuity-based candidacy. There is no code path in this design
  where `POTENTIALLY_AVOIDABLE` can be reached after `red_flag`, `icu`,
  `admitted`, or `major_procedure` is true.
- **Values used come directly from the data** — no invented thresholds
  (e.g., no invented cost cutoff, no invented diagnosis whitelist).
- **Coverage:** 28.76% of all historical ED encounters are protected from
  ever being labeled avoidable, regardless of triage level.

---

## 6. Label States and Member-Level Target Mapping

### 6.1 Three encounter-level states (not forced binary)

`POTENTIALLY_AVOIDABLE`, `PROTECTED_OR_HIGH_ACUITY`, `UNCERTAIN` — defined
in §4.3–§4.4.

### 6.2 Handling of `UNCERTAIN` — chosen approach: **excluded from positive
label construction, folded into the negative class at the member level**

Three options were evaluated:

1. **Treat `UNCERTAIN` as positive** — rejected: would mean guessing that
   ambiguous triage-3, non-excluded visits are avoidable, contradicting the
   brief's conservatism requirement.
2. **Exclude members whose *only* outcome-window encounters are
   `UNCERTAIN`** — rejected for the *production* target: it would leave
   those members with no defined label at all, which a binary classifier
   cannot consume, and would silently shrink/bias the training population
   toward members with clearer-cut encounters.
3. **Chosen: `UNCERTAIN` encounters never *cause* a positive label, but a
   member is not penalized or excluded for having one.** The member-level
   target (§7) is defined purely by the presence of a
   `POTENTIALLY_AVOIDABLE` encounter in the outcome window. A member whose
   only outcome-window encounter is `UNCERTAIN` (or `PROTECTED_OR_HIGH_
   ACUITY`, or who has no ED encounter at all) is labeled negative.

**Justification:** this is the most conservative option that still yields a
usable, fully-labeled binary target for every member — it never asserts
avoidability without triage-4/5 + all-clear-on-exclusions evidence, and it
never discards members from the dataset. The tradeoff, documented as an
open risk in §28, is that some genuinely avoidable triage-3 visits will be
undercounted as negative — an intentional bias toward under- rather than
over-labeling.

---

## 7. Member-Level Prediction Target

> **`potentially_avoidable_ed_risk`** = P(member has ≥1 `POTENTIALLY_AVOIDABLE`
> ED encounter during the 90-day outcome window following their index date)

This is a **binary member-level classification target** for training
(1 = at least one qualifying encounter occurred in the outcome window;
0 = otherwise), scored by the model as a probability, exactly mirroring the
current pipeline's `predict_proba` mechanics but pointed at a different,
forward-looking, avoidability-aware outcome instead of raw frequency.

---

## 8. Prediction Horizon — Chosen: 90 Days

### 8.1 Candidates evaluated, using the actual data

Outcome-window member-level prevalence, measured directly (index date =
`data_end − horizon`, outcome window = `(index date, data_end]`):

| Horizon | Index date | Members with ≥1 `POTENTIALLY_AVOIDABLE` event in window | Prevalence |
|---:|---|---:|---:|
| 30d | 2026-06-02 | 305 / 10,000 | 3.05% |
| 60d | 2026-05-03 | 616 / 10,000 | 6.16% |
| **90d** | **2026-04-03** | **909 / 10,000** | **9.09%** |
| 120d | 2026-03-04 | 1,219 / 10,000 | 12.19% |
| 180d | 2026-01-03 | 1,800 / 10,000 | 18.00% |

### 8.2 Feasibility of a 3-way **non-overlapping temporal** split (§9) at each horizon

A genuine period-based train/validation/test split needs **3 consecutive,
non-overlapping outcome windows**, each preceded by enough history to build
features. Total span available is 547 days.

- `30d × 3 = 90d` consumed → 457d left for observation history. Feasible,
  but outcome-window prevalence (3.05%) is too thin to reliably populate
  3 separate splits with enough positive examples for stable evaluation.
- `60d × 3 = 180d` consumed → 367d left. Workable but still thin
  (~600 positive events total, before splitting further into 3 snapshots).
- **`90d × 3 = 270d` consumed → 277d left for observation history.**
  Feasible, and prevalence (~9%) yields roughly 900 positive events **per
  snapshot** (measured directly at all three chosen index dates, §9.2) —
  enough for stable training and evaluation.
- `180d × 3 = 540d` consumed → only **7 days** left for observation
  history before the earliest (train) snapshot. **Infeasible** — there is
  not enough runway left to compute even a 30-day utilization feature
  before the training index date.

### 8.3 Decision

**90 days** is the largest horizon that (a) still permits a genuine 3-way
non-overlapping temporal split within the dataset's actual 547-day span,
(b) produces stable, non-trivial prevalence (~9%, confirmed consistent
across all three chosen snapshots in §9.2, not just the most recent one),
and (c) is operationally meaningful — a 90-day forward risk window is
long enough to plan and deliver a care-management or navigation
intervention, and short enough to remain actionable (180 days out is too
distant to act on with confidence; 30 days is too thin statistically and
operationally reactive rather than proactive).

---

## 9. Observation Window, Index Date, and Temporal Split Design

### 9.1 Structure

```
 OBSERVATION WINDOW (270 days, history only)  →  INDEX DATE  →  OUTCOME WINDOW (90 days, future only)
 [features computed here,                        [prediction    [label computed here,
  visit_date <= index_date]                        made here]     visit_date > index_date]
```

### 9.2 Three non-overlapping snapshots (chosen design — see §9.4 for why
period-based over member-level random)

| Snapshot | Index date | Observation window (270d, capped) | Outcome window (90d) | Prevalence (measured) |
|---|---|---|---|---:|
| **TRAIN** | 2025-10-05 | 2025-01-08 → 2025-10-05 | 2025-10-05 → 2026-01-03 | 907 / 10,000 = **9.07%** |
| **VALIDATION** | 2026-01-03 | 2025-04-08 → 2026-01-03 | 2026-01-03 → 2026-04-03 | 957 / 10,000 = **9.57%** |
| **TEST** | 2026-04-03 | 2025-07-07 → 2026-04-03 | 2026-04-03 → 2026-07-02 (dataset end) | 909 / 10,000 = **9.09%** |

All three observation windows fall entirely within the dataset's actual
range (earliest observation start, 2025-01-08, is 7 days after
`data_start`). All three outcome windows are strictly consecutive and
non-overlapping — no calendar day is ever used as both training-outcome
and validation/test-outcome. The 270-day observation cap (rather than the
initially-considered 365d) was chosen specifically because it is the
largest window that still fits before the earliest (TRAIN) snapshot's
index date without truncating history.

### 9.3 Minimum required history / cold-start members

No minimum history is required to generate a row — a member with zero
prior ED/care encounters before their snapshot's index date simply
receives 0-valued (or median-imputed, for distance-type fields) utilization
features, consistent with the current pipeline's `fillna(0)` convention.
This is an accepted, documented limitation (§28): the model will have
limited signal for genuinely history-free members, and their risk will
default toward whatever the static (demographic/access/chronic-condition)
features imply.

### 9.4 Handling members who appear in more than one snapshot

Because all three snapshots draw from the same 10,000-member population at
different points in time, a member can legitimately appear as a row in
more than one snapshot (e.g., once in TRAIN, once in TEST). This is **not**
outcome leakage — each row's label is computed strictly from that
snapshot's own, non-overlapping outcome window, and each row's features are
computed strictly from that snapshot's own, prior observation window; no
future information ever crosses from a later snapshot back into an earlier
one. It is measured to be a modest effect: only 66–72 members overlap
between any two snapshots' *positive* sets (out of ~900–950 positives per
snapshot, i.e., ~7–8%). This is treated as an accepted, documented
non-independence between splits (common in claims-style rolling-cohort
designs), with three concrete guardrails required at implementation time:

1. **Preprocessing (imputers/encoders) must be fit only on the TRAIN
   snapshot** and applied unchanged to VALIDATION and TEST.
2. **Risk-tier thresholds must be selected using VALIDATION only.**
3. **Final reported performance must come from a single, untouched pass
   over TEST** — no iterating back into TEST to adjust the model.

A fully member-disjoint split (each member assigned to exactly one
snapshot) was considered and rejected for this phase: with only ~900
positive events per snapshot already, further fragmenting the 10,000-member
population by identity as well as by time would meaningfully shrink and
destabilize each split's positive count. It remains a candidate refinement
for a later phase if overlap-driven optimism is observed in practice.

### 9.5 Why period-based (not random member-level) splitting

The brief explicitly asks to replace the current random split with a
temporal design. §9.2's measurements confirm a genuine 3-period design is
feasible within the actual data span and produces stable prevalence — so
the "safest realistic alternative" fallback (a random member-level split)
is not needed here; the full temporal design is used.

### 9.6 Single production snapshot vs. this training design

For **training/evaluation**, the 3-snapshot design above is used. For
**live production inference** (i.e., scoring an uploaded "current" member
file through the eventual retrained model), only one index date is
relevant at a time — "today" — with a single 270-day observation window
looking backward from the upload's effective reference date. This mirrors
how `predict.py` already treats `reference_date` as the newest available
date; Phase 3 will need to redefine that reference date as the *index
date* for feature computation and stop using it to compute any label-only
quantity.

---

## 10. Evaluation Metrics

| Metric | Relevance to UC07 |
|---|---|
| **PR-AUC (primary)** | With ~9% prevalence, PR-AUC is far more informative than ROC-AUC about how well the model ranks true positives given the operational reality that most members are negative in any 90-day window |
| **Recall / Sensitivity at a fixed operating precision (primary, paired with PR-AUC)** | Care-management capacity is finite; the operational question is "how many true avoidable-risk members do we catch at an outreach volume we can actually staff," which recall-at-precision answers directly |
| ROC-AUC (secondary) | Standard, comparable across model iterations; less informative than PR-AUC alone under class imbalance but useful as a stability check |
| Precision / PPV (secondary) | Directly informs how many navigation outreach contacts will be "wasted" on members who were not actually going to have a qualifying event |
| Specificity (secondary) | Confirms the model isn't flagging the majority-negative population indiscriminately |
| F1 (secondary, single-threshold summary) | Useful only after a threshold is chosen from PR-AUC/recall-precision analysis; not a selection criterion on its own |
| Brier score / calibration curve (required, not optional) | Risk tiers (§14) and any probability shown to a care-management user must be trustworthy as a probability, not just correctly *ranked* — this is what calibration checks |
| Confusion matrix at the chosen operating threshold | Required for operational sign-off — makes false-positive/false-negative *counts*, not just rates, visible to stakeholders |

**Accuracy is explicitly excluded as a model-selection metric.** At ~9%
prevalence, a trivial always-negative classifier scores ~91% accuracy while
being clinically useless — this exact failure mode is why Phase 1's
reliance on `accuracy_score` (among other metrics) needs to be
de-prioritized going forward, not repeated.

**Asymmetric cost of errors:** a **false negative** (a member who will have
a potentially avoidable ED visit but is scored low-risk) means a missed
care-management opportunity — a soft cost, recoverable at the next
scoring cycle, but a lost chance to intervene. A **false positive** (a
member flagged high-risk who will not have such a visit) drives an
unnecessary outreach contact — a real but bounded cost (staff time, member
annoyance) that does **not** touch anyone's actual emergency care access,
because (per §5, §22) this system never blocks or discourages ED use for
any individual. Given that asymmetry, and that the Safety & Policy Agent
(§22) guarantees emergency care is never gated regardless of model output,
this design tolerates a moderate false-positive rate in exchange for higher
recall — to be tuned concretely using the VALIDATION split once a model
exists (§14).

---

## 11. Risk Tier Semantics (Thresholds Deferred)

Three tiers, operational meaning only — **no numeric threshold is set in
this document**; thresholds will be derived from the VALIDATION split
once a model exists, per §9.4's guardrail #2.

| Tier | Operational meaning for UC07 |
|---|---|
| **LOW** | No proactive navigation escalation. Model sees no elevated pattern of potentially avoidable ED utilization risk. Member still sees the same standard app experience; nothing about ED access changes. |
| **MODERATE** | A navigation opportunity exists. Appropriate response is passive/light-touch: surfacing lower-acuity options (PCP/Urgent Care/Telehealth) for *future, non-emergency* needs, and general access-barrier education — never a same-visit intervention. |
| **HIGH** | A stronger, more confident navigation opportunity. Appropriate response adds a Care-Management review candidate flag (a human decision point, not an automated action) on top of the MODERATE-tier navigation surfacing. |

Tier assignment is entirely the Risk Detection Agent's output; what happens
*because of* a tier (which care option, whether Care Management is
triggered) is entirely the Care Navigation Agent's responsibility (§21),
and whether anything is shown at all is entirely gated by the Safety &
Policy Agent (§22).

---

## 12. Leakage Policy

### 12.1 Resolving the two Phase 1 leakage findings

Phase 1 found `days_since_last_ED` and unwindowed `diagnosis_*` counts
leaking because both were computed **using the entire dataset relative to
a single global reference date that was itself the source of the target
window** — i.e., information from the very encounters used to build the
label was also available to the model as a feature for that same label.
The fix in this design is **not** to remove ED-history features — it is to
enforce that every feature is computed using **only encounters with
`visit_date <= index_date`** (strictly prior to the outcome window), for
whichever snapshot's index date applies to that row. Recomputed this way:

- `days_since_last_ED` becomes "days between the member's most recent
  *prior* ED visit and this snapshot's index date" — a legitimate,
  non-leaking recency signal.
- `diagnosis_*` counts become "how many prior ED visits (before the index
  date) fell into each diagnosis category" — legitimate history, though
  §4.5 already shows diagnosis carries limited acuity signal in this data,
  so its predictive value should be evaluated rather than assumed once
  real training happens.

The distinction that matters is **"historical utilization known before the
prediction point" (safe) vs. "information from the event(s) being
predicted" (label-only, prohibited as a feature for that row)** — exactly
the framing requested in Step 7.

### 12.2 Leakage classification table

| Feature / Source | Allowed as feature? | Conditions | Leakage rationale |
|---|---|---|---|
| `triage_level`, `red_flag`, `admitted`, `icu`, `major_procedure`, `diagnosis` **of an outcome-window encounter** | **No — label-only** | These fields define the label for that encounter; using them as features for the same row would be using the answer to predict the answer | Direct leakage |
| Count of prior `POTENTIALLY_AVOIDABLE` encounters, count of prior `PROTECTED_OR_HIGH_ACUITY` encounters, count of prior `UNCERTAIN` encounters | **Yes** | Computed only from encounters with `visit_date <= index_date` | Historical pattern, not the outcome itself |
| `diagnosis_*` prior-encounter counts | **Yes** | Strictly prior to index date | Same reasoning as above; low expected signal per §4.5, but not leakage once point-in-time |
| `days_since_last_ED` (recomputed relative to index date, using prior visits only) | **Yes** | Must use the snapshot's own index date, never the dataset's global max date | Resolves Phase 1 finding — the *concept* is fine, the *reference point* was wrong |
| `ED_visits_{30,90,180,270}d` (prior-only, i.e. windows ending at index date) | **Yes** | Window must end at `index_date`, not extend past it | Legitimate historical utilization rate |
| Raw ED count for the **current/outcome** window (i.e., anything equivalent to today's `ED_visits_365d` computed forward from index date) | **No** | This is mechanically the label's own source field | Direct leakage (same root cause as Phase 1's original `frequent_ED_user` target) |
| `care_PCP`/`care_Urgent_Care`/`care_Telehealth`/`care_Care_Management` counts and recency | **Yes** | Prior-only | Non-ED alternative care, independent of the ED-based label by construction |
| Chronic-condition flags, `num_chronic_conditions`, `clinical_burden` | **Yes** | Static, no time dependency | No leakage risk |
| `transportation_barrier`, `telehealth_available`, `pcp_distance_miles`, `urgent_care_distance_miles`, `access_burden` | **Yes** | Static | No leakage risk |
| `age`, `gender` | **Yes, with a fairness caveat** | Static | No leakage risk; subgroup validation (§17 acceptance criteria) required before production use given gender is a protected attribute |
| `cost`, `ED_total_cost`, `ED_avg_cost` of outcome-window encounters | **No** | Downstream consequence of the encounter being labeled | Direct leakage |
| Any care/ED record with `visit_date > index_date` for that row's snapshot | **No, categorically** | n/a | Prohibited future information (Step 7C) — the single rule that supersedes every other row in this table |

---

## 13. Feature Window Design

All windows below are computed relative to each row's own snapshot index
date, using only `visit_date <= index_date` records, nested inside the
270-day observation cap established in §9.2.

### MUST HAVE
- Prior ED utilization counts: `ED_count_30d`, `ED_count_90d`,
  `ED_count_270d`
- Prior potentially-avoidable ED count: `avoidable_ED_count_90d`,
  `avoidable_ED_count_270d`
- Prior high-acuity (`PROTECTED_OR_HIGH_ACUITY`) ED count: `high_acuity_
  ED_count_270d` — included as a feature (not the label) because a
  member's *history* of high-acuity visits is a legitimate prospective risk
  signal, distinct from using a *current/outcome* high-acuity visit as a
  feature
- Recency: `days_since_last_ED`, `days_since_last_avoidable_ED` (relative
  to index date)
- Recency of alternative care: `days_since_last_PCP`, `days_since_last_
  urgent_care`, `days_since_last_telehealth`, `days_since_last_care_
  management`
- Prior alternative-care counts: `PCP_count_270d`, `urgent_care_count_
  270d`, `telehealth_count_270d`, `care_management_count_270d`
- Chronic burden: all 6 condition flags, `num_chronic_conditions`,
  `clinical_burden`
- Access: `transportation_barrier`, `telehealth_available`,
  `pcp_distance_miles`, `urgent_care_distance_miles`, `access_burden`
- Demographics: `age`

### OPTIONAL (evaluate during Phase 3 modeling, do not assume value)
- `diagnosis_*` prior-encounter counts (§4.5 — limited acuity signal
  measured, but may still help identify recurring symptom patterns; keep
  as a candidate, drop if it doesn't earn its place)
- Velocity/trend: short-window vs. long-window ratio, e.g.
  `ED_count_90d / (ED_count_270d + 1)`, to capture recent acceleration
  distinct from steady-state utilization
- `gender` (include only alongside the subgroup-validation acceptance
  criterion in §17)
- Prior `UNCERTAIN`-state ED count (may carry residual signal even though
  it doesn't drive the label itself)

### DO NOT USE
- Anything computed from encounters with `visit_date > index_date`
  (categorical prohibition, §12.2)
- `triage_level`/`red_flag`/`admitted`/`icu`/`major_procedure`/`diagnosis`
  of the outcome-window encounter(s)
- `cost` of outcome-window encounters
- Any feature computed against the dataset's global max date rather than
  the row's own snapshot index date (the specific Phase 1 bug pattern)
- `num_chronic_conditions` **and** the 6 individual chronic flags **and**
  `clinical_burden` all together without acknowledging redundancy (Phase 1
  §25) — not unsafe, but should be reviewed for multicollinearity during
  Phase 3 modeling rather than included reflexively

---

## 14. Temporal Split Design (Summary)

Already detailed in §9. Summary table:

| Split | Index date | Observation window | Outcome window | Rows | Prevalence |
|---|---|---|---|---:|---:|
| TRAIN | 2025-10-05 | 2025-01-08 → 2025-10-05 (270d) | 2025-10-05 → 2026-01-03 (90d) | 10,000 | 9.07% |
| VALIDATION | 2026-01-03 | 2025-04-08 → 2026-01-03 (270d) | 2026-01-03 → 2026-04-03 (90d) | 10,000 | 9.57% |
| TEST | 2026-04-03 | 2025-07-07 → 2026-04-03 (270d) | 2026-04-03 → 2026-07-02 (90d, dataset end) | 10,000 | 9.09% |

Guardrails (restated from §9.4): fit preprocessing on TRAIN only; select
thresholds on VALIDATION only; compute final reported metrics on TEST
exactly once.

---

## 15. Three-Agent Architecture — ASCII Diagram (Future State)

```
┌────────────────────────────────────────────────────────────────────┐
│                      UPLOADED / SCORED MEMBER DATA                   │
│      raw_members.csv-shaped, raw_ed_visits.csv-shaped,                │
│      raw_care_history.csv-shaped (unseen data, same schema)           │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │   POINT-IN-TIME FEATURE LAYER   │
                 │  (shared, versioned module —     │
                 │   NOT duplicated across train/     │
                 │   inference as it is today)          │
                 │  index_date = "now" for live scoring  │
                 │  • filters all history to               │
                 │    visit_date <= index_date               │
                 │  • builds MUST-HAVE feature groups (§13)   │
                 │  • NEVER touches outcome-window/label-only  │
                 │    fields (§12)                              │
                 └───────────────────┬───────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │        AGENT 1 — RISK DETECTION AGENT      │
                 │  in: leakage-safe historical features only  │
                 │  model: trained on §14 temporal split         │
                 │  out: risk_probability, risk_tier,             │
                 │       top_risk_factors, model_version,           │
                 │       prediction_index_date                       │
                 │  MUST NOT: diagnose, tell member where to go,      │
                 │            block ED use, decide safety              │
                 └───────────────────┬─────────────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │       AGENT 2 — CARE NAVIGATION AGENT       │
                 │  in: risk output + historical care usage +   │
                 │      access/transportation/telehealth +        │
                 │      utilization pattern                         │
                 │  logic: deterministic rule tree (as today's        │
                 │         _build_alternative_care_recommendation,     │
                 │         extended per §24)                              │
                 │  out: recommended_option ∈ {PCP, Urgent Care,           │
                 │       Telehealth, Care Management}, reason_text,          │
                 │       priority_rank                                        │
                 │  MUST NOT emit: "do not go to ER" / "avoid the ED" /        │
                 │       "you don't need emergency care" / "unnecessary"        │
                 └───────────────────┬─────────────────────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │      AGENT 3 — SAFETY & POLICY AGENT        │
                 │           (FINAL AUTHORITY, deterministic)    │
                 │  in: Agent 1 output + Agent 2 output +          │
                 │      raw safety-relevant history (red_flag,       │
                 │      triage_level, admitted, icu — HISTORICAL       │
                 │      only, never real-time)                          │
                 │  checks: prohibited-phrase scan, safety-state          │
                 │      classification (CLEAR / CAUTION / OVERRIDE),        │
                 │      mandatory disclaimer injection                        │
                 │  out: final_response (may reframe or suppress               │
                 │       Agent 2's text; NEVER upgrades urgency of                │
                 │       Agent 2's suggestion, only downgrades/blocks)             │
                 └───────────────────┬─────────────────────────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │     FASTAPI RESPONSE  →  REACT DASHBOARD    │
                 │  (existing /predict, /predict-json,           │
                 │   /explain-member endpoints, extended            │
                 │   with agent outputs — behavior change              │
                 │   deferred to Phase 3)                                │
                 └─────────────────────────────────────────────────────┘
```

---

## 16. Agent 1 — Risk Detection Agent Contract

**Responsibility:** estimate the probability of ≥1 `POTENTIALLY_AVOIDABLE`
ED encounter in the next 90 days, using only leakage-safe historical
features (§12, §13).

**Allowed inputs:** the MUST-HAVE / OPTIONAL feature groups from §13,
computed strictly before the index date. No outcome-window field, ever.

**Required outputs:**

| Field | Description |
|---|---|
| `risk_probability` | calibrated model probability, 0–1 |
| `risk_tier` | `LOW` / `MODERATE` / `HIGH` (§11 semantics; thresholds set in Phase 3 from VALIDATION) |
| `top_risk_factors` | ranked list of contributing features (technical SHAP output — see §25 for the human-facing translation layer, which this agent does **not** own) |
| `model_version` | artifact identifier, so every prediction is traceable to a specific trained model |
| `prediction_index_date` | the point-in-time date features/label were anchored to (mirrors this design's index-date concept, not the dataset's global max date) |

**Explicit prohibitions:** must not diagnose emergency symptoms, must not
tell a member where to seek care (that is Agent 2's job), must not block
ED use, must not make any safety decision (that is Agent 3's exclusive
authority).

---

## 17. Agent 2 — Care Navigation Agent Contract

**Responsibility:** given Agent 1's output plus historical care-usage and
access data, select the most appropriate *future, non-emergency* lower-
acuity option from `{Primary Care, Urgent Care, Telehealth, Care
Management}`.

**Inputs:**
- Agent 1's `risk_probability` / `risk_tier`
- Prior care-history counts/recency (`care_PCP`, `care_Urgent_Care`,
  `care_Telehealth`, `care_Care_Management` and their recency, §13)
- Access variables: `transportation_barrier`, `telehealth_available`,
  `pcp_distance_miles`, `urgent_care_distance_miles`
- Prior ED utilization pattern (counts, recency, avoidable-vs-high-acuity
  mix) — used to distinguish "recurring pattern" from "isolated event"

**Outputs:**

| Field | Description |
|---|---|
| `recommended_option` | one of `PCP`, `Urgent Care`, `Telehealth`, `Care Management` |
| `reason_text` | access/utilization-grounded justification (extends today's `alternative_care_reason` pattern) |
| `priority_rank` | ordered list of the other three options, for cases where the top choice is later reframed/suppressed by Agent 3 |

**Deterministic vs. ML logic:** rule-based/deterministic, as today —
this preserves the existing, already-reasonable
`_build_alternative_care_recommendation` structure (access/distance/
telehealth-driven) and extends it with the utilization-pattern-driven
Care Management trigger (§24). No ML model chooses the navigation option
directly; Agent 1's risk output is one deterministic *input* to Agent 2's
rule tree, not a black box making the choice itself.

**Fallback behavior:** if a required access field is missing, default to
the most conservative, lowest-barrier option (`Telehealth`, if available;
otherwise the geographically nearest of `PCP`/`Urgent Care`) rather than
failing the request outright — mirrors the "fail toward conservative
messaging" principle from §22/§23.

**Absolute prohibitions (wording):** never emit "do not go to ER," "avoid
the ED," "you don't need emergency care," "this visit is unnecessary," or
any semantic equivalent. Agent 2 proposes options for **future,
non-emergency** needs only — it is never phrased as a response to, or
override of, a current/live situation.

---

## 18. Agent 3 — Safety & Policy Agent Contract

**Final authority.** Deterministic/rule-based for the production
prototype (no ML in this agent). Every response from Agents 1 and 2 passes
through this agent before reaching the API/frontend.

**Responsibilities:** enforce safety wording, apply emergency-care
protection, suppress or reframe unsafe navigation output, detect
prohibited phrases (§17's list plus any semantic near-miss), enforce
policy constraints, attach the approved safety disclaimer (already proven
effective in the current frontend, §14 of the Phase 1 baseline), and
prevent Agent 2's recommendation from ever overriding emergency-care
safety.

**Input/output contract:**

| Direction | Field | Description |
|---|---|---|
| in | Agent 1 output | `risk_probability`, `risk_tier` |
| in | Agent 2 output | `recommended_option`, `reason_text` |
| in | Historical safety fields | prior `red_flag` / `PROTECTED_OR_HIGH_ACUITY` encounter history (historical only — see distinction below) |
| out | `safety_state` | `CLEAR` / `CAUTION` / `OVERRIDE` (§18.1) |
| out | `final_recommendation_text` | Agent 2's text, unmodified (CLEAR), reframed with added caution language (CAUTION), or replaced with a generic disclaimer-only message (OVERRIDE) |
| out | `disclaimer_text` | always present on every response, regardless of safety state |
| out | `blocked_phrases_detected` | audit trail of any phrase this agent intercepted, for governance/testing (§27) |

### 18.1 Safety states

| State | When it applies | Effect |
|---|---|---|
| **CLEAR** | No prohibited phrase detected; member has no recent `PROTECTED_OR_HIGH_ACUITY` history flagged for extra caution | Agent 2's recommendation passes through unchanged, disclaimer attached |
| **CAUTION** | Member has a recent (e.g., within the observation window) `PROTECTED_OR_HIGH_ACUITY` encounter on record, or Agent 2's `reason_text` trips a soft-warning heuristic (e.g., mentions symptoms) | Recommendation still shown, but with additional explicit language reinforcing that this is not a substitute for emergency judgment; exact heuristic thresholds are a Phase 3 implementation detail, not fixed here |
| **OVERRIDE** | A prohibited phrase (§17 list or equivalent) is detected in Agent 2's output, or a required safety field is missing/unparseable | Agent 2's `recommended_option` text is suppressed/replaced with a neutral, disclaimer-only response; the underlying risk/tier data may still be returned for internal use, but no care-navigation suggestion is shown to the member until the issue is fixed |

### 18.2 Historical vs. real-time — explicit boundary

This system scores **historical claims/utilization-style data**. It has
no access to a member's real-time symptoms, vitals, or current complaint.
Therefore:

- **A. Historical safety classification** (this design, §4–§6): a
  retrospective judgment about *past* encounters, used only to build
  training labels and historical-pattern features.
- **B. Real-time emergency assessment**: this system does **not** perform
  this and must never imply that it does. The Safety & Policy Agent's
  `CAUTION`/`OVERRIDE` states are about *what the system is allowed to
  say*, not about assessing whether a member's current situation is an
  emergency.

Every response, at every safety state, must make clear (via the
disclaimer) that the system is not a real-time emergency diagnostic tool
and that a member's own judgment to seek emergency care is never
overridden.

---

## 19. Agent Orchestration and Failure/Fallback Behavior

### 19.1 Runtime flow

```
member data → point-in-time feature generation → Risk Detection Agent
            → Care Navigation Agent → Safety & Policy Agent → final response
```

### 19.2 Failure behavior (fail toward conservative messaging, never toward
aggressive redirection)

| Failure | Behavior |
|---|---|
| Model unavailable (Risk Detection Agent cannot run) | Navigation **still runs**, using only deterministic access/utilization rules (no risk-tier input) — mirrors today's behavior where `_build_alternative_care_recommendation` never depends on the model. Response includes a flag that risk scoring was unavailable; Safety & Policy Agent still runs and still attaches the full disclaimer. |
| Required features missing for a given member | That member is scored with whatever partial feature set is available (consistent with today's `fillna(0)`/median-impute convention); if too much is missing to produce a meaningful `risk_probability`, Agent 1 returns `risk_tier = null`/"insufficient data" rather than guessing, and Agent 2 falls back to its access-only rule path. |
| Navigation Agent returns an invalid/unrecognized option | Safety & Policy Agent treats this as an `OVERRIDE` condition — the invalid output is never shown; a neutral disclaimer-only response is returned instead. |
| Safety Agent rejects/overrides an output | The member always still receives a valid response containing at minimum the safety disclaimer — the system never returns an empty or error response in place of safety messaging. |
| Any agent throws an unhandled exception | Treated the same as "Safety Agent rejects" — conservative disclaimer-only fallback, never a raw error surfaced as if it were a recommendation. |

**Guiding principle, restated:** every failure mode degrades toward *less*
navigation content and *more* conservative, disclaimer-forward messaging —
never toward a more assertive redirection away from any care setting,
and never toward silence on safety messaging.

---

## 20. Care Management — Conceptual Trigger Design

Care Management is currently unreachable (Phase 1 finding). This design
makes it reachable via a **priority-ordered** set of signals evaluated by
Agent 2, ahead of the existing PCP/Urgent Care/Telehealth logic:

**Conceptual priority (highest to lowest signal strength):**

1. **High predicted risk** (`risk_tier == HIGH` from Agent 1) **combined
   with** a recurring utilization pattern — not risk alone, to avoid
   triggering Care Management off a single noisy prediction.
2. **Repeated potentially-avoidable historical ED use** — e.g., more than
   one `POTENTIALLY_AVOIDABLE`-state prior encounter within the
   observation window (§13's `avoidable_ED_count_270d`).
3. **Multiple chronic conditions** (`num_chronic_conditions >= 2` or
   `clinical_burden` above a to-be-validated cutoff) **combined with**
   elevated ED utilization — chronic burden alone, without a utilization
   signal, is better served by routine PCP care, not Care Management.
4. **Meaningful access barriers** (`transportation_barrier == 1` and/or
   both distance fields elevated) **combined with** any of the above —
   access barriers alone (with otherwise low utilization) point toward
   Telehealth, not Care Management.
5. **Prior Care Management engagement already on record**
   (`care_Care_Management` count > 0) — continuity signal; a member
   already known to Care Management is a strong candidate for continued
   Care Management rather than being redirected elsewhere.

**Decision priority statement:** Care Management is proposed when a
**risk or repeated-utilization signal co-occurs with either chronic burden
or access barriers (or prior CM engagement)** — it is deliberately not
triggered by any single signal in isolation, to keep it a targeted,
higher-effort intervention rather than a default high-risk output. Exact
numeric cutoffs are explicitly deferred to Phase 3 validation-set analysis,
consistent with §11's threshold deferral.

---

## 21. Explainability Language Rules

**Technical explanation** (internal/clinician-facing, Agent 1's
`top_risk_factors`): may use SHAP values, feature names, and signed
numeric contributions, as the current `predict.py` SHAP pipeline already
produces.

**Member/user-facing explanation** (anything reaching the dashboard):
must be reworded into pattern language, never predictive/causal/diagnostic
language.

**5 examples of SAFE explanation language:**
1. "Recent utilization patterns contributed to a higher predicted
   navigation opportunity."
2. "This member's history shows more frequent use of lower-acuity care
   settings, which the model associates with this risk level."
3. "Limited recent contact with a primary care provider was one of the
   larger contributing factors to this score."
4. "Access-related factors, such as distance to care, were associated
   with this recommendation."
5. "This score reflects patterns in past care-seeking behavior, not a
   prediction about any specific future event."

**5 examples of PROHIBITED language:**
1. "You will visit the ED again." (implies certainty/prediction of a
   specific future event)
2. "This diagnosis indicates the visit was unnecessary." (diagnostic +
   retroactive judgment)
3. "Your chest pain is not an emergency." (diagnostic, real-time,
   dangerous)
4. "Avoid the ER for this condition." (directly prohibited navigation
   phrase)
5. "This member is a frequent flyer / abuses the ED." (judgmental,
   stigmatizing, and not supported by anything the model actually
   measures — the model measures *pattern association*, not intent or
   behavior character)

---

## 22. System Claim Boundaries

### WHAT THIS SYSTEM CAN CLAIM
- Identifies members whose historical utilization patterns are associated
  with a higher likelihood of a future potentially avoidable ED encounter,
  as defined in §4–§7.
- Supports care-navigation prioritization for care-management teams.
- Recommends lower-acuity options (PCP, Urgent Care, Telehealth, Care
  Management) for **future, non-emergency** needs.
- Provides model explanations describing which historical factors
  contributed to a score.
- States, explicitly and consistently, that it does not override a
  member's judgment to seek emergency care.

### WHAT THIS SYSTEM MUST NOT CLAIM
- Determine whether a current/real-time symptom constitutes an emergency.
- Declare any specific past or future ED visit "unnecessary" or
  "inappropriate."
- Prevent, block, or discourage ED utilization for any individual.
- Provide medical diagnosis of any kind.
- Replace clinical judgment — of the member, a caregiver, or a clinician.
- Guarantee that any predicted event will or will not occur — outputs are
  probabilistic pattern associations, not forecasts of certainty.

---

## 23. Production Acceptance Criteria (Phase 3 Definition of Done)

| Area | Acceptance criteria |
|---|---|
| **Target** | Label built exactly per §4–§7; `UNCERTAIN` never drives a positive label; safety exclusions verified to have absolute precedence in code (unit-tested) |
| **Features** | Zero features computed using `visit_date > index_date` for any row (automated check); `days_since_last_ED` and all `ED_*` window features verified to use each row's own index date, not a global dataset date |
| **Model** | Trained on the §14 temporal split; preprocessing fit only on TRAIN; PR-AUC/recall-at-precision/calibration reported on TEST exactly once; artifact stores `model_version`, training date, and the exact feature list |
| **Agents** | All three agents implemented as separable modules/functions with the input/output contracts in §16–§18; Care Navigation Agent can return `Care Management`; Safety & Policy Agent runs on 100% of responses with no bypass path |
| **Safety** | All phrases listed in §17/§21 (and documented near-miss equivalents) demonstrably blocked/reframed by automated tests; disclaimer present on every response regardless of safety state |
| **API** | Response schema extended to carry agent outputs (`risk_tier`, `recommended_option`, `safety_state`, etc.) without breaking existing `/predict`, `/predict-json`, `/explain-member` consumers; model reload-per-request behavior addressed |
| **Testing** | Unit tests for label construction, leakage-boundary features, each agent's contract, and the orchestration failure modes in §19.2; at minimum one end-to-end test per safety state (`CLEAR`/`CAUTION`/`OVERRIDE`) |
| **Docker** | Image builds, includes the model artifact and its version metadata, exposes a health check that actually loads the model (not just checks file existence) |
| **Azure** | Deployment target and secrets-handling approach documented before any resource is created; no credentials committed to the repo |
| **Documentation** | `docs/` updated to reflect the as-built system at the end of Phase 3, mirroring this design document's structure |

---

## 24. Open Design Risks

1. **`UNCERTAIN` (triage-3) conservatism trade-off** — folding all
   `UNCERTAIN` encounters into the negative label (§6.2) is intentionally
   conservative but will undercount some genuinely avoidable triage-3
   visits, likely suppressing achievable recall. Worth revisiting once a
   baseline model exists.
2. **`diagnosis` predictive value is unproven** — §4.5 shows it carries
   little *acuity* signal; it may still carry *utilization-pattern* signal
   as a feature, but this is unverified until Phase 3 modeling.
3. **Cross-snapshot member overlap** (§9.4) — a modest (~7–8%) but real
   non-independence between splits; if TEST-set performance looks
   optimistic relative to a fully independent holdout, a member-disjoint
   split should be revisited.
4. **Cold-start members** (§9.3) — no minimum history requirement means
   members with zero prior activity are scored from static features alone;
   their real-world risk is unvalidated by this design.
5. **`gender` as a feature** — retained as OPTIONAL pending the subgroup
   validation acceptance criterion (§23); should not ship without it.
6. **CAUTION-state heuristics are not fully specified** — §18.1
   intentionally defers the exact triggering logic (e.g., how "recent" a
   `PROTECTED_OR_HIGH_ACUITY` encounter must be to trigger CAUTION) to
   Phase 3, where it can be tuned against real Agent 2 outputs rather than
   guessed at here.
7. **Only one full 90-day-outcome dataset era exists** (Jan 2025–Jul
   2026) — the 3-snapshot design reuses the same 547-day span three times
   rather than drawing on genuinely independent years of data; results
   should be read as internally consistent, not as proof of long-run
   temporal stability.

---

## 25. Recommended Phase 3 Implementation Sequence

1. Implement the point-in-time feature-generation module as a single
   shared function (used identically by training and inference), replacing
   the current hand-duplicated `feature_engineering.py`/`train_model.py`
   logic, enforcing the `visit_date <= index_date` boundary from §12 by
   construction.
2. Implement label construction (§4–§7) as its own testable function,
   independent of feature generation.
3. Build the three §14 snapshot datasets (TRAIN/VALIDATION/TEST) as
   **derived** files (not modifying the three raw CSVs), and retrain the
   model on TRAIN only.
4. Select risk-tier thresholds and Care Management trigger cutoffs using
   VALIDATION only (§11, §20); report final metrics on TEST exactly once.
5. Refactor `predict.py`/`main.py` into the three agent contracts (§16–
   §18), preserving current API endpoints where possible and extending
   response schemas additively.
6. Implement the Safety & Policy Agent's phrase-detection and state logic
   (§18) with unit tests covering every prohibited phrase and every safety
   state.
7. Update the frontend to surface `risk_tier`, the reachable `Care
   Management` option, and any `CAUTION`/`OVERRIDE` messaging — without
   removing the existing, already-good disclaimer components.
8. Add automated tests per §23, then Docker packaging, then Azure
   deployment design — in that order, only after the above is verified
   correct.

---

## 26. Design Quality Self-Check (Step 21)

1. **Does the target predict FUTURE potentially avoidable utilization?**
   Yes — §7's target is defined strictly over a 90-day outcome window
   after the index date, using the §4 avoidability label, not raw
   frequency.
2. **Could any proposed feature see into the future?** No, by
   construction — §12's leakage table and §23's acceptance criterion both
   require every feature to use only `visit_date <= index_date` records;
   this is called out as an automated-check requirement, not just a
   convention.
3. **Are diagnosis/triage/admission/ICU/red-flag variables correctly
   separated between label construction and prediction features?** Yes —
   §12.2 explicitly distinguishes "these fields on the outcome encounter"
   (label-only) from "counts of these fields on prior encounters" (safe
   features).
4. **Does the Safety Agent have final authority?** Yes — §18 defines it as
   running on 100% of responses with no bypass path, and §19.2 makes every
   failure mode resolve through it.
5. **Is Care Management reachable?** Yes — §20 gives it an explicit,
   priority-ordered trigger design, distinct from being permanently
   excluded as it is today.
6. **Does the system ever imply a real emergency should be avoided?**
   No — §17, §18.2, and §22 all separately and consistently prohibit this,
   and §18.2 explicitly distinguishes historical classification from
   real-time emergency assessment so the system never claims capability it
   doesn't have.
7. **Are uncertain historical ED visits handled conservatively?** Yes —
   §6.2 excludes `UNCERTAIN` from ever driving a positive label.
8. **Is the prediction horizon supported by actual date coverage?** Yes —
   §8 shows the calculation, including why 180 days is infeasible and why
   90 days is the largest feasible choice.
9. **Is the temporal split practical with available data?** Yes — §9.2
   shows all three observation windows fit within the actual data range,
   with measured, stable prevalence across all three snapshots.
10. **Can this design realistically be implemented within the existing
    repo?** Yes — §25's sequence extends the existing `feature_engineering.
    py` / `predict.py` / `main.py` patterns (pandas window/crosstab
    features, a joblib-pickled sklearn pipeline, FastAPI routes) rather
    than introducing new frameworks; the agent split is a refactor of
    existing logic into three contracts, not a rewrite.

All ten checks pass under this design; no revision was required before
finishing.

---

## 27. Statement of No Functional Change

**No functional code, feature-engineering logic, model logic, API
behavior, or frontend behavior was changed in Phase 2.** No dataset was
modified, no derived training dataset was created, no model was retrained,
and no agent code was written. The only filesystem changes made in this
phase are `docs/02_UC07_AND_DATA_DESIGN.md` (this file) and the Phase 2
entries appended to `docs/DECISION_LOG.md` and `docs/CHANGELOG.md`.
