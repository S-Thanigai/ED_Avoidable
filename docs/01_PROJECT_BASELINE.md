# UC07 — Avoidable ED Utilization Navigator: Project Baseline (Phase 1 Audit)

**Audit date:** 2026-08-15
**Phase:** 1 — Repository Audit / Freeze (read-only, documentation-only)
**Branch audited:** `Venkii`

---

## 1. Executive Summary

The repository implements a working end-to-end prototype: three CSV datasets →
pandas feature engineering → a Random Forest classifier → a FastAPI backend →
a React dashboard. The pipeline is a faithful re-implementation of an original
Google Colab notebook (`AVOIDABLE.docx`) into `backend/train_model.py` /
`backend/feature_engineering.py`.

The prototype is functionally complete for a demo (upload → score → explain →
recommend), and its frontend already carries genuinely good safety language
("never a reason to delay care", "call 911"). However, several of the
concerns listed in the task brief are **confirmed** by this audit, most
importantly:

- The target is literally "2+ ED visits in 365 days" — a **frequency** label,
  not a clinically defined "avoidable" label. Nothing in the three datasets or
  the code encodes avoidability (e.g., ED visits that could have been handled
  by PCP/Urgent Care/Telehealth).
- The two dominant model features (`days_since_last_ED`, 40.8% importance,
  and the `diagnosis_*` crosstab, 36.3% combined importance) are themselves
  derived from ED visit history and correlate 0.55–0.79 with the target and
  with raw ED visit counts. This is measurable, confirmed data leakage — the
  model is substantially predicting "has this member used the ED recently /
  a lot" using ED usage as the predictor of ED usage.
- Care Management is present in the source data (`care_type == "Care
  Management"`, 2,612 rows) and is engineered into a feature
  (`care_Care_Management`), but the navigation/recommendation function only
  ever returns `PCP`, `Urgent Care`, or `Telehealth`. Care Management is not
  a reachable recommendation output.
- Risk scoring (Random Forest), navigation (rule-based recommendation), SHAP
  explainability, and safety messaging are four genuinely separate pieces of
  logic today, but they live un-separated inside two files
  (`backend/predict.py`, `backend/main.py`) with no architectural boundary,
  no interfaces, and no independent tests.
- There is no Docker, no Azure configuration, no CI/CD, and no automated
  tests anywhere in the repository.

No functional code, data, or model artifacts were changed to produce this
report.

---

## 2. UC07 Business Objective

Detect patterns of potentially avoidable emergency department utilization and
recommend a lower-acuity next step (Primary Care, Urgent Care, Telehealth, or
Care-Management follow-up) — while never discouraging or blocking
appropriate emergency care.

---

## 3. Repository Structure

```
UC07/
├── raw_members.csv                    # immutable source dataset (10,000 rows)
├── raw_ed_visits.csv                  # immutable source dataset (13,798 rows)
├── raw_care_history.csv               # immutable source dataset (26,289 rows)
├── AVOIDABLE.docx                     # original Colab-notebook spec the pipeline was ported from
├── docx_out.txt                       # plaintext extraction of AVOIDABLE.docx (pipeline reference)
├── docx_content.txt                   # empty (0 bytes) — stale/failed extraction artifact
├── csv_info.txt                       # ad-hoc shape/dtype dump of the 3 datasets
├── ed_risk_predictions.xlsx           # generated output artifact (sample run), sits at repo root
├── unseen_patient_risk_results (1).csv# generated output artifact (sample run), sits at repo root
├── zip.zip                            # full point-in-time backup of the repo (not test data)
├── backend/
│   ├── main.py                        # FastAPI app (routes, CORS, static dashboard mount)
│   ├── feature_engineering.py         # extract_features() — inference-time feature pipeline
│   ├── train_model.py                 # build_training_features() + train() — training pipeline
│   ├── predict.py                     # load_model(), predict(), explain_member(), risk/recommendation logic
│   ├── ed_risk_model.pkl              # trained model artifact (joblib dict)
│   └── requirements.txt               # backend Python dependencies
├── frontend/                          # React 19 + TypeScript + Vite dashboard (current)
│   └── src/{App.tsx, api.ts, types.ts, components/*}
├── frontend_legacy/
│   └── index.html                     # earlier vanilla-JS single-file prototype dashboard (superseded, still present)
└── docs/                              # created in this phase
    ├── 01_PROJECT_BASELINE.md
    ├── CHANGELOG.md
    └── DECISION_LOG.md
```

No `tests/` directory, no `Dockerfile`, no `.github/` or other CI config,
and no Azure configuration exist anywhere in the repository.

---

## 4. Current Architecture Diagram (AS-IS)

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ raw_members.csv  │  │raw_ed_visits.csv │  │raw_care_history  │
│   (10,000 rows)  │  │  (13,798 rows)   │  │  .csv (26,289)   │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────┬───────┴──────────┬───────────┘
                          ▼                  ▼
        ┌───────────────────────────────────────────────────┐
        │  feature_engineering.py :: extract_features()      │
        │  train_model.py         :: build_training_features()│
        │  (near-duplicate logic, kept in sync by hand)       │
        │                                                     │
        │  • parse visit_date                                 │
        │  • reference_date = MAX(ed.visit_date)  (global,     │
        │    not per-member / not point-in-time)               │
        │  • ED_visits_{30,90,180,365}d windows                │
        │  • diagnosis_* crosstab  (ALL-TIME, unwindowed)       │
        │  • ED_admissions/ICU/major_procedure/red_flag/cost   │
        │  • care_{PCP,Urgent_Care,Telehealth,Care_Management}  │
        │  • days_since_last_ED, days_since_last_care           │
        │  • access_burden, clinical_burden, alt_care_visits    │
        └───────────────────┬───────────────────────┬─────────┘
                             │                       │
                 (train only)▼                       │ (train + inference)
        ┌────────────────────────────┐               │
        │ TARGET (train_model.py)     │               │
        │ frequent_ED_user =           │               │
        │   ED_visits_365d >= 2         │               │
        │ prevalence: 20.17% positive   │               │
        └───────────────┬────────────┘               │
                         │                             │
                         ▼                             ▼
        ┌─────────────────────────────┐   ┌──────────────────────────┐
        │ DROP_BEFORE_MODEL            │   │ Same drop list applied    │
        │ (feature_engineering.py)     │   │ at inference time         │
        │ drops: member_id, dates,      │   │ (feature_engineering.py)  │
        │ ED_visits_30/90/180/365d,     │   │                            │
        │ ED_to_alternative_ratio,      │   │  ⚠ diagnosis_* and         │
        │ ED_admissions/ICU/proc/       │   │  days_since_last_ED are    │
        │ red_flag/cost                 │   │  NOT dropped — retained    │
        │  ⚠ diagnosis_* NOT dropped    │   │  as model inputs (see      │
        │  ⚠ days_since_last_ED NOT     │   │  §9 Leakage Assessment)    │
        │  dropped                      │   │                            │
        └───────────────┬───────────────┘   └─────────────┬──────────────┘
                         ▼                                 │
        ┌───────────────────────────────────────────┐      │
        │ train_model.py :: train()                   │      │
        │ • random 80/20 split, stratified, seed=42    │      │
        │   (NOT temporal / NOT point-in-time)          │      │
        │ • ColumnTransformer: median-impute numeric,   │      │
        │   most-frequent-impute + OneHot categorical   │      │
        │ • RandomForestClassifier(n_estimators=400,    │      │
        │   max_depth=12, min_samples_leaf=5,           │      │
        │   class_weight="balanced", random_state=42)   │      │
        │ • metrics: accuracy/precision/recall/F1/AUC   │      │
        │   (no PR-AUC, no confusion matrix, no CV,     │      │
        │   no calibration, no subgroup validation)     │      │
        └───────────────────┬───────────────────────┘      │
                             ▼                               │
              backend/ed_risk_model.pkl                      │
              {model, feature_columns(37), target,           │
               target_definition}  — no version/date/         │
               python-or-sklearn metadata stored               │
                             │                               │
                             └───────────────┬───────────────┘
                                              ▼
                         ┌─────────────────────────────────────┐
                         │ predict.py :: predict()               │
                         │ • joblib.load() the pkl on EVERY call │
                         │   (no startup caching)                 │
                         │ • proba = pipeline.predict_proba(X)    │
                         │ • risk_category: High>=.60,            │
                         │   Medium>=.35, else Low (hardcoded)    │
                         │ • predicted_frequent_ED = proba>=.50   │
                         │ • merges demo/access/clinical cols      │
                         │   back on for display                  │
                         └───────────┬─────────────┬─────────────┘
                                     │             │
                    ┌────────────────┘             └────────────────┐
                    ▼                                                ▼
   ┌───────────────────────────────┐            ┌──────────────────────────────┐
   │ _build_alternative_care_       │            │ _compute_shap_explanations()  │
   │ recommendation() (predict.py)  │            │ shap.TreeExplainer, top-3      │
   │ • pure rule tree on:            │            │ features + signed contribution │
   │   telehealth_available,         │            │ • /predict-json: skipped        │
   │   transportation_barrier,       │            │   (perf: ~70ms/row)             │
   │   pcp_distance_miles,           │            │ • /explain-member: computed     │
   │   urgent_care_distance_miles    │            │   on demand, single row         │
   │ • outputs: PCP | Urgent Care |  │            │ • feature names are raw sklearn │
   │   Telehealth ONLY               │            │   transformed names (e.g.       │
   │ ⚠ Care Management unreachable   │            │   "numeric__diagnosis_Other")   │
   │ ⚠ does not use risk_probability │            └──────────────────────────────┘
   │   or red_flag/admitted/icu       │
   └───────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────┐
        │ backend/main.py — FastAPI                     │
        │ GET  /            health/root                 │
        │ GET  /health       {status, model_loaded}      │
        │ GET  /dashboard     serves frontend/dist        │
        │ POST /predict        → .xlsx download            │
        │ POST /predict-json    → JSON (no SHAP)            │
        │ POST /explain-member   → JSON (single-row SHAP)    │
        │ CORS: allow_origins=["*"] / methods=["*"] / headers=["*"] │
        └───────────────────────┬───────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────┐
        │ frontend/ (React 19 + TS + Vite)               │
        │ App.tsx → Header, DisclaimerBanner (static),    │
        │ UploadPanel (3 CSV dropzones), StatCards,        │
        │ RiskDistributionChart, PatientTable,              │
        │ PatientDetailPanel (on-demand SHAP fetch +         │
        │ recommendation + per-row disclaimer)                │
        │ API_BASE_URL = VITE_API_URL, default                │
        │ http://127.0.0.1:8001 (hardcoded fallback)           │
        └─────────────────────────────────────────────┘
```

---

## 5. Current End-to-End Data Flow

1. **Training** (`backend/train_model.py`, run manually/offline): reads the
   three raw CSVs from the repo root, calls `build_training_features()`,
   constructs the target, splits, fits the pipeline, prints metrics, saves
   `backend/ed_risk_model.pkl`.
2. **Inference** (`backend/main.py` → `feature_engineering.py` →
   `predict.py`): a user uploads three CSVs (same schema) via the frontend or
   directly to the API. `extract_features()` rebuilds the same feature set
   (minus the target and leakage columns dropped for training), `predict()`
   loads the pickled pipeline fresh, scores every row, attaches
   demographic/access/clinical display columns, computes the rule-based
   recommendation, and optionally SHAP.
3. **Explainability** is decoupled from bulk scoring for performance reasons:
   `/predict-json` never computes SHAP; the frontend fetches
   `/explain-member` per row only when a user opens that patient's detail
   drawer.
4. **Display**: the frontend renders risk category, the model's probability,
   the rule-based recommendation, and (on demand) the SHAP explanation, all
   behind a static, always-visible safety disclaimer.

There is no scheduled retraining, no data pipeline orchestration, and no
persistence layer — every request is stateless and self-contained (uploaded
files live only in-memory for the duration of the request).

---

## 6. Dataset Summary (Data Dictionary)

Source: direct inspection of the three immutable CSVs (row/column counts,
dtypes, uniques, nulls, duplicates, join behavior). No dataset was modified.

### 6.1 `raw_members.csv` — 10,000 rows × 14 columns, 0 duplicate `member_id`, 0 missing values anywhere

| Column | Type | Role | Notes |
|---|---|---|---|
| `member_id` | string | join key | e.g. `M09953`, unique, present in all downstream joins |
| `age` | int | demographic feature | range 18–90 |
| `gender` | string | demographic feature | values `M`/`F` only |
| `diabetes`, `copd`, `hypertension`, `chf`, `asthma`, `ckd` | int (0/1) | chronic-condition flags | static, member-level |
| `num_chronic_conditions` | int (0–5) | derived summary | exactly equals the sum of the 6 flags above (0 mismatches) — redundant with them |
| `transportation_barrier` | int (0/1) | access feature | |
| `telehealth_available` | int (0/1) | access feature | |
| `pcp_distance_miles` | float | access feature | 0.3–30.0, no missing |
| `urgent_care_distance_miles` | float | access feature | 0.2–25.0, no missing |

Role: static member registry. Available at prediction time by construction
(no leakage risk — none of these fields are outcome-derived).

### 6.2 `raw_ed_visits.csv` — 13,798 rows × 10 columns, 0 duplicate `visit_id`, 0 missing values, 8,447 unique members (all present in `raw_members.csv`)

| Column | Type | Role | Notes |
|---|---|---|---|
| `visit_id` | string | row key | unique |
| `member_id` | string | join key | all 8,447 values exist in `raw_members.csv` |
| `visit_date` | string→datetime | temporal | range **2025-01-01 → 2026-07-02** (~18 months) |
| `diagnosis` | string (14 categories) | clinical | `Diabetes-related symptoms`, `Other`, `UTI`, `Asthma exacerbation`, `Minor injury`, `Chest pain`, `Back pain`, `Respiratory infection`, `Fever`, `Dehydration`, `Migraine`, `COPD exacerbation`, `Abdominal pain`, `Hypertension` |
| `triage_level` | int 1–5 | acuity | 1 = most acute |
| `admitted` | int (0/1) | ED-outcome feature | dropped before model |
| `icu` | int (0/1) | ED-outcome feature | dropped before model |
| `major_procedure` | int (0/1) | ED-outcome feature | dropped before model |
| `cost` | float | ED-outcome feature | $120–$11,422.50; dropped before model |
| `red_flag` | int (0/1) | safety-relevant flag | 6.7% of visits; correlates with low `triage_level` (1–2), never used downstream (see §9, §15) |

Longitudinal: yes — 1–8 visits per member (mean 1.63), i.e. repeat encounters
per member_id are real and expected.

### 6.3 `raw_care_history.csv` — 26,289 rows × 4 columns, 0 duplicate `care_id`, 0 missing values, 9,160 unique members (all present in `raw_members.csv`)

| Column | Type | Role | Notes |
|---|---|---|---|
| `care_id` | string | row key | unique |
| `member_id` | string | join key | |
| `visit_date` | string→datetime | temporal | same range as ED, 2025-01-01 → 2026-07-02 |
| `care_type` | string (4 categories) | non-ED encounter type | `PCP` (12,378), `Telehealth` (6,978), `Urgent Care` (4,321), `Care Management` (2,612) |

### 6.4 Joins

- All joins are on `member_id`, `how="left"` from `members` outward.
- `raw_members.csv` is the join anchor; every `member_id` in the ED and care
  files exists in `raw_members.csv` (no orphan records).
- `165` of 10,000 members have **neither** an ED visit nor a care-history
  record (isolated members) — these rows get all utilization features
  filled with 0/NaN→imputed and are still scored.
- Grouped aggregation (`.groupby("member_id")`, `pd.crosstab`) is used before
  merging, so the ED/care tables are collapsed to one row per member before
  the left-join — **this prevents row duplication** in the final feature
  frame (verified: feature matrix row count == `len(members)`).
- Encounters themselves (pre-aggregation) are genuinely longitudinal/repeat
  per member, but the model only ever sees the aggregated, one-row-per-member
  view.

---

## 7. Current Target Definition

- **Column name:** `frequent_ED_user`
- **Defined in:** `backend/train_model.py::build_training_features()`, line
  `features["frequent_ED_user"] = (features["ED_visits_365d"] >= 2).astype(int)`
  (byte-for-byte match to `AVOIDABLE.docx` step 20).
- **Source field:** `ED_visits_365d` — count of ED visits per member within
  365 days of `reference_date`, where `reference_date` is the **single
  global maximum** `visit_date` across the *entire* `raw_ed_visits.csv`
  (2026-07-02), not a per-member enrollment or prediction date.
- **Threshold:** `>= 2` visits in that trailing 365-day window.
- **Prevalence:** 2,017 / 10,000 members positive → **20.17%** positive
  class (moderately imbalanced; `class_weight="balanced"` is used to
  compensate).
- **What it represents:** **(a) ED frequency.** It is a pure utilization
  count threshold. Nothing in the label construction inspects diagnosis,
  triage level, admission status, or any clinical-appropriateness signal
  that would distinguish an *avoidable* ED visit from a necessary one.

**Alignment with UC07:** Does **not** align. UC07 asks for "potentially
avoidable ED utilization." The current target measures *how often* someone
uses the ED, not *whether those visits were avoidable*. A member with two
genuinely emergent visits (e.g., two separate MI events) is labeled
identically to a member with two low-acuity visits that a PCP could have
handled — the dataset even carries `triage_level`, `admitted`, `icu`,
`red_flag`, and `diagnosis` fields that could inform an avoidability
definition, but none of them enter target construction today.

Target definition was **not changed** in this phase.

---

## 8. Feature Inventory

All features below are produced identically by `feature_engineering.py`
(inference) and `train_model.py` (training) — the two implementations are
hand-duplicated, not shared, beyond the two constants imported from
`feature_engineering.py` (`EXPECTED_CARE`, `DROP_BEFORE_MODEL`).

| Feature / group | Source | Time window | Static/Longitudinal | Available at prediction time? | Leakage risk | Used by saved model |
|---|---|---|---|---|---|---|
| `age`, `gender` | members | n/a | static | yes | none | yes |
| `diabetes`,`copd`,`hypertension`,`chf`,`asthma`,`ckd`,`num_chronic_conditions` | members | n/a | static | yes | none | yes |
| `transportation_barrier`,`telehealth_available`,`pcp_distance_miles`,`urgent_care_distance_miles` | members | n/a | static | yes | none | yes |
| `ED_visits_{365,180,90,30}d` | ed_visits | trailing 30/90/180/365d from global `reference_date` | longitudinal, aggregated | technically yes at scoring time, but 365d window **is the target's own source field** | **direct** — explicitly dropped | **no** (in `DROP_BEFORE_MODEL`) |
| `last_ED_date` | ed_visits | n/a (raw date) | longitudinal | intermediate only | n/a (dropped) | no |
| `days_since_last_ED` | ed_visits | n/a (recency) | longitudinal, aggregated | yes, but strongly reflects the same ED-usage history the target is built from | **indirect — confirmed** (top feature at 40.8% importance; median 97 days for positives vs 261 for negatives) | **yes** — NOT in `DROP_BEFORE_MODEL` |
| `diagnosis_*` (14 columns) | ed_visits | **all-time, unwindowed** crosstab (not restricted to the 365d target window) | longitudinal, aggregated | yes, but each column is literally a per-diagnosis ED-visit count | **indirect — confirmed** (combined importance 36.3%; row-sum correlates 0.79 with `ED_visits_365d`, 0.69 with the target) | **yes** — NOT in `DROP_BEFORE_MODEL` |
| `ED_admissions`,`ED_ICU_visits`,`ED_major_procedures`,`ED_red_flags`,`ED_total_cost`,`ED_avg_cost` | ed_visits | all-time aggregate | longitudinal | yes | **direct** — explicitly dropped | **no** (in `DROP_BEFORE_MODEL`) |
| `care_PCP`,`care_Urgent_Care`,`care_Telehealth`,`care_Care_Management` | care_history | all-time crosstab | longitudinal, aggregated | yes | none (non-ED alternative care, independent of target) | yes |
| `days_since_last_care`,`last_care_date` | care_history | n/a | longitudinal | yes (date dropped, days kept) | none | days kept, date dropped |
| `alternative_care_visits`,`total_non_ED_care` | derived from care_* | all-time | derived | yes | none | yes |
| `ED_to_alternative_ratio` | derived from `ED_visits_365d` | same window as target | derived | uses the target's source field | **direct** — explicitly dropped | **no** (in `DROP_BEFORE_MODEL`) |
| `access_burden` | derived (transportation_barrier + distance thresholds) | n/a | derived, static | yes | none | yes |
| `clinical_burden` | derived (sum of 6 chronic flags) | n/a | derived, static | yes | none (redundant with individual flags + `num_chronic_conditions`) | yes |

**Special-attention items requested in the audit brief:**
- **ED count features:** correctly excluded from the model (`DROP_BEFORE_MODEL`).
- **Diagnosis count/crosstab features:** **not excluded** — confirmed
  indirect leakage source (see §9).
- **Recency features:** `days_since_last_ED` **not excluded** — confirmed
  indirect leakage source and the single most important feature in the
  model.
- **Admission/ICU/procedure/red-flag features:** correctly excluded from the
  model, but also **never re-enter** the recommendation or safety logic —
  `red_flag` in particular is collected but goes nowhere (§15).
- **Care history / PCP / urgent-care / telehealth / access variables:**
  correctly retained, no leakage — these are the genuinely prospective,
  actionable signals in the dataset.
- **Chronic-condition and demographic variables:** correctly retained, no
  leakage.

---

## 9. Leakage Assessment (Direct Verification)

Computed directly against the raw CSVs (read-only), not from documentation
claims:

- `corr(diagnosis_all_time_total_count, ED_visits_365d) = 0.794`
- `corr(diagnosis_all_time_total_count, frequent_ED_user target) = 0.687`
- `days_since_last_ED`: median **97 days** for `frequent_ED_user=1` vs
  **261 days** for `frequent_ED_user=0`.
- In the trained model, `days_since_last_ED` alone carries **40.77%** of
  total Random Forest feature importance; the 14 `diagnosis_*` columns
  together carry **36.25%** — combined, **~77%** of everything the model
  relies on is drawn from ED-visit-derived recency/diagnosis-count signals
  that are structurally close to the label itself.

**Conclusion:** the explicit leakage guard in the code
(`DROP_BEFORE_MODEL`) correctly removes the *literal* window-count and
severity/cost aggregates, but misses two indirect-but-strong leakage paths:
`days_since_last_ED` and the unwindowed `diagnosis_*` crosstab. This is a
**confirmed, material finding**, not a hypothesis.

---

## 10. Training / Validation Design

- **Algorithm:** `RandomForestClassifier` (scikit-learn), 400 trees,
  `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`,
  `random_state=42`, `n_jobs=-1`.
- **Preprocessing:** `ColumnTransformer` — numeric features:
  `SimpleImputer(strategy="median")`; categorical features (`gender` only,
  in the current schema): `SimpleImputer(strategy="most_frequent")` →
  `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- **Split:** single random `train_test_split(test_size=0.20,
  random_state=42, stratify=y)`. **Not temporal, not point-in-time** — rows
  are split at random regardless of visit dates, even though every feature
  and the target are built from a fixed global `reference_date`.
- **Class balancing:** via `class_weight="balanced"` only; no resampling
  (SMOTE/undersampling).
- **Metrics reported (`train_model.py`):** accuracy, precision, recall, F1,
  ROC-AUC, full `classification_report`. **Not reported:** confusion matrix
  (present in the original `AVOIDABLE.docx` notebook but dropped from
  `train_model.py`'s import list and output), PR-AUC, calibration
  (Brier score / reliability curve), cross-validation, subgroup
  (age/gender/chronic-condition) validation, or any temporal/out-of-time
  validation.
- **Realism of the evaluation:** because there is one fixed global
  `reference_date` for the whole dataset and a random (not time-based)
  split, the reported metrics describe how well the model separates the two
  classes **within a single historical snapshot** — they do **not**
  estimate how the model would perform predicting forward from an earlier
  point in time to a later outcome window, which is the actual production
  use case. Combined with the leakage in §9, the reported ROC-AUC/F1 numbers
  are very likely optimistic relative to true prospective performance.
- Model was **not retrained** as part of this audit.

---

## 11. Current Model and Artifacts

- **File:** `backend/ed_risk_model.pkl`, `joblib`-serialized Python `dict`:
  `{"model": <sklearn Pipeline>, "feature_columns": [...37 names...],
  "target": "frequent_ED_user", "target_definition": "2 or more ED visits
  within 365 days"}`.
- **Pipeline steps:** `preprocessor` (ColumnTransformer) → `model`
  (RandomForestClassifier). `rf.n_features_in_ = 38` (post-one-hot-expansion
  of `gender`), vs. 37 pre-transform `feature_columns`.
- **No stored metadata for:** training date, Python version used at
  training time, scikit-learn version used at training time, git commit/data
  version the model was trained on, or a model/schema version identifier.
  The only reproducibility control is `backend/requirements.txt` pinning
  `scikit-learn==1.7.2` (all other libraries unpinned).
- **Runtime environment observed during this audit:** Python 3.11.9,
  scikit-learn 1.7.2 (matches the pin) — the artifact loads and predicts
  correctly in the current `.venv`, but nothing in the repo would catch a
  future environment drift before it silently breaks or subtly changes
  `predict_proba` outputs.
- **Thresholds** (`0.35`, `0.60` for Medium/High risk category; `0.50` for
  `predicted_frequent_ED`) are hardcoded in `predict.py`, not stored in the
  artifact and not derived from any calibration analysis — if the model is
  retrained, these cutoffs will silently stop being meaningful unless a
  human remembers to re-derive them.
- **Feature-order assumption:** `predict()` calls `pipeline.predict_proba(X)`
  directly on whatever column order `extract_features()` produces; the
  pipeline's `ColumnTransformer` selects columns by name (not position), so
  this is safe as long as all named columns are present — but there is no
  explicit schema-validation step comparing incoming `X.columns` against the
  stored `feature_columns` list before scoring.

---

## 12. Navigation Logic

- **Function:** `_build_alternative_care_recommendation(row)` in
  `backend/predict.py`.
- **Inputs:** `telehealth_available`, `transportation_barrier`,
  `pcp_distance_miles`, `urgent_care_distance_miles` — all static
  member/access fields from `raw_members.csv`. **Not** an input: the model's
  own `risk_probability`/`risk_category`, nor any ED severity/safety field
  (`red_flag`, `admitted`, `icu`, `triage_level`).
- **Decision rules (deterministic, hardcoded if/elif chain):**
  1. Telehealth available **and** (transport barrier **or** PCP > 10mi **or**
     UC > 10mi) → **Telehealth**
  2. else PCP ≤ 5mi and no transport barrier → **PCP**
  3. else Urgent Care ≤ 5mi → **Urgent Care**
  4. else PCP ≤ 10mi → **PCP**
  5. else telehealth available → **Telehealth**
  6. else → **Urgent Care** (default fallback)
- **Care Management:** **not implemented.** It is neither an output of this
  function nor represented in `types.ts::RecommendedCare` or the frontend's
  `CARE_CLASS` pill styling — even though `care_type == "Care Management"`
  is present in the raw data (2,612 rows) and engineered as
  `care_Care_Management` / included in `total_non_ED_care`.
- **Model coupling:** the recommendation runs independently of the risk
  model's prediction — it is applied via `result.apply(...)` to every row
  regardless of `risk_category`, using only access/geography fields. So the
  navigation suggestion and the ED-risk score are computed by two unrelated
  code paths that happen to be displayed together.
- **Configurability:** all thresholds (5mi, 10mi) are hardcoded Python
  literals, not sourced from config.

---

## 13. Explainability

- **Method:** SHAP `TreeExplainer` on the Random Forest, computed in
  `predict.py::_compute_shap_explanations()`.
- **Granularity:** individual/local only — no global (dataset-level) SHAP
  summary is computed or exposed anywhere.
- **Cost management:** SHAP is skipped entirely on the bulk endpoint
  (`/predict-json`, used for the table view) for performance (~70ms/row →
  minutes on a full census) and computed on demand, one row at a time, via a
  dedicated `/explain-member` endpoint when a user opens a patient's detail
  drawer.
- **Feature naming exposed to the frontend:** raw post-`ColumnTransformer`
  names (e.g. `numeric__diagnosis_Other`, `categorical__gender_F`) — the
  frontend does only a cosmetic `feature.replace(/_/g, " ")`, so labels like
  "numeric  diagnosis Other" can reach the UI rather than a curated
  human-readable name.
- **Human-readability:** a templated summary string
  (`"<feature> increases/reduces risk (<value>)"`) is generated, plus a raw
  JSON array of the top-3 contributions.
- **Clinical-diagnosis risk:** because several of the top SHAP-driving
  features are literally named `diagnosis_*`, a SHAP explanation such as
  "diagnosis Chest pain increases risk (0.041)" is one plausible output —
  this reads uncomfortably close to a clinical statement even though it is
  a statistical association with future *utilization*, not a diagnostic
  claim. There is no rewording/guardrail layer between raw SHAP output and
  the UI to prevent this framing.

---

## 14. Safety Behavior

Full-repository search for emergency/safety-relevant language and logic
(`emergency|ER|ED|red_flag|safety|disclaimer|urgent symptoms|delay|avoid|
instead of ER|before ER|unnecessary|non-emergency`) surfaced matches in
frontend copy, backend docstrings/comments, the raw ED data, and
documentation — **not** in any deterministic backend safety-override logic.

- **Explicit disclaimers (frontend only, static text):**
  - `DisclaimerBanner.tsx` — always visible at the top of the app:
    *"For care navigation only — never a reason to delay care... no output
    here should ever be read as discouraging emergency care. If you or a
    member may be experiencing a medical emergency, call 911 or go to the
    nearest emergency department immediately."*
  - `PatientDetailPanel.tsx` — per-patient recommendation card repeats:
    *"This is a navigation suggestion for care management outreach, not a
    clinical or emergency-necessity determination. It never overrides the
    member's own judgment to seek emergency care."*
- **Deterministic safety layer: does not exist.** There is no backend code
  path that inspects `red_flag`, `triage_level`, `admitted`, or `icu` and
  overrides/suppresses/flags a recommendation. `red_flag` is read from
  `raw_ed_visits.csv`, aggregated into `ED_red_flags` in feature
  engineering, and then unconditionally dropped by `DROP_BEFORE_MODEL` — it
  never reaches the model, the recommendation function, or the API
  response. It is effectively collected and discarded.
- **Gatekeeping risk assessment:** the current recommendation wording
  ("Telehealth is available and removes travel barriers...", "Urgent care is
  ... appropriate for non-emergency needs instead of the ER") is written
  defensively and does not itself instruct a user to avoid the ED. Combined
  with the two static disclaimers, the *UI-level* framing is reasonably
  safe today. The gap is architectural, not textual: safety currently
  depends entirely on two pieces of always-shown static copy, with no
  server-side, data-driven safety check that could catch/flag a
  high-severity case (e.g., a member with recent `red_flag=1` visits) before
  a lower-acuity recommendation is even generated.

---

## 15. FastAPI Backend

- **Entrypoint:** `backend/main.py`, single-file FastAPI app,
  `title="ED Risk Prediction API"`, `version="1.0.0"`.
- **Routes:**
  | Method | Path | Purpose |
  |---|---|---|
  | GET | `/` | root/liveness — `{status, message}` |
  | GET | `/health` | `{status, model_loaded}` — checks only that the `.pkl` file exists on disk, not that it loads/predicts |
  | GET | `/dashboard` | serves `frontend/dist/index.html` if built; 503 otherwise |
  | (mount) | `/dashboard/*` | `StaticFiles` mount of `frontend/dist` (only if the directory exists) |
  | POST | `/predict` | 3 CSV uploads → streamed `.xlsx` |
  | POST | `/predict-json` | 3 CSV uploads → JSON rows (SHAP skipped) |
  | POST | `/explain-member` | `member_id` + 3 CSV uploads → single-row SHAP JSON |
- **Request schemas:** multipart `UploadFile`s, validated post-hoc by
  `_validate_columns()` (checks required column names only — no dtype, row
  count, or value-range validation).
- **Response schemas:** ad-hoc dicts / `StreamingResponse`; no Pydantic
  response models declared, so FastAPI's OpenAPI schema for responses is
  effectively untyped.
- **Error handling:** CSV parse errors → 400; missing columns → 422; missing
  model file → 503; feature-engineering exceptions → 500; prediction
  exceptions → 500; SHAP exceptions → 500 (member-not-found → 404). All
  reasonably mapped, but error bodies pass raw exception text
  (`str(exc)`) back to the client, which could leak internal details.
- **Model loading:** `joblib.load(MODEL_PATH)` is called **inside** every
  `predict()`/`explain_member()` call — the model is deserialized from disk
  on every single request rather than cached once at app startup. This is a
  functional-behavior observation only (not changed in this phase), but is
  relevant to production readiness/performance.
- **Config/env vars:** only the frontend reads an env var
  (`VITE_API_URL`); the backend has no environment-variable-driven
  configuration at all (CORS origins, host/port, model path are all
  Python-literal constants).
- **CORS:** `allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]`
  — fully open, no allow-list, no credentials handling considered.
- **Logging / request IDs / model version exposure:** none present. No
  structured logging, no correlation/request IDs, no endpoint exposes the
  loaded model's `target_definition`/version metadata even though it is
  already stored in the pickle.
- **Startup validation:** none — the app will start even if the model file
  is missing or corrupt; failure only surfaces on first `/predict*` call
  (`/health` only checks file existence, not loadability).

Backend was **not modified** during this audit.

---

## 16. Frontend Architecture

- **Framework:** React 19.2 + TypeScript + Vite 8, no router (single page),
  `oxlint` for linting.
- **Pages/components:** `App.tsx` orchestrates `Header`, `DisclaimerBanner`,
  `UploadPanel` (3 drag/drop CSV zones with basic `.csv` extension check),
  `StatCards`, `RiskDistributionChart` (CSS-bar chart, no charting library),
  `PatientTable`, `PatientDetailPanel` (drawer: risk, recommendation +
  per-row disclaimer, on-demand SHAP bars, patient snapshot, chronic
  condition chips), `EmptyState`, `ErrorBanner`.
- **API URL handling:** `frontend/src/api.ts` reads
  `import.meta.env.VITE_API_URL`, falling back to the hardcoded literal
  `http://127.0.0.1:8001` if unset — same fallback is hardcoded again in the
  legacy static dashboard (`frontend_legacy/index.html`).
- **Upload flow:** three independent dropzones (members / ED visits / care);
  "Run risk analysis" is disabled until all three are selected; original
  `File` objects are kept in React state so they can be re-submitted later
  for the on-demand `/explain-member` call.
- **Prediction flow:** `runPrediction()` → `POST /predict-json`; on success
  renders `StatCards` + `RiskDistributionChart` + `PatientTable`; on failure
  shows `ErrorBanner`.
- **Risk visualization:** simple stat tiles + a bar-per-category chart, both
  purely client-side aggregation of the returned rows (no separate backend
  aggregate endpoint).
- **Recommendation display:** `CarePill` shows the backend's
  `recommended_alternative_care` string; the `CARE_CLASS` lookup table only
  knows `PCP` / `Urgent Care` / `Telehealth` (falls back to the PCP pill
  style for any unrecognized value, which would silently mis-style a
  hypothetical "Care Management" value if the backend ever returned one).
- **Explainability display:** SHAP bars with a legend distinguishing
  "increases"/"reduces" risk, fetched lazily per selected patient.
- **Safety disclaimer:** present twice (global banner + per-patient card,
  see §14) — no duplicated *business logic* (risk scoring, recommendation
  rules) exists client-side; the frontend is a thin display layer over the
  API's output.
- **Error/loading states:** `loading` boolean gates the run button/spinner;
  `ErrorBanner` is dismissible; the detail drawer has its own
  loading/error state for the SHAP fetch.
- **Hard-coded URLs:** the `127.0.0.1:8001` fallback in both `api.ts` and
  `frontend_legacy/index.html`.
- **Production configuration readiness:** no `.env` committed (only
  `.env.example`, good practice); no build-time validation that
  `VITE_API_URL` is set for a non-local deployment; no error boundary
  component; `frontend_legacy/index.html` is dead code still present in the
  repo pointing at the same hardcoded local URL.

Frontend was **not modified** during this audit.

---

## 17. Testing Status

**No tests exist anywhere in the repository** — no `tests/` directory, no
`pytest`/`unittest` files, no `jest`/`vitest` config or spec files, no API
contract tests, no fixtures. Categorized gap (all rated MISSING):

| Category | Status |
|---|---|
| Unit (feature engineering, target, recommendation rules) | Missing |
| Integration (train → artifact → predict round-trip) | Missing |
| API (FastAPI route/contract tests) | Missing |
| Model (metric-regression / performance-floor tests) | Missing |
| Feature-engineering (train/inference parity tests) | Missing |
| Navigation (recommendation-rule tests) | Missing |
| Safety (red-flag / disclaimer regression tests) | Missing |
| Frontend (component/unit tests) | Missing |
| End-to-end | Missing |

No tests were created in this phase (per scope).

---

## 18. Docker Readiness

No `Dockerfile`, `.dockerignore`, `docker-compose.yml`, or any container
entrypoint script exists anywhere in the repository (verified by
repository-wide search). Consequently: no non-root execution policy, no
container healthcheck, no defined strategy for baking the model artifact
into an image vs. mounting it, and no deterministic/locked dependency
installation (`requirements.txt` mostly unpinned, see §11 and §20).
**Not assessed further because there is nothing to assess** — this is a
from-scratch gap for the next phase, not a partial implementation.

---

## 19. Azure Readiness

No Azure configuration of any kind exists in the repository — no Bicep/ARM
templates, no `azure-pipelines.yml`, no Container Apps/App Service config,
no `.env` scaffolding for Azure-specific variables, no managed-identity or
Key Vault references, and no `.github/workflows` (or any other CI/CD
definitions). This is a from-scratch gap.

---

## 20. Security / Privacy Engineering Observations

(Engineering-level observations only — no regulatory/compliance claim made.)

- **PII/PHI surface:** the three datasets carry `member_id`, `age`,
  `gender`, and clinical/chronic-condition flags — no name, DOB, SSN, or
  address fields are present in the current schema, which limits (but does
  not eliminate) direct-identifiability risk. `member_id` + demographics +
  diagnosis history is still sensitive health data and flows unredacted
  through the API response, the generated `.xlsx`/JSON, and the SHAP JSON.
- **No authentication/authorization** on any endpoint — anyone who can reach
  the API can upload arbitrary matching CSVs and receive predictions/SHAP.
- **CORS is fully open** (`*` origins/methods/headers) — acceptable for
  local development, a real exposure if deployed as-is.
- **No upload validation beyond column-name presence** — no file-size cap,
  no row-count cap, no content-type enforcement beyond a client-side `.csv`
  extension check (trivially bypassable) — large/malformed uploads could
  drive expensive `pd.crosstab`/one-hot operations (resource-exhaustion /
  DoS surface).
- **No rate limiting** on any endpoint.
- **File handling:** uploads are read into memory (`io.BytesIO`) and never
  written to disk — this is good practice and avoids temp-file leakage, but
  also means large uploads are held fully in process memory.
- **No secrets found in the repository** (verified by pattern search across
  all tracked source/text files) — no committed `.env`, no hardcoded API
  keys/passwords/tokens. `frontend/.env.example` documents the one
  configurable value (`VITE_API_URL`) without a real secret.
- **No structured audit logging** — no record of who uploaded what, when,
  or which member records were scored.
- **Static dashboard mount (`/dashboard`) is unauthenticated**, same as
  every other route.
- **Dependency pinning:** only `scikit-learn==1.7.2` is pinned in
  `backend/requirements.txt`; `fastapi`, `uvicorn`, `pandas`, `numpy`,
  `joblib`, `openpyxl`, `shap` are all unpinned, which risks silent
  behavior drift on a fresh `pip install`.
- **Debug behavior:** raw exception messages (`str(exc)`) are returned in
  HTTP error bodies, which could leak internal file paths/stack detail to a
  client.

---

## 21. Documentation Status

Before this phase, the only documentation-like artifacts were
`AVOIDABLE.docx` (the original notebook spec, still authoritative for the
feature-engineering/training logic), its plaintext dump `docx_out.txt`, an
empty stray file `docx_content.txt`, and an ad-hoc `csv_info.txt` shape
dump — no `docs/` directory, no README beyond `frontend/README.md` (dev/build
instructions only, no architecture or data documentation), and no changelog
or decision log. This phase creates `docs/01_PROJECT_BASELINE.md`,
`docs/CHANGELOG.md`, and `docs/DECISION_LOG.md` to close that gap.

---

## 22. Current Strengths

- Feature-engineering and training logic is a faithful, traceable port of
  the original `AVOIDABLE.docx` spec — the two Python implementations match
  it step-for-step, which makes auditing straightforward.
- The frontend already carries genuinely good, prominent, twice-repeated
  safety disclaimer language that correctly avoids discouraging emergency
  care.
- Explicit (if incomplete) leakage-avoidance already exists
  (`DROP_BEFORE_MODEL`) — the team clearly reasoned about this problem, they
  just didn't cover every leakage path.
- SHAP explainability is wired end-to-end with a sensible performance
  tradeoff (skip on bulk, compute on demand per patient).
- Clean separation between raw uploaded files (never persisted) and
  in-memory processing — good baseline data-handling hygiene.
- The care-history dataset already contains a `Care Management` category,
  meaning the data foundation for a 4th navigation option already exists —
  it's a wiring gap, not a data gap.
- The `scikit-learn` version is pinned, which is the one dependency most
  likely to break the pickled model artifact if it drifted.

---

## 23. Critical Gaps

1. **Target does not represent avoidable ED utilization** — it is a raw
   visit-frequency threshold with no clinical/avoidability signal (§7).
2. **Confirmed indirect leakage** via `days_since_last_ED` (40.8% importance)
   and unwindowed `diagnosis_*` crosstab (36.3% importance) — together ~77%
   of model reliance (§8, §9).
3. **No temporal/point-in-time validation** — single random split against a
   single global reference date; reported metrics likely overstate real
   prospective performance (§10).
4. **Care Management is not a reachable recommendation**, despite existing
   in the source data and feature set (§12).
5. **No deterministic safety layer** — `red_flag`/`triage_level`/`admitted`/
   `icu` are collected but never used anywhere downstream of feature
   engineering; safety currently relies entirely on static frontend text
   (§14).
6. **Zero automated tests** anywhere in the repository (§17).
7. **No Docker and no Azure/CI-CD readiness** whatsoever (§18, §19).

---

## 24. Medium-Priority Gaps

- No model versioning/training-date/environment metadata stored in the
  artifact; thresholds (0.35/0.60/0.50) hardcoded separately from the model
  and not derived from calibration.
- No calibration assessment, no PR-AUC, no confusion matrix in the current
  `train_model.py` (present in the original notebook but dropped),
  no cross-validation, no subgroup (age/gender/chronic-condition) validation.
- Model is loaded from disk (`joblib.load`) on every API request instead of
  once at startup.
- CORS fully open; no authentication on any endpoint; no upload size/row
  limits; no rate limiting.
- `feature_engineering.py` and `train_model.py` duplicate the feature-build
  logic by hand rather than sharing one function — a drift risk every time
  either is edited.
- Risk scoring, navigation, explainability, and safety are all
  intermingled inside `predict.py`/`main.py` with no module/interface
  boundaries (explicitly flagged as a future-architecture concern in the
  brief).
- Dependency pinning is inconsistent (only `scikit-learn` pinned).

---

## 25. Low-Priority Gaps

- `frontend_legacy/index.html` is dead code still present in the repo,
  pointing at a hardcoded local API URL.
- Stray root-level generated/scratch files (`ed_risk_predictions.xlsx`,
  `unseen_patient_risk_results (1).csv`, `csv_info.txt`, empty
  `docx_content.txt`, `zip.zip` full-repo backup) are not organized into an
  `outputs/`/`scratch/` location and are not `.gitignore`d.
  `.gitignore` at the repo root currently ignores only `.venv/`.
  `frontend/.gitignore` was not separately deep-audited beyond confirming
  its presence.
  `num_chronic_conditions` is fully redundant with the 6 individual
  chronic-condition flags (harmless but unnecessary duplication).
- SHAP feature names surfaced via the API use raw
  `ColumnTransformer`-transformed names (e.g. `numeric__diagnosis_Other`)
  rather than a curated display-name mapping.
- No `/health`-distinct readiness probe that actually attempts to load the
  model (current `/health` only checks file existence).

---

## 26. Recommended Next Phase

Phase 2 should be scoped as **design-and-decision work, not implementation**,
covering (in priority order): (1) defining an avoidability-aware target in
consultation with a clinical/business stakeholder — using `diagnosis`,
`triage_level`, `admitted`, and `red_flag` as *candidate inputs* to that
definition rather than as model features; (2) closing the two confirmed
leakage paths (`days_since_last_ED`, `diagnosis_*`) or explicitly justifying
their retention if a redefined target still calls for them; (3) designing a
point-in-time / temporal validation strategy; (4) designing the
architectural split between risk detection, care navigation, and
safety/policy (including wiring `red_flag`/severity into a real safety
gate, and making Care Management a reachable recommendation); (5) only then
moving to retraining, testing, Docker, and Azure packaging. This phase
should **not** begin implementation — it is design/decision work to be
ratified before any code, data, or model changes occur.

---

## 27. Statement of No Functional Change

**No functional code, feature-engineering logic, model logic, API
behavior, or frontend behavior was changed in Phase 1.** No dataset was
modified, renamed, deleted, or regenerated. No model was retrained. The
only filesystem changes made in this phase are the creation of
`docs/01_PROJECT_BASELINE.md`, `docs/CHANGELOG.md`, and
`docs/DECISION_LOG.md`.

---

## 28. Baseline Status Table

| Area | Status | Basis |
|---|:---:|---|
| Dataset foundation | 🟢 GREEN | Clean joins, no duplicates, no missing values, all members cross-referenced correctly |
| UC07 target definition | 🔴 RED | Frequency-based (`ED_visits_365d>=2`), not avoidability-based (§7) |
| Label methodology | 🔴 RED | No clinical/avoidability signal used; single hardcoded threshold |
| Point-in-time design | 🔴 RED | Single global reference date + random split, not temporal (§10) |
| Feature engineering | 🟡 YELLOW | Sound structure, but hand-duplicated between train/inference scripts |
| Leakage prevention | 🔴 RED | Direct leakage correctly dropped; indirect leakage (recency + diagnosis crosstab, ~77% of model importance) not addressed (§9) |
| Model baseline | 🟡 YELLOW | Reasonable RF baseline, but no comparison model was ever tried |
| Model comparison | 🔴 RED | No alternative algorithms evaluated |
| Calibration | 🔴 RED | Not assessed at all |
| Threshold design | 🟡 YELLOW | Fixed 0.35/0.60/0.50 cutoffs work for a demo but are not calibration-derived or stored with the artifact |
| Navigation | 🟡 YELLOW | Deterministic and reasonable rules, but decoupled from risk score and severity |
| Care Management support | 🔴 RED | Data exists; not a reachable recommendation output (§12) |
| Multi-agent architecture | 🔴 RED | Does not exist; risk/navigation/safety/explainability are intermingled in 2 files |
| Safety | 🟡 YELLOW | Good static disclaimers; no deterministic red-flag/severity safety layer (§14) |
| Explainability | 🟡 YELLOW | SHAP wired end-to-end and performance-conscious; raw feature names exposed, no global view |
| Bias validation | 🔴 RED | No subgroup validation performed |
| FastAPI | 🟡 YELLOW | Functionally complete; no auth, open CORS, model reloaded per request, no structured logging (§15) |
| Frontend | 🟢 GREEN | Complete, well-organized, good safety UX; minor hardcoded-URL/dead-code issues |
| Automated testing | 🔴 RED | Zero tests anywhere (§17) |
| Model governance/versioning | 🔴 RED | No version/date/env metadata in the artifact; only sklearn pinned |
| Docker | 🔴 RED | Does not exist (§18) |
| Azure readiness | 🔴 RED | Does not exist (§19) |
| Monitoring | 🔴 RED | No logging, metrics, or observability of any kind |
| Security/privacy engineering | 🟡 YELLOW | No secrets/PII-fields found, no on-disk persistence of uploads; but open CORS, no auth, no upload limits (§20) |
| Documentation | 🟢 GREEN (as of this phase) | `docs/` now exists with baseline, changelog, and decision log |
