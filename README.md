Avoidable Emergency Department Utilization Navigator

> AI-powered care-management system for identifying patterns of potentially avoidable Emergency Department (ED) utilization, estimating future ED utilization risk, explaining predictions, recommending lower-acuity care-navigation options, and supporting safe member communication.

---

## 📌 Overview

 Avoidable Emergency Department Utilization Navigator** is a healthcare care-management application designed to help identify members who may benefit from proactive care navigation.

Emergency Department visits are expensive and, in some situations, a member's needs may potentially be addressed through lower-acuity alternatives such as:

- Primary Care
- Urgent Care
- Telehealth
- Care Management

However, emergency care must never be discouraged when a genuine emergency may be present.

For this reason, UC07 is designed as a **care-navigation and pattern-detection system — not an emergency-care gatekeeping system**.

The system combines:

- Point-in-time feature engineering
- Machine-learning risk prediction
- Risk stratification
- SHAP-based model explainability
- Rule-based care-navigation recommendations
- Safety guardrails
- Generative AI explanations
- Interactive population analytics
- Member-level investigation
- PDF report generation
- Email-based member communication

---

# 🎯 Use Case

### UC07 — Avoidable Emergency Department Utilization Navigator

**Domain:** Care Management

### Problem

Some Emergency Department visits may potentially have been addressed through a lower-acuity care setting such as a same-day clinic, urgent care, telehealth consultation, primary-care visit, or care-management intervention.

The objective is to identify utilization patterns suggesting opportunities for proactive navigation while ensuring that the system **never discourages legitimate emergency care**.

### Core Goal

> Identify members with elevated modeled ED-utilization risk and support proactive care navigation toward appropriate lower-acuity alternatives when safe, without blocking or discouraging emergency care.

---

# 🚨 Important Safety Disclaimer

This project is a **care-management decision-support prototype**.

It is **not**:

- a diagnostic system,
- an emergency triage system,
- a medical-necessity determination system,
- a replacement for clinical judgment,
- or a mechanism for denying or blocking emergency care.

The ML risk score represents modeled patterns of future ED utilization.

It does **not** determine whether an individual ED visit is medically necessary.

When safety indicators suggest possible emergency or high-acuity conditions, safety logic takes priority over proactive navigation.

---

# 🧠 System Architecture

The application follows an end-to-end care-management pipeline:

```text
Historical Data
      │
      ▼
Point-in-Time Feature Engineering
      │
      ▼
Machine-Learning Risk Model
      │
      ▼
Risk Probability + Risk Tier
      │
      ├──────────────► SHAP Explainability
      │
      ▼
Care Navigation Agent
      │
      ▼
Safety Policy Agent
      │
      ▼
Authoritative UC07 Decision
      │
      ├──────────────► GenAI Explanation
      │                 Groq
      │                   ↓ fallback
      │                 Ollama
      │                   ↓ fallback
      │                 Deterministic Explanation
      │
      ▼
Interactive Care-Manager Dashboard
      │
      ├── Population Analytics
      ├── Member Prioritization
      ├── Member Details
      ├── Why Flagged / SHAP
      ├── AI Explanation
      ├── Safety Review
      │
      ▼
Member Communication
      │
      ├── PDF Report
      └── Email + PDF Attachment
```

---

# 🗂️ Data Sources

The system operates primarily on three historical datasets.

## 1. Members

Contains member-level demographic and clinical attributes.

Example information may include:

- Member ID
- Demographics
- Chronic-condition indicators
- Access-related attributes

---

## 2. ED Visits

Contains historical Emergency Department utilization.

Example fields may include:

- Member ID
- Visit history
- Diagnosis information
- Triage information
- Admission / ICU indicators
- Procedures
- Cost/utilization information
- Emergency indicators

---

## 3. Care History

Contains historical lower-acuity and care-management interactions.

Examples:

- Primary-care utilization
- Urgent-care utilization
- Telehealth utilization
- Care-management activity
- Access information

---

## Current Safety Context

The application can additionally accept **current safety context**.

This information is used by the Safety Agent and is separate from historical risk-model inputs.

Possible fields include:

- `member_id`
- `red_flag`
- `icu`
- `admitted`
- `major_procedure`
- `triage_level`

Safety context is used to prevent care-navigation recommendations from being interpreted as a reason to delay emergency care.

---

# ⚙️ Feature Engineering

Historical events are aggregated into member-level point-in-time features.

Examples include:

- ED visits over 30 / 90 / 180 / 365-day windows
- Recent ED utilization
- ED visit frequency
- Potentially avoidable ED patterns
- Admissions / ICU utilization
- Primary-care utilization
- Urgent-care utilization
- Telehealth utilization
- Care-management engagement
- Chronic-condition burden
- Access-related features
- Alternative-care availability

Point-in-time construction is important to reduce temporal leakage between historical features and future outcomes.

---

# 🤖 Machine-Learning Layer

During the training phase, multiple candidate models are evaluated.

Candidate algorithms include:

- Logistic Regression
- Gradient Boosting / XGBoost
- LightGBM
- Random Forest

The selected model produces a probability representing modeled future ED-utilization risk.

```text
Member Features
      │
      ▼
Risk Model
      │
      ▼
Risk Probability
      │
      ▼
Risk Tier
```

---

# 📊 Risk Stratification

The model output is converted into an interpretable risk tier.

The current configured thresholds are:

| Risk Tier | Probability |
|---|---:|
| 🟢 LOW | `< 0.15` |
| 🟠 MODERATE | `0.15 – 0.35` |
| 🔴 HIGH | `> 0.35` |

Thresholds are controlled by the backend/model configuration.

The frontend should display the authoritative backend result and must not independently recreate clinical or decision logic.

---

# 🔍 Model Explainability — SHAP

UC07 uses **SHAP-based explainability** to help care managers understand which model features contributed to a member's estimated risk.

The interface separates factors into:

### Factors increasing the estimate

Examples may include:

- Recent ED utilization
- Repeated potentially avoidable ED utilization
- Clinical burden
- Elevated historical utilization

### Factors decreasing the estimate

Examples may include:

- Primary-care engagement
- Telehealth availability
- Alternative-care access

The member workspace displays these contributions using a diverging visualization.

> SHAP values explain model behavior. They do not establish clinical causation.

---

# 🧭 Care Navigation

After risk estimation, the system evaluates potential lower-acuity care-navigation opportunities.

Possible destinations include:

| Destination | Purpose |
|---|---|
| 🔵 Primary Care | Ongoing or routine outpatient care |
| 🟠 Urgent Care | Same-day lower-acuity evaluation |
| 🟦 Telehealth | Remote care when appropriate |
| 🟣 Care Management | Outreach / coordination / follow-up |
| ⚪ No Proactive Navigation | No navigation action recommended |

Navigation recommendations are intended for **future non-emergency care-management opportunities**.

They must never be interpreted as instructions to avoid emergency care.

---

# 🛡️ Safety Guardrails

Safety is a first-class component of UC07.

The Safety Agent evaluates available safety context independently of the ML risk prediction.

The system uses three safety states:

### 🟢 CLEAR

No supplied safety indicator currently prevents proactive navigation.

### 🟠 CAUTION

Safety information is incomplete or requires additional care-manager consideration.

### 🔴 OVERRIDE

Safety indicators prevent the system from promoting lower-acuity navigation.

When an override is present, emergency/clinical safety takes precedence.

```text
Risk Prediction
      │
      ▼
Navigation Candidate
      │
      ▼
Safety Agent
      │
      ├── CLEAR
      ├── CAUTION
      └── OVERRIDE
              │
              ▼
      Do not promote lower-acuity
      navigation over emergency care
```

---

# 🧠 Generative AI Explanation Layer

UC07 includes a GenAI explanation layer that converts the **existing authoritative decision** into a readable care-manager explanation.

The LLM does **not** make the risk, navigation, or safety decision.

It only explains existing structured outputs.

The provider chain is:

```text
Authoritative Decision
        │
        ▼
      Groq
        │
   failure/rejection
        ▼
      Ollama
        │
   failure/rejection
        ▼
Deterministic Explanation
```

---

## Groq

Groq is the primary configured GenAI provider.

Example model configuration:

```env
GENAI_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile
```

Model availability may change over time. Verify the currently supported Groq models before deployment.

---

## Ollama

Ollama provides an optional local fallback.

Example:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
```

---

## Deterministic Fallback

If external/local LLM generation fails or an explanation does not pass validation, UC07 can return a deterministic explanation.

This keeps explanation functionality available without allowing an LLM failure to affect the authoritative decision.

---

# 🔐 GenAI Safety Design

Generated explanations are validated against structured decision outputs.

The system protects against contradictions involving:

- Risk tier
- Navigation destination
- Safety state
- Safety override behavior

The LLM is therefore treated as an **explanation layer**, not a decision authority.

---

# ⚡ AI Explanation Caching

The frontend caches generated explanations for the current decision state.

This means:

1. A care manager opens **AI Explanation**.
2. The explanation is generated.
3. The user navigates to another tab.
4. The user returns to **AI Explanation**.
5. The previously generated explanation is reused.

The LLM is not unnecessarily called again for the same unchanged decision.

The cache is invalidated when the underlying decision changes.

---

# 🖥️ Frontend

The frontend is a healthcare-oriented care-management dashboard.

The primary workflow is:

```text
Upload Data
   ↓
Run Risk Analysis
   ↓
Population Overview
   ↓
Interactive Analytics
   ↓
Member Prioritization
   ↓
Member Workspace
   ↓
Why Flagged
   ↓
AI Explanation
   ↓
Safety Review
   ↓
Communication
```

---

# 📈 Population Analytics

The population dashboard provides metrics such as:

- Total members
- High-risk members
- Moderate-risk members
- Navigation opportunities
- Safety caution
- Safety override

Interactive visualizations include:

### Risk Distribution

Donut chart showing:

- Low
- Moderate
- High

### Navigation Distribution

Displays:

- Primary Care
- Urgent Care
- Telehealth
- Care Management
- No Proactive Navigation

### Safety Distribution

Displays:

- Clear
- Caution
- Override

### Risk Probability Distribution

Histogram showing the distribution of predicted probabilities and configured risk thresholds.

---

# 🎛️ Cross-Filtering

Analytics charts interact with the member table.

For example:

```text
Click HIGH RISK
      ↓
Member table filters to HIGH-risk members
```

Filters remain synchronized across:

- Charts
- Dropdowns
- Active filter chips
- Member table

---

# 👥 Member Prioritization

The member table supports operational care-management review.

Typical columns include:

- Member ID
- Risk probability
- Risk tier
- Navigation recommendation
- Safety state
- Member action

The interface displays **15 members per page**.

Filtering is available by:

- Member ID
- Risk
- Navigation
- Safety

---

# 👤 Member Workspace

Selecting a member opens a detailed member workspace.

Tabs include:

```text
Overview
Why Flagged
AI Explanation
Current Safety
Communication
```

The workspace keeps the member's core decision visible:

- Risk probability
- Risk tier
- Navigation destination
- Safety state

---

# 📄 Member PDF Reports

Care managers can generate a structured PDF report containing the current UC07 decision.

The PDF reporting layer is intended to provide a professional care-management document rather than a raw API response.

Reports can contain:

- Report metadata
- Member identifier
- Risk assessment
- Navigation recommendation
- Safety state
- Important explanatory information
- Care-navigation context
- Required disclaimers

PDF generation is separate from model inference and does not alter the authoritative decision.

---

# ✉️ Member Email Communication

UC07 supports sending a care-navigation report to a member.

The workflow is:

```text
Member Workspace
      │
      ▼
Communication
      │
      ▼
Compose Email
      │
      ├── Recipient
      ├── Subject
      ├── Editable Message
      └── PDF Attachment
      │
      ▼
Review & Send
      │
      ▼
Confirm Send
      │
      ▼
SMTP Provider
```

The email composer allows care managers to review/edit communication before sending.

---

# 📬 SMTP Email

SMTP is currently supported for email delivery.

Example Gmail configuration:

```env
EMAIL_ENABLED=true
EMAIL_PROVIDER=smtp

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=YOUR_APP_PASSWORD
SMTP_FROM_EMAIL=your-email@example.com
SMTP_FROM_NAME=ED Navigator System
SMTP_USE_TLS=true

EMAIL_TIMEOUT_SECONDS=30
```

For Gmail, use an **App Password** rather than the account's normal password when applicable.

Never commit SMTP credentials.

---

# 📧 Email Diagnostics

The email service includes safe diagnostics for failures such as:

- Connection failure
- Timeout
- TLS failure
- Authentication failure
- Sender rejection
- Recipient rejection
- Message rejection
- Rate limiting
- Temporary provider failure
- Permanent provider failure

Sensitive credentials and message contents should never be written to logs.

---

# 📝 Communication Audit

Communication operations can produce audit events for actions such as:

```text
PDF_GENERATED
EMAIL_SENT
EMAIL_FAILED
```

Audit information can include:

- Event ID
- Member ID
- Report ID
- Masked recipient
- Provider
- Result
- Timestamp

Sensitive email credentials must never appear in audit logs.

---

# 🎨 UI / UX Design

The frontend uses a healthcare-oriented design language.

Semantic colors communicate meaning.

### Risk

- 🟢 Green — Low
- 🟠 Amber — Moderate
- 🔴 Red — High

### Safety

- 🟢 Green — Clear
- 🟠 Amber — Caution
- 🔴 Red — Override

### Navigation

- 🔵 Blue — Primary Care
- 🟠 Orange — Urgent Care
- 🩵 Cyan / Teal — Telehealth
- 🟣 Violet — Care Management
- ⚪ Slate — No Proactive Navigation

### Explainability

Warm colors indicate factors increasing the modeled estimate.

Cool colors indicate factors decreasing the modeled estimate.

Both light and dark themes are supported.

---

# 🏗️ Technology Stack

## Backend

- Python
- FastAPI
- Uvicorn
- Pandas / NumPy
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- Groq API integration
- Ollama integration
- SMTP
- ReportLab / PDF reporting utilities
- Pytest

## Frontend

- React
- TypeScript
- Vite
- Recharts
- CSS design-token system
- Component-level testing

---

# 📁 Repository Structure

A representative structure is:

```text
UC07/
│
├── backend/
│   │
│   ├── agents/
│   │   ├── risk_detection.py
│   │   ├── care_navigation.py
│   │   ├── safety_policy.py
│   │   ├── model_explainability.py
│   │   ├── genai_explanation.py
│   │   └── orchestrator.py
│   │
│   ├── modeling/
│   │   ├── preprocessing.py
│   │   ├── feature_spec.py
│   │   ├── risk_tiers.py
│   │   ├── train.py
│   │   └── metrics.py
│   │
│   ├── pit/
│   │   ├── features.py
│   │   ├── windows.py
│   │   ├── validation.py
│   │   └── target.py
│   │
│   ├── services/
│   │   └── email_service.py
│   │
│   ├── tests/
│   │
│   ├── main.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── .env.example
│   └── .env                 # DO NOT COMMIT
│
├── frontend/
│   │
│   ├── src/
│   │   ├── uc07/
│   │   ├── __tests__/
│   │   └── ...
│   │
│   ├── package.json
│   └── package-lock.json
│
├── artifacts/
│
├── docs/
│   ├── DECISION_LOG.md
│   ├── CHANGELOG.md
│   └── ...
│
├── .gitignore
└── README.md
```

The exact repository structure may evolve as the project develops.

---

# 🚀 Local Development Setup

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

---

## 2. Create Python virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install backend dependencies

```powershell
pip install -r backend/requirements.txt
```

---

# 🔑 Environment Configuration

Create:

```text
backend/.env
```

Example:

```env
# ==========================================================
# GENAI
# ==========================================================

GENAI_ENABLED=true
GENAI_PROVIDER=groq

GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=YOUR_SUPPORTED_GROQ_MODEL

GENAI_TIMEOUT_SECONDS=30


# ==========================================================
# OPTIONAL OLLAMA FALLBACK
# ==========================================================

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b


# ==========================================================
# EMAIL
# ==========================================================

EMAIL_ENABLED=true
EMAIL_PROVIDER=smtp

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=YOUR_EMAIL
SMTP_PASSWORD=YOUR_APP_PASSWORD
SMTP_FROM_EMAIL=YOUR_EMAIL
SMTP_FROM_NAME=ED Navigator System
SMTP_USE_TLS=true

EMAIL_TIMEOUT_SECONDS=30
```

Never commit this file.

---

# 🔒 Secret Management

Your `.gitignore` should include:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
.venv/
venv/

# Secrets
.env
backend/.env
frontend/.env

# Keep examples
!.env.example
!backend/.env.example

# Node
node_modules/
frontend/node_modules/

# Builds
dist/
build/

# Caches
.pytest_cache/

# Logs
*.log

# OS / Editor
.DS_Store
Thumbs.db
.vscode/
```

If a secret has ever been committed to Git, adding it to `.gitignore` afterward is **not sufficient**.

Rotate the exposed credential and remove it from repository history if necessary.

---

# ▶️ Start the Backend

From the repository root:

```powershell
cd backend
python -m uvicorn main:app --reload --port 8001
```

The backend should become available at:

```text
http://127.0.0.1:8001
```

Swagger/OpenAPI documentation is normally available at:

```text
http://127.0.0.1:8001/docs
```

---

# 🖥️ Start the Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL displayed by Vite in your browser.

---

# 🦙 Optional Local Ollama Setup

If Ollama fallback is required, install Ollama and obtain the configured model.

Example:

```bash
ollama pull qwen3:8b
```

Verify:

```bash
ollama list
```

The configured Ollama service should match:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

---

# 🧪 Testing

## Backend

From the backend directory:

```powershell
pytest tests/
```

Or:

```powershell
python -m pytest tests/
```

---

## Frontend

```bash
cd frontend
npm test
```

Run lint:

```bash
npm run lint
```

Create a production build:

```bash
npm run build
```

---

# 🔌 Major API Responsibilities

The application exposes backend endpoints for operations including:

### Health

```text
GET /health
```

Used to verify backend/service status.

### Decision

```text
POST /uc07/decide
```

Produces the authoritative UC07 decision.

### Explanation

```text
POST /uc07/explain
```

Produces an explanation of the existing decision using:

```text
Groq
  ↓
Ollama
  ↓
Deterministic fallback
```

### Report

```text
POST /uc07/report
```

Generates the member PDF report.

### Email

```text
POST /uc07/email
```

Sends member communication with the generated report.

Refer to the application's OpenAPI/Swagger documentation for the exact current request and response schemas.

---

# 🔄 End-to-End Workflow

```text
1. Care manager uploads historical data
                │
                ▼
2. Backend validates inputs
                │
                ▼
3. Point-in-time features are generated
                │
                ▼
4. ML model estimates ED-utilization risk
                │
                ▼
5. Member receives LOW / MODERATE / HIGH tier
                │
                ▼
6. SHAP explains model contribution patterns
                │
                ▼
7. Navigation Agent evaluates lower-acuity opportunity
                │
                ▼
8. Safety Agent evaluates safety context
                │
                ▼
9. Authoritative UC07 decision is produced
                │
                ▼
10. Care manager reviews population analytics
                │
                ▼
11. Care manager opens member workspace
                │
                ├── Overview
                ├── Why Flagged
                ├── AI Explanation
                ├── Current Safety
                └── Communication
                │
                ▼
12. Optional GenAI explanation
                │
                ▼
13. Care manager may generate PDF
                │
                ▼
14. Care manager may edit communication
                │
                ▼
15. PDF can be emailed to member
```

---

# 🧩 Design Principles

UC07 follows several important architectural principles.

## 1. ML predicts — it does not determine emergency necessity

The risk model identifies utilization patterns.

It does not decide whether emergency care is medically necessary.

---

## 2. Safety is independent

Safety logic is separated from risk prediction.

A high or low risk score cannot bypass safety rules.

---

## 3. GenAI explains — it does not decide

The LLM receives an existing structured decision.

It cannot independently modify:

- Risk tier
- Navigation destination
- Safety state

---

## 4. Deterministic fallback is valid behavior

LLM availability is not required for the core decision pipeline.

If GenAI fails, the application can still provide a deterministic explanation.

---

## 5. The frontend is not a decision engine

The frontend displays authoritative backend outputs.

Clinical/safety logic should remain server-side.

---

## 6. Human review remains central

UC07 supports care managers.

It does not replace professional clinical or care-management judgment.

---

# 🛡️ Privacy & Security Considerations

Before production deployment:

- Never commit API keys.
- Never commit SMTP passwords.
- Use managed secret storage.
- Use HTTPS.
- Apply authentication and authorization.
- Implement role-based access control.
- Protect member data.
- Minimize data sent to external AI providers.
- Maintain audit trails.
- Apply appropriate retention policies.
- Review logs for sensitive information.
- Validate uploaded files.
- Rate-limit communication endpoints.
- Add appropriate email-delivery controls.
- Perform security testing.
- Conduct privacy/compliance review appropriate to the deployment environment.

---

# ☁️ Production Deployment

The application is intended to be deployable to a cloud environment such as Microsoft Azure.

A future production architecture may include:

```text
User Browser
     │
     ▼
Frontend Hosting
     │
     ▼
HTTPS / API Layer
     │
     ▼
FastAPI Backend
     │
     ├── ML Inference
     ├── SHAP
     ├── Safety / Navigation Agents
     ├── GenAI Provider
     ├── PDF Reporting
     └── Email Provider
```

Production deployment should additionally consider:

- Azure-hosted application services
- Managed identity
- Key Vault
- Application Insights
- centralized logging
- secure email provider configuration
- domain authentication
- authentication / authorization
- network restrictions
- CORS hardening
- monitoring
- backup/recovery
- rate limiting
- privacy/security review

Deployment infrastructure is intentionally separate from the core UC07 decision logic.

---

# ⚠️ Current Project Limitations

Important limitations include:

1. The current model is trained/evaluated using synthetic data and is **not clinically validated**.
2. Model predictions should not be interpreted as medical diagnoses.
3. SHAP explains model behavior, not causality.
4. Free-text GenAI safety validation cannot replace structured safety controls.
5. External GenAI providers may be unavailable or change supported model names.
6. SMTP delivery does not guarantee inbox placement.
7. Email messages may be classified as spam depending on sender/domain reputation and authentication.
8. Current safety information may be incomplete.
9. Production security, privacy, authentication, authorization, monitoring, and compliance require additional work.
10. Clinical validation would be required before any real-world healthcare use.

---

# 🗺️ Future Enhancements

Potential future improvements include:

- Azure deployment
- Managed secret storage
- Enterprise identity / SSO
- Role-based access control
- Production email provider integration
- Domain-level SPF / DKIM / DMARC configuration
- Persistent communication history
- Advanced audit dashboards
- Care-manager work queues
- Outreach status tracking
- Member engagement tracking
- Longitudinal risk trends
- Model monitoring
- Drift detection
- Fairness monitoring
- Calibration monitoring
- Production observability
- Clinically reviewed safety policies
- Real-world validation

---

# 📊 Project Status

Current implemented capabilities include:

- ✅ Historical data ingestion
- ✅ Point-in-time feature engineering
- ✅ ED-utilization risk prediction
- ✅ Risk stratification
- ✅ SHAP explainability
- ✅ Care-navigation recommendations
- ✅ Safety guardrails
- ✅ Groq GenAI integration
- ✅ Ollama fallback
- ✅ Deterministic explanation fallback
- ✅ Explanation consistency validation
- ✅ Frontend explanation caching
- ✅ Interactive analytics dashboard
- ✅ Cross-filtering
- ✅ Member prioritization
- ✅ Member workspace
- ✅ Light / dark UI
- ✅ PDF member reports
- ✅ Editable email communication
- ✅ PDF email attachment
- ✅ SMTP diagnostics
- ✅ Communication audit events
- ✅ Automated backend/frontend testing

---

# 🏁 Summary

UC07 combines predictive analytics, explainability, safety rules, care navigation, generative AI, reporting, and member communication into a single care-management workflow.

The central philosophy is:

> **Predict patterns, explain risk, prioritize opportunities, navigate proactively, and protect emergency care.**

The system is designed to help care managers identify potential opportunities for lower-acuity future care while maintaining a strict separation between utilization-pattern prediction and emergency-care decision making.

---

## Disclaimer

This repository is intended for demonstration, research, and development purposes.

It is not a clinically validated medical device and should not be used to diagnose, treat, triage, deny, authorize, or determine medical necessity for patient care without the appropriate clinical validation, governance, regulatory review, security controls, and human oversight.
