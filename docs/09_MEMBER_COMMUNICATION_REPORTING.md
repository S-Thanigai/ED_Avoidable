# Phase 9 — Member Communication & Reporting (PDF Report + Email)

**Date:** 2026-08-17
**Status:** Complete. Model, thresholds, SHAP, Risk/Navigation/Safety Agents, GenAI decision authority, `/uc07/decide` behavior, training datasets, and model artifacts: **unchanged**. This phase is a purely operational communication/reporting layer that CONSUMES an already-computed `FinalUC07Decision` (and an already-approved `MemberExplanationResponse`); it never creates or influences one.

---

## 1. Purpose

Gives a care manager two actions on a member they already have a decision open for:

- **Download PDF Report** — a professional "Member Care Navigation & Risk Summary" PDF.
- **Send to Member** — an editable email composer that attaches the *same* PDF and sends it, with explicit confirmation.

Not a clinical/diagnosis report. Never implies emergency care should be delayed or skipped.

## 2. Does the member dataset have an email address?

No. `raw_members.csv` (and every UC07 API response) was inspected directly — it has `member_id, age, gender, diabetes, copd, hypertension, chf, asthma, ckd, num_chronic_conditions, transportation_barrier, telehealth_available, pcp_distance_miles, urgent_care_distance_miles` and nothing resembling contact information; there is no name field either.

Rather than inventing one, this phase adds a small **frontend-only contact store** (`frontend/src/uc07/memberContacts.ts`) — a `member_id -> {name?, email?}` map in `localStorage`. It exists purely so a care manager's typed-in email is remembered across sessions in the same browser; it is:

- **never** sent to `POST /uc07/decide`
- **never** read by any agent
- **never** used as an ML feature or joined into any snapshot/training data

The email composer's "Recipient" field is always an editable text input — there is no trusted, backend-verified contact source to make it read-only against (Section 18's "read-only if trusted" case does not apply here; documented, not silently ignored).

## 3. Architecture

```
Frontend (MemberDetailsDrawer)
  MemberReportActions            "Download PDF Report" / "Send to Member"
      |                                    |
      | builds ReportRequestPayload        | opens
      | (api.ts: buildReportRequest)       v
      |                          EmailComposerModal (editable To/Subject/Body,
      |                          Preview PDF, two-step Send confirmation)
      v                                    |
POST /uc07/report                 POST /uc07/email
      |                                    |
      v                                    v
backend/main.py: _build_report_context()  (SAME helper, both endpoints)
      |                                    |
      v                                    v
report_service.generate_report_pdf()  --> pdf_bytes  --> email_service.EmailService
      |                                                        |
      v                                                        v
 application/pdf response                            SmtpEmailProvider.send()
```

`backend/services/report_service.py` and `backend/services/email_service.py` are pure, decision-agent-free modules:

- **Never import** `risk_detection.py`, `care_navigation.py`, `safety_policy.decide()`, `model_explainability.py`, or `orchestrator.py` (enforced by construction, same pattern as `genai_explanation.py`; see `test_no_model_decision_authority_imports` in `test_email_service.py`).
- `report_service.py` does no I/O and no network calls — it renders `bytes` from a plain `ReportContext` dataclass.
- `email_service.py` does no PDF rendering — it only attaches whatever bytes it is given.

## 4. Same PDF, one rendering path (Section 15/16)

`backend/main.py`'s `_build_report_context()` is the **one** place a `ReportRequest` becomes a `report_service.ReportContext`. Both `POST /uc07/report` and `POST /uc07/email` call it, then call `report_service.generate_report_pdf()` — there is exactly one PDF-rendering code path in this project. `POST /uc07/email` never independently reconstructs an attachment.

(The two are not *byte-identical* across two separate HTTP calls — each call gets its own `report_id`/timestamp, which is expected and correct. `test_email_endpoint_success_attaches_same_pdf_as_report_endpoint` asserts the *content* — everything except the report-id/timestamp lines — is identical.)

## 5. Report content

Rendered with **ReportLab** (Platypus), A4, header/footer with page numbers + report ID + generated timestamp on every page, semantic colors (green=LOW/CLEAR, amber=MODERATE/CAUTION, red=HIGH/OVERRIDE, purple=explainability), grid-based key/value tables, multi-page support. No unicode glyphs outside the base Helvetica font's WinAnsi/Latin-1 coverage are used (an earlier draft used ▲/▼/⚠, which render as broken glyphs in real PDF viewers even though `pypdf` text-extraction shows the intended character — fixed to plain ASCII `(+)`/`(-)`/`— PRIORITY`).

Sections (A–I, per spec): Report Metadata, Member Information, 90-Day Risk Assessment ("*Predicted likelihood of potentially avoidable ED utilization within the next 90 days*" — never "emergency risk"), Key Model Factors (human-readable `display_name` + direction only, never a raw feature slug; "*model-attribution signals, do not establish causation*"), Care Navigation, Current Safety Status, Explanation, Important Safety Notice, Model Disclosure (shown only when `synthetic_model` is true).

### OVERRIDE behavior (Section 5)

When `safety.state == "OVERRIDE"`:

- The **Safety section moves ahead of** the Risk Assessment section and is labeled "— PRIORITY".
- The Risk section is re-labeled "(reference only — see Safety, above)" — never suppressed, just visually secondary.
- The Care Navigation section leads with: *"Current supplied information triggered a safety override. Proactive navigation information must not delay appropriate emergency evaluation."*

Verified in `test_report_service.py::test_13_*` (section ordering, wording) and `EmailComposerModal`/`MemberReportActions` inherit this for free since they always render whatever the backend returns.

## 6. Explanation reuse — no extra LLM call (Section 4-G / 14)

`_resolve_report_explanation()` in `main.py`:

- If the request includes an `explanation` (i.e. the frontend already called `POST /uc07/explain` for this member and has it cached — see `explanationCache.ts`), it is reused **verbatim**.
- If not, it calls `genai_explanation.generate_explanation()` with a `GenAIConfig` forced to `enabled = False` — this returns the exact same deterministic, template-based explanation the GenAI agent always falls back to, with **zero network/LLM calls**. `test_report_endpoint_uses_deterministic_fallback_when_no_explanation_given` asserts this by monkeypatching Groq/Ollama's HTTP calls to raise if invoked.

GenAI is never called *because of* a report/email action — only because the frontend had already, separately, called `/uc07/explain` for that member.

## 7. Backend endpoints

- **`POST /uc07/report`** → `application/pdf`, `Content-Disposition: attachment; filename="Member_Care_Navigation_Report_<member_id>.pdf"`. Input: `ReportRequest` (member contact fields + the same risk/navigation/safety shape `ExplainRequest` uses, plus `safety.message` and an optional `explanation`). Malformed input → FastAPI's standard 422.
- **`POST /uc07/email`** → always `200` with `{sent, provider, message, error_code, report_id}` — same "never surface a business failure as an HTTP error" convention `POST /uc07/explain` already uses. Input: `EmailSendRequest = { report: ReportRequest, to_email, subject, body }`.
- **`GET /health`** — extended with `email_configured: bool` and `email_provider: "smtp"`. Never exposes credentials.

**Trust model, stated explicitly:** like `POST /uc07/explain` before it, these endpoints trust the frontend's echoed risk/navigation/safety values rather than re-deriving them from an uploaded CSV. This app has no server-side persisted-decision store — `POST /uc07/decide` is stateless per-upload — so "regenerate/validate server-side" would require a new persistence layer, which is out of this phase's scope (operational communication layer only, not a data-architecture change). Documented here rather than silently assumed; see Limitations.

## 8. Email service (Section 9–13)

`backend/services/email_service.py`:

- `EmailProvider` ABC + `SmtpEmailProvider` (only implemented provider). A future `AzureEmailProvider` can be added as a second class without changing `EmailService`'s public API.
- Validates recipient (`validate_recipient_email`) and rejects header/newline injection in the recipient or subject **before** any network call.
- Never logs or returns `SMTP_PASSWORD`/any credential; every failure path returns a generic, safe `EmailSendResult`.
- `EMAIL_ENABLED=false` (the default) short-circuits before any SMTP connection is attempted.

### 8a. SMTP reliability (Phase 9 follow-up — intermittent TIMEOUT/PROVIDER_ERROR investigation)

`SmtpEmailProvider.send()` runs each SMTP protocol step **individually**, each in its own try/except:

```
connect → ehlo → [starttls → ehlo again] → [authenticate] → send
```

**Port 587 uses `smtplib.SMTP` + `.starttls()`** (never `smtplib.SMTP_SSL`, which is port 465's implicit-TLS semantics — confirmed absent from this codepath by `test_smtp_ssl_never_used_for_port_587_starttls_flow`). After `starttls()` succeeds, a **second, explicit `ehlo()`** is sent over the now-encrypted channel — RFC 3207 requires this because some servers (Gmail included) only advertise `AUTH` capabilities post-STARTTLS.

**Root cause of the reported TIMEOUT/PROVIDER_ERROR reports:** `EMAIL_TIMEOUT_SECONDS` was already being applied correctly to every stage — `smtplib.SMTP(timeout=X)` sets the underlying socket's timeout once, and that same socket (and therefore that same timeout) is reused for connect *and* every subsequent read (EHLO/STARTTLS/AUTH/DATA), verified directly from CPython's `smtplib` source. The actual problem was **observability, not the timeout value**: every failure — a slow DNS lookup, a stalled TLS handshake, a rejected `AUTH`, a 4xx during message transmission — all collapsed into the same generic `TIMEOUT`/`PROVIDER_ERROR` bucket with no indication of *where* it happened. Gmail's `smtp.gmail.com:587` itself is not unusually slow; occasional multi-second STARTTLS/AUTH round-trips are normal internet-SMTP behavior that a single 30s bound already tolerates.

**Fix:** each stage is now individually attributable (`SmtpStageError.stage`) and categorized (`.error_code`, drawn from `smtplib`'s actual exception hierarchy plus the server's numeric SMTP response code where one exists) into: `TIMEOUT`, `CONNECTION_FAILED`, `TLS_FAILED`, `AUTH_FAILED`, `SENDER_REJECTED`, `RECIPIENT_REJECTED`, `MESSAGE_REJECTED`, `RATE_LIMITED` (SMTP 421), `PROVIDER_TEMPORARY_ERROR` (4xx), `PROVIDER_PERMANENT_ERROR` (5xx), `UNKNOWN_PROVIDER_ERROR`. (`TIMEOUT` is intentionally kept as ONE code rather than split into `CONNECT_TIMEOUT`/`SEND_TIMEOUT` — `stage` already disambiguates *where*, so this avoids doubling the code surface for the same information.)

**Safe stage logging** (`uc07.communication.email` logger, Section 5's exact allowed shape):
```
smtp_send provider=smtp stage=authenticate result=AUTH_FAILED smtp_code=535 attempt=1/2
smtp_send provider=smtp result=SENT connection_ms=612.4 send_ms=88.1 attempt=1/2
```
Never the SMTP password, the raw provider response text, the report content, or the email body — verified by `test_20_smtp_send_log_line_never_contains_password` / `test_20_successful_send_log_line_has_no_message_content`.

**Follow-up: `stage=starttls result=UNKNOWN_PROVIDER_ERROR` diagnosability.** Every branch of `_categorize_smtp_exception()` now also tags `SmtpStageError.exception_type` with the raised exception's **class name only** (e.g. `SSLCertVerificationError`, `SMTPNotSupportedError`) — including the final catch-all, so even a genuinely unrecognized exception type is no longer a diagnostic dead end. Logged as `exc_type=` in the `smtp_send` line. `ssl.SSLCertVerificationError` (a subclass of `ssl.SSLError`) is now checked explicitly, ahead of the generic TLS branch, with its own message ("Could not verify the email server's TLS certificate.") rather than the generic one — and `ConnectionResetError`/`BrokenPipeError` are named explicitly rather than only falling through the generic `OSError` branch. **TLS certificate verification was NOT weakened to chase this** — `ssl.create_default_context()` is called with no arguments (spied on directly by `test_starttls_uses_create_default_context_with_no_overrides`, which also asserts the actual context handed to `starttls()` still has `verify_mode == ssl.CERT_REQUIRED` and `check_hostname is True`); `test_tls_verification_is_never_weakened` guards against `CERT_NONE`/`check_hostname=False`/an unverified context ever being introduced.

**Actual root cause, found and confirmed against a real `smtp.gmail.com:587` handshake:** the "connect" stage's own hardening (deferring `smtplib.SMTP()` construction, then calling `smtp.connect(host, port)` separately so connect had its own attributable/timeable stage) broke STARTTLS on every call. `smtplib.SMTP.__init__` is the ONLY place that sets the instance's internal `self._host`; `smtp.connect(host, port)` establishes the socket using its own local `host` parameter but never writes it back to `self._host`. `starttls()` later calls `context.wrap_socket(self.sock, server_hostname=self._host)` for TLS SNI/hostname verification — with `self._host` stuck at the constructor default `''`, `ssl`'s `wrap_socket()` raised a plain `ValueError("check_hostname requires server_hostname")`, which matched none of the categorizer's `smtplib`/`ssl`/`socket`/`OSError` branches and fell through to `UNKNOWN_PROVIDER_ERROR` on 100% of calls. **Fix:** `SmtpEmailProvider._connect()` now constructs `smtplib.SMTP(host, port, timeout=...)` directly (host/port passed to the constructor, exactly like the original pre-hardening implementation), restoring correct internal state, while keeping "connect" as its own individually-categorized stage (any exception raised during construction is still tagged `stage="connect"`). Verified directly: a real connect→EHLO→STARTTLS→EHLO handshake against `smtp.gmail.com:587` now succeeds (login/send were stubbed out for that verification — no real email was sent). See `docs/DECISION_LOG.md` #127.

**Retry policy — at most 1 additional attempt, transient failures only:** a pre-"send"-stage `TIMEOUT`/`CONNECTION_FAILED`/`RATE_LIMITED`/`PROVIDER_TEMPORARY_ERROR` (connect/ehlo/starttls/tls_ehlo/authenticate — the message was never transmitted) is retried once after a 1s delay. An explicit 4xx/rate-limit response received **during** the send stage is also retried (the server gave an unambiguous "try again" signal, RFC 5321). **Never retried:** `AUTH_FAILED`, `TLS_FAILED`, any `*_REJECTED`, `PROVIDER_PERMANENT_ERROR`, `UNKNOWN_PROVIDER_ERROR`, `NOT_CONFIGURED`/`INVALID_INPUT` (permanent/config problems a retry cannot fix) — **and, critically, a `TIMEOUT`/`CONNECTION_FAILED` that happens *during* the send stage is never retried either**, because the client cannot tell whether the server already queued the message before the connection was lost. Resending blindly there could produce a duplicate email. **This is a real, documented limitation**: there is no idempotency key / `Message-ID` pre-registration to detect or suppress a true duplicate if a *human* manually clicks Send again after seeing this specific failure — the response message explicitly warns the care manager ("Delivery could not be confirmed — check with the recipient before sending again").

**Connection cleanup (Section 7):** `SmtpEmailProvider.send()` always attempts `smtp.quit()` in a `finally` block, and falls back to `smtp.close()` if `quit()` itself fails (e.g. the connection already dropped) — a failed send never leaves a stale connection for the next request. Each `send_report_email()` call builds a brand-new `SmtpEmailProvider`/`smtplib.SMTP` instance, so one request's failure cannot affect a later, independent request either (`test_11_second_independent_send_succeeds_after_a_prior_failure`).

**Event loop (Section 8):** `POST /uc07/report` and `POST /uc07/email` are plain `def` handlers (not `async def`) — FastAPI/Starlette automatically runs a synchronous route in its own worker thread (`run_in_threadpool`), so the blocking SMTP I/O never runs on, and cannot starve, the main asyncio event loop. This was **verified**, not assumed: `test_uc07_email_concurrency.py::test_slow_smtp_send_does_not_block_health` fires a `POST /uc07/email` whose (mocked) SMTP send takes 1.5s, and confirms a concurrent `GET /health` still responds in well under 1s. No `ProcessPoolExecutor` is used (unlike the legacy `/predict` SHAP path) — SMTP is ordinary blocking network I/O, which releases the GIL while waiting on the socket; a thread handles that correctly. A `ProcessPoolExecutor` would be the wrong tool here and was deliberately not added.

**Deliverability hygiene (Section 13/16) — what this app CAN control:** the message is plain-text only (no HTML, no tracking pixels, no decorative markup), a single professionally-named PDF attachment, a matching `From` display name/address (no mismatch), and now explicit `Date` and `Message-ID` headers (`smtplib` does not add either automatically; their absence is a common, easily-avoided spam-classifier signal). **What this app CANNOT control or guarantee:** which inbox tab/folder a recipient's provider chooses to place the message in. Gmail-SMTP-relay-as-a-generic-sender (rather than a properly authenticated custom domain) has no application-level fix — the durable fix is a transactional email provider on an authenticated domain with SPF/DKIM/DMARC configured, which is future work (see Limitations), not something `EmailService` can paper over. **Primary Inbox placement is never guaranteed by this or any application code.**

## 9. Audit trail (Section 22)

No database in this prototype, so "audit trail" = structured logging (`backend/services/audit.py`, logger `uc07.communication.audit`, one INFO line per event: `event_id, action (PDF_GENERATED|EMAIL_SENT|EMAIL_FAILED), member_id, report_id, recipient (masked, e.g. j***@example.com), provider, result, timestamp`). Never logs the SMTP password, any credential, the full report content, or the full email body. **Production follow-up (not implemented here):** persist these events to a real audit store/database instead of process logs.

## 10. Privacy (Section 24)

PDFs are generated in memory (`io.BytesIO`) and never written to disk. No temp files are created. Explanation/report payloads are not logged beyond the audit event's allow-listed fields above. **This does not make the prototype HIPAA-compliant** — there is no encryption-at-rest for the in-memory buffer beyond process memory protections, no access control on the endpoints themselves, and synthetic data is used throughout the demo. A real deployment needs transport security (HTTPS), authenticated/authorized endpoints, and a compliant email provider/BAA before handling real PHI.

## 11. Environment variables

See `backend/.env.example` for the full block. Summary:

| Variable | Default | Notes |
|---|---|---|
| `EMAIL_ENABLED` | `false` | Master switch |
| `EMAIL_PROVIDER` | `smtp` | Only `smtp` implemented |
| `SMTP_HOST` | *(blank)* | |
| `SMTP_PORT` | `587` | |
| `SMTP_USERNAME` | *(blank)* | |
| `SMTP_PASSWORD` | *(blank)* | Never committed, never logged |
| `SMTP_FROM_EMAIL` | *(blank)* | Required for sending |
| `SMTP_FROM_NAME` | `UC07 Care Management` | |
| `SMTP_USE_TLS` | `true` | |
| `EMAIL_TIMEOUT_SECONDS` | `30` | Uniform socket timeout for every SMTP stage; raise (e.g. 60) only for a genuinely slow local/demo relay — see 8a |

## 12. Tests

- `backend/tests/test_report_service.py` — 19 tests against `generate_report_pdf()` directly (via `pypdf` text extraction): required fields, synthetic disclosure, human-readable-only factors, multipage, OVERRIDE ordering/wording, deterministic content given fixed input, filename sanitization.
- `backend/tests/test_email_service.py` — 59 tests, `smtplib.SMTP` always mocked, split across two layers: `SmtpEmailProvider` staged-execution/categorization tests (timeout/auth/TLS incl. certificate verification failure/recipient/sender/message rejection, 4xx/5xx/421, connection cleanup, timeout applied, TLS verification never weakened) and `EmailService` tests (validation, retry policy — transient-retried/permanent-not-retried/ambiguous-send-never-retried, same-PDF/MIME/filename/Date/Message-ID, credentials never logged or returned, exception class name always logged for diagnosability).
- `backend/tests/test_uc07_communication_api.py` — 16 HTTP-level tests: PDF content-type/disposition, 422 on malformed input, deterministic-fallback vs. reused explanation, disabled-by-default email, same-content download/email attachment, OVERRIDE wording preserved through the email path, `/uc07/decide` unaffected, `/health` fields, no leaked SMTP internals in the HTTP response.
- `backend/tests/test_uc07_email_concurrency.py` — 1 test: a slow (mocked) SMTP send does not block a concurrent `GET /health`.
- `frontend/src/uc07/__tests__/MemberReportActions.test.tsx` — 5 tests: buttons render, download request shape, clean error surface, composer prefill from saved contact, composer close.
- `frontend/src/uc07/__tests__/EmailComposerModal.test.tsx` — 14 tests: prefill/edit fields, email validation, two-step confirm-before-send, success/failure states (no stack trace ever shown), PDF preview, Cancel/Escape, focus-on-open.

## 13. Limitations / future work

- No server-side persisted decision store — endpoints trust the frontend's echoed decision (see Section 7). A future phase could add a real backend member/decision store and revalidate server-side.
- No real database audit trail (process logging only — see Section 9).
- Recipient email always comes from `localStorage`/manual entry, never a verified directory — see Section 2.
- **No duplicate-send protection for a manually-repeated action after an ambiguous timeout** (see 8a) — if the SMTP connection is lost mid-transmission (not before, not with an explicit server response), this app cannot tell whether the message was queued, so it does not auto-retry; a care manager who manually clicks Send again after seeing that specific warning could still produce a duplicate. A real idempotency mechanism (e.g. a client-supplied request id deduplicated server-side) is future work.
- **Deliverability is best-effort, not guaranteed** (see 8a) — this app controls MIME construction, headers (`Date`/`Message-ID`/matching `From`), and content hygiene, but Primary Inbox vs. Spam/Promotions placement is decided entirely by the recipient's provider. A Gmail-relay-as-generic-sender setup has no application-level fix for this; a transactional provider on an authenticated custom domain (SPF/DKIM/DMARC) is the real fix, and is future work.
- **Azure email provider:** not implemented. `EmailProvider` is already an abstract base specifically so a future `AzureEmailProvider(EmailProvider)` can be added and selected via `EMAIL_PROVIDER=azure` without changing `EmailService`, the endpoints, or the frontend. **Not started per instruction — stop here.**
