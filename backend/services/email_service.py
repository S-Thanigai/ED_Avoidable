"""
email_service.py
-----------------
Sends the care-navigation PDF report (rendered by report_service.py,
NEVER independently reconstructed here) to a member's email address.

Responsibilities (and ONLY these):
    - validate the recipient address and reject header/newline injection
    - build a MIME email with the exact PDF bytes the caller supplies
    - send it through a configured EmailProvider, with a small bounded
      retry for genuinely transient failures only (see RETRY POLICY)
    - return a safe, structured EmailSendResult (never a raw SMTP
      exception, stack trace, or credential)

No model/agent decision logic lives here. Credentials are read from
os.environ (backend/.env, gitignored) once per EmailService construction
and are NEVER logged, NEVER returned in an EmailSendResult, and NEVER
sent to the frontend.

Provider abstraction: `EmailProvider` is an abstract base so a future
Azure Communication Services / Azure Email provider can be added as a
second concrete class without changing EmailService's public API or any
caller. Only SmtpEmailProvider is implemented today.

STAGED EXECUTION (Phase 9 SMTP reliability pass)
-------------------------------------------------
`SmtpEmailProvider.send()` runs each SMTP protocol step (connect, ehlo,
starttls, a SECOND ehlo over the now-encrypted channel, authenticate,
send) as its own try/except, via `_stage()`. This exists so a failure
can be attributed to WHERE it happened (`SmtpStageError.stage`) and
WHAT kind of failure it was (`.error_code`, drawn from Python's
`smtplib` exception hierarchy and, where the server sent one, the bare
numeric SMTP response code) -- not collapsed into one generic
"PROVIDER_ERROR"/"NETWORK_ERROR" bucket. This is purely diagnostic
instrumentation: `smtplib.SMTP(..., timeout=X)`'s timeout already
applied to every one of these steps before this change too (it sets the
underlying socket's timeout once, for the life of the connection -- see
CPython's `smtplib.SMTP.connect()`/`getreply()`), so EMAIL_TIMEOUT_SECONDS
was never the bug; not being able to tell WHICH step timed out was.

Port 587 + STARTTLS (Gmail's documented configuration) is used
throughout: `smtplib.SMTP` (plain, unencrypted-at-connect) is used, NOT
`smtplib.SMTP_SSL` (which implies port 465's implicit-TLS-from-the-
start semantics and would be the wrong tool for 587). After `starttls()`
succeeds, a second explicit `ehlo()` is sent over the encrypted channel
per RFC 3207 -- some servers (Gmail included) only advertise `AUTH`
capabilities post-STARTTLS, so skipping this second EHLO can make
`login()` behave inconsistently depending on what the pre-TLS EHLO
happened to report.

RETRY POLICY (Section 9)
-------------------------
At most ONE additional attempt, and only for a FAILURE THAT PROVES THE
MESSAGE WAS NOT ACCEPTED:
    - any pre-"send"-stage TIMEOUT/CONNECTION_FAILED/RATE_LIMITED/
      PROVIDER_TEMPORARY_ERROR (connect, ehlo, starttls, the post-TLS
      ehlo, authenticate) -- the message was never even transmitted.
    - an EXPLICIT 4xx response received DURING the "send" stage itself
      (`PROVIDER_TEMPORARY_ERROR`/`RATE_LIMITED`) -- the server
      explicitly told us "temporary failure, try again" (RFC 5321),
      which is not ambiguous.
NEVER retried: AUTH_FAILED, TLS_FAILED, *_REJECTED, PROVIDER_PERMANENT_ERROR,
UNKNOWN_PROVIDER_ERROR, NOT_CONFIGURED, INVALID_INPUT (all either
permanent or a configuration problem a retry cannot fix) -- and, most
importantly, a TIMEOUT/CONNECTION_FAILED that happens DURING the "send"
stage itself is also never retried, because at that point the client
does not know whether the server already queued the message before the
connection was lost. Resending blindly in that specific case could
produce a duplicate email to the member. This is a real, documented
limitation of this implementation (no server-side idempotency key /
Message-ID pre-registration exists to detect or suppress a duplicate);
see docs/09_MEMBER_COMMUNICATION_REPORTING.md.
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
import socket
import ssl
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Callable

email_logger = logging.getLogger("uc07.communication.email")

# A deliberately simple, conservative recipient-address check -- this is
# an operational communication feature, not a form of medical/identity
# verification. It exists to reject obviously malformed input and (most
# importantly) header/newline injection attempts before anything ever
# reaches smtplib, not to fully validate RFC 5322.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_MAX_ATTEMPTS = 2  # 1 initial + at most 1 retry (Section 9)
_RETRY_DELAY_SECONDS = 1.0

# Stages that happen strictly BEFORE the message body is transmitted --
# a transient failure here means the message was DEFINITELY not sent.
_PRE_SEND_STAGES = frozenset({"connect", "ehlo", "starttls", "tls_ehlo", "authenticate"})
_RETRYABLE_ERROR_CODES = frozenset({"TIMEOUT", "CONNECTION_FAILED", "PROVIDER_TEMPORARY_ERROR", "RATE_LIMITED"})


class EmailValidationError(ValueError):
    """Raised for a malformed recipient address, a header-injection
    attempt in subject/recipient, or any other input-shape problem --
    always a client error (never a provider/network failure)."""


def validate_recipient_email(address: str) -> str:
    """Returns the trimmed address if it is well-formed and free of
    header/newline-injection characters; raises EmailValidationError
    otherwise. Never silently coerces or truncates a bad address."""
    if not address or not isinstance(address, str):
        raise EmailValidationError("Recipient email is required.")
    trimmed = address.strip()
    if "\n" in trimmed or "\r" in trimmed:
        raise EmailValidationError("Recipient email must not contain line breaks.")
    if not _EMAIL_RE.match(trimmed):
        raise EmailValidationError(f"'{trimmed}' is not a valid email address.")
    return trimmed


def _reject_header_injection(field_name: str, value: str) -> str:
    if "\n" in value or "\r" in value:
        raise EmailValidationError(f"{field_name} must not contain line breaks.")
    return value


def mask_email(address: str) -> str:
    """Privacy-safe representation for audit/diagnostic logs -- e.g.
    'j***@example.com'. Never used for the actual send, only for
    logging/observability (Section 22)."""
    local, _, domain = address.partition("@")
    if not domain:
        return "***"
    visible = local[:1] or "*"
    return f"{visible}***@{domain}"


@dataclass(frozen=True)
class EmailSendResult:
    sent: bool
    provider: str
    message: str
    error_code: str | None = None


@dataclass(frozen=True)
class SmtpSendTiming:
    """Coarse, non-sensitive timing (Section 12) -- never includes
    message content. `connection_ms` covers connect through
    authenticate; `send_ms` covers the DATA/message-transmission stage
    only, so a slow report is never confused with a slow network."""
    connection_ms: float
    send_ms: float


class SmtpStageError(RuntimeError):
    """Raised by an EmailProvider for a failure at one specific SMTP
    protocol stage. `safe_message` is always generic and provider-
    response-free -- never the raw exception text, which for some
    providers could echo back parts of the request. `smtp_code`, when
    the server sent one, is a bare 3-digit integer -- safe to log
    (Section 5 explicitly allows `smtp_code=XXX`), never a full response
    line. `exception_type` is the raised exception's CLASS NAME only
    (e.g. "SSLCertVerificationError") -- never its message/args, which
    for some exception types can echo request details -- logged
    alongside `stage`/`error_code`/`smtp_code` so a failure that the
    fixed `error_code` taxonomy can't distinguish (e.g. an unusual OS-
    level TLS error) is still individually diagnosable from the log
    line alone, without needing a code change to find out what it was."""

    def __init__(
        self, stage: str, error_code: str, safe_message: str,
        smtp_code: int | None = None, exception_type: str | None = None,
    ) -> None:
        super().__init__(safe_message)
        self.stage = stage
        self.error_code = error_code
        self.safe_message = safe_message
        self.smtp_code = smtp_code
        self.exception_type = exception_type


def _categorize_smtp_exception(stage: str, exc: Exception) -> SmtpStageError:
    """Maps a raw smtplib/socket/ssl exception to a safe, categorized
    SmtpStageError. Order matters: subclasses are checked before their
    parent classes (e.g. SMTPAuthenticationError before the generic
    SMTPResponseException; ssl.SSLCertVerificationError before the
    generic ssl.SSLError; TimeoutError before the generic OSError).
    Every return path is funneled through `_tag()` so `exception_type`
    is ALWAYS populated, even for a genuinely unrecognized exception --
    that is precisely the case where knowing the real class name matters
    most (see docs/09_MEMBER_COMMUNICATION_REPORTING.md section 8a)."""

    def _tag(error: SmtpStageError) -> SmtpStageError:
        error.exception_type = type(exc).__name__
        return error

    if isinstance(exc, (socket.timeout, TimeoutError)):
        return _tag(SmtpStageError(stage, "TIMEOUT", f"Timed out during the '{stage}' step."))
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return _tag(SmtpStageError(
            stage, "AUTH_FAILED", "The email provider rejected the configured credentials.",
            smtp_code=exc.smtp_code,
        ))
    # ssl.SSLCertVerificationError is a subclass of ssl.SSLError -- must
    # be checked FIRST so a certificate problem gets its own, more
    # specific safe_message rather than the generic TLS one. Neither
    # branch touches how verification itself is performed (Section 4/5
    # of the investigation): the SSLContext used by starttls() below is
    # always ssl.create_default_context()'s untouched default (hostname
    # checking + certificate verification both ON) -- see
    # test_default_ssl_context_used_with_verification_enabled.
    if isinstance(exc, ssl.SSLCertVerificationError):
        return _tag(SmtpStageError(
            stage, "TLS_FAILED", "Could not verify the email server's TLS certificate.",
        ))
    if isinstance(exc, ssl.SSLError):
        return _tag(SmtpStageError(stage, "TLS_FAILED", "Could not establish a secure connection to the email server."))
    if isinstance(exc, smtplib.SMTPNotSupportedError):
        code = "TLS_FAILED" if stage in ("starttls", "tls_ehlo") else "AUTH_FAILED"
        return _tag(SmtpStageError(stage, code, "The email server does not support a required capability."))
    if isinstance(exc, smtplib.SMTPSenderRefused):
        return _tag(SmtpStageError(
            stage, "SENDER_REJECTED", "The email provider rejected the sender address.",
            smtp_code=getattr(exc, "smtp_code", None),
        ))
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return _tag(SmtpStageError(stage, "RECIPIENT_REJECTED", "The email provider refused the recipient address."))
    if isinstance(exc, smtplib.SMTPDataError):
        code = exc.smtp_code
        if code == 421:
            category = "RATE_LIMITED"
        elif 400 <= code < 500:
            category = "PROVIDER_TEMPORARY_ERROR"
        else:
            category = "MESSAGE_REJECTED"
        return _tag(SmtpStageError(stage, category, "The email provider rejected the message.", smtp_code=code))
    if isinstance(exc, smtplib.SMTPResponseException):
        code = exc.smtp_code
        if code == 421:
            return _tag(SmtpStageError(stage, "RATE_LIMITED", "The email provider is temporarily rate-limiting this connection.", smtp_code=code))
        if 400 <= code < 500:
            return _tag(SmtpStageError(stage, "PROVIDER_TEMPORARY_ERROR", "The email provider temporarily rejected this request.", smtp_code=code))
        if 500 <= code < 600:
            return _tag(SmtpStageError(stage, "PROVIDER_PERMANENT_ERROR", "The email provider permanently rejected this request.", smtp_code=code))
        return _tag(SmtpStageError(stage, "UNKNOWN_PROVIDER_ERROR", "The email provider returned an unexpected response.", smtp_code=code))
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return _tag(SmtpStageError(stage, "CONNECTION_FAILED", "The email server closed the connection unexpectedly."))
    if isinstance(exc, smtplib.SMTPException):
        return _tag(SmtpStageError(stage, "UNKNOWN_PROVIDER_ERROR", "The email provider could not complete this request."))
    if isinstance(exc, socket.gaierror):
        return _tag(SmtpStageError(stage, "CONNECTION_FAILED", "Could not resolve the email server address."))
    if isinstance(exc, ConnectionRefusedError):
        return _tag(SmtpStageError(stage, "CONNECTION_FAILED", "The email server refused the connection."))
    # Explicit (rather than relying on the generic OSError branch below)
    # per the investigation's requested coverage list ("connection
    # reset ... OSError") -- same CONNECTION_FAILED outcome either way,
    # but named explicitly so it reads as deliberate, not incidental.
    if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
        return _tag(SmtpStageError(stage, "CONNECTION_FAILED", "The connection to the email server was reset."))
    if isinstance(exc, OSError):
        return _tag(SmtpStageError(stage, "CONNECTION_FAILED", "Could not connect to the email server."))
    # Truly unrecognized exception type -- error_code stays the coarse
    # UNKNOWN_PROVIDER_ERROR bucket (the frontend/API contract only ever
    # needs a bounded set of codes), but `exception_type` below still
    # captures exactly what it was, so this is never actually a dead
    # end for diagnosis -- see the "smtp_send" log line.
    return _tag(SmtpStageError(stage, "UNKNOWN_PROVIDER_ERROR", "The email provider could not complete this request."))


class EmailProvider(ABC):
    """Abstract send transport. A concrete provider knows how to deliver
    one already-built EmailMessage; it does not validate addresses or
    build attachments (EmailService does that, once, provider-agnostic)."""

    name: str = "abstract"

    @abstractmethod
    def send(self, message: EmailMessage, timeout_seconds: float) -> SmtpSendTiming:
        """Returns timing on success. Raises `SmtpStageError` (or, if
        the provider is entirely unusable -- e.g. no host configured --
        `EmailProviderUnavailableError`) on any failure. EmailService
        translates either into a safe EmailSendResult; this method never
        lets a raw smtplib/socket/ssl exception escape."""


class SmtpEmailProvider(EmailProvider):
    name = "smtp"

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        use_tls: bool,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._use_tls = use_tls

    @staticmethod
    def _stage(stage: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, immediately re-categorized
            raise _categorize_smtp_exception(stage, exc) from exc

    @staticmethod
    def _close_quietly(smtp: smtplib.SMTP) -> None:
        """Connection cleanup (Section 7): ALWAYS attempts a clean QUIT,
        and ALWAYS falls back to closing the raw socket if QUIT itself
        fails (e.g. the connection already dropped) -- a failed send
        must never leave a stale, half-open SMTP connection around for
        the next request. Called from a `finally` block unconditionally."""
        try:
            smtp.quit()
        except Exception:
            try:
                smtp.close()
            except Exception:
                pass

    def _connect(self, timeout_seconds: float) -> smtplib.SMTP:
        """Constructs AND connects in a single call -- host/port MUST be
        passed directly to the `smtplib.SMTP` constructor rather than via
        a separate `smtp.connect(host, port)` call afterward.

        ROOT CAUSE of the `stage=starttls result=UNKNOWN_PROVIDER_ERROR`
        regression: `smtplib.SMTP.__init__` is the ONLY place that sets
        the instance's internal `self._host` attribute (`self._host =
        host`, unconditionally, before it even attempts to connect).
        `smtplib.SMTP.connect(host, port)` establishes the socket using
        its own LOCAL `host` parameter but never assigns it back to
        `self._host`. `starttls()` later calls
        `context.wrap_socket(self.sock, server_hostname=self._host)` for
        TLS SNI/hostname verification -- if the object was constructed
        with no host (as the previous staged-connect version did, to
        make "connect" its own catchable/timeable stage, then called
        `smtp.connect(self._host, self._port)` afterward), `self._host`
        is stuck at the constructor's default `''`, and
        `ssl.SSLContext.wrap_socket()` raises `ValueError("check_hostname
        requires server_hostname")` -- a plain ValueError, not any
        smtplib/ssl/socket/OSError subclass this module's categorizer
        recognized, so it fell all the way through to the generic
        UNKNOWN_PROVIDER_ERROR catch-all on EVERY call. Verified directly
        against a real `smtp.gmail.com:587` handshake (both the failure
        and this fix) -- see docs/09_MEMBER_COMMUNICATION_REPORTING.md
        section 8a. "connect" stays its own attributable stage: any
        exception raised during construction (including the implicit
        connect it performs) is still categorized with stage="connect"."""
        try:
            return smtplib.SMTP(self._host, self._port, timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, immediately re-categorized
            raise _categorize_smtp_exception("connect", exc) from exc

    def send(self, message: EmailMessage, timeout_seconds: float) -> SmtpSendTiming:
        # `smtplib.SMTP(host, port, timeout=...)` applies `timeout_seconds`
        # to the underlying socket for connect() AND every subsequent
        # read (ehlo/starttls/login/send all reuse the same socket) --
        # this was already correct before this pass; see module
        # docstring. Timing starts here (before _connect) so
        # connection_ms includes the connect itself, same as before.
        connection_start = time.monotonic()
        smtp = self._connect(timeout_seconds)
        try:
            self._stage("ehlo", lambda: smtp.ehlo())
            if self._use_tls:
                self._stage("starttls", lambda: smtp.starttls(context=ssl.create_default_context()))
                # RFC 3207: capabilities (notably AUTH) must be
                # re-discovered over the now-encrypted channel. Explicit
                # here rather than relying on login()'s implicit
                # ehlo_or_helo_if_needed() so this stage is individually
                # attributable in logs/errors.
                self._stage("tls_ehlo", lambda: smtp.ehlo())
            if self._username:
                self._stage("authenticate", lambda: smtp.login(self._username, self._password))
            connection_ms = (time.monotonic() - connection_start) * 1000

            send_start = time.monotonic()
            self._stage("send", lambda: smtp.send_message(message))
            send_ms = (time.monotonic() - send_start) * 1000
        finally:
            self._close_quietly(smtp)
        return SmtpSendTiming(connection_ms=connection_ms, send_ms=send_ms)


class EmailProviderUnavailableError(RuntimeError):
    """Raised when a provider is selected but not usable at all (e.g. an
    unsupported EMAIL_PROVIDER value) -- distinct from a runtime send
    failure so EmailService can report a clearer, still-safe message."""


class EmailConfig:
    """Read fresh from the environment on every construction (mirrors
    genai_explanation.GenAIConfig's convention) so tests can monkeypatch
    os.environ per-test and a running server picks up a changed
    EMAIL_ENABLED/SMTP_* without a restart."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("EMAIL_ENABLED", "false").strip().lower() == "true"
        self.provider = os.environ.get("EMAIL_PROVIDER", "smtp").strip().lower() or "smtp"

        self.smtp_host = os.environ.get("SMTP_HOST", "").strip()
        try:
            self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        except ValueError:
            self.smtp_port = 587
        self.smtp_username = os.environ.get("SMTP_USERNAME", "").strip()
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.environ.get("SMTP_USE_TLS", "true").strip().lower() == "true"

        self.from_email = os.environ.get("SMTP_FROM_EMAIL", "").strip()
        self.from_name = os.environ.get("SMTP_FROM_NAME", "UC07 Care Management").strip()

        # A single, uniform socket timeout applied to every SMTP stage
        # (see module docstring) -- bounded, never unlimited. 30s is a
        # reasonable default for Gmail-class providers; a slower local/
        # demo SMTP relay (e.g. MailHog under load) can raise this via
        # the env var -- see backend/.env.example.
        try:
            self.timeout_seconds = float(os.environ.get("EMAIL_TIMEOUT_SECONDS", "30"))
        except ValueError:
            self.timeout_seconds = 30.0

    @property
    def configured(self) -> bool:
        """Safe to expose via GET /health -- true only if enabled AND
        the active provider has the minimum fields it needs to attempt a
        send. Never reveals which fields are missing, and never reveals
        credential values."""
        if not self.enabled:
            return False
        if self.provider == "smtp":
            return bool(self.smtp_host and self.from_email)
        return False


def load_email_config() -> EmailConfig:
    return EmailConfig()


def _is_retryable(error: SmtpStageError) -> bool:
    """See module docstring's RETRY POLICY section -- the short version:
    retry anything transient that happened before the message was
    transmitted, plus an EXPLICIT 4xx/rate-limit response received
    DURING transmission; never retry a TIMEOUT/CONNECTION_FAILED that
    happens DURING transmission, since it is impossible to know from the
    client side whether the server already queued the message."""
    if error.error_code not in _RETRYABLE_ERROR_CODES:
        return False
    if error.stage in _PRE_SEND_STAGES:
        return True
    return error.stage == "send" and error.error_code in ("PROVIDER_TEMPORARY_ERROR", "RATE_LIMITED")


class EmailService:
    """Provider-agnostic facade. `send_report_email` is the only method
    callers (backend/main.py's POST /uc07/email) need."""

    def __init__(self, config: EmailConfig | None = None) -> None:
        self._config = config or load_email_config()

    def _build_provider(self) -> EmailProvider:
        if self._config.provider == "smtp":
            return SmtpEmailProvider(
                host=self._config.smtp_host,
                port=self._config.smtp_port,
                username=self._config.smtp_username,
                password=self._config.smtp_password,
                use_tls=self._config.smtp_use_tls,
            )
        raise EmailProviderUnavailableError(f"Unsupported EMAIL_PROVIDER '{self._config.provider}'.")

    @staticmethod
    def _build_message(
        *, from_name: str, from_email: str, recipient: str, subject: str, body: str,
        attachment_bytes: bytes, attachment_filename: str,
    ) -> EmailMessage:
        """Plain-text, single-attachment MIME message -- deliberately
        simple (Section 13/16): no HTML body, no tracking pixels, no
        marketing-style formatting. `Date` and `Message-ID` are set
        explicitly (Section 13) -- smtplib does NOT add either
        automatically, and their absence is a common, easily-avoided
        spam-classifier signal."""
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = f"{from_name} <{from_email}>"
        message["To"] = recipient
        message["Date"] = formatdate(localtime=True)
        domain = from_email.rsplit("@", 1)[-1] if "@" in from_email else None
        message["Message-ID"] = make_msgid(domain=domain)
        message.set_content(body)
        message.add_attachment(
            attachment_bytes,
            maintype="application",
            subtype="pdf",
            filename=attachment_filename,
        )
        return message

    def _result_from_error(self, error: SmtpStageError) -> EmailSendResult:
        message = error.safe_message
        if error.stage == "send" and error.error_code in ("TIMEOUT", "CONNECTION_FAILED"):
            # Ambiguous send-state (Section 9) -- surfaced to the care
            # manager explicitly rather than silently, since retrying
            # by hand (clicking Send again) could produce a duplicate.
            message += (
                " Delivery could not be confirmed -- check with the recipient before sending again, "
                "to avoid a possible duplicate email."
            )
        return EmailSendResult(sent=False, provider=self._config.provider, message=message, error_code=error.error_code)

    def send_report_email(
        self,
        *,
        to_email: str,
        subject: str,
        body: str,
        attachment_bytes: bytes,
        attachment_filename: str,
    ) -> EmailSendResult:
        """Builds and sends one email with `attachment_bytes` (the SAME
        PDF report_service.generate_report_pdf() produced -- this method
        never regenerates or alters the attachment, and never rebuilds
        it again for a retry) as an application/pdf attachment. Never
        raises: any failure (disabled, misconfigured, validation,
        network, auth, timeout) is returned as a non-`sent`
        EmailSendResult with a safe, generic message."""
        if not self._config.enabled:
            return EmailSendResult(
                sent=False, provider=self._config.provider,
                message="Email sending is disabled (EMAIL_ENABLED is not \"true\").",
                error_code="EMAIL_DISABLED",
            )

        try:
            recipient = validate_recipient_email(to_email)
            _reject_header_injection("Subject", subject)
        except EmailValidationError as exc:
            return EmailSendResult(
                sent=False, provider=self._config.provider, message=str(exc), error_code="INVALID_INPUT",
            )

        if not subject.strip():
            return EmailSendResult(
                sent=False, provider=self._config.provider,
                message="Subject must not be empty.", error_code="INVALID_INPUT",
            )
        if not body.strip():
            return EmailSendResult(
                sent=False, provider=self._config.provider,
                message="Message body must not be empty.", error_code="INVALID_INPUT",
            )
        if not attachment_bytes:
            return EmailSendResult(
                sent=False, provider=self._config.provider,
                message="Report attachment is empty or was not generated.", error_code="MISSING_ATTACHMENT",
            )

        if not self._config.from_email:
            return EmailSendResult(
                sent=False, provider=self._config.provider,
                message="Email sender is not configured (SMTP_FROM_EMAIL is unset).",
                error_code="NOT_CONFIGURED",
            )
        if self._config.provider == "smtp" and not self._config.smtp_host:
            return EmailSendResult(
                sent=False, provider=self._config.provider,
                message="Email server is not configured (SMTP_HOST is unset).",
                error_code="NOT_CONFIGURED",
            )

        try:
            provider = self._build_provider()
        except EmailProviderUnavailableError as exc:
            return EmailSendResult(
                sent=False, provider=self._config.provider, message=str(exc), error_code="NOT_CONFIGURED",
            )

        # Built ONCE, outside the retry loop (Section 12) -- a retry
        # re-sends this exact same message object, it never regenerates
        # or re-reads the PDF attachment.
        message = self._build_message(
            from_name=self._config.from_name, from_email=self._config.from_email, recipient=recipient,
            subject=subject, body=body, attachment_bytes=attachment_bytes, attachment_filename=attachment_filename,
        )

        last_error: SmtpStageError | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                timing = provider.send(message, timeout_seconds=self._config.timeout_seconds)
            except SmtpStageError as exc:
                last_error = exc
                # exc_type is the raised exception's CLASS NAME only
                # (e.g. "SSLCertVerificationError", "SMTPNotSupportedError")
                # -- never its message/args. Safe to log (Section 5) and
                # the key diagnostic field for a failure the fixed
                # error_code taxonomy can't fully distinguish on its own.
                email_logger.warning(
                    "smtp_send provider=%s stage=%s result=%s smtp_code=%s exc_type=%s attempt=%d/%d",
                    self._config.provider, exc.stage, exc.error_code, exc.smtp_code, exc.exception_type,
                    attempt, _MAX_ATTEMPTS,
                )
                if attempt < _MAX_ATTEMPTS and _is_retryable(exc):
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue
                return self._result_from_error(exc)
            else:
                email_logger.info(
                    "smtp_send provider=%s result=SENT connection_ms=%.1f send_ms=%.1f attempt=%d/%d",
                    self._config.provider, timing.connection_ms, timing.send_ms, attempt, _MAX_ATTEMPTS,
                )
                return EmailSendResult(
                    sent=True, provider=self._config.provider,
                    message=f"Report sent to {mask_email(recipient)}.",
                )

        assert last_error is not None  # loop always returns; satisfies static analysis only
        return self._result_from_error(last_error)
