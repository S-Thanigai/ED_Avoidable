"""
Phase 9 -- backend/services/email_service.py unit tests, split across
two layers matching the module's own architecture:

    - SmtpEmailProvider tests: mock `smtplib.SMTP` itself (its
      connect/ehlo/starttls/login/send_message methods) and exercise the
      REAL SmtpEmailProvider.send() to verify exception categorization,
      stage attribution, and connection cleanup.
    - EmailService tests: mock `EmailService._build_provider()` to
      return a fake provider that raises `SmtpStageError` directly (the
      actual contract EmailService depends on) to verify retry policy,
      validation, and result mapping.

`smtplib.SMTP` is ALWAYS mocked -- no test in this file ever opens a
real network connection or sends a real email (Section 18's explicit
requirement). `time.sleep` is patched to a no-op wherever a retry is
exercised, so these tests stay fast.
"""
import smtplib
import socket
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.email_service import (  # noqa: E402
    EmailConfig,
    EmailService,
    EmailValidationError,
    SmtpEmailProvider,
    SmtpStageError,
    mask_email,
    validate_recipient_email,
)

PDF_BYTES = b"%PDF-1.4 fake pdf bytes for testing only"


def _enabled_config(**overrides) -> EmailConfig:
    cfg = EmailConfig()
    cfg.enabled = True
    cfg.provider = "smtp"
    cfg.smtp_host = "smtp.example.com"
    cfg.smtp_port = 587
    cfg.smtp_username = "user@example.com"
    cfg.smtp_password = "super-secret-password"
    cfg.smtp_use_tls = True
    cfg.from_email = "care@example.com"
    cfg.from_name = "UC07 Care Management"
    cfg.timeout_seconds = 5.0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ===========================================================================
# Layer 1 -- SmtpEmailProvider: real staged execution, mocked smtplib.SMTP
# ===========================================================================

def _mock_smtp(**method_side_effects) -> MagicMock:
    """A MagicMock standing in for an smtplib.SMTP instance. Each kwarg
    names a method (connect/ehlo/starttls/login/send_message/quit/close)
    and its side_effect (an exception instance/class, or a callable)."""
    instance = MagicMock()
    for method, effect in method_side_effects.items():
        getattr(instance, method).side_effect = effect
    return instance


def _provider(**overrides) -> SmtpEmailProvider:
    defaults = dict(host="smtp.example.com", port=587, username="user@example.com", password="pw", use_tls=True)
    defaults.update(overrides)
    return SmtpEmailProvider(**defaults)


class TestSmtpEmailProviderStagedExecution:
    def test_successful_send_runs_all_stages_in_order_then_quits(self):
        instance = _mock_smtp()
        with patch("services.email_service.smtplib.SMTP", return_value=instance) as smtp_cls:
            timing = _provider().send(EmailMessage(), timeout_seconds=7.5)

        # host/port MUST be passed directly to the constructor (this IS
        # the STARTTLS regression fix -- see SmtpEmailProvider._connect's
        # docstring): smtplib.SMTP.__init__ is the only place that sets
        # the instance's internal `_host`, which starttls() later reads
        # for TLS server_hostname/SNI. A separate smtp.connect(host,
        # port) call afterward does NOT update it.
        smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=7.5)
        instance.connect.assert_not_called()
        assert instance.ehlo.call_count == 2  # once pre-TLS, once post-TLS (RFC 3207)
        instance.starttls.assert_called_once()
        instance.login.assert_called_once_with("user@example.com", "pw")
        instance.send_message.assert_called_once()
        instance.quit.assert_called_once()
        assert timing.connection_ms >= 0
        assert timing.send_ms >= 0

    def test_exact_protocol_order_smtp_ehlo_starttls_ehlo_login_send(self):
        """Explicitly proves the exact required sequence for port 587
        (STARTTLS regression fix + prior hardening, combined): construct
        with host/port -> EHLO -> STARTTLS (default, verified context)
        -> EHLO again (RFC 3207) -> LOGIN -> SEND -- in that order, with
        nothing extra in between, and never smtplib.SMTP_SSL."""
        instance = _mock_smtp()
        call_order: list[str] = []
        for name in ("ehlo", "starttls", "login", "send_message"):
            getattr(instance, name).side_effect = lambda *a, _n=name, **k: call_order.append(_n)

        with patch("services.email_service.smtplib.SMTP", return_value=instance) as smtp_cls:
            _provider().send(EmailMessage(), timeout_seconds=5)

        smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=5)
        assert call_order == ["ehlo", "starttls", "ehlo", "login", "send_message"]
        instance.connect.assert_not_called()  # host/port went to the constructor, not a separate connect()

    def test_no_username_skips_authenticate_stage(self):
        instance = _mock_smtp()
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            _provider(username="").send(EmailMessage(), timeout_seconds=5)
        instance.login.assert_not_called()

    def test_use_tls_false_skips_starttls_and_second_ehlo(self):
        instance = _mock_smtp()
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            _provider(use_tls=False).send(EmailMessage(), timeout_seconds=5)
        instance.starttls.assert_not_called()
        assert instance.ehlo.call_count == 1

    # ---- 2. connection timeout ----

    def test_2_connection_timeout_categorized_as_connect_stage(self):
        # "connect" now happens INSIDE the smtplib.SMTP(...) constructor
        # call (see _connect()'s docstring) -- so a connect-stage failure
        # is simulated via the constructor itself raising, not via a
        # separate instance.connect() mock.
        with patch("services.email_service.smtplib.SMTP", side_effect=socket.timeout("timed out")):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "connect"
        assert exc_info.value.error_code == "TIMEOUT"

    # ---- 3. authentication failure ----

    def test_3_authentication_failure_categorized(self):
        instance = _mock_smtp(login=smtplib.SMTPAuthenticationError(535, b"bad credentials"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "authenticate"
        assert exc_info.value.error_code == "AUTH_FAILED"
        assert exc_info.value.smtp_code == 535
        assert "bad credentials" not in exc_info.value.safe_message

    # ---- 4. STARTTLS failure ----

    def test_4_starttls_failure_categorized(self):
        instance = _mock_smtp(starttls=ssl.SSLError("handshake failure"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "starttls"
        assert exc_info.value.error_code == "TLS_FAILED"
        assert exc_info.value.exception_type == "SSLError"

    # ---- STARTTLS certificate verification failure (requested follow-up) ----

    def test_starttls_certificate_verification_failure_categorized(self):
        """ssl.SSLCertVerificationError is a SUBCLASS of ssl.SSLError --
        must be recognized as its own, more specific case (a distinct
        safe_message mentioning "certificate"), not silently absorbed
        into the generic TLS_FAILED wording, and never previously
        collapsed into UNKNOWN_PROVIDER_ERROR."""
        cert_error = ssl.SSLCertVerificationError(
            1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate",
        )
        instance = _mock_smtp(starttls=cert_error)
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "starttls"
        assert exc_info.value.error_code == "TLS_FAILED"
        assert exc_info.value.exception_type == "SSLCertVerificationError"
        assert "certificate" in exc_info.value.safe_message.lower()
        # never the raw OpenSSL error text (could vary by platform/lib
        # version and is not something a care manager needs to see)
        assert "CERTIFICATE_VERIFY_FAILED" not in exc_info.value.safe_message

    def test_starttls_smtp_not_supported_error_categorized(self):
        """Section: SMTPNotSupportedError during STARTTLS specifically
        (server doesn't advertise the STARTTLS extension) -- was already
        handled before the diagnostic-logging follow-up, re-asserted
        here alongside the certificate-failure test per the explicit
        request to cover both."""
        instance = _mock_smtp(starttls=smtplib.SMTPNotSupportedError("STARTTLS extension not supported by server."))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "starttls"
        assert exc_info.value.error_code == "TLS_FAILED"
        assert exc_info.value.exception_type == "SMTPNotSupportedError"

    def test_starttls_connection_reset_categorized_as_connection_failed(self):
        instance = _mock_smtp(starttls=ConnectionResetError("connection reset by peer"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "starttls"
        assert exc_info.value.error_code == "CONNECTION_FAILED"
        assert exc_info.value.exception_type == "ConnectionResetError"

    def test_starttls_timeout_categorized(self):
        instance = _mock_smtp(starttls=socket.timeout("timed out"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "starttls"
        assert exc_info.value.error_code == "TIMEOUT"
        assert exc_info.value.exception_type in ("timeout", "TimeoutError")

    def test_starttls_unrecognized_exception_still_tagged_with_real_class_name(self):
        """Even a genuinely unmapped exception type must still carry its
        real class name in `exception_type` -- this is the actual fix
        for the reported "stage=starttls result=UNKNOWN_PROVIDER_ERROR"
        with no further diagnostic information: the error_code stays a
        coarse UNKNOWN_PROVIDER_ERROR (a bounded, small set is what the
        API/frontend need), but the log line is no longer a dead end."""

        class _WeirdPlatformError(Exception):
            pass

        instance = _mock_smtp(starttls=_WeirdPlatformError("something platform-specific"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "starttls"
        assert exc_info.value.error_code == "UNKNOWN_PROVIDER_ERROR"
        assert exc_info.value.exception_type == "_WeirdPlatformError"

    # ---- 5. recipient rejection ----

    def test_5_recipient_rejection_categorized(self):
        instance = _mock_smtp(
            send_message=smtplib.SMTPRecipientsRefused({"bad@example.com": (550, b"no such user")}),
        )
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "send"
        assert exc_info.value.error_code == "RECIPIENT_REJECTED"

    # ---- 6. sender rejection ----

    def test_6_sender_rejection_categorized(self):
        instance = _mock_smtp(send_message=smtplib.SMTPSenderRefused(550, b"bad sender", "from@example.com"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "send"
        assert exc_info.value.error_code == "SENDER_REJECTED"
        assert exc_info.value.smtp_code == 550

    # ---- 7. message rejection (permanent, 5xx during DATA) ----

    def test_7_message_rejection_permanent(self):
        instance = _mock_smtp(send_message=smtplib.SMTPDataError(550, b"message content rejected"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "send"
        assert exc_info.value.error_code == "MESSAGE_REJECTED"
        assert exc_info.value.smtp_code == 550

    # ---- 8. temporary 4xx provider error ----

    def test_8_temporary_4xx_during_send(self):
        instance = _mock_smtp(send_message=smtplib.SMTPDataError(451, b"temporary local problem"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "send"
        assert exc_info.value.error_code == "PROVIDER_TEMPORARY_ERROR"
        assert exc_info.value.smtp_code == 451

    def test_8b_421_categorized_as_rate_limited(self):
        instance = _mock_smtp(send_message=smtplib.SMTPDataError(421, b"too many connections"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.error_code == "RATE_LIMITED"

    # ---- 9. permanent 5xx provider error (general, not DATA-specific) ----

    def test_9_permanent_5xx_general_response_exception(self):
        instance = _mock_smtp(ehlo=smtplib.SMTPResponseException(550, b"not permitted"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "ehlo"
        assert exc_info.value.error_code == "PROVIDER_PERMANENT_ERROR"

    def test_temporary_4xx_general_response_exception(self):
        instance = _mock_smtp(ehlo=smtplib.SMTPResponseException(452, b"insufficient storage"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.error_code == "PROVIDER_TEMPORARY_ERROR"

    def test_dns_failure_categorized_as_connection_failed(self):
        with patch("services.email_service.smtplib.SMTP", side_effect=socket.gaierror("Name or service not known")):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "connect"
        assert exc_info.value.error_code == "CONNECTION_FAILED"

    def test_connection_refused_categorized(self):
        with patch("services.email_service.smtplib.SMTP", side_effect=ConnectionRefusedError("refused")):
            with pytest.raises(SmtpStageError) as exc_info:
                _provider().send(EmailMessage(), timeout_seconds=5)
        assert exc_info.value.stage == "connect"
        assert exc_info.value.error_code == "CONNECTION_FAILED"

    # ---- 10. connection cleanup after failure ----

    def test_10_connection_cleanup_after_failure_calls_quit(self):
        instance = _mock_smtp(login=smtplib.SMTPAuthenticationError(535, b"bad"))
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError):
                _provider().send(EmailMessage(), timeout_seconds=5)
        instance.quit.assert_called_once()

    def test_10_connection_cleanup_falls_back_to_close_if_quit_fails(self):
        instance = _mock_smtp(
            login=smtplib.SMTPAuthenticationError(535, b"bad"),
            quit=smtplib.SMTPServerDisconnected("already gone"),
        )
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError):
                _provider().send(EmailMessage(), timeout_seconds=5)
        instance.quit.assert_called_once()
        instance.close.assert_called_once()

    def test_10_cleanup_never_raises_even_if_close_also_fails(self):
        # A post-connect stage failure (ehlo) -- cleanup only runs for a
        # `smtp` object that was actually constructed (see _connect()'s
        # docstring: a connect-stage failure means no object exists yet,
        # nothing to clean up, same as the original pre-regression code).
        instance = _mock_smtp(
            ehlo=socket.timeout("timed out"),
            quit=RuntimeError("boom"),
            close=RuntimeError("boom again"),
        )
        with patch("services.email_service.smtplib.SMTP", return_value=instance):
            with pytest.raises(SmtpStageError):  # not the cleanup RuntimeError
                _provider().send(EmailMessage(), timeout_seconds=5)
        instance.quit.assert_called_once()
        instance.close.assert_called_once()

    # ---- 12. timeout value actually applied ----

    def test_12_timeout_value_passed_to_smtp_constructor(self):
        instance = _mock_smtp()
        with patch("services.email_service.smtplib.SMTP", return_value=instance) as smtp_cls:
            _provider().send(EmailMessage(), timeout_seconds=42.0)
        smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=42.0)


# ===========================================================================
# Layer 2 -- EmailService: validation, retry policy, result mapping.
# The fake provider raises SmtpStageError directly, matching the real
# contract EmailProvider.send() is documented to uphold.
# ===========================================================================

class _FakeProvider:
    """Returns/raises exactly what the test tells it to, once per call,
    in order -- lets a test simulate "fails once, then succeeds"."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def send(self, message, timeout_seconds):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _service_with_fake_provider(outcomes, **config_overrides) -> tuple[EmailService, _FakeProvider]:
    service = EmailService(_enabled_config(**config_overrides))
    fake = _FakeProvider(outcomes)
    service._build_provider = lambda: fake  # type: ignore[method-assign]
    return service, fake


from services.email_service import SmtpSendTiming  # noqa: E402

_SUCCESS = SmtpSendTiming(connection_ms=10.0, send_ms=5.0)


class TestEmailServiceValidation:
    def test_disabled_email_never_attempts_a_connection(self):
        service, fake = _service_with_fake_provider([_SUCCESS])
        service._config.enabled = False
        with patch("services.email_service.smtplib.SMTP") as smtp_cls:
            result = service.send_report_email(
                to_email="member@example.com", subject="Subject", body="Body",
                attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
            )
        smtp_cls.assert_not_called()
        assert result.sent is False
        assert result.error_code == "EMAIL_DISABLED"

    @pytest.mark.parametrize("bad_address", ["not-an-email", "", "a@b", "user@@example.com"])
    def test_7_invalid_recipient_rejected(self, bad_address):
        with pytest.raises(EmailValidationError):
            validate_recipient_email(bad_address)

    def test_7_header_injection_rejected(self):
        with pytest.raises(EmailValidationError):
            validate_recipient_email("victim@example.com\r\nBcc: attacker@evil.example")

    def test_invalid_recipient_never_builds_a_provider(self):
        service, fake = _service_with_fake_provider([_SUCCESS])
        result = service.send_report_email(
            to_email="not-an-email", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert result.sent is False
        assert result.error_code == "INVALID_INPUT"
        assert fake.calls == 0

    def test_missing_from_email_handled(self):
        service, fake = _service_with_fake_provider([_SUCCESS], from_email="")
        result = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert result.sent is False
        assert result.error_code == "NOT_CONFIGURED"
        assert fake.calls == 0

    def test_missing_smtp_host_handled(self):
        service, fake = _service_with_fake_provider([_SUCCESS], smtp_host="")
        result = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert result.sent is False
        assert result.error_code == "NOT_CONFIGURED"
        assert fake.calls == 0


class TestEmailServiceRetryPolicy:
    """13-15: transient retry, permanent-not-retried, ambiguous-send
    never retried (Section 9)."""

    def test_1_valid_send_succeeds(self):
        service, fake = _service_with_fake_provider([_SUCCESS])
        result = service.send_report_email(
            to_email="member@example.com", subject="Your Care Navigation Summary",
            body="Hello, please see the attached report.",
            attachment_bytes=PDF_BYTES, attachment_filename="Member_Care_Navigation_Report_M1.pdf",
        )
        assert result.sent is True
        assert result.error_code is None
        assert fake.calls == 1

    def test_13_transient_pre_send_failure_retried_once_then_succeeds(self):
        service, fake = _service_with_fake_provider([
            SmtpStageError("connect", "TIMEOUT", "Timed out during the 'connect' step."),
            _SUCCESS,
        ])
        with patch("services.email_service.time.sleep") as sleep_mock:
            result = service.send_report_email(
                to_email="member@example.com", subject="Subject", body="Body",
                attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
            )
        assert result.sent is True
        assert fake.calls == 2
        sleep_mock.assert_called_once()

    def test_13_transient_4xx_during_send_is_retried(self):
        service, fake = _service_with_fake_provider([
            SmtpStageError("send", "PROVIDER_TEMPORARY_ERROR", "The email provider temporarily rejected this request.", smtp_code=451),
            _SUCCESS,
        ])
        with patch("services.email_service.time.sleep"):
            result = service.send_report_email(
                to_email="member@example.com", subject="Subject", body="Body",
                attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
            )
        assert result.sent is True
        assert fake.calls == 2

    def test_13_retry_never_exceeds_one_additional_attempt(self):
        service, fake = _service_with_fake_provider([
            SmtpStageError("connect", "TIMEOUT", "t1"),
            SmtpStageError("connect", "TIMEOUT", "t2"),
        ])
        with patch("services.email_service.time.sleep"):
            result = service.send_report_email(
                to_email="member@example.com", subject="Subject", body="Body",
                attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
            )
        assert result.sent is False
        assert fake.calls == 2  # 1 initial + 1 retry, never more

    def test_14_permanent_error_is_not_retried(self):
        service, fake = _service_with_fake_provider([
            SmtpStageError("authenticate", "AUTH_FAILED", "The email provider rejected the configured credentials.", smtp_code=535),
        ])
        with patch("services.email_service.time.sleep") as sleep_mock:
            result = service.send_report_email(
                to_email="member@example.com", subject="Subject", body="Body",
                attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
            )
        assert result.sent is False
        assert result.error_code == "AUTH_FAILED"
        assert fake.calls == 1
        sleep_mock.assert_not_called()

    @pytest.mark.parametrize("error_code", ["RECIPIENT_REJECTED", "SENDER_REJECTED", "MESSAGE_REJECTED", "PROVIDER_PERMANENT_ERROR", "TLS_FAILED", "UNKNOWN_PROVIDER_ERROR"])
    def test_14_every_permanent_code_is_not_retried(self, error_code):
        service, fake = _service_with_fake_provider([
            SmtpStageError("send", error_code, "safe message"),
        ])
        result = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert result.sent is False
        assert fake.calls == 1

    def test_15_ambiguous_send_timeout_never_retried(self):
        """A TIMEOUT during the 'send' stage means we cannot tell if the
        server already queued the message -- retrying here could send a
        duplicate, so it must NEVER be retried even though TIMEOUT is
        retryable in every other stage."""
        service, fake = _service_with_fake_provider([
            SmtpStageError("send", "TIMEOUT", "Timed out during the 'send' step."),
        ])
        result = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert result.sent is False
        assert fake.calls == 1
        assert "duplicate" in result.message.lower()

    def test_15_ambiguous_connection_failed_during_send_never_retried(self):
        service, fake = _service_with_fake_provider([
            SmtpStageError("send", "CONNECTION_FAILED", "The email server closed the connection unexpectedly."),
        ])
        result = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert result.sent is False
        assert fake.calls == 1

    def test_11_second_independent_send_succeeds_after_a_prior_failure(self):
        """A failed EmailService.send_report_email() call must not leave
        state that breaks a LATER, independent call (a fresh
        _build_provider() per call already guarantees this by
        construction -- this test proves it observably)."""
        service, fake = _service_with_fake_provider([
            SmtpStageError("authenticate", "AUTH_FAILED", "bad creds"),
        ])
        first = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert first.sent is False

        fake._outcomes = [_SUCCESS]
        second = service.send_report_email(
            to_email="member2@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        assert second.sent is True


class TestEmailServiceResultAndAttachment:
    def test_2_3_4_same_pdf_correct_mime_and_filename_attached(self):
        service = EmailService(_enabled_config())
        captured: dict = {}

        class _CapturingProvider:
            def send(self, message, timeout_seconds):
                captured["message"] = message
                return _SUCCESS

        service._build_provider = lambda: _CapturingProvider()  # type: ignore[method-assign]
        result = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="Member_Care_Navigation_Report_M1.pdf",
        )
        assert result.sent is True
        message: EmailMessage = captured["message"]
        attachments = list(message.iter_attachments())
        assert len(attachments) == 1
        attachment = attachments[0]
        assert attachment.get_content_type() == "application/pdf"
        assert attachment.get_filename() == "Member_Care_Navigation_Report_M1.pdf"
        assert attachment.get_payload(decode=True) == PDF_BYTES
        assert message["Date"] is not None
        assert message["Message-ID"] is not None

    def test_5_6_editable_subject_and_body_used_verbatim(self):
        service = EmailService(_enabled_config())
        captured: dict = {}

        class _CapturingProvider:
            def send(self, message, timeout_seconds):
                captured["message"] = message
                return _SUCCESS

        service._build_provider = lambda: _CapturingProvider()  # type: ignore[method-assign]
        service.send_report_email(
            to_email="member@example.com",
            subject="A custom, care-manager-edited subject line",
            body="A custom, care-manager-edited message body.",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        message: EmailMessage = captured["message"]
        assert message["Subject"] == "A custom, care-manager-edited subject line"
        body_text = message.get_body(preferencelist=("plain",)).get_content()
        assert "A custom, care-manager-edited message body." in body_text

    def test_22_frontend_never_sees_a_raw_smtp_code_or_python_exception_name(self):
        service, fake = _service_with_fake_provider([
            SmtpStageError("send", "MESSAGE_REJECTED", "The email provider rejected the message.", smtp_code=550),
        ])
        result = service.send_report_email(
            to_email="member@example.com", subject="Subject", body="Body",
            attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
        )
        for leak in ("550", "SMTPDataError", "SMTPException", "Traceback", "smtplib"):
            assert leak not in result.message


# ---- 19-20. secrets never returned or logged ----

def test_19_result_never_contains_password_or_credentials():
    service, fake = _service_with_fake_provider([
        SmtpStageError("authenticate", "AUTH_FAILED", "The email provider rejected the configured credentials."),
    ])
    result = service.send_report_email(
        to_email="member@example.com", subject="Subject", body="Body",
        attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
    )
    result_text = f"{result.sent} {result.provider} {result.message} {result.error_code}"
    assert "super-secret-password" not in result_text


def test_20_smtp_send_log_line_never_contains_password(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="uc07.communication.email")
    service, fake = _service_with_fake_provider([
        SmtpStageError("authenticate", "AUTH_FAILED", "bad creds", smtp_code=535),
    ])
    service.send_report_email(
        to_email="member@example.com", subject="Subject", body="Body",
        attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
    )
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "super-secret-password" not in log_text
    assert "stage=authenticate" in log_text
    assert "result=AUTH_FAILED" in log_text
    assert "smtp_code=535" in log_text


def test_20_successful_send_log_line_has_no_message_content(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="uc07.communication.email")
    service, fake = _service_with_fake_provider([_SUCCESS])
    service.send_report_email(
        to_email="member@example.com", subject="A very specific subject nobody should log",
        body="A very specific body nobody should log.",
        attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
    )
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "A very specific subject nobody should log" not in log_text
    assert "A very specific body nobody should log" not in log_text
    assert "result=SENT" in log_text


def test_20_log_line_includes_exception_type_for_diagnosability(caplog):
    """The actual fix for a repeated 'stage=starttls
    result=UNKNOWN_PROVIDER_ERROR' report with no further information:
    the raw exception's CLASS NAME (never its message/args) is now
    always present in the log line, even when the coarse `error_code`
    taxonomy can't distinguish it further."""
    import logging
    caplog.set_level(logging.WARNING, logger="uc07.communication.email")
    service, fake = _service_with_fake_provider([
        SmtpStageError(
            "starttls", "TLS_FAILED", "Could not verify the email server's TLS certificate.",
            exception_type="SSLCertVerificationError",
        ),
    ])
    service.send_report_email(
        to_email="member@example.com", subject="Subject", body="Body",
        attachment_bytes=PDF_BYTES, attachment_filename="report.pdf",
    )
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "stage=starttls" in log_text
    assert "result=TLS_FAILED" in log_text
    assert "exc_type=SSLCertVerificationError" in log_text


# ---- TLS certificate verification must never be weakened (Section 4/5) ----

def test_tls_verification_is_never_weakened():
    source = (BACKEND_DIR / "services" / "email_service.py").read_text(encoding="utf-8")
    for forbidden in ("_create_unverified_context", "CERT_NONE", "check_hostname = False", "check_hostname=False"):
        assert forbidden not in source, f"email_service.py must never weaken TLS verification via {forbidden}"
    # the ONLY SSL context this module ever constructs is the secure,
    # untouched stdlib default (hostname checking + certificate
    # verification both on)
    assert "ssl.create_default_context()" in source


def test_starttls_uses_create_default_context_with_no_overrides():
    """Guards specifically against a future change quietly passing
    `check_hostname=False`/`verify_mode=CERT_NONE` into the context used
    for the real Gmail STARTTLS flow -- verified by SPYING on the real
    ssl.create_default_context (not mocking it away), so this fails if
    the call is ever made with weakening arguments."""
    instance = _mock_smtp()
    with patch("services.email_service.smtplib.SMTP", return_value=instance), \
         patch("services.email_service.ssl.create_default_context", wraps=ssl.create_default_context) as ctx_spy:
        _provider().send(EmailMessage(), timeout_seconds=5)
    ctx_spy.assert_called_once_with()  # no arguments -- fully default, unweakened
    used_context = instance.starttls.call_args.kwargs["context"]
    assert used_context.verify_mode == ssl.CERT_REQUIRED
    assert used_context.check_hostname is True


# ---- mask_email ----

def test_mask_email_hides_local_part():
    assert mask_email("jordan.lee@example.com").startswith("j***@")
    assert mask_email("jordan.lee@example.com").endswith("example.com")


# ---- no model/agent decision authority ----

def test_no_model_decision_authority_imports():
    source = (BACKEND_DIR / "services" / "email_service.py").read_text(encoding="utf-8")
    for forbidden in ("risk_detection", "care_navigation", "safety_policy", "orchestrator", "model_explainability"):
        assert forbidden not in source, f"email_service.py must never import {forbidden}"


def test_smtp_ssl_never_used_for_port_587_starttls_flow():
    """Guards against accidentally reintroducing smtplib.SMTP_SSL (465's
    implicit-TLS semantics) into the 587/STARTTLS code path -- checked
    as actual usage (a call), not merely a mention in prose/comments."""
    source = (BACKEND_DIR / "services" / "email_service.py").read_text(encoding="utf-8")
    assert "smtplib.SMTP_SSL(" not in source
