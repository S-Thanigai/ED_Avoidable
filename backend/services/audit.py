"""
audit.py
--------
Lightweight communication audit trail (Section 22). This prototype has
no database, so "audit trail" means structured application logging via
the standard `logging` module -- one INFO record per event, on the
`uc07.communication.audit` logger, which backend/main.py's
logging.basicConfig(...) call already routes to stderr (same pattern as
genai_explanation's provider-event logging).

Records ONLY safe metadata: event id, member id, MASKED recipient
email, timestamp, action, report id, provider, result status. NEVER the
SMTP password, any API credential, the full report content, or the full
email body -- see `record_communication_event`'s docstring for the
exact allow-list. Persisting this to a real database/audit store is
documented as a production follow-up in
docs/09_MEMBER_COMMUNICATION_REPORTING.md, not implemented here.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

audit_logger = logging.getLogger("uc07.communication.audit")

AuditAction = Literal["PDF_GENERATED", "EMAIL_SENT", "EMAIL_FAILED"]


def record_communication_event(
    *,
    action: AuditAction,
    member_id: str,
    report_id: str,
    masked_recipient: str | None = None,
    provider: str | None = None,
    result_status: str = "OK",
) -> str:
    """Logs one structured audit line and returns the generated event
    id. `masked_recipient` must already be masked (see
    email_service.mask_email) by the caller -- this function does not
    mask or otherwise transform it, so it must never be passed a raw
    address."""
    event_id = f"AUD-{uuid.uuid4().hex[:12]}"
    audit_logger.info(
        "communication_event event_id=%s action=%s member_id=%s report_id=%s "
        "recipient=%s provider=%s result=%s timestamp=%s",
        event_id, action, member_id, report_id,
        masked_recipient or "-", provider or "-", result_status,
        datetime.now(timezone.utc).isoformat(),
    )
    return event_id
