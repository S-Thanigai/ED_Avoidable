"""
safety_context_schema.py
--------------------------
Pydantic schema for the `current_safety_context` JSON payload accepted
by POST /uc07/decide (backend/main.py). A field the caller OMITS from a
member's entry stays None (ABSENT) here -- this schema never invents a
default of 0/false for an omitted key; only a key the caller explicitly
included is ever coerced to 0 or 1
(docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md sections 16-17).

This module owns ONLY schema validation of the raw JSON. Turning a
validated entry into the runtime CurrentSafetyContext dataclass
(backend/agents/contracts.py) that the Safety & Policy Agent actually
consumes is main.py's job (a one-line construction per entry) -- keeping
this module free of any agent-orchestration import, and keeping
main.py's route handler free of validation business logic.
"""
from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, RootModel, field_validator

BINARY_VALID_VALUES = (0, 1)
TRIAGE_VALID_VALUES = (1, 2, 3, 4, 5)


def _reject_non_finite(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"must be a finite value, got {value!r}")
    return value


class SafetyContextEntry(BaseModel):
    """One member's current-safety-context entry. `extra="forbid"`:
    an unrecognized key (e.g. a typo like `trige_level`) is rejected
    with a clear 422, never silently ignored."""

    model_config = ConfigDict(extra="forbid")

    red_flag: Optional[int] = None
    icu: Optional[int] = None
    admitted: Optional[int] = None
    major_procedure: Optional[int] = None
    triage_level: Optional[int] = None

    @field_validator("red_flag", "icu", "admitted", "major_procedure", mode="before")
    @classmethod
    def _validate_binary(cls, value):
        if value is None:
            return None
        value = _reject_non_finite(value)
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, (int, float)) or value not in BINARY_VALID_VALUES:
            raise ValueError(f"must be 0 or 1, got {value!r}")
        return int(value)

    @field_validator("triage_level", mode="before")
    @classmethod
    def _validate_triage(cls, value):
        if value is None:
            return None
        value = _reject_non_finite(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in TRIAGE_VALID_VALUES:
            raise ValueError(f"must be an integer 1-5, got {value!r}")
        return int(value)


class SafetyContextPayload(RootModel[dict[str, SafetyContextEntry]]):
    """The full `current_safety_context` JSON object: member_id -> entry.
    A member_id absent from this payload entirely is a DIFFERENT thing
    from an entry present with every field null -- both currently
    resolve to the same ABSENT completeness state once converted to a
    CurrentSafetyContext, but the distinction is preserved up to that
    point (a caller can address a member with an explicitly empty `{}`
    entry, or omit the member_id key entirely; main.py treats both as
    "no context supplied for this member")."""
