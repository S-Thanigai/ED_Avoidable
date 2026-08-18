"""
sessions.py
-----------
Server-side session repository. The session `id` is an opaque random
token (never a JWT/derivable value) -- see backend/auth.py for creation.
Revocation is instant and server-side (DELETE-equivalent via
revoked_at), unlike a stateless signed cookie.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import UserSession


def create_session(db: Session, *, session_id: str, user_id: int, ttl: dt.timedelta) -> UserSession:
    now = dt.datetime.now(dt.timezone.utc)
    session = UserSession(id=session_id, user_id=user_id, created_at=now, expires_at=now + ttl)
    db.add(session)
    db.commit()
    return session


def get_valid_session(db: Session, session_id: str) -> UserSession | None:
    now = dt.datetime.now(dt.timezone.utc)
    session = db.get(UserSession, session_id)
    if session is None:
        return None
    if session.revoked_at is not None:
        return None
    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=dt.timezone.utc)
    if expires_at < now:
        return None
    return session


def revoke_session(db: Session, session_id: str) -> None:
    session = db.get(UserSession, session_id)
    if session is not None and session.revoked_at is None:
        session.revoked_at = dt.datetime.now(dt.timezone.utc)
        db.commit()
