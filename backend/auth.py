"""
auth.py
-------
Password hashing + session cookie plumbing for the Azure SQL persistence
feature. This module is the ONLY place that decides "who is the
authenticated caller" -- every ownership check downstream (populations,
members, delete) takes that identity as a plain argument and never
re-derives it from anything the client sends in a body/query/path.

Session model: an opaque random token (never a JWT -- nothing about the
user is encoded IN the cookie) stored server-side in the `sessions`
table (backend/db/models.py). This makes logout / expiry instant and
server-enforced, at the cost of one indexed lookup per authenticated
request -- an acceptable trade for a care-management application at this
scale.
"""
from __future__ import annotations

import datetime as dt
import os
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from fastapi import Depends, HTTPException, Request, Response

from db.engine import get_db
from db.models import User
from db.repositories import sessions as sessions_repo
from db.repositories import users as users_repo

SESSION_COOKIE_NAME = "uc07_session"
SESSION_TTL = dt.timedelta(days=7)

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _cookie_secure() -> bool:
    # Local dev runs over plain http -- a Secure cookie would silently
    # never be sent by the browser at all, breaking login. Default false
    # (dev-safe); set COOKIE_SECURE=true for any real (https) deployment.
    return os.environ.get("COOKIE_SECURE", "false").strip().lower() == "true"


def issue_session_cookie(response: Response, db, user_id: int) -> None:
    token = secrets.token_urlsafe(32)
    sessions_repo.create_session(db, session_id=token, user_id=user_id, ttl=SESSION_TTL)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
        path="/",
    )


def clear_session_cookie(response: Response, db, request: Request) -> None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        sessions_repo.revoke_session(db, token)
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def get_current_user(request: Request, db=Depends(get_db)) -> User:
    """The sole FastAPI dependency that establishes identity. Raises 401
    for a missing/invalid/expired/revoked session -- never falls back to
    any caller-supplied user id."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    session = sessions_repo.get_valid_session(db, token)
    if session is None:
        raise HTTPException(status_code=401, detail="Session expired or invalid.")
    user = users_repo.get_user_by_id(db, session.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Session expired or invalid.")
    return user
