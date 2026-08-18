"""
users.py
--------
User account repository. Password hashing lives in backend/auth.py --
this module only stores/retrieves the already-hashed value, never a
plaintext password.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import User


def get_user_by_email(db: Session, email: str) -> User | None:
    normalized = email.strip().lower()
    return db.execute(select(User).where(User.email == normalized)).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def create_user(db: Session, *, email: str, password_hash: str) -> User:
    user = User(email=email.strip().lower(), password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
