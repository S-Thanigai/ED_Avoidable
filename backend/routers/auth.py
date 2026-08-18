"""
routers/auth.py
----------------
Sign up / sign in / sign out / current-user endpoints. See backend/auth.py
for the session/password mechanics this router calls into -- this module
is just the HTTP boundary (request validation, status codes, cookie
wiring via the Response object).
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import auth
from db.engine import get_db
from db.models import User
from db.repositories import users as users_repo

router = APIRouter(prefix="/auth", tags=["Auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=200)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        value = value.strip()
        if not _EMAIL_RE.match(value):
            raise ValueError("must be a valid email address")
        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=200)


class UserResponse(BaseModel):
    id: int
    email: str


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(id=user.id, email=user.email)


@router.post("/signup", response_model=UserResponse, status_code=201)
def signup(payload: SignupRequest, response: Response, db: Session = Depends(get_db)) -> UserResponse:
    existing = users_repo.get_user_by_email(db, payload.email)
    if existing is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    password_hash = auth.hash_password(payload.password)
    try:
        user = users_repo.create_user(db, email=payload.email, password_hash=password_hash)
    except IntegrityError:
        # Race: two signups for the same email at once. Unique constraint
        # catches it even if the pre-check above raced.
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    auth.issue_session_cookie(response, db, user.id)
    return _to_user_response(user)


@router.post("/login", response_model=UserResponse)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> UserResponse:
    user = users_repo.get_user_by_email(db, payload.email)
    # Same generic message whether the email doesn't exist or the
    # password is wrong -- never confirms which via the message.
    if user is None or not auth.verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    auth.issue_session_cookie(response, db, user.id)
    return _to_user_response(user)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> Response:
    # Mutate + return the SAME injected Response instance -- returning a
    # newly-constructed Response here would silently drop the
    # delete_cookie() header just set on `response`.
    auth.clear_session_cookie(response, db, request)
    response.status_code = 204
    return response


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(auth.get_current_user)) -> UserResponse:
    return _to_user_response(current_user)
