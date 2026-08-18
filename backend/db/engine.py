"""
engine.py
---------
SQLAlchemy engine + session factory for Azure SQL. Lazily constructed
(mirrors the existing lazy UC07Orchestrator pattern in backend/main.py)
so the app can still start even if the database is temporarily
unreachable or unconfigured -- failures surface as a clean error on
first actual use, not a startup crash.
"""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import DatabaseNotConfiguredError, load_azure_sql_config

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        config = load_azure_sql_config()
        _engine = create_engine(
            config.sqlalchemy_url(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            pool_recycle=1800,
            fast_executemany=True,
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()  # ensures _SessionLocal is initialized
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields one Session per request, always closed
    afterward. Callers are responsible for commit()/rollback() -- this
    dependency does not auto-commit, so a handler that raises leaves the
    transaction rolled back by SQLAlchemy's session teardown semantics
    (nothing is implicitly persisted on error)."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def check_connection() -> bool:
    """Cheap pooled `SELECT 1` used only by GET /health. Never raises --
    any failure (unconfigured, unreachable, auth failure) returns False.
    Never logs or returns connection details."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except DatabaseNotConfiguredError:
        return False
    except Exception:
        return False


def is_configured() -> bool:
    try:
        load_azure_sql_config()
        return True
    except DatabaseNotConfiguredError:
        return False
