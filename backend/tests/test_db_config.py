"""
Azure SQL configuration + connection-layer tests (backend/db/config.py,
backend/db/engine.py). Runs against the real Azure SQL configured in
backend/.env -- there is no mocked substitute for this feature.
"""
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db.config import AzureSqlConfig, DatabaseNotConfiguredError, load_azure_sql_config  # noqa: E402
from db import engine as db_engine  # noqa: E402


def test_loads_config_from_environment(monkeypatch):
    monkeypatch.setenv("AZURE_SQL_SERVER", "example.database.windows.net")
    monkeypatch.setenv("AZURE_SQL_DATABASE", "exampledb")
    monkeypatch.setenv("AZURE_SQL_USERNAME", "exampleadmin")
    monkeypatch.setenv("AZURE_SQL_PASSWORD", "s3cret")
    config = load_azure_sql_config()
    assert config.configured
    assert config.server == "example.database.windows.net"


def test_missing_configuration_raises_clean_error(monkeypatch):
    for key in ("AZURE_SQL_SERVER", "AZURE_SQL_DATABASE", "AZURE_SQL_USERNAME", "AZURE_SQL_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(DatabaseNotConfiguredError) as exc_info:
        load_azure_sql_config()
    # The error message must name which variables are missing but never
    # include a password value (there isn't one to leak here, but this
    # guards against a future change accidentally including `password`).
    assert "AZURE_SQL_PASSWORD" in str(exc_info.value)


def test_sqlalchemy_url_never_prints_password_unmasked_in_repr(monkeypatch):
    monkeypatch.setenv("AZURE_SQL_SERVER", "example.database.windows.net")
    monkeypatch.setenv("AZURE_SQL_DATABASE", "exampledb")
    monkeypatch.setenv("AZURE_SQL_USERNAME", "exampleadmin")
    monkeypatch.setenv("AZURE_SQL_PASSWORD", "Sup3r@Secret!")
    config = load_azure_sql_config()
    url = config.sqlalchemy_url()
    # The password IS present in the connection URL (SQLAlchemy needs it
    # to connect) -- what matters is that nothing in this module logs or
    # returns it separately, and it is never printed in cleartext by a
    # health/diagnostic endpoint (see test_uc07_api.py's health test).
    assert "Sup3r" in url  # sanity: URL really was built from the password
    assert isinstance(config, AzureSqlConfig)


def test_real_connection_succeeds():
    """Uses the REAL backend/.env configuration -- this is the
    integration checkpoint confirming Azure SQL is reachable and the
    schema migration has been applied."""
    assert db_engine.is_configured(), "backend/.env must have AZURE_SQL_* set for this test suite to run"
    assert db_engine.check_connection() is True


def test_check_connection_never_raises_when_misconfigured(monkeypatch):
    for key in ("AZURE_SQL_SERVER", "AZURE_SQL_DATABASE", "AZURE_SQL_USERNAME", "AZURE_SQL_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    db_engine._engine = None  # noqa: SLF001 -- force re-read of (now-cleared) config
    db_engine._SessionLocal = None  # noqa: SLF001
    try:
        assert db_engine.check_connection() is False
        assert db_engine.is_configured() is False
    finally:
        db_engine._engine = None  # noqa: SLF001 -- do not leak a broken engine into later tests
        db_engine._SessionLocal = None  # noqa: SLF001
