"""
config.py
---------
Azure SQL configuration, read exclusively from environment variables
(backend/.env in local dev; real env vars in any real deployment -- see
backend/main.py's load_dotenv() call, which runs before this module is
ever imported).

NEVER logs, returns, or otherwise exposes AZURE_SQL_PASSWORD or the
assembled connection string (which embeds it) -- see `configured` /
`describe()` below, which are the only safe things to surface to a
caller (e.g. GET /health).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when a database operation is attempted but one or more
    required AZURE_SQL_* environment variables are missing. Never
    includes the values of any variable that WAS set, in case a caller
    logs the exception message."""


@dataclass(frozen=True)
class AzureSqlConfig:
    server: str
    port: str
    database: str
    username: str
    password: str
    encrypt: bool
    trust_server_certificate: bool
    connection_timeout: int

    @property
    def configured(self) -> bool:
        return bool(self.server and self.database and self.username and self.password)

    def sqlalchemy_url(self) -> str:
        """Builds a `mssql+pyodbc://` SQLAlchemy URL. The password is
        URL-quoted (it may contain characters like '@' or '/') and never
        logged -- callers must treat the return value itself as a secret."""
        driver = quote_plus("ODBC Driver 18 for SQL Server")
        user = quote_plus(self.username)
        pwd = quote_plus(self.password)
        encrypt = "yes" if self.encrypt else "no"
        trust_cert = "yes" if self.trust_server_certificate else "no"
        return (
            f"mssql+pyodbc://{user}:{pwd}@{self.server}:{self.port}/{self.database}"
            f"?driver={driver}&Encrypt={encrypt}&TrustServerCertificate={trust_cert}"
            f"&Connection+Timeout={self.connection_timeout}"
        )


def load_azure_sql_config() -> AzureSqlConfig:
    """Reads AZURE_SQL_* from the process environment. Does not validate
    reachability -- only that the required fields are present. Raises
    DatabaseNotConfiguredError (never a raw KeyError) if a required
    field is missing, so callers get a clean, safe error message."""
    server = os.environ.get("AZURE_SQL_SERVER", "")
    database = os.environ.get("AZURE_SQL_DATABASE", "")
    username = os.environ.get("AZURE_SQL_USERNAME", "")
    password = os.environ.get("AZURE_SQL_PASSWORD", "")
    port = os.environ.get("AZURE_SQL_PORT", "1433")
    encrypt = os.environ.get("AZURE_SQL_ENCRYPT", "true").strip().lower() != "false"
    trust_server_certificate = os.environ.get(
        "AZURE_SQL_TRUST_SERVER_CERTIFICATE", "false"
    ).strip().lower() == "true"
    try:
        connection_timeout = int(os.environ.get("AZURE_SQL_CONNECTION_TIMEOUT", "30"))
    except ValueError:
        connection_timeout = 30

    config = AzureSqlConfig(
        server=server,
        port=port,
        database=database,
        username=username,
        password=password,
        encrypt=encrypt,
        trust_server_certificate=trust_server_certificate,
        connection_timeout=connection_timeout,
    )
    if not config.configured:
        missing = [
            name
            for name, value in (
                ("AZURE_SQL_SERVER", server),
                ("AZURE_SQL_DATABASE", database),
                ("AZURE_SQL_USERNAME", username),
                ("AZURE_SQL_PASSWORD", password),
            )
            if not value
        ]
        raise DatabaseNotConfiguredError(
            f"Azure SQL is not configured -- missing environment variable(s): {missing}"
        )
    return config
