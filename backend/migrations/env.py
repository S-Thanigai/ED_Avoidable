"""
Alembic environment -- loads the DB URL from backend/.env via
backend/db/config.py (the same code path the running app uses), so
there is exactly one place that knows how to build an Azure SQL
connection string. Never hard-codes a URL or credential in this file or
in alembic.ini.
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

from alembic import context

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from db.config import load_azure_sql_config  # noqa: E402
from db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Built as a plain Python variable, deliberately NOT passed through
# config.set_main_option()/ConfigParser -- a password containing a "%"
# (percent-encoded by urllib, e.g. "%40" for "@") trips ConfigParser's
# string-interpolation parser (`invalid interpolation syntax`). Every
# database URL used below is built the same way the running app builds
# it (backend/db/config.py) -- never derived from alembic.ini.
DB_URL = load_azure_sql_config().sqlalchemy_url()


def run_migrations_offline() -> None:
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(DB_URL, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
