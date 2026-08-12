"""Shared database location/engine helpers for every tool module.

Tool functions get imported from lots of different places over the life of
this project: a notebook running with `solution/` as its working directory
today, a pytest run tomorrow, and eventually a standalone MCP server process
that could be launched from anywhere. A path like ``"data/external/cultpass.db"``
only resolves correctly in the first case, so every DB path here is anchored
to *this file's* location instead of the current working directory.
"""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# This file lives at solution/agentic/tools/db.py, so two levels up is the
# solution/ root (agentic/tools -> agentic -> solution).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CULTPASS_DB_PATH = PROJECT_ROOT / "data" / "external" / "cultpass.db"
UDAHUB_DB_PATH = PROJECT_ROOT / "data" / "core" / "udahub.db"

# One SQLAlchemy engine per DB file, reused across calls instead of being
# recreated (and reopening a connection pool) on every tool invocation.
_engines: dict[str, Engine] = {}


def _get_engine(db_path: Path) -> Engine:
    key = str(db_path)
    if key not in _engines:
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {db_path}. Run the setup notebooks "
                "(01_external_db_setup / 02_core_db_setup) first."
            )
        _engines[key] = create_engine(f"sqlite:///{db_path}", echo=False)
    return _engines[key]


def get_cultpass_engine() -> Engine:
    """Engine for the external CultPass database (customer's own system)."""
    return _get_engine(CULTPASS_DB_PATH)


def get_udahub_engine() -> Engine:
    """Engine for UDA-Hub's own core database (accounts, tickets, knowledge)."""
    return _get_engine(UDAHUB_DB_PATH)
