"""
pytest configuration for backend/tests.

Adds backend/pit, backend/modeling, and backend/agents to sys.path so
tests can use flat imports (`from encounter_classification import ...`,
`from feature_spec import ...`, `from orchestrator import ...`) matching
the same sibling-module import convention already used throughout
backend/ (e.g. backend/main.py's `from feature_engineering import
extract_features`).
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
for subdir in ("pit", "modeling", "agents"):
    p = str(BACKEND_DIR / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)
