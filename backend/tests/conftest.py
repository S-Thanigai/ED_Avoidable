"""
pytest configuration for backend/tests.

Adds backend/pit, backend/modeling, and backend/agents to sys.path so
tests can use flat imports (`from encounter_classification import ...`,
`from feature_spec import ...`, `from orchestrator import ...`) matching
the same sibling-module import convention already used throughout
backend/ (e.g. backend/main.py's `from feature_engineering import
extract_features`).
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
for subdir in ("pit", "modeling", "agents"):
    p = str(BACKEND_DIR / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

# `import main` (done by several test modules, e.g. test_uc07_api.py,
# test_genai_explanation_authority.py) triggers backend/main.py's
# load_dotenv(backend/.env) as a side effect. Left alone, a developer's
# own local backend/.env (e.g. GENAI_ENABLED=true, for their own manual
# testing against a real running Ollama) would silently leak into the
# WHOLE pytest session from whichever test module happens to `import
# main` first -- making tests that assume GenAI is off (and every test
# that doesn't explicitly monkeypatch these vars) depend on one
# developer's local machine state, non-deterministically. This runs at
# collection time, before any test module's `import main`, and pins a
# clean, known baseline; load_dotenv()'s override=False means it will
# never overwrite a value already present here. Any test that actually
# wants to exercise a different value still does so explicitly via
# monkeypatch.setenv(...), same as the existing convention throughout
# the GenAI test files -- this only removes the AMBIENT, implicit
# dependency on whatever is in backend/.env right now.
for _key, _default in (
    ("GENAI_ENABLED", "false"),
    ("GENAI_PROVIDER", "groq"),
    ("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    ("OLLAMA_MODEL", "qwen3:8b"),
    ("GENAI_TIMEOUT_SECONDS", "20"),
    # Deliberately pinned to blank, same reasoning as above: a developer's
    # real backend/.env may have a real GROQ_API_KEY for their own manual
    # testing against live Groq. Without this, that key would silently
    # leak into the whole pytest session (since load_dotenv's
    # override=False never overwrites a value already set here) and any
    # test relying on "Groq is unconfigured, so it's skipped" would become
    # non-deterministic depending on one developer's local machine state.
    ("GROQ_API_KEY", ""),
    ("GROQ_MODEL", "llama-3.3-70b-versatile"),
    # Phase 9 -- same reasoning: a developer's real backend/.env may have
    # EMAIL_ENABLED=true and real SMTP_* values for their own manual
    # testing. Pinned off/blank here so no test that assumes email is
    # disabled (the default posture) becomes non-deterministic depending
    # on one developer's local machine state. Tests that want a
    # configured/enabled EmailService construct their own EmailConfig or
    # monkeypatch.setenv(...) explicitly, same convention as GenAI.
    ("EMAIL_ENABLED", "false"),
    ("EMAIL_PROVIDER", "smtp"),
    ("SMTP_HOST", ""),
    ("SMTP_PORT", "587"),
    ("SMTP_USERNAME", ""),
    ("SMTP_PASSWORD", ""),
    ("SMTP_FROM_EMAIL", ""),
    ("SMTP_FROM_NAME", "UC07 Care Management"),
    ("SMTP_USE_TLS", "true"),
    ("EMAIL_TIMEOUT_SECONDS", "30"),
):
    os.environ.setdefault(_key, _default)
