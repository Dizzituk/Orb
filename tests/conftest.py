# FILE: tests/conftest.py
# Purpose: Pytest configuration for Orb test suite.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-07-02 (LANE E: hermetic embedding-provider pin)
"""
Pytest configuration for Orb test suite.

Configures:
- pytest-asyncio for async test support
- LANE E: pins the embedding provider env to the historical gemini defaults
  BEFORE any test can import main.py. main's load_dotenv(.env) does not
  override pre-set variables, so without this pin the first test that
  imports main leaks the LIVE .env's EMBEDDINGS_TEXT_PROVIDER=local into
  the shared process and every later embedding test silently changes
  provider (rows stamp local, queries stay gemini → empty results).
  Lane E's own tests opt into local explicitly via monkeypatch.setenv.
"""
import os

import pytest

os.environ.setdefault("EMBEDDINGS_TEXT_PROVIDER", "gemini")
os.environ.setdefault("EMBEDDINGS_TEXT_QUERY_CUTOVER", "0")
os.environ.setdefault("EMBEDDINGS_MULTIMODAL_PROVIDER", "gemini")

# Configure pytest-asyncio to use auto mode
pytest_plugins = ["pytest_asyncio"]
