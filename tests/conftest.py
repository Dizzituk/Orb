# FILE: tests/conftest.py
# Purpose: Pytest configuration for Orb test suite.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pytest configuration for Orb test suite.

Configures:
- pytest-asyncio for async test support
"""
import pytest

# Configure pytest-asyncio to use auto mode
pytest_plugins = ["pytest_asyncio"]
