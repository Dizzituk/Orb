# FILE: tests/conftest.py
"""
Pytest configuration for Orb test suite.

Configures:
- pytest-asyncio for async test support
"""
import pytest

# Configure pytest-asyncio to use auto mode
pytest_plugins = ["pytest_asyncio"]
