# FILE: app/settings/__init__.py
# Purpose: ASTRA Settings Module.
# Called-by: app.db
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
ASTRA Settings Module.

Provides a UI-friendly settings interface for managing:
- API keys (encrypted, stored in DB instead of .env)
- System configuration
- Feature toggles

API keys are encrypted at rest using the existing master key
crypto system (EncryptedText column type).
"""
