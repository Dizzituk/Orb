# FILE: app/pot_spec/errors.py
# Purpose: errors
# Called-by: app.pot_spec.service
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
class SpecError(Exception):
    """Base class for PoT Spec errors."""


class SpecNotFoundError(SpecError):
    pass


class SpecMismatchError(SpecError):
    """Raised when DB/file/spec_hash do not agree."""
