# FILE: app/pot_spec/errors.py
class SpecError(Exception):
    """Base class for PoT Spec errors."""


class SpecNotFoundError(SpecError):
    pass


class SpecMismatchError(SpecError):
    """Raised when DB/file/spec_hash do not agree."""
