# FILE: app/memory/domains/preference_registry.py
"""
Preference domain registry.

Maps preference namespace names to descriptions and validation rules.
The astra_preferences.namespace column accepts arbitrary strings —
this registry validates and documents the known namespaces.

Usage:
    from app.memory.domains.preference_registry import (
        is_valid_domain, get_domain_info, list_domains,
    )

    if not is_valid_domain("foobar"):
        raise ValueError("Unknown preference domain")
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DomainInfo:
    """Metadata for a preference domain/namespace."""
    name: str
    description: str
    example_keys: tuple[str, ...]


# =========================================================================
# Registry
# =========================================================================

_DOMAINS: dict[str, DomainInfo] = {
    "development": DomainInfo(
        name="development",
        description=(
            "Code style, file size limits, tooling preferences, "
            "build/test conventions, and engineering workflow rules."
        ),
        example_keys=(
            "file_size_target_kb",
            "modularity_first",
            "no_git_commands",
            "evidence_first",
            "indent_style",
        ),
    ),
    "content": DomainInfo(
        name="content",
        description=(
            "Writing style, tone, formatting preferences for "
            "documents, blog posts, and generated text."
        ),
        example_keys=(
            "tone_formal",
            "prefer_british_english",
            "markdown_style",
        ),
    ),
    "video": DomainInfo(
        name="video",
        description=(
            "Video editing, production, and publishing preferences."
        ),
        example_keys=(
            "resolution_default",
            "export_format",
            "thumbnail_style",
        ),
    ),
    "fitness": DomainInfo(
        name="fitness",
        description=(
            "Personal training, health monitoring, and exercise "
            "programming preferences."
        ),
        example_keys=(
            "measurement_unit",
            "training_split",
            "rest_period_default",
        ),
    ),
    "catering": DomainInfo(
        name="catering",
        description=(
            "Food preparation, hospitality, menu planning, and "
            "catering management preferences."
        ),
        example_keys=(
            "portion_size_default",
            "allergen_alert_level",
            "costing_method",
        ),
    ),
    "finance": DomainInfo(
        name="finance",
        description=(
            "Accounting, invoicing, business financial management, "
            "and reporting preferences."
        ),
        example_keys=(
            "currency_default",
            "vat_rate",
            "invoice_format",
        ),
    ),
    "general": DomainInfo(
        name="general",
        description=(
            "Catch-all namespace for preferences that don't fit "
            "a specific domain. Also used for cross-domain defaults."
        ),
        example_keys=(
            "timezone",
            "date_format",
            "language",
        ),
    ),
}


# =========================================================================
# Public API
# =========================================================================

def is_valid_domain(domain: str) -> bool:
    """Check if a domain name is registered."""
    return domain in _DOMAINS


def get_domain_info(domain: str) -> Optional[DomainInfo]:
    """Get metadata for a domain. Returns None if unknown."""
    return _DOMAINS.get(domain)


def list_domains() -> list[DomainInfo]:
    """List all registered domains."""
    return list(_DOMAINS.values())


def list_domain_names() -> list[str]:
    """List just the domain name strings."""
    return list(_DOMAINS.keys())


def validate_domain(domain: str) -> str:
    """
    Validate and return domain name.

    Raises ValueError if domain is not registered.
    """
    if not is_valid_domain(domain):
        known = ", ".join(_DOMAINS.keys())
        raise ValueError(
            f"Unknown preference domain: '{domain}'. "
            f"Known domains: {known}"
        )
    return domain
