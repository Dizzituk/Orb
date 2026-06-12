# FILE: app/grounding/freshness_policy.py
# Purpose: Freshness Policy for the Grounded Intelligence System.
# Called-by: app.grounding.grounding_gate
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Freshness Policy for the Grounded Intelligence System.

Defines maximum acceptable data age per domain. When ASTRA is about
to answer a claims-dependent query, the freshness policy determines
whether a web search is mandatory based on how quickly the domain's
information changes.

Configurable via environment variables or the defaults below.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Freshness rules per domain
# =========================================================================

@dataclass(frozen=True)
class FreshnessRule:
    """Freshness constraint for a topic domain."""
    domain: str
    max_age_hours: int          # Max acceptable age of data in hours
    search_required: bool       # Whether search is always required
    description: str = ""


# Defaults — overridable via env vars (GROUNDING_FRESHNESS_{DOMAIN}_HOURS)
_DEFAULT_RULES: dict[str, FreshnessRule] = {
    "finance": FreshnessRule(
        domain="finance",
        max_age_hours=4,
        search_required=True,
        description="Financial markets, crypto, exchange rates, economic indicators",
    ),
    "ai_tech": FreshnessRule(
        domain="ai_tech",
        max_age_hours=168,   # 7 days
        search_required=True,
        description="AI models, tech releases, benchmarks, industry moves",
    ),
    "geopolitics": FreshnessRule(
        domain="geopolitics",
        max_age_hours=72,    # 3 days
        search_required=True,
        description="Wars, elections, treaties, international relations",
    ),
    "uk_affairs": FreshnessRule(
        domain="uk_affairs",
        max_age_hours=72,    # 3 days
        search_required=True,
        description="UK politics, NHS, economy, legislation",
    ),
    "health": FreshnessRule(
        domain="health",
        max_age_hours=336,   # 14 days
        search_required=True,
        description="Medical studies, treatments, nutrition research",
    ),
    "surf_weather": FreshnessRule(
        domain="surf_weather",
        max_age_hours=6,
        search_required=True,
        description="Surf conditions, swell forecasts, weather",
    ),
    "general": FreshnessRule(
        domain="general",
        max_age_hours=720,   # 30 days
        search_required=False,
        description="General knowledge — training data acceptable for stable facts",
    ),
}


def _load_env_override(domain: str, default_hours: int) -> int:
    """Check for env var override: GROUNDING_FRESHNESS_{DOMAIN}_HOURS."""
    env_key = f"GROUNDING_FRESHNESS_{domain.upper()}_HOURS"
    val = os.getenv(env_key, "").strip()
    if val:
        try:
            hours = int(val)
            if hours > 0:
                logger.info(
                    "[freshness_policy] Override: %s = %d hours (was %d)",
                    env_key, hours, default_hours,
                )
                return hours
        except ValueError:
            logger.warning("[freshness_policy] Invalid env var %s=%s", env_key, val)
    return default_hours


def get_freshness_rule(domain: str) -> FreshnessRule:
    """
    Get the freshness rule for a domain.
    
    Falls back to 'general' if domain not recognised.
    Checks environment variable overrides.
    """
    rule = _DEFAULT_RULES.get(domain, _DEFAULT_RULES["general"])
    # Apply env override if present
    actual_hours = _load_env_override(rule.domain, rule.max_age_hours)
    if actual_hours != rule.max_age_hours:
        return FreshnessRule(
            domain=rule.domain,
            max_age_hours=actual_hours,
            search_required=rule.search_required,
            description=rule.description,
        )
    return rule


def is_search_required(domain: str) -> bool:
    """Quick check: does this domain require a web search?"""
    rule = get_freshness_rule(domain)
    return rule.search_required


def get_all_rules() -> dict[str, FreshnessRule]:
    """Return all freshness rules (with env overrides applied)."""
    return {
        domain: get_freshness_rule(domain)
        for domain in _DEFAULT_RULES
    }


__all__ = [
    "FreshnessRule",
    "get_freshness_rule",
    "is_search_required",
    "get_all_rules",
]
