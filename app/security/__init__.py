# FILE: app/security/__init__.py
"""Security Module: Firewall and access control for Orb.

CRITICAL SAFETY: This module enforces one-way control between Main Orb
and Zombie Orb. The sandbox can NEVER communicate back to Main Orb.

Usage in main.py:
    from app.security.firewall import FirewallMiddleware
    app.add_middleware(FirewallMiddleware)
"""

from app.security.firewall import (
    FirewallMiddleware,
    LocalhostOnlyMiddleware,
    is_sandbox_ip,
    get_firewall_status,
    SANDBOX_IP_RANGES,
    ALLOWED_IPS,
)

__all__ = [
    "FirewallMiddleware",
    "LocalhostOnlyMiddleware",
    "is_sandbox_ip",
    "get_firewall_status",
    "SANDBOX_IP_RANGES",
    "ALLOWED_IPS",
]
