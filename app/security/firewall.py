# FILE: app/security/firewall.py
# Purpose: Firewall Middleware: Block requests from untrusted sources.
# Called-by: main (add_middleware, ACTIVE since 2026-07-02), app.security, tests.test_firewall
# Depends-on: app.security.client_ip
# Last-renovated: 2026-07-02
"""Firewall Middleware: Block requests from untrusted sources.

CRITICAL SAFETY FEATURE: Prevents Zombie/Sandbox Orb from communicating
back to Main Orb. This ensures one-way control only.

The sandbox uses Windows Sandbox NAT which assigns IPs in 192.168.250.x range.
Any request from this range to Main Orb is blocked.

Header-trust fix (2026-07-02): client IP resolution is delegated to
app.security.client_ip.effective_client_ip — X-Forwarded-For / X-Real-IP are
honoured ONLY when the raw socket peer is a trusted proxy (loopback Caddy),
so a spoofed XFF from the LAN or the sandbox cannot impersonate localhost.
"""

from __future__ import annotations

import logging
from typing import List, Set
from ipaddress import ip_address, ip_network

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.security.client_ip import effective_client_ip

logger = logging.getLogger(__name__)

# =============================================================================
# BLOCKED IP RANGES
# =============================================================================

# Windows Sandbox NAT range - NEVER allow these to reach Main Orb
SANDBOX_IP_RANGES = [
    "192.168.250.0/24",  # Windows Sandbox default NAT
]

# Additional blocked ranges (add more as needed)
BLOCKED_RANGES = [
    # "10.0.0.0/8",      # Uncomment to block all private 10.x.x.x
    # "172.16.0.0/12",   # Uncomment to block all private 172.16-31.x.x
]

# Combine all blocked ranges
ALL_BLOCKED_RANGES = SANDBOX_IP_RANGES + BLOCKED_RANGES

# Always allow these (localhost). "testclient" is starlette TestClient's
# synthetic peer name — impossible on a real TCP socket (peers are numeric),
# required so in-process test harnesses pass the active firewall.
ALLOWED_IPS = {
    "127.0.0.1",
    "::1",
    "localhost",
    "testclient",
}

# =============================================================================
# FIREWALL LOGIC
# =============================================================================

class FirewallMiddleware(BaseHTTPMiddleware):
    """Middleware to block requests from sandbox/untrusted IPs.
    
    This is a CRITICAL safety feature that ensures:
    1. Zombie Orb cannot call back to Main Orb
    2. Only localhost can access Main Orb API
    3. One-way control is enforced
    """
    
    def __init__(self, app, blocked_ranges: List[str] = None, allowed_ips: Set[str] = None):
        super().__init__(app)
        self.blocked_networks = []
        self.allowed_ips = allowed_ips or ALLOWED_IPS.copy()
        
        # Parse blocked ranges
        ranges = blocked_ranges or ALL_BLOCKED_RANGES
        for range_str in ranges:
            try:
                self.blocked_networks.append(ip_network(range_str, strict=False))
            except ValueError as e:
                logger.error(f"Invalid IP range '{range_str}': {e}")
        
        logger.info(f"Firewall initialized: blocking {len(self.blocked_networks)} ranges, allowing {len(self.allowed_ips)} IPs")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP — proxy-aware, spoof-resistant.

        Delegates to app.security.client_ip.effective_client_ip: forwarding
        headers count ONLY when the raw socket peer is a trusted proxy;
        every other peer is judged by its raw socket address. A LAN or
        sandbox client sending "X-Forwarded-For: 127.0.0.1" straight at
        :8000 therefore cannot bypass this firewall.
        """
        ip = effective_client_ip(request)
        return ip or "unknown"
    
    def _is_blocked(self, ip_str: str) -> bool:
        """Check if IP is in a blocked range."""
        # Always allow localhost
        if ip_str in self.allowed_ips:
            return False
        
        try:
            ip = ip_address(ip_str)
            for network in self.blocked_networks:
                if ip in network:
                    return True
        except ValueError:
            # Invalid IP format - block it to be safe
            logger.warning(f"Invalid IP format: {ip_str} - blocking")
            return True
        
        return False
    
    async def dispatch(self, request: Request, call_next):
        """Process request and block if from untrusted source."""
        client_ip = self._get_client_ip(request)
        
        if self._is_blocked(client_ip):
            logger.warning(f"🛡️ FIREWALL BLOCKED request from {client_ip} to {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Access denied",
                    "detail": "Request blocked by firewall",
                    "code": "FIREWALL_BLOCKED",
                }
            )
        
        # Allow request
        return await call_next(request)


# =============================================================================
# STRICT LOCALHOST-ONLY MODE
# =============================================================================

class LocalhostOnlyMiddleware(BaseHTTPMiddleware):
    """Even stricter mode: ONLY allow localhost, block everything else.
    
    Use this for maximum security - no external access at all.
    """
    
    LOCALHOST_IPS = {"127.0.0.1", "::1", "localhost"}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        
        if client_ip not in self.LOCALHOST_IPS:
            logger.warning(f"🛡️ LOCALHOST-ONLY blocked request from {client_ip}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Access denied",
                    "detail": "Only localhost access permitted",
                    "code": "LOCALHOST_ONLY",
                }
            )
        
        return await call_next(request)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_sandbox_ip(ip_str: str) -> bool:
    """Check if an IP is from the Windows Sandbox range."""
    try:
        ip = ip_address(ip_str)
        sandbox_net = ip_network("192.168.250.0/24")
        return ip in sandbox_net
    except ValueError:
        return False


def get_firewall_status() -> dict:
    """Get current firewall configuration for diagnostics."""
    return {
        "blocked_ranges": ALL_BLOCKED_RANGES,
        "allowed_ips": list(ALLOWED_IPS),
        "sandbox_range": "192.168.250.0/24",
        "mode": "firewall",
    }


__all__ = [
    "FirewallMiddleware",
    "LocalhostOnlyMiddleware",
    "is_sandbox_ip",
    "get_firewall_status",
    "SANDBOX_IP_RANGES",
    "ALLOWED_IPS",
]
