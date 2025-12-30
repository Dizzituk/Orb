# FILE: tests/test_firewall.py
"""Tests for firewall middleware - CRITICAL SAFETY TESTS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, AsyncMock
from app.security.firewall import (
    FirewallMiddleware,
    LocalhostOnlyMiddleware,
    is_sandbox_ip,
    get_firewall_status,
)


class TestSandboxIPDetection:
    """Test sandbox IP detection."""
    
    def test_sandbox_ip_detected(self):
        """Should detect sandbox IPs."""
        assert is_sandbox_ip("192.168.250.1") is True
        assert is_sandbox_ip("192.168.250.2") is True
        assert is_sandbox_ip("192.168.250.255") is True
    
    def test_non_sandbox_ip_not_detected(self):
        """Should not flag non-sandbox IPs."""
        assert is_sandbox_ip("127.0.0.1") is False
        assert is_sandbox_ip("192.168.1.1") is False
        assert is_sandbox_ip("10.0.0.1") is False
        assert is_sandbox_ip("8.8.8.8") is False
    
    def test_localhost_not_sandbox(self):
        """Localhost should not be detected as sandbox."""
        assert is_sandbox_ip("127.0.0.1") is False
        assert is_sandbox_ip("::1") is False


class TestFirewallMiddleware:
    """Test firewall middleware blocking."""
    
    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        app = Mock()
        return FirewallMiddleware(app)
    
    def test_sandbox_ip_blocked(self, middleware):
        """Sandbox IPs should be blocked."""
        assert middleware._is_blocked("192.168.250.1") is True
        assert middleware._is_blocked("192.168.250.2") is True
        assert middleware._is_blocked("192.168.250.100") is True
    
    def test_localhost_allowed(self, middleware):
        """Localhost should always be allowed."""
        assert middleware._is_blocked("127.0.0.1") is False
        assert middleware._is_blocked("::1") is False
    
    def test_external_ips_allowed(self, middleware):
        """External IPs should be allowed (not blocked by default)."""
        # These are allowed because they're not in blocked ranges
        assert middleware._is_blocked("192.168.1.100") is False
        assert middleware._is_blocked("10.0.0.50") is False
    
    def test_invalid_ip_blocked(self, middleware):
        """Invalid IPs should be blocked for safety."""
        assert middleware._is_blocked("not-an-ip") is True
        assert middleware._is_blocked("") is True


class TestLocalhostOnlyMiddleware:
    """Test strict localhost-only mode."""
    
    def test_localhost_ips(self):
        """Should recognize localhost IPs."""
        allowed = LocalhostOnlyMiddleware.LOCALHOST_IPS
        assert "127.0.0.1" in allowed
        assert "::1" in allowed
        assert "localhost" in allowed


class TestFirewallStatus:
    """Test firewall status reporting."""
    
    def test_status_contains_sandbox_range(self):
        """Status should show sandbox range."""
        status = get_firewall_status()
        assert "192.168.250.0/24" in status["blocked_ranges"]
        assert status["sandbox_range"] == "192.168.250.0/24"
    
    def test_status_contains_allowed_ips(self):
        """Status should show allowed IPs."""
        status = get_firewall_status()
        assert "127.0.0.1" in status["allowed_ips"]


class TestCriticalSafety:
    """CRITICAL: These tests verify sandbox isolation."""
    
    @pytest.fixture
    def middleware(self):
        app = Mock()
        return FirewallMiddleware(app)
    
    def test_all_sandbox_subnet_blocked(self, middleware):
        """CRITICAL: Every IP in sandbox subnet must be blocked."""
        for i in range(1, 255):
            ip = f"192.168.250.{i}"
            assert middleware._is_blocked(ip) is True, f"SECURITY FAILURE: {ip} not blocked!"
    
    def test_zombie_cannot_reach_main(self, middleware):
        """CRITICAL: Zombie Orb IP must be blocked."""
        # Standard sandbox controller IP
        assert middleware._is_blocked("192.168.250.2") is True
        # Any other sandbox IP
        assert middleware._is_blocked("192.168.250.50") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
