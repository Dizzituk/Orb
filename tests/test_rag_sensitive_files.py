# tests/test_rag_sensitive_files.py
"""Tests for sensitive file detection."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from app.rag.security.sensitive_files import (
    is_sensitive_file,
    should_skip_directory,
)


class TestSensitiveFile:
    def test_env_blocked(self):
        is_sens, reason = is_sensitive_file(".env", "/project/.env", "repo")
        assert is_sens is True
    
    def test_env_local_blocked(self):
        is_sens, reason = is_sensitive_file(".env.local", "/x/.env.local", "repo")
        assert is_sens is True
    
    def test_pem_blocked(self):
        is_sens, reason = is_sensitive_file("server.pem", "/certs/server.pem", "repo")
        assert is_sens is True
    
    def test_id_rsa_blocked(self):
        is_sens, reason = is_sensitive_file("id_rsa", "/home/.ssh/id_rsa", "user")
        assert is_sens is True
    
    def test_ssh_path_blocked(self):
        is_sens, reason = is_sensitive_file("config", "/home/user/.ssh/config", "user")
        assert is_sens is True
    
    def test_normal_py_allowed(self):
        is_sens, reason = is_sensitive_file("router.py", "/app/router.py", "repo")
        assert is_sens is False
    
    def test_tokenizer_allowed_repo(self):
        """tokenizer.py should NOT be blocked in repo (false positive risk)."""
        is_sens, reason = is_sensitive_file("tokenizer.py", "/app/tokenizer.py", "repo")
        assert is_sens is False
    
    def test_tokenizer_blocked_user(self):
        """But 'token' substring blocked in user zone."""
        is_sens, reason = is_sensitive_file("my_token.txt", "/user/token.txt", "user")
        assert is_sens is True


class TestSkipDirectory:
    def test_ssh_skipped(self):
        assert should_skip_directory(".ssh") is True
    
    def test_gnupg_skipped(self):
        assert should_skip_directory(".gnupg") is True
    
    def test_app_allowed(self):
        assert should_skip_directory("app") is False
