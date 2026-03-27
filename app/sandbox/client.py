# FILE: app/overwatcher/sandbox_client.py
"""Sandbox Client: HTTP interface to sandbox_controller.

Provides safe, isolated execution environment for:
- File operations (read/write)
- Shell commands (pytest, ruff, mypy)
- Repo tree scanning

The sandbox_controller runs in Windows Sandbox at http://<ip>:8765
with strict security boundaries (no secrets, no system commands).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SANDBOX_URL = "http://192.168.250.2:8765"
CONTROLLER_ADDR_FILE = r"C:\Orb\_sandbox_cache\controller_addr.txt"
DEFAULT_TIMEOUT = 30
MAX_FILE_BYTES = 512_000  # 500KB - matches sandbox_controller limit


# =============================================================================
# Response Models
# =============================================================================

@dataclass
class HealthResponse:
    """Response from /health endpoint."""
    status: str
    repo_root: str
    cache_root: str
    artifact_root: str
    scratch_root: str
    version: str


@dataclass
class FileEntry:
    """Single file in repo tree."""
    path: str
    size_bytes: int
    sha256: Optional[str] = None


@dataclass
class FileContent:
    """Response from /repo/file endpoint."""
    path: str
    sha256: str
    bytes: int
    content: str


@dataclass
class WriteResult:
    ok: bool
    path: str
    bytes: int
    sha256: str = ""


@dataclass
class ShellResult:
    ok: bool
    exit_code: int
    duration_ms: int
    stdout: str
    stderr: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


# =============================================================================
# Client
# =============================================================================

class SandboxClient:
    """HTTP client for sandbox_controller API.
    
    Usage:
        client = SandboxClient()
        
        # Check health
        health = client.health()
        
        # Get repo tree
        tree = client.repo_tree(include_hashes=True)
        
        # Read file
        content = client.repo_file("app/main.py")
        
        # Write file
        result = client.write_file("SCRATCH", "test.py", "print('hello')")
        
        # Run shell command
        result = client.shell_run("pytest tests/ -v", timeout_seconds=120)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize sandbox client.
        
        Args:
            base_url: Override sandbox URL (auto-discovers if None)
            timeout: Default request timeout in seconds
        """
        self.base_url = base_url or self._discover_base_url()
        self.timeout = timeout
        self._connected = False
        
        logger.info(f"[sandbox] Client initialized: {self.base_url}")
    
    def _discover_base_url(self) -> str:
        """Discover sandbox controller URL from cache file or default."""
        # Try reading from controller_addr.txt (written by start_controller.ps1)
        try:
            addr_path = Path(CONTROLLER_ADDR_FILE)
            if addr_path.exists():
                url = addr_path.read_text(encoding="utf-8").strip()
                if url.startswith("http://") or url.startswith("https://"):
                    logger.info(f"[sandbox] Discovered URL from cache: {url}")
                    return url
        except Exception as e:
            logger.debug(f"[sandbox] Could not read addr file: {e}")
        
        # Fall back to default
        return DEFAULT_SANDBOX_URL
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to sandbox controller.
        
        Args:
            method: HTTP method (GET, POST)
            path: API path (e.g., "/health")
            params: Query parameters
            json_body: JSON request body
            timeout: Request timeout override
        
        Returns:
            Parsed JSON response
        
        Raises:
            SandboxError: On connection or HTTP errors
        """
        url = f"{self.base_url.rstrip('/')}{path}"
        
        if params:
            url += "?" + urlencode(params)
        
        headers = {"Accept": "application/json"}
        data = None
        
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(json_body).encode("utf-8")
        
        req = Request(url, data=data, headers=headers, method=method)
        
        print(f"[SANDBOX_DEBUG] HTTP {method} {url}")  # Force print
        
        try:
            with urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise SandboxError(
                f"HTTP {e.code} from {path}: {body[:500]}",
                status_code=e.code,
                response_body=body,
            )
        except URLError as e:
            raise SandboxError(f"Connection failed to {url}: {e.reason}")
        except Exception as e:
            raise SandboxError(f"Request failed: {e}")
    
    # -------------------------------------------------------------------------
    # API Methods
    # -------------------------------------------------------------------------
    
    def health(self) -> HealthResponse:
        """Check sandbox controller health.
        
        Returns:
            HealthResponse with status and paths
        """
        data = self._request("GET", "/health")
        self._connected = data.get("status") == "ok"
        
        return HealthResponse(
            status=data.get("status", ""),
            repo_root=data.get("repo_root", ""),
            cache_root=data.get("cache_root", ""),
            artifact_root=data.get("artifact_root", ""),
            scratch_root=data.get("scratch_root", ""),
            version=data.get("version", ""),
        )
    
    def is_connected(self) -> bool:
        """Check if client can reach sandbox controller."""
        try:
            health = self.health()
            return health.status == "ok"
        except Exception:
            return False
    
    def repo_tree(
        self,
        include_hashes: bool = False,
        max_files: int = 80000,
    ) -> List[FileEntry]:
        """Get repository file tree.
        
        Args:
            include_hashes: Include SHA256 hashes for files
            max_files: Maximum files to return
        
        Returns:
            List of FileEntry objects
        """
        params = {
            "include_hashes": str(include_hashes).lower(),
            "max_files": max_files,
        }
        
        data = self._request("GET", "/repo/tree", params=params)
        
        return [
            FileEntry(
                path=entry.get("path", ""),
                size_bytes=entry.get("size_bytes", 0),
                sha256=entry.get("sha256"),
            )
            for entry in data
        ]
    
    def repo_file(self, path: str) -> FileContent:
        """Read file from repository.
        
        Args:
            path: Relative POSIX path (e.g., "app/main.py")
        
        Returns:
            FileContent with content and metadata
        
        Raises:
            SandboxError: If file not found or access denied
        """
        params = {"path": path}
        data = self._request("GET", "/repo/file", params=params)
        
        result = FileContent(
            path=data.get("path", ""),
            sha256=data.get("sha256", ""),
            bytes=data.get("bytes", 0),
            content=data.get("content", ""),
        )

        # IO Tracking: record sandbox reads when IOTracker is active
        try:
            from app.transparency.io_tracker import get_active_tracker
            tracker = get_active_tracker()
            if tracker:
                tracker.record_read(
                    path=path,
                    source="sandbox",
                    bytes_count=result.bytes,
                )
        except ImportError:
            pass

        return result
    
    def write_file(
        self,
        target: str,
        filename: str,
        content: str,
        subdir: Optional[str] = None,
        overwrite: bool = False,
    ) -> WriteResult:
        target_map = {
            "REPO": r"D:\Orb",
            "SCRATCH": None,
            "ARTIFACT": None,
            "DESKTOP": r"C:\Users\dizzi\Desktop",
            "DOCUMENTS": r"C:\Users\dizzi\Documents",
        }
        base = target_map.get(target.upper())
        if target.upper() == "SCRATCH":
            base = self.health().scratch_root
        elif target.upper() == "ARTIFACT":
            base = self.health().artifact_root
        if not base:
            raise SandboxError(f"Unsupported write target: {target}")

        parts = [base]
        if subdir:
            parts.append(subdir)
        parts.append(filename)
        abs_path = str(Path(*parts))

        data = self._request(
            "POST",
            "/fs/write",
            json_body={
                "path": abs_path,
                "content": content,
                "overwrite": overwrite,
            },
        )

        return WriteResult(
            ok=data.get("status") == "ok" or data.get("ok", False),
            path=data.get("path", ""),
            bytes=data.get("bytes", 0),
            sha256=data.get("sha256", ""),
        )

    def shell_run(
        self,
        command: str,
        cwd_target: str = "REPO",
        timeout_seconds: int = 60,
    ) -> ShellResult:
        cwd_map = {
            "REPO": r"D:\Orb",
            "SCRATCH": self.health().scratch_root,
            "ARTIFACT": self.health().artifact_root,
        }
        cwd = cwd_map.get(cwd_target.upper())
        body = {
            "cmd": ["powershell", "-NoProfile", "-Command", command],
            "cwd": cwd,
            "timeout_sec": min(timeout_seconds, 300),
        }

        logger.info(f"[sandbox] shell_run request: cmd={body['cmd']}, cwd={cwd}")
        print(f"[SANDBOX_DEBUG] shell_run request: cmd={body['cmd']}, cwd={cwd}")

        try:
            data = self._request(
                "POST",
                "/shell/run",
                json_body=body,
                timeout=timeout_seconds + 10,
            )
            exit_code = data.get("exit_code")
            if exit_code is None:
                exit_code = data.get("returncode", -1)
            return ShellResult(
                ok=exit_code == 0,
                exit_code=exit_code,
                duration_ms=data.get("duration_ms", 0),
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
            )
        except SandboxError as e:
            logger.error(
                f"[sandbox] shell_run FAILED: {e}, "
                f"status={e.status_code}, "
                f"body={e.response_body[:200] if e.response_body else None}"
            )
            print(
                f"[SANDBOX_DEBUG] shell_run FAILED: {e}, "
                f"status={e.status_code}, "
                f"body={e.response_body[:200] if e.response_body else None}"
            )
            raise

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    def run_pytest(
        self,
        test_path: str = "tests/",
        extra_args: str = "-v",
        timeout_seconds: int = 120,
    ) -> ShellResult:
        """Run pytest in sandbox.
        
        Args:
            test_path: Path to tests (relative to repo root)
            extra_args: Additional pytest arguments
            timeout_seconds: Timeout for test run
        
        Returns:
            ShellResult with test output
        """
        command = f"python -m pytest {test_path} {extra_args}"
        return self.shell_run(command, timeout_seconds=timeout_seconds)
    
    def run_ruff(
        self,
        paths: Optional[List[str]] = None,
        timeout_seconds: int = 60,
    ) -> ShellResult:
        """Run ruff linter in sandbox.
        
        Args:
            paths: Specific paths to lint (defaults to ".")
            timeout_seconds: Timeout
        
        Returns:
            ShellResult with lint output
        """
        target = " ".join(paths) if paths else "."
        command = f"ruff check {target}"
        return self.shell_run(command, timeout_seconds=timeout_seconds)
    
    def run_mypy(
        self,
        paths: Optional[List[str]] = None,
        timeout_seconds: int = 120,
    ) -> ShellResult:
        """Run mypy type checker in sandbox.
        
        Args:
            paths: Specific paths to check (defaults to ".")
            timeout_seconds: Timeout
        
        Returns:
            ShellResult with type check output
        """
        target = " ".join(paths) if paths else "."
        command = f"python -m mypy {target} --ignore-missing-imports"
        return self.shell_run(command, timeout_seconds=timeout_seconds)


# =============================================================================
# Exceptions
# =============================================================================

class SandboxError(Exception):
    """Error communicating with sandbox controller."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


# =============================================================================
# Singleton / Factory
# =============================================================================

_default_client: Optional[SandboxClient] = None


def get_sandbox_client(
    base_url: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> SandboxClient:
    """Get or create sandbox client singleton.
    
    Args:
        base_url: Override sandbox URL
        timeout: Request timeout
    
    Returns:
        SandboxClient instance
    """
    global _default_client
    
    if _default_client is None or base_url:
        _default_client = SandboxClient(base_url=base_url, timeout=timeout)
    
    return _default_client


__all__ = [
    # Models
    "HealthResponse",
    "FileEntry",
    "FileContent",
    "WriteResult",
    "ShellResult",
    # Client
    "SandboxClient",
    "SandboxError",
    # Factory
    "get_sandbox_client",
]
