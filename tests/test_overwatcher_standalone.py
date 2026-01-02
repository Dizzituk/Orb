# FILE: test_overwatcher_standalone.py
"""Standalone Overwatcher Smoke Test - works without full app structure.

This test runs the core Overwatcher logic directly without requiring
the full app.* import hierarchy. Use this to verify the logic before
integrating into the main codebase.

Run with:
    python test_overwatcher_standalone.py
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Inline implementations (copied from overwatcher_command.py for standalone test)
# =============================================================================

SMOKE_TEST_FILENAME = "hello.txt"
SMOKE_TEST_CONTENT = "ASTRA OK"
SMOKE_TEST_TARGET = "DESKTOP"


@dataclass
class ResolvedSpec:
    """Resolved spec context."""
    spec_id: str
    spec_hash: str
    project_id: int
    title: Optional[str] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at,
        }


@dataclass
class ImplementerResult:
    """Result from Implementer execution."""
    success: bool
    output_path: Optional[str] = None
    sha256: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0
    sandbox_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "sha256": self.sha256,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "sandbox_used": self.sandbox_used,
        }


@dataclass
class VerificationResult:
    """Result from verification step."""
    passed: bool
    file_exists: bool = False
    content_matches: bool = False
    actual_content: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "file_exists": self.file_exists,
            "content_matches": self.content_matches,
            "actual_content": self.actual_content,
            "error": self.error,
        }


@dataclass
class OverwatcherCommandResult:
    """Complete result from 'run overwatcher' command."""
    success: bool
    job_id: str
    spec: Optional[ResolvedSpec] = None
    overwatcher_decision: Optional[str] = None
    overwatcher_diagnosis: Optional[str] = None
    implementer_result: Optional[ImplementerResult] = None
    verification_result: Optional[VerificationResult] = None
    error: Optional[str] = None
    stage_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "job_id": self.job_id,
            "spec": self.spec.to_dict() if self.spec else None,
            "overwatcher_decision": self.overwatcher_decision,
            "overwatcher_diagnosis": self.overwatcher_diagnosis,
            "implementer_result": self.implementer_result.to_dict() if self.implementer_result else None,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None,
            "error": self.error,
            "stage_trace": self.stage_trace,
        }
    
    def add_trace(self, stage: str, status: str, details: Optional[Dict] = None):
        """Add entry to stage trace."""
        self.stage_trace.append({
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })


def create_smoke_test_spec() -> ResolvedSpec:
    """Create a minimal spec for smoke testing."""
    spec_content = json.dumps({
        "title": "Sandbox Hello Test",
        "goal": "Create hello.txt on desktop with content 'ASTRA OK'",
        "deliverables": [
            {
                "type": "file",
                "target": "DESKTOP",
                "filename": "hello.txt",
                "content": "ASTRA OK"
            }
        ],
        "verification": {
            "file_exists": True,
            "content_match": "ASTRA OK"
        }
    }, sort_keys=True)
    
    spec_hash = hashlib.sha256(spec_content.encode()).hexdigest()
    spec_id = f"smoke-test-{uuid4().hex[:8]}"
    
    return ResolvedSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        project_id=0,
        title="Sandbox Hello Test",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# Command Detection (from overwatcher_routing.py)
# =============================================================================

class OverwatcherCommandType:
    RUN = "run"
    STATUS = "status"
    RETRY = "retry"
    CANCEL = "cancel"


OVERWATCHER_COMMAND_PATTERNS = [
    r"(?:astra[,:]?\s+)?(?:command[:\s]+)?run\s+overwatcher",
    r"(?:astra[,:]?\s+)?execute\s+overwatcher",
    r"(?:astra[,:]?\s+)?start\s+overwatcher",
    r"overwatcher\s+run",
    r"run\s+the\s+overwatcher",
    r"invoke\s+overwatcher",
    r"trigger\s+overwatcher",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in OVERWATCHER_COMMAND_PATTERNS]


def detect_overwatcher_command(message: str) -> Optional[str]:
    """Detect if message is an Overwatcher command."""
    message = message.strip()
    
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(message):
            return OverwatcherCommandType.RUN
    
    if re.search(r"overwatcher\s+status", message, re.IGNORECASE):
        return OverwatcherCommandType.STATUS
    
    if re.search(r"(?:retry|rerun)\s+overwatcher", message, re.IGNORECASE):
        return OverwatcherCommandType.RETRY
    
    if re.search(r"(?:cancel|stop)\s+overwatcher", message, re.IGNORECASE):
        return OverwatcherCommandType.CANCEL
    
    return None


# =============================================================================
# Sandbox Client (simplified for standalone test)
# =============================================================================

@dataclass
class ShellResult:
    """Result from shell command."""
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int = 0


class SimpleSandboxClient:
    """Simplified sandbox client for testing."""
    
    def __init__(self, base_url: str = "http://192.168.250.2:8765"):
        self.base_url = base_url
        self._connected = None
    
    def is_connected(self) -> bool:
        """Check if sandbox is reachable."""
        if self._connected is not None:
            return self._connected
        
        try:
            import urllib.request
            import json
            
            req = urllib.request.Request(
                f"{self.base_url}/health",
                headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                self._connected = data.get("status") == "ok"
                return self._connected
        except Exception as e:
            logger.debug(f"Sandbox not reachable: {e}")
            self._connected = False
            return False
    
    def write_file(self, target: str, filename: str, content: str, overwrite: bool = True) -> Dict:
        """Write file via sandbox."""
        import urllib.request
        import json
        
        body = json.dumps({
            "target": target,
            "filename": filename,
            "content": content,
            "overwrite": overwrite,
        }).encode()
        
        req = urllib.request.Request(
            f"{self.base_url}/fs/write",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    
    def shell_run(self, command: str, timeout_seconds: int = 60) -> ShellResult:
        """Run shell command via sandbox."""
        import urllib.request
        import json
        
        body = json.dumps({
            "command": command,
            "cwd_target": "REPO",
            "timeout_seconds": timeout_seconds,
        }).encode()
        
        req = urllib.request.Request(
            f"{self.base_url}/shell/run",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=timeout_seconds + 10) as resp:
            data = json.loads(resp.read().decode())
            return ShellResult(
                ok=data.get("ok", False),
                exit_code=data.get("exit_code", -1),
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
                duration_ms=data.get("duration_ms", 0),
            )


# =============================================================================
# Implementer (simplified)
# =============================================================================

async def run_implementer(
    spec: ResolvedSpec,
    client: Optional[SimpleSandboxClient] = None,
) -> ImplementerResult:
    """Execute the smoke test implementation."""
    import time
    start_time = time.time()
    
    if client is None:
        client = SimpleSandboxClient()
    
    try:
        if client.is_connected():
            logger.info("[implementer] Writing file via sandbox")
            
            result = client.write_file(
                target=SMOKE_TEST_TARGET,
                filename=SMOKE_TEST_FILENAME,
                content=SMOKE_TEST_CONTENT,
                overwrite=True,
            )
            
            if result.get("ok"):
                return ImplementerResult(
                    success=True,
                    output_path=result.get("path"),
                    sha256=result.get("sha256"),
                    duration_ms=int((time.time() - start_time) * 1000),
                    sandbox_used=True,
                )
            else:
                return ImplementerResult(
                    success=False,
                    error="Sandbox write returned not ok",
                    duration_ms=int((time.time() - start_time) * 1000),
                    sandbox_used=True,
                )
        else:
            # Local fallback
            logger.warning("[implementer] Sandbox not connected, using local fallback")
            
            # Try common desktop paths
            desktop_paths = [
                Path.home() / "OneDrive" / "Desktop",
                Path.home() / "Desktop",
                Path("/tmp"),  # Linux fallback
            ]
            
            desktop = None
            for dp in desktop_paths:
                if dp.exists():
                    desktop = dp
                    break
            
            if desktop is None:
                desktop = Path("/tmp")
                desktop.mkdir(exist_ok=True)
            
            out_path = desktop / SMOKE_TEST_FILENAME
            content_bytes = SMOKE_TEST_CONTENT.encode("utf-8")
            out_path.write_bytes(content_bytes)
            sha256 = hashlib.sha256(content_bytes).hexdigest()
            
            return ImplementerResult(
                success=True,
                output_path=str(out_path),
                sha256=sha256,
                duration_ms=int((time.time() - start_time) * 1000),
                sandbox_used=False,
            )
            
    except Exception as e:
        logger.error(f"[implementer] Execution failed: {e}")
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


# =============================================================================
# Verification (simplified)
# =============================================================================

async def run_verification(
    impl_result: ImplementerResult,
    expected_content: str = SMOKE_TEST_CONTENT,
    client: Optional[SimpleSandboxClient] = None,
) -> VerificationResult:
    """Verify implementation result."""
    
    if not impl_result.success:
        return VerificationResult(
            passed=False,
            error=f"Implementation failed: {impl_result.error}",
        )
    
    if not impl_result.output_path:
        return VerificationResult(
            passed=False,
            error="No output path from Implementer",
        )
    
    if client is None:
        client = SimpleSandboxClient()
    
    try:
        if client.is_connected() and impl_result.sandbox_used:
            # PowerShell verification
            ps_path = impl_result.output_path.replace("/", "\\")
            
            # Check exists
            exists_result = client.shell_run(f'Test-Path -Path "{ps_path}"', timeout_seconds=10)
            file_exists = exists_result.ok and "True" in exists_result.stdout
            
            if not file_exists:
                return VerificationResult(
                    passed=False,
                    file_exists=False,
                    error=f"File not found at {impl_result.output_path}",
                )
            
            # Read content
            read_result = client.shell_run(f'Get-Content -Path "{ps_path}" -Raw', timeout_seconds=10)
            
            if not read_result.ok:
                return VerificationResult(
                    passed=False,
                    file_exists=True,
                    error=f"Failed to read file: {read_result.stderr}",
                )
            
            actual_content = read_result.stdout.strip()
            content_matches = actual_content == expected_content
            
            return VerificationResult(
                passed=content_matches,
                file_exists=True,
                content_matches=content_matches,
                actual_content=actual_content,
                error=None if content_matches else f"Content mismatch: expected '{expected_content}', got '{actual_content}'",
            )
        else:
            # Local verification
            file_path = Path(impl_result.output_path)
            file_exists = file_path.exists()
            
            if not file_exists:
                return VerificationResult(
                    passed=False,
                    file_exists=False,
                    error=f"File not found at {impl_result.output_path}",
                )
            
            actual_content = file_path.read_text(encoding="utf-8").strip()
            content_matches = actual_content == expected_content
            
            return VerificationResult(
                passed=content_matches,
                file_exists=True,
                content_matches=content_matches,
                actual_content=actual_content,
                error=None if content_matches else f"Content mismatch: expected '{expected_content}', got '{actual_content}'",
            )
            
    except Exception as e:
        logger.error(f"[verification] Failed: {e}")
        return VerificationResult(
            passed=False,
            error=str(e),
        )


# =============================================================================
# Main Command Handler (simplified)
# =============================================================================

async def run_overwatcher_command(
    project_id: int = 0,
    job_id: Optional[str] = None,
    message: str = "",
) -> OverwatcherCommandResult:
    """Execute the 'run overwatcher' command (simplified for standalone test)."""
    
    job_id = job_id or str(uuid4())
    result = OverwatcherCommandResult(success=False, job_id=job_id)
    
    logger.info(f"[overwatcher_command] Starting: job_id={job_id}")
    result.add_trace("OVERWATCHER_COMMAND_START", "started", {"project_id": project_id})
    
    # Step 1: Create smoke test spec
    spec = create_smoke_test_spec()
    result.spec = spec
    result.add_trace("SPEC_RESOLVE", "smoke_test", {"spec_id": spec.spec_id})
    logger.info(f"[overwatcher_command] Spec created: {spec.spec_id}")
    
    # Step 2: Overwatcher decision (auto-approve for standalone test)
    result.overwatcher_decision = "PASS"
    result.overwatcher_diagnosis = "Auto-approved (standalone smoke test)"
    result.add_trace("OVERWATCHER_SKIP", "auto_approved", {"reason": "standalone_test"})
    
    # Step 3: Run Implementer
    logger.info("[overwatcher_command] Running Implementer")
    result.add_trace("IMPLEMENTER_ENTER", "running")
    
    client = SimpleSandboxClient()
    impl_result = await run_implementer(spec=spec, client=client)
    result.implementer_result = impl_result
    
    if impl_result.success:
        result.add_trace("IMPLEMENTER_EXIT", "success", {
            "output_path": impl_result.output_path,
            "sandbox_used": impl_result.sandbox_used,
        })
        logger.info(f"[overwatcher_command] Implementer success: {impl_result.output_path}")
    else:
        result.error = f"Implementer failed: {impl_result.error}"
        result.add_trace("IMPLEMENTER_EXIT", "failed", {"error": impl_result.error})
        logger.error(f"[overwatcher_command] {result.error}")
        return result
    
    # Step 4: Run Verification
    logger.info("[overwatcher_command] Running verification")
    result.add_trace("VERIFICATION_ENTER", "running")
    
    verify_result = await run_verification(impl_result=impl_result, client=client)
    result.verification_result = verify_result
    
    if verify_result.passed:
        result.success = True
        result.add_trace("VERIFICATION_EXIT", "passed", verify_result.to_dict())
        result.add_trace("JOB_COMPLETE", "success", {"job_id": job_id})
        logger.info(f"[overwatcher_command] ✓ Job {job_id} COMPLETE - verification passed")
    else:
        result.error = f"Verification failed: {verify_result.error}"
        result.add_trace("VERIFICATION_EXIT", "failed", verify_result.to_dict())
        logger.error(f"[overwatcher_command] {result.error}")
    
    return result


# =============================================================================
# Tests
# =============================================================================

async def test_smoke_test_spec_creation():
    """Test smoke test spec creation."""
    spec = create_smoke_test_spec()
    
    assert spec.spec_id.startswith("smoke-test-"), f"Unexpected spec_id: {spec.spec_id}"
    assert len(spec.spec_hash) == 64, f"Invalid hash length: {len(spec.spec_hash)}"
    assert spec.project_id == 0
    assert spec.title == "Sandbox Hello Test"
    
    logger.info(f"✓ Smoke test spec created: {spec.spec_id}")
    return True


async def test_command_detection():
    """Test command pattern detection."""
    test_cases = [
        ("run overwatcher", OverwatcherCommandType.RUN),
        ("Astra, command: run overwatcher", OverwatcherCommandType.RUN),
        ("execute overwatcher", OverwatcherCommandType.RUN),
        ("start overwatcher", OverwatcherCommandType.RUN),
        ("overwatcher run", OverwatcherCommandType.RUN),
    ]
    
    for message, expected in test_cases:
        result = detect_overwatcher_command(message)
        assert result == expected, f"Failed for '{message}': expected {expected}, got {result}"
        logger.info(f"  ✓ '{message}' → {result}")
    
    # Should NOT detect
    non_commands = ["what is overwatcher", "hello world"]
    for message in non_commands:
        result = detect_overwatcher_command(message)
        assert result is None, f"False positive for '{message}'"
        logger.info(f"  ✓ '{message}' → None (correct)")
    
    logger.info("✓ Command detection working correctly")
    return True


async def test_sandbox_connection():
    """Test sandbox connection (informational)."""
    client = SimpleSandboxClient()
    connected = client.is_connected()
    
    if connected:
        logger.info("✓ Sandbox connected")
    else:
        logger.warning("⚠ Sandbox not connected - will use local fallback")
    
    return True  # Don't fail


async def test_full_smoke_test():
    """Run full end-to-end smoke test."""
    logger.info("=" * 60)
    logger.info("STARTING FULL SMOKE TEST")
    logger.info("=" * 60)
    
    result = await run_overwatcher_command(
        project_id=0,
        message="run overwatcher",
    )
    
    logger.info("-" * 60)
    logger.info("RESULT SUMMARY:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Job ID: {result.job_id}")
    logger.info(f"  Spec ID: {result.spec.spec_id if result.spec else 'N/A'}")
    logger.info(f"  Decision: {result.overwatcher_decision}")
    logger.info(f"  Error: {result.error or 'None'}")
    
    if result.implementer_result:
        logger.info(f"  Output Path: {result.implementer_result.output_path}")
        logger.info(f"  Sandbox Used: {result.implementer_result.sandbox_used}")
    
    if result.verification_result:
        logger.info(f"  File Exists: {result.verification_result.file_exists}")
        logger.info(f"  Content Matches: {result.verification_result.content_matches}")
    
    logger.info("-" * 60)
    logger.info("STAGE TRACE:")
    for entry in result.stage_trace:
        logger.info(f"  [{entry['stage']}] {entry['status']}")
    
    logger.info("=" * 60)
    
    if result.success:
        logger.info("✓ SMOKE TEST PASSED")
    else:
        logger.error(f"✗ SMOKE TEST FAILED: {result.error}")
    
    return result.success


async def run_all_tests():
    """Run all tests."""
    tests = [
        ("Smoke Test Spec Creation", test_smoke_test_spec_creation),
        ("Command Detection", test_command_detection),
        ("Sandbox Connection", test_sandbox_connection),
        ("Full Smoke Test", test_full_smoke_test),
    ]
    
    passed = 0
    failed = 0
    
    logger.info("\n" + "=" * 60)
    logger.info("OVERWATCHER STANDALONE SMOKE TEST")
    logger.info("=" * 60 + "\n")
    
    for name, test_fn in tests:
        logger.info(f"\n--- Running: {name} ---\n")
        try:
            result = await test_fn()
            if result:
                passed += 1
                logger.info(f"\n✓ {name}: PASSED\n")
            else:
                failed += 1
                logger.error(f"\n✗ {name}: FAILED\n")
        except Exception as e:
            failed += 1
            logger.exception(f"\n✗ {name}: EXCEPTION - {e}\n")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
