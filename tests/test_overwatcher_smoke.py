# FILE: tests/test_overwatcher_smoke.py
"""Overwatcher Smoke Test: End-to-end test of Overwatcher pipeline.

This test verifies:
1. Spec resolution (smoke test spec creation)
2. Evidence bundle building
3. Overwatcher decision flow (auto-approve without LLM)
4. Implementer execution (file creation)
5. Verification (file exists + content match)
6. STAGE_TRACE logging

Run with:
    python -m pytest tests/test_overwatcher_smoke.py -v
    
Or directly:
    python tests/test_overwatcher_smoke.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_smoke_test_spec_creation():
    """Test that smoke test spec is created correctly."""
    from app.overwatcher.overwatcher_command import create_smoke_test_spec
    
    spec = create_smoke_test_spec()
    
    assert spec.spec_id.startswith("smoke-test-"), f"Unexpected spec_id: {spec.spec_id}"
    assert len(spec.spec_hash) == 64, f"Invalid hash length: {len(spec.spec_hash)}"
    assert spec.project_id == 0
    assert spec.title == "Sandbox Hello Test"
    
    logger.info(f"✓ Smoke test spec created: {spec.spec_id}")
    return True


async def test_evidence_bundle_building():
    """Test that evidence bundle is built correctly."""
    from app.overwatcher.overwatcher_command import (
        create_smoke_test_spec,
        build_overwatcher_evidence,
    )
    from uuid import uuid4
    
    spec = create_smoke_test_spec()
    job_id = str(uuid4())
    
    evidence = build_overwatcher_evidence(
        job_id=job_id,
        spec=spec,
        artifacts={},
        task_description="Test task",
    )
    
    assert evidence.job_id == job_id
    assert evidence.spec_id == spec.spec_id
    assert evidence.spec_hash == spec.spec_hash
    assert evidence.strike_number == 1
    assert len(evidence.file_changes) == 1
    
    logger.info(f"✓ Evidence bundle built: chunk_id={evidence.chunk_id}")
    return True


async def test_command_detection():
    """Test that Overwatcher command patterns are detected."""
    from app.overwatcher.overwatcher_routing import (
        detect_overwatcher_command,
        OverwatcherCommandType,
    )
    
    # Should detect
    test_cases = [
        ("run overwatcher", OverwatcherCommandType.RUN),
        ("Astra, command: run overwatcher", OverwatcherCommandType.RUN),
        ("astra run overwatcher", OverwatcherCommandType.RUN),
        ("execute overwatcher", OverwatcherCommandType.RUN),
        ("start overwatcher", OverwatcherCommandType.RUN),
        ("overwatcher run", OverwatcherCommandType.RUN),
        ("invoke overwatcher", OverwatcherCommandType.RUN),
        ("trigger overwatcher", OverwatcherCommandType.RUN),
    ]
    
    for message, expected in test_cases:
        result = detect_overwatcher_command(message)
        assert result == expected, f"Failed for '{message}': expected {expected}, got {result}"
        logger.info(f"  ✓ '{message}' → {result}")
    
    # Should NOT detect
    non_commands = [
        "what is overwatcher",
        "tell me about the overwatcher",
        "hello world",
    ]
    
    for message in non_commands:
        result = detect_overwatcher_command(message)
        assert result is None, f"False positive for '{message}': got {result}"
        logger.info(f"  ✓ '{message}' → None (correct)")
    
    logger.info("✓ Command detection working correctly")
    return True


async def test_full_smoke_test():
    """Run full end-to-end smoke test (without LLM).
    
    This tests the complete flow:
    1. Create smoke test spec
    2. Build evidence
    3. Skip Overwatcher LLM call (auto-approve)
    4. Run Implementer (create file)
    5. Run verification
    """
    from app.overwatcher.overwatcher_command import run_overwatcher_command
    
    logger.info("=" * 60)
    logger.info("STARTING FULL SMOKE TEST")
    logger.info("=" * 60)
    
    result = await run_overwatcher_command(
        project_id=0,  # Smoke test project
        message="run overwatcher",
        llm_call_fn=None,  # No LLM - auto-approve
        use_smoke_test=True,
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


async def test_sandbox_connection():
    """Test sandbox connection (informational, not failing)."""
    from app.overwatcher.sandbox_client import get_sandbox_client
    
    client = get_sandbox_client()
    connected = client.is_connected()
    
    if connected:
        health = client.health()
        logger.info(f"✓ Sandbox connected: {health.status}")
        logger.info(f"  Repo root: {health.repo_root}")
        logger.info(f"  Scratch root: {health.scratch_root}")
    else:
        logger.warning("⚠ Sandbox not connected - tests will use local fallback")
    
    return True  # Don't fail if sandbox unavailable


async def run_all_tests():
    """Run all smoke tests."""
    tests = [
        ("Smoke Test Spec Creation", test_smoke_test_spec_creation),
        ("Evidence Bundle Building", test_evidence_bundle_building),
        ("Command Detection", test_command_detection),
        ("Sandbox Connection", test_sandbox_connection),
        ("Full Smoke Test", test_full_smoke_test),
    ]
    
    passed = 0
    failed = 0
    
    logger.info("\n" + "=" * 60)
    logger.info("OVERWATCHER SMOKE TEST SUITE")
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


# Pytest compatibility
def test_smoke_test_spec_creation_sync():
    assert asyncio.run(test_smoke_test_spec_creation())

def test_evidence_bundle_building_sync():
    assert asyncio.run(test_evidence_bundle_building())

def test_command_detection_sync():
    assert asyncio.run(test_command_detection())

def test_full_smoke_test_sync():
    assert asyncio.run(test_full_smoke_test())


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
