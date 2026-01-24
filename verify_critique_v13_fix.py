#!/usr/bin/env python3
"""
Verification script for Critique v1.3 - Deterministic Spec-Compliance Check

Run: python verify_critique_v13_fix.py

Expected output: All tests PASS
"""

import sys
import json

# Add the app directory to path
sys.path.insert(0, "D:\\Orb")

from app.llm.pipeline.critique import run_deterministic_spec_compliance_check
from app.llm.pipeline.critique_schemas import APPROVED_ARCHITECTURE_BLOCKER_TYPES


def test_new_blocker_types():
    """Test that new blocker types are added to approved list."""
    print("=" * 60)
    print("TEST: New blocker types added to APPROVED_ARCHITECTURE_BLOCKER_TYPES")
    print("=" * 60)
    
    required_types = [
        "platform_mismatch",
        "stack_mismatch",
        "scope_inflation",
        "spec_compliance",
    ]
    
    missing = []
    for bt in required_types:
        if bt in APPROVED_ARCHITECTURE_BLOCKER_TYPES:
            print(f"  ✅ '{bt}' present")
        else:
            print(f"  ❌ '{bt}' MISSING")
            missing.append(bt)
    
    if missing:
        print(f"\n❌ FAIL: Missing blocker types: {missing}")
        return False
    else:
        print(f"\n✅ PASS: All new blocker types present")
        return True


def test_stack_mismatch_pygame_vs_electron():
    """Test that Pygame discussed + Electron proposed = BLOCKER."""
    print("\n" + "=" * 60)
    print("TEST: Stack mismatch - Pygame discussed, Electron proposed")
    print("=" * 60)
    
    arch_content = """
    # Architecture Document
    
    ## Executive Summary
    Using Electron + React + TypeScript for a desktop Tetris game.
    
    ## Technology Stack
    - Runtime: Electron
    - Frontend: React 18 + TypeScript
    - Bundler: Vite
    - Testing: Playwright E2E
    """
    
    spec_json = json.dumps({
        "goal": "Build a classic Tetris game",
        "summary": "Python + Pygame implementation, bare minimum playable",
    })
    
    original_request = "I want to build a Tetris game using Python and Pygame"
    
    issues = run_deterministic_spec_compliance_check(
        arch_content=arch_content,
        spec_json=spec_json,
        original_request=original_request,
    )
    
    print(f"  Found {len(issues)} issue(s)")
    
    stack_mismatch = any(i.category == "stack_mismatch" for i in issues)
    
    if issues and stack_mismatch:
        for issue in issues:
            print(f"  ✅ [{issue.category}] {issue.description[:80]}...")
        print(f"\n✅ PASS: Stack mismatch correctly detected")
        return True
    else:
        print(f"\n❌ FAIL: Stack mismatch NOT detected")
        return False


def test_scope_inflation():
    """Test that minimal spec + production architecture = BLOCKER."""
    print("\n" + "=" * 60)
    print("TEST: Scope inflation - minimal spec, production architecture")
    print("=" * 60)
    
    arch_content = """
    # Architecture Document
    
    ## Technology Stack
    - Electron-builder for packaging
    - Playwright E2E testing
    - Telemetry and crash reporting
    - SQLite persistence in %APPDATA%
    - Settings UI with overlays
    - Authentication module
    """
    
    spec_json = json.dumps({
        "goal": "Build a simple game",
        "summary": "Bare minimum playable prototype, minimal scope",
    })
    
    issues = run_deterministic_spec_compliance_check(
        arch_content=arch_content,
        spec_json=spec_json,
        original_request="",
    )
    
    print(f"  Found {len(issues)} issue(s)")
    
    scope_inflation = any(i.category == "scope_inflation" for i in issues)
    
    if issues and scope_inflation:
        for issue in issues:
            print(f"  ✅ [{issue.category}] {issue.description[:80]}...")
        print(f"\n✅ PASS: Scope inflation correctly detected")
        return True
    else:
        print(f"\n❌ FAIL: Scope inflation NOT detected")
        return False


def test_platform_mismatch():
    """Test that Desktop spec + Mobile architecture = BLOCKER."""
    print("\n" + "=" * 60)
    print("TEST: Platform mismatch - Desktop spec, Mobile architecture")
    print("=" * 60)
    
    arch_content = """
    # Architecture Document
    
    ## Target Platform
    Building for Android and iOS mobile devices.
    """
    
    spec_json = json.dumps({
        "goal": "Build a Desktop application",
        "summary": "Desktop platform, dark mode",
    })
    
    issues = run_deterministic_spec_compliance_check(
        arch_content=arch_content,
        spec_json=spec_json,
        original_request="",
    )
    
    print(f"  Found {len(issues)} issue(s)")
    
    platform_mismatch = any(i.category == "platform_mismatch" for i in issues)
    
    if issues and platform_mismatch:
        for issue in issues:
            print(f"  ✅ [{issue.category}] {issue.description[:80]}...")
        print(f"\n✅ PASS: Platform mismatch correctly detected")
        return True
    else:
        print(f"\n❌ FAIL: Platform mismatch NOT detected")
        return False


def test_no_false_positives():
    """Test that aligned architecture passes."""
    print("\n" + "=" * 60)
    print("TEST: No false positives - aligned architecture should pass")
    print("=" * 60)
    
    arch_content = """
    # Architecture Document
    
    ## Technology Stack
    - Language: Python 3.11
    - Graphics: Pygame 2.5
    - Platform: Desktop (Windows)
    
    ## Scope
    Minimal Tetris implementation:
    - 10x20 playfield
    - 7 tetrominoes
    - Basic scoring
    """
    
    spec_json = json.dumps({
        "goal": "Build a classic Tetris game",
        "summary": "Python + Pygame, Desktop, minimal playable",
    })
    
    original_request = "Build me a Tetris game using Python and Pygame"
    
    issues = run_deterministic_spec_compliance_check(
        arch_content=arch_content,
        spec_json=spec_json,
        original_request=original_request,
    )
    
    print(f"  Found {len(issues)} issue(s)")
    
    if not issues:
        print(f"\n✅ PASS: No false positives - aligned architecture passed")
        return True
    else:
        for issue in issues:
            print(f"  ❌ False positive: [{issue.category}] {issue.description[:80]}...")
        print(f"\n❌ FAIL: False positive(s) detected")
        return False


def main():
    print("\n" + "=" * 60)
    print("Critique v1.3 - Deterministic Spec-Compliance Check")
    print("Verification Script")
    print("=" * 60)
    
    results = []
    
    results.append(("New blocker types", test_new_blocker_types()))
    results.append(("Stack mismatch (Pygame vs Electron)", test_stack_mismatch_pygame_vs_electron()))
    results.append(("Scope inflation", test_scope_inflation()))
    results.append(("Platform mismatch", test_platform_mismatch()))
    results.append(("No false positives", test_no_false_positives()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - v1.3 fix is working correctly!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
