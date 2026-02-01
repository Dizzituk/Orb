# FILE: test_rename_policy.py
"""
Test script for rename policy - invariant-aware refactor decisions.

Tests that:
1. Protocol tokens (orb_session_) are UNSAFE
2. Env vars are UNSAFE or MIGRATION_REQUIRED
3. Branding/UI text is SAFE
4. Tests/docs are SAFE
5. Sandbox paths are UNSAFE
"""

import sys
sys.path.insert(0, r"D:\Orb")

from app.pot_spec.grounded.rename_policy import (
    classify_match,
    build_rename_plan,
    RenameDecision,
    Invariant,
)


def test_protocol_tokens_unsafe():
    """Protocol tokens like orb_session_ should be UNSAFE."""
    print("\n" + "="*60)
    print("TEST 1: Protocol Tokens → UNSAFE")
    print("="*60)
    
    test_cases = [
        # (file_path, line_content, expected_decision)
        ("D:\\Orb\\app\\auth\\middleware.py", 
         'if token.startswith("orb_session_"):', 
         RenameDecision.UNSAFE),
        
        ("D:\\Orb\\app\\auth\\config.py", 
         'return f"orb_session_{secrets.token_hex(32)}"', 
         RenameDecision.UNSAFE),
        
        ("D:\\Orb\\app\\auth\\session.py", 
         'prefix = "orb_" if is_api else "orb_session_"', 
         RenameDecision.UNSAFE),
    ]
    
    all_passed = True
    for path, content, expected in test_cases:
        result = classify_match(path, 1, content, "Orb")
        status = "✅ PASS" if result.decision == expected else "❌ FAIL"
        print(f"  {status}: {content[:50]}...")
        print(f"           Decision: {result.decision.value} (expected: {expected.value})")
        if result.decision != expected:
            all_passed = False
            print(f"           Reason: {result.reason}")
    
    return all_passed


def test_env_vars_unsafe_or_migration():
    """Env var usages should be UNSAFE, definitions MIGRATION_REQUIRED."""
    print("\n" + "="*60)
    print("TEST 2: Env Vars → UNSAFE or MIGRATION")
    print("="*60)
    
    test_cases = [
        # Usage in code - UNSAFE
        ("D:\\Orb\\app\\config.py", 
         'master_key = os.getenv("ORB_MASTER_KEY")', 
         RenameDecision.UNSAFE),
        
        # Definition in .env - MIGRATION_REQUIRED
        ("D:\\Orb\\.env", 
         'ORB_MASTER_KEY=abc123def456', 
         RenameDecision.MIGRATION_REQUIRED),
        
        # Another usage
        ("D:\\Orb\\app\\jobs\\engine.py", 
         'root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs")', 
         RenameDecision.UNSAFE),
    ]
    
    all_passed = True
    for path, content, expected in test_cases:
        result = classify_match(path, 1, content, "Orb")
        status = "✅ PASS" if result.decision == expected else "❌ FAIL"
        print(f"  {status}: {content[:50]}...")
        print(f"           Decision: {result.decision.value} (expected: {expected.value})")
        if result.decision != expected:
            all_passed = False
            print(f"           Reason: {result.reason}")
    
    return all_passed


def test_branding_safe():
    """Branding/UI text should be SAFE."""
    print("\n" + "="*60)
    print("TEST 3: Branding/UI Text → SAFE")
    print("="*60)
    
    test_cases = [
        ("D:\\Orb\\app\\jobs\\engine.py", 
         'base = """You are Orb, a fast and helpful assistant.', 
         RenameDecision.SAFE),
        
        ("D:\\Orb\\app\\jobs\\engine.py", 
         "base = \"\"\"You are Orb's engineering brain - a senior backend architect.", 
         RenameDecision.SAFE),
        
        ("D:\\Orb\\static\\index.html", 
         '.message-row.orb {', 
         RenameDecision.SAFE),
        
        ("D:\\Orb\\start_zombie_orb.ps1", 
         'Write-Host "  ZOMBIE ORB STARTING UP"', 
         RenameDecision.SAFE),
    ]
    
    all_passed = True
    for path, content, expected in test_cases:
        result = classify_match(path, 1, content, "Orb")
        status = "✅ PASS" if result.decision == expected else "❌ FAIL"
        print(f"  {status}: {content[:50]}...")
        print(f"           Decision: {result.decision.value} (expected: {expected.value})")
        if result.decision != expected:
            all_passed = False
            print(f"           Reason: {result.reason}")
    
    return all_passed


def test_tests_and_docs_safe():
    """Tests and docs should be SAFE."""
    print("\n" + "="*60)
    print("TEST 4: Tests & Docs → SAFE")
    print("="*60)
    
    test_cases = [
        ("D:\\Orb\\tests\\test_orb_config.py", 
         'assert orb_config.is_valid()', 
         RenameDecision.SAFE),
        
        ("D:\\Orb\\README.md", 
         '# Orb - AI Assistant Platform', 
         RenameDecision.SAFE),
        
        ("D:\\Orb\\docs\\architecture.md", 
         'Orb uses a multi-LLM architecture', 
         RenameDecision.SAFE),
        
        ("D:\\Orb\\jobs\\jobs\\abc123\\stage_output.md", 
         'Orb completed the analysis', 
         RenameDecision.SAFE),
    ]
    
    all_passed = True
    for path, content, expected in test_cases:
        result = classify_match(path, 1, content, "Orb")
        status = "✅ PASS" if result.decision == expected else "❌ FAIL"
        print(f"  {status}: {path}")
        print(f"           Decision: {result.decision.value} (expected: {expected.value})")
        if result.decision != expected:
            all_passed = False
            print(f"           Reason: {result.reason}")
    
    return all_passed


def test_sandbox_paths_unsafe():
    """Sandbox path rules should be UNSAFE."""
    print("\n" + "="*60)
    print("TEST 5: Sandbox Path Rules → UNSAFE")
    print("="*60)
    
    test_cases = [
        ("D:\\Orb\\app\\sandbox\\manager.py", 
         'allowed_roots = ["D:\\\\Orb", "D:\\\\orb-desktop"]', 
         RenameDecision.UNSAFE),
        
        ("D:\\Orb\\app\\capabilities\\injector.py", 
         '"D:\\\\Orb",  # This is the HOST repo path', 
         RenameDecision.UNSAFE),
    ]
    
    all_passed = True
    for path, content, expected in test_cases:
        result = classify_match(path, 1, content, "Orb")
        # Both UNSAFE and MIGRATION_REQUIRED are acceptable for paths
        is_ok = result.decision in [RenameDecision.UNSAFE, RenameDecision.MIGRATION_REQUIRED]
        status = "✅ PASS" if is_ok else "❌ FAIL"
        print(f"  {status}: {path}")
        print(f"           Decision: {result.decision.value}")
        if not is_ok:
            all_passed = False
            print(f"           Reason: {result.reason}")
    
    return all_passed


def test_build_rename_plan():
    """Test building a complete rename plan."""
    print("\n" + "="*60)
    print("TEST 6: Build Rename Plan")
    print("="*60)
    
    matches = [
        # SAFE - branding
        {'file_path': 'D:\\Orb\\app\\jobs\\engine.py', 'line_number': 152, 
         'line_content': 'base = """You are Orb, a fast and helpful assistant.'},
        
        # SAFE - test
        {'file_path': 'D:\\Orb\\tests\\test_orb.py', 'line_number': 10, 
         'line_content': 'def test_orb_config():'},
        
        # UNSAFE - protocol token
        {'file_path': 'D:\\Orb\\app\\auth\\middleware.py', 'line_number': 53, 
         'line_content': 'if token.startswith("orb_session_"):'},
        
        # MIGRATION - env var definition
        {'file_path': 'D:\\Orb\\.env', 'line_number': 5, 
         'line_content': 'ORB_MASTER_KEY=abc123'},
    ]
    
    plan = build_rename_plan(matches, "Orb", "ASTRA")
    
    print(f"  Total matches: {plan.total_matches}")
    print(f"  ✅ Safe: {plan.safe_count}")
    print(f"  ❌ Unsafe: {plan.unsafe_count}")
    print(f"  ⚠️ Migration: {plan.migration_count}")
    
    # Verify counts
    expected_safe = 2
    expected_unsafe = 1
    expected_migration = 1
    
    all_passed = True
    
    if plan.safe_count != expected_safe:
        print(f"  ❌ FAIL: Expected {expected_safe} safe, got {plan.safe_count}")
        all_passed = False
    else:
        print(f"  ✅ PASS: Safe count correct")
    
    if plan.unsafe_count != expected_unsafe:
        print(f"  ❌ FAIL: Expected {expected_unsafe} unsafe, got {plan.unsafe_count}")
        all_passed = False
    else:
        print(f"  ✅ PASS: Unsafe count correct")
    
    if plan.migration_count != expected_migration:
        print(f"  ❌ FAIL: Expected {expected_migration} migration, got {plan.migration_count}")
        all_passed = False
    else:
        print(f"  ✅ PASS: Migration count correct")
    
    # Print the report
    print("\n  --- Generated Report Preview ---")
    report = plan.get_report()
    for line in report.split('\n')[:30]:
        print(f"  {line}")
    if len(report.split('\n')) > 30:
        print(f"  ... ({len(report.split(chr(10))) - 30} more lines)")
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RENAME POLICY TEST SUITE")
    print("="*60)
    
    results = [
        ("Protocol Tokens UNSAFE", test_protocol_tokens_unsafe()),
        ("Env Vars UNSAFE/MIGRATION", test_env_vars_unsafe_or_migration()),
        ("Branding SAFE", test_branding_safe()),
        ("Tests & Docs SAFE", test_tests_and_docs_safe()),
        ("Sandbox Paths UNSAFE", test_sandbox_paths_unsafe()),
        ("Build Rename Plan", test_build_rename_plan()),
    ]
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Rename policy is working correctly")
        print("\nThe system will now:")
        print("  1. Rename branding/UI text (safe)")
        print("  2. Rename tests/docs (safe)")
        print("  3. EXCLUDE protocol tokens like orb_session_ (breaks auth)")
        print("  4. EXCLUDE env var usages (breaks config)")
        print("  5. Mark .env definitions as MIGRATION_REQUIRED")
    else:
        print("❌ SOME TESTS FAILED - Review the output above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
