#!/usr/bin/env python
"""Test script for multi-target file read functionality (Level 2.5)."""

import sys
sys.path.insert(0, 'D:/Orb')

from app.pot_spec.grounded.sandbox_discovery import extract_file_targets, is_multi_target_request

print("=" * 60)
print("MULTI-TARGET FILE READ TESTS (Level 2.5)")
print("=" * 60)

test_cases = [
    # (input_text, expected_target_count, description)
    ("read test on desktop and test2 on D drive", 2, "Basic: desktop + D drive"),
    ("read test on desktop and test2 on the D drive", 2, "With 'the': desktop + D drive"),
    ("show me file1 from documents and file2 from E:", 2, "Documents + E:"),
    ("read D:\\test.txt and C:\\data.txt", 2, "Explicit paths"),
    ("read test on desktop", 1, "Single file - should NOT be multi-target"),
    ("find all py files", 0, "Level 3 search - no specific targets"),
    ("read test on desktop and data on desktop", 2, "Same anchor, two files"),
    ("read file1 on D drive and file2 on D drive and file3 on E drive", 3, "Three files"),
]

passed = 0
failed = 0

for text, expected_count, description in test_cases:
    targets = extract_file_targets(text)
    is_multi = is_multi_target_request(text)
    actual_count = len(targets)
    
    # Check if count matches expected
    count_ok = actual_count == expected_count
    # Multi-target should be true if 2+ targets
    multi_ok = is_multi == (expected_count >= 2)
    
    status = "✓ PASS" if (count_ok and multi_ok) else "✗ FAIL"
    
    if count_ok and multi_ok:
        passed += 1
    else:
        failed += 1
    
    print(f"\n{status}: {description}")
    print(f"  Input: '{text}'")
    print(f"  Expected targets: {expected_count}, Got: {actual_count}")
    print(f"  Is multi-target: {is_multi}")
    if targets:
        for t in targets:
            print(f"    - {t}")

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)

# Exit with error code if any failed
sys.exit(0 if failed == 0 else 1)
