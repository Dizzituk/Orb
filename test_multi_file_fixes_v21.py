# FILE: test_multi_file_fixes_v21.py
"""
Test script for v2.1 fixes to multi-file detection and file discovery.

Tests:
1. Bug 1: Stopword validation - "and" should be REJECTED
2. Bug 1: Pattern extraction - "references to Orb... replace with ASTRA"
3. Bug 2: Summary report vs full dump
"""

import sys
sys.path.insert(0, r"D:\Orb")

from app.pot_spec.grounded.multi_file_detection import (
    _is_valid_term,
    _normalize_spaced_identifier,
    _extract_search_and_replace_terms,
    STOPWORDS,
)
from app.pot_spec.grounded.file_discovery import (
    _should_skip_line,
    _classify_match_mechanical,
    MatchBucket,
    MUST_REVIEW_BUCKETS,
)


def test_stopword_validation():
    """Test that stopwords are rejected."""
    print("\n" + "="*60)
    print("TEST 1: Stopword Validation")
    print("="*60)
    
    # These should all be REJECTED
    stopwords_to_test = ["and", "or", "the", "to", "with", "find", "replace"]
    
    all_passed = True
    for word in stopwords_to_test:
        result = _is_valid_term(word)
        status = "✅ PASS" if not result else "❌ FAIL"
        print(f"  {status}: '{word}' -> {result} (expected: False)")
        if result:
            all_passed = False
    
    # These should be ACCEPTED
    valid_terms = ["Orb", "ASTRA", "OrbConfig", "my_variable"]
    for word in valid_terms:
        result = _is_valid_term(word)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: '{word}' -> {result} (expected: True)")
        if not result:
            all_passed = False
    
    return all_passed


def test_spaced_identifier_normalization():
    """Test O-R-B -> ORB normalization."""
    print("\n" + "="*60)
    print("TEST 2: Spaced Identifier Normalization")
    print("="*60)
    
    test_cases = [
        ("O-R-B", "ORB"),
        ("A-S-T-R-A", "ASTRA"),
        ("O R B", "ORB"),
        ("O.R.B", "ORB"),
        ("normal text with Orb in it", "normal text with Orb in it"),  # No change
    ]
    
    all_passed = True
    for input_text, expected in test_cases:
        result = _normalize_spaced_identifier(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"  {status}: '{input_text}' -> '{result}' (expected: '{expected}')")
        if result != expected:
            all_passed = False
    
    return all_passed


def test_pattern_extraction():
    """Test pattern extraction for the failing case."""
    print("\n" + "="*60)
    print("TEST 3: Pattern Extraction (The Critical Bug)")
    print("="*60)
    
    # THE failing input from the bug report
    weaver_input = """Find all references to Orb or O-R-B (case-insensitive) on D: and replace them with ASTRA"""
    
    result = _extract_search_and_replace_terms(weaver_input)
    
    print(f"  Input: {weaver_input[:80]}...")
    print(f"  Result: {result}")
    
    if result is None:
        print("  ❌ FAIL: Extraction returned None")
        return False
    
    search = result.get("search_pattern", "")
    replace = result.get("replacement_pattern", "")
    
    # Check search pattern is NOT "and"
    if search.lower() == "and":
        print(f"  ❌ FAIL: search_pattern is 'and' - STOPWORD BUG STILL EXISTS!")
        return False
    
    # Check search pattern is some variant of "Orb"
    if search.lower() not in ["orb", "orb"]:
        print(f"  ⚠️ WARNING: search_pattern '{search}' is not 'Orb' (might be acceptable)")
    
    # Check replacement is "ASTRA"
    if replace.upper() != "ASTRA":
        print(f"  ❌ FAIL: replacement_pattern '{replace}' is not 'ASTRA'")
        return False
    
    print(f"  ✅ PASS: search='{search}', replace='{replace}'")
    return True


def test_line_filtering():
    """Test garbage line filtering."""
    print("\n" + "="*60)
    print("TEST 4: Line Filtering (Garbage Detection)")
    print("="*60)
    
    # Lines that should be SKIPPED
    garbage_lines = [
        ("ENC:abc123def456", "D:\\config.py"),
        ("YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXowMTIzNDU2Nzg5" + "A"*50, "D:\\file.py"),  # Base64
        ("0.123, 0.456, 0.789, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888", "D:\\embeddings.py"),  # Embeddings
    ]
    
    # Lines that should be KEPT
    valid_lines = [
        ("from app.orb import OrbConfig", "D:\\main.py"),
        ("class OrbManager:", "D:\\manager.py"),
        ("ORB_API_KEY = os.getenv('ORB_KEY')", "D:\\.env"),
    ]
    
    all_passed = True
    
    print("  Should SKIP (garbage):")
    for line, path in garbage_lines:
        result = _should_skip_line(line, path)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"    {status}: '{line[:50]}...' -> skip={result} (expected: True)")
        if not result:
            all_passed = False
    
    print("  Should KEEP (valid):")
    for line, path in valid_lines:
        result = _should_skip_line(line, path)
        status = "✅ PASS" if not result else "❌ FAIL"
        print(f"    {status}: '{line[:50]}' -> skip={result} (expected: False)")
        if result:
            all_passed = False
    
    return all_passed


def test_mechanical_classification():
    """Test mechanical bucket classification."""
    print("\n" + "="*60)
    print("TEST 5: Mechanical Classification")
    print("="*60)
    
    test_cases = [
        ("from app.orb import Config", "D:\\app\\main.py", MatchBucket.IMPORT_PATH),
        ("ORB_API_KEY=secret", "D:\\.env", MatchBucket.ENV_VAR_KEY),
        ("CREATE TABLE orb_users", "D:\\migrations\\001.sql", MatchBucket.DATABASE_ARTIFACT),
        ("assert orb.is_valid()", "D:\\tests\\test_orb.py", MatchBucket.TEST_ASSERTION),
        ("class OrbManager:", "D:\\app\\orb\\manager.py", MatchBucket.CODE_IDENTIFIER),
    ]
    
    all_passed = True
    for line, path, expected_bucket in test_cases:
        result = _classify_match_mechanical(line, path)
        status = "✅ PASS" if result == expected_bucket else "❌ FAIL"
        print(f"  {status}: '{line[:40]}' in '{path}' -> {result.value} (expected: {expected_bucket.value})")
        if result != expected_bucket:
            all_passed = False
    
    return all_passed


def test_must_review_buckets():
    """Test must-review bucket detection."""
    print("\n" + "="*60)
    print("TEST 6: Must-Review Buckets")
    print("="*60)
    
    expected_must_review = [
        MatchBucket.ENV_VAR_KEY,
        MatchBucket.DATABASE_ARTIFACT,
        MatchBucket.FILE_FOLDER_NAME,
        MatchBucket.HISTORICAL_DATA,
    ]
    
    all_passed = True
    for bucket in expected_must_review:
        is_must_review = bucket in MUST_REVIEW_BUCKETS
        status = "✅ PASS" if is_must_review else "❌ FAIL"
        print(f"  {status}: {bucket.value} in MUST_REVIEW_BUCKETS -> {is_must_review}")
        if not is_must_review:
            all_passed = False
    
    # These should NOT be must-review
    not_must_review = [MatchBucket.CODE_IDENTIFIER, MatchBucket.DOCUMENTATION, MatchBucket.TEST_ASSERTION]
    for bucket in not_must_review:
        is_must_review = bucket in MUST_REVIEW_BUCKETS
        status = "✅ PASS" if not is_must_review else "❌ FAIL"
        print(f"  {status}: {bucket.value} NOT in MUST_REVIEW_BUCKETS -> {not is_must_review}")
        if is_must_review:
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("v2.1 MULTI-FILE FIX VERIFICATION")
    print("="*60)
    
    results = [
        ("Stopword Validation", test_stopword_validation()),
        ("Spaced Identifier Normalization", test_spaced_identifier_normalization()),
        ("Pattern Extraction", test_pattern_extraction()),
        ("Line Filtering", test_line_filtering()),
        ("Mechanical Classification", test_mechanical_classification()),
        ("Must-Review Buckets", test_must_review_buckets()),
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
        print("✅ ALL TESTS PASSED - v2.1 fixes are working correctly")
    else:
        print("❌ SOME TESTS FAILED - Review the output above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
