# FILE: app/llm/local_tools/zobie/test_multiline_parser.py
"""Test multi-line quoted content parsing fix (v5.7).

Run with: python -m app.llm.local_tools.zobie.test_multiline_parser
"""

import sys
from .fs_command_parser import parse_command_mode, _extract_quoted_arguments


def test_tokenizer():
    """Test _extract_quoted_arguments directly."""
    print("\n=== TOKENIZER TESTS ===\n")
    
    # Test 1: Simple two quoted args
    text1 = '"path/to/file" "content here"'
    result1 = _extract_quoted_arguments(text1, debug=True)
    print(f"Test 1: {result1}")
    assert result1 == ['path/to/file', 'content here'], f"Expected 2 args, got {result1}"
    print("✓ Test 1 passed\n")
    
    # Test 2: Multi-line quoted content
    text2 = '''"D:\\Orb\\test.py" "# Comment
def hello():
    print('world')"'''
    result2 = _extract_quoted_arguments(text2, debug=True)
    print(f"Test 2: {len(result2)} args")
    assert len(result2) == 2, f"Expected 2 args, got {len(result2)}"
    assert result2[0] == 'D:\\Orb\\test.py'
    assert 'def hello():' in result2[1]
    assert "print('world')" in result2[1]
    print("✓ Test 2 passed\n")
    
    # Test 3: Unquoted path + quoted content
    text3 = 'D:\\Orb\\file.py "some content"'
    result3 = _extract_quoted_arguments(text3, debug=True)
    print(f"Test 3: {result3}")
    assert len(result3) == 2
    assert result3[0] == 'D:\\Orb\\file.py'
    assert result3[1] == 'some content'
    print("✓ Test 3 passed\n")


def test_parse_write_command():
    """Test full write command parsing."""
    print("\n=== WRITE COMMAND TESTS ===\n")
    
    # Test 1: Single-line overwrite (should still work)
    cmd1 = 'overwrite "D:\\Orb\\test.py" "TEST_CONTENT"'
    result1 = parse_command_mode(cmd1)
    print(f"Test 1: {result1}")
    assert result1 is not None
    assert result1['command'] == 'overwrite'
    assert result1['content'] == 'TEST_CONTENT'
    print("✓ Test 1 passed\n")
    
    # Test 2: Multi-line overwrite (THE BUG FIX)
    cmd2 = '''overwrite "D:\\Orb\\app\\utils\\utils.py" "# FILE: orb/utils/math.py
+def add_tax(price: float) -> float:
    total = price * (1 + TAX_RATE)
    return round(total, 2)"'''
    result2 = parse_command_mode(cmd2)
    print(f"Test 2: path={result2.get('path')}")
    print(f"Test 2: content_len={len(result2.get('content') or '')}")
    print(f"Test 2: content preview: {repr((result2.get('content') or '')[:60])}")
    assert result2 is not None, "parse_command_mode returned None"
    assert result2['command'] == 'overwrite'
    assert result2['content'] is not None, "Content is None! Bug not fixed."
    assert 'add_tax' in result2['content'], "Content truncated"
    assert 'return round(total, 2)' in result2['content'], "Content truncated"
    print("✓ Test 2 passed - MULTI-LINE BUG FIXED!\n")
    
    # Test 3: Multi-line append
    cmd3 = '''append "D:\\Orb\\log.txt" "Entry 1
Entry 2
Entry 3"'''
    result3 = parse_command_mode(cmd3)
    print(f"Test 3: content={repr(result3.get('content'))}")
    assert result3 is not None
    assert result3['command'] == 'append'
    assert result3['content'] is not None
    assert 'Entry 2' in result3['content']
    print("✓ Test 3 passed\n")
    
    # Test 4: Fenced block (should still work)
    cmd4 = '''overwrite D:\\Orb\\test.py
```
def hello():
    pass
```'''
    result4 = parse_command_mode(cmd4)
    print(f"Test 4: content={repr(result4.get('content'))}")
    assert result4 is not None
    assert result4['command'] == 'overwrite'
    assert result4['content'] is not None
    assert 'def hello():' in result4['content']
    print("✓ Test 4 passed\n")


def main():
    print("=" * 60)
    print("Testing fs_command_parser v5.7 multi-line fix")
    print("=" * 60)
    
    try:
        test_tokenizer()
        test_parse_write_command()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
