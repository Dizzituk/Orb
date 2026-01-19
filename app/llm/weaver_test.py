# FILE: app/llm/weaver_test.py
r"""
Simple test harness for the new Weaver (v3.0).

Usage:
    cd D:\Orb
    python -m app.llm.weaver_test
    
    Or for interactive testing:
    python -m app.llm.weaver_test --interactive

v1.1 (2026-01-19): Fixed syntax warning.
v1.0 (2026-01-19): Initial test harness for locked Weaver behaviour.
"""
from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_weave_simple(text: str) -> str:
    """Test the simple weave function directly."""
    from app.llm.weaver_simple import weave
    return weave(text)


def run_tests():
    """Run the 3 standard test cases."""
    print("=" * 70)
    print("WEAVER v3.0 - LOCKED BEHAVIOUR TEST HARNESS")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Test 1: Normal ramble",
            "input": "I want an app that runs on Android, it needs to be secure, and it should track my deliveries...",
            "expected": ["Android", "secure", "deliveries"],
            "not_expected": ["React", "Firebase", "Node.js"],  # Should not invent tech
        },
        {
            "name": "Test 2: Contradiction preserved",
            "input": "It must be offline only... but also it should sync live to the cloud...",
            "expected": ["offline", "cloud"],  # Both should appear
            "not_expected": [],
        },
        {
            "name": "Test 3: No goal - question allowed",
            "input": "Just make it better",
            "expected": ["?", "clarification"],  # Should ask a question
            "not_expected": [],
        },
    ]
    
    all_passed = True
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"{tc['name']}")
        print(f"{'='*70}")
        print(f"\n📝 INPUT:\n{tc['input']}\n")
        
        try:
            result = test_weave_simple(tc["input"])
            print(f"📤 OUTPUT:\n{result}\n")
            
            # Check expected content
            print("✓ CHECKS:")
            passed = True
            
            for expected in tc["expected"]:
                if expected.lower() in result.lower():
                    print(f"  ✅ Contains '{expected}'")
                else:
                    print(f"  ❌ Missing '{expected}'")
                    passed = False
                    all_passed = False
            
            for not_expected in tc["not_expected"]:
                if not_expected.lower() not in result.lower():
                    print(f"  ✅ Does not contain '{not_expected}' (good)")
                else:
                    print(f"  ⚠️  Unexpectedly contains '{not_expected}'")
            
            print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"{'='*70}")
    
    return all_passed


def interactive_mode():
    """Interactive testing mode."""
    print("=" * 70)
    print("WEAVER v3.0 - INTERACTIVE TEST MODE")
    print("=" * 70)
    print("\nEnter your ramble text (type 'quit' to exit):\n")
    
    while True:
        try:
            text = input(">>> ")
            if text.lower() in ('quit', 'exit', 'q'):
                break
            if not text.strip():
                continue
            
            print("\n📤 OUTPUT:")
            result = test_weave_simple(text)
            print(result)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
    
    print("\nGoodbye!")


def main():
    """Main entry point."""
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, skip
    
    args = sys.argv[1:]
    
    if not args:
        # No arguments - run standard tests
        success = run_tests()
        sys.exit(0 if success else 1)
    
    if args[0] in ('--interactive', '-i'):
        interactive_mode()
    elif args[0] in ('--test', '-t'):
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        # Treat arguments as input text
        text = " ".join(args)
        print("📝 INPUT:", text)
        print()
        result = test_weave_simple(text)
        print("📤 OUTPUT:")
        print(result)


if __name__ == "__main__":
    main()
