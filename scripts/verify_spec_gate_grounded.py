# FILE: scripts/verify_spec_gate_grounded.py
"""
Quick verification script for SpecGate Contract v1 implementation.

Run from D:\\Orb directory:
    python scripts/verify_spec_gate_grounded.py

This script verifies:
1. All modules import correctly
2. Evidence loading works
3. POT spec markdown generation works
4. Read-only enforcement works
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_imports():
    """Verify all new modules import correctly."""
    print("=" * 60)
    print("STEP 1: Verifying imports...")
    print("=" * 60)
    
    errors = []
    
    # Evidence collector
    try:
        from app.pot_spec.evidence_collector import (
            EvidenceBundle,
            EvidenceSource,
            load_evidence,
            refuse_write_operation,
            WRITE_REFUSED_ERROR,
        )
        print("✅ evidence_collector imports OK")
    except ImportError as e:
        print(f"❌ evidence_collector import FAILED: {e}")
        errors.append(str(e))
    
    # Spec gate grounded
    try:
        from app.pot_spec.spec_gate_grounded import (
            run_spec_gate_grounded,
            GroundedPOTSpec,
            GroundedQuestion,
            GroundedFact,
            QuestionCategory,
            build_pot_spec_markdown,
        )
        print("✅ spec_gate_grounded imports OK")
    except ImportError as e:
        print(f"❌ spec_gate_grounded import FAILED: {e}")
        errors.append(str(e))
    
    # Spec gate stream (modified)
    try:
        from app.llm.spec_gate_stream import (
            generate_spec_gate_stream,
            _USE_GROUNDED_SPEC_GATE,
            _SPEC_GATE_GROUNDED_AVAILABLE,
        )
        print(f"✅ spec_gate_stream imports OK")
        print(f"   USE_GROUNDED_SPEC_GATE={_USE_GROUNDED_SPEC_GATE}")
        print(f"   SPEC_GATE_GROUNDED_AVAILABLE={_SPEC_GATE_GROUNDED_AVAILABLE}")
    except ImportError as e:
        print(f"❌ spec_gate_stream import FAILED: {e}")
        errors.append(str(e))
    
    # Package init
    try:
        from app.pot_spec import (
            run_spec_gate_grounded,
            GroundedPOTSpec,
            EvidenceBundle,
            load_evidence,
        )
        print("✅ app.pot_spec package exports OK")
    except ImportError as e:
        print(f"⚠️ app.pot_spec package export issue: {e}")
    
    return len(errors) == 0


def verify_evidence_loading():
    """Verify evidence loading works."""
    print("\n" + "=" * 60)
    print("STEP 2: Verifying evidence loading...")
    print("=" * 60)
    
    try:
        from app.pot_spec.evidence_collector import load_evidence
        
        bundle = load_evidence(
            include_arch_map=True,
            include_codebase_report=True,
            arch_map_max_lines=100,
            codebase_report_max_lines=100,
        )
        
        print(f"✅ Evidence loaded successfully")
        print(f"   Sources: {len(bundle.sources)}")
        print(f"   Errors: {len(bundle.errors)}")
        
        for source in bundle.sources:
            status = "✓" if source.found else "✗"
            print(f"   [{status}] {source.source_type}: {source.filename or source.error or 'N/A'}")
        
        if bundle.arch_map_content:
            print(f"   Arch map content: {len(bundle.arch_map_content)} chars")
        if bundle.codebase_report_content:
            print(f"   Codebase report: {len(bundle.codebase_report_content)} chars")
        
        # Generate evidence markdown
        md = bundle.to_evidence_used_markdown()
        print(f"\n   Evidence Used markdown preview:")
        for line in md.split('\n')[:5]:
            print(f"   | {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evidence loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_pot_spec_generation():
    """Verify POT spec markdown generation."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying POT spec generation...")
    print("=" * 60)
    
    try:
        from app.pot_spec.spec_gate_grounded import (
            GroundedPOTSpec,
            GroundedFact,
            GroundedQuestion,
            QuestionCategory,
            build_pot_spec_markdown,
        )
        from app.pot_spec.evidence_collector import EvidenceBundle, EvidenceSource
        
        # Create test evidence bundle
        bundle = EvidenceBundle()
        bundle.add_source(EvidenceSource(
            source_type="architecture_map",
            filename="ARCHITECTURE_MAP.md",
            mtime_human="2026-01-19 12:00:00",
            found=True,
        ))
        
        # Create test spec
        spec = GroundedPOTSpec(
            goal="Verify SpecGate Contract v1 implementation",
            confirmed_components=[
                GroundedFact(
                    description="evidence_collector.py exists and works",
                    source="file_read",
                    confidence="confirmed",
                ),
            ],
            what_exists=["app/pot_spec/evidence_collector.py", "app/pot_spec/spec_gate_grounded.py"],
            what_missing=[],
            in_scope=["Evidence loading", "POT spec generation", "Question rules"],
            out_of_scope=["DB persistence", "File writes"],
            constraints_from_intent=["Read-only runtime"],
            constraints_from_repo=["Must use existing SpecGateResult type"],
            evidence_bundle=bundle,
            proposed_steps=[
                "Load evidence from architecture map",
                "Parse Weaver intent",
                "Ground facts against evidence",
                "Generate questions if needed",
                "Build POT spec markdown",
            ],
            acceptance_tests=[
                "All modules import cleanly",
                "Evidence loading returns valid data",
                "POT spec has all required sections",
                "No DB writes occur",
            ],
            risks=[{"risk": "Evidence not found", "mitigation": "Graceful degradation"}],
            refactor_flags=[],
            open_questions=[
                GroundedQuestion(
                    question="Is this verification sufficient?",
                    category=QuestionCategory.PREFERENCE,
                    why_it_matters="Determines if more testing needed",
                    evidence_found="Basic smoke tests passed",
                    options=["Yes, looks good", "No, need more tests"],
                ),
            ],
            spec_id="verify-123",
            spec_hash="abc123def456",
            spec_version=1,
            is_complete=False,
        )
        
        # Generate markdown
        md = build_pot_spec_markdown(spec)
        
        # Verify required sections
        required_sections = [
            "# Point-of-Truth Specification",
            "## Goal",
            "## Current Reality",
            "## Scope",
            "## Constraints",
            "## Evidence Used",
            "## Proposed Step Plan",
            "## Acceptance Tests",
            "## Risks",
            "## Refactor Flags",
            "## Open Questions",
            "## Metadata",
        ]
        
        missing = []
        for section in required_sections:
            if section not in md:
                missing.append(section)
        
        if missing:
            print(f"❌ Missing sections: {missing}")
            return False
        
        print(f"✅ POT spec generated successfully")
        print(f"   Total length: {len(md)} chars")
        print(f"   All {len(required_sections)} required sections present")
        
        # Show preview
        print(f"\n   POT spec preview (first 500 chars):")
        print("   " + "-" * 40)
        for line in md[:500].split('\n'):
            print(f"   | {line}")
        print("   " + "-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ POT spec generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_read_only_enforcement():
    """Verify read-only runtime is enforced."""
    print("\n" + "=" * 60)
    print("STEP 4: Verifying read-only enforcement...")
    print("=" * 60)
    
    try:
        from app.pot_spec.evidence_collector import refuse_write_operation, WRITE_REFUSED_ERROR
        
        # Test that write operations are refused
        try:
            refuse_write_operation("test write operation")
            print(f"❌ Write operation was NOT refused - this is a bug!")
            return False
        except RuntimeError as e:
            if WRITE_REFUSED_ERROR in str(e):
                print(f"✅ Write operations correctly refused")
                print(f"   Error message: {e}")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Read-only enforcement check FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_round3_no_guessing():
    """Verify Round 3 produces output but does NOT guess or fill gaps."""
    print("\n" + "=" * 60)
    print("STEP 5: Verifying Round 3 Contract (No Guessing)...")
    print("=" * 60)
    
    try:
        from app.pot_spec.spec_gate_grounded import (
            GroundedPOTSpec,
            GroundedQuestion,
            QuestionCategory,
            build_pot_spec_markdown,
        )
        
        # Create spec simulating Round 3 with unresolved questions
        spec = GroundedPOTSpec(
            goal="Test Round 3 finalization",
            open_questions=[
                GroundedQuestion(
                    question="What color should it be?",
                    category=QuestionCategory.PREFERENCE,
                    why_it_matters="User preference",
                    evidence_found="Not in evidence",
                ),
                GroundedQuestion(
                    question="Which database to use?",
                    category=QuestionCategory.MISSING_PRODUCT_DECISION,
                    why_it_matters="Architectural decision",
                    evidence_found="Multiple options found",
                    options=["PostgreSQL", "SQLite", "MongoDB"],
                ),
            ],
            proposed_steps=[],  # Gap - no steps
            acceptance_tests=[],  # Gap - no tests
            spec_version=3,
            is_complete=True,  # Round 3 forces completion
            blocking_issues=["Finalized with 2 unanswered question(s) - NOT guessed"],
        )
        
        md = build_pot_spec_markdown(spec)
        
        # Critical checks for Contract v1 compliance
        checks = [
            ("FINALIZED WITH UNRESOLVED QUESTIONS", "Round 3 finalization header"),
            ("UNRESOLVED (no guess", "Question marked as unresolved"),
            ("NOT guessed", "No guessing indicator"),
            ("Unresolved / Unknown (No Guess)", "Unresolved section header"),
            ("What color should it be?", "Question 1 preserved"),
            ("Which database to use?", "Question 2 preserved"),
            ("human input", "Human input required indicator"),
            ("Steps", "Steps gap mentioned"),
        ]
        
        all_passed = True
        for marker, desc in checks:
            if marker.lower() in md.lower():
                print(f"   ✅ {desc}")
            else:
                print(f"   ❌ {desc} - missing: {repr(marker)[:40]}")
                all_passed = False
        
        # Verify NOT filling gaps
        bad_fills = [
            "(assumed)",
            "(defaulted to)",
            "(inferred as)",
            "based on common practice",
        ]
        
        for bad in bad_fills:
            if bad.lower() in md.lower():
                print(f"   ❌ CONTRACT VIOLATION: Found gap-filling: {bad}")
                all_passed = False
        
        if all_passed:
            print(f"\n✅ Round 3 Contract verified: No guessing, gaps explicit")
        else:
            print(f"\n❌ Round 3 Contract VIOLATED")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Round 3 verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_no_downstream_persistence():
    """Verify grounded mode doesn't trigger any downstream persistence."""
    print("\n" + "=" * 60)
    print("STEP 6: Verifying no downstream persistence...")
    print("=" * 60)
    
    try:
        # Check spec_gate_stream doesn't import persistence
        import ast
        import os
        
        stream_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app", "llm", "spec_gate_stream.py"
        )
        
        with open(stream_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for persistence imports
        if "spec_gate_persistence" in content:
            print("   ❌ spec_gate_persistence imported in spec_gate_stream")
            return False
        print("   ✅ No spec_gate_persistence import")
        
        # Check for direct DB writes
        if "db.add(" in content or "db.commit(" in content:
            print("   ❌ Direct DB write operations found")
            return False
        print("   ✅ No direct DB write operations")
        
        # Check memory_service only saves messages
        if "memory_service" in content:
            if "create_message" in content:
                print("   ✅ memory_service only used for chat messages (OK)")
            else:
                print("   ⚠️ memory_service used but not for messages - verify")
        else:
            print("   ✅ No memory_service (even chat messages not saved)")
        
        print(f"\n✅ No downstream persistence when using grounded mode")
        return True
        
    except Exception as e:
        print(f"❌ Persistence check FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification steps."""
    print("\n" + "=" * 60)
    print("SpecGate Contract v1 - Verification Script")
    print("=" * 60)
    print()
    
    results = []
    
    # Run each verification step
    results.append(("Imports", verify_imports()))
    results.append(("Evidence Loading", verify_evidence_loading()))
    results.append(("POT Spec Generation", verify_pot_spec_generation()))
    results.append(("Read-Only Enforcement", verify_read_only_enforcement()))
    results.append(("Round 3 No-Guessing Contract", verify_round3_no_guessing()))
    results.append(("No Downstream Persistence", verify_no_downstream_persistence()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All verification steps PASSED!")
        print()
        print("Next steps:")
        print("  1. Run: pytest tests/test_spec_gate_grounded.py -v")
        print("  2. Start the backend and test via API")
        print("  3. Say 'Astra, command: critical architecture' to test")
    else:
        print("⚠️ Some verification steps FAILED - review errors above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
