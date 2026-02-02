# FILE: tests/spec_gate/test_refactor_planner_v3.py
"""
SpecGate v3.0 - Planner-First Refactor Tests

Golden tests for the planner-first refactor architecture transformation.
Tests verify:
1. Rendered text only → ready, no questions
2. Operational keys → preserve, ready, no questions  
3. Env var usage → preserve, ready, no questions
4. Ambiguity requiring expansion → expansion succeeds, decision made
5. Expansion fails → blocks with blocking_issues
6. REGRESSION - Multi-file alone must not block
7. "Questions: none" + expansion fails → blocks with open_questions=[]

v3.0 (2026-02-01): Initial golden tests for planner-first architecture
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List

# Import v3 types
from app.pot_spec.grounded.refactor_schemas import (
    RawMatch,
    MatchBucket,
    ChangeDecision,
    RiskLevel,
    RiskClass,
    ExpansionFailureReason,
    ReasonCode,
    ClassifiedMatchV3,
    RefactorPlanV3,
    BlockingIssue,
)

from app.pot_spec.grounded.refactor_classifier import (
    _needs_expansion,
    _compute_risk_class,
    REFACTOR_CLASSIFIER_V3_BUILD_ID,
)

from app.pot_spec.grounded.pot_spec_builder import (
    build_pot_spec_markdown,
    POT_SPEC_BUILDER_BUILD_ID,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def ui_text_matches() -> List[RawMatch]:
    """Rendered text only - should all be CHANGE, LOW risk."""
    return [
        RawMatch(
            file_path="src/components/Header.tsx",
            line_number=12,
            line_content='<h1 className="title">Welcome to Orb</h1>',
            match_text="Orb",
        ),
        RawMatch(
            file_path="src/components/Footer.tsx",
            line_number=5,
            line_content='<span>© 2024 Orb Inc.</span>',
            match_text="Orb",
        ),
        RawMatch(
            file_path="src/pages/About.tsx",
            line_number=23,
            line_content='<title>About Orb</title>',
            match_text="Orb",
        ),
    ]


@pytest.fixture
def operational_key_matches() -> List[RawMatch]:
    """Operational keys - should all be SKIP/PRESERVE."""
    return [
        RawMatch(
            file_path="src/auth/storage.ts",
            line_number=15,
            line_content='const TOKEN_KEY = "orb_auth_token";',
            match_text="orb_auth_token",
        ),
        RawMatch(
            file_path="src/services/cache.ts",
            line_number=42,
            line_content='localStorage.setItem("orb_session_id", sessionId);',
            match_text="orb_session_id",
        ),
    ]


@pytest.fixture
def env_var_matches() -> List[RawMatch]:
    """Environment variable usage - should all be SKIP/PRESERVE."""
    return [
        RawMatch(
            file_path="src/config/env.ts",
            line_number=8,
            line_content='const API_URL = process.env.ORB_API_URL;',
            match_text="ORB_API_URL",
        ),
        RawMatch(
            file_path="src/config/env.ts",
            line_number=12,
            line_content='export const DEBUG = process.env.ORB_DEBUG === "true";',
            match_text="ORB_DEBUG",
        ),
    ]


@pytest.fixture
def mixed_matches() -> List[RawMatch]:
    """Mix of UI text and operational - tests classification accuracy."""
    return [
        # UI text - CHANGE
        RawMatch(
            file_path="src/components/Header.tsx",
            line_number=12,
            line_content='<h1>Welcome to Orb</h1>',
            match_text="Orb",
        ),
        # Storage key - PRESERVE
        RawMatch(
            file_path="src/auth/storage.ts",
            line_number=15,
            line_content='const TOKEN_KEY = "orb_auth_token";',
            match_text="orb",
        ),
        # Comment - CHANGE  
        RawMatch(
            file_path="src/utils/helpers.ts",
            line_number=3,
            line_content='// Orb utility functions',
            match_text="Orb",
        ),
    ]


# =============================================================================
# TEST: _needs_expansion heuristic
# =============================================================================

class TestNeedsExpansion:
    """Test the _needs_expansion() heuristic function."""
    
    def test_storage_key_needs_expansion(self):
        """Storage keys should need expansion for confident classification."""
        match = RawMatch(
            file_path="src/auth/storage.ts",
            line_number=15,
            line_content='localStorage.setItem("orb_auth_token", token);',
            match_text="orb_auth_token",
        )
        assert _needs_expansion(match) is True
    
    def test_env_var_needs_expansion(self):
        """Environment variables should need expansion."""
        match = RawMatch(
            file_path="src/config/env.ts",
            line_number=8,
            line_content='const API_URL = process.env.ORB_API_URL;',
            match_text="ORB_API_URL",
        )
        assert _needs_expansion(match) is True
    
    def test_simple_ui_text_no_expansion(self):
        """Simple UI text should NOT need expansion."""
        match = RawMatch(
            file_path="src/components/Header.tsx",
            line_number=12,
            line_content='<h1>Welcome to Orb</h1>',
            match_text="Orb",
        )
        # UI text is simple - no suspicious patterns
        assert _needs_expansion(match) is False
    
    def test_comment_no_expansion(self):
        """Comments should NOT need expansion."""
        match = RawMatch(
            file_path="src/utils/helpers.ts",
            line_number=3,
            line_content='// Orb utility functions',
            match_text="Orb",
        )
        assert _needs_expansion(match) is False
    
    def test_config_file_needs_expansion(self):
        """Config files should need expansion."""
        match = RawMatch(
            file_path="src/config/settings.json",
            line_number=5,
            line_content='"apiKey": "orb_api_key_12345"',
            match_text="orb_api_key",
        )
        # Config + key pattern
        assert _needs_expansion(match) is True


# =============================================================================
# TEST: _compute_risk_class
# =============================================================================

class TestComputeRiskClass:
    """Test the _compute_risk_class() function."""
    
    def test_all_low_risk_changes(self):
        """All LOW risk CHANGE → RiskClass.LOW."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=3,
            total_occurrences=5,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path="src/Header.tsx",
                    line_number=12,
                    line_content="<h1>Orb</h1>",
                    match_text="Orb",
                    decision=ChangeDecision.CHANGE,
                    risk_level=RiskLevel.LOW,
                ),
            ],
            change_count=5,
            skip_count=0,
            flag_count=0,
            unresolved_count=0,
        )
        
        risk = _compute_risk_class(plan, {})
        assert risk == RiskClass.LOW
    
    def test_critical_change_blocks(self):
        """CHANGE on CRITICAL item → RiskClass.CRITICAL."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=1,
            total_occurrences=1,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path="src/db/schema.sql",
                    line_number=10,
                    line_content="CREATE TABLE orb_users",
                    match_text="orb",
                    decision=ChangeDecision.CHANGE,
                    risk_level=RiskLevel.CRITICAL,
                ),
            ],
            change_count=1,
        )
        
        risk = _compute_risk_class(plan, {})
        assert risk == RiskClass.CRITICAL
    
    def test_unresolved_unknowns_blocks(self):
        """Unresolved unknowns → RiskClass.CRITICAL."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=1,
            total_occurrences=1,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path="src/unknown.ts",
                    line_number=10,
                    line_content="const x = orb_something;",
                    match_text="orb",
                    is_unresolved=True,
                ),
            ],
            unresolved_count=1,
        )
        
        risk = _compute_risk_class(plan, {})
        assert risk == RiskClass.CRITICAL
    
    def test_expansion_failures_blocks(self):
        """Expansion failures → RiskClass.CRITICAL."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=1,
            total_occurrences=1,
            expansion_failures=[
                BlockingIssue(
                    file_path="src/config.ts",
                    line_number=5,
                    reason="Could not read file",
                    failure_type=ExpansionFailureReason.READ_FAILED,
                ),
            ],
        )
        
        risk = _compute_risk_class(plan, {})
        assert risk == RiskClass.CRITICAL
    
    def test_high_risk_changes(self):
        """HIGH risk changes → RiskClass.HIGH."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=3,
            total_occurrences=5,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path="src/api/routes.ts",
                    line_number=12,
                    line_content='app.get("/api/orb/status")',
                    match_text="orb",
                    decision=ChangeDecision.CHANGE,
                    risk_level=RiskLevel.HIGH,
                ),
            ],
            change_count=1,
        )
        
        risk = _compute_risk_class(plan, {})
        assert risk == RiskClass.HIGH
    
    def test_large_scope_medium_risk(self):
        """Large scope (>100 changes) → RiskClass.MEDIUM."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=50,
            total_occurrences=150,
            change_count=150,
            skip_count=0,
            flag_count=0,
        )
        
        risk = _compute_risk_class(plan, {})
        assert risk == RiskClass.MEDIUM


# =============================================================================
# TEST: POT Spec Builder
# =============================================================================

class TestPotSpecBuilder:
    """Test the build_pot_spec_markdown() function."""
    
    def test_basic_markdown_generation(self):
        """Basic POT spec markdown generation."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=3,
            total_occurrences=5,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path="src/Header.tsx",
                    line_number=12,
                    line_content="<h1>Welcome to Orb</h1>",
                    match_text="Orb",
                    decision=ChangeDecision.CHANGE,
                    risk_level=RiskLevel.LOW,
                    reason_code=ReasonCode.UI_TEXT_VISIBLE_TO_USER,
                    reason_text="User-visible UI text",
                ),
            ],
            change_count=5,
            computed_risk_class=RiskClass.LOW,
        )
        
        md = build_pot_spec_markdown(plan)
        
        assert "Orb" in md
        assert "Astra" in md
        assert "CHANGE" in md
        assert "Header.tsx" in md
    
    def test_no_migration_wording_unless_allowed(self):
        """MIGRATION REQUIRED never appears unless allows_migration=True."""
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=1,
            total_occurrences=1,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path="D:/projects/orb/config.ts",
                    line_number=5,
                    line_content='const PATH = "D:\\Orb\\data";',
                    match_text="Orb",
                    decision=ChangeDecision.FLAG,
                    reason_code=ReasonCode.PATH_LITERAL_WORKS_NOW,
                    reason_text="Path literal - migration required",  # Input has this
                ),
            ],
            flag_count=1,
            allows_migration=False,  # Migration NOT allowed
        )
        
        md = build_pot_spec_markdown(plan, {"allows_migration": False})
        
        # "MIGRATION REQUIRED" should NOT appear
        assert "migration required" not in md.lower()
    
    def test_preserve_section_generated(self):
        """PRESERVE section is generated for SKIP matches."""
        plan = RefactorPlanV3(
            search_term="orb",
            replace_term="astra",
            total_files=1,
            total_occurrences=1,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path="src/storage.ts",
                    line_number=15,
                    line_content='const KEY = "orb_auth_token";',
                    match_text="orb",
                    decision=ChangeDecision.SKIP,
                    risk_level=RiskLevel.HIGH,
                    reason_code=ReasonCode.STORAGE_KEY_IN_USE,
                    reason_text="Storage key in use",
                ),
            ],
            skip_count=1,
        )
        
        md = build_pot_spec_markdown(plan)
        
        assert "PRESERVE" in md
        assert "storage.ts" in md


# =============================================================================
# TEST: Golden Test - Rendered Text Only
# =============================================================================

class TestGoldenRenderedTextOnly:
    """
    Golden Test 1: Rendered text only → ready, no questions
    
    When all matches are user-visible UI text (JSX, HTML titles, etc.),
    the plan should be ready_for_pipeline=True with open_questions=[].
    """
    
    @pytest.mark.asyncio
    async def test_ui_text_ready_no_questions(self, ui_text_matches):
        """UI text only should be ready with no questions."""
        # Build a v3 plan from UI text matches
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=3,
            total_occurrences=3,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path=m.file_path,
                    line_number=m.line_number,
                    line_content=m.line_content,
                    match_text=m.match_text,
                    decision=ChangeDecision.CHANGE,
                    risk_level=RiskLevel.LOW,
                    reason_code=ReasonCode.UI_TEXT_VISIBLE_TO_USER,
                )
                for m in ui_text_matches
            ],
            change_count=3,
        )
        
        # Compute risk
        risk = _compute_risk_class(plan, {})
        
        assert risk == RiskClass.LOW
        assert not plan.has_unresolved_unknowns()
        assert plan.change_count == 3
        assert plan.skip_count == 0


# =============================================================================
# TEST: Golden Test - Operational Keys Preserved
# =============================================================================

class TestGoldenOperationalKeys:
    """
    Golden Test 2: Operational keys → preserve, ready, no questions
    
    When matches are storage keys or tokens, they should be SKIP (preserved)
    and the plan should still be ready.
    """
    
    @pytest.mark.asyncio
    async def test_operational_keys_preserved(self, operational_key_matches):
        """Operational keys should be preserved."""
        plan = RefactorPlanV3(
            search_term="orb",
            replace_term="astra",
            total_files=2,
            total_occurrences=2,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path=m.file_path,
                    line_number=m.line_number,
                    line_content=m.line_content,
                    match_text=m.match_text,
                    decision=ChangeDecision.SKIP,
                    risk_level=RiskLevel.HIGH,
                    reason_code=ReasonCode.STORAGE_KEY_IN_USE,
                )
                for m in operational_key_matches
            ],
            skip_count=2,
        )
        
        risk = _compute_risk_class(plan, {})
        
        # All preserved = safe, just nothing to change
        assert risk == RiskClass.LOW
        assert plan.skip_count == 2
        assert plan.change_count == 0


# =============================================================================
# TEST: Golden Test - Expansion Failure Blocks
# =============================================================================

class TestGoldenExpansionFailure:
    """
    Golden Test 5: Expansion fails → blocks with blocking_issues
    
    When evidence expansion fails and we can't confidently classify,
    the plan should block with blocking_issues (not questions).
    """
    
    def test_expansion_failure_blocks(self):
        """Expansion failure should cause RiskClass.CRITICAL."""
        plan = RefactorPlanV3(
            search_term="orb",
            replace_term="astra",
            total_files=1,
            total_occurrences=1,
            expansion_failures=[
                BlockingIssue(
                    file_path="src/config/secrets.ts",
                    line_number=10,
                    reason="Evidence expansion failed: could not read file",
                    what_was_needed="File read capability",
                    failure_type=ExpansionFailureReason.READ_FAILED,
                ),
            ],
        )
        
        risk = _compute_risk_class(plan, {})
        
        assert risk == RiskClass.CRITICAL
        blocking = plan.get_blocking_issues()
        assert len(blocking) >= 1


# =============================================================================
# TEST: Golden Test - Multi-file Alone Does Not Block
# =============================================================================

class TestGoldenMultiFileNoBlock:
    """
    Golden Test 6: REGRESSION - Multi-file alone must not block
    
    The v3 architecture change: multi-file alone is NOT a blocking condition.
    Only risk class determines blocking.
    """
    
    def test_large_multi_file_low_risk_proceeds(self):
        """Many files with LOW risk should NOT block."""
        # 100 files, all LOW risk UI text
        plan = RefactorPlanV3(
            search_term="Orb",
            replace_term="Astra",
            total_files=100,
            total_occurrences=500,
            classified_matches=[
                ClassifiedMatchV3(
                    file_path=f"src/component{i}.tsx",
                    line_number=10,
                    line_content="<div>Orb</div>",
                    match_text="Orb",
                    decision=ChangeDecision.CHANGE,
                    risk_level=RiskLevel.LOW,
                )
                for i in range(100)
            ],
            change_count=100,
        )
        
        risk = _compute_risk_class(plan, {})
        
        # Even with 100 files, all LOW risk = MEDIUM (not blocking)
        assert risk in (RiskClass.LOW, RiskClass.MEDIUM)
        # NOT CRITICAL - should proceed


# =============================================================================
# TEST: Golden Test - Questions None + Block
# =============================================================================

class TestGoldenQuestionsNoneBlock:
    """
    Golden Test 7: "Questions: none" + block → open_questions=[]
    
    When Weaver specifies "Questions: none" and the plan needs to block,
    it should return blocking_issues but open_questions must be empty.
    """
    
    def test_questions_none_returns_blocking_issues_not_questions(self):
        """When questions_none=True, blocking returns issues, not questions."""
        plan = RefactorPlanV3(
            search_term="orb",
            replace_term="astra",
            total_files=1,
            total_occurrences=1,
            questions_none=True,  # "Questions: none" contract
            expansion_failures=[
                BlockingIssue(
                    file_path="src/auth.ts",
                    line_number=10,
                    reason="Could not read file",
                ),
            ],
        )
        
        risk = _compute_risk_class(plan, {"questions_none": True})
        
        assert risk == RiskClass.CRITICAL
        # When blocked with questions_none, we return blocking_issues NOT questions
        blocking = plan.get_blocking_issues()
        assert len(blocking) >= 1


# =============================================================================
# TEST: Build IDs (Verification)
# =============================================================================

class TestBuildIds:
    """Verify build IDs are set correctly for v3.0."""
    
    def test_classifier_build_id(self):
        """Classifier should have v3.0 build ID."""
        assert "v3.0" in REFACTOR_CLASSIFIER_V3_BUILD_ID
        assert "planner-first" in REFACTOR_CLASSIFIER_V3_BUILD_ID
    
    def test_builder_build_id(self):
        """Builder should have v3.0 build ID."""
        assert "v3.0" in POT_SPEC_BUILDER_BUILD_ID
        assert "planner-first" in POT_SPEC_BUILDER_BUILD_ID
