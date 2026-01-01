# FILE: tests/test_job3_routing_controls.py
"""
Tests for Job 3: Routing Controls (Standalone)
"""

import sys
from pathlib import Path

_job3_dir = Path(__file__).parent
if str(_job3_dir) not in sys.path:
    sys.path.insert(0, str(_job3_dir))

import pytest
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# INLINE OUTPUT VALIDATOR (for standalone testing)
# =============================================================================

class ViolationType(str, Enum):
    CODE_FENCE = "code_fence"
    SHELL_PROMPT = "shell_prompt"
    PYTHON_CODE = "python_code"
    DIFF_PATCH = "diff_patch"
    COMMAND_SEQUENCE = "command_sequence"
    JAVASCRIPT_CODE = "javascript_code"
    SQL_CODE = "sql_code"
    FILE_CONTENT = "file_content"
    INLINE_CODE = "inline_code"


@dataclass
class Violation:
    violation_type: ViolationType
    pattern_matched: str
    line_number: Optional[int] = None
    excerpt: str = ""
    
    def to_dict(self) -> dict:
        return {
            "type": self.violation_type.value,
            "pattern": self.pattern_matched,
            "line": self.line_number,
            "excerpt": self.excerpt[:100] if self.excerpt else "",
        }


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[Violation] = field(default_factory=list)
    reprompt_message: Optional[str] = None


# Patterns
SHELL_PROMPT_PATTERNS = [
    (r"^\s*PS\s*[A-Za-z]?:?[>\s]", "PowerShell prompt"),
    (r"^\s*\$\s+\w", "Unix shell prompt"),
    (r"^\s*C:\\[>\s]", "Windows cmd prompt"),
]

PYTHON_CODE_PATTERNS = [
    (r"^\s{4,}def\s+\w+\s*\(", "Python function definition"),
    (r"^\s{4,}class\s+\w+\s*[\(:]", "Python class definition"),
    (r"^\s{4,}import\s+\w+", "Python import (indented)"),
    (r"^\s{4,}from\s+\w+\s+import", "Python from-import (indented)"),
]

DIFF_PATTERNS = [
    (r"^\s*[-+]{3}\s+[ab]/", "Unified diff header"),
    (r"^\s*@@\s*-\d+", "Diff hunk header"),
    (r"^\s*\+\s*def\s+\w+", "Diff adding function"),
    (r"^\s*diff\s+--git", "Git diff header"),
    (r"^\s*index\s+[a-f0-9]+\.\.[a-f0-9]+", "Git index line"),
]

COMMAND_SEQUENCE_PATTERNS = [
    (r"(?i)step\s*\d+[:.]\s*(run|execute|type|enter)", "Step-by-step commands"),
]

FILE_CONTENT_PATTERNS = [
    (r"^#!\s*/usr/bin/", "Shebang line"),
    (r"^#!\s*/bin/", "Shebang line"),
]


def _check_patterns(text: str, patterns: List[Tuple[str, str]], violation_type: ViolationType) -> List[Violation]:
    violations = []
    lines = text.split("\n")
    for line_num, line in enumerate(lines, 1):
        for pattern, description in patterns:
            if re.search(pattern, line, re.MULTILINE):
                violations.append(Violation(
                    violation_type=violation_type,
                    pattern_matched=description,
                    line_number=line_num,
                    excerpt=line.strip()[:80],
                ))
                break
    return violations


def _check_code_fences(text: str) -> List[Violation]:
    violations = []
    fence_pattern = r"```(\w*)\n([\s\S]*?)```"
    matches = re.finditer(fence_pattern, text)
    for match in matches:
        lang = match.group(1) or "unknown"
        line_num = text[:match.start()].count("\n") + 1
        violations.append(Violation(
            violation_type=ViolationType.CODE_FENCE,
            pattern_matched=f"Code fence ({lang})",
            line_number=line_num,
            excerpt=f"```{lang}...",
        ))
    return violations


def validate_overwatcher_output(text: str) -> ValidationResult:
    if not text:
        return ValidationResult(is_valid=True)
    
    all_violations: List[Violation] = []
    all_violations.extend(_check_code_fences(text))
    all_violations.extend(_check_patterns(text, SHELL_PROMPT_PATTERNS, ViolationType.SHELL_PROMPT))
    all_violations.extend(_check_patterns(text, PYTHON_CODE_PATTERNS, ViolationType.PYTHON_CODE))
    all_violations.extend(_check_patterns(text, DIFF_PATTERNS, ViolationType.DIFF_PATCH))
    all_violations.extend(_check_patterns(text, COMMAND_SEQUENCE_PATTERNS, ViolationType.COMMAND_SEQUENCE))
    all_violations.extend(_check_patterns(text, FILE_CONTENT_PATTERNS, ViolationType.FILE_CONTENT))
    
    if all_violations:
        violation_summary = ", ".join(set(v.pattern_matched for v in all_violations[:5]))
        reprompt = f"CONTRACT VIOLATION: Detected: {violation_summary}. Re-emit without code."
        return ValidationResult(is_valid=False, violations=all_violations, reprompt_message=reprompt)
    
    return ValidationResult(is_valid=True)


def build_reprompt_messages(original_messages, invalid_response, validation_result):
    messages = list(original_messages)
    messages.append({"role": "assistant", "content": invalid_response})
    violation_types = set(v.violation_type.value for v in validation_result.violations[:5])
    messages.append({"role": "user", "content": f"Contract violated: {', '.join(violation_types)}. Re-emit."})
    return messages


# =============================================================================
# INLINE COST GUARD (for standalone testing)
# =============================================================================

class ModelRole(str, Enum):
    OVERWATCHER = "overwatcher"
    SPEC_GATE = "spec_gate"
    IMPLEMENTER = "implementer"
    ARCHITECT = "architect"
    CRITIQUE = "critique"
    MEDIATOR = "mediator"


class BudgetStatus(str, Enum):
    WITHIN_BUDGET = "within_budget"
    APPROACHING_LIMIT = "approaching_limit"
    EXCEEDED = "exceeded"
    BREAK_GLASS_REQUIRED = "break_glass_required"


@dataclass
class TokenBudget:
    role: ModelRole
    max_output_tokens: int
    max_input_tokens: int
    warning_threshold: float = 0.8
    break_glass_multiplier: float = 2.0
    
    def get_effective_limit(self, break_glass: bool = False) -> int:
        if break_glass:
            return int(self.max_output_tokens * self.break_glass_multiplier)
        return self.max_output_tokens


DEFAULT_BUDGETS = {
    ModelRole.OVERWATCHER: TokenBudget(ModelRole.OVERWATCHER, 2000, 120000),
    ModelRole.SPEC_GATE: TokenBudget(ModelRole.SPEC_GATE, 4000, 100000),
    ModelRole.IMPLEMENTER: TokenBudget(ModelRole.IMPLEMENTER, 16000, 100000),
    ModelRole.ARCHITECT: TokenBudget(ModelRole.ARCHITECT, 32000, 100000),
    ModelRole.CRITIQUE: TokenBudget(ModelRole.CRITIQUE, 8000, 100000),
    ModelRole.MEDIATOR: TokenBudget(ModelRole.MEDIATOR, 4000, 50000),
}


@dataclass
class UsageRecord:
    job_id: str
    role: ModelRole
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate: float
    break_glass_used: bool = False
    stage: Optional[str] = None


@dataclass
class JobBudgetSummary:
    job_id: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    calls_by_role: Dict[str, int] = field(default_factory=dict)
    tokens_by_role: Dict[str, int] = field(default_factory=dict)
    break_glass_count: int = 0
    budget_warnings: List[str] = field(default_factory=list)
    
    def add_usage(self, record: UsageRecord):
        self.total_prompt_tokens += record.prompt_tokens
        self.total_completion_tokens += record.completion_tokens
        self.total_tokens += record.total_tokens
        self.total_cost += record.cost_estimate
        role_key = record.role.value
        self.calls_by_role[role_key] = self.calls_by_role.get(role_key, 0) + 1
        self.tokens_by_role[role_key] = self.tokens_by_role.get(role_key, 0) + record.total_tokens
        if record.break_glass_used:
            self.break_glass_count += 1


@dataclass
class BudgetCheckResult:
    status: BudgetStatus
    allowed_tokens: int
    requested_tokens: int
    message: str
    break_glass_available: bool = True
    
    def is_allowed(self) -> bool:
        return self.status in (BudgetStatus.WITHIN_BUDGET, BudgetStatus.APPROACHING_LIMIT)


@dataclass
class BreakGlassRequest:
    job_id: str
    role: ModelRole
    reason: str
    requested_tokens: int
    normal_limit: int
    break_glass_limit: int
    approved: bool = False
    approved_by: Optional[str] = None
    
    def approve(self, approver: str = "system"):
        self.approved = True
        self.approved_by = approver


class CostGuard:
    def __init__(self):
        self._budgets = DEFAULT_BUDGETS.copy()
        self._job_summaries: Dict[str, JobBudgetSummary] = {}
        self._usage_records: List[UsageRecord] = []
        self._break_glass_requests: List[BreakGlassRequest] = []
    
    def get_budget(self, role: ModelRole) -> TokenBudget:
        return self._budgets.get(role, DEFAULT_BUDGETS[ModelRole.IMPLEMENTER])
    
    def check_budget(self, role: ModelRole, requested_output_tokens: int, break_glass: bool = False) -> BudgetCheckResult:
        budget = self.get_budget(role)
        normal_limit = budget.max_output_tokens
        break_glass_limit = budget.get_effective_limit(break_glass=True)
        
        if requested_output_tokens <= normal_limit:
            warning_threshold = int(normal_limit * budget.warning_threshold)
            if requested_output_tokens >= warning_threshold:
                return BudgetCheckResult(BudgetStatus.APPROACHING_LIMIT, normal_limit, requested_output_tokens, "Approaching limit")
            return BudgetCheckResult(BudgetStatus.WITHIN_BUDGET, normal_limit, requested_output_tokens, "Within budget")
        
        # Exceeds normal budget
        if break_glass and requested_output_tokens <= break_glass_limit:
            return BudgetCheckResult(BudgetStatus.WITHIN_BUDGET, break_glass_limit, requested_output_tokens, "Within break-glass")
        
        if not break_glass and requested_output_tokens <= break_glass_limit:
            return BudgetCheckResult(BudgetStatus.BREAK_GLASS_REQUIRED, normal_limit, requested_output_tokens, "Break-glass required", True)
        
        return BudgetCheckResult(BudgetStatus.EXCEEDED, break_glass_limit if break_glass else normal_limit, requested_output_tokens, "Exceeded", False)
    
    def get_max_tokens_for_role(self, role: ModelRole, break_glass: bool = False) -> int:
        return self.get_budget(role).get_effective_limit(break_glass)
    
    def record_usage(self, job_id: str, role: ModelRole, provider: str, model: str, prompt_tokens: int, completion_tokens: int, cost_estimate: float = 0.0, break_glass_used: bool = False, stage: Optional[str] = None) -> UsageRecord:
        record = UsageRecord(job_id, role, provider, model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, cost_estimate, break_glass_used, stage)
        self._usage_records.append(record)
        
        if job_id not in self._job_summaries:
            self._job_summaries[job_id] = JobBudgetSummary(job_id)
        
        summary = self._job_summaries[job_id]
        summary.add_usage(record)
        
        budget = self.get_budget(role)
        if completion_tokens > budget.max_output_tokens:
            summary.budget_warnings.append(f"{role.value} exceeded: {completion_tokens} > {budget.max_output_tokens}")
        
        return record
    
    def get_job_summary(self, job_id: str) -> Optional[JobBudgetSummary]:
        return self._job_summaries.get(job_id)
    
    def create_break_glass_request(self, job_id: str, role: ModelRole, reason: str, requested_tokens: int) -> BreakGlassRequest:
        budget = self.get_budget(role)
        request = BreakGlassRequest(job_id, role, reason, requested_tokens, budget.max_output_tokens, budget.get_effective_limit(True))
        self._break_glass_requests.append(request)
        return request
    
    def approve_break_glass(self, request: BreakGlassRequest, approver: str = "auto") -> bool:
        if request.requested_tokens > request.break_glass_limit:
            return False
        request.approve(approver)
        return True


# Convenience functions
def check_budget(role: ModelRole, requested_tokens: int, break_glass: bool = False) -> BudgetCheckResult:
    return CostGuard().check_budget(role, requested_tokens, break_glass)


def get_max_tokens(role: ModelRole, break_glass: bool = False) -> int:
    return DEFAULT_BUDGETS[role].get_effective_limit(break_glass)


def record_usage(job_id: str, role: ModelRole, provider: str, model: str, prompt_tokens: int, completion_tokens: int, **kwargs) -> UsageRecord:
    cg = CostGuard()
    return cg.record_usage(job_id, role, provider, model, prompt_tokens, completion_tokens, **kwargs)


# =============================================================================
# OUTPUT VALIDATOR TESTS
# =============================================================================

class TestOutputValidator:
    """Test output validation for Overwatcher contract compliance."""
    
    # --- Clean outputs (should pass) ---
    
    def test_clean_diagnosis_passes(self):
        text = """
        The error is caused by a missing import in the router module.
        The function references a variable that was never defined.
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is True
        assert len(result.violations) == 0
    
    def test_clean_fix_actions_pass(self):
        text = """
        1. Add the missing import statement to the router module
        2. Initialize the configuration variable before use
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is True
    
    def test_json_schema_output_passes(self):
        text = '''
        {
          "decision": "FAIL",
          "diagnosis": "Missing import in router module",
          "fix_actions": [
            {
              "order": 1,
              "target_file": "app/router.py",
              "action_type": "add_import",
              "description": "Add the missing import for the configuration module"
            }
          ]
        }
        '''
        result = validate_overwatcher_output(text)
        assert result.is_valid is True
    
    def test_empty_text_passes(self):
        result = validate_overwatcher_output("")
        assert result.is_valid is True
    
    def test_inline_backticks_for_filenames_pass(self):
        text = "Update the file `app/router.py` to add the import."
        result = validate_overwatcher_output(text)
        assert result.is_valid is True
    
    # --- Violations (should fail) ---
    
    def test_code_fence_detected(self):
        text = '''
        Here's the fix:
        ```python
        def fix_router():
            import config
            return config.get_value()
        ```
        '''
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
        assert any(v.violation_type == ViolationType.CODE_FENCE for v in result.violations)
    
    def test_shell_prompt_detected(self):
        text = """
        Run these commands:
        PS> cd D:\\Orb
        PS> python -m pytest tests/
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
        assert any(v.violation_type == ViolationType.SHELL_PROMPT for v in result.violations)
    
    def test_unix_shell_prompt_detected(self):
        text = """
        Execute:
        $ pip install package
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
        assert any(v.violation_type == ViolationType.SHELL_PROMPT for v in result.violations)
    
    def test_python_function_in_text_detected(self):
        text = """
        Add this function:
            def process_request(data):
                result = validate(data)
                return result
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
        assert any(v.violation_type == ViolationType.PYTHON_CODE for v in result.violations)
    
    def test_diff_detected(self):
        text = """
        Apply this patch:
        --- a/app/router.py
        +++ b/app/router.py
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
        assert any(v.violation_type == ViolationType.DIFF_PATCH for v in result.violations)
    
    def test_git_diff_header_detected(self):
        text = """
        diff --git a/app/main.py b/app/main.py
        index abc123..def456 100644
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
    
    def test_step_by_step_commands_detected(self):
        text = """
        Step 1: Run the migration script
        Step 2: Execute the database update
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
        assert any(v.violation_type == ViolationType.COMMAND_SEQUENCE for v in result.violations)
    
    def test_shebang_detected(self):
        text = """
        #!/usr/bin/env python
        import sys
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
    
    def test_reprompt_message_generated(self):
        text = "```python\ncode here\n```"
        result = validate_overwatcher_output(text)
        assert result.is_valid is False
        assert result.reprompt_message is not None
        assert "CONTRACT VIOLATION" in result.reprompt_message


class TestOutputValidatorEdgeCases:
    
    def test_mentions_code_without_including_it(self):
        text = """
        The function definition is missing a return statement.
        Add a class method to handle the edge case.
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is True
    
    def test_file_path_references_pass(self):
        text = """
        Modify app/router.py to add the import.
        Update tests/test_router.py with new test cases.
        """
        result = validate_overwatcher_output(text)
        assert result.is_valid is True


# =============================================================================
# COST GUARD TESTS
# =============================================================================

class TestCostGuard:
    
    def test_within_budget_passes(self):
        cg = CostGuard()
        result = cg.check_budget(ModelRole.OVERWATCHER, 1500)
        assert result.status == BudgetStatus.WITHIN_BUDGET
        assert result.is_allowed()
    
    def test_approaching_limit_warning(self):
        cg = CostGuard()
        result = cg.check_budget(ModelRole.OVERWATCHER, 1800)  # 90% of 2000
        assert result.status == BudgetStatus.APPROACHING_LIMIT
        assert result.is_allowed()
    
    def test_exceeds_budget_requires_break_glass(self):
        cg = CostGuard()
        result = cg.check_budget(ModelRole.OVERWATCHER, 3000, break_glass=False)
        assert result.status == BudgetStatus.BREAK_GLASS_REQUIRED
        assert not result.is_allowed()
        assert result.break_glass_available
    
    def test_break_glass_allows_higher_limit(self):
        cg = CostGuard()
        result = cg.check_budget(ModelRole.OVERWATCHER, 3000, break_glass=True)
        assert result.status == BudgetStatus.WITHIN_BUDGET
        assert result.is_allowed()
    
    def test_exceeds_even_break_glass(self):
        cg = CostGuard()
        result = cg.check_budget(ModelRole.OVERWATCHER, 5000, break_glass=True)
        assert result.status == BudgetStatus.EXCEEDED
        assert not result.is_allowed()
    
    def test_implementer_has_higher_budget(self):
        cg = CostGuard()
        result = cg.check_budget(ModelRole.IMPLEMENTER, 10000)
        assert result.status == BudgetStatus.WITHIN_BUDGET
    
    def test_architect_has_highest_budget(self):
        cg = CostGuard()
        result = cg.check_budget(ModelRole.ARCHITECT, 25000)
        assert result.status == BudgetStatus.WITHIN_BUDGET
    
    def test_record_usage(self):
        cg = CostGuard()
        record = cg.record_usage(
            job_id="job-123",
            role=ModelRole.OVERWATCHER,
            provider="openai",
            model="gpt-5.2-pro",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_estimate=0.05,
        )
        assert record.job_id == "job-123"
        assert record.total_tokens == 1500
        
        summary = cg.get_job_summary("job-123")
        assert summary is not None
        assert summary.total_tokens == 1500
    
    def test_multiple_usages_aggregate(self):
        cg = CostGuard()
        cg.record_usage("job-456", ModelRole.OVERWATCHER, "openai", "gpt-5.2-pro", 1000, 500)
        cg.record_usage("job-456", ModelRole.IMPLEMENTER, "anthropic", "claude-sonnet", 2000, 1500)
        
        summary = cg.get_job_summary("job-456")
        assert summary.total_tokens == 5000
        assert summary.calls_by_role["overwatcher"] == 1
        assert summary.calls_by_role["implementer"] == 1
    
    def test_break_glass_request_creation(self):
        cg = CostGuard()
        request = cg.create_break_glass_request(
            job_id="job-789",
            role=ModelRole.OVERWATCHER,
            reason="Complex analysis",
            requested_tokens=3500,
        )
        assert request.job_id == "job-789"
        assert request.normal_limit == 2000
        assert request.break_glass_limit == 4000
        assert not request.approved
    
    def test_break_glass_approval(self):
        cg = CostGuard()
        request = cg.create_break_glass_request("job-bg", ModelRole.OVERWATCHER, "Required", 3000)
        approved = cg.approve_break_glass(request, "user")
        assert approved is True
        assert request.approved is True
    
    def test_break_glass_denied_if_too_high(self):
        cg = CostGuard()
        request = cg.create_break_glass_request("job-deny", ModelRole.OVERWATCHER, "Extreme", 10000)
        approved = cg.approve_break_glass(request)
        assert approved is False
        assert request.approved is False


class TestCostGuardBudgetWarnings:
    
    def test_exceeding_budget_generates_warning(self):
        cg = CostGuard()
        cg.record_usage("job-warn", ModelRole.OVERWATCHER, "openai", "gpt-5.2-pro", 1000, 3000)
        summary = cg.get_job_summary("job-warn")
        assert len(summary.budget_warnings) > 0
        assert "exceeded" in summary.budget_warnings[0].lower()


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    
    def test_check_budget_function(self):
        result = check_budget(ModelRole.IMPLEMENTER, 10000)
        assert result.is_allowed()
    
    def test_get_max_tokens_function(self):
        max_tokens = get_max_tokens(ModelRole.OVERWATCHER)
        assert max_tokens == 2000
        
        max_tokens_bg = get_max_tokens(ModelRole.OVERWATCHER, break_glass=True)
        assert max_tokens_bg == 4000
    
    def test_record_usage_function(self):
        record = record_usage(
            job_id="job-conv",
            role=ModelRole.CRITIQUE,
            provider="google",
            model="gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=800,
        )
        assert record.total_tokens == 1300


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    
    def test_validated_output_with_budget_check(self):
        cg = CostGuard()
        
        # Check budget
        budget_result = cg.check_budget(ModelRole.OVERWATCHER, 1500)
        assert budget_result.is_allowed()
        
        # Validate clean output
        clean_output = '{"decision": "FAIL", "diagnosis": "Missing config"}'
        validation = validate_overwatcher_output(clean_output)
        assert validation.is_valid
        
        # Record usage
        cg.record_usage("job-int", ModelRole.OVERWATCHER, "openai", "gpt-5.2-pro", 1000, len(clean_output) // 4)
        summary = cg.get_job_summary("job-int")
        assert summary.total_tokens > 0
    
    def test_rejected_output_triggers_reprompt(self):
        invalid_output = "```python\ndef solve(): return 42\n```"
        validation = validate_overwatcher_output(invalid_output)
        assert not validation.is_valid
        
        original_messages = [
            {"role": "system", "content": "You are Overwatcher"},
            {"role": "user", "content": "Analyze this error"},
        ]
        
        reprompt_messages = build_reprompt_messages(original_messages, invalid_output, validation)
        assert len(reprompt_messages) == 4
        assert reprompt_messages[-1]["role"] == "user"
        assert "violated" in reprompt_messages[-1]["content"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
