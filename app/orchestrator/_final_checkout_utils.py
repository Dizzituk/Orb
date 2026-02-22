from .construction_planner_models import ConstructionPlan
from .strike_tracker import StrikeVerdict
from typing import Any, Callable, Optional


FINAL_CHECKOUT_BUILD_ID = "2026-02-15-v3.3-investigation-model-upgrade"

async def _fix_interface_contract_issue(
    issue: Any,
    sandbox_base: str,
    sandbox_client: Any,
    verdict: StrikeVerdict,
    emit: Optional[Callable] = None,
) -> bool:
    """
    Fix an interface contract violation.

    If a segment promised to expose a name but doesn't, check if it
    exists under a slightly different name and add a re-export.
    """
    _emit = emit or (lambda msg: None)
    # Interface contract fixes are less common and more complex.
    # For now, log and skip — the boot test will catch real breakage.
    _emit(f"    [INFO] Interface contract fix deferred to boot test")
    return False

def _find_closest_name(target: str, candidates: set) -> Optional[str]:
    """Find the closest matching name from a set of candidates."""
    if not candidates:
        return None

    target_lower = target.lower()

    # Exact match (case-insensitive)
    for c in candidates:
        if c.lower() == target_lower:
            return c

    # Substring match
    for c in candidates:
        if target_lower in c.lower() or c.lower() in target_lower:
            return c

    # Word overlap
    import re as _re
    target_words = set(_re.findall(r'[a-z]+', target_lower))
    best = None
    best_score = 0
    for c in candidates:
        c_words = set(_re.findall(r'[a-z]+', c.lower()))
        overlap = len(target_words & c_words)
        if overlap > best_score:
            best_score = overlap
            best = c

    return best if best_score > 0 else None

_REVIEW_PRIORITY_PATTERNS = [
    r"main\.py$",                    # Entry point
    r"__init__\.py$",                # Package init (import chains)
    r"auth|security|password",       # Security-sensitive
    r"api|router|endpoint|views",    # External-facing
    r"database|models|migration",    # Data integrity
    r"config|settings|env",          # Configuration
]

_MAX_REVIEW_FILES = 12       # Cap total files reviewed

_MAX_REVIEW_CHARS = 8000     # Cap per-file content in prompt

_REVIEW_SYSTEM_PROMPT = """\
You are a senior software reviewer performing a final quality gate assessment.
Review the code files against the original specification.

Focus on:
1. QUALITY: Code structure, error handling, naming consistency, dead code
2. SECURITY: Input validation, injection risks, hardcoded secrets, auth gaps
3. PERFORMANCE: Obvious bottlenecks, N+1 queries, unbounded loops, missing caching

Be concise and actionable. Only flag real issues, not style preferences.
Respond with valid JSON only.\
"""

def _build_minimal_state_from_plan(
    plan: ConstructionPlan,
    sandbox_base: str,
) -> Any:
    """
    Build a minimal state-like object from a ConstructionPlan for boot fix mapping.
    """
    class _MinimalSegState:
        def __init__(self, output_files):
            self.output_files = output_files
            self.status = "complete"

    class _MinimalState:
        def __init__(self):
            self.segments = {}

    state = _MinimalState()
    for phase in plan.phases:
        state.segments[phase.phase_id] = _MinimalSegState(phase.file_scope)

    return state
