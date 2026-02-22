from __future__ import annotations
import os
from dataclasses import dataclass
_STAGE_MODELS_AVAILABLE = True
get_overwatcher_config = None
get_stage_config = None


def _get_overwatcher_config():
    """
    Get Overwatcher configuration from centralized stage_models.
    
    v3.1: NO HARDCODED DEFAULTS. All config comes from env vars via stage_models.
    
    Returns:
        StageConfig with provider, model, max_output_tokens, timeout_seconds
    
    Raises:
        RuntimeError if stage_models unavailable
    """
    if not _STAGE_MODELS_AVAILABLE or get_overwatcher_config is None:
        raise RuntimeError(
            "FATAL: stage_models not available. Ensure app/llm/stage_models.py exists "
            "and OVERWATCHER_PROVIDER + OVERWATCHER_MODEL env vars are set."
        )
    return get_overwatcher_config()

def _get_fallback_config():
    """
    Get fallback configuration for Overwatcher.
    
    Uses OVERWATCHER_FALLBACK stage config if available.
    Returns None if no fallback configured.
    """
    if not _STAGE_MODELS_AVAILABLE or get_stage_config is None:
        return None
    try:
        # Only return fallback if explicitly configured in env
        fallback_provider = os.getenv("OVERWATCHER_FALLBACK_PROVIDER", "").strip()
        fallback_model = os.getenv("OVERWATCHER_FALLBACK_MODEL", "").strip()
        if fallback_provider and fallback_model:
            return get_stage_config("OVERWATCHER_FALLBACK")
        return None
    except Exception:
        return None

OVERWATCHER_MAX_INPUT_TOKENS = 120_000

@dataclass
class FixAction:
    """A single fix action (no code allowed)."""
    
    order: int
    target_file: str
    action_type: str  # "add_function" | "modify_function" | "fix_import" | etc.
    description: str  # What to do (high-level, no code)
    rationale: str  # Why this fixes the issue
    
    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "target_file": self.target_file,
            "action_type": self.action_type,
            "description": self.description,
            "rationale": self.rationale,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FixAction":
        return cls(
            order=data.get("order", 0),
            target_file=data.get("target_file", ""),
            action_type=data.get("action_type", ""),
            description=data.get("description", ""),
            rationale=data.get("rationale", ""),
        )

@dataclass
class VerificationStep:
    """A verification step with expected outcome."""
    
    command: str
    expected_outcome: str
    timeout_seconds: int = 60
    
    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "expected_outcome": self.expected_outcome,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VerificationStep":
        return cls(
            command=data.get("command", ""),
            expected_outcome=data.get("expected_outcome", ""),
            timeout_seconds=data.get("timeout_seconds", 60),
        )

CODE_PATTERNS = [
    r"```\w*\n",  # Code fences
    r"^\s{4,}def\s+\w+",  # Python function definition
    r"^\s{4,}class\s+\w+",  # Python class definition
    r"^\s{4,}import\s+",  # Import statement (indented = in code block)
    r"^\s{4,}from\s+\w+\s+import",  # From import (indented)
    r"^\+\s*def\s+",  # Diff with function
    r"^\+\s*class\s+",  # Diff with class
]

OVERWATCHER_SYSTEM = """You are an expert software engineering supervisor (Overwatcher).

YOUR ROLE:
- Diagnose failures and define fix actions
- Enforce constraints and spec compliance
- Decide PASS, FAIL, or NEEDS_INFO

CRITICAL RULES - YOU MUST FOLLOW:
1. NEVER write code, patches, diffs, or file contents
2. NEVER include code blocks in your response
3. Only output JSON with decision, diagnosis, and fix actions
4. Fix actions describe WHAT to do, not HOW (no code)

You must respond with ONLY a valid JSON object matching this schema:
{{
  "decision": "PASS" | "FAIL" | "NEEDS_INFO",
  "diagnosis": "Root cause hypothesis (1-2 sentences)",
  "fix_actions": [
    {{
      "order": 1,
      "target_file": "path/to/file.py",
      "action_type": "add_function|modify_function|fix_import|add_error_handling|etc",
      "description": "High-level description of what to change (NO CODE)",
      "rationale": "Why this fixes the issue"
    }}
  ],
  "constraints": ["List of invariants to respect"],
  "verification": [
    {{
      "command": "pytest tests/test_foo.py -v",
      "expected_outcome": "all tests pass",
      "timeout_seconds": 60
    }}
  ],
  "blockers": ["Issues that must be fixed"],
  "nonblockers": ["Issues that can be deferred"],
  "confidence": 0.0-1.0,
  "needs_deep_research": true|false
}}

SPEC HASH LOCK: {spec_hash}
You must preserve this spec hash. Do not suggest changes that would alter the spec."""

OVERWATCHER_USER = """Analyze this evidence bundle and provide your decision.

{evidence_text}

Remember:
- Output ONLY JSON
- NO code in fix_actions descriptions
- decision must be PASS, FAIL, or NEEDS_INFO
- Strike {strike_number}/3 - {strike_hint}"""


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "build_overwatcher_prompt": "_overwatcher_utils_3",
    "contains_code": "_overwatcher_utils_3",
    "run_pot_spec_execution": "_overwatcher_utils_3",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.overwatcher.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
