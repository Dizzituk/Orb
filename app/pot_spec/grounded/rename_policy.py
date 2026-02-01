# FILE: app/pot_spec/grounded/rename_policy.py
"""
Rename Policy Engine (v1.0)

Implements the "features > data" policy for multi-file refactors.
Classifies matches into SAFE / UNSAFE / MIGRATION-REQUIRED based on
whether they break core invariants.

Core Invariants (must not break):
1. Backend boots cleanly
2. Desktop boots (electron)
3. Auth still works (session/token creation + validation)
4. Encryption still works (master key load + DB decrypt/encrypt)
5. Job pipeline runs end-to-end (Weaver→SpecGate→Critical→Implementer)
6. Sandbox safety rules still block forbidden paths
7. Tool routing still works (local tools / stream endpoints)

Not Core (allowed to break/wipe):
- stored memory DB contents
- old job artifacts
- backups, logs
- historical reports
- cached embeddings/indexes

Token Categories:
A) Branding/UI text → SAFE to rename
B) Code identifiers → SAFE if fully refactored (must compile)
C) Protocol/compatibility tokens → UNSAFE unless migration added

Version: v1.0 (2026-02-01)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Core Invariants Definition
# =============================================================================

class Invariant(str, Enum):
    """Core system invariants that must not break."""
    BACKEND_BOOT = "backend_boot"
    DESKTOP_BOOT = "desktop_boot"
    AUTH_WORKS = "auth_works"
    ENCRYPTION_WORKS = "encryption_works"
    JOB_PIPELINE = "job_pipeline"
    SANDBOX_SAFETY = "sandbox_safety"
    TOOL_ROUTING = "tool_routing"


class RenameDecision(str, Enum):
    """Decision for each match/category."""
    SAFE = "safe"                    # Can rename without breaking invariants
    UNSAFE = "unsafe"                # Would break invariant - DO NOT RENAME
    MIGRATION_REQUIRED = "migration" # Safe only with compatibility layer


# =============================================================================
# Protocol Token Patterns (HIGH RISK - breaks invariants if renamed)
# =============================================================================

# These are exact patterns that are protocol/compatibility tokens
# Renaming them breaks auth, encryption, or persistence
PROTOCOL_TOKENS: Dict[str, Tuple[Invariant, str]] = {
    # Auth tokens
    r'orb_session_': (Invariant.AUTH_WORKS, "Session token prefix used in validation"),
    r'orb_api_': (Invariant.AUTH_WORKS, "API key prefix"),
    r'orb_': (Invariant.AUTH_WORKS, "Generic token prefix (if at start of token)"),
    
    # Encryption
    r'ORB_MASTER_KEY': (Invariant.ENCRYPTION_WORKS, "Master encryption key env var"),
    r'ORB_ENCRYPTION': (Invariant.ENCRYPTION_WORKS, "Encryption config env var"),
    
    # Database paths (persisted)
    r'orb_memory': (Invariant.BACKEND_BOOT, "Database filename - persisted on disk"),
    r'\.db': (Invariant.BACKEND_BOOT, "Database file extension context"),
}

# Env var patterns that are protocol tokens
PROTOCOL_ENV_VARS: Set[str] = {
    'ORB_MASTER_KEY',
    'ORB_SESSION_SECRET',
    'ORB_API_KEY',
    'ORB_ENCRYPTION_KEY',
    'ORB_DB_PATH',
    'ORB_JOB_ARTIFACT_ROOT',
}

# Files that contain protocol-critical code
INVARIANT_FILES: Dict[str, Invariant] = {
    'auth/middleware.py': Invariant.AUTH_WORKS,
    'auth/config.py': Invariant.AUTH_WORKS,
    'auth/session.py': Invariant.AUTH_WORKS,
    'encryption/': Invariant.ENCRYPTION_WORKS,
    'crypto': Invariant.ENCRYPTION_WORKS,
    'master_key': Invariant.ENCRYPTION_WORKS,
    'sandbox/manager.py': Invariant.SANDBOX_SAFETY,
    'sandbox/tools.py': Invariant.SANDBOX_SAFETY,
    'overwatcher/': Invariant.JOB_PIPELINE,
    'translation/': Invariant.TOOL_ROUTING,
    'stream_router': Invariant.TOOL_ROUTING,
}


# =============================================================================
# Safe Categories (can rename without breaking invariants)
# =============================================================================

# File patterns that are safe to rename (won't break invariants)
SAFE_FILE_PATTERNS: List[str] = [
    # Documentation
    r'\.md$',
    r'README',
    r'docs/',
    r'\.txt$',
    
    # Tests (won't break prod)
    r'tests/',
    r'test_',
    r'_test\.py$',
    
    # Static/UI
    r'static/',
    r'\.html$',
    r'\.css$',
    
    # Old artifacts (allowed to break)
    r'jobs/jobs/',  # Old job outputs
    r'backup',
    r'/logs/',
    r'\.log$',
    r'leak_scan',
    r'_backup',
]

# Content patterns that are safe to rename
SAFE_CONTENT_PATTERNS: List[str] = [
    # Branding/UI text
    r'You are Orb',
    r"Orb's engineering brain",
    r'Starting Orb',
    r'ZOMBIE ORB',
    r'Orb Desktop',
    
    # Comments/docs
    r'^#.*Orb',
    r'^//.*Orb',
    r'""".*Orb',
    r"'''.*Orb",
    
    # CSS classes (UI only)
    r'\.orb\s*\{',
    r'class.*orb',
    r'className.*orb',
]


# =============================================================================
# Rename Decision Engine
# =============================================================================

@dataclass
class MatchDecision:
    """Decision for a single match."""
    file_path: str
    line_number: int
    line_content: str
    decision: RenameDecision
    reason: str
    invariant: Optional[Invariant] = None
    migration_needed: Optional[str] = None


@dataclass
class RenamePlan:
    """
    Complete rename plan with decisions.
    
    This is what SpecGate outputs to Implementer.
    """
    search_pattern: str
    replace_pattern: str
    
    # Categorized matches
    safe_to_rename: List[MatchDecision] = field(default_factory=list)
    unsafe_excluded: List[MatchDecision] = field(default_factory=list)
    migration_required: List[MatchDecision] = field(default_factory=list)
    
    # Summary counts
    total_matches: int = 0
    safe_count: int = 0
    unsafe_count: int = 0
    migration_count: int = 0
    
    # Invariant checks for Implementer to run
    required_checks: List[str] = field(default_factory=list)
    
    def add_match(self, decision: MatchDecision):
        """Add a match decision to the appropriate category."""
        self.total_matches += 1
        
        if decision.decision == RenameDecision.SAFE:
            self.safe_to_rename.append(decision)
            self.safe_count += 1
        elif decision.decision == RenameDecision.UNSAFE:
            self.unsafe_excluded.append(decision)
            self.unsafe_count += 1
        elif decision.decision == RenameDecision.MIGRATION_REQUIRED:
            self.migration_required.append(decision)
            self.migration_count += 1
    
    def get_report(self) -> str:
        """Generate the rename plan report for SpecGate output."""
        lines = [
            "# Rename Plan",
            "",
            f"**Pattern:** `{self.search_pattern}` → `{self.replace_pattern}`",
            "",
            "## Summary",
            f"- **Total Matches:** {self.total_matches}",
            f"- **✅ Safe to Rename:** {self.safe_count}",
            f"- **❌ Excluded (breaks invariants):** {self.unsafe_count}",
            f"- **⚠️ Requires Migration:** {self.migration_count}",
            "",
        ]
        
        # Excluded items (most important - these are NOT being renamed)
        if self.unsafe_excluded:
            lines.append("## ❌ EXCLUDED - Will Break Invariants")
            lines.append("")
            lines.append("These items are **NOT being renamed** because they would break core features:")
            lines.append("")
            
            # Group by invariant
            by_invariant: Dict[Invariant, List[MatchDecision]] = {}
            for m in self.unsafe_excluded:
                inv = m.invariant or Invariant.BACKEND_BOOT
                if inv not in by_invariant:
                    by_invariant[inv] = []
                by_invariant[inv].append(m)
            
            for invariant, matches in by_invariant.items():
                lines.append(f"### Breaks: {invariant.value}")
                for m in matches[:5]:  # Show first 5
                    lines.append(f"- `{m.file_path}` L{m.line_number}")
                    lines.append(f"  - Content: `{m.line_content[:60]}...`")
                    lines.append(f"  - Reason: {m.reason}")
                if len(matches) > 5:
                    lines.append(f"- ... and {len(matches) - 5} more")
                lines.append("")
        
        # Migration required items
        if self.migration_required:
            lines.append("## ⚠️ MIGRATION REQUIRED")
            lines.append("")
            lines.append("These items can be renamed but require a compatibility layer:")
            lines.append("")
            
            for m in self.migration_required[:10]:
                lines.append(f"- `{m.file_path}` L{m.line_number}")
                lines.append(f"  - Migration: {m.migration_needed or 'TBD'}")
            if len(self.migration_required) > 10:
                lines.append(f"- ... and {len(self.migration_required) - 10} more")
            lines.append("")
        
        # Safe items summary (just counts by category)
        if self.safe_to_rename:
            lines.append("## ✅ SAFE TO RENAME")
            lines.append("")
            
            # Group by file type
            by_type: Dict[str, int] = {}
            for m in self.safe_to_rename:
                if 'test' in m.file_path.lower():
                    key = "Tests"
                elif '.md' in m.file_path.lower() or 'readme' in m.file_path.lower():
                    key = "Documentation"
                elif '.tsx' in m.file_path.lower() or '.jsx' in m.file_path.lower():
                    key = "UI Components"
                elif 'jobs/jobs/' in m.file_path.lower():
                    key = "Old Job Artifacts"
                else:
                    key = "Code"
                by_type[key] = by_type.get(key, 0) + 1
            
            for category, count in sorted(by_type.items(), key=lambda x: -x[1]):
                lines.append(f"- {category}: {count} matches")
            lines.append("")
        
        # Required invariant checks
        lines.append("## Required Invariant Checks (Implementer Must Run)")
        lines.append("")
        for check in self.required_checks:
            lines.append(f"- [ ] {check}")
        lines.append("")
        
        return "\n".join(lines)


def classify_match(
    file_path: str,
    line_number: int,
    line_content: str,
    search_pattern: str,
) -> MatchDecision:
    """
    Classify a single match as SAFE / UNSAFE / MIGRATION_REQUIRED.
    
    This is the core decision logic.
    """
    path_lower = file_path.lower()
    content_lower = line_content.lower()
    
    # ==========================================================================
    # CHECK 1: Is this in a safe file? (tests, docs, old artifacts)
    # ==========================================================================
    
    for pattern in SAFE_FILE_PATTERNS:
        if re.search(pattern, path_lower, re.IGNORECASE):
            return MatchDecision(
                file_path=file_path,
                line_number=line_number,
                line_content=line_content,
                decision=RenameDecision.SAFE,
                reason=f"Safe file pattern: {pattern}",
            )
    
    # ==========================================================================
    # CHECK 2: Is this a .env file? (MIGRATION_REQUIRED for definitions)
    # Must check BEFORE protocol tokens to catch env var definitions
    # ==========================================================================
    
    if '.env' in path_lower:
        # Check if it's an env var definition (KEY=value format)
        if re.match(r'^[A-Z][A-Z0-9_]*\s*=', line_content):
            return MatchDecision(
                file_path=file_path,
                line_number=line_number,
                line_content=line_content,
                decision=RenameDecision.MIGRATION_REQUIRED,
                reason="Env var definition in .env - must update file AND all usages atomically",
                invariant=Invariant.BACKEND_BOOT,
                migration_needed="Rename env var in .env AND update all os.getenv() calls in same commit",
            )
    
    # ==========================================================================
    # CHECK 3: Is this a protocol token? (UNSAFE)
    # ==========================================================================
    
    for token_pattern, (invariant, reason) in PROTOCOL_TOKENS.items():
        if re.search(token_pattern, line_content, re.IGNORECASE):
            # Check if this is actually a protocol usage, not just a mention
            # Protocol usage: token creation, validation, comparison
            if any(x in content_lower for x in ['startswith', '==', 'token', 'prefix', 'session', 'key', 'getenv', 'environ']):
                return MatchDecision(
                    file_path=file_path,
                    line_number=line_number,
                    line_content=line_content,
                    decision=RenameDecision.UNSAFE,
                    reason=reason,
                    invariant=invariant,
                )
    
    # ==========================================================================
    # CHECK 4: Is this a protocol env var usage? (UNSAFE)
    # ==========================================================================
    
    for env_var in PROTOCOL_ENV_VARS:
        if env_var in line_content:
            # If it's a getenv call, it's a usage - UNSAFE
            if 'getenv' in content_lower or 'environ' in content_lower:
                return MatchDecision(
                    file_path=file_path,
                    line_number=line_number,
                    line_content=line_content,
                    decision=RenameDecision.UNSAFE,
                    reason=f"Protocol env var {env_var} - renaming breaks config",
                    invariant=Invariant.BACKEND_BOOT,
                )
    
    # ==========================================================================
    # CHECK 5: Is this in an invariant-critical file?
    # ==========================================================================
    
    for file_pattern, invariant in INVARIANT_FILES.items():
        if file_pattern in path_lower:
            # Check if it's branding (safe) or protocol (unsafe)
            if any(re.search(p, line_content, re.IGNORECASE) for p in SAFE_CONTENT_PATTERNS):
                return MatchDecision(
                    file_path=file_path,
                    line_number=line_number,
                    line_content=line_content,
                    decision=RenameDecision.SAFE,
                    reason="Branding/UI text in critical file (safe to rename)",
                )
            
            # If it's not clearly branding, be cautious
            if any(x in content_lower for x in ['token', 'session', 'key', 'secret', 'encrypt', 'decrypt']):
                return MatchDecision(
                    file_path=file_path,
                    line_number=line_number,
                    line_content=line_content,
                    decision=RenameDecision.UNSAFE,
                    reason=f"Protocol-sensitive code in {file_pattern}",
                    invariant=invariant,
                )
    
    # ==========================================================================
    # CHECK 6: Is this branding/UI text? (SAFE)
    # ==========================================================================
    
    for pattern in SAFE_CONTENT_PATTERNS:
        if re.search(pattern, line_content, re.IGNORECASE):
            return MatchDecision(
                file_path=file_path,
                line_number=line_number,
                line_content=line_content,
                decision=RenameDecision.SAFE,
                reason="Branding/UI text",
            )
    
    # ==========================================================================
    # CHECK 7: Is this a database file reference?
    # ==========================================================================
    
    if '.db' in content_lower and search_pattern.lower() in content_lower:
        return MatchDecision(
            file_path=file_path,
            line_number=line_number,
            line_content=line_content,
            decision=RenameDecision.MIGRATION_REQUIRED,
            reason="Database filename - requires file rename + path updates",
            invariant=Invariant.BACKEND_BOOT,
            migration_needed="Rename physical .db file AND update all path references atomically",
        )
    
    # ==========================================================================
    # CHECK 8: Is this a hardcoded path with search pattern?
    # ==========================================================================
    
    if ('D:\\' in line_content or 'D:/' in line_content) and search_pattern in line_content:
        # Paths in sandbox config are UNSAFE (break sandbox safety)
        if 'sandbox' in path_lower or 'allow' in content_lower or 'deny' in content_lower:
            return MatchDecision(
                file_path=file_path,
                line_number=line_number,
                line_content=line_content,
                decision=RenameDecision.UNSAFE,
                reason="Sandbox path rule - renaming may break safety constraints",
                invariant=Invariant.SANDBOX_SAFETY,
            )
        # Other paths might just need folder rename
        return MatchDecision(
            file_path=file_path,
            line_number=line_number,
            line_content=line_content,
            decision=RenameDecision.MIGRATION_REQUIRED,
            reason="Hardcoded path - requires folder rename",
            migration_needed="Rename D:\\Orb folder AND update path references",
        )
    
    # ==========================================================================
    # DEFAULT: Code identifier - SAFE (but must compile after)
    # ==========================================================================
    
    return MatchDecision(
        file_path=file_path,
        line_number=line_number,
        line_content=line_content,
        decision=RenameDecision.SAFE,
        reason="Code identifier (must compile after rename)",
    )


def build_rename_plan(
    matches: List[Dict[str, Any]],
    search_pattern: str,
    replace_pattern: str,
) -> RenamePlan:
    """
    Build a complete rename plan from discovery matches.
    
    Args:
        matches: List of match dicts with file_path, line_number, line_content
        search_pattern: Pattern being searched (e.g., "Orb")
        replace_pattern: Replacement text (e.g., "ASTRA")
    
    Returns:
        RenamePlan with categorized matches and required checks
    """
    plan = RenamePlan(
        search_pattern=search_pattern,
        replace_pattern=replace_pattern,
    )
    
    for match in matches:
        decision = classify_match(
            file_path=match.get('file_path', match.get('path', '')),
            line_number=match.get('line_number', 0),
            line_content=match.get('line_content', match.get('content', '')),
            search_pattern=search_pattern,
        )
        plan.add_match(decision)
    
    # Add required invariant checks based on what we're touching
    touched_invariants: Set[Invariant] = set()
    
    for m in plan.safe_to_rename:
        path_lower = m.file_path.lower()
        if 'auth' in path_lower:
            touched_invariants.add(Invariant.AUTH_WORKS)
        if 'encrypt' in path_lower or 'crypto' in path_lower:
            touched_invariants.add(Invariant.ENCRYPTION_WORKS)
        if 'sandbox' in path_lower:
            touched_invariants.add(Invariant.SANDBOX_SAFETY)
        if 'job' in path_lower or 'overwatcher' in path_lower:
            touched_invariants.add(Invariant.JOB_PIPELINE)
    
    # Always check boot
    plan.required_checks = [
        "Backend boots cleanly: `python -c 'from app.main import app'`",
    ]
    
    if Invariant.AUTH_WORKS in touched_invariants:
        plan.required_checks.append(
            "Auth works: Create session token, validate it, check middleware accepts"
        )
    
    if Invariant.ENCRYPTION_WORKS in touched_invariants:
        plan.required_checks.append(
            "Encryption works: Load master key, encrypt/decrypt test data"
        )
    
    if Invariant.SANDBOX_SAFETY in touched_invariants:
        plan.required_checks.append(
            "Sandbox safety: Verify forbidden paths still blocked"
        )
    
    if Invariant.JOB_PIPELINE in touched_invariants:
        plan.required_checks.append(
            "Job pipeline: Run trivial job end-to-end (Weaver→SpecGate→Implementer)"
        )
    
    # Always check compile
    plan.required_checks.append(
        "Code compiles: `python -m py_compile app/**/*.py`"
    )
    
    return plan


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Invariant",
    "RenameDecision",
    "MatchDecision",
    "RenamePlan",
    "classify_match",
    "build_rename_plan",
    "PROTOCOL_TOKENS",
    "PROTOCOL_ENV_VARS",
    "SAFE_FILE_PATTERNS",
]
