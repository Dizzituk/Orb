# FILE: app/translation/gates.py
"""
Safety gates for ASTRA Translation Layer.
- Directive vs Story Gate: Blocks past tense, questions, future planning
- Context Gate: Ensures required context is present
- Confirmation Gate: Requires explicit Yes for high-stakes operations
- Overwatcher Gate: Ensures validated spec + pipeline completion (v1.2)
"""
from __future__ import annotations
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from .schemas import (
    GateResult,
    DirectiveGateResult,
    ContextGateResult,
    ConfirmationGateResult,
    CanonicalIntent,
)
from .intents import get_intent_definition

logger = logging.getLogger(__name__)


# =============================================================================
# DIRECTIVE VS STORY GATE (VERY IMPORTANT)
# =============================================================================
# Inside Command-Capable mode, ONLY true imperatives may become commands.
# Block the following from ever triggering commands:
# - past tense: "that time you mapped..."
# - future planning: "next week we'll map..."
# - questions: "how do you map...?"
# - talking about commands: "when we run start your zombie..."

# Patterns that indicate NON-DIRECTIVE speech
NON_DIRECTIVE_PATTERNS = {
    # Past tense indicators
    "past_tense": [
        r"\bthat time\b",
        r"\bwhen you\b.*\bed\b",          # "when you mapped", "when you started"
        r"\byou\s+\w+ed\b",                # "you mapped", "you created"
        r"\bi\s+\w+ed\b",                  # "I asked", "I ran"
        r"\bwe\s+\w+ed\b",                 # "we mapped", "we discussed"
        r"\blast\s+(?:time|week|month)\b",
        r"\byesterday\b",
        r"\bpreviously\b",
        r"\bearlier\b",
        r"\bbefore\b",
        r"\bremember when\b",
        r"\brecall when\b",
    ],
    
    # Future planning indicators (not commands)
    "future_planning": [
        r"\bnext\s+(?:time|week|month)\b",
        r"\bwe(?:'ll| will)\b",           # "we'll map", "we will run"
        r"\bi(?:'ll| will)\b",            # "I'll do", "I will start"
        r"\bgoing to\b",
        r"\bplan(?:ning)? to\b",
        r"\bshould we\b",
        r"\bcould we\b",
        r"\bwould be nice to\b",
        r"\bmaybe\s+(?:we|i)\b",
        r"\beventually\b",
        r"\blater\b",
        r"\bsomeday\b",
    ],
    
    # Question indicators
    "question": [
        r"\?$",                            # Ends with question mark
        r"^(?:how|what|when|where|why|who|which|can|could|would|should|is|are|do|does|did)\b",
        r"\bhow do(?:es)?\b",
        r"\bwhat (?:is|are|does|do)\b",
        r"\bcan you\b",
        r"\bcould you\b",
        r"\bwould you\b",
        r"\btell me about\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bwhat happens when\b",
    ],
    
    # Talking ABOUT commands (meta-discussion)
    "meta_discussion": [
        r"\bwhen (?:we|you|i) (?:run|start|create|update)\b",
        r"\bif (?:we|you|i) (?:run|start|create|update)\b",
        r"\babout the\b.*\bcommand\b",
        r"\babout\b.*\bpipeline\b",
        r"\bhow does the\b",
        r"\bwhat does\b.*\bdo\b",
        r"\bthe\b.*\bsystem\b",
        r"\byour\b.*\bsubsystem\b",        # "your Overwatch subsystem"
        r"\bthe\b.*\bsubsystem\b",
        r"\btell me\b",                    # "tell me about"
        r"\bshow me\b",                    # "show me the"
    ],
    
    # Hypothetical/conditional
    "hypothetical": [
        r"\bif\s+(?:we|you|i)\b",
        r"\bwhat if\b",
        r"\bsuppose\b",
        r"\bimagine\b",
        r"\bhypothetically\b",
        r"\bin theory\b",
    ],
    
    # Storytelling
    "storytelling": [
        r"\bonce upon\b",
        r"\bthere was\b",
        r"\blong ago\b",
        r"\bback when\b",
    ],
}

# Compiled patterns for efficiency
_COMPILED_NON_DIRECTIVE = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in NON_DIRECTIVE_PATTERNS.items()
}


def check_directive_gate(text: str) -> DirectiveGateResult:
    """
    Check if text is a true directive (imperative command) vs story/question/planning.
    
    Returns:
        DirectiveGateResult with:
        - passed=True if this looks like a genuine imperative command
        - passed=False if this looks like chat (question, past tense, planning, etc.)
    """
    text_lower = text.lower().strip()
    
    # Check each category of non-directive patterns
    for category, patterns in _COMPILED_NON_DIRECTIVE.items():
        for pattern in patterns:
            if pattern.search(text_lower):
                return DirectiveGateResult(
                    passed=False,
                    gate_name="directive_vs_story",
                    reason=f"Detected {category} pattern - not a command",
                    blocked_by=category,
                    detected_pattern=category,
                    original_text_snippet=text[:100],
                )
    
    # If no non-directive patterns found, it passes
    return DirectiveGateResult(
        passed=True,
        gate_name="directive_vs_story",
        reason="No non-directive patterns detected",
    )


def is_obvious_chat(text: str) -> Tuple[bool, Optional[str]]:
    """
    Quick check for messages that are OBVIOUSLY chat.
    Used for Tier 0 short-circuit to avoid classifier entirely.
    
    Returns:
        (is_chat, reason)
    """
    # Check directive gate
    result = check_directive_gate(text)
    if not result.passed:
        return True, result.detected_pattern
    
    # Additional quick checks
    text_lower = text.lower().strip()
    
    # Very short messages without command keywords are chat
    if len(text_lower) < 10:
        command_keywords = ["create", "start", "run", "update", "execute", "launch"]
        if not any(kw in text_lower for kw in command_keywords):
            return True, "short_non_command"
    
    # Messages starting with certain words are chat
    chat_starters = [
        "i think", "i'm", "i am", "it's", "it is", "that's", "that is",
        "well", "so", "hmm", "huh", "ok", "okay", "sure", "yeah", "yes",
        "no", "nope", "thanks", "thank you", "please", "hey", "hi", "hello",
        "good", "great", "nice", "cool", "interesting", "actually", "basically",
    ]
    for starter in chat_starters:
        if text_lower.startswith(starter):
            return True, "chat_starter"
    
    return False, None


# =============================================================================
# CONTEXT GATE
# =============================================================================

def check_context_gate(
    intent: CanonicalIntent,
    provided_context: Dict[str, Any],
) -> ContextGateResult:
    """
    Check if required context is present for the given intent.
    
    Args:
        intent: The resolved canonical intent
        provided_context: Context provided (from UI, previous messages, etc.)
        
    Returns:
        ContextGateResult indicating if context requirements are met
    """
    defn = get_intent_definition(intent)
    required = defn.requires_context
    
    if not required:
        return ContextGateResult(
            passed=True,
            gate_name="context",
            reason="No context required for this intent",
            provided_context=provided_context,
        )
    
    missing = []
    for key in required:
        if key not in provided_context or provided_context[key] is None:
            missing.append(key)
    
    if missing:
        return ContextGateResult(
            passed=False,
            gate_name="context",
            reason=f"Missing required context: {', '.join(missing)}",
            missing_context=missing,
            provided_context=provided_context,
        )
    
    return ContextGateResult(
        passed=True,
        gate_name="context",
        reason="All required context present",
        missing_context=[],
        provided_context=provided_context,
    )


def extract_context_from_text(
    text: str,
    intent: CanonicalIntent,
) -> Dict[str, Any]:
    """
    Attempt to extract context from the message text itself.
    E.g., "Run critical pipeline for job abc123" -> {"job_id": "abc123"}
    
    This is a best-effort extraction. Missing context will still
    require clarification.
    """
    context = {}
    text_lower = text.lower()
    
    # Extract job_id
    job_patterns = [
        r"for job\s+([a-zA-Z0-9\-_]+)",
        r"job[:\s]+([a-zA-Z0-9\-_]+)",
        r"job_id[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in job_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["job_id"] = match.group(1)
            break
    
    # Extract sandbox_id
    sandbox_patterns = [
        r"for sandbox\s+([a-zA-Z0-9\-_]+)",
        r"sandbox[:\s]+([a-zA-Z0-9\-_]+)",
        r"sandbox_id[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in sandbox_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["sandbox_id"] = match.group(1)
            break
    
    # Extract change_set_id (optional - Overwatcher can derive if missing)
    changeset_patterns = [
        r"change(?:_?set)?[:\s]+([a-zA-Z0-9\-_]+)",
        r"changes[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in changeset_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["change_set_id"] = match.group(1)
            break
    
    return context


# =============================================================================
# OVERWATCHER GATE (v1.2)
# =============================================================================
# Overwatcher-specific gating that checks:
# - REQUIRED: validated spec exists (spec_id + spec_hash resolvable)
# - REQUIRED: Critical Pipeline completed for that spec
# - NOT REQUIRED: change_set_id (Overwatcher derives internally)
# - NOT REQUIRED: zero blocking issues (Overwatcher evaluates these)
# =============================================================================

def check_overwatcher_gate(
    project_id: int,
    db_session: Any = None,
) -> ContextGateResult:
    """
    Check Overwatcher-specific requirements.
    
    Unlike the generic context gate, this performs actual DB lookups to verify:
    1. A validated spec exists for this project
    2. Critical Pipeline has completed for that spec
    
    Args:
        project_id: The project ID to check
        db_session: SQLAlchemy session (optional - will create if needed)
        
    Returns:
        ContextGateResult indicating if Overwatcher can run
    """
    missing = []
    provided = {"project_id": project_id}
    
    # Check project_id
    if not project_id:
        return ContextGateResult(
            passed=False,
            gate_name="overwatcher",
            reason="Missing required context: project_id",
            missing_context=["project_id"],
            provided_context={},
        )
    
    # Try to resolve validated spec from DB
    spec_info = _resolve_validated_spec(project_id, db_session)
    
    if spec_info is None:
        return ContextGateResult(
            passed=False,
            gate_name="overwatcher",
            reason="No validated spec found for this project. Run Spec Gate first.",
            missing_context=["validated_spec"],
            provided_context=provided,
        )
    
    provided["spec_id"] = spec_info.get("spec_id")
    provided["spec_hash"] = spec_info.get("spec_hash")
    
    # Check if Critical Pipeline completed for this spec
    pipeline_completed = _check_critical_pipeline_completed(
        project_id, 
        spec_info.get("spec_id"),
        db_session
    )
    
    if not pipeline_completed:
        return ContextGateResult(
            passed=False,
            gate_name="overwatcher",
            reason=f"Critical Pipeline has not completed for spec {spec_info.get('spec_id')}. Run 'Astra, command: run critical pipeline' first.",
            missing_context=["critical_pipeline_completion"],
            provided_context=provided,
        )
    
    provided["pipeline_completed"] = True
    
    # All checks passed
    logger.info(f"[overwatcher_gate] PASSED: project={project_id}, spec={spec_info.get('spec_id')}")
    
    return ContextGateResult(
        passed=True,
        gate_name="overwatcher",
        reason="Overwatcher gate passed: validated spec and completed pipeline found",
        missing_context=[],
        provided_context=provided,
    )


def _resolve_validated_spec(
    project_id: int,
    db_session: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve the latest validated spec for a project.
    
    Returns:
        Dict with spec_id, spec_hash, spec_version if found
        None if no validated spec exists
    """
    try:
        # Try to import spec service
        from app.spec_gate.spec_persistence import get_latest_validated_spec
        
        if db_session is None:
            # Create session if not provided
            from app.database import SessionLocal
            db_session = SessionLocal()
            close_session = True
        else:
            close_session = False
        
        try:
            spec = get_latest_validated_spec(db_session, project_id)
            if spec:
                return {
                    "spec_id": spec.spec_id,
                    "spec_hash": spec.spec_hash,
                    "spec_version": spec.version,
                    "status": spec.status,
                }
            return None
        finally:
            if close_session:
                db_session.close()
                
    except ImportError as e:
        logger.warning(f"[overwatcher_gate] Could not import spec_persistence: {e}")
        # Fallback: try to find spec in jobs directory
        return _resolve_spec_from_jobs(project_id)
    except Exception as e:
        logger.error(f"[overwatcher_gate] Error resolving spec: {e}")
        return None


def _resolve_spec_from_jobs(project_id: int) -> Optional[Dict[str, Any]]:
    """
    Fallback: resolve spec from jobs directory if DB unavailable.
    Looks for the most recent spec_v*.json in the project's job folders.
    """
    import os
    import json
    from pathlib import Path
    
    jobs_root = Path(os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs/jobs"))
    
    if not jobs_root.exists():
        return None
    
    # Find spec files
    latest_spec = None
    latest_time = 0
    
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        
        spec_dir = job_dir / "spec"
        if not spec_dir.exists():
            continue
        
        for spec_file in spec_dir.glob("spec_v*.json"):
            try:
                mtime = spec_file.stat().st_mtime
                if mtime > latest_time:
                    with open(spec_file, "r", encoding="utf-8") as f:
                        spec_data = json.load(f)
                    
                    # Check if validated
                    if spec_data.get("status") == "validated":
                        latest_spec = {
                            "spec_id": spec_data.get("spec_id"),
                            "spec_hash": spec_data.get("spec_hash"),
                            "spec_version": spec_data.get("version", 1),
                            "status": "validated",
                        }
                        latest_time = mtime
            except Exception as e:
                logger.debug(f"[overwatcher_gate] Could not read {spec_file}: {e}")
                continue
    
    return latest_spec


def _check_critical_pipeline_completed(
    project_id: int,
    spec_id: str,
    db_session: Any = None,
) -> bool:
    """
    Check if Critical Pipeline has completed for the given spec.
    
    Returns:
        True if pipeline completed, False otherwise
    """
    try:
        # Try DB lookup first
        from app.jobs.service import get_completed_pipeline_for_spec
        
        if db_session is None:
            from app.database import SessionLocal
            db_session = SessionLocal()
            close_session = True
        else:
            close_session = False
        
        try:
            pipeline = get_completed_pipeline_for_spec(db_session, spec_id)
            return pipeline is not None
        finally:
            if close_session:
                db_session.close()
                
    except ImportError:
        logger.debug("[overwatcher_gate] get_completed_pipeline_for_spec not available, using fallback")
        # Fallback: check jobs directory for architecture artifacts
        return _check_pipeline_from_jobs(spec_id)
    except Exception as e:
        logger.error(f"[overwatcher_gate] Error checking pipeline: {e}")
        # Fallback to jobs directory check
        return _check_pipeline_from_jobs(spec_id)


def _check_pipeline_from_jobs(spec_id: str) -> bool:
    """
    Fallback: check if pipeline completed by looking for architecture artifacts.
    """
    import os
    import json
    from pathlib import Path
    
    jobs_root = Path(os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs/jobs"))
    
    if not jobs_root.exists():
        return False
    
    # Look for cp-* (critical pipeline) job directories
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        if not job_dir.name.startswith("cp-"):
            continue
        
        # Check for architecture artifact
        arch_dir = job_dir / "arch"
        if not arch_dir.exists():
            continue
        
        # Check if any arch file references our spec_id
        for arch_file in arch_dir.glob("arch_v*.md"):
            try:
                content = arch_file.read_text(encoding="utf-8", errors="ignore")
                if spec_id in content:
                    logger.debug(f"[overwatcher_gate] Found pipeline completion in {arch_file}")
                    return True
            except Exception:
                continue
        
        # Also check arch_v*.json
        for arch_file in arch_dir.glob("arch_v*.json"):
            try:
                with open(arch_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("spec_id") == spec_id:
                    logger.debug(f"[overwatcher_gate] Found pipeline completion in {arch_file}")
                    return True
            except Exception:
                continue
    
    return False


# =============================================================================
# CONFIRMATION GATE (High-Stakes)
# =============================================================================

class ConfirmationState:
    """
    Tracks pending confirmations for high-stakes operations.
    This would typically be stored in session/conversation state.
    """
    
    def __init__(self):
        self._pending: Dict[str, Dict[str, Any]] = {}
    
    def request_confirmation(
        self,
        confirmation_id: str,
        intent: CanonicalIntent,
        context: Dict[str, Any],
    ) -> str:
        """
        Request confirmation for a high-stakes operation.
        Returns the confirmation prompt to show the user.
        """
        defn = get_intent_definition(intent)
        prompt = defn.confirmation_prompt or (
            f"⚠️ HIGH-STAKES OPERATION\n"
            f"You are about to execute: {intent.value}\n"
            f"Type 'Yes' to confirm."
        )
        
        # Format with context
        try:
            prompt = prompt.format(**context)
        except KeyError:
            pass  # Keep unformatted if context missing
        
        self._pending[confirmation_id] = {
            "intent": intent,
            "context": context,
            "prompt": prompt,
        }
        
        return prompt
    
    def check_confirmation(
        self,
        confirmation_id: str,
        user_response: str,
    ) -> Tuple[bool, Optional[CanonicalIntent], Optional[Dict[str, Any]]]:
        """
        Check if user response confirms the pending operation.
        
        Returns:
            (confirmed, intent, context) if confirmed
            (False, None, None) if not confirmed or no pending
        """
        if confirmation_id not in self._pending:
            return False, None, None
        
        pending = self._pending[confirmation_id]
        
        # Check for explicit "Yes" confirmation
        response = user_response.strip().lower()
        if response in ("yes", "y", "confirm", "confirmed"):
            # Remove from pending and return confirmed
            del self._pending[confirmation_id]
            return True, pending["intent"], pending["context"]
        
        # Not confirmed - remove pending
        del self._pending[confirmation_id]
        return False, None, None
    
    def clear_pending(self, confirmation_id: str) -> None:
        """Clear a pending confirmation."""
        self._pending.pop(confirmation_id, None)
    
    def has_pending(self, confirmation_id: str) -> bool:
        """Check if there's a pending confirmation."""
        return confirmation_id in self._pending


def check_confirmation_gate(
    intent: CanonicalIntent,
    context: Dict[str, Any],
    confirmation_state: Optional[ConfirmationState] = None,
    confirmation_id: Optional[str] = None,
) -> ConfirmationGateResult:
    """
    Check if high-stakes confirmation is required/provided.
    
    Args:
        intent: The resolved intent
        context: Provided context
        confirmation_state: State tracker for pending confirmations
        confirmation_id: ID for this confirmation (e.g., conversation_id)
        
    Returns:
        ConfirmationGateResult indicating confirmation status
    """
    defn = get_intent_definition(intent)
    
    if not defn.requires_confirmation:
        return ConfirmationGateResult(
            passed=True,
            gate_name="confirmation",
            reason="No confirmation required for this intent",
            requires_confirmation=False,
        )
    
    # Requires confirmation
    prompt = defn.confirmation_prompt or f"Confirm execution of {intent.value}?"
    try:
        prompt = prompt.format(**context)
    except KeyError:
        pass
    
    return ConfirmationGateResult(
        passed=False,
        gate_name="confirmation",
        reason="High-stakes operation requires explicit confirmation",
        requires_confirmation=True,
        confirmation_prompt=prompt,
        awaiting_confirmation=True,
    )
