# FILE: app/llm/critical_pipeline_stream.py
"""
Critical Pipeline streaming handler for ASTRA command flow.

v2.2 (2026-01): Quickcheck Validation for Micro Jobs
- Added MicroQuickcheckResult and micro_quickcheck() for deterministic validation
- Micro jobs now get fast tick-box checks before "Ready for Overwatcher"
- No LLM critique for micro jobs - pure deterministic validation

v2.1 (2026-01-04): Artifact Binding Support
- Extracts artifact bindings from spec for Overwatcher
- Includes content_verbatim, location, scope_constraints in architecture prompt
- Generates concrete file paths for implementation

v2.0: Real pipeline integration with Block 4-6.
"""

import json
import logging
import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict
from uuid import uuid4

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# =============================================================================
# Pipeline Imports (Block 4-6)
# =============================================================================

try:
    from app.llm.pipeline.high_stakes import (
        run_high_stakes_with_critique,
        store_architecture_artifact,
        get_environment_context,
        HIGH_STAKES_JOB_TYPES,
    )
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    _PIPELINE_AVAILABLE = False
    logger.warning(f"[critical_pipeline] Pipeline modules not available: {e}")

try:
    from app.llm.pipeline.critique_schemas import CritiqueResult
except ImportError:
    CritiqueResult = None

# =============================================================================
# Schema Imports
# =============================================================================

try:
    from app.llm.schemas import LLMTask, JobType
    from app.jobs.schemas import (
        JobEnvelope,
        JobType as Phase4JobType,
        Importance,
        DataSensitivity,
        Modality,
        JobBudget,
        OutputContract,
    )
    _SCHEMAS_AVAILABLE = True
except ImportError as e:
    _SCHEMAS_AVAILABLE = False
    logger.warning(f"[critical_pipeline] Schema imports failed: {e}")

# =============================================================================
# Spec Service Imports
# =============================================================================

try:
    from app.specs.service import get_spec, get_latest_validated_spec, get_spec_schema
    _SPECS_SERVICE_AVAILABLE = True
except ImportError:
    _SPECS_SERVICE_AVAILABLE = False
    get_spec = None
    get_latest_validated_spec = None
    get_spec_schema = None

# =============================================================================
# Evidence Collector Import (for grounding Critical Pipeline)
# =============================================================================

try:
    from app.pot_spec.evidence_collector import load_evidence, EvidenceBundle
    _EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[critical_pipeline] Evidence collector not available: {e}")
    _EVIDENCE_AVAILABLE = False
    load_evidence = None
    EvidenceBundle = None

# =============================================================================
# Memory Service Imports
# =============================================================================

try:
    from app.memory import service as memory_service, schemas as memory_schemas
except ImportError:
    memory_service = None
    memory_schemas = None

# =============================================================================
# Audit Logger Imports
# =============================================================================

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

# =============================================================================
# Stage Models (env-driven model resolution)
# =============================================================================

try:
    from app.llm.stage_models import get_critical_pipeline_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

def _get_pipeline_model_config() -> dict:
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_critical_pipeline_config()
            return {"provider": cfg.provider, "model": cfg.model}
        except Exception:
            pass
    return {
        "provider": os.getenv("CRITICAL_PIPELINE_PROVIDER", "anthropic"),
        "model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    }


# =============================================================================
# Job Type Classification (v2.2)
# =============================================================================

class JobKind:
    """Job classification for pipeline routing."""
    MICRO_EXECUTION = "micro_execution"  # Simple read/write/answer tasks
    ARCHITECTURE = "architecture"         # Design/build/refactor tasks


def _classify_job_kind(spec_data: Dict[str, Any], message: str) -> str:
    """
    Classify job as MICRO_EXECUTION or ARCHITECTURE.
    
    v2.3 FIX: Now checks spec_data.get("job_kind") FIRST.
    SpecGate's deterministic classification takes priority.
    Only falls back to keyword matching if job_kind is missing.
    
    Returns: JobKind.MICRO_EXECUTION or JobKind.ARCHITECTURE
    """
    # ========================================================================
    # v2.3 FIX: Check for pre-classified job_kind from SpecGate FIRST
    # ========================================================================
    
    spec_job_kind = spec_data.get("job_kind", "")
    spec_job_kind_confidence = spec_data.get("job_kind_confidence", 0.0)
    spec_job_kind_reason = spec_data.get("job_kind_reason", "")
    
    if spec_job_kind and spec_job_kind != "unknown":
        # SpecGate already classified this - OBEY IT
        logger.info(
            "[critical_pipeline] v2.3 USING SPEC JOB_KIND: %s (confidence=%.2f, reason='%s')",
            spec_job_kind, spec_job_kind_confidence, spec_job_kind_reason
        )
        
        if spec_job_kind == "micro_execution":
            return JobKind.MICRO_EXECUTION
        elif spec_job_kind == "repo_change":
            # repo_change is faster than architecture but not as fast as micro
            # For now, treat as architecture (needs some design work)
            return JobKind.ARCHITECTURE
        elif spec_job_kind == "architecture":
            return JobKind.ARCHITECTURE
        else:
            # Unknown or unexpected value - escalate to architecture
            logger.warning(
                "[critical_pipeline] v2.3 Unknown job_kind '%s' - escalating to architecture",
                spec_job_kind
            )
            return JobKind.ARCHITECTURE
    
    # ========================================================================
    # FALLBACK: SpecGate didn't classify (or returned "unknown")
    # Use local classification logic
    # ========================================================================
    
    logger.warning(
        "[critical_pipeline] v2.3 job_kind not set by SpecGate (found='%s') - using fallback classification",
        spec_job_kind
    )
    
    # Combine all text for analysis
    summary = spec_data.get("summary", "").lower()
    objective = spec_data.get("objective", "").lower()
    title = spec_data.get("title", "").lower()
    goal = spec_data.get("goal", "").lower()
    msg_lower = message.lower()
    
    all_text = f"{summary} {objective} {title} {goal} {msg_lower}"
    
    # ========================================================================
    # Check for sandbox/file paths resolved by SpecGate (STRONGEST signal)
    # ========================================================================
    
    # Primary fields from GroundedPOTSpec
    has_sandbox_input = bool(spec_data.get("sandbox_input_path"))
    has_sandbox_output = bool(spec_data.get("sandbox_output_path"))
    has_sandbox_reply = bool(spec_data.get("sandbox_generated_reply"))
    sandbox_discovery_used = spec_data.get("sandbox_discovery_used", False)
    
    # Also check for nested fields or alternate names
    if not has_sandbox_input:
        has_sandbox_input = bool(spec_data.get("input_file_path"))
    if not has_sandbox_output:
        has_sandbox_output = bool(spec_data.get("output_file_path") or spec_data.get("planned_output_path"))
    
    # Check constraints for file paths (SpecGate adds these)
    constraints_from_repo = spec_data.get("constraints_from_repo", [])
    for constraint in constraints_from_repo:
        if isinstance(constraint, str):
            if "planned output path" in constraint.lower() or "reply.txt" in constraint.lower():
                has_sandbox_output = True
    
    # Check what_exists for sandbox input
    what_exists = spec_data.get("what_exists", [])
    for item in what_exists:
        if isinstance(item, str) and "sandbox input" in item.lower():
            has_sandbox_input = True
    
    # ========================================================================
    # MICRO FAST PATH: If sandbox discovery resolved files, it's micro
    # ========================================================================
    
    if sandbox_discovery_used and has_sandbox_input:
        logger.info(
            "[critical_pipeline] v2.3 FALLBACK MICRO: sandbox_discovery_used=True, input=%s, output=%s",
            has_sandbox_input, has_sandbox_output
        )
        return JobKind.MICRO_EXECUTION
    
    if has_sandbox_input and has_sandbox_output and has_sandbox_reply:
        logger.info(
            "[critical_pipeline] v2.3 FALLBACK MICRO: Full sandbox resolution found (input+output+reply)"
        )
        return JobKind.MICRO_EXECUTION
    
    # ========================================================================
    # Keyword-based classification (last resort fallback)
    # ========================================================================
    
    # MICRO indicators (simple file operations)
    micro_indicators = [
        "read the", "read file", "open file", "find the file",
        "answer the question", "answer question", "reply to",
        "write reply", "write answer", "print answer",
        "summarize", "summarise", "extract", "copy",
        "find document", "find the document",
        "what does", "what is", "tell me",
        "underneath", "below", "same folder",
        "sandbox", "desktop", "test folder",
        "read-only", "reply (read-only)",
    ]
    
    # ARCHITECTURE indicators (design/build work)
    arch_indicators = [
        "design", "architect", "build system", "create system",
        "implement feature", "add feature", "new module",
        "refactor", "restructure", "redesign",
        "api endpoint", "database schema", "migration",
        "integration", "pipeline", "service",
        "authentication", "authorization",
        "full implementation", "complete implementation",
        "specgate", "spec gate", "overwatcher",  # Orb system design
    ]
    
    # Count matches
    micro_score = sum(1 for ind in micro_indicators if ind in all_text)
    arch_score = sum(1 for ind in arch_indicators if ind in all_text)
    
    # Boost micro score if we have resolved file paths
    if has_sandbox_input and has_sandbox_output:
        micro_score += 5  # Strong signal - SpecGate already resolved paths
    elif has_sandbox_input:
        micro_score += 3
    elif has_sandbox_output:
        micro_score += 2
    
    # Check step count from spec (micro jobs typically have ≤5 steps)
    steps = spec_data.get("proposed_steps", spec_data.get("steps", []))
    if isinstance(steps, list):
        if len(steps) <= 5:
            micro_score += 1
        elif len(steps) > 10:
            arch_score += 2
    
    # Log classification
    logger.info(
        "[critical_pipeline] v2.3 FALLBACK classification: micro_score=%d, arch_score=%d, "
        "sandbox_discovery=%s, has_paths=%s/%s",
        micro_score, arch_score, sandbox_discovery_used, has_sandbox_input, has_sandbox_output
    )
    
    # Decision: prefer MICRO if scores are close and paths are resolved
    if micro_score > arch_score:
        return JobKind.MICRO_EXECUTION
    elif arch_score > micro_score:
        return JobKind.ARCHITECTURE
    elif has_sandbox_input or has_sandbox_output:
        # Tie-breaker: if any paths are resolved, it's micro
        return JobKind.MICRO_EXECUTION
    else:
        # Default to architecture for safety
        return JobKind.ARCHITECTURE


# =============================================================================
# Micro Quickcheck Validation (v2.2)
# =============================================================================

@dataclass
class MicroQuickcheckResult:
    """Result of micro-execution quickcheck validation.
    
    This is a fast, deterministic validation - NO LLM calls.
    Pure tick-box checks to verify spec/plan alignment.
    """
    passed: bool
    issues: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


def micro_quickcheck(spec_data: Dict[str, Any], plan_text: str) -> MicroQuickcheckResult:
    """
    Fast deterministic validation for micro-execution jobs.
    NO LLM calls - pure tick-box checks.
    
    v2.2 Checks:
    1. sandbox_input_path exists in spec
    2. sandbox_output_path is resolved
    3. Plan paths match spec paths
    4. Plan has only safe operations (no destructive commands)
    5. If plan says "write output" but no generated_reply exists → fail
    
    Returns:
        MicroQuickcheckResult with pass/fail and any issues found
    """
    issues: List[Dict[str, str]] = []
    
    # =========================================================================
    # Check 1: Input path resolved
    # =========================================================================
    
    input_path = (
        spec_data.get("sandbox_input_path") or
        spec_data.get("input_file_path") or
        ""
    )
    
    if not input_path:
        issues.append({
            "id": "MICRO-CHECK-001",
            "description": "sandbox_input_path not resolved in spec - cannot verify input source",
            "severity": "blocking",
        })
    
    # =========================================================================
    # Check 2: Output path resolved
    # =========================================================================
    
    output_path = (
        spec_data.get("sandbox_output_path") or
        spec_data.get("output_file_path") or
        spec_data.get("planned_output_path") or
        ""
    )
    
    if not output_path:
        issues.append({
            "id": "MICRO-CHECK-002",
            "description": "sandbox_output_path not resolved in spec - cannot verify output destination",
            "severity": "blocking",
        })
    
    # =========================================================================
    # Check 3: Plan references correct paths
    # =========================================================================
    
    if input_path and input_path not in plan_text:
        # Check for path variations (forward/back slashes, case)
        input_normalized = input_path.replace("\\", "/").lower()
        plan_normalized = plan_text.replace("\\", "/").lower()
        
        if input_normalized not in plan_normalized:
            issues.append({
                "id": "MICRO-CHECK-003",
                "description": f"Plan does not reference spec input path: {input_path}",
                "severity": "blocking",
            })
    
    if output_path and output_path not in plan_text:
        # Check for path variations
        output_normalized = output_path.replace("\\", "/").lower()
        plan_normalized = plan_text.replace("\\", "/").lower()
        
        if output_normalized not in plan_normalized:
            issues.append({
                "id": "MICRO-CHECK-004",
                "description": f"Plan does not reference spec output path: {output_path}",
                "severity": "blocking",
            })
    
    # =========================================================================
    # Check 4: Unsafe operations
    # =========================================================================
    
    unsafe_patterns = [
        "rm -rf", "rmdir /s", "del /f /q",
        "format c:", "format d:",
        "DROP TABLE", "DROP DATABASE", "DELETE FROM",
        "TRUNCATE TABLE",
        ":(){:|:&};:",  # Fork bomb
        "shutdown", "reboot",
        "reg delete", "regedit",
    ]
    
    plan_lower = plan_text.lower()
    for pattern in unsafe_patterns:
        if pattern.lower() in plan_lower:
            issues.append({
                "id": "MICRO-CHECK-005",
                "description": f"Plan contains potentially unsafe operation: '{pattern}'",
                "severity": "blocking",
            })
    
    # =========================================================================
    # Check 5: Reply existence (v2.2 addition per feedback)
    # If plan says "write output" but spec has no generated_reply → problem
    # =========================================================================
    
    reply_content = spec_data.get("sandbox_generated_reply", "")
    
    # Check if plan includes write step but no reply exists
    write_keywords = ["write output", "write reply", "write file", "create output", "save reply"]
    plan_has_write = any(kw in plan_lower for kw in write_keywords)
    
    if plan_has_write and not reply_content:
        issues.append({
            "id": "MICRO-CHECK-006",
            "description": "Plan includes write step but spec has no sandbox_generated_reply - nothing to write",
            "severity": "blocking",
        })
    
    # =========================================================================
    # Build result
    # =========================================================================
    
    passed = len(issues) == 0
    
    if passed:
        summary = "✅ All quickchecks passed"
    else:
        blocking_count = sum(1 for i in issues if i.get("severity") == "blocking")
        summary = f"❌ {blocking_count} blocking issue(s) found"
    
    logger.info(
        "[micro_quickcheck] Result: passed=%s, issues=%d, input=%s, output=%s, reply=%s",
        passed, len(issues),
        bool(input_path), bool(output_path), bool(reply_content)
    )
    
    return MicroQuickcheckResult(passed=passed, issues=issues, summary=summary)


def _generate_micro_execution_plan(spec_data: Dict[str, Any], job_id: str) -> str:
    """
    Generate a minimal execution plan for MICRO jobs.
    
    No architecture design needed - just a simple step-by-step plan
    that Overwatcher can execute directly.
    
    Uses the sandbox fields populated by SpecGate.
    """
    # Get paths from SpecGate's sandbox resolution
    input_path = (
        spec_data.get("sandbox_input_path") or
        spec_data.get("input_file_path") or
        "(input path not resolved)"
    )
    output_path = (
        spec_data.get("sandbox_output_path") or
        spec_data.get("output_file_path") or
        spec_data.get("planned_output_path") or
        "(output path not resolved)"
    )
    
    # Get content from SpecGate's sandbox discovery
    input_excerpt = spec_data.get("sandbox_input_excerpt", "")
    reply_content = spec_data.get("sandbox_generated_reply", "")
    content_type = spec_data.get("sandbox_selected_type", "unknown")
    
    # Get summary/goal from spec
    summary = spec_data.get("goal", spec_data.get("summary", spec_data.get("objective", "Execute task per spec")))
    
    # Build the plan
    plan = f"""# Micro-Execution Plan

**Job ID:** {job_id}
**Type:** MICRO_EXECUTION (no architecture required)

## Task Summary
{summary}

## Resolved Paths (by SpecGate)
- **Input:** `{input_path}`
- **Output:** `{output_path}`
- **Content Type:** {content_type}

## Execution Steps

1. **Read Input File**
   - Path: `{input_path}`
   - Action: Read file contents

2. **Process Content**
   - Parse the content
   - Generate response based on file content

3. **Write Output File**
   - Path: `{output_path}`
   - Action: Write generated reply

4. **Verify**
   - Confirm output file exists
   - Validate content is correct
"""
    
    # Add input preview if available
    if input_excerpt:
        plan += f"""
## Input Preview
```
{input_excerpt[:500] if input_excerpt else '(content will be read at execution time)'}
```
"""
    
    # Add expected output if SpecGate already generated the reply
    if reply_content:
        plan += f"""
## Generated Reply (from SpecGate)
```
{reply_content}
```

**Note:** SpecGate has already generated this reply. Overwatcher just needs to write it.
"""
    else:
        plan += """
## Expected Output
(to be generated by Overwatcher based on input content)
"""
    
    plan += """
## Notes
- This is a simple file operation task
- No architectural changes required
- All paths are pre-resolved by SpecGate
- Overwatcher can execute directly

---
✅ **Ready for Overwatcher** - Say 'Astra, command: send to overwatcher' to execute.
"""
    
    return plan


# =============================================================================
# Artifact Binding (v2.1)
# =============================================================================

# Path template variables for artifact binding
PATH_VARIABLES = {
    "{JOB_ID}": lambda ctx: ctx.get("job_id", "unknown"),
    "{JOB_ROOT}": lambda ctx: os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
    "{SANDBOX_DESKTOP}": lambda ctx: "C:/Users/WDAGUtilityAccount/Desktop",
    "{REPO_ROOT}": lambda ctx: ctx.get("repo_root", "."),
}


def _resolve_path_template(template: str, context: Dict[str, Any]) -> str:
    """Resolve path template variables."""
    result = template
    for var, resolver in PATH_VARIABLES.items():
        if var in result:
            result = result.replace(var, str(resolver(context)))
    return result


def _extract_artifact_bindings(spec_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and resolve artifact bindings from spec for Overwatcher.
    
    Returns list of bindings with resolved paths:
    [
        {
            "artifact_id": "output_1",
            "action": "create",
            "path": "/resolved/path/to/file.txt",
            "content_type": "text",
            "content_verbatim": "hello",  # if specified
            "description": "Output file"
        }
    ]
    """
    bindings: List[Dict[str, Any]] = []
    
    # Get outputs from spec
    outputs = spec_data.get("outputs", [])
    if not outputs:
        # Try metadata
        metadata = spec_data.get("metadata", {}) or {}
        outputs = metadata.get("outputs", [])
    
    # Get content preservation fields
    content_verbatim = (
        spec_data.get("content_verbatim") or
        spec_data.get("context", {}).get("content_verbatim") or
        spec_data.get("metadata", {}).get("content_verbatim")
    )
    location = (
        spec_data.get("location") or
        spec_data.get("context", {}).get("location") or
        spec_data.get("metadata", {}).get("location")
    )
    
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            output = {"name": output, "path": "", "description": ""}
        
        name = output.get("name", f"output_{i+1}")
        path = output.get("path", "")
        description = output.get("description", output.get("notes", ""))
        
        # Resolve path
        if path:
            resolved_path = _resolve_path_template(path, context)
        elif location:
            # Use location from content preservation
            resolved_path = _resolve_path_template(location, context)
            if name and not resolved_path.endswith(name):
                resolved_path = os.path.join(resolved_path, name)
        else:
            # Default to job artifacts directory
            resolved_path = os.path.join(
                context.get("job_root", "jobs"),
                "jobs",
                context.get("job_id", "unknown"),
                "outputs",
                name
            )
        
        binding = {
            "artifact_id": f"output_{i+1}",
            "action": "create",
            "path": resolved_path,
            "content_type": _infer_content_type(name),
            "description": description or name,
        }
        
        # Include content_verbatim if this is the primary output
        if i == 0 and content_verbatim:
            binding["content_verbatim"] = content_verbatim
        
        bindings.append(binding)
    
    logger.info("[critical_pipeline] Extracted %d artifact bindings", len(bindings))
    return bindings


def _infer_content_type(filename: str) -> str:
    """Infer content type from filename."""
    ext = os.path.splitext(filename.lower())[1]
    type_map = {
        ".txt": "text",
        ".md": "markdown",
        ".json": "json",
        ".py": "python",
        ".js": "javascript",
        ".html": "html",
        ".css": "css",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return type_map.get(ext, "text")


def _build_artifact_binding_prompt(bindings: List[Dict[str, Any]]) -> str:
    """Build prompt section for artifact bindings."""
    if not bindings:
        return ""
    
    lines = [
        "\n## ARTIFACT BINDINGS (for Overwatcher)",
        "",
        "The following artifacts MUST be created with these EXACT paths:",
        ""
    ]
    
    for binding in bindings:
        lines.append(f"- **{binding['artifact_id']}**: `{binding['path']}`")
        lines.append(f"  - Action: {binding['action']}")
        lines.append(f"  - Type: {binding['content_type']}")
        if binding.get("content_verbatim"):
            lines.append(f"  - Content: \"{binding['content_verbatim']}\" (EXACT)")
        lines.append("")
    
    lines.append("Overwatcher will use these bindings to write files. Do NOT invent different paths.")
    
    return "\n".join(lines)


# =============================================================================
# Main Stream Handler
# =============================================================================

async def generate_critical_pipeline_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: Optional[str] = None,
):
    """
    Generate SSE stream for Critical Pipeline execution with artifact binding (v2.1).
    
    v2.2: Added micro_quickcheck validation before "Ready for Overwatcher"
    
    Flow:
    1. Load validated spec from DB
    2. Classify job type (MICRO vs ARCHITECTURE)
    3a. MICRO: Generate plan → quickcheck validation → ready for Overwatcher
    3b. ARCHITECTURE: Extract bindings → run Block 4-6 pipeline → stream result
    """
    response_parts = []
    
    model_cfg = _get_pipeline_model_config()
    pipeline_provider = model_cfg["provider"]
    pipeline_model = model_cfg["model"]
    
    try:
        yield "data: " + json.dumps({"type": "token", "content": "⚙️ **Critical Pipeline**\n\n"}) + "\n\n"
        response_parts.append("⚙️ **Critical Pipeline**\n\n")
        
        # =====================================================================
        # Validation
        # =====================================================================
        
        if not _PIPELINE_AVAILABLE:
            error_msg = (
                "❌ **Pipeline modules not available.**\n\n"
                "The high-stakes pipeline modules (app.llm.pipeline.*) failed to import.\n"
            )
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message="Pipeline modules not available")
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        if not _SCHEMAS_AVAILABLE:
            error_msg = "❌ **Schema imports failed.** Check backend logs.\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # =====================================================================
        # Step 1: Load validated spec
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "📋 **Loading validated spec...**\n"}) + "\n\n"
        response_parts.append("📋 **Loading validated spec...**\n")
        
        db_spec = None
        spec_json = None
        
        if spec_id and _SPECS_SERVICE_AVAILABLE and get_spec:
            try:
                db_spec = get_spec(db, spec_id)
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to get spec by ID: {e}")
        
        if not db_spec and _SPECS_SERVICE_AVAILABLE and get_latest_validated_spec:
            try:
                db_spec = get_latest_validated_spec(db, project_id)
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to get latest validated spec: {e}")
        
        if not db_spec:
            error_msg = (
                "❌ **No validated spec found.**\n\n"
                "Please complete Spec Gate validation first:\n"
                "1. Describe what you want to build\n"
                "2. Say `Astra, command: how does that look all together`\n"
                "3. Say `Astra, command: critical architecture` to validate\n"
                "4. Once validated, retry `run critical pipeline`\n"
            )
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        spec_id = db_spec.spec_id
        spec_hash = db_spec.spec_hash
        spec_json = db_spec.content_json
        
        # Parse spec JSON
        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else (spec_json or {})
        except Exception:
            spec_data = {}
        
        yield "data: " + json.dumps({"type": "token", "content": f"✅ Spec loaded: `{spec_id[:16]}...`\n"}) + "\n\n"
        response_parts.append(f"✅ Spec loaded: `{spec_id[:16]}...`\n")
        
        # =====================================================================
        # Step 1b: Classify Job Type (v2.2)
        # =====================================================================
        
        job_kind = _classify_job_kind(spec_data, message)
        
        yield "data: " + json.dumps({"type": "token", "content": f"🏷️ **Job Type:** `{job_kind}`\n"}) + "\n\n"
        response_parts.append(f"🏷️ **Job Type:** `{job_kind}`\n")
        
        # =====================================================================
        # MICRO_EXECUTION PATH: Skip architecture, generate minimal plan + quickcheck
        # =====================================================================
        
        if job_kind == JobKind.MICRO_EXECUTION:
            yield "data: " + json.dumps({"type": "token", "content": "\n⚡ **Fast Path:** This is a micro-execution job.\n"}) + "\n\n"
            response_parts.append("\n⚡ **Fast Path:** This is a micro-execution job.\n")
            yield "data: " + json.dumps({"type": "token", "content": "No architecture design required - generating execution plan...\n\n"}) + "\n\n"
            response_parts.append("No architecture design required - generating execution plan...\n\n")
            
            # Create job ID
            if not job_id:
                job_id = f"micro-{uuid4().hex[:8]}"
            
            # Generate minimal execution plan (no LLM call needed)
            micro_plan = _generate_micro_execution_plan(spec_data, job_id)
            
            # =================================================================
            # v2.2: Run quickcheck validation BEFORE showing plan
            # =================================================================
            
            yield "data: " + json.dumps({"type": "token", "content": "🧪 **Running Quickcheck...**\n"}) + "\n\n"
            response_parts.append("🧪 **Running Quickcheck...**\n")
            
            quickcheck_result = micro_quickcheck(spec_data, micro_plan)
            
            if quickcheck_result.passed:
                # Quickcheck PASSED - show plan and mark ready
                yield "data: " + json.dumps({"type": "token", "content": f"{quickcheck_result.summary}\n\n"}) + "\n\n"
                response_parts.append(f"{quickcheck_result.summary}\n\n")
                
                yield "data: " + json.dumps({"type": "token", "content": micro_plan}) + "\n\n"
                response_parts.append(micro_plan)
                
                # Extract artifact bindings for Overwatcher
                binding_context = {
                    "job_id": job_id,
                    "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
                    "repo_root": os.getenv("REPO_ROOT", "."),
                }
                artifact_bindings = _extract_artifact_bindings(spec_data, binding_context)
                
                # Emit completion event
                yield "data: " + json.dumps({
                    "type": "work_artifacts",
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": True,
                    "artifact_bindings": artifact_bindings,
                }) + "\n\n"
                
                # Save to memory
                full_response = "".join(response_parts)
                if memory_service and memory_schemas:
                    try:
                        memory_service.create_message(db, memory_schemas.MessageCreate(
                            project_id=project_id, role="assistant", content=full_response,
                            provider="local", model="micro-execution"
                        ))
                    except Exception as e:
                        logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
                
                if trace:
                    trace.finalize(success=True)
                
                yield "data: " + json.dumps({
                    "type": "done",
                    "provider": "local",
                    "model": "micro-execution",
                    "total_length": len(full_response),
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": True,
                    "artifact_bindings": len(artifact_bindings),
                }) + "\n\n"
                return  # Exit early - micro job complete
            
            else:
                # Quickcheck FAILED - show issues and do NOT mark ready
                yield "data: " + json.dumps({"type": "token", "content": f"{quickcheck_result.summary}\n\n"}) + "\n\n"
                response_parts.append(f"{quickcheck_result.summary}\n\n")
                
                # List the issues
                for issue in quickcheck_result.issues:
                    issue_msg = f"❌ **{issue['id']}:** {issue['description']}\n"
                    yield "data: " + json.dumps({"type": "token", "content": issue_msg}) + "\n\n"
                    response_parts.append(issue_msg)
                
                # Show the plan anyway for debugging
                yield "data: " + json.dumps({"type": "token", "content": "\n### Generated Plan (for review):\n"}) + "\n\n"
                response_parts.append("\n### Generated Plan (for review):\n")
                yield "data: " + json.dumps({"type": "token", "content": micro_plan}) + "\n\n"
                response_parts.append(micro_plan)
                
                # Show next steps
                fail_msg = """
---
⚠️ **Quickcheck Failed** - Job NOT ready for Overwatcher.

Please check:
1. Did SpecGate resolve the input/output paths correctly?
2. Is the spec complete with sandbox_input_path and sandbox_output_path?
3. If the plan needs to write output, does the spec have a sandbox_generated_reply?

You may need to re-run Spec Gate with more details about the file locations.
"""
                yield "data: " + json.dumps({"type": "token", "content": fail_msg}) + "\n\n"
                response_parts.append(fail_msg)
                
                # Save to memory (even on failure)
                full_response = "".join(response_parts)
                if memory_service and memory_schemas:
                    try:
                        memory_service.create_message(db, memory_schemas.MessageCreate(
                            project_id=project_id, role="assistant", content=full_response,
                            provider="local", model="micro-execution"
                        ))
                    except Exception as e:
                        logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
                
                if trace:
                    trace.finalize(success=False, error_message="Quickcheck failed")
                
                yield "data: " + json.dumps({
                    "type": "done",
                    "provider": "local",
                    "model": "micro-execution",
                    "total_length": len(full_response),
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": False,
                    "quickcheck_issues": len(quickcheck_result.issues),
                }) + "\n\n"
                return  # Exit - quickcheck failed
        
        # =====================================================================
        # ARCHITECTURE PATH: Full pipeline (continues below)
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "\n🏗️ **Architecture Mode:** Full design pipeline required.\n\n"}) + "\n\n"
        response_parts.append("\n🏗️ **Architecture Mode:** Full design pipeline required.\n\n")
        
        # =====================================================================
        # Step 2: Create job ID and extract artifact bindings (v2.1)
        # =====================================================================
        
        if not job_id:
            job_id = f"cp-{uuid4().hex[:8]}"
        
        # Build context for path resolution
        binding_context = {
            "job_id": job_id,
            "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
            "repo_root": os.getenv("REPO_ROOT", "."),
        }
        
        # Extract artifact bindings
        artifact_bindings = _extract_artifact_bindings(spec_data, binding_context)
        
        yield "data: " + json.dumps({"type": "token", "content": f"📁 **Job ID:** `{job_id}`\n"}) + "\n\n"
        response_parts.append(f"📁 **Job ID:** `{job_id}`\n")
        
        if artifact_bindings:
            binding_msg = f"📦 **Artifact Bindings:** {len(artifact_bindings)} output(s)\n"
            for b in artifact_bindings[:3]:  # Show first 3
                binding_msg += f"  - `{b['path']}`\n"
            if len(artifact_bindings) > 3:
                binding_msg += f"  - ... and {len(artifact_bindings) - 3} more\n"
            yield "data: " + json.dumps({"type": "token", "content": binding_msg}) + "\n\n"
            response_parts.append(binding_msg)
        
        # =====================================================================
        # Step 2b: Load evidence for grounded architecture generation
        # =====================================================================
        
        evidence_excerpt = ""
        if _EVIDENCE_AVAILABLE and load_evidence:
            try:
                evidence = load_evidence(
                    include_arch_map=True,
                    include_codebase_report=False,  # Spec already has details
                    arch_map_max_lines=200,
                )
                
                if evidence.arch_map_content:
                    evidence_excerpt = evidence.arch_map_content[:8000]  # Cap at 8k chars
                    yield "data: " + json.dumps({"type": "token", "content": "📚 **Evidence loaded:** Architecture map\n"}) + "\n\n"
                    response_parts.append("📚 **Evidence loaded:** Architecture map\n")
                    logger.info("[critical_pipeline] Loaded architecture map evidence (%d chars)", len(evidence_excerpt))
                else:
                    logger.warning("[critical_pipeline] No architecture map content available")
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to load evidence: {e}")
        
        # =====================================================================
        # Step 3: Build prompt with content preservation and bindings
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "🔧 **Building architecture prompt...**\n\n"}) + "\n\n"
        response_parts.append("🔧 **Building architecture prompt...**\n\n")
        
        # Extract content preservation fields
        content_verbatim = (
            spec_data.get("content_verbatim") or
            spec_data.get("context", {}).get("content_verbatim") or
            spec_data.get("metadata", {}).get("content_verbatim")
        )
        location = (
            spec_data.get("location") or
            spec_data.get("context", {}).get("location") or
            spec_data.get("metadata", {}).get("location")
        )
        scope_constraints = (
            spec_data.get("scope_constraints") or
            spec_data.get("context", {}).get("scope_constraints") or
            spec_data.get("metadata", {}).get("scope_constraints") or
            []
        )
        
        # Build artifact binding prompt section
        binding_prompt = _build_artifact_binding_prompt(artifact_bindings)
        
        # Build system prompt with all context
        # FIXED: Extract actual task description, not generic titles
        original_request = message
        if spec_data:
            # Priority order for finding the actual task:
            # 1. summary - often contains the real task description
            # 2. objective - if it's not generic
            # 3. First input's content/example if it looks like a task
            # 4. Fall back to message
            
            summary = spec_data.get("summary", "")
            objective = spec_data.get("objective", "")
            
            # Check if objective is generic/placeholder
            generic_objectives = [
                "job description", "weaver", "build spec", "create spec",
                "draft", "generated", "placeholder"
            ]
            objective_is_generic = any(
                g in (objective or "").lower() for g in generic_objectives
            ) or len(objective or "") < 20
            
            # Use summary if it's more descriptive
            if summary and len(summary) > len(objective or ""):
                original_request = summary
            elif objective and not objective_is_generic:
                original_request = objective
            elif summary:
                original_request = summary
            else:
                # Try to get from inputs or fall back to message
                inputs = spec_data.get("inputs", [])
                if inputs and isinstance(inputs, list) and len(inputs) > 0:
                    first_input = inputs[0]
                    if isinstance(first_input, dict):
                        input_example = first_input.get("example", "")
                        if input_example and len(input_example) > 20:
                            original_request = f"Task: {input_example}"
                        else:
                            original_request = message
                    else:
                        original_request = message
                else:
                    original_request = message
            
            logger.info(f"[critical_pipeline] Extracted objective: {original_request[:100]}...")
        
        system_prompt = f"""You are Claude Opus, generating a detailed architecture document.

SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

You are working from a validated PoT Spec. Your architecture MUST:
1. Address all MUST requirements from the spec
2. Consider all constraints
3. Be buildable by a solo developer on Windows 11
4. Include the SPEC_ID and SPEC_HASH header at the top of your output

## CONTENT PRESERVATION (CRITICAL)
"""
        
        if content_verbatim:
            system_prompt += f"""
**EXACT FILE CONTENT REQUIRED:**
The file content MUST be EXACTLY: "{content_verbatim}"
Do NOT paraphrase, summarize, or modify this content in any way.
"""
        
        if location:
            system_prompt += f"""
**EXACT LOCATION REQUIRED:**
The output MUST be written to: {location}
Use this EXACT path - do not substitute or normalize it.
"""
        
        if scope_constraints:
            system_prompt += f"""
**SCOPE CONSTRAINTS:**
{chr(10).join(f'- {c}' for c in scope_constraints)}
The implementation MUST NOT operate outside these boundaries.
"""
        
        system_prompt += binding_prompt
        
        # Add evidence excerpt if available (for grounded architecture generation)
        if evidence_excerpt:
            system_prompt += f"""\n\n## ARCHITECTURE MAP (excerpts for context)\n\nThis is an excerpt from the existing codebase architecture. Use this to understand:\n- Existing code structure and patterns\n- Available modules and services\n- Integration points\n\n```\n{evidence_excerpt[:6000]}\n```\n\nReference these patterns when designing the architecture.\n"""
        
        system_prompt += "\n\nGenerate a complete, detailed architecture document."
        
        task_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate architecture for:\n\n{original_request}\n\nSpec:\n{json.dumps(spec_data, indent=2)}"},
        ]
        
        # Build LLMTask
        task = LLMTask(
            messages=task_messages,
            job_type=JobType.ARCHITECTURE_DESIGN if hasattr(JobType, 'ARCHITECTURE_DESIGN') else list(JobType)[0],
            attachments=[],
        )
        
        # Build JobEnvelope with artifact bindings in metadata
        envelope = JobEnvelope(
            job_id=job_id,
            session_id=conversation_id or f"session-{uuid4().hex[:8]}",
            project_id=project_id,
            job_type=getattr(Phase4JobType, "APP_ARCHITECTURE", list(Phase4JobType)[0]),
            importance=Importance.CRITICAL,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=16384,
                max_cost_estimate=1.00,
                max_wall_time_seconds=600,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=task_messages,
            metadata={
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "pipeline": "critical",
                # v2.1: Include artifact bindings for Overwatcher
                "artifact_bindings": artifact_bindings,
                "content_verbatim": content_verbatim,
                "location": location,
                "scope_constraints": scope_constraints,
            },
            allow_multi_model_review=True,
            needs_tools=[],
        )
        
        # =====================================================================
        # Step 4: Run the pipeline
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n"}) + "\n\n"
        response_parts.append(f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n")
        
        yield "data: " + json.dumps({"type": "token", "content": "This may take 2-5 minutes. Stages:\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  1. 📝 Architecture generation\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  2. 🔍 Critique (real blockers only)\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  3. ✏️ Revision loop (stops early if clean)\n\n"}) + "\n\n"
        
        yield "data: " + json.dumps({
            "type": "pipeline_started",
            "stage": "critical_pipeline",
            "job_id": job_id,
            "spec_id": spec_id,
            "critique_mode": "deep",
            "artifact_bindings": len(artifact_bindings),
        }) + "\n\n"
        
        try:
            result = await run_high_stakes_with_critique(
                task=task,
                provider_id=pipeline_provider,
                model_id=pipeline_model,
                envelope=envelope,
                job_type_str="architecture_design",
                file_map=None,
                db=db,
                spec_id=spec_id,
                spec_hash=spec_hash,
                spec_json=spec_json,
                use_json_critique=True,
            )
            
        except Exception as e:
            logger.exception(f"[critical_pipeline] Pipeline failed: {e}")
            error_msg = f"❌ **Pipeline failed:** {e}\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts), "error": str(e)
            }) + "\n\n"
            return
        
        # =====================================================================
        # Step 5: Stream the result
        # =====================================================================
        
        if not result or not result.content:
            error_msg = "❌ **Pipeline returned empty result.**\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        routing_decision = getattr(result, 'routing_decision', {}) or {}
        arch_id = routing_decision.get('arch_id', 'unknown')
        final_version = routing_decision.get('final_version', 1)
        critique_passed = routing_decision.get('critique_passed', False)
        blocking_issues = routing_decision.get('blocking_issues', 0)
        
        summary_header = "✅ **Pipeline Complete**\n\n"
        yield "data: " + json.dumps({"type": "token", "content": summary_header}) + "\n\n"
        response_parts.append(summary_header)
        
        summary_details = f"""**Architecture ID:** `{arch_id}`
**Final Version:** v{final_version}
**Critique Mode:** deep (blocker filtering enabled)
**Critique Status:** {"✅ PASSED" if critique_passed else f"⚠️ {blocking_issues} blocking issues"}
**Provider:** {result.provider}
**Model:** {result.model}
**Tokens:** {result.total_tokens:,}
**Cost:** ${result.cost_usd:.4f}
**Artifact Bindings:** {len(artifact_bindings)}

---

"""
        yield "data: " + json.dumps({"type": "token", "content": summary_details}) + "\n\n"
        response_parts.append(summary_details)
        
        # Stream architecture content
        yield "data: " + json.dumps({"type": "token", "content": "### Architecture Document\n\n"}) + "\n\n"
        response_parts.append("### Architecture Document\n\n")
        
        content = result.content
        chunk_size = 200
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
            response_parts.append(chunk)
            await asyncio.sleep(0.01)
        
        # =====================================================================
        # Step 6: Emit completion events with artifact bindings
        # =====================================================================
        
        yield "data: " + json.dumps({
            "type": "work_artifacts",
            "spec_id": spec_id,
            "job_id": job_id,
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_mode": "deep",
            "critique_passed": critique_passed,
            "artifact_bindings": artifact_bindings,  # v2.1: Include for Overwatcher
            "artifacts": [
                f"arch_v{final_version}.md",
                f"critique_v{final_version}.json",
            ],
        }) + "\n\n"
        
        if critique_passed:
            next_step = f"""

---
✅ **Ready for Implementation**

Architecture approved with {len(artifact_bindings)} artifact binding(s).
Critique mode: deep (blocker filtering enabled, stops early when clean)

🔧 **Next Step:** Say **'Astra, command: send to overwatcher'** to implement.
"""
        else:
            next_step = f"""

---
⚠️ **Critique Not Fully Passed**

{blocking_issues} blocking issues remain (after filtering for real blockers only).

You may:
- Re-run with updated spec
- Proceed to Overwatcher with caution
"""
        
        yield "data: " + json.dumps({"type": "token", "content": next_step}) + "\n\n"
        response_parts.append(next_step)
        
        # Save to memory
        full_response = "".join(response_parts)
        if memory_service and memory_schemas:
            try:
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id, role="assistant", content=full_response,
                    provider=pipeline_provider, model=pipeline_model
                ))
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done",
            "provider": pipeline_provider,
            "model": pipeline_model,
            "total_length": len(full_response),
            "spec_id": spec_id,
            "job_id": job_id,
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_mode": "deep",
            "critique_passed": critique_passed,
            "artifact_bindings": len(artifact_bindings),
            "tokens": result.total_tokens,
            "cost_usd": result.cost_usd,
        }) + "\n\n"
        
    except Exception as e:
        logger.exception("[critical_pipeline] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


__all__ = [
    "generate_critical_pipeline_stream",
    "JobKind",
    "MicroQuickcheckResult",
    "micro_quickcheck",
]
