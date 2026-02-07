# FILE: app/llm/critical_pipeline/prompt_builder.py
"""
Architecture system-prompt construction for Critical Pipeline.

Builds the complete system prompt used when running architecture-mode jobs
through the Block 4-6 high-stakes pipeline.
"""

import logging
from typing import Dict, Any, List

from app.llm.critical_pipeline.artifact_binding import build_artifact_binding_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Refactor Job Detection
# =============================================================================

def is_refactor_job(spec_data: Dict[str, Any], message: str) -> bool:
    """Detect whether this job involves refactoring / restructuring."""
    job_kind = (spec_data.get("job_kind") or "").lower()
    if job_kind in ("refactor", "restructure", "optimize"):
        logger.info("[prompt_builder] is_refactor_job: True (job_kind=%s)", job_kind)
        return True

    summary = (spec_data.get("summary", "") or "").lower()
    objective = (spec_data.get("objective", "") or "").lower()
    msg_lower = (message or "").lower()
    key_reqs = spec_data.get("key_requirements", [])
    key_reqs_text = " ".join(str(r).lower() for r in key_reqs) if key_reqs else ""
    all_text = f"{summary} {objective} {msg_lower} {key_reqs_text}"

    for kw in ["refactor", "restructure", "split file", "tech debt",
                "file too large", "reorgani", "decompose", "break apart"]:
        if kw in all_text:
            logger.info("[prompt_builder] is_refactor_job: True (keyword=%s)", kw)
            return True

    logger.info("[prompt_builder] is_refactor_job: False")
    return False


# =============================================================================
# Objective Extraction
# =============================================================================

def extract_original_request(spec_data: Dict[str, Any], message: str) -> str:
    """Extract the best description of the task from spec data."""
    summary = spec_data.get("summary", "")
    objective = spec_data.get("objective", "")

    GENERIC = ["job description", "weaver", "build spec", "create spec",
               "draft", "generated", "placeholder"]
    obj_generic = any(g in (objective or "").lower() for g in GENERIC) or len(objective or "") < 20

    if summary and len(summary) > len(objective or ""):
        return summary
    if objective and not obj_generic:
        return objective
    if summary:
        return summary

    inputs = spec_data.get("inputs", [])
    if inputs and isinstance(inputs, list) and inputs:
        first = inputs[0]
        if isinstance(first, dict):
            example = first.get("example", "")
            if example and len(example) > 20:
                return f"Task: {example}"
    return message


# =============================================================================
# Spec Constraint Extraction
# =============================================================================

def extract_spec_constraints(spec_data: Dict[str, Any]) -> List[str]:
    """Extract explicit constraints for INVIOLABLE enforcement."""
    constraints: List[str] = []

    KEYWORDS = [
        "don't rewrite", "don't rebuild", "as-is", "do not",
        "use existing", "never", "must not", "phase 1 only",
        "in-memory", "no disk", "no cross-platform",
        "don't implement", "don't add", "don't create",
        "only", "not implement",
    ]

    for req in spec_data.get("key_requirements", []):
        if isinstance(req, str) and any(kw in req.lower() for kw in KEYWORDS):
            constraints.append(req)

    for pref in spec_data.get("design_preferences", []):
        if isinstance(pref, str):
            constraints.append(pref)

    for c in spec_data.get("constraints", []):
        if isinstance(c, str):
            constraints.append(c)

    for c in spec_data.get("grounding_data", {}).get("constraints", []):
        if isinstance(c, str) and c not in constraints:
            constraints.append(c)

    return constraints


# =============================================================================
# System Prompt Assembly
# =============================================================================

def build_architecture_system_prompt(
    *,
    spec_id: str,
    spec_hash: str,
    spec_data: Dict[str, Any],
    artifact_bindings: List[Dict[str, Any]],
    evidence_context: str,
    spec_constraints: List[str],
) -> str:
    """Assemble the full system prompt for architecture generation."""

    content_verbatim = (
        spec_data.get("content_verbatim")
        or spec_data.get("context", {}).get("content_verbatim")
        or spec_data.get("metadata", {}).get("content_verbatim")
    )
    location = (
        spec_data.get("location")
        or spec_data.get("context", {}).get("location")
        or spec_data.get("metadata", {}).get("location")
    )
    scope_constraints = (
        spec_data.get("scope_constraints")
        or spec_data.get("context", {}).get("scope_constraints")
        or spec_data.get("metadata", {}).get("scope_constraints")
        or []
    )

    parts = []

    parts.append(f"""You are Claude Opus, generating a detailed architecture document.

SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

You are working from a validated PoT Spec. Your architecture MUST:
1. Address all MUST requirements from the spec
2. Consider all constraints
3. Be buildable by a solo developer on Windows 11
4. Include the SPEC_ID and SPEC_HASH header at the top of your output

## CONTENT PRESERVATION (CRITICAL)
""")

    if content_verbatim:
        parts.append(f"""
**EXACT FILE CONTENT REQUIRED:**
The file content MUST be EXACTLY: "{content_verbatim}"
Do NOT paraphrase, summarize, or modify this content in any way.
""")

    if location:
        parts.append(f"""
**EXACT LOCATION REQUIRED:**
The output MUST be written to: {location}
Use this EXACT path - do not substitute or normalize it.
""")

    if scope_constraints:
        parts.append(
            "\n**SCOPE CONSTRAINTS:**\n"
            + "\n".join(f"- {c}" for c in scope_constraints)
            + "\nThe implementation MUST NOT operate outside these boundaries.\n"
        )

    parts.append(build_artifact_binding_prompt(artifact_bindings))

    if spec_constraints:
        parts.append(
            "\n## INVIOLABLE SPEC CONSTRAINTS\n\n"
            "The following constraints are ABSOLUTE. Your architecture MUST NOT violate any of them.\n"
            "A DECISION block can NEVER override these \u2014 they come directly from the user\u2019s spec.\n"
            "If your architecture would violate any constraint, STOP and redesign.\n\n"
            + "\n".join(f"- \u274c VIOLATION IF BROKEN: {c}" for c in spec_constraints)
            + "\n\nThese are not suggestions. Breaking ANY of these constraints means the architecture is WRONG.\n"
        )
        logger.info("[prompt_builder] Injected %d spec constraints", len(spec_constraints))

    if evidence_context and len(evidence_context) > 50:
        parts.append(f"""
## CODEBASE EVIDENCE (Comprehensive)

This is comprehensive evidence gathered from the codebase. Use this to understand:
- Existing code structure and patterns
- Available modules and services
- Integration points
- Actual file contents for context

{evidence_context}

Reference these patterns when designing the architecture.
""")

    parts.append("""
## OUTPUT FORMAT REQUIREMENTS

Your architecture document MUST include a `## File Inventory` section with two markdown tables:

### New Files
| File | Purpose |
|------|--------|
| `path/to/file.ext` | Brief description |

### Modified Files
| File | Purpose |
|------|--------|
| `path/to/file.ext` | Brief description |

This section is REQUIRED \u2014 the downstream executor parses it to know which files to create and modify.

**CRITICAL \u2014 MULTI-ROOT PATH RULES:**
This project has TWO separate root directories:
- **Backend** (`D:\\Orb`): Python/FastAPI. Paths start with `app/`, `tests/`, `main.py`, `requirements.txt`, etc.
- **Frontend** (`D:\\orb-desktop`): Electron/React/TypeScript. Paths MUST use the `orb-desktop/` prefix.

Path format rules:
- Backend files: relative to D:\\Orb (e.g. `app/routers/voice.py`, `main.py`, `.env`)
- Frontend files: MUST start with `orb-desktop/` (e.g. `orb-desktop/src/components/VoiceInput.tsx`, `orb-desktop/package.json`)
- NEVER use bare `src/` for frontend files \u2014 always prefix with `orb-desktop/`
- The architecture map uses these same conventions \u2014 follow them exactly.

If no files are modified, include the table header with no rows.

Generate a complete, detailed architecture document.""")

    return "".join(parts)
