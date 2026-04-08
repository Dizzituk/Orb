# FILE: app/pipeline_v2/bvl/cross_model_diagnostic.py
"""
Cross-Model Diagnostic — Opus reviews what GPT-5.4 built.

When the BVL retry loop detects a failure, this module sends the
failing code + error signals to a DIFFERENT model (Opus) for
diagnosis. The idea: the builder model has systematic blind spots.
A different model family catches anomalies the builder missed.

Flow:
  1. GPT-5.4 builds the code (agentic builder)
  2. BVL tier fails
  3. Opus reads the failing file + error + spec slice
  4. Opus produces a structured diagnosis: what is wrong, why, what to fix
  5. That diagnosis goes back to GPT-5.4 as a handover context for the fix

This gives you two model perspectives on every failure instead of one
model trying to debug its own assumptions.

v1.0 (2026-04-06): Initial implementation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)

# Diagnosis prompt template
_DIAGNOSIS_SYSTEM = """\
You are a diagnostic reviewer. A different AI model (GPT-5.4) wrote \
the code below and it failed verification.

CRITICAL RULES:
- You are the DIAGNOSTICIAN, not the surgeon. You do NOT rewrite code.
- You produce a SHORT, SURGICAL diagnosis only.
- Your output must be UNDER 500 WORDS. No exceptions.
- Do NOT suggest rewriting entire files or restructuring architecture.
- Do NOT second-guess the builder's architectural decisions.
- ONLY identify the specific, small errors causing the failure.
- Reference exact line numbers, function names, import paths, variable names.
- The builder wrote 10+ files that work together. Your job is to find \
  the tiny mismatches, wrong imports, missing fields, or typos — not \
  to redesign anything.

You have full project context below. Use it to verify imports, schemas, \
and API contracts. Check whether a referenced class/function actually \
exists in the file it is imported from.

Output format (strict, keep it short):
<DIAGNOSIS>
PROBLEM: One sentence. What specifically broke.
ROOT_CAUSE: One sentence. Why the builder got this specific thing wrong.
FIXES:
- [file:line] Exact change needed (e.g. "change import X to import Y")
- [file:line] Exact change needed
CONFIDENCE: HIGH | MEDIUM | LOW
</DIAGNOSIS>

Keep FIXES to 1-5 items maximum. Each fix is ONE line change, not a \
paragraph. If you cannot identify a specific fix, say so — do not \
invent one.
"""

_DIAGNOSIS_USER = """\
## Project Context
{project_context}

## Failed Component
File: {component_path}

## Error Signal
{error_signal}

## Logcat / Build Output
{logcat_excerpt}

## Full Spec (SPoT)
{spec_slice}

## Source Code of Failing File
```
{source_code}
```

## Related Files (imports/dependencies)
{related_files}

## Previous Fix Attempts (if any)
{previous_errors}

Diagnose the root cause. Use the project context and related files to \
verify imports, schemas, and API contracts. Be specific about what the \
builder got wrong.
"""


async def diagnose_failure(
    component_path: str,
    error_signal: str,
    logcat_excerpt: str,
    spec_slice: str,
    source_code: str,
    previous_errors: Optional[List[str]] = None,
    profile: Optional["BuildTargetProfile"] = None,
    emit: Optional[Callable[[str], None]] = None,
    full_spec: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Send a failure to Opus for cross-model diagnosis.

    Opus gets full project context so it understands the architectural
    intent, but its output is capped and constrained to surgical
    diagnosis only — no rewrites, no restructuring.

    Args:
        component_path: Path to the failing file.
        error_signal: Error message / stack trace from the failed tier.
        logcat_excerpt: Logcat or build output excerpt.
        spec_slice: Relevant portion of the spec for this component.
        source_code: The actual source code that failed.
        previous_errors: List of error strings from prior retry attempts.
        profile: Build target profile (for context).
        emit: Progress callback.
        full_spec: The complete SPoT spec (for architectural context).
        manifest: The segment manifest (for file relationships).

    Returns:
        Structured diagnosis string to hand to the builder, or None on failure.
    """
    emit = emit or (lambda msg: None)

    # Get Overwatcher (Opus) config
    try:
        from app.llm.stage_models import get_overwatcher_config
        config = get_overwatcher_config()
        provider = config.provider
        model = config.model
        max_tokens = min(config.max_output, 1500)  # Surgical: cap output for diagnosis
    except Exception as e:
        logger.error("[cross_model] Failed to load overwatcher config: %s", e)
        return None

    emit(f"   🔬 Cross-model diagnosis: sending to {provider}/{model}")
    logger.info(
        "[cross_model] Diagnosing %s via %s/%s",
        component_path, provider, model,
    )

    # Build project context so Opus understands the full picture
    project_context = _build_project_context(profile, manifest)

    # Gather related files (imports/dependencies of the failing file)
    related_files_text = await _gather_related_files(
        source_code, component_path, profile
    )

    # Use full spec if available, otherwise fall back to slice
    spec_text = spec_slice[:2000]
    if full_spec:
        spec_text = json.dumps(full_spec, indent=2)[:6000]

    prev_errors_text = "None"
    if previous_errors:
        prev_errors_text = "\n".join(
            f"Attempt {i+1}: {err[:300]}"
            for i, err in enumerate(previous_errors)
        )

    user_prompt = _DIAGNOSIS_USER.format(
        project_context=project_context,
        component_path=component_path,
        error_signal=error_signal[:2000],
        logcat_excerpt=logcat_excerpt[:1000],
        spec_slice=spec_text,
        source_code=source_code[:8000],
        related_files=related_files_text,
        previous_errors=prev_errors_text,
    )

    # Call the overwatcher model
    try:
        from app.pipeline_v2.llm_caller import call_llm

        response = await call_llm(
            provider=provider,
            model=model,
            system_prompt=_DIAGNOSIS_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

        if not response:
            logger.warning("[cross_model] Empty response from %s/%s", provider, model)
            return None

        logger.info(
            "[cross_model] Diagnosis received: %d chars from %s/%s",
            len(response), provider, model,
        )
        emit(f"   🔬 Diagnosis received ({len(response)} chars)")

        return _format_diagnosis_for_builder(response, component_path)

    except Exception as e:
        logger.error("[cross_model] Diagnosis call failed: %s", e)
        emit(f"   ⚠️ Cross-model diagnosis failed: {e}")
        return None


def _format_diagnosis_for_builder(
    diagnosis: str,
    component_path: str,
) -> str:
    """Format the Opus diagnosis as a builder handover context.

    Wraps the diagnosis so the builder understands this is an external
    review, not its own analysis.
    """
    return (
        f"## Cross-Model Diagnostic Report\n"
        f"A different model (Overwatcher) reviewed the failing code and "
        f"produced the following diagnosis. Use this to guide your fix.\n\n"
        f"**Component:** {component_path}\n\n"
        f"{diagnosis}\n\n"
        f"Apply the fix instructions above. Read the file first, then "
        f"make targeted changes. Do not rewrite the entire file unless "
        f"the diagnosis says the structure is fundamentally wrong."
    )


def _build_project_context(
    profile: Optional["BuildTargetProfile"],
    manifest: Optional[Dict[str, Any]],
) -> str:
    """Build a concise project context string for Opus.

    Gives Opus the big picture: what project, what language, what
    architecture, what files are in the build, what the segments are.
    """
    parts = []

    if profile:
        parts.append(
            f"Project: {profile.project_name} ({profile.project_id})\n"
            f"Language: {profile.language} / {profile.framework}\n"
            f"Architecture: {profile.architecture_pattern}\n"
            f"Root: {profile.project_root}\n"
            f"Source root: {profile.absolute_source_root}\n"
            f"Package: {profile.package_name}"
        )

    if manifest:
        seg_names = [
            s.get("segment_id", "?") for s in manifest.get("segments", [])
        ]
        file_list = []
        for seg in manifest.get("segments", []):
            file_list.extend(seg.get("file_scope", []))
        parts.append(
            f"Segments: {', '.join(seg_names[:15])}\n"
            f"All files in build: {', '.join(file_list[:20])}"
        )

    return "\n\n".join(parts) if parts else "No project context available."


async def _gather_related_files(
    source_code: str,
    component_path: str,
    profile: Optional["BuildTargetProfile"],
) -> str:
    """Read files that the failing component imports or depends on.

    Parses import statements from the source code and reads the
    referenced files so Opus can verify cross-file contracts.
    Returns a condensed summary (first 50 lines of each, max 3 files).
    """
    import re

    related = []

    if not source_code:
        return "No related files available."

    # Kotlin imports: import com.astra.astrabridge.network.models.ChatRequest
    kt_imports = re.findall(r"import\s+([\w.]+)", source_code)

    # Python imports: from app.bridge.schemas import BridgeChatRequestV2
    py_from = re.findall(r"from\s+([\w.]+)\s+import", source_code)
    py_import = re.findall(r"^import\s+([\w.]+)", source_code, re.MULTILINE)

    all_modules = list(set(kt_imports + py_from + py_import))

    # Try to resolve module paths to files and read them
    files_read = 0
    for mod in all_modules[:5]:
        if files_read >= 3:
            break

        # Skip standard library / framework imports
        if mod.startswith(("java.", "javax.", "android.", "kotlin.",
                           "androidx.", "kotlinx.", "os", "sys",
                           "logging", "json", "time", "asyncio",
                           "typing", "dataclasses", "enum")):
            continue

        # Try to resolve to a file path
        file_path = _module_to_path(mod, profile)
        if not file_path:
            continue

        try:
            content = await read_source_file(file_path, profile)
            if content and not content.startswith("[Could not"):
                # Truncate to first 60 lines for context
                lines = content.split("\n")[:60]
                truncated = "\n".join(lines)
                related.append(
                    f"### {file_path} (first 60 lines)\n```\n{truncated}\n```"
                )
                files_read += 1
        except Exception:
            pass

    return "\n\n".join(related) if related else "No related files resolved."


def _module_to_path(
    module: str,
    profile: Optional["BuildTargetProfile"],
) -> Optional[str]:
    """Attempt to convert a module/import path to a file path."""
    if not profile:
        return None

    root = profile.project_root.replace("\\", "/").rstrip("/")

    # Kotlin: com.astra.astrabridge.network.models.ChatRequest
    # -> app/src/main/java/com/astra/astrabridge/network/models/ChatRequest.kt
    if profile.language == "kotlin" and module.startswith(profile.package_name):
        relative = module.replace(profile.package_name, "").lstrip(".")
        parts = relative.rsplit(".", 1)  # Last part might be a class, not file
        file_parts = parts[0].replace(".", "/") if parts[0] else ""
        if file_parts:
            src = profile.source_root.replace("\\", "/")
            return f"{root}/{src}/{file_parts}.kt"

    # Python: app.bridge.schemas -> app/bridge/schemas.py
    if profile.language == "python":
        file_path = module.replace(".", "/") + ".py"
        return f"{root}/{file_path}"

    return None

async def read_source_file(
    component_path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> str:
    """Read the source code of a component for diagnosis.

    Uses the host filesystem for Android builds, sandbox for ASTRA repos.
    """
    try:
        from app.pipeline_v2.sandbox_tools import read_file
        result = await read_file(component_path, profile=profile)
        if isinstance(result, dict):
            return result.get("content", "")
        return str(result) if result else ""
    except Exception as e:
        logger.warning("[cross_model] Could not read %s: %s", component_path, e)
        return f"[Could not read file: {e}]"