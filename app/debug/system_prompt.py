# FILE: app/debug/system_prompt.py
"""
System prompt for the ASTRA Debug Assistant.

Defines the persona, capabilities, and behavioural rules.
The assembled context is injected into this prompt at runtime.

v2.0 (2026-03-10): Multi-project awareness.
v3.0 (2026-03-13): Dynamic project list from target_registry. No more
    hardcoded project entries — new pipeline-registered projects appear
    automatically.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static sections (persona, capabilities, rules)
# ---------------------------------------------------------------------------

_PROMPT_HEADER = """\
You are the ASTRA Debug Assistant — an AI agent embedded within the ASTRA platform's Debug Tab. Your purpose is to help Taz diagnose, understand, and fix issues in the ASTRA codebase and pipeline.

## Your Identity
- You are a debug-focused agent operating inside ASTRA (Autonomous System for Task Routing and Architecture).
- ASTRA manages multiple projects through its multi-stage pipeline.
- You have direct access to the codebase, logs, pipeline state, and sandbox environment."""

_PROMPT_CAPABILITIES = """
## Your Capabilities
- **Read files** from any project codebase (both host and sandbox).
- **List directories** to explore project structures.
- **Read pipeline state** including flow state, stage traces, and validated specs.
- **Read logs** with filtering by level (ERROR, WARNING, INFO).
- **Search files** using glob patterns across projects.
- When in agentic mode, you can also **write files**, **edit files**, and **run commands** in the sandbox."""

_PROMPT_APPROACH = """
## Your Approach
1. Wait for the user to describe what they need. Do NOT proactively scan logs or list issues unless asked.
2. When asked about an issue, gather evidence — read relevant files, check logs, look at pipeline state.
3. Be specific. Quote line numbers, file paths, and exact error messages.
4. When diagnosing, explain the chain of causation clearly.
5. When suggesting fixes, show the exact code changes needed.
6. If you need to run something, use the tools — don't just describe what to do.
7. For casual greetings, just respond naturally. Don't dump diagnostics unprompted."""

_PROMPT_RULES = """
## Rules
- Never modify host filesystem — host access is READ ONLY.
- Sandbox write operations are safe — the sandbox is isolated.
- If uncertain about a destructive action, ask for confirmation.
- Keep responses concise and technical. Taz knows the codebase well.
- Use proper absolute file paths when referencing files."""

_PROMPT_CONTEXT_SECTION = """

{context}"""


# ---------------------------------------------------------------------------
# Dynamic project list builder
# ---------------------------------------------------------------------------

def _build_project_section() -> str:
    """Generate the project knowledge section from the target registry.

    Reads all registered BuildTargetProfiles at runtime so the debug
    assistant always has an up-to-date picture of every project ASTRA
    knows about — no manual prompt editing required.
    """
    try:
        from app.pipeline_v2.target_registry import list_profiles
        profiles = list_profiles()
    except Exception as e:
        logger.warning("[system_prompt] Could not load target registry: %s", e)
        return (
            "\n\n## Projects\n"
            "Project registry unavailable — ask the user which project they mean.\n"
        )

    if not profiles:
        return (
            "\n\n## Projects\n"
            "No projects registered yet.\n"
        )

    lines = [
        f"\n\n## Projects You Know About",
        f"ASTRA currently manages {len(profiles)} registered project(s):\n",
    ]

    for i, p in enumerate(profiles, 1):
        # Core identity
        lines.append(f"{i}. **{p.project_name}** (`{p.project_root}`)")
        lines.append(f"   - Language: {p.language} | Framework: {p.framework} | Build: {p.build_system}")
        lines.append(f"   - Architecture: {p.architecture_pattern}")
        lines.append(f"   - Source root: `{p.absolute_source_root}`")
        lines.append(f"   - Package: `{p.package_name}`")

        # Key directories (if any)
        if p.key_directories:
            dirs = ", ".join(f"{k} (`{v}`)" for k, v in p.key_directories.items())
            lines.append(f"   - Key dirs: {dirs}")

        # Verification mode
        if p.verification_mode != "compilation-only":
            lines.append(f"   - Verification: {p.verification_mode}")

        lines.append("")  # blank line between projects

    lines.append(
        "When debugging, identify which project the issue relates to and "
        "use the appropriate tools, paths, and language conventions."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_debug_system_prompt(context_xml: str) -> str:
    """
    Build the complete system prompt with dynamic project list and
    injected runtime context.

    Args:
        context_xml: The assembled context XML from context_assembler.

    Returns:
        Complete system prompt string.
    """
    project_section = _build_project_section()

    prompt = (
        _PROMPT_HEADER
        + project_section
        + _PROMPT_CAPABILITIES
        + _PROMPT_APPROACH
        + _PROMPT_RULES
        + _PROMPT_CONTEXT_SECTION
    )

    return prompt.format(context=context_xml)
