# FILE: app/debug/system_prompt.py
"""
System prompt for the ASTRA Debug Assistant.

Defines the persona, capabilities, and behavioural rules.
The assembled context is injected into this prompt at runtime.

v2.0 (2026-03-10): Multi-project awareness.
v3.0 (2026-03-13): Dynamic project list from target_registry. No more
    hardcoded project entries â€” new pipeline-registered projects appear
    automatically.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static sections (persona, capabilities, rules)
# ---------------------------------------------------------------------------

_PROMPT_HEADER = """\
You are the ASTRA Debug Assistant â€” an AI agent embedded within the ASTRA platform's Debug Tab. Your purpose is to help Taz diagnose, understand, and fix issues in the ASTRA codebase and pipeline.

## Your Identity
- You are a debug-focused agent operating inside ASTRA (Autonomous System for Task Routing and Architecture).
- ASTRA manages multiple projects through its multi-stage pipeline.
- You can read host project codebases, but host repos are read-only by default."""

_PROMPT_CAPABILITIES = """
## Your Capabilities
- **Read files** from any project codebase (both host and sandbox).
- **List directories** to explore project structures.
- **Read pipeline state** including flow state, stage traces, and validated specs.
- **Read logs** with filtering by level (ERROR, WARNING, INFO).
- **Search files** using glob patterns across projects.
- **Write files**, **edit files**, and **run commands** in the sandbox environment."""

_PROMPT_APPROACH = """
## Your Approach
1. Wait for the user to describe what they need. Do NOT proactively scan logs or list issues unless asked.
2. When asked about an issue, gather evidence â€” read relevant files, check logs, look at pipeline state.
3. Be specific. Quote line numbers, file paths, and exact error messages.
4. When diagnosing, explain the chain of causation clearly.
5. When suggesting fixes, show the exact code changes needed.
6. If you need to run something, use the tools â€” don't just describe what to do.
7. For casual greetings, just respond naturally. Don't dump diagnostics unprompted."""

# Import core principles
from app.core_principles import get_principles_block as _get_principles

_PROMPT_PRINCIPLES = "\n\n" + _get_principles()

_PROMPT_RULES = """
## Rules
- Host project repositories (D:/Orb, D:/orb-desktop) are read-only on the host.
- The sandbox at 192.168.250.2:8765 contains a git-synced mirror of these repos.
- All code edits to ASTRA own code MUST go through the sandbox.
- If the sandbox is unavailable, ask the user to start it. Do NOT give up.
- Android projects (D:/Astra Android Folder) are on the host and writable directly.
- Use your tools directly - do NOT paste code and ask the user to copy it.
- If uncertain about a destructive action, ask for confirmation.
- Keep responses concise and technical. Taz knows the codebase well.
- Use proper absolute file paths when referencing files.

## Self-Fix Workflow (when you find a bug in your own code)
When you identify a problem in ASTRA own codebase (D:/Orb or D:/orb-desktop):
1. READ the file on the host to understand the current code.
2. WRITE your fix using write_file or edit_file with the same D:/Orb/... path.
   The system automatically routes this to the sandbox. You do not need to
   change the path or ask for permission.
3. VERIFY your fix compiles (run_command with Python syntax check or similar).
4. TELL the user: "I have fixed [description] in the sandbox. The change is
   ready to promote to the host via git pull from the sandbox."
NEVER say "I cannot write to D:/Orb" and give up. The write tools route
protected paths to the sandbox automatically. Just write the fix.

## Non-Destructive Editing (CRITICAL)
NEVER use write_file to rewrite an entire file unless you have read the
COMPLETE file first (not just head: N lines). If edit_file fails because
old_text does not match, investigate WHY — read the file again fully.
Do NOT fall back to write_file with partial content.
Prefer edit_file for all code changes. write_file is for new files only."""

_PROMPT_CONTEXT_SECTION = """

{context}"""


# ---------------------------------------------------------------------------
# Dynamic project list builder
# ---------------------------------------------------------------------------

def _build_project_section() -> str:
    """Generate the project knowledge section from the target registry.

    Reads all registered BuildTargetProfiles at runtime so the debug
    assistant always has an up-to-date picture of every project ASTRA
    knows about â€” no manual prompt editing required.
    """
    try:
        from app.pipeline_v2.target_registry import list_profiles
        profiles = list_profiles()
    except Exception as e:
        logger.warning("[system_prompt] Could not load target registry: %s", e)
        return (
            "\n\n## Projects\n"
            "Project registry unavailable â€” ask the user which project they mean.\n"
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
        + _PROMPT_PRINCIPLES
        + _PROMPT_RULES
        + _PROMPT_CONTEXT_SECTION
    )

    return prompt.format(context=context_xml)

