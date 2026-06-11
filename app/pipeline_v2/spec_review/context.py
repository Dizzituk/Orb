# FILE: app/pipeline_v2/spec_review/context.py
"""
Builds the full context package the reviewer sees.

Pulls:
  - Weaver intent text (what the user wanted)
  - SpecGate spec (what was agreed to build)
  - Builder's list of files written + its final text message
  - Build / compile output
  - BVL tier outcomes (if already run)
  - Source of every file the builder wrote

Opus 4.7 has a 1M-token context window, so we can afford to send the
full source of every file without aggressive truncation. We do cap at
~200KB of source to protect against pathological cases.

v1.0 (2026-04-18): Initial implementation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile
    from app.pipeline_v2.models import BuildResult

logger = logging.getLogger(__name__)


# Rough caps so we don't blow the context on a weird edge case. 1M tokens
# is ~4MB of text; 200KB of source keeps us comfortably inside that while
# leaving room for the spec + prompt + thinking budget.
MAX_SOURCE_CHARS = 200_000
MAX_PER_FILE_CHARS = 15_000
MAX_SPEC_CHARS = 30_000
MAX_WEAVER_CHARS = 15_000
MAX_BUILD_OUTPUT_CHARS = 8_000


async def assemble_review_context(
    spec: Dict[str, Any],
    build_result: "BuildResult",
    profile: "BuildTargetProfile",
    intent_text: str = "",
    bvl_report: Optional[Any] = None,
    build_output: str = "",
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Build the context dictionary ready to format into the user prompt.

    Every key maps to a plain string so the prompt template can do
    simple .format() substitution. No nested structures.

    Returns:
        Dict with keys: weaver_intent, spec_text, project_name, language,
        framework, architecture, package_name, project_root,
        files_written, build_output, tier_outcomes, source_concatenation.
    """
    ctx: Dict[str, str] = {}

    # ── Weaver intent ────────────────────────────────────────────────
    ctx["weaver_intent"] = _truncate(
        intent_text or "(no Weaver intent text available)",
        MAX_WEAVER_CHARS,
    )

    # ── SpecGate spec ────────────────────────────────────────────────
    spec_text = _spec_to_text(spec)
    ctx["spec_text"] = _truncate(spec_text, MAX_SPEC_CHARS)

    # ── Build target profile fields ─────────────────────────────────
    ctx["project_name"] = str(getattr(profile, "project_name", "(unknown)"))
    ctx["language"] = str(getattr(profile, "language", "(unknown)"))
    ctx["framework"] = str(getattr(profile, "framework", "(unknown)"))
    ctx["architecture"] = str(getattr(profile, "architecture_pattern", "(unknown)"))
    ctx["package_name"] = str(getattr(profile, "package_name", "(unknown)"))
    ctx["project_root"] = str(getattr(profile, "project_root", "(unknown)"))

    # ── Files written ────────────────────────────────────────────────
    files_written = list(build_result.all_files_written or [])
    # JOB 2 (2026-06-10): the reviewer reviews the WHOLE feature file-set,
    # not just this run's diff. Re-runs that touched one file were handing
    # the reviewer a single file of source, producing false "never created"
    # criticals about files that exist on disk.
    feature_files = _collect_feature_files(spec, manifest, files_written)
    extra_files = [f for f in feature_files if f not in files_written]
    if files_written:
        ctx["files_written"] = "\n".join(f"- {f}" for f in files_written)
        if extra_files:
            ctx["files_written"] += (
                "\n\nAdditional feature files included for review "
                "(written in earlier runs, part of this feature's file scope):\n"
                + "\n".join(f"- {f}" for f in extra_files)
            )
    else:
        ctx["files_written"] = "(builder reported no files)"

    # ── Build output ─────────────────────────────────────────────────
    ctx["build_output"] = _truncate(
        build_output or "(no build output captured)",
        MAX_BUILD_OUTPUT_CHARS,
    )

    # ── Tier outcomes ────────────────────────────────────────────────
    ctx["tier_outcomes"] = _bvl_summary(bvl_report)

    # ── Source concatenation ────────────────────────────────────────
    ctx["source_concatenation"] = await _concat_sources(
        files_written=feature_files or files_written,
        profile=profile,
    )

    return ctx


def _collect_feature_files(
    spec: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    files_written: List[str],
) -> List[str]:
    """JOB 2 (2026-06-10): union of this run's files + the feature's full
    file scope, so re-runs review the whole feature.

    Priority order: files written this run first, then manifest segment
    file_scope, then spec segment file_scope. Deduped case-insensitively on
    normalised separators, original casing preserved.
    """
    ordered: List[str] = []
    seen = set()

    def _add(path: Any) -> None:
        if not path or not isinstance(path, str):
            return
        key = path.replace("\\", "/").lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(path)

    for f in files_written or []:
        _add(f)
    for source in (manifest, spec):
        if not isinstance(source, dict):
            continue
        for seg in source.get("segments", []) or []:
            if isinstance(seg, dict):
                for f in seg.get("file_scope", []) or []:
                    _add(f)
    return ordered


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _spec_to_text(spec: Any) -> str:
    """Render the spec as readable text for the prompt."""
    if isinstance(spec, dict):
        try:
            return json.dumps(spec, indent=2, ensure_ascii=False)
        except Exception:
            return str(spec)
    return str(spec or "")


def _bvl_summary(bvl_report: Optional[Any]) -> str:
    """Render BVL tier outcomes as a compact summary string."""
    if bvl_report is None:
        return "(BVL not run for this target)"
    try:
        components = getattr(bvl_report, "component_results", []) or []
        if not components:
            return "(BVL report has no component results)"

        lines: List[str] = []
        for comp in components:
            path = getattr(comp, "component_path", "?")
            lines.append(f"Component: {path}")
            tiers = getattr(comp, "tier_results", []) or []
            for tr in tiers:
                tier = getattr(tr, "tier", "?")
                status = getattr(tr, "status", "?")
                tier_val = tier.value if hasattr(tier, "value") else str(tier)
                status_val = status.value if hasattr(status, "value") else str(status)
                lines.append(f"  - {tier_val}: {status_val}")
                checks = getattr(tr, "checks", []) or []
                for c in checks[:6]:
                    mark = "\u2713" if getattr(c, "passed", False) else "\u2717"
                    name = getattr(c, "name", "?")
                    msg = getattr(c, "message", "")[:100]
                    lines.append(f"    {mark} {name}: {msg}")
            if getattr(comp, "blocked", False):
                lines.append(f"  BLOCKED: {getattr(comp, 'block_reason', '')[:200]}")
            if getattr(comp, "signed_off", False):
                lines.append("  SIGNED OFF")
        return "\n".join(lines) or "(empty BVL report)"
    except Exception as exc:
        logger.debug("[spec_review] Failed to summarise BVL report: %s", exc)
        return f"(could not summarise BVL report: {exc})"


async def _concat_sources(
    files_written: List[str],
    profile: "BuildTargetProfile",
) -> str:
    """Read and concatenate the source of every file the builder wrote.

    Each file is wrapped with a header + fenced code block so the reviewer
    can easily cite "file:line" references. Files are read via sandbox_tools
    which handles host vs sandbox routing based on the profile.
    """
    if not files_written:
        return "(no files to concatenate)"

    from app.pipeline_v2 import sandbox_tools

    pieces: List[str] = []
    total = 0

    for f in files_written:
        if total >= MAX_SOURCE_CHARS:
            pieces.append(
                f"\n\n[TRUNCATED: source concatenation exceeded "
                f"{MAX_SOURCE_CHARS} chars; {len(files_written)} total files "
                f"in build but only showing up to this point]"
            )
            break

        try:
            content = await sandbox_tools.read_file(f, profile=profile)
        except Exception as exc:
            pieces.append(f"\n\n### {f}\n[Could not read file: {exc}]")
            continue

        if content is None:
            pieces.append(f"\n\n### {f}\n[File not found on disk]")
            continue

        if len(content) > MAX_PER_FILE_CHARS:
            truncated = (
                content[:MAX_PER_FILE_CHARS]
                + f"\n\n[TRUNCATED: file is {len(content)} chars total, "
                  f"showing first {MAX_PER_FILE_CHARS}]"
            )
            content = truncated

        # Number the lines so the reviewer can cite line numbers accurately.
        numbered = _prefix_line_numbers(content)
        piece = f"\n\n### {f}\n\n```\n{numbered}\n```"
        pieces.append(piece)
        total += len(piece)

    return "".join(pieces)


def _prefix_line_numbers(content: str) -> str:
    """Prefix each line with `NNNN: ` so the reviewer can cite lines.

    Matches the convention used elsewhere in the project (view tool, etc.).
    Uses a 4-digit width, sufficient for files up to 9999 lines.
    """
    lines = content.split("\n")
    return "\n".join(
        f"{i:>4}: {line}"
        for i, line in enumerate(lines, start=1)
    )


def _truncate(text: str, cap: int) -> str:
    """Truncate text to `cap` chars with a trailing marker if cut."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n\n[TRUNCATED at {cap} chars]"
