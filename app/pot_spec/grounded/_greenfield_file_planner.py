# FILE: app/pot_spec/grounded/_greenfield_file_planner.py
# Purpose: live9 — APEX plans the modular file layout for greenfield specs that name no files.
# Called-by: app.pot_spec.grounded.spec_runner
# Depends-on: app.llm.stage_roles, app.pipeline_v2.llm_caller
# Last-renovated: 2026-07-04
"""Greenfield file planner (live9, 2026-07-04).

The first live greenfield run (Tazza's Tetris, python/generic) died silently:
_build_generic_spec embeds NO file paths, _extract_file_scope_from_spec found
nothing, the manifest shipped with 0 files, and the scaffold engine had
nothing to build. The android/react/fastapi templates carry file plans; the
generic template cannot know a job's layout — so the SPECGATE_MAIN (APEX)
model plans it at spec time, which is exactly the "architecture generation"
the pipeline promises.

Deterministic containment: the LLM only PROPOSES lines; every path must pass
a strict flat-layout regex (src/ or tests/, single segment, whitelisted
extension — guaranteed extractable by _extract_file_scope_from_spec), is
deduped and capped. Garbage output -> None, and the caller leaves the spec
untouched (the 0-file hard-stop downstream still protects the run).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Flat layout only: one segment under src/ or tests/. Nested dirs are
# rejected because the downstream extractor's relative-path pattern only
# matches a single segment after the known prefix.
_PLANNED_PATH_RE = re.compile(
    r"^(?:src|tests)/[A-Za-z0-9_\-]+\.(?:py|json|md|yaml|yml|js|ts|css)$"
)

_MAX_PLANNED_FILES = 24

_PLANNER_SYSTEM = """You are the ASTRA build architect. Plan the file layout for a brand-new project.

Rules:
- Output ONLY a bullet list, one file per line, no prose before or after.
- Each line: - `src/<name>.<ext>` — <one-line responsibility>
- FLAT layout: every file lives directly in src/ (tests in tests/). No nested folders.
- One responsibility per file, one public class or entry function each.
- Size discipline: each file must plausibly stay under 20 KB (hard cap 30 KB) — split concerns until that is credible.
- Include exactly one entry point: `src/main.py` (or the language's equivalent).
- 6-16 files for a small app; only what the requirements need.
- NO asset files: plan code-only modules. Audio and visuals must be procedural
  (synthesized waveforms, drawn shapes, system fonts via e.g. pygame.font.SysFont)
  or absent. Never plan an assets/ directory or any file the code merely
  "loads" — the build will not create it and the app must boot without it.
"""


def _parse_planned_files(text: str) -> List[Tuple[str, str]]:
    """Extract validated (path, responsibility) pairs from planner output."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for raw_line in (text or "").split("\n"):
        line = raw_line.strip().lstrip("-*").strip()
        if not line:
            continue
        # Path is the backticked token, else the first whitespace token.
        m = re.search(r"`([^`]+)`", line)
        token = (m.group(1) if m else line.split()[0]).strip()
        token = token.replace("\\", "/").lstrip("./")
        if not _PLANNED_PATH_RE.match(token):
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        # Responsibility: text after the token, minus separators.
        tail = line[m.end():] if m else line[len(token):]
        desc = tail.strip().lstrip("—-–:").strip()
        desc = " ".join(desc.split())[:140] or "module"
        out.append((token, desc))
        if len(out) >= _MAX_PLANNED_FILES:
            break
    return out


def render_files_section(files: List[Tuple[str, str]]) -> str:
    lines = ["## Files to create", ""]
    lines += [f"- `{path}` — {desc}" for path, desc in files]
    lines.append("")
    return "\n".join(lines)


async def plan_greenfield_files(
    goal: str,
    what_to_do: str,
    build_profile: Dict,
    job_id: Optional[str] = None,
) -> Optional[str]:
    """Return a '## Files to create' markdown section, or None on failure."""
    try:
        from app.llm.stage_roles import resolve_stage_role
        from app.pipeline_v2.llm_caller import call_llm

        role = resolve_stage_role("SPECGATE_MAIN")
        user_prompt = (
            f"Project: {build_profile.get('project_name', 'Untitled')}\n"
            f"Language: {build_profile.get('language', 'unknown')} | "
            f"Framework: {build_profile.get('framework', 'generic')} | "
            f"Build system: {build_profile.get('build_system', 'unknown')}\n\n"
            f"Goal: {(goal or '').strip()[:500]}\n\n"
            f"Requirements:\n{(what_to_do or '').strip()[:6000]}\n\n"
            "Plan the file layout now. Bullet list only."
        )
        text = await call_llm(
            provider=role.provider,
            model=role.model,
            system_prompt=_PLANNER_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=2000,
            stage="specgate_plan",
            job_id=job_id,
        )
    except Exception as exc:
        logger.warning("[greenfield_planner] Planning call failed: %s", exc)
        return None

    files = _parse_planned_files(text)
    if not files:
        logger.warning(
            "[greenfield_planner] Planner output had no valid flat src/ paths "
            "(%d chars) — leaving spec without a file plan", len(text or ""),
        )
        return None

    logger.info(
        "[greenfield_planner] Planned %d file(s): %s",
        len(files), ", ".join(p for p, _ in files[:8]),
    )
    print(f"[spec_runner] live9 GREENFIELD FILE PLAN: {len(files)} file(s) planned by {role.provider}/{role.model}")
    return render_files_section(files)


__all__ = ["plan_greenfield_files", "render_files_section", "_parse_planned_files"]
