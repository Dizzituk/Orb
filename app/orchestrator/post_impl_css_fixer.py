"""
Post-Implementation CSS Fix Loop (Fix 5b).

When post_impl_validation detects CSS class mismatches (classes used
in TSX but undefined in CSS), this module re-runs the CSS segment's
implementer with the class inventory as binding context.

Flow:
  1. Collect all className values from implemented TSX files (sandbox)
  2. Collect all .class-name definitions from implemented CSS files (sandbox)
  3. Compute the delta (used but undefined)
  4. If delta is non-empty, call the implementer for the CSS segment
     with the class inventory injected as a hard constraint
  5. Re-validate after fix

This runs between Phase 3B (validation) and Phase 4 (reconciliation).
Max 2 fix attempts to avoid infinite loops.

v1.0 (2026-03-03): Initial implementation
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --- Class extraction (reuse logic from post_impl_validation) ------------- #

_PROJECT_CLASS_PREFIXES = (
    "education-", "course-", "investments-", "content-", "social-",
    "finance-", "lifestyle-", "builds-", "debug-", "settings-",
    "ch-",
)


def _extract_tsx_classes(content: str) -> List[str]:
    """Extract all className values from TSX content."""
    classes = set()
    for m in re.finditer(r'className="([^"]+)"', content):
        for cls in m.group(1).split():
            classes.add(cls)
    for m in re.finditer(r'className=\{`([^`]+)`\}', content):
        for part in re.findall(r"([\w][\w-]*(?:__[\w-]+)?(?:--[\w-]+)?)", m.group(1)):
            if any(part.startswith(p) for p in _PROJECT_CLASS_PREFIXES):
                classes.add(part)
    return sorted(classes)


def _extract_css_classes(content: str) -> List[str]:
    """Extract all class selectors from CSS content."""
    classes = set()
    for m in re.finditer(r"\.([\w][\w-]*(?:__[\w-]+)?(?:--[\w-]+)?)", content):
        cls = m.group(1)
        if not cls[0].isdigit():
            classes.add(cls)
    return sorted(classes)


def _read_sandbox_file(path: str) -> Optional[str]:
    """Read a single file from the sandbox."""
    import requests
    try:
        resp = requests.post(
            "http://192.168.250.2:8765/fs/contents",
            json={"paths": [path]},
            timeout=15,
        )
        data = resp.json()
        for f in data.get("files", []):
            if f.get("error"):
                return None
            lines = []
            for line in f["content"].split("\n"):
                cleaned = re.sub(r"^\s*\d+:\s?", "", line)
                lines.append(cleaned)
            return "\n".join(lines)
    except Exception as e:
        logger.warning("[css_fixer] Failed to read %s: %s", path, e)
    return None


def _list_sandbox_files(directory: str, pattern: str) -> List[str]:
    """List files in a sandbox directory."""
    import requests
    cmd = (
        f"Get-ChildItem '{directory}' -File -Filter '{pattern}' "
        f"-Recurse -ErrorAction SilentlyContinue | "
        f"ForEach-Object {{ $_.FullName }}"
    )
    try:
        resp = requests.post(
            "http://192.168.250.2:8765/shell/run",
            json={"cmd": ["powershell", "-NoProfile", "-Command", cmd],
                  "timeout_sec": 10},
            timeout=15,
        )
        data = resp.json()
        stdout = (data.get("stdout") or "").strip()
        return [l.strip() for l in stdout.split("\n") if l.strip()]
    except Exception as e:
        logger.warning("[css_fixer] Failed to list %s: %s", directory, e)
    return []


def _write_sandbox_file(path: str, content: str) -> bool:
    """Write content to a file in the sandbox."""
    import requests
    import base64
    try:
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = (
            f"$bytes = [System.Convert]::FromBase64String('{b64}'); "
            f"[System.IO.File]::WriteAllBytes('{path}', $bytes)"
        )
        resp = requests.post(
            "http://192.168.250.2:8765/shell/run",
            json={"cmd": ["powershell", "-NoProfile", "-Command", cmd],
                  "timeout_sec": 10},
            timeout=15,
        )
        return True
    except Exception as e:
        logger.warning("[css_fixer] Failed to write %s: %s", path, e)
    return False

# --- Core fix logic ------------------------------------------------------- #

def _collect_class_inventory(
    segment_files: Dict[str, List[str]],
    frontend_root: str = r"D:\orb-desktop",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Collect TSX classes and CSS classes from implemented segment files.

    Returns: (tsx_classes, css_classes, css_file_paths)
    """
    tsx_classes = set()
    css_classes = set()
    css_files = []

    for seg_id, files in segment_files.items():
        for rel_path in files:
            # Build absolute path
            if rel_path.startswith("src"):
                abs_path = f"{frontend_root}\\{rel_path.replace('/', chr(92))}"
            elif rel_path.startswith("orb-desktop"):
                stripped = rel_path[len("orb-desktop/"):]
                abs_path = f"{frontend_root}\\{stripped.replace('/', chr(92))}"
            else:
                abs_path = rel_path

            content = _read_sandbox_file(abs_path)
            if content is None:
                logger.warning("[css_fixer] Could not read: %s", abs_path)
                continue

            if abs_path.endswith((".tsx", ".ts")):
                for cls in _extract_tsx_classes(content):
                    tsx_classes.add(cls)
            elif abs_path.endswith(".css"):
                css_files.append(abs_path)
                for cls in _extract_css_classes(content):
                    css_classes.add(cls)

    return sorted(tsx_classes), sorted(css_classes), css_files


def _compute_mismatches(
    tsx_classes: List[str], css_classes: List[str],
) -> List[str]:
    """Return TSX classes that have no corresponding CSS definition."""
    css_set = set(css_classes)

    # Framework/utility classes to skip
    _FRAMEWORK = {
        "active", "selected", "disabled", "hidden", "visible", "open",
        "closed", "loading", "error", "success", "warning", "flex",
        "grid", "block", "inline", "relative", "absolute", "sr-only",
        "container", "wrapper", "row", "col", "btn", "icon", "text",
        "link", "badge", "card", "list", "item",
    }

    missing = []
    for cls in tsx_classes:
        if cls in css_set:
            continue
        if cls in _FRAMEWORK:
            continue
        if not any(cls.startswith(p) for p in _PROJECT_CLASS_PREFIXES):
            continue
        missing.append(cls)

    return missing


def _build_css_fix_prompt(
    missing_classes: List[str],
    existing_css: str,
    tsx_context: Dict[str, str],
) -> str:
    """
    Build a prompt for the LLM to regenerate CSS with all required classes.

    Args:
        missing_classes: Classes used in TSX but not defined in CSS.
        existing_css: The current CSS content.
        tsx_context: Dict of filename -> TSX content for context.
    """
    tsx_snippets = []
    for fname, content in tsx_context.items():
        tsx_snippets.append(f"--- {fname} ---\n{content}")
    tsx_block = "\n\n".join(tsx_snippets)

    return f"""You are fixing a CSS file that has class name mismatches with its React components.

## THE PROBLEM
The following {len(missing_classes)} CSS classes are used in the TSX components but are NOT defined in the CSS file:

{chr(10).join(f"  .{cls}" for cls in missing_classes)}

## THE COMPONENTS (read these to understand what each class does)

{tsx_block}

## THE CURRENT CSS

{existing_css}

## YOUR TASK

Rewrite the COMPLETE CSS file so that:
1. Every class used in the TSX components has a corresponding CSS rule
2. The styles match the dark theme using CSS variables (var(--bg-primary), var(--text-primary), etc.)
3. The layout follows what the component structure implies (grid, flex, etc.)
4. Keep all existing styles that are correctly matched
5. Output ONLY the CSS file content, no markdown fences, no explanations

CRITICAL: Output the raw CSS only. No \`\`\`css blocks. No commentary."""

# --- LLM call for CSS regeneration --------------------------------------- #

def _call_llm_for_css_fix(
    prompt: str,
    emit: Callable,
) -> Optional[str]:
    """
    Call the LLM to regenerate CSS with all required classes.
    Uses the same model routing as the implementer.
    """
    try:
        from app.llm.stage_models import get_model_for_stage
        model_id = get_model_for_stage("IMPLEMENTER")
    except Exception:
        model_id = "anthropic/claude-sonnet-4-6"

    logger.info("[css_fixer] Calling LLM for CSS fix: model=%s", model_id)
    emit(f"  \U0001f3a8 Regenerating CSS via {model_id}...")

    try:
        from app.llm.routing.core import call_llm, LLMTask
        from app.llm.routing.types import JobType

        task = LLMTask(
            messages=[
                {"role": "user", "content": prompt},
            ],
            system_prompt="You are a CSS expert. Output only valid CSS. No markdown fences, no commentary.",
            job_type=JobType.CODE_MEDIUM,
            model=str(model_id),
        )
        result = call_llm(task)
        css_text = (result.content or "").strip()

        # Strip any accidental markdown fences
        if css_text.startswith("```"):
            lines = css_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            css_text = "\n".join(lines)

        if not css_text or len(css_text) < 100:
            logger.error("[css_fixer] LLM returned empty/too-short CSS (%d chars)", len(css_text))
            return None

        return css_text

    except Exception as e:
        logger.exception("[css_fixer] LLM call failed: %s", e)
        emit(f"  \u26a0\ufe0f CSS fix LLM call failed: {e}")
        return None


# --- Main entry point ----------------------------------------------------- #

def run_css_fix_loop(
    job_dir: str,
    frontend_root: str = r"D:\orb-desktop",
    emit: Callable = None,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """
    Post-implementation CSS fix loop.

    Detects CSS class mismatches between TSX and CSS, then re-generates
    the CSS file(s) with the full class inventory as context.

    Args:
        job_dir: Path to the segment job directory.
        frontend_root: Path to frontend project root.
        emit: Progress callback.
        max_attempts: Maximum fix attempts (default 2).

    Returns:
        Dict with: passed, attempts, fixed_files, remaining_mismatches
    """
    import json
    import os

    if emit is None:
        emit = lambda msg: None

    result = {
        "passed": False,
        "attempts": 0,
        "fixed_files": [],
        "remaining_mismatches": [],
    }

    # Load manifest to get segment files
    manifest_path = os.path.join(job_dir, "segments", "manifest.json")
    if not os.path.isfile(manifest_path):
        logger.warning("[css_fixer] No manifest at %s", manifest_path)
        emit("  ⚠️ CSS fix skipped: no manifest found")
        result["passed"] = True
        return result

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest_data = json.load(fh)

    # Build segment_files map: {seg_id: [rel_paths]}
    segment_files = {}
    for seg in manifest_data.get("segments", []):
        seg_id = seg.get("id", "")
        files = seg.get("file_scope", [])
        if files:
            segment_files[seg_id] = files

    if not segment_files:
        emit("  ⚠️ CSS fix skipped: no segment files")
        result["passed"] = True
        return result

    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt
        emit(f"\n  🔍 CSS cohesion check (attempt {attempt}/{max_attempts})...")

        # Step 1: Collect classes from sandbox
        tsx_classes, css_classes, css_files = _collect_class_inventory(
            segment_files, frontend_root,
        )

        logger.info(
            "[css_fixer] Attempt %d: %d TSX classes, %d CSS classes, %d CSS files",
            attempt, len(tsx_classes), len(css_classes), len(css_files),
        )

        # Step 2: Compute mismatches
        missing = _compute_mismatches(tsx_classes, css_classes)

        if not missing:
            emit(f"  ✅ CSS cohesion PASSED — all {len(tsx_classes)} TSX classes defined in CSS")
            result["passed"] = True
            result["remaining_mismatches"] = []
            return result

        emit(
            f"  ⚠️ {len(missing)} CSS class(es) missing: "
            f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"
        )
        logger.warning("[css_fixer] Missing %d classes: %s", len(missing), missing)

        if not css_files:
            emit("  ⚠️ No CSS files found in segments — cannot fix")
            result["remaining_mismatches"] = missing
            return result

        # Step 3: Gather TSX context for the prompt
        tsx_context = {}
        for seg_id, files in segment_files.items():
            for rel_path in files:
                if not rel_path.endswith((".tsx", ".ts")):
                    continue
                if rel_path.startswith("src"):
                    abs_path = f"{frontend_root}\\{rel_path.replace('/', chr(92))}"
                elif rel_path.startswith("orb-desktop"):
                    stripped = rel_path[len("orb-desktop/"):]
                    abs_path = f"{frontend_root}\\{stripped.replace('/', chr(92))}"
                else:
                    abs_path = rel_path
                content = _read_sandbox_file(abs_path)
                if content:
                    tsx_context[rel_path] = content

        # Step 4: For each CSS file, regenerate
        for css_path in css_files:
            existing_css = _read_sandbox_file(css_path)
            if existing_css is None:
                continue

            prompt = _build_css_fix_prompt(missing, existing_css, tsx_context)
            new_css = _call_llm_for_css_fix(prompt, emit)

            if new_css is None:
                emit(f"  ⚠️ Failed to regenerate {css_path}")
                continue

            # Validate the new CSS has the missing classes
            new_css_classes = set(_extract_css_classes(new_css))
            fixed_count = sum(1 for c in missing if c in new_css_classes)
            still_missing = [c for c in missing if c not in new_css_classes]

            logger.info(
                "[css_fixer] Regenerated %s: fixed %d/%d missing classes, %d still missing",
                css_path, fixed_count, len(missing), len(still_missing),
            )

            if fixed_count == 0:
                emit(f"  ⚠️ Regenerated CSS didn't fix any mismatches — skipping write")
                continue

            # Write the fixed CSS
            if _write_sandbox_file(css_path, new_css):
                emit(
                    f"  ✅ Wrote fixed CSS: {css_path} "
                    f"({fixed_count}/{len(missing)} classes fixed, {len(new_css)} chars)"
                )
                result["fixed_files"].append(css_path)
            else:
                emit(f"  ⚠️ Failed to write fixed CSS to {css_path}")

        # If we fixed all of them, next iteration will confirm
        result["remaining_mismatches"] = missing

    # Final check after all attempts
    tsx_final, css_final, _ = _collect_class_inventory(segment_files, frontend_root)
    final_missing = _compute_mismatches(tsx_final, css_final)

    if not final_missing:
        emit(f"  ✅ CSS cohesion PASSED after {result['attempts']} attempt(s)")
        result["passed"] = True
        result["remaining_mismatches"] = []
    else:
        emit(
            f"  ⚠️ CSS cohesion FAILED — {len(final_missing)} class(es) still missing "
            f"after {result['attempts']} attempt(s)"
        )
        result["remaining_mismatches"] = final_missing

    return result