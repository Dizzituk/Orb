# FILE: app/pipeline_v2/stages/builder.py
# Purpose: Stage 4 — The Builder.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.agentic_pipeline, app.agentic_pipeline.loop_controller, app.overwatcher.architecture_executor, app.pipeline_v2 (+4 more)
# Last-renovated: 2026-06-11
"""
Stage 4 — The Builder.

Receives one tier at a time from the Architect's build plan.
For each tier:
  1. Build context with grounding brief (real files from prior tiers)
  2. ONE LLM call — fill template gaps with complete production code
  3. Extract code deterministically
  4. Write files to sandbox
  5. Check own work (syntax, imports)
  6. Fix anything broken before moving on

Uses Claude Opus 4.6 for code generation.
Has full tool access: sandbox read, write, shell execution.

v1.0 (2026-03-06): Initial implementation for ASTRA v2.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from app.pipeline_v2.config import (
    BUILDER_PROVIDER, BUILDER_MODEL, BUILDER_MAX_OUTPUT,
)
from app.pipeline_v2.models import (
    BuildPlan, BuildResult, TierBuildResult, LaidFile,
)

logger = logging.getLogger(__name__)

BUILDER_SYSTEM = """You are ASTRA's Builder. You write production-ready code.

You receive a tier of segments with deterministic templates that have
[LLM_FILL_N] markers. You fill the gaps with complete, working code.

IMPORTANT — CREATE vs MODIFY:
- Files marked [CREATE] → you write the COMPLETE file.
- Files marked [MODIFY] → DO NOT write code for these. They are handled
  separately by the Surgical Editor which reads the real file and applies
  targeted edits. Skip them in your output.

OUTPUT FORMAT — MANDATORY:

For EVERY segment and EVERY [LLM_FILL_N] marker, output:

=== FILL: <segment_id> / <fill_number> ===
<your content>
=== END FILL ===

FILL RULES:

[LLM_FILL_1] = Architecture vision (1-2 paragraphs).

[LLM_FILL_2] = COMPLETE code for every [CREATE] file in the segment.
  Write a ### heading with the file path, then a fenced code block.
  FIRST LINE of every code block MUST be:
    # FILE: app/example.py       (Python)
    // FILE: src/Example.tsx      (TypeScript/JS)
  Code must be COMPLETE. No placeholders. No TODOs.
  DO NOT include [MODIFY] files — they are edited separately.

[LLM_FILL_3] = Brief notes (optional).

WORKMANSHIP:
- Check your own work. If an import path is wrong, fix it.
- If a type doesn't match, fix it.
- Use exact names from skeleton contracts for cross-segment imports.
- Reference the GROUNDED TRUTH section — those files are REAL and exist.
- Each [CREATE] file must be self-contained and complete.
- Never put two files in one code block.

EXAMPLE (correct [LLM_FILL_2]):

### `app/debug/models.py`

```python
# FILE: app/debug/models.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.db import Base

class DebugProject(Base):
    __tablename__ = "debug_projects"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
```

CRITICAL:
- Output ALL fills for ALL segments. Skipping fills = build failure.
- Every [CREATE] file needs its own separate fenced code block.
- NEVER combine multiple files into one code block.
- NEVER output code for [MODIFY] files.
"""


async def run_builder(
    build_plan: BuildPlan,
    manifest: Dict[str, Any],
    preflight_result: Any,
    job_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> BuildResult:
    """Run the Builder across all tiers in the build plan."""
    result = BuildResult()
    t_start = time.time()
    emit = on_progress or (lambda msg: None)

    segments_data = manifest.get("segments", [])
    seg_by_id = {s.get("segment_id", ""): s for s in segments_data}

    # Grounded truth — accumulates as tiers complete
    laid_files: Dict[str, str] = {}
    prior_summaries: Dict[str, str] = {}

    from app.pipeline_v2.llm_caller import call_llm
    from app.agentic_pipeline.loop_controller import parse_fills, splice_fills_into_templates
    from app.agentic_pipeline.tier_context import build_tier_context

    for layer in build_plan.layers:
        emit(f"\n{'='*50}")
        emit(f"🔨 BUILDER — Tier {layer.tier}: {len(layer.segment_ids)} segment(s), "
             f"{len(layer.file_scope)} file(s)")
        emit(f"{'='*50}")

        tier_result = TierBuildResult(tier=layer.tier)
        t_tier = time.time()

        # Gather this tier's segment specs
        tier_segments = [seg_by_id[sid] for sid in layer.segment_ids if sid in seg_by_id]

        # Build context with grounding brief
        context = build_tier_context(
            tier_segments=tier_segments,
            numbered_templates=build_plan.templates,
            skeleton_contract=build_plan.skeleton_contract,
            preflight_result=preflight_result,
            job_dir=job_dir,
            laid_files=laid_files,
            prior_arch_summaries=prior_summaries,
        )

        # Count fills for this tier
        total_fills = 0
        checklist_parts = []
        for seg in tier_segments:
            sid = seg.get("segment_id", "")
            tmpl = build_plan.templates.get(sid, "")
            seg_fills = tmpl.count("[LLM_FILL_")
            total_fills += seg_fills
            checklist_parts.append(f"  {sid}: fills 1..{seg_fills}")
        checklist = "\n".join(checklist_parts)

        emit(f"   Fills: {total_fills} | Context: {len(context):,} chars")
        emit(f"   🤖 Calling {BUILDER_PROVIDER}/{BUILDER_MODEL}...")

        # Classify files for the LLM prompt
        from app.pipeline_v2.agentic_editor import classify_file_operation
        create_file_list = []
        modify_file_list = []
        for seg in tier_segments:
            for fp in seg.get("file_scope", []):
                op = classify_file_operation(fp, preflight_result)
                if op == "modify":
                    modify_file_list.append(fp)
                else:
                    create_file_list.append(fp)
        if modify_file_list:
            emit(f"   ✂️ MODIFY files ({len(modify_file_list)}): {', '.join(modify_file_list[:5])}")
        emit(f"   📦 CREATE files ({len(create_file_list)}): {', '.join(create_file_list[:5])}{'...' if len(create_file_list) > 5 else ''}")
        logger.info("[builder] Tier %d: %d CREATE, %d MODIFY", layer.tier, len(create_file_list), len(modify_file_list))

        file_labels = ""
        if create_file_list:
            file_labels += "[CREATE] files (write complete code):\n"
            file_labels += "\n".join(f"  - {f}" for f in create_file_list)
        if modify_file_list:
            file_labels += "\n[MODIFY] files (DO NOT write — handled by Surgical Editor):\n"
            file_labels += "\n".join(f"  - {f}" for f in modify_file_list)

        # ONE LLM call for the tier (CREATE files only)
        if total_fills > 0:
            user_prompt = (
                f"Fill ALL {total_fills} [LLM_FILL_N] markers for {len(tier_segments)} segment(s).\n\n"
                f"FILE CLASSIFICATION:\n{file_labels}\n\n"
                f"CHECKLIST — output every one:\n{checklist}\n\n"
                f"[LLM_FILL_2]: write code ONLY for [CREATE] files. "
                f"Each file gets its own fenced code block starting with # FILE: or // FILE:.\n"
                f"Do NOT write code for [MODIFY] files.\n\n"
                f"{context}"
            )
        else:
            user_prompt = (
                f"Write COMPLETE production-ready code for {len(tier_segments)} segment(s).\n\n"
                f"FILE CLASSIFICATION:\n{file_labels}\n\n"
                f"Write code ONLY for [CREATE] files. Skip [MODIFY] files.\n"
                f"Each file gets its own fenced code block with # FILE: or // FILE: as first line.\n"
                f"Code must be COMPLETE. No placeholders. No TODOs.\n\n"
                f"{context}"
            )

        try:
            raw_output = await call_llm(
                provider=BUILDER_PROVIDER,
                model=BUILDER_MODEL,
                system_prompt=BUILDER_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=BUILDER_MAX_OUTPUT,
            )
        except RuntimeError as e:
            emit(f"   ❌ LLM call failed: {e}")
            tier_result.error = str(e)
            result.tier_results.append(tier_result)
            result.errors.append(f"Tier {layer.tier}: {e}")
            continue

        tier_result.llm_calls = 1
        result.total_llm_calls += 1
        emit(f"   📝 Got {len(raw_output)} chars")

        # Parse fills and splice into templates
        fills = parse_fills(raw_output)
        emit(f"   Parsed {len(fills)} fill(s) (expected {total_fills})")

        tier_templates = {
            sid: build_plan.templates.get(sid, "")
            for sid in layer.segment_ids
        }

        # Check if we have templates to splice into
        has_templates = any(t.strip() for t in tier_templates.values())

        if has_templates and fills:
            arch_docs = splice_fills_into_templates(tier_templates, fills)
        elif fills:
            # Fills but no templates — splice into empty templates still works
            arch_docs = splice_fills_into_templates(tier_templates, fills)
        else:
            # No fills parsed — try segment-delimited or use raw output
            from app.agentic_pipeline.loop_parser import parse_agentic_output
            parsed = parse_agentic_output(raw_output, layer.segment_ids)
            arch_docs = parsed.get_all_arch_docs()
            if not arch_docs and len(layer.segment_ids) == 1:
                # Single segment, no delimiters — raw output IS the arch doc
                arch_docs = {layer.segment_ids[0]: raw_output}

        tier_result.arch_docs = arch_docs
        emit(f"   Spliced {len(arch_docs)} arch doc(s)")

        # Extract, write, verify per segment
        from app.pipeline_v2.agentic_editor import apply_surgical_edits

        for seg in tier_segments:
            seg_id = seg.get("segment_id", "")
            file_scope = seg.get("file_scope", [])
            arch_doc = arch_docs.get(seg_id, "")

            if not arch_doc:
                emit(f"   ⚠️ {seg_id}: No arch doc")
                tier_result.files_failed.extend(file_scope)
                continue

            # Save arch doc
            _save_arch_doc(seg_id, arch_doc, job_dir)

            # Split files into CREATE vs MODIFY (reuse the classification we already did)
            # Normalise paths for comparison — file_scope may use \ while lists use /
            create_set = {f.replace('\\', '/').lower() for f in create_file_list}
            modify_set = {f.replace('\\', '/').lower() for f in modify_file_list}
            create_files = [fp for fp in file_scope if fp.replace('\\', '/').lower() in create_set]
            modify_files = [fp for fp in file_scope if fp.replace('\\', '/').lower() in modify_set]

            # --- CREATE files: extract from arch doc and write ---
            if create_files:
                emit(f"   📦 {seg_id}: Extracting {len(create_files)} CREATE file(s)...")
                written, failed = await _extract_and_write(arch_doc, create_files, emit)
                for fp in written:
                    tier_result.files_written.append(LaidFile(path=fp))
                    laid_files[fp] = f"Laid by {seg_id} (tier {layer.tier})"
                tier_result.files_failed.extend(failed)
                result.all_files_written.extend(written)

            # --- MODIFY files: surgical agentic edits ---
            if modify_files:
                emit(f"   ✂️ {seg_id}: Editing {len(modify_files)} MODIFY file(s)...")
                for fp in modify_files:
                    # Build edit instructions from the arch doc
                    edit_instructions = _extract_modify_instructions(arch_doc, fp, seg, all_create_files=create_file_list)
                    ok, err = await apply_surgical_edits(
                        file_path=fp,
                        edit_instructions=edit_instructions,
                        segment_context=context[:8000],  # Trim context to keep prompt manageable
                        emit=emit,
                    )
                    if ok:
                        tier_result.files_written.append(LaidFile(path=fp))
                        laid_files[fp] = f"Edited by {seg_id} (tier {layer.tier})"
                        result.all_files_written.append(fp)
                    else:
                        # Fallback: try extracting from arch doc like a CREATE
                        emit(f"      🔄 Falling back to full extraction for {fp}")
                        w, f = await _extract_and_write(arch_doc, [fp], emit)
                        if w:
                            tier_result.files_written.append(LaidFile(path=fp))
                            laid_files[fp] = f"Laid (fallback) by {seg_id} (tier {layer.tier})"
                            result.all_files_written.extend(w)
                        else:
                            tier_result.files_failed.append(fp)

            written_this_seg = [lf.path for lf in tier_result.files_written if laid_files.get(lf.path, "").endswith(f"(tier {layer.tier})")]
            if written_this_seg:
                file_list = ", ".join(f"`{f}`" for f in written_this_seg[:4])
                prior_summaries[seg_id] = f"Laid {len(written_this_seg)} files: {file_list}"

        # Verify the tier
        if tier_result.files_written:
            emit(f"   🔍 Verifying tier {layer.tier}...")
            await _verify_tier(tier_result, emit)

        tier_result.duration_seconds = time.time() - t_tier
        tier_result.success = len(tier_result.files_written) > 0 and not tier_result.files_failed
        result.tier_results.append(tier_result)

        emit(f"   ⏱️ Tier {layer.tier}: {len(tier_result.files_written)} written, "
             f"{len(tier_result.files_failed)} failed ({tier_result.duration_seconds:.1f}s)")

    result.total_duration_seconds = time.time() - t_start
    result.success = all(t.success for t in result.tier_results) if result.tier_results else False

    emit(f"\n🔨 Builder complete: {len(result.all_files_written)} files, "
         f"{result.total_llm_calls} LLM calls, {result.total_duration_seconds:.1f}s")

    return result


def _save_arch_doc(seg_id: str, arch_doc: str, job_dir: str) -> None:
    """Save arch doc to disk."""
    seg_dir = os.path.join(job_dir, "segments", seg_id, "arch")
    os.makedirs(seg_dir, exist_ok=True)
    path = os.path.join(seg_dir, "arch_v1.md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(arch_doc)
    except Exception as e:
        logger.error("[builder] Failed to save arch doc %s: %s", path, e)


async def _extract_and_write(
    arch_doc: str, file_scope: List[str], emit: Callable,
) -> tuple:
    """Extract code and write to sandbox. Returns (written, failed)."""
    written, failed = [], []
    try:
        from app.overwatcher.architecture_executor.arch_code_extractor import extract_code_for_files
    except ImportError:
        return written, file_scope

    extraction = extract_code_for_files(arch_doc, file_scope)

    from app.pipeline_v2.sandbox_tools import write_file
    for fp in file_scope:
        content = extraction.get_content_for_file(fp)
        if not content:
            failed.append(fp)
            emit(f"      ⚠️ No code extracted: {fp}")
            continue
        if await write_file(fp, content):
            written.append(fp)
            emit(f"      ✅ {fp}")
        else:
            failed.append(fp)
            emit(f"      ❌ Write failed: {fp}")
    return written, failed


def _extract_modify_instructions(
    arch_doc: str,
    file_path: str,
    segment: Dict,
    all_create_files: Optional[List[str]] = None,
) -> str:
    """Build concrete edit instructions for a MODIFY file.

    Instead of vague "modify as described", this produces specific instructions:
    - What imports to add
    - What routes/components to register
    - What the new CREATE files export (so the editor knows what to wire in)

    The agentic editor receives these instructions + the real file content
    and produces targeted search/replace blocks.
    """
    instructions = []
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    norm_path = file_path.replace("\\", "/")

    # 1. Segment requirements
    reqs = segment.get("requirements", [])
    if reqs:
        instructions.append("REQUIREMENTS:")
        for r in reqs:
            instructions.append(f"  - {r}")

    # 2. Extract file-specific content from arch doc
    lines = arch_doc.split("\n")
    capturing = False
    file_section = []
    for line in lines:
        if basename in line or norm_path in line:
            capturing = True
            file_section.append(line)
            continue
        if capturing:
            if (line.startswith("### ") or line.startswith("## ") or "FILE:" in line):
                if basename not in line and norm_path not in line:
                    capturing = False
                    continue
            file_section.append(line)
    if file_section:
        instructions.append("\nARCHITECTURE NOTES:")
        instructions.extend(file_section)

    # 3. Context about CREATE files in this segment (what to wire in)
    create_files = all_create_files or []
    seg_creates = [f for f in segment.get("file_scope", []) if f in create_files]
    if seg_creates:
        instructions.append("\nNEW FILES CREATED IN THIS SEGMENT (wire these in):")
        for cf in seg_creates:
            instructions.append(f"  - {cf}")

    # 4. Specific guidance based on the file being modified
    if basename == "main.py":
        instructions.append("\nSPECIFIC EDITS NEEDED FOR main.py:")
        instructions.append("  - Add import for the new debug project router")
        instructions.append("  - Register it with app.include_router() in the startup section")
        instructions.append("  - Follow the EXACT pattern used by other routers (try/except ImportError)")
        instructions.append("  - Place it right after the existing debug_chat_router registration")
        instructions.append("  - DO NOT rewrite the file. Only add the import and include_router lines.")
    elif basename in ("App.tsx",):
        instructions.append("\nSPECIFIC EDITS NEEDED FOR App.tsx:")
        instructions.append("  - Only add imports/routes needed for the new debug feature")
        instructions.append("  - DO NOT rewrite existing routes or components")
    elif basename == "DebugView.tsx":
        instructions.append("\nSPECIFIC EDITS NEEDED FOR DebugView.tsx:")
        instructions.append("  - Add useState for activeProjectId (number | null)")
        instructions.append("  - Add import for DebugProjectGrid and DebugWorkspace")
        instructions.append("  - Wrap the existing chat view: if activeProjectId is null, show DebugProjectGrid")
        instructions.append("  - If activeProjectId is set, show DebugWorkspace with a back button")
        instructions.append("  - Keep the existing DebugView function body intact for when a project is selected")
    elif basename == "debug_chat.py":
        instructions.append("\nSPECIFIC EDITS NEEDED FOR debug_chat.py:")
        instructions.append("  - Add project_id parameter support to the chat endpoint")
        instructions.append("  - Import and use the new crud module for message persistence")
        instructions.append("  - DO NOT rewrite the entire file. Only add the new parameters and imports.")
    elif basename == "context_assembler.py":
        instructions.append("\nSPECIFIC EDITS NEEDED FOR context_assembler.py:")
        instructions.append("  - Add a cross_tab_events section to the context assembly")
        instructions.append("  - Import from the context_lock module")
        instructions.append("  - DO NOT rewrite existing collector logic")

    if not instructions:
        instructions.append(f"Modify {file_path} to integrate the new debug project features.")
        instructions.append("Make minimal, targeted changes. Do not rewrite the file.")

    return "\n".join(instructions)


async def _verify_tier(tier_result: TierBuildResult, emit: Callable) -> None:
    """Verify all files in the tier. Updates LaidFile.syntax_ok."""
    from app.pipeline_v2.sandbox_tools import check_python_syntax
    for laid in tier_result.files_written:
        if laid.path.endswith(".py"):
            ok, err = await check_python_syntax(laid.path)
            laid.syntax_ok = ok
            if not ok:
                laid.error = err
                emit(f"      ⚠️ Syntax issue: {laid.path} — {err}")
            else:
                emit(f"      ✓ {laid.path} syntax OK")
