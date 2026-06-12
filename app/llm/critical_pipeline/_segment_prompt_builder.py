# FILE: app/llm/critical_pipeline/_segment_prompt_builder.py
# Purpose: Build segment-scoped prompt injection for architecture generation.
# Called-by: app.llm.critical_pipeline.stream_handler
# Depends-on: app.pot_spec.pattern_cache, app.sandbox_fs
# Last-renovated: 2026-06-11
"""
Build segment-scoped prompt injection for architecture generation.

Extracted from stream_handler.py _handle_architecture() — the large
block that constructs segment scope, enrichment, source evidence,
sibling exports, cohesion feedback, and import validation feedback
into a single prompt injection string.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.sandbox_fs import sandbox_isfile, sandbox_read_text

logger = logging.getLogger(__name__)


def build_segment_injection(
    segment_context: dict,
    job_id: str,
) -> str:
    """Build the segment-scoped prompt injection from segment_context.

    This constructs all segment-related sections:
    - Segment scope (file list with CREATE/MODIFY/READ-ONLY markers)
    - Requirements and acceptance criteria
    - Interface contract
    - Sibling interfaces (AST-extracted)
    - Source file evidence
    - Stage 4B enrichment (constants, functions, classes, cross-segment contracts)
    - Duplicate function prohibition
    - Upstream evidence
    - Cohesion regen feedback
    - Import validation feedback
    - Implementation failure feedback
    - Complete sibling export map

    Args:
        segment_context: Dict with segment metadata from the segment loop
        job_id: Job ID (may be sub-job ID like "sg-xxx__seg-04-...")

    Returns:
        Joined prompt injection string, or "" if segment_context is empty
    """
    if not segment_context:
        return ""

    _si_parts: List[str] = []

    _si_files = segment_context.get("file_scope", [])
    _si_reqs = segment_context.get("requirements", [])
    _si_seg_id = segment_context.get("segment_id", "unknown")
    _si_ac = segment_context.get("acceptance_criteria", [])

    _si_parts.append(f"## Segment Scope: {_si_seg_id}\n")
    _si_parts.append(
        "**IMPORTANT**: You are generating architecture for ONE SEGMENT "
        "of a multi-segment job, not the full specification. Only design "
        "and produce code for the files listed below. "
        "Files marked CREATE do not exist on disk yet — do NOT emit "
        "EVIDENCE_REQUEST to read them. All source code you need is provided "
        "in the Source File Evidence and Segment Enrichment sections below.\n\n"
        "**NO-CHANGES ESCAPE HATCH**: If after reviewing the source evidence "
        "you determine that ALL files in this segment ALREADY support the "
        "feature without any code changes (e.g. existing generic routing "
        "already handles the new case), you MUST output ONLY the following "
        "instead of a full architecture document:\n\n"
        "```\n"
        "SPEC_ID: <spec_id>\n"
        "SPEC_HASH: <spec_hash>\n\n"
        "## NO_CHANGES_NEEDED\n\n"
        "Reason: <one-line explanation of why no changes are required>\n"
        "```\n\n"
        "Do NOT reproduce files verbatim with zero changes. That wastes "
        "tokens and risks regressions. If the existing code already works, "
        "say so and move on.\n"
    )

    # --- File scope with CREATE/MODIFY/READ-ONLY markers ---
    if _si_files:
        _si_parts.append("### Files in Scope (ONLY these files)")
        _si_source_evidence = segment_context.get("source_file_evidence", {})
        _si_source_evidence_norm = {
            k.replace("\\", "/"): v for k, v in _si_source_evidence.items()
        }
        for _f in _si_files:
            _f_norm = _f.replace("\\", "/")
            _exists = _f_norm in _si_source_evidence_norm or any(
                k == _f_norm for k in _si_source_evidence_norm
            )
            if _exists:
                _f_stem = _f_norm.rsplit(".py", 1)[0]
                _is_monolith_source = any(
                    other_f.replace("\\", "/").startswith(_f_stem + "/")
                    for other_f in _si_files if other_f != _f
                )
                if _is_monolith_source:
                    _si_parts.append(
                        f"- `{_f}` — **READ-ONLY** (source monolith being refactored — "
                        f"provided as evidence only. Do NOT include in File Inventory. "
                        f"Do NOT generate MODIFY operations for this file.)"
                    )
                else:
                    _si_parts.append(f"- `{_f}` — **MODIFY** (exists on disk)")
            else:
                _si_parts.append(f"- `{_f}` — **CREATE** (new file — do NOT try to read)")
        _si_parts.append("")

    if _si_reqs:
        _si_parts.append("### Segment Requirements")
        for _r in _si_reqs[:15]:
            _si_parts.append(f"- {_r}")
        if len(_si_reqs) > 15:
            _si_parts.append(f"- ... (+{len(_si_reqs)-15} more)")
        _si_parts.append("")

    if _si_ac:
        _si_parts.append("### Acceptance Criteria")
        for _a in _si_ac[:10]:
            _si_parts.append(f"- {_a}")
        if len(_si_ac) > 10:
            _si_parts.append(f"- ... (+{len(_si_ac)-10} more)")
        _si_parts.append("")

    # Interface contract
    _si_contract = segment_context.get("interface_contract", "")
    if _si_contract:
        _si_parts.append(_si_contract)

    # Deterministic sibling interface evidence
    _si_sibling_ifaces = segment_context.get("sibling_interfaces", "")
    if _si_sibling_ifaces:
        _si_parts.append(_si_sibling_ifaces)
        logger.info(
            "[segment_prompt] Injected %d chars of deterministic sibling interface evidence",
            len(_si_sibling_ifaces),
        )

    # v1.0: Upstream architecture evidence (for deps that are approved but not complete)
    _si_upstream_arch = segment_context.get("upstream_architecture_evidence", "")
    if _si_upstream_arch:
        _si_parts.append(_si_upstream_arch)
        logger.info(
            "[segment_prompt] v1.0 Injected %d chars of upstream architecture evidence",
            len(_si_upstream_arch),
        )

    # Source file evidence
    _si_source_files = segment_context.get("source_file_evidence", {})
    if _si_source_files:
        _si_parts.append("### Source File Evidence (EXISTING code — copy verbatim)\n")
        _si_parts.append(
            "**CRITICAL**: The following file(s) exist on disk and are being "
            "refactored. You MUST copy all function signatures, constant values, "
            "parameter names, and return types EXACTLY as they appear below. "
            "Do NOT invent, guess, or approximate any values.\n"
        )
        _si_parts.append(
            "**ENUM/TYPE PRESERVATION**: If the source code uses an enum like "
            "`SegmentStatus.COMPLETE.value` or `SegmentStatus.FAILED`, you MUST "
            "use that same enum in your output. Do NOT replace enums with invented "
            "string constants.\n"
        )

        # Source evidence ownership warning (v5.34)
        _si_other_seg_symbols = set()
        _enrich_for_ownership = segment_context.get("enrichment", {})
        _consumes_for_ownership = _enrich_for_ownership.get("consumes", {}) if _enrich_for_ownership else {}
        for _own_seg, _own_syms in _consumes_for_ownership.items():
            if isinstance(_own_syms, list):
                _si_other_seg_symbols.update(_own_syms)
        if _si_other_seg_symbols:
            _si_parts.append("#### ⚠️ Source Evidence Ownership Warning (v5.34)\n")
            _si_parts.append(
                "The source file below contains functions that belong to "
                "**OTHER segments**, not yours. When you see these functions "
                "in the source code, do NOT copy their bodies into your file. "
                "Instead, IMPORT them from the upstream segment module.\n"
            )
            _si_parts.append("**Functions in source that you must IMPORT, not copy:**")
            for _own_sym in sorted(_si_other_seg_symbols):
                _si_parts.append(f"  - `{_own_sym}` — IMPORT this, do NOT copy its body")
            _si_parts.append("")

        for _sf_path, _sf_content in _si_source_files.items():
            _si_parts.append(f"**`{_sf_path}`** ({len(_sf_content):,} chars)")
            _sf_inject = _sf_content[:120_000]
            if len(_sf_content) > 120_000:
                _sf_inject += f"\n... (truncated from {len(_sf_content):,} chars)"
            _si_parts.append(f"```python\n{_sf_inject}\n```\n")

    # Stage 4B Segment Enrichment
    _enrichment = segment_context.get("enrichment")
    if _enrichment:
        _build_enrichment_section(_si_parts, _enrichment)

    # Upstream evidence
    _si_evidence = segment_context.get("evidence", [])
    if _si_evidence:
        _build_upstream_evidence_section(_si_parts, _si_evidence)

        # v4.3: Evidence Ledger — decisions, constraints, facts from earlier stages
    _si_ledger = segment_context.get("ledger_evidence", "")
    if _si_ledger:
        _si_parts.append(_si_ledger)
        logger.info(
            "[segment_prompt] Injected %d chars of evidence ledger",
            len(_si_ledger),
        )

    # Cohesion regen feedback
    _si_cohesion = segment_context.get("cohesion_issues", "")
    if _si_cohesion:
        _si_parts.append("### ⚠️ COHESION ISSUES (from previous architecture attempt)\n")
        _si_parts.append("The previous architecture for this segment had cross-segment compatibility issues.")
        _si_parts.append("You MUST fix these issues in this regeneration:\n")
        _si_parts.append(f"{_si_cohesion}\n")
        _si_parts.append("Ensure all import names, module names, and function signatures match what other segments expect.\n")

    # Import validation failure feedback (v5.38)
    _si_import_feedback = segment_context.get("import_validation_feedback", "")
    if _si_import_feedback:
        _si_parts.append("### ❌ IMPORT VALIDATION FAILED (deterministic check)\n")
        _si_parts.append("The previous architecture for this segment contained cross-segment imports")
        _si_parts.append(" that reference symbols which DO NOT EXIST in the target modules.")
        _si_parts.append(" You MUST fix every violation listed below.\n")
        _si_parts.append(f"{_si_import_feedback}\n")

    # Implementation failure feedback (v5.25)
    _si_impl_feedback = segment_context.get("implementation_feedback", "")
    if _si_impl_feedback:
        _si_parts.append("### PREVIOUS IMPLEMENTATION FAILED\n")
        _si_parts.append("The previous architecture for this segment was implemented but the Implementer")
        _si_parts.append("failed to produce working code. The specific failure reasons were:\n")
        _si_parts.append(f"{_si_impl_feedback}\n")
        _si_parts.append("DO NOT repeat the design patterns that caused these failures.\n")

    # Complete sibling export map (v5.36)
    # v2.0: Codebase pattern reference for CREATE-only segments.
    # When a segment has no existing source files, find an existing sibling
    # component to use as a pattern reference (e.g. InvestmentsView for
    # EducationView).  This gives the architecture LLM real codebase
    # patterns: icon usage, API fetch style, loading states, CSS classes.
    _inject_codebase_pattern_reference(_si_parts, segment_context)

    _build_sibling_export_map(_si_parts, segment_context, job_id)

    return "\n".join(_si_parts)


def _build_enrichment_section(parts: List[str], enrichment: dict) -> None:
    """Build the Stage 4B enrichment section into parts list."""
    parts.append("### Segment Enrichment (Stage 4B — Grounded Evidence)\n")
    parts.append(
        "**IMPORTANT**: The following symbols were extracted from the original source "
        "file using AST parsing. You MUST include ALL of them in your architecture. "
        "However, this list may NOT be exhaustive — the source file may contain "
        "additional functions, classes, or helpers. If your segment logically needs "
        "a symbol not listed below but exists in the source file, INCLUDE it.\n"
    )

    # Constants
    _enrich_constants = enrichment.get("constants", [])
    if _enrich_constants:
        parts.append("#### Constants (MUST preserve exact names and values)")
        for _ec in _enrich_constants:
            _ec_val = _ec.get("value", "")
            if _ec_val:
                parts.append(f"```python\n{_ec_val}\n```")
            else:
                parts.append(f"- `{_ec.get('name', '?')}`")
        parts.append("")

    # Functions
    _enrich_functions = enrichment.get("functions", [])
    if _enrich_functions:
        parts.append("#### Functions (MUST preserve exact signatures)")
        parts.append(
            "Each function's line count is from AST analysis. "
            "Do NOT create extra helper files to split a function."
        )
        for _ef in _enrich_functions:
            _sig = _ef.get("signature", _ef.get("name", "?"))
            _line_range = _ef.get("line_range")
            _body = _ef.get("body", "")
            _line_count = 0
            if _line_range and len(_line_range) == 2:
                _line_count = _line_range[1] - _line_range[0] + 1
            elif _body:
                _line_count = _body.count("\n") + 1
            _is_async = _ef.get("is_async", False)
            _async_tag = " (async)" if _is_async else ""
            if _line_count:
                parts.append(f"- `{_sig}`{_async_tag} — **{_line_count} lines**")
            else:
                parts.append(f"- `{_sig}`{_async_tag}")
        parts.append("")

    # Classes
    _enrich_classes = enrichment.get("classes", [])
    if _enrich_classes:
        parts.append("#### Classes (MUST preserve)")
        for _ecl in _enrich_classes:
            parts.append(f"- `class {_ecl.get('name', '?')}` "
                         f"(methods: {', '.join(_ecl.get('methods', [])[:10])})")
        parts.append("")

    # Cross-segment contract
    _enrich_consumed_by = enrichment.get("consumed_by", {})
    if _enrich_consumed_by:
        parts.append("#### Cross-Segment Contract (other segments import these from YOU)")
        for _cb_seg, _cb_syms in _enrich_consumed_by.items():
            if isinstance(_cb_syms, list):
                parts.append(f"- **{_cb_seg}** imports: {', '.join(f'`{s}`' for s in _cb_syms)}")
        parts.append("")

    # Dependencies + duplicate prohibition
    _enrich_consumes = enrichment.get("consumes", {})
    if _enrich_consumes:
        parts.append("#### Dependencies (symbols YOU need from other segments)")
        for _cn_seg, _cn_syms in _enrich_consumes.items():
            if isinstance(_cn_syms, list):
                parts.append(f"- From **{_cn_seg}**: {', '.join(f'`{s}`' for s in _cn_syms)}")
        parts.append("")

        _all_dep_symbols = []
        for _cn_syms_list in _enrich_consumes.values():
            if isinstance(_cn_syms_list, list):
                _all_dep_symbols.extend(_cn_syms_list)
        if _all_dep_symbols:
            parts.append("#### ⛔ DUPLICATE FUNCTION PROHIBITION (v5.30)")
            parts.append(
                "The following symbols are ALREADY DEFINED in your dependency "
                "segments. You MUST import them — NEVER redefine them."
            )
            parts.append("")
            parts.append("**DO NOT define any of these in your code:**")
            for _ds in _all_dep_symbols:
                parts.append(f"  - ❌ `{_ds}` — import it, do NOT redefine it")
            parts.append("")
            parts.append(
                "If your function body needs to call a dependency symbol, "
                "write `from ._dependencies import symbol_name` (or the "
                "appropriate sibling module). NEVER copy the function body."
            )
            parts.append("")

    # Design guidance
    _enrich_guidance = enrichment.get("design_guidance", "")
    if _enrich_guidance:
        parts.append(f"#### Design Guidance\n{_enrich_guidance}\n")

    # Source size budget
    _source_extract = enrichment.get("source_extract")
    if _source_extract and isinstance(_source_extract, dict):
        _total_source_lines = sum(
            v.count("\n") + 1 for v in _source_extract.values() if isinstance(v, str)
        )
        if _total_source_lines > 0:
            parts.append(f"#### Source Size Budget")
            parts.append(
                f"Total source code being transplanted: "
                f"**{_total_source_lines} lines** across {len(_source_extract)} function(s). "
                f"Budget ~{int(_total_source_lines * 1.15)} lines total. "
                f"This FITS in a single file — do NOT split into helper submodules."
            )
            parts.append("")

    # Risk flags
    _enrich_risk = enrichment.get("risk_level", "low")
    if _enrich_risk in ("medium", "high"):
        _enrich_risk_notes = enrichment.get("risk_notes", "")
        parts.append(f"⚠️ **Risk: {_enrich_risk.upper()}** — {_enrich_risk_notes}\n")

    # Unresolved symbols
    _enrich_unresolved = enrichment.get("unresolved", [])
    if _enrich_unresolved:
        parts.append("❌ **UNRESOLVED SYMBOLS (will cause boot failure):**")
        for _eu in _enrich_unresolved:
            parts.append(f"- {_eu}")
        parts.append("")


def _build_upstream_evidence_section(parts: List[str], evidence) -> None:
    """Build upstream evidence section from completed segments."""
    parts.append("### Upstream Evidence (from completed segments)\n")
    if isinstance(evidence, dict):
        evidence = list(evidence.values()) if evidence else []
    elif not isinstance(evidence, list):
        evidence = []
    for _ev in evidence[:10]:
        if isinstance(_ev, dict):
            _ev_path = _ev.get("file_path", _ev.get("path", ""))
            _ev_sigs = _ev.get("signatures", [])
            parts.append(f"**{_ev_path}**")
            for _sig in _ev_sigs[:5]:
                parts.append(f"  - `{_sig}`")
        elif isinstance(_ev, str):
            parts.append(f"- {_ev}")
    parts.append("")


def _build_sibling_export_map(
    parts: List[str],
    segment_context: dict,
    job_id: str,
) -> None:
    """Build complete sibling export map (v5.36)."""
    try:
        _job_artifact_root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs")
        _parent_job_id = job_id.split("__")[0] if "__" in job_id else job_id
        _job_dir_for_siblings = os.path.join(_job_artifact_root, "jobs", _parent_job_id)
        _segments_dir = os.path.join(_job_dir_for_siblings, "segments")
        _sibling_exports: Dict[str, List[str]] = {}
        _current_seg_id = segment_context.get("segment_id", "")

        if not os.path.isdir(_segments_dir):
            return

        for _sib_dir_name in sorted(os.listdir(_segments_dir)):
            if _sib_dir_name == _current_seg_id:
                continue
            _sib_enrich_path = os.path.join(
                _segments_dir, _sib_dir_name, "enrichment.json"
            )
            if not os.path.isfile(_sib_enrich_path):
                continue
            try:
                with open(_sib_enrich_path, "r", encoding="utf-8") as _sef:
                    _sib_enrich = json.load(_sef)
                _sib_symbols = list(_sib_enrich.get("exports", []))
                for _sf in _sib_enrich.get("functions", []):
                    _fn_name = _sf.get("name", "")
                    if _fn_name and _fn_name not in _sib_symbols:
                        _sib_symbols.append(_fn_name)
                for _sc in _sib_enrich.get("constants", []):
                    _cn_name = _sc.get("name", "")
                    if _cn_name and _cn_name not in _sib_symbols:
                        _sib_symbols.append(_cn_name)
                if _sib_symbols:
                    _sibling_exports[_sib_dir_name] = _sib_symbols
            except Exception:
                pass

        if _sibling_exports:
            parts.append("#### 📋 Complete Sibling Export Map (v5.36)")
            parts.append("")
            parts.append(
                "**REFERENCE**: These are ALL real function/constant names "
                "exported by sibling segments. When your code needs to call "
                "a function in another segment, pick from THIS list."
            )
            parts.append("")
            _total_sib_symbols = 0
            for _sib_id, _sib_syms in _sibling_exports.items():
                parts.append(
                    f"- **{_sib_id}** exports: "
                    f"{', '.join(f'`{s}`' for s in _sib_syms)}"
                )
                _total_sib_symbols += len(_sib_syms)
            parts.append("")
            parts.append(
                "If the function you need is NOT in this list, it either "
                "doesn't exist (define it yourself) or belongs to stdlib/"
                "third-party (import normally)."
            )
            parts.append("")
            logger.info(
                "[segment_prompt] v5.36 Sibling export map: %d segment(s), %d symbol(s)",
                len(_sibling_exports), _total_sib_symbols,
            )
    except Exception as _sib_err:
        logger.warning(
            "[segment_prompt] v5.36 Sibling export map failed (non-fatal): %s",
            _sib_err,
        )


# =========================================================================
# v2.0: CODEBASE PATTERN REFERENCE
# =========================================================================

# Mapping of parent directory patterns to known sibling reference files.
# Key = parent dir pattern (lowercase), Value = list of candidate reference
# paths to try (first found wins).  Paths are relative to project roots.
_SIBLING_REFERENCE_MAP = {
    # Frontend view components — reference an existing view
    "components/": [
        "src/components/investments/InvestmentsView.tsx",
        "src/components/finance/FinanceView.tsx",
        "src/components/content/ContentView.tsx",
        "src/components/lifestyle/LifestyleView.tsx",
    ],
    # Frontend types — reference existing type definitions
    "types/": [
        "src/types/index.ts",
    ],
    # Backend routers — reference an existing router
    "routers/": [
        "app/content/router.py",
        "app/astra_memory/router.py",
        "app/introspection/router.py",
    ],
    # Backend models — reference an existing model
    "models/": [
        "app/content/project_models.py",
        "app/investments/models.py",
        "app/finance/models.py",
    ],
    # Backend app modules (app/education/, app/foo/) — reference existing module
    "app/": [
        "app/content/router.py",
        "app/investments/models.py",
    ],
}

_PROJECT_ROOTS = ["D:\\Orb", "D:\\orb-desktop"]
_MAX_REFERENCE_CHARS = 8000  # Cap per file to avoid bloating the prompt


def _inject_codebase_pattern_reference(
    parts: List[str],
    segment_context: dict,
) -> None:
    """v2.0: Inject an existing sibling component as a pattern reference.

    Only triggers when ALL files in the segment are CREATE targets (no
    existing source evidence).  Finds the best matching existing file
    from the codebase and injects it as a read-only pattern reference
    so the architecture LLM can match existing codebase conventions.
    """
    file_scope = segment_context.get("file_scope", [])
    if not file_scope:
        return

    # Check if THIS segment owns any existing source files (not just
    # whether the global source_evidence dict is non-empty).
    source_evidence = segment_context.get("source_file_evidence", {})
    norm_scope = {f.replace("\\", "/").lower() for f in file_scope}
    has_own_sources = any(
        k.replace("\\", "/").lower() in norm_scope
        for k in source_evidence
    )
    if has_own_sources:
        # Segment owns existing source files — no need for pattern reference
        return

    # Determine what kind of files we're creating
    # Normalise to forward slashes for matching
    norm_files = [f.replace("\\", "/").lower() for f in file_scope]

    # Find best matching sibling reference
    reference_path = None
    for pattern, candidates in _SIBLING_REFERENCE_MAP.items():
        if any(pattern in f for f in norm_files):
            for candidate in candidates:
                for root in _PROJECT_ROOTS:
                    abs_path = os.path.join(
                        root, candidate.replace("/", os.sep)
                    )
                    if sandbox_isfile(abs_path):
                        reference_path = abs_path
                        break
                if reference_path:
                    break
        if reference_path:
            break

    if not reference_path:
        return

    try:
        # v2.1 (Job 9): Check pattern cache before filesystem read
        content = None
        _cache_hit = False
        try:
            from app.pot_spec.pattern_cache import get_pattern_cache
            _pcache = get_pattern_cache()
            _cached = _pcache.get(reference_path)
            if _cached:
                content = _cached.content
                _cache_hit = True
                logger.info(
                    "[segment_prompt] v2.1 Pattern cache HIT: %s (hits=%d)",
                    reference_path, _cached.hit_count,
                )
        except Exception:
            pass

        if content is None:
            content = sandbox_read_text(reference_path)
            if content:
                content = content[:_MAX_REFERENCE_CHARS]
            else:
                return

            # v2.1: Cache the result for next time
            if len(content) >= 50:
                try:
                    _pcache = get_pattern_cache()
                    _pext = os.path.splitext(reference_path)[1]
                    _ptype = "component" if _pext in (".tsx", ".ts") else "module"
                    _pcache.put(reference_path, content, pattern_type=_ptype)
                except Exception:
                    pass

        if len(content) < 50:
            return

        rel_path = reference_path
        for root in _PROJECT_ROOTS:
            if reference_path.startswith(root):
                rel_path = reference_path[len(root) + 1:].replace(os.sep, "/")
                break

        _cache_tag = " [cached]" if _cache_hit else ""
        parts.append("### 📚 Codebase Pattern Reference (v2.1)\n")
        parts.append(
            "The following existing file serves the same role as what you are "
            "building. Use it as a **pattern reference** for: import style, "
            "component structure, API fetch patterns, icon usage, CSS class "
            "naming, loading/error state handling, and export conventions.\n"
        )
        parts.append(
            "**DO NOT copy this file.** Use it to understand the conventions, "
            "then build your new components following the same patterns.\n"
        )

        truncated = ""
        if len(content) >= _MAX_REFERENCE_CHARS:
            truncated = f"  [truncated at {_MAX_REFERENCE_CHARS} chars]"

        ext = os.path.splitext(reference_path)[1]
        lang = "typescript" if ext in (".tsx", ".ts") else "python"
        parts.append(f"**`{rel_path}`** ({len(content):,} chars){truncated}{_cache_tag}")
        parts.append(f"```{lang}\n{content}\n```\n")

        logger.info(
            "[segment_prompt] v2.1 Codebase pattern reference: %s (%d chars%s)",
            rel_path, len(content), _cache_tag,
        )
    except Exception as e:
        logger.warning(
            "[segment_prompt] v2.1 Pattern reference failed (non-fatal): %s", e
        )
