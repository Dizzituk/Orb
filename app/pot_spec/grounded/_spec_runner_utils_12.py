from __future__ import annotations
import logging
import os
import re
from app.pot_spec.grounded._spec_runner_utils_10 import _FALLBACK_ALL_PATHS, _FALLBACK_BACKEND_PATHS, _FALLBACK_FRONTEND_PATHS
from app.pot_spec.grounded._spec_runner_utils_11 import SCOPE_BACKEND, SCOPE_FRONTEND, _detect_search_replace_terms
from typing import List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _extract_project_paths(text: str, search_term: str = None, replace_term: str = None) -> List[str]:
    """
    v4.3: Scope-aware path extraction.
    
    Key improvements:
    1. Detect search/replace terms and EXCLUDE them from path matching
    2. Detect explicit scope (UI/frontend vs backend) and RESPECT it
    3. Only match bare 'orb'/'astra' when they're explicit project names
    
    Examples:
    - "change the front-end UI so it's called Astra" -> D:\\orb-desktop ONLY
    - "rename Orb to Astra in Orb Desktop" -> D:\\orb-desktop ONLY  
    - "rename Orb to Astra across the codebase" -> D:\\orb-desktop + D:\\Orb
    """
    from ._spec_runner_utils_13 import _discover_project_roots
    if not text:
        return []
    
    text_lower = text.lower()
    paths = []
    
    print(f"[spec_runner] v4.3.4 SCOPE-AWARE PATH EXTRACTION: input={len(text)} chars")
    
    # Step 1: Detect search/replace terms (don't treat these as project names)
    detected_search, detected_replace = _detect_search_replace_terms(text)
    if detected_search:
        search_term = detected_search
        replace_term = detected_replace
        print(f"[spec_runner] v4.3.4 DETECTED SEARCH/REPLACE: '{search_term}' -> '{replace_term}'")
    
    excluded_terms = set()
    if search_term:
        excluded_terms.add(search_term.lower())
    if replace_term:
        excluded_terms.add(replace_term.lower())
    print(f"[spec_runner] v4.3.4 EXCLUDED TERMS: {excluded_terms}")
    
    # Step 2: Check for EXPLICIT scope indicators
    has_frontend_scope = any(pattern in text_lower for pattern in SCOPE_FRONTEND)
    has_backend_scope = any(pattern in text_lower for pattern in SCOPE_BACKEND)
    
    print(f"[spec_runner] v4.3.4 SCOPE: frontend={has_frontend_scope}, backend={has_backend_scope}")
    
    # Step 3: Check for project name patterns via DYNAMIC DISCOVERY
    # v4.5: Uses architecture index instead of hardcoded patterns
    discovery = _discover_project_roots()
    for alias, alias_paths in discovery["aliases"].items():
        if alias in text_lower:
            # Don't match aliases that are search/replace terms
            if alias not in excluded_terms:
                print(f"[spec_runner] v4.5 DISCOVERED PROJECT: '{alias}' -> {alias_paths}")
                paths.extend(alias_paths)
    
    # Step 3b: SCOPE-AWARE ROOT INJECTION (v4.6)
    # If alias matching found some paths but the scope flags indicate
    # frontend/backend is needed and we're missing the corresponding root,
    # inject it from discovery.
    #
    # v4.6.1: Also detect frontend scope from VISUAL INTENT SIGNALS.
    # Users describe frontend work by talking about how things LOOK
    # (dashboard, cards, dark theme, progress bar) without literally
    # saying "frontend". These semantic signals are just as valid.
    if not has_frontend_scope:
        from ._simple_create_utils_17 import _VISUAL_INTENT_SIGNALS
        visual_matches = [s for s in _VISUAL_INTENT_SIGNALS if s in text_lower]
        if visual_matches:
            has_frontend_scope = True
            print(f"[spec_runner] v4.6.1 VISUAL INTENT detected frontend scope: {visual_matches[:5]}")

    if paths:
        fe_paths = discovery["frontend_paths"] or _FALLBACK_FRONTEND_PATHS
        be_paths = discovery["backend_paths"] or _FALLBACK_BACKEND_PATHS
        has_fe = any(p.lower().replace('/', '\\') in {fp.lower().replace('/', '\\') for fp in fe_paths} for p in paths)
        has_be = any(p.lower().replace('/', '\\') in {bp.lower().replace('/', '\\') for bp in be_paths} for p in paths)

        if has_frontend_scope and not has_fe:
            for fp in fe_paths:
                if fp not in paths:
                    print(f"[spec_runner] v4.6 SCOPE INJECTION: frontend root '{fp}' added (scope=frontend but no frontend path matched)")
                    paths.append(fp)
        if has_backend_scope and not has_be:
            for bp in be_paths:
                if bp not in paths:
                    print(f"[spec_runner] v4.6 SCOPE INJECTION: backend root '{bp}' added (scope=backend but no backend path matched)")
                    paths.append(bp)

    # Step 4: Check for explicit paths like "D:\orb-desktop" or "D:\Orb"
    # v4.3.4: Only match short, valid folder names (max 20 chars)
    # This prevents garbage like "D:\Orb Desktop front-end UI text"
    for match in re.findall(r'([A-Za-z]:[\\/][A-Za-z][A-Za-z0-9_\-]{0,17})', text):
        cleaned = match.rstrip(' \t')
        # Skip if too short or too long
        if len(cleaned) < 4 or len(cleaned) > 20:
            continue
        # Skip if it contains newlines  
        if '\n' in cleaned or '\r' in cleaned:
            continue
        # Check if this looks like a path to a known project
        name_part = cleaned[3:].lower().replace(' ', '-').replace('\\', '')
        if name_part not in excluded_terms:
            print(f"[spec_runner] v4.3.4 EXPLICIT PATH: '{cleaned}'")
            paths.append(cleaned)
            # Also add hyphenated version
            if ' ' in cleaned:
                drive = cleaned[:3]
                folder = cleaned[3:].replace(' ', '-').lower()
                paths.append(drive + folder)
    
    # Step 5: If no explicit paths found, use scope with DISCOVERED paths
    # v4.5: Uses dynamically discovered frontend/backend paths
    if not paths:
        fe_paths = discovery["frontend_paths"] or _FALLBACK_FRONTEND_PATHS
        be_paths = discovery["backend_paths"] or _FALLBACK_BACKEND_PATHS
        all_paths = discovery["all_paths"] or _FALLBACK_ALL_PATHS

        if has_frontend_scope and not has_backend_scope:
            # User explicitly mentioned UI/frontend -> frontend only
            print(f"[spec_runner] v4.5 SCOPE-BASED: frontend only -> {fe_paths}")
            paths = list(fe_paths)
        elif has_backend_scope and not has_frontend_scope:
            # User explicitly mentioned backend -> backend only
            print(f"[spec_runner] v4.5 SCOPE-BASED: backend only -> {be_paths}")
            paths = list(be_paths)
        elif has_frontend_scope and has_backend_scope:
            # User mentioned both -> all paths
            print(f"[spec_runner] v4.5 SCOPE-BASED: both frontend + backend -> {all_paths}")
            paths = list(all_paths)
        # else: no scope indicators and no explicit paths -> return empty
    
    # Step 6: "X drive" + project name detection (fallback)
    # v4.5: Uses discovered aliases to resolve "D drive" + project name
    if not paths:
        drive_match = re.search(r'\b([A-Za-z])\s+drive\b', text, re.IGNORECASE)
        if drive_match:
            drive = drive_match.group(1).upper()
            # Check discovered aliases for project name patterns in the text
            for alias, alias_paths in discovery["aliases"].items():
                # Only check multi-word aliases ("orb desktop", not bare "orb")
                if ' ' in alias or '-' in alias:
                    # Build regex from alias: "orb desktop" -> r'\borb[\s-]*desktop\b'
                    alias_parts = re.split(r'[-\s]', alias)
                    alias_pattern = r'\b' + r'[\s-]*'.join(re.escape(p) for p in alias_parts) + r'\b'
                    if re.search(alias_pattern, text_lower) and alias_parts[0] not in excluded_terms:
                        # Use the discovered root but on the specified drive
                        for ap in alias_paths:
                            # Replace drive letter with user-specified drive
                            folder_part = ap[2:]  # Strip "D:" prefix
                            paths.append(f"{drive}:{folder_part}")
                            print(f"[spec_runner] v4.5 DRIVE+ALIAS: '{alias}' on {drive}: -> {drive}:{folder_part}")
    
    # Dedupe while preserving order
    seen = set()
    unique = []
    for p in paths:
        key = p.lower().replace('/', '\\').rstrip('\\')
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    print(f"[spec_runner] v4.3.4 FINAL PATHS: {unique}")
    return unique

def _dedup_evidence_requests(spec_markdown: str) -> str:
    """
    v4.7: Remove duplicate EVIDENCE_REQUEST blocks from spec markdown.

    Scans for EVIDENCE_REQUEST blocks (delimited by 'EVIDENCE_REQUEST:' headers),
    extracts the 'id' field from each, and removes duplicates (keeping first occurrence).

    Returns the cleaned markdown with duplicates removed.
    """
    if not spec_markdown or 'EVIDENCE_REQUEST' not in spec_markdown:
        return spec_markdown

    lines = spec_markdown.split('\n')
    output_lines = []
    seen_er_ids = set()
    in_er_block = False
    er_block_lines = []
    er_block_id = None
    skip_current_block = False
    duplicates_removed = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect start of an EVIDENCE_REQUEST block
        if stripped.startswith('EVIDENCE_REQUEST:') or stripped == 'EVIDENCE_REQUEST:':
            # If we were already in an ER block, flush it
            if in_er_block and er_block_lines:
                if not skip_current_block:
                    output_lines.extend(er_block_lines)
                else:
                    duplicates_removed += 1

            # Start new ER block
            in_er_block = True
            er_block_lines = [line]
            er_block_id = None
            skip_current_block = False
            i += 1
            continue

        if in_er_block:
            # Check if this line has the id field
            id_match = re.match(r'\s+id:\s*["\']?(ER-[\w-]+)["\']?', line)
            if id_match:
                er_block_id = id_match.group(1)
                if er_block_id in seen_er_ids:
                    skip_current_block = True
                    logger.info("[spec_runner] v4.7 DEDUP: dropping duplicate %s", er_block_id)
                    print(f"[spec_runner] v4.7 ER DEDUP: dropping duplicate {er_block_id}")
                else:
                    seen_er_ids.add(er_block_id)

            # Check if we've hit the end of this ER block
            # An ER block ends when we hit: another EVIDENCE_REQUEST, a markdown header,
            # an empty line followed by non-indented content, or end of file
            next_is_new_section = False
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                next_is_new_section = (
                    next_stripped.startswith('EVIDENCE_REQUEST:') or
                    next_stripped.startswith('# ') or
                    next_stripped.startswith('## ') or
                    next_stripped.startswith('### ') or
                    # A new top-level YAML key after the ER block (not indented)
                    (next_stripped and not next_stripped.startswith(' ') and
                     not next_stripped.startswith('-') and
                     not next_stripped.startswith('EVIDENCE_REQUEST') and
                     ':' not in next_stripped and
                     stripped == '')
                )

            er_block_lines.append(line)

            # If next line starts a new section, or we're at EOF, flush this block
            if next_is_new_section or i == len(lines) - 1:
                if not skip_current_block:
                    output_lines.extend(er_block_lines)
                else:
                    duplicates_removed += 1
                in_er_block = False
                er_block_lines = []
                er_block_id = None
                skip_current_block = False
        else:
            output_lines.append(line)

        i += 1

    # Flush any remaining ER block
    if in_er_block and er_block_lines:
        if not skip_current_block:
            output_lines.extend(er_block_lines)
        else:
            duplicates_removed += 1

    if duplicates_removed > 0:
        print(f"[spec_runner] v4.7 ER DEDUP COMPLETE: removed {duplicates_removed} duplicate block(s), "
              f"{len(seen_er_ids)} unique ER(s) remain")
        logger.info("[spec_runner] v4.7 ER dedup: removed %d duplicate(s), %d unique remain",
                    duplicates_removed, len(seen_er_ids))

    return '\n'.join(output_lines)

def _write_segmentation_output(job_id: str, manifest) -> None:
    """
    v4.9: Write manifest and segment specs to the job directory.
    
    Creates:
        <artifact_root>/jobs/<job-id>/segments/manifest.json
        <artifact_root>/jobs/<job-id>/segments/seg-XX/spec.json (per segment)
    
    v4.9 (2026-02-08): Uses spec_gate_persistence.artifact_root() + job_dir()
    for path construction instead of relative paths. The previous version used
    os.path.join('jobs', job_id) which resolves relative to cwd — wrong when
    the FastAPI server's working directory differs from the project root.
    """
    from .segment_schemas import SegmentManifest
    
    # Use the same path construction as spec_gate_persistence.py
    try:
        from ..spec_gate_persistence import artifact_root as _artifact_root, job_dir as _job_dir
        job_dir_path = _job_dir(_artifact_root(), job_id)
    except ImportError:
        # Fallback: replicate the logic directly
        _root = os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))
        job_dir_path = os.path.join(_root, "jobs", job_id)
    
    os.makedirs(job_dir_path, exist_ok=True)
    
    segments_dir = os.path.join(job_dir_path, 'segments')
    os.makedirs(segments_dir, exist_ok=True)
    
    # Write manifest
    manifest_path = os.path.join(segments_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(manifest.to_json(indent=2))
    
    logger.info("[spec_runner] v4.8 Wrote manifest: %s", manifest_path)
    
    # Write per-segment specs
    for seg in manifest.segments:
        seg_dir = os.path.join(segments_dir, seg.segment_id)
        os.makedirs(seg_dir, exist_ok=True)
        seg_spec_path = os.path.join(seg_dir, 'spec.json')
        with open(seg_spec_path, 'w', encoding='utf-8') as f:
            f.write(seg.to_json(indent=2))
        logger.info("[spec_runner] v4.8 Wrote segment spec: %s", seg_spec_path)

def _build_single_segment_manifest(
    spec_markdown: str,
    spec_id: str,
    spec_hash: str,
    goal: str,
    file_scope: List[str],
    requirements: List[str],
    acceptance_criteria: List[str],
    job_kind: str = "architecture",
) -> "SegmentManifest":
    """
    v5.4 PHASE 1: Wrap a non-segmented spec into a single-segment manifest.
    
    This ensures SpecGate ALWAYS outputs a manifest, whether the job has 1 or N
    segments. Downstream consumers (segment loop, critical pipeline) only need
    to handle one format.
    
    The single segment gets:
    - segment_id: "seg-01" (consistent with multi-segment naming)
    - title: the goal from the spec
    - No dependencies (it's the only segment)
    - No interface contracts (nothing to expose/consume)
    - Full file scope and requirements from the parent spec
    """
    from .segment_schemas import SegmentManifest, SegmentSpec
    
    segment = SegmentSpec(
        segment_id="seg-01",
        title=goal or "Single-segment job",
        parent_spec_id=spec_id,
        requirements=requirements,
        file_scope=file_scope,
        evidence_files=[],
        dependencies=[],
        exposes=None,
        consumes=None,
        acceptance_criteria=acceptance_criteria,
        estimated_files=len(file_scope),
    )
    
    manifest = SegmentManifest(
        parent_spec_id=spec_id,
        parent_spec_hash=spec_hash,
        segments=[segment],
        requirement_map={r: ["seg-01"] for r in requirements},
        total_segments=1,
        total_files=len(file_scope),
        manifest_version="1.0",
    )
    
    logger.info(
        "[spec_runner] v5.4 Built single-segment manifest: spec_id=%s, files=%d",
        spec_id, len(file_scope),
    )
    
    return manifest
