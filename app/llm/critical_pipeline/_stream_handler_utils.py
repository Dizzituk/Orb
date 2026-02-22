from __future__ import annotations
import os
from app.llm.critical_pipeline.artifact_binding import extract_artifact_bindings
from app.llm.critical_pipeline.evidence import gather_critical_pipeline_evidence
from app.llm.critical_pipeline.job_classification import JobKind
from app.llm.critical_pipeline.plan_micro import generate_micro_execution_plan
from app.llm.critical_pipeline.plan_scan import generate_scan_execution_plan
from app.llm.critical_pipeline.quickcheck_micro import micro_quickcheck
from app.llm.critical_pipeline.quickcheck_scan import scan_quickcheck
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _build_segment_critique_spec(
    segment_context: Dict[str, Any],
    parent_spec_markdown: Optional[str] = None,
) -> str:
    """Build a segment-scoped spec for critique evaluation.

    When processing a segment within a segmented job, the critique must
    evaluate the architecture against THIS segment's contract, not the
    parent job spec. The parent spec describes the whole job (e.g.
    'refactor this file, no new files') which may contradict what
    individual segments need to do.

    The segment-scoped spec:
    1. Defines the segment's own requirements and acceptance criteria
    2. Lists the segment's file_scope (what files it owns)
    3. Describes what other segments handle (so critique doesn't flag
       'missing' functionality that lives in sibling segments)
    4. Preserves parent spec context as reference, clearly scoped

    v5.26: Fixes critique false-positives where segment architectures
    were flagged for violating parent-level constraints that don't
    apply at the segment level.
    """
    seg_id = segment_context.get("segment_id", "unknown")
    seg_spec = segment_context.get("segment_spec", {})
    file_scope = segment_context.get("file_scope", [])
    requirements = segment_context.get("requirements", [])
    acceptance_criteria = segment_context.get("acceptance_criteria", [])
    dependencies = segment_context.get("dependencies", [])
    exposes = segment_context.get("exposes")
    consumes = segment_context.get("consumes")

    parts: List[str] = []

    # Header — make it clear this is a segment, not a standalone job
    parts.append(f"# Segment Spec: {seg_id}")
    parts.append(f"**Title:** {seg_spec.get('title', seg_id)}")
    parts.append("")
    parts.append(
        "This is ONE SEGMENT within a segmented job. The architecture "
        "should be evaluated against THIS segment's requirements below, "
        "NOT against the parent job spec. Other segments handle other "
        "parts of the work."
    )
    parts.append("")

    # Goal — the segment's actual job
    parts.append("## Goal")
    parts.append(seg_spec.get("title", "Implement this segment's file scope."))
    parts.append("")

    # File scope — authoritative list of files this segment owns
    if file_scope:
        parts.append("## File Scope (ONLY these files)")
        parts.append(
            "This segment is responsible for ONLY these files. "
            "Creating, modifying, or referencing other files is acceptable "
            "if they are dependencies or standard library imports."
        )
        for f in file_scope:
            parts.append(f"- `{f}`")
        parts.append("")

    # Requirements — the segment's own requirements
    if requirements:
        parts.append("## Requirements")
        for r in requirements:
            parts.append(f"- {r}")
        parts.append("")

    # Acceptance criteria — what the segment must achieve
    if acceptance_criteria:
        parts.append("## Acceptance Criteria")
        for ac in acceptance_criteria:
            parts.append(f"- {ac}")
        parts.append("")

    # Dependencies — what other segments this one depends on
    if dependencies:
        parts.append("## Dependencies")
        parts.append(
            "This segment depends on the following sibling segments. "
            "Their output files already exist or will exist when this "
            "segment executes. Importing from them is EXPECTED and CORRECT."
        )
        for dep in dependencies:
            parts.append(f"- `{dep}`")
        parts.append("")

    # v5.27: Available symbols from dependencies (from enrichment)
    # Prevents critique false-positives like "imports X from seg-02
    # but X is not in available exports" by giving the full symbol list.
    _enrichment = segment_context.get("enrichment")
    if _enrichment:
        _consumes = _enrichment.get("consumes", {})
        if _consumes:
            parts.append("## Available Symbols from Dependencies")
            parts.append(
                "These symbols are confirmed available from sibling segments "
                "(extracted from the source monolith via AST parsing). "
                "Importing any of these is CORRECT and should NOT be flagged."
            )
            for _dep_seg, _dep_syms in _consumes.items():
                if isinstance(_dep_syms, list) and _dep_syms:
                    parts.append(f"- From **{_dep_seg}**: {', '.join(f'`{s}`' for s in _dep_syms)}")
            parts.append("")

    if exposes:
        parts.append("## Exposes (what this segment provides to others)")
        if isinstance(exposes, dict):
            for k, v in exposes.items():
                parts.append(f"- **{k}**: {v}")
        else:
            parts.append(str(exposes))
        parts.append("")

    if consumes:
        parts.append("## Consumes (what this segment needs from dependencies)")
        if isinstance(consumes, dict):
            for k, v in consumes.items():
                parts.append(f"- **{k}**: {v}")
        else:
            parts.append(str(consumes))
        parts.append("")

    # Parent spec as REFERENCE (not authoritative for this segment)
    if parent_spec_markdown:
        parts.append("## Parent Job Spec (REFERENCE ONLY)")
        parts.append(
            "The following is the parent job spec for context. "
            "Constraints in the parent spec (e.g. 'no new files', "
            "'refactor single file') apply to the OVERALL job, not "
            "to this individual segment. This segment may legitimately "
            "create new files, import from sibling-created modules, or "
            "perform operations that appear to violate parent-level "
            "constraints but are correct at the segment level."
        )
        parts.append("")
        # Include truncated parent spec for context
        _parent_truncated = parent_spec_markdown[:4000]
        if len(parent_spec_markdown) > 4000:
            _parent_truncated += f"\n... (truncated from {len(parent_spec_markdown):,} chars)"
        parts.append(_parent_truncated)
        parts.append("")

    return "\n".join(parts)

def _format_enrichment_for_critique(enrichment: dict) -> str:
    """Format enrichment data into concise markdown for critique/revision.

    v5.18: Gives Gemini critique and Opus revision the same symbol
    awareness that GPT draft gets, so they don't flag missing functions
    that simply weren't in the AST extraction list.
    """
    if not enrichment:
        return ""
    parts = []

    constants = enrichment.get("constants", [])
    if constants:
        parts.append("**Constants:**")
        for c in constants:
            parts.append(f"- `{c.get('name', '?')}`")

    functions = enrichment.get("functions", [])
    if functions:
        parts.append("**Functions:**")
        for f in functions:
            sig = f.get("signature", f.get("name", "?"))
            parts.append(f"- `{sig}`")

    classes = enrichment.get("classes", [])
    if classes:
        parts.append("**Classes:**")
        for cl in classes:
            methods = ", ".join(cl.get("methods", [])[:10])
            parts.append(f"- `class {cl.get('name', '?')}` (methods: {methods})")

    consumed_by = enrichment.get("consumed_by", {})
    if consumed_by:
        parts.append("**Other segments import from this segment:**")
        for seg_id, syms in consumed_by.items():
            if isinstance(syms, list):
                parts.append(f"- {seg_id}: {', '.join(f'`{s}`' for s in syms)}")

    consumes = enrichment.get("consumes", {})
    if consumes:
        parts.append("**This segment imports from:**")
        for seg_id, syms in consumes.items():
            if isinstance(syms, list):
                parts.append(f"- {seg_id}: {', '.join(f'`{s}`' for s in syms)}")

    guidance = enrichment.get("design_guidance", "")
    if guidance:
        parts.append(f"**Design guidance:** {guidance}")

    if not parts:
        return ""
    return "\n".join(parts)

async def _handle_micro(
    spec_data, message, spec_id, job_id, project_id, db,
    trace, provider, model, response_parts,
):
    def _emit(text):
        response_parts.append(text)
        return _token(text)

    yield _emit("\n\u26a1 **Fast Path:** This is a micro-execution job.\n")
    yield _emit("No architecture design required - generating execution plan...\n\n")

    if not job_id:
        job_id = f"micro-{uuid4().hex[:8]}"

    # Gather evidence (light)
    micro_evidence = gather_critical_pipeline_evidence(
        spec_data=spec_data, message=message,
        include_arch_map=False, include_codebase_report=False,
        include_file_evidence=True,
    )
    if micro_evidence.file_evidence_loaded:
        yield _emit(
            f"\ud83d\udcda **File evidence loaded:** {len(micro_evidence.multi_target_files)} file(s)\n"
        )

    micro_plan = generate_micro_execution_plan(spec_data, job_id)

    # Quickcheck
    yield _emit("\ud83e\uddea **Running Quickcheck...**\n")
    qc = micro_quickcheck(spec_data, micro_plan)

    if qc.passed:
        yield _emit(f"{qc.summary}\n\n")
        yield _emit(micro_plan)

        binding_ctx = {
            "job_id": job_id,
            "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
            "repo_root": os.getenv("REPO_ROOT", "."),
        }
        bindings = extract_artifact_bindings(spec_data, binding_ctx)

        yield _sse("work_artifacts",
            spec_id=spec_id, job_id=job_id, job_kind=JobKind.MICRO_EXECUTION,
            critique_mode="quickcheck", critique_passed=True,
            artifact_bindings=bindings,
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "micro-execution")
        if trace:
            trace.finalize(success=True)

        yield _done(
            provider="local", model="micro-execution",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.MICRO_EXECUTION, critique_mode="quickcheck",
            critique_passed=True, artifact_bindings=len(bindings),
        )
    else:
        yield _emit(f"{qc.summary}\n\n")
        for issue in qc.issues:
            yield _emit(f"\u274c **{issue['id']}:** {issue['description']}\n")

        yield _emit("\n### Generated Plan (for review):\n")
        yield _emit(micro_plan)
        yield _emit(
            "\n---\n\u26a0\ufe0f **Quickcheck Failed** \u2014 Job NOT ready for Overwatcher.\n\n"
            "Please check:\n"
            "1. Did SpecGate resolve the input/output paths correctly?\n"
            "2. Is the spec complete with sandbox_input_path and sandbox_output_path?\n"
            "3. If the plan needs to write output, does the spec have a sandbox_generated_reply?\n\n"
            "You may need to re-run Spec Gate with more details about the file locations.\n"
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "micro-execution")
        if trace:
            trace.finalize(success=False, error_message="Quickcheck failed")

        yield _done(
            provider="local", model="micro-execution",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.MICRO_EXECUTION, critique_mode="quickcheck",
            critique_passed=False, quickcheck_issues=len(qc.issues),
        )

async def _handle_scan(
    spec_data, message, spec_id, job_id, project_id, db,
    trace, provider, model, response_parts,
):
    def _emit(text):
        response_parts.append(text)
        return _token(text)

    yield _emit("\n\ud83d\udd0d **Scan Mode:** Read-only filesystem scan.\n")
    yield _emit("No architecture design required - generating scan execution plan...\n\n")

    if not job_id:
        job_id = f"scan-{uuid4().hex[:8]}"

    scan_evidence = gather_critical_pipeline_evidence(
        spec_data=spec_data, message=message,
        include_arch_map=True, include_codebase_report=False,
        include_file_evidence=False, arch_map_max_lines=300,
    )
    if scan_evidence.arch_map_loaded:
        yield _emit(
            f"\ud83d\udcda **Architecture context loaded:** "
            f"{len(scan_evidence.arch_map_content or '')} chars\n"
        )

    scan_plan = generate_scan_execution_plan(spec_data, job_id)

    yield _emit("\ud83e\uddea **Running Scan Quickcheck...**\n")
    sqc = scan_quickcheck(spec_data, scan_plan)

    if sqc.passed:
        yield _emit(f"{sqc.summary}\n\n")
        for issue in sqc.issues:
            if issue.get("severity") == "warning":
                yield _emit(f"\u26a0\ufe0f **{issue['id']}:** {issue['description']}\n")
        yield _emit(scan_plan)

        yield _sse("work_artifacts",
            spec_id=spec_id, job_id=job_id, job_kind=JobKind.SCAN_ONLY,
            critique_mode="quickcheck", critique_passed=True,
            scan_roots=spec_data.get("scan_roots", []),
            scan_terms=spec_data.get("scan_terms", []),
            artifact_bindings=[],
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "scan-only")
        if trace:
            trace.finalize(success=True)

        yield _done(
            provider="local", model="scan-only",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.SCAN_ONLY, critique_mode="quickcheck",
            critique_passed=True,
        )
    else:
        yield _emit(f"{sqc.summary}\n\n")
        for issue in sqc.issues:
            icon = "\u274c" if issue.get("severity") == "blocking" else "\u26a0\ufe0f"
            yield _emit(f"{icon} **{issue['id']}:** {issue['description']}\n")
        yield _emit("\n### Generated Plan (for review):\n")
        yield _emit(scan_plan)
        yield _emit(
            "\n---\n\u26a0\ufe0f **Scan Quickcheck Failed** \u2014 Job NOT ready for execution.\n\n"
            "Please check:\n"
            "1. Did SpecGate resolve the scan_roots correctly?\n"
            "2. Did SpecGate extract the scan_terms from your request?\n"
            "3. Is the output_mode set to CHAT_ONLY?\n\n"
            "You may need to re-run Spec Gate with more details about what to scan.\n"
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "scan-only")
        if trace:
            trace.finalize(success=False, error_message="Scan quickcheck failed")

        yield _done(
            provider="local", model="scan-only",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.SCAN_ONLY, critique_mode="quickcheck",
            critique_passed=False, quickcheck_issues=len(sqc.issues),
        )
