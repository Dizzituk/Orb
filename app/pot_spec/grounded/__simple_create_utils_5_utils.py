import logging
import os
from app.pot_spec.grounded._simple_create_utils import CreateEvidence, _detect_tech_stack
from app.pot_spec.grounded._simple_create_utils import _extract_constraints, _extract_task_keywords, _suggest_new_files
from app.pot_spec.grounded._simple_create_utils import _extract_patterns, _find_integration_points, build_create_spec
from app.pot_spec.grounded._simple_create_utils import _resolve_mentioned_files
from typing import Any, Callable, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


CREATE_ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing a feature request.

You will receive:
1. A feature description (from the Weaver stage)
2. Detected tech stack
3. Discovered integration points (existing files)
4. Extracted constraints

YOUR TASK:
Produce a structured analysis with these sections:

## Architecture Overview
Brief description of the feature's architecture (3-5 sentences).

## Implementation Steps
Numbered, actionable implementation steps. Each step should reference specific 
files (existing or new) and describe what changes are needed. Order by dependency.

## Files to Modify
For each existing file that needs changes, explain WHAT changes are needed and WHY.

## New Files to Create
For each new file, explain its purpose and key contents.

## Acceptance Criteria
Task-specific, testable acceptance criteria. Include criteria derived from 
explicit constraints (e.g., "no network traffic during transcription").

IMPORTANT:
- Be specific to THIS feature, not generic
- Respect ALL constraints listed
- Reference actual integration points provided
- Keep implementation steps concrete and actionable
- Do NOT suggest cloud services if constraints say local-only
- Do NOT suggest files/features outside the stated phase scope

AMBIGUITY HANDLING:
When a design decision has multiple valid approaches (e.g., "should the endpoint
accept multipart/form-data or raw bytes?"), do NOT flag it as HUMAN_REQUIRED if
a safe default exists that covers both options. Instead:
- Pick the most flexible default (e.g., "support both multipart and raw body")
- State it as a DECISION_ALLOWED with the chosen default
- Only flag HUMAN_REQUIRED when there is genuine ambiguity with no safe default,
  or when the choice has significant architectural consequences that cannot be
  reversed without major rework
- Implementation details like exact response field names, optional metadata fields,
  or input format variants are NOT architectural decisions — adopt sensible defaults

ENTRYPOINT IDENTIFICATION (CRITICAL):
When generating EVIDENCE_REQUESTs to locate backend entrypoints (e.g., main.py):
- The goal is to find the file that instantiates FastAPI() (or calls an app factory
  returning FastAPI) and registers routers via include_router().
- Ignore any main.py files under /static/, /dist/, /build/, /public/, or frontend
  project roots. These are NOT the backend entrypoint.
- success_criteria MUST include: "Evidence must include the lines showing
  app = FastAPI(...) (or equivalent factory) and at least one include_router(...)."
- If multiple main.py files exist, the ER must distinguish them by checking content,
  not just path.

CONFIGURATION FILE EVIDENCE:
When the task references external configuration files (e.g., config.ini, .env, YAML)
that are loaded by service wrappers or modules being integrated:
- Include an EVIDENCE_REQUEST to read the configuration file directly.
- Confirm that the sections, keys, and values match what the consuming code expects.
- This prevents runtime mismatches between config parsers and actual config content.
- Fold this into an existing ER if it already reads the consuming module's code,
  or create a dedicated ER if the config file is on a separate path.

FUNCTION SIGNATURE VERIFICATION (CRITICAL):
When your implementation plan proposes CALLING an existing function (e.g., store_embedding(),
search_embeddings(), generate_embedding(), or any other existing function):
- You MUST emit an EVIDENCE_REQUEST to read that function's definition BEFORE proposing
  parameter names or return types in your spec.
- Verify: exact parameter names, required vs optional params, return type/structure.
- Do NOT assume parameter names like 'top_k', 'source_type', 'chunk_index' exist —
  the actual function may use different names (e.g., 'limit', 'content_type', 'idx').
- If your plan says "call X(a=1, b=2)", you must have CITED evidence showing X accepts
  parameters 'a' and 'b'. Otherwise emit an ER to read the function.
- This applies equally to ORM model fields — if you propose filtering by a column,
  verify that column exists on the model.

CALLER CHAIN VERIFICATION (CRITICAL):
When your implementation plan proposes injecting code at a specific point (e.g., adding
a call inside an existing function), you MUST verify:
- WHO calls that function and WHAT data they pass in.
- If your plan depends on a field being available (e.g., project_id in task metadata),
  emit an EVIDENCE_REQUEST to read at least one caller to confirm the field is populated.
- Do NOT propose regex-parsing free text fields to extract structured data that should
  be passed explicitly. If a field isn't available, the spec should say "add this field"
  rather than "parse it out of an unrelated string".

ER ID UNIQUENESS:
- Every EVIDENCE_REQUEST must have a unique id (ER-001, ER-002, etc.).
- NEVER emit two EVIDENCE_REQUEST blocks with the same id.
- If you need to request evidence for two different things, use different ids.

EVIDENCE_REQUEST FORMAT (you MUST use this exact YAML format — the parser is strict):
EVIDENCE_REQUEST:
  id: "ER-NNN"
  severity: "CRITICAL" | "NONCRITICAL"
  need: "What you need to know"
  why: "What breaks if you guess wrong"
  scope:
    roots: ["where to look"]
    max_files: 500
  tool_calls:
    - tool: "sandbox_inspector.read_sandbox_file"
      args: {file_path: "full/path/to/file.py"}
      expect: "What you expect to find"
  success_criteria: "What counts as having the answer"
  fallback_if_not_found: "DECISION_ALLOWED" | "HUMAN_REQUIRED"

CRITICAL FORMATTING RULES:
- The block MUST start with 'EVIDENCE_REQUEST:' on its own line (no prefix, no markdown header)
- Fields MUST be indented with 2 spaces under EVIDENCE_REQUEST:
- The id field MUST be quoted: id: "ER-001" (not id: ER-001)
- Do NOT wrap EVIDENCE_REQUESTs in markdown code blocks or headers
- Do NOT use prose-style descriptions — use the YAML structure above

TOOL USAGE:
- Use tool 'sandbox_inspector.read_sandbox_file' with args: {file_path: "FULL_PATH"} for reading files
- Use tool 'sandbox_inspector.run_sandbox_discovery_chain' with args: {anchor: "FULL_PATH"} for listing directories
- ALWAYS use FULL ABSOLUTE PATHS from the Integration Points list provided to you
- Do NOT guess paths. Only request files that appear in the Integration Points list,
  or that you discovered from a previous directory listing or file read.
- If you need to find files not in the Integration Points list, first request a
  directory listing of the parent directory, then request specific files from the results.

When you need to examine files to ground your analysis, emit EVIDENCE_REQUEST blocks
in this exact format. The orchestrator will read the files and re-prompt you with
the actual contents so you can produce a grounded analysis instead of guessing."""

async def build_grounded_create_spec(
    goal: str,
    what_to_do: str,
    project_paths: List[str],
    sandbox_client: Any = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    llm_call_func: Optional[Callable] = None,
) -> Tuple[str, CreateEvidence]:
    """
    v2.0: Build a grounded spec for CREATE tasks with LLM analysis.
    
    Now accepts provider_id, model_id, and llm_call_func to enable
    LLM-powered analysis using the model allocated by the spec_gate_stream.
    Falls back to template-only mode if LLM unavailable.
    
    Returns:
        Tuple of (spec_markdown, evidence)
    """
    logger.info("[simple_create] v2.0 Building LLM-grounded CREATE spec")
    print(f"[simple_create] v2.0 GROUNDED CREATE: {goal[:60]}...")
    
    # v2.0: Extract CONCEPTS (not raw keywords)
    combined_text = f"{goal} {what_to_do}"
    concepts = _extract_task_keywords(combined_text)
    print(f"[simple_create] v2.0 Concepts: {concepts[:10]}")
    
    # v2.0: Extract constraints
    constraints = _extract_constraints(combined_text)
    print(f"[simple_create] v2.0 Constraints: {constraints}")
    
    # Detect tech stack for each project path
    tech_stack = TechStack()
    for path in project_paths:
        if os.path.isdir(path):
            detected = _detect_tech_stack(path, sandbox_client)
            for attr in ['frontend_framework', 'frontend_language', 'backend_framework',
                        'backend_language', 'styling', 'state_management', 'api_pattern']:
                if getattr(detected, attr) and not getattr(tech_stack, attr):
                    setattr(tech_stack, attr, getattr(detected, attr))
    
    print(f"[simple_create] v2.0 Tech stack: {tech_stack.frontend_framework}/{tech_stack.backend_framework}")
    
    # v2.0: Find integration points using CONCEPTS (not raw keywords)
    all_points = []
    for path in project_paths:
        if os.path.isdir(path):
            points = _find_integration_points(path, concepts, sandbox_client)
            all_points.extend(points)
    
    print(f"[simple_create] v2.0 Found {len(all_points)} integration points")
    
    # Extract patterns from integration points
    patterns = _extract_patterns(all_points, tech_stack)
    print(f"[simple_create] v2.0 Extracted {len(patterns)} patterns")
    
    # v2.0: Suggest new files with CONSTRAINT awareness
    suggested_files = _suggest_new_files(concepts, constraints, tech_stack, project_paths)
    
    # v5.1: PRE-RESOLVE mentioned filenames to real paths BEFORE LLM call
    # The LLM should never have to guess file paths — resolve them proactively.
    resolved_target_files = _resolve_mentioned_files(combined_text, project_paths)
    if resolved_target_files:
        print(f"[simple_create] v5.1 RESOLVED {len(resolved_target_files)} target file(s):")
        for _rtf in resolved_target_files:
            print(f"[simple_create] v5.1   {_rtf['mentioned']} → {_rtf['resolved_path']}")
    else:
        print(f"[simple_create] v5.1 No explicit filenames found in job description")

    # v2.0: Run LLM analysis if model available
    llm_analysis = None
    if provider_id and model_id:
        # Import llm_call if not provided
        if llm_call_func is None:
            try:
                from app.providers.registry import llm_call as registry_llm_call
                llm_call_func = registry_llm_call
                print(f"[simple_create] v2.0 Loaded llm_call from registry")
            except ImportError:
                print(f"[simple_create] v2.0 WARNING: Could not import llm_call from registry")
        
        if llm_call_func:
            llm_analysis = await _run_llm_analysis(
                goal=goal,
                what_to_do=what_to_do,
                tech_stack=tech_stack,
                integration_points=all_points,
                constraints=constraints,
                suggested_files=suggested_files,
                provider_id=provider_id,
                model_id=model_id,
                llm_call_func=llm_call_func,
                resolved_target_files=resolved_target_files,  # v5.1
            )

            # v4.0: Fulfil EVIDENCE_REQUESTs from the LLM analysis
            # If the LLM produced ERs asking to read specific files, read them
            # and re-prompt with real evidence for a grounded spec.
            if llm_analysis and 'EVIDENCE_REQUEST' in llm_analysis:
                logger.info("[SPEC_GATE_EVIDENCE] LLM analysis contains EVIDENCE_REQUESTs — starting fulfilment")
                print("[SPEC_GATE_EVIDENCE] EVIDENCE_REQUESTs detected — starting fulfilment loop")
                llm_analysis = await _fulfil_evidence_requests(
                    llm_analysis=llm_analysis,
                    provider_id=provider_id,
                    model_id=model_id,
                    llm_call_func=llm_call_func,
                    project_paths=project_paths,
                    goal=goal,
                    what_to_do=what_to_do,
                )
                print(f"[SPEC_GATE_EVIDENCE] Fulfilment complete: {len(llm_analysis)} chars")
            elif llm_analysis:
                logger.info("[SPEC_GATE_EVIDENCE] No EVIDENCE_REQUESTs in LLM analysis — skipping fulfilment")
    else:
        print(f"[simple_create] v2.0 NO LLM: provider_id={provider_id}, model_id={model_id}")
    
    # Build evidence bundle
    evidence = CreateEvidence(
        tech_stack=tech_stack,
        integration_points=all_points,
        existing_patterns=patterns,
        suggested_files=suggested_files,
        keywords_found={c: [] for c in concepts},
        constraints=constraints,
        llm_analysis=llm_analysis,
    )
    
    # Build spec
    spec = build_create_spec(
        goal=goal,
        what_to_do=what_to_do,
        evidence=evidence,
        project_paths=project_paths,
    )
    
    print(f"[simple_create] v2.0 SPEC READY: {len(spec)} chars (LLM={'yes' if llm_analysis else 'no'})")
    
    return spec, evidence
