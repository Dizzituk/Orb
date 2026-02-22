import logging
import os
from app.pot_spec.grounded._simple_create_utils import CreateEvidence, IntegrationPoint, _detect_tech_stack
from app.pot_spec.grounded._simple_create_utils import _CREATE_ANALYSIS_MODEL, _FALLBACK_MODELS
from app.pot_spec.grounded._simple_create_utils import _CREATE_ANALYSIS_TIMEOUT, _extract_constraints, _extract_task_keywords, _suggest_new_files
from app.pot_spec.grounded._simple_create_utils import _extract_patterns, _find_integration_points, build_create_spec
from app.pot_spec.grounded._simple_create_utils import _resolve_mentioned_files
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


@dataclass 
class TechStack:
    """Detected technology stack."""
    frontend_framework: Optional[str] = None  # React, Vue, Angular, vanilla
    frontend_language: Optional[str] = None   # TypeScript, JavaScript
    backend_framework: Optional[str] = None   # FastAPI, Express, Django
    backend_language: Optional[str] = None    # Python, Node, Go
    styling: Optional[str] = None             # CSS, Tailwind, styled-components
    state_management: Optional[str] = None    # Redux, Zustand, Context
    api_pattern: Optional[str] = None         # REST, GraphQL

async def _run_llm_analysis(
    goal: str,
    what_to_do: str,
    tech_stack: TechStack,
    integration_points: List[IntegrationPoint],
    constraints: List[str],
    suggested_files: List[str],
    provider_id: str,
    model_id: str,
    llm_call_func: Optional[Callable],
    resolved_target_files: Optional[List[Dict]] = None,  # v5.1
) -> Optional[str]:
    """
    v2.1: Use an LLM to analyze the feature request.
    
    Model selection priority:
    1. ASTRA_CREATE_ANALYSIS_MODEL env var (if set)
    2. Allocated model from spec_gate_stream
    3. On timeout: retry with faster fallback model
    
    Returns LLM-generated analysis or None if all attempts fail.
    """
    if not llm_call_func:
        logger.warning("[simple_create] v2.1 LLM unavailable, falling back to template")
        return None
    
    # v2.1: Model override from env
    use_provider = provider_id
    use_model = model_id
    if _CREATE_ANALYSIS_MODEL:
        use_model = _CREATE_ANALYSIS_MODEL
        logger.info("[simple_create] v2.1 MODEL OVERRIDE: %s (from ASTRA_CREATE_ANALYSIS_MODEL)", use_model)
        print(f"[simple_create] v2.1 MODEL OVERRIDE: {use_model}")
    
    # Build context for LLM
    stack_desc = []
    if tech_stack.frontend_framework:
        stack_desc.append(f"Frontend: {tech_stack.frontend_framework}" +
                         (f" ({tech_stack.frontend_language})" if tech_stack.frontend_language else ""))
    if tech_stack.backend_framework:
        stack_desc.append(f"Backend: {tech_stack.backend_framework}" +
                         (f" ({tech_stack.backend_language})" if tech_stack.backend_language else ""))
    if tech_stack.styling:
        stack_desc.append(f"Styling: {tech_stack.styling}")
    
    integration_desc = []
    for p in integration_points[:20]:  # v4.2: Increased from 10 to give LLM more valid paths for ERs
        integration_desc.append(f"- {p.file_path} ({p.action}): {p.relevance}")
    
    constraints_desc = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"
    
    # v4.7: Removed [:3000] cap on what_to_do. The weaver output is typically
    # 5-7k chars and the second half contains critical requirements (canonicalization,
    # retrieval, provenance, etc.) that were being silently dropped.
    # v5.1: Build target files section with RESOLVED paths
    target_files_desc = ""
    if resolved_target_files:
        target_lines = []
        for rtf in resolved_target_files:
            size_kb = rtf.get('size_bytes', 0) / 1024
            target_lines.append(
                f"- RESOLVED: `{rtf['mentioned']}` → `{rtf['resolved_path']}` "
                f"({size_kb:.1f}KB)"
            )
        target_files_desc = (
            "\n\nTARGET FILES (PRE-RESOLVED — use these EXACT paths in EVIDENCE_REQUESTs):\n"
            + chr(10).join(target_lines)
            + "\n\nIMPORTANT: The paths above are ground truth from the filesystem. "
            "Do NOT guess alternative paths for these files. Use the resolved paths exactly "
            "as shown when emitting EVIDENCE_REQUESTs to read them."
        )

    user_prompt = f"""Feature Request:
{goal}

Full Description:
{what_to_do}
{target_files_desc}

Tech Stack:
{chr(10).join(stack_desc) if stack_desc else 'Not detected'}

Existing Integration Points (VERIFIED — these files exist and can be read via EVIDENCE_REQUESTs):
{chr(10).join(integration_desc) if integration_desc else 'None found'}

Suggested New Files:
{chr(10).join(f'- {f}' for f in suggested_files) if suggested_files else 'None'}

Constraints:
{constraints_desc}

Please provide your structured analysis."""

    # v2.1: Build attempt list — primary model, then fallbacks on timeout
    attempts = [(use_provider, use_model, _CREATE_ANALYSIS_TIMEOUT)]
    for fb_provider, fb_model in _FALLBACK_MODELS:
        # Don't add fallback if it's the same as primary
        if fb_provider != use_provider or fb_model != use_model:
            attempts.append((fb_provider, fb_model, 90))  # Fallbacks get standard timeout
    
    for attempt_idx, (attempt_provider, attempt_model, attempt_timeout) in enumerate(attempts):
        is_retry = attempt_idx > 0
        
        try:
            if is_retry:
                logger.info("[simple_create] v2.1 RETRY with fallback: %s/%s (timeout=%ds)",
                           attempt_provider, attempt_model, attempt_timeout)
                print(f"[simple_create] v2.1 RETRY: {attempt_provider}/{attempt_model} (timeout={attempt_timeout}s)")
            else:
                logger.info("[simple_create] v2.1 LLM ANALYSIS CALL: provider=%s, model=%s, timeout=%ds",
                           attempt_provider, attempt_model, attempt_timeout)
                print(f"[simple_create] v2.1 LLM ANALYSIS: calling {attempt_provider}/{attempt_model} (timeout={attempt_timeout}s)")
            
            result = await llm_call_func(
                provider_id=attempt_provider,
                model_id=attempt_model,
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=CREATE_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=8192,  # v4.1: Increased from 4096 — LLM needs room for analysis + YAML ERs
                timeout_seconds=attempt_timeout,
            )
            
            if result.is_success() and result.content:
                analysis = result.content.strip()
                model_label = f"{attempt_provider}/{attempt_model}"
                if is_retry:
                    model_label += " (fallback)"
                logger.info("[simple_create] v2.1 LLM ANALYSIS SUCCESS: %d chars via %s", len(analysis), model_label)
                print(f"[simple_create] v2.1 LLM ANALYSIS SUCCESS: {len(analysis)} chars via {model_label}")
                return analysis
            
            error_msg = getattr(result, 'error_message', 'Unknown error')
            logger.warning("[simple_create] v2.1 LLM ANALYSIS FAILED (%s/%s): %s",
                          attempt_provider, attempt_model, error_msg)
            print(f"[simple_create] v2.1 LLM ANALYSIS FAILED: {error_msg}")
            
            # Only retry on timeout-like errors
            is_timeout = 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower()
            if not is_timeout:
                # Non-timeout error — don't bother retrying with a different model
                return None
            
        except Exception as e:
            error_str = str(e)
            logger.warning("[simple_create] v2.1 LLM ANALYSIS EXCEPTION (%s/%s): %s",
                          attempt_provider, attempt_model, error_str)
            print(f"[simple_create] v2.1 LLM ANALYSIS EXCEPTION: {error_str}")
            
            is_timeout = 'timeout' in error_str.lower() or 'timed out' in error_str.lower()
            if not is_timeout:
                return None
    
    # All attempts failed
    logger.warning("[simple_create] v2.1 ALL LLM ATTEMPTS FAILED, falling back to template")
    print("[simple_create] v2.1 ALL LLM ATTEMPTS FAILED — using template fallback")
    return None

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
