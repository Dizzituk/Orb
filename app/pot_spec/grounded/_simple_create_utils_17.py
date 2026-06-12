# Purpose: simple create utils 17
# Called-by: app.pot_spec.grounded._simple_create_utils_14, app.pot_spec.grounded._simple_create_utils_15, app.pot_spec.grounded._simple_create_utils_16, app.pot_spec.grounded._spec_runner_utils_12 (+1 more)
# Depends-on: app.pot_spec.grounded._dependency_presence, app.pot_spec.grounded._sbx_fs, app.pot_spec.grounded._simple_create_review, app.pot_spec.grounded._simple_create_utils_12 (+8 more)
# Last-renovated: 2026-06-11
from __future__ import annotations
import logging
from app.pot_spec.grounded._simple_create_utils_12 import _CREATE_ANALYSIS_MODEL, _FALLBACK_MODELS
from app.pot_spec.grounded._simple_create_utils_13 import _resolve_mentioned_files
from app.pot_spec.grounded._simple_create_utils_14 import _CREATE_ANALYSIS_TIMEOUT, _extract_constraints, _extract_task_keywords, _suggest_new_files
from app.pot_spec.grounded._simple_create_utils_15 import _extract_patterns, _find_integration_points, build_create_spec
from app.pot_spec.grounded._simple_create_utils_16 import CreateEvidence, IntegrationPoint, _detect_tech_stack
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.pot_spec.grounded._sbx_fs import _sbx_isfile, _sbx_isdir
from app.pot_spec.grounded._simple_create_utils_18 import inject_brief_mentioned_roots, priority_sort_project_paths, run_dependency_gate, filter_phantom_files
from app.pot_spec.grounded._stage_reasoning import spec_gate_reasoning, spec_gate_timeout_seconds
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.


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

# ── v2.2: Frontend root injection ────────────────────────────────────
# Visual/UI terms that imply frontend work without needing the literal
# word "frontend".  These are semantic signals — things humans say when
# describing how something LOOKS, not how the backend works.
_VISUAL_INTENT_SIGNALS = {
    # Layout & structure
    'dashboard', 'card', 'cards', 'grid', 'layout', 'sidebar', 'panel',
    'tab', 'tabs', 'modal', 'dialog', 'drawer', 'navbar', 'header',
    'footer', 'page', 'view', 'screen',
    # Style & appearance
    'dark theme', 'dark mode', 'light theme', 'colour', 'color',
    'hex', '#111', '#1e1', 'background', 'accent', 'icon', 'icons',
    'font', 'typography', 'css', 'tailwind', 'styled',
    # Interaction & UX
    'button', 'click', 'hover', 'toggle', 'dropdown', 'input field',
    'form', 'checkbox', 'progress bar', 'percentage bar', 'slider',
    'tooltip', 'notification', 'toast',
    # Component names (common in React/Vue)
    '.tsx', '.jsx', '.vue', '.svelte', 'component', 'components',
    'react', 'vue', 'angular',
    # Visual concepts
    'how it looks', 'visual', 'display', 'render', 'ui design',
    'user interface', 'screenshot', 'mockup', 'wireframe',
    'aesthetic', 'style', 'theme', 'branded',
}


def _inject_frontend_root_if_needed(
    project_paths: list[str],
    combined_text: str,
) -> list[str]:
    """Ensure frontend root is in project_paths when the job implies UI work.

    Uses semantic signals (visual terms, component names, style references)
    rather than requiring the user to literally say 'frontend'.  If the job
    description talks about how something LOOKS, it needs frontend files.

    Returns project_paths unchanged if no frontend injection is needed,
    or a new list with the frontend root appended.
    """
    from ._spec_runner_utils_13 import _discover_project_roots

    discovery = _discover_project_roots()
    fe_roots = discovery.get('frontend_paths', [])
    if not fe_roots:
        return project_paths  # No frontend root known — nothing to inject

    # Already have a frontend root?
    paths_lower = {p.lower().replace('/', '\\').rstrip('\\') for p in project_paths}
    for fr in fe_roots:
        if fr.lower().replace('/', '\\').rstrip('\\') in paths_lower:
            return project_paths  # Frontend root already present

    # Check for visual/UI intent in the combined text
    text_lower = combined_text.lower()
    matched = [sig for sig in _VISUAL_INTENT_SIGNALS if sig in text_lower]

    if matched:
        new_paths = list(project_paths) + list(fe_roots)
        print(
            f"[simple_create] v2.2 FRONTEND ROOT INJECTED: {fe_roots} "
            f"(visual signals: {matched[:5]})"
        )
        logger.info(
            "[simple_create] v2.2 Frontend root injected: %s (signals: %s)",
            fe_roots, matched[:5],
        )
        return new_paths

    return project_paths


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
    patterns: Optional[Dict[str, str]] = None,  # v2.8 Fix C
    project_paths: Optional[List[str]] = None,  # v2.8 Fix D
) -> Optional[str]:
    """
    v2.1: Use an LLM to analyze the feature request.
    
    Model selection priority:
    1. ASTRA_CREATE_ANALYSIS_MODEL env var (if set)
    2. Allocated model from spec_gate_stream
    3. On timeout: retry with faster fallback model
    
    Returns LLM-generated analysis or None if all attempts fail.
    """
    from .simple_create import CREATE_ANALYSIS_SYSTEM_PROMPT
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


    # v2.10 Fix C: format extracted patterns for prompt injection.
    # Sort by category importance so the most architecturally relevant patterns
    # appear first if truncation occurs. Cap at 25 — generous enough for 5+ files
    # across 5 categories without flooding the prompt.
    patterns_desc = ""
    if patterns:
        try:
            _priority = ("persistence", "http_client", "state_mgmt", "di",
                         "router", "models", "component", "async_style", "import_block")
            def _sort_key(item):
                k = item[0]
                cat = k.split(":", 1)[0]
                try:
                    return (_priority.index(cat), k)
                except ValueError:
                    return (len(_priority), k)
            _sorted = sorted(patterns.items(), key=_sort_key)
            _lines = []
            for k, v in _sorted[:25]:
                _v = (v or "").strip().replace("\n", " ")[:200]
                _lines.append(f"- {k}: {_v}")
            patterns_desc = "\n".join(_lines)
        except Exception:
            pass


    # v2.8 Fix D: walk project_paths and enumerate declared dependencies
    declared_deps_desc = ""
    try:
        from app.pot_spec.grounded._dependency_presence import get_declared_dependencies
        _all_deps = set()
        for _root in (project_paths or []):
            if _root:
                try:
                    _all_deps.update(get_declared_dependencies(_root))
                except Exception:
                    continue
        if _all_deps:
            _sorted = sorted(_all_deps)[:50]  # Cap at 50 to keep prompt focused
            declared_deps_desc = ", ".join(_sorted)
            if len(_all_deps) > 50:
                declared_deps_desc += f" ... (+{len(_all_deps) - 50} more)"
    except Exception as _e:
        logger.warning("[simple_create] v2.8 Fix D dep enumeration failed: %s", _e)
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

Existing Patterns (from project — match these in any new code):
{patterns_desc if patterns_desc else 'None extracted'}

Declared Dependencies (libraries the project ALREADY has — use ONLY these):
{declared_deps_desc if declared_deps_desc else 'Not enumerated'}

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
                temperature=1.0,  # stripped by registry when the model requires it
                max_tokens=16384,  # floored upward by registry per effort level
                timeout_seconds=max(attempt_timeout, spec_gate_timeout_seconds()),
                reasoning=spec_gate_reasoning(),  # v2.8: env-overridable (SPEC_GATE_REASONING_EFFORT)
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
    from .simple_create import _fulfil_evidence_requests
    logger.info("[simple_create] v2.0 Building LLM-grounded CREATE spec")
    print(f"[simple_create] v2.0 GROUNDED CREATE: {goal[:60]}...")

    # v2.2: FRONTEND ROOT INJECTION
    # If the job description implies frontend work (visual terms, UI concepts,
    # component names, style references) but no frontend root is in project_paths,
    # inject the discovered frontend root so the LLM sees real frontend integration
    # points instead of inventing paths under the backend root.
    project_paths = _inject_frontend_root_if_needed(project_paths, combined_text=f"{goal} {what_to_do}")
    
    # v2.0: Extract CONCEPTS (not raw keywords)
    combined_text = f"{goal} {what_to_do}"

    # v2.3: Brief-mentioned root injection — see _simple_create_utils_18.inject_brief_mentioned_roots
    project_paths, _added_roots = inject_brief_mentioned_roots(project_paths, combined_text)
    if _added_roots:
        print(f"[simple_create] v2.3 BRIEF-ROOT INJECTION: added {len(_added_roots)} root(s): {_added_roots}")
        logger.info("[simple_create] v2.3 Brief-root injection added: %s", _added_roots)
    concepts = _extract_task_keywords(combined_text)
    print(f"[simple_create] v2.0 Concepts: {concepts[:10]}")
    
    # v2.0: Extract constraints
    constraints = _extract_constraints(combined_text)
    print(f"[simple_create] v2.0 Constraints: {constraints}")
    
    # v2.4: Priority-sort — see _simple_create_utils_18.priority_sort_project_paths
    project_paths = priority_sort_project_paths(project_paths)

    # Detect tech stack for each project path
    tech_stack = TechStack()
    for path in project_paths:
        if _sbx_isdir(path):
            detected = _detect_tech_stack(path, sandbox_client)
            for attr in ['frontend_framework', 'frontend_language', 'backend_framework',
                        'backend_language', 'styling', 'state_management', 'api_pattern']:
                _new_val = getattr(detected, attr)
                _cur_val = getattr(tech_stack, attr)
                if not _new_val:
                    continue
                # v2.4: Android-specific values always override React/Vite/Vue/etc.
                # Mobile-specificity > web defaults when we're building for a mobile target.
                _is_mobile = 'Android' in str(_new_val) or 'Kotlin' in str(_new_val) or 'Gradle' in str(_new_val)
                if not _cur_val or _is_mobile:
                    setattr(tech_stack, attr, _new_val)
    
    print(f"[simple_create] v2.0 Tech stack: {tech_stack.frontend_framework}/{tech_stack.backend_framework}")
    
    # v2.0: Find integration points using CONCEPTS (not raw keywords)
    all_points = []
    for path in project_paths:
        if _sbx_isdir(path):
            points = _find_integration_points(path, concepts, sandbox_client)
            all_points.extend(points)
    
    print(f"[simple_create] v2.0 Found {len(all_points)} integration points")

    # v2.6: Phantom-file filter — see _simple_create_utils_18.filter_phantom_files
    all_points = filter_phantom_files(all_points, tech_stack)
    
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
                patterns=patterns,  # v2.8 Fix C: existing patterns for prompt
                project_paths=project_paths,  # v2.8 Fix D: for declared deps lookup
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

            # Self-review pass: audit the finished draft against the intent and
            # verify every reuse-target is actually grounded (read), not guessed.
            # Any core reuse-target only assumed triggers a fresh EVIDENCE_REQUEST
            # so the real file is read before the spec is finalised.
            if llm_analysis:
                from ._simple_create_review import run_spec_self_review
                llm_analysis = await run_spec_self_review(
                    llm_analysis=llm_analysis,
                    provider_id=provider_id,
                    model_id=model_id,
                    llm_call_func=llm_call_func,
                    project_paths=project_paths,
                    goal=goal,
                    what_to_do=what_to_do,
                    integration_points=all_points,
                )
    else:
        print(f"[simple_create] v2.0 NO LLM: provider_id={provider_id}, model_id={model_id}")
    

    # v2.5: Dependency gate — see _simple_create_utils_18.run_dependency_gate
    llm_analysis = run_dependency_gate(llm_analysis, project_paths)
    # Build evidence bundle
    evidence = CreateEvidence(
        tech_stack=tech_stack,
        integration_points=all_points,
        existing_patterns=patterns,
        suggested_files=suggested_files,
        keywords_found={c: [] for c in concepts},
        constraints=constraints,
        llm_analysis=llm_analysis,
        resolved_target_files=resolved_target_files,  # v5.2: Survive into spec independently
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
