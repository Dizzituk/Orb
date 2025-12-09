# FILE: app/llm/router.py
"""
LLM Router (PHASE 4 - v0.13.7)

Uses job_classifier for automatic routing based on message content and attachments.

v0.13.7 Changes:
- CRITICAL FIX: Separate critique prompts for architecture_design vs security_review
- Architecture critique HARD-CODES environment constraints (single-host, local repos, solo dev)
- Architecture critique AGGRESSIVELY prevents enterprise infra drift (no K8s/VLANs/containers unless explicit)
- Security critique keeps strong hardening focus (unchanged)
- Environment context object passed to critic for all architecture jobs
- Critique prompt selection based on normalized job_type

v0.13.6 Changes:
- CRITICAL FIX: High-stakes critique pipeline now triggers for architecture jobs
- Added normalize_job_type_for_high_stakes() to map orchestrator → architecture_design
- Architecture design requests with file uploads now route through critique pipeline

8-ROUTE CLASSIFICATION SYSTEM:
- CHAT_LIGHT → OpenAI (gpt-4.1-mini) - casual chat
- TEXT_HEAVY → OpenAI (gpt-4.1) - heavy text, text-only PDFs
- CODE_MEDIUM → Anthropic Sonnet - scoped code (1-3 files)
- ORCHESTRATOR → Anthropic Opus - architecture, multi-file
- IMAGE_SIMPLE → Gemini Flash - simple screenshots
- IMAGE_COMPLEX → Gemini 2.5 Pro - PDFs with images, multi-image
- VIDEO_HEAVY → Gemini 3.0 Pro - video >10MB OR deep semantic analysis
- OPUS_CRITIC → Gemini 3.0 Pro - explicit Opus review only

HARD RULES:
- Images/video NEVER go to Claude
- PDFs NEVER go to Claude
- opus.critic is EXPLICIT ONLY (no fuzzy matching)

ATTACHMENT SAFETY RULE (v0.13.1):
- If attachments present AND no job_type specified:
  - Classify using classifier
  - Do NOT route to Claude unless classifier explicitly returns code.medium or orchestrator

HIGH-STAKES CRITIQUE PIPELINE (v0.13.7):
- When Opus used for high-stakes tasks: Opus draft → Gemini 3 critique → Opus revision
- Triggers when: Anthropic + Opus + high-stakes job type + response >= 1500 chars
- High-stakes types: architecture_design, security_review, complex_code_change, etc.
- Critique prompt customized by job_type with HARD-CODED constraints for architecture
- Environment context passed to critic for architecture jobs
- Graceful degradation: Returns original draft on any failure

BUG FIXES:
- v0.13.2.1: Fixed call_opus_revision() to not use system role in messages
- v0.13.2.2: Added print() statements for all [critic] logs (logger wasn't showing)

Provider selection priority:
1. task.force_provider (explicit override from caller)
2. Job classifier (analyzes message + attachments)
3. Attachment safety rule enforcement
4. task.provider (legacy hint)
5. Fallback to CHAT_LIGHT
"""
import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4

from app.llm.schemas import (
    LLMTask,
    LLMResult,
    JobType,
    Provider,
    RoutingConfig,
    RoutingOptions,
    RoutingDecision,
)

# Phase 4 imports
from app.jobs.schemas import (
    JobEnvelope,
    JobType as Phase4JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
    validate_job_envelope,
    ValidationError,
)
from app.providers.registry import llm_call as registry_llm_call

# Job classifier (8-route system)
from app.llm.job_classifier import (
    classify_job,
    classify_and_route as classifier_classify_and_route,
    get_routing_for_job_type,
    get_model_config,
    is_claude_forbidden,
    is_claude_allowed,
    prepare_attachments,
)

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER DEBUG MODE
# =============================================================================
# Set ORB_ROUTER_DEBUG=1 in .env to enable detailed routing diagnostics
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"

def _debug_log(msg: str):
    """Print debug message if ROUTER_DEBUG is enabled."""
    if ROUTER_DEBUG:
        print(f"[router-debug] {msg}")


# =============================================================================
# HIGH-STAKES CRITIQUE PIPELINE CONFIGURATION (v0.13.7)
# =============================================================================

# High-stakes job types that trigger critique pipeline
# These are FINE-GRAINED strings from the classifier, NOT the coarse 5-type enum
HIGH_STAKES_JOB_TYPES = {
    "architecture_design",       # System architecture design
    "big_architecture",          # Large-scale architecture changes
    "high_stakes_infra",         # Infrastructure with high impact
    "security_review",           # Security audits and reviews
    "privacy_sensitive_change",  # Privacy-impacting changes
    "security_sensitive_change", # Alias for security work
    "complex_code_change",       # Complex multi-file code changes
    "implementation_plan",       # Detailed implementation plans
    "spec_review",               # Specification reviews
    "architecture",              # Legacy alias
    "deep_planning",             # Deep strategic planning
    "orchestrator",              # Generic orchestrator (will be normalized)
}

# Minimum response length to trigger critique (characters, ~250 tokens)
MIN_CRITIQUE_CHARS = int(os.getenv("ORB_MIN_CRITIQUE_CHARS", "1500"))

# Gemini critic model (must be gemini-3-pro or better)
GEMINI_CRITIC_MODEL = os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3-pro")

# Token limits for critique pipeline (cost control)
GEMINI_CRITIC_MAX_TOKENS = int(os.getenv("GEMINI_CRITIC_MAX_TOKENS", "1024"))
OPUS_REVISION_MAX_TOKENS = int(os.getenv("OPUS_REVISION_MAX_TOKENS", "2048"))


# =============================================================================
# ENVIRONMENT CONTEXT (v0.13.7)
# =============================================================================

def get_environment_context() -> Dict[str, Any]:
    """
    Get current environment context for architecture critique.
    
    v0.13.7: NEW - Hard-coded constraints to prevent enterprise infra drift
    
    Returns:
        Environment context dict with deployment constraints
    """
    return {
        "deployment_type": "single_host",
        "os": "Windows 11",
        "repos": ["D:\\Orb\\", "D:\\SandboxOrb\\"],
        "team_size": "solo_developer",
        "infrastructure": "local_only",
        "phase": "early_self_improvement_pipeline",
        "constraints": {
            "no_kubernetes": True,
            "no_docker_orchestration": True,
            "no_multi_host": True,
            "no_vlans": True,
            "no_external_ci": True,
            "no_separate_vms": True,
            "prefer_local_controls": True,
            "prefer_file_permissions": True,
            "prefer_process_isolation": True,
        },
        "acceptable_infra": [
            "Windows security features",
            "File-level permissions",
            "Process isolation",
            "Local sandboxing",
            "Windows Credential Manager",
            "Local SQLite encryption",
        ],
        "forbidden_unless_explicit": [
            "Kubernetes",
            "Docker Swarm",
            "Multiple VLANs",
            "External CI runners",
            "Separate VMs/hosts",
            "Egress proxies",
            "Documentation mirrors",
            "Container orchestration",
        ]
    }


def normalize_job_type_for_high_stakes(job_type_str: str, reason: str = "") -> str:
    """
    Normalize job type to specific high-stakes string.
    
    Maps generic types like 'orchestrator' to specific high-stakes types
    based on classification reason.
    
    v0.13.6: Enables high-stakes critique pipeline for architecture jobs
    that come through as 'orchestrator'.
    
    Args:
        job_type_str: Raw job type string from classifier
        reason: Classification reason (may contain keywords)
    
    Returns:
        Normalized job type string (for high-stakes matching)
    """
    # Already a specific high-stakes type - pass through
    if job_type_str in HIGH_STAKES_JOB_TYPES and job_type_str != "orchestrator":
        return job_type_str
    
    # Normalize 'orchestrator' based on reason
    if job_type_str == "orchestrator":
        reason_lower = reason.lower()
        
        # Check for architecture keywords
        architecture_keywords = ["architecture", "system design", "architect", "system architecture"]
        if any(kw in reason_lower for kw in architecture_keywords):
            print(f"[router] Normalized orchestrator → architecture_design (reason: {reason[:80]})")
            return "architecture_design"
        
        # Check for security keywords
        security_keywords = ["security", "security review", "security audit"]
        if any(kw in reason_lower for kw in security_keywords):
            print(f"[router] Normalized orchestrator → security_review (reason: {reason[:80]})")
            return "security_review"
        
        # Check for infrastructure keywords
        infra_keywords = ["infrastructure", "infra", "deployment"]
        if any(kw in reason_lower for kw in infra_keywords):
            print(f"[router] Normalized orchestrator → high_stakes_infra (reason: {reason[:80]})")
            return "high_stakes_infra"
        
        # Default: treat orchestrator as architecture_design
        print(f"[router] Normalized orchestrator → architecture_design (default)")
        return "architecture_design"
    
    # Not a high-stakes job
    return job_type_str


def is_high_stakes_job(job_type_str: str) -> bool:
    """Check if job type string is in high-stakes set."""
    return job_type_str in HIGH_STAKES_JOB_TYPES


def is_opus_model(model_id: str) -> bool:
    """Check if model is Opus (not Sonnet)."""
    return "opus" in model_id.lower()


def is_long_enough_for_critique(text: str) -> bool:
    """Check if response is long enough to warrant critique."""
    return len(text or "") >= MIN_CRITIQUE_CHARS


# =============================================================================
# CRITIQUE PROMPT TEMPLATES (v0.13.7)
# =============================================================================

def build_critique_prompt_for_architecture(
    draft_text: str, 
    original_request: str,
    env_context: Dict[str, Any]
) -> str:
    """
    Build critique prompt for architecture design tasks.
    
    v0.13.7: HARD-CODED constraints to prevent enterprise infra drift
    
    Focus:
    - AGGRESSIVELY respect current environment (single host, local repos, solo dev)
    - EXPLICITLY forbid enterprise infra unless user asked for it
    - Prefer pragmatic, implementable designs
    - Flag over-engineering directly
    - Separate "future hardening" from core design
    
    Args:
        draft_text: Opus draft response
        original_request: User's original request
        env_context: Environment context dict with constraints
    
    Returns:
        Formatted critique prompt with HARD constraints
    """
    forbidden_infra = env_context.get("forbidden_unless_explicit", [])
    acceptable_infra = env_context.get("acceptable_infra", [])
    
    return f"""You are reviewing an architecture design proposal.

ORIGINAL REQUEST:
{original_request}

PROPOSED ARCHITECTURE:
{draft_text}

===== MANDATORY ENVIRONMENT CONSTRAINTS =====

DEPLOYMENT CONTEXT (DO NOT DEVIATE):
- Single Windows 11 workstation (D:\\Orb\\, D:\\SandboxOrb\\)
- Solo developer, limited time/budget
- No existing Kubernetes, Docker orchestration, or multi-host infrastructure
- Current setup: FastAPI backend + Electron desktop client
- Phase: Early self-improvement pipeline

HARD CONSTRAINTS (ENFORCED):
You MUST NOT recommend the following unless the user explicitly requested them:
{chr(10).join(f'  - {item}' for item in forbidden_infra)}

ACCEPTABLE SOLUTIONS (PREFER THESE):
{chr(10).join(f'  - {item}' for item in acceptable_infra)}

===== YOUR REVIEW TASK =====

Critically review the proposed architecture with these priorities:

1. **OVER-ENGINEERING CHECK (HIGHEST PRIORITY)**
   - Does this design introduce Kubernetes, containers, VLANs, separate VMs, external CI, or doc mirrors?
   - If YES and the user didn't ask for them → FLAG AS OVER-ENGINEERED
   - Are there simpler alternatives using file permissions, process isolation, or Windows security features?
   - Is the complexity justified for a solo developer on a single machine?

2. **ENVIRONMENT FIT**
   - Can this be implemented by one person in 2-4 weeks?
   - Does it respect the single-host deployment context?
   - Are safety goals achievable with local controls instead of network segmentation?

3. **TECHNICAL CORRECTNESS**
   - Are there actual errors in the design logic?
   - Are proposed patterns and technologies sound?
   - Are critical components missing for stated goals?

4. **PRAGMATIC SAFETY**
   - Are security invariants appropriate for local deployment (not theoretical enterprise)?
   - Can goals be met with simpler solutions?
   - Is defense-in-depth achieved without infrastructure sprawl?

===== OUTPUT RULES =====

If the design is OVER-ENGINEERED:
  → State it directly: "This design is over-engineered for your single-host setup."
  → Explain which components are unnecessary (K8s, containers, VLANs, etc.)
  → Suggest simpler alternatives using local controls

If the design mentions enterprise hardening:
  → Require it be moved to a clearly separated "Future Hardening (Optional)" section
  → Do NOT allow it mixed into the core implementable design

If the design is appropriate:
  → Confirm it respects environment constraints
  → Highlight pragmatic safety approaches
  → Suggest minor improvements if needed

Keep your critique under 800 words. Be direct and specific. Your job is to prevent architectural drift toward enterprise infrastructure that the user cannot implement."""


def build_critique_prompt_for_security(draft_text: str, original_request: str) -> str:
    """
    Build critique prompt for security review tasks.
    
    Focus:
    - Strong threat modeling and attack surface analysis
    - Aggressive hardening recommendations
    - Deep security verification
    - No holds barred on security concerns
    
    v0.13.7: Security critique keeps strong posture (unchanged from original)
    """
    return f"""You are performing a security review.

ORIGINAL REQUEST:
{original_request}

SECURITY ANALYSIS:
{draft_text}

YOUR TASK:
Provide a critical security review focusing on:

1. **Threat Model Completeness**
   - Are all attack vectors identified?
   - Are there missing threat scenarios?
   - Is the risk assessment accurate?

2. **Security Controls**
   - Are proposed controls sufficient?
   - Are there bypasses or weaknesses?
   - What additional hardening is needed?

3. **Implementation Risks**
   - Are there security pitfalls in the implementation approach?
   - Are cryptographic/auth patterns correct?
   - Are there timing/side-channel concerns?

4. **Defense in Depth**
   - Are there sufficient layers of security?
   - What happens if one layer fails?
   - Are there single points of failure?

Be aggressive in identifying security issues. Recommend hardening measures even if they add complexity.
This is a security review - err on the side of caution.

Keep your critique under 800 words. Be direct and specific about security concerns."""


def build_critique_prompt_for_general(draft_text: str, original_request: str, job_type_str: str) -> str:
    """
    Build critique prompt for general high-stakes tasks.
    
    v0.13.7: Fallback for other high-stakes types
    """
    return f"""You are reviewing high-stakes technical work.

ORIGINAL REQUEST:
{original_request}

PROPOSED SOLUTION:
{draft_text}

YOUR TASK:
Provide a critical technical review focusing on:
1. Correctness and accuracy
2. Completeness and thoroughness
3. Potential risks or oversights
4. Areas needing improvement

Be direct and specific. Identify concrete issues, not generic praise.
Keep your critique under 800 words."""


def build_critique_prompt(
    draft_text: str, 
    original_request: str, 
    job_type_str: str,
    env_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build critique prompt based on job type.
    
    v0.13.7: Routes to appropriate template with environment context for architecture
    
    Args:
        draft_text: The Opus draft response
        original_request: The user's original request
        job_type_str: Normalized job type (architecture_design, security_review, etc.)
        env_context: Environment context dict (for architecture jobs)
    
    Returns:
        Formatted critique prompt for Gemini
    """
    # Architecture design tasks - respect environment constraints
    if job_type_str in ["architecture_design", "big_architecture", "high_stakes_infra", 
                         "architecture", "orchestrator"]:
        if not env_context:
            env_context = get_environment_context()
        return build_critique_prompt_for_architecture(draft_text, original_request, env_context)
    
    # Security review tasks - aggressive hardening focus
    elif job_type_str in ["security_review", "security_sensitive_change", 
                           "privacy_sensitive_change"]:
        return build_critique_prompt_for_security(draft_text, original_request)
    
    # Other high-stakes tasks - general critique
    else:
        return build_critique_prompt_for_general(draft_text, original_request, job_type_str)


# =============================================================================
# DEFAULT MODELS PER PROVIDER (8-route)
# =============================================================================

DEFAULT_MODELS: Dict[str, str] = {
    "openai": os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini"),
    "openai_heavy": os.getenv("OPENAI_MODEL_HEAVY_TEXT", "gpt-4.1"),
    "anthropic": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
    "anthropic_opus": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20250514"),
    "google": os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash"),
    "google_complex": os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro"),
    "google_video": os.getenv("GEMINI_VIDEO_HEAVY_MODEL", "gemini-3.0-pro-preview"),
    "google_critic": os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3-pro"),
}


# =============================================================================
# JOB TYPE MAPPING (Legacy → Phase 4)
# =============================================================================

_LEGACY_TO_PHASE4_JOB_TYPE: Dict[str, Phase4JobType] = {
    # TEXT types
    JobType.TEXT_ADMIN.value: Phase4JobType.CHAT_SIMPLE,
    JobType.CASUAL_CHAT.value: Phase4JobType.CHAT_SIMPLE,
    JobType.CHAT_LIGHT.value: Phase4JobType.CHAT_SIMPLE,
    JobType.TEXT_HEAVY.value: Phase4JobType.CHAT_SIMPLE,
    JobType.QUICK_QUESTION.value: Phase4JobType.CHAT_RESEARCH,
    JobType.PROMPT_SHAPING.value: Phase4JobType.CHAT_SIMPLE,
    JobType.SUMMARY.value: Phase4JobType.CHAT_SIMPLE,
    JobType.EXPLANATION.value: Phase4JobType.CHAT_SIMPLE,
    
    # Code types
    JobType.SMALL_CODE.value: Phase4JobType.CODE_SMALL,
    JobType.CODE_MEDIUM.value: Phase4JobType.CODE_SMALL,
    JobType.SIMPLE_CODE_CHANGE.value: Phase4JobType.CODE_SMALL,
    JobType.SMALL_BUGFIX.value: Phase4JobType.CODE_SMALL,
    JobType.BIG_ARCHITECTURE.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.ORCHESTRATOR.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.COMPLEX_CODE_CHANGE.value: Phase4JobType.CODE_REPO,
    JobType.CODEGEN_FULL_FILE.value: Phase4JobType.CODE_REPO,
    
    # Architecture / critique
    JobType.ARCHITECTURE_DESIGN.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.CODE_REVIEW.value: Phase4JobType.CRITIQUE_REVIEW,
    JobType.SPEC_REVIEW.value: Phase4JobType.CRITIQUE_REVIEW,
    JobType.HIGH_STAKES_INFRA.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.OPUS_CRITIC.value: Phase4JobType.CRITIQUE_REVIEW,
    
    # Vision types
    JobType.SIMPLE_VISION.value: Phase4JobType.VISION_SIMPLE,
    JobType.IMAGE_SIMPLE.value: Phase4JobType.VISION_SIMPLE,
    JobType.HEAVY_MULTIMODAL_CRITIQUE.value: Phase4JobType.VISION_COMPLEX,
    JobType.IMAGE_COMPLEX.value: Phase4JobType.VISION_COMPLEX,
    JobType.VIDEO_HEAVY.value: Phase4JobType.VISION_COMPLEX,
}


def _map_to_phase4_job_type(job_type: JobType) -> Phase4JobType:
    """Map our JobType to Phase 4 JobType."""
    return _LEGACY_TO_PHASE4_JOB_TYPE.get(
        job_type.value,
        Phase4JobType.CHAT_SIMPLE,
    )


def _default_importance_for_job_type(job_type: JobType) -> Importance:
    """Determine default Importance based on JobType."""
    if job_type in {JobType.BIG_ARCHITECTURE, JobType.HIGH_STAKES_INFRA, JobType.ORCHESTRATOR}:
        return Importance.HIGH
    if job_type in {JobType.SMALL_CODE, JobType.HEAVY_MULTIMODAL_CRITIQUE, JobType.CODE_MEDIUM}:
        return Importance.MEDIUM
    return Importance.LOW


def _default_modalities_for_job_type(job_type: JobType) -> List[Modality]:
    """Determine default modalities based on JobType."""
    if job_type in {
        JobType.SIMPLE_VISION, JobType.HEAVY_MULTIMODAL_CRITIQUE,
        JobType.IMAGE_SIMPLE, JobType.IMAGE_COMPLEX, JobType.VIDEO_HEAVY,
    }:
        return [Modality.TEXT, Modality.IMAGE]
    return [Modality.TEXT]


# =============================================================================
# CLASSIFY AND ROUTE
# =============================================================================

def classify_and_route(
    task: LLMTask,
) -> Tuple[Provider, str, JobType, str]:
    """
    Classify task and determine routing using job_classifier.
    
    Uses 8-route system with HARD RULES enforcement.
    
    Args:
        task: LLMTask with messages and optional attachments
    
    Returns:
        (provider, model, job_type, reason)
    """
    # Extract message text from task
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    message_text = user_messages[-1].get("content", "") if user_messages else ""
    
    # Get requested type from task
    requested_type = task.job_type.value if task.job_type else None
    
    # Get metadata
    metadata = task.metadata or {}
    
    # Classify using job_classifier
    decision = classifier_classify_and_route(
        message=message_text,
        attachments=task.attachments,
        job_type=requested_type,
        metadata=metadata,
    )
    
    return (decision.provider, decision.model, decision.job_type, decision.reason)


# =============================================================================
# ENVELOPE SYNTHESIS
# =============================================================================

def synthesize_envelope_from_task(
    task: LLMTask,
    session_id: Optional[str] = None,
    project_id: int = 1,
) -> JobEnvelope:
    """
    Synthesize a JobEnvelope from LLMTask.
    
    v0.13.5: Now properly injects system_prompt and project_context into messages.
    """
    phase4_job_type = _map_to_phase4_job_type(task.job_type)
    importance = _default_importance_for_job_type(task.job_type)
    modalities = _default_modalities_for_job_type(task.job_type)

    routing = task.routing
    max_tokens = routing.max_tokens if routing else 8000
    max_cost = routing.max_cost_usd if routing else 1.0
    timeout = routing.timeout_seconds if routing else 60

    budget = JobBudget(
        max_tokens=max_tokens,
        max_cost_estimate=float(max_cost),
        max_wall_time_seconds=timeout,
    )

    
    # ==========================================================================
    # v0.13.5: BUILD COMPLETE MESSAGES ARRAY WITH CONTEXT INJECTION
    # ==========================================================================
    # Combine system_prompt + project_context into a system message,
    # then append the conversation messages
    
    final_messages = []
    
    # Build system message content
    system_parts = []
    if task.system_prompt:
        system_parts.append(task.system_prompt)
    if task.project_context:
        system_parts.append(task.project_context)
    
    # If we have system content, add it as the first message
    if system_parts:
        system_content = "\n\n".join(system_parts)
        final_messages.append({"role": "system", "content": system_content})
    
    # Add the conversation messages
    final_messages.extend(task.messages)

    envelope = JobEnvelope(
        job_id=str(uuid4()),
        session_id=session_id or f"legacy-{uuid4()}",
        project_id=project_id,
        job_type=phase4_job_type,
        importance=importance,
        data_sensitivity=DataSensitivity.INTERNAL,
        modalities_in=modalities,
        budget=budget,
        output_contract=OutputContract.TEXT_RESPONSE,
        messages=final_messages,  # v0.13.5: Use injected messages instead of task.messages
        metadata={
            "legacy_provider_hint": task.provider.value if task.provider else None,
            "legacy_routing": routing.model_dump() if routing else None,
            "legacy_context": task.project_context,
        },
        allow_multi_model_review=False,
        needs_tools=[],
    )

    try:
        validate_job_envelope(envelope)
    except ValidationError as ve:
        logger.warning("[router] Envelope validation failed: %s", ve)
        raise

    return envelope


# =============================================================================
# ATTACHMENT SAFETY CHECK
# =============================================================================

def _check_attachment_safety(
    task: LLMTask,
    decision: RoutingDecision,
    has_attachments: bool,
    job_type_specified: bool,
) -> Tuple[Provider, str, str]:
    """
    Enforce attachment safety rule.
    
    If attachments are present AND no job_type was specified:
    - Do NOT route to Claude unless classifier explicitly returned code.medium or orchestrator
    
    Returns:
        (provider, model, reason) - possibly modified from original decision
    """
    provider_id = decision.provider.value
    model_id = decision.model
    reason = decision.reason
    
    # Only apply safety rule when:
    # 1. Attachments are present
    # 2. No job_type was specified by the frontend
    # 3. Classifier returned a Claude route
    if has_attachments and not job_type_specified and provider_id == "anthropic":
        # Check if the classified job type is explicitly allowed for Claude
        if not is_claude_allowed(decision.job_type):
            # Redirect to GPT instead of Claude
            logger.warning(
                "[router] Attachment safety: Blocked Claude route for %s with attachments",
                decision.job_type.value,
            )
            provider_id = "openai"
            model_id = DEFAULT_MODELS["openai_heavy"]
            reason = f"Attachment safety: {decision.job_type.value} not allowed on Claude"
    
    return (provider_id, model_id, reason)


# =============================================================================
# HIGH-STAKES CRITIQUE PIPELINE (v0.13.7 - Environment Context)
# =============================================================================

async def call_gemini_critic(
    original_task: LLMTask,
    draft_result: LLMResult,
    job_type_str: str,
) -> Optional[LLMResult]:
    """
    Call Gemini 3 Pro to critique the Opus draft.
    
    v0.13.7: Now uses job-type-specific critique prompts with environment context
    
    Args:
        original_task: The original LLMTask
        draft_result: The Opus draft response
        job_type_str: Normalized job type string (architecture_design, security_review, etc.)
    
    Returns:
        LLMResult with critique, or None on failure
    """
    # Extract original user request
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""
    
    # v0.13.7: Get environment context for architecture jobs
    env_context = None
    if job_type_str in ["architecture_design", "big_architecture", "high_stakes_infra", 
                         "architecture", "orchestrator"]:
        env_context = get_environment_context()
        print(f"[critic] Passing environment context to architecture critique: {env_context['deployment_type']}, {env_context['team_size']}")
    
    # v0.13.7: Build critique prompt based on job type with environment context
    critique_prompt = build_critique_prompt(
        draft_text=draft_result.content,
        original_request=original_request,
        job_type_str=job_type_str,
        env_context=env_context
    )

    critique_messages = [
        {"role": "user", "content": critique_prompt}
    ]
    
    print(f"[critic] Calling Gemini 3 Pro for critique of {job_type_str} task")
    logger.info("[critic] Calling Gemini 3 Pro for critique of %s task", job_type_str)
    
    try:
        # Create minimal envelope for critic call
        critic_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=f"critic-{uuid4()}",
            project_id=1,
            job_type=Phase4JobType.CRITIQUE_REVIEW,
            importance=Importance.MEDIUM,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=GEMINI_CRITIC_MAX_TOKENS,
                max_cost_estimate=0.05,
                max_wall_time_seconds=30,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=critique_messages,
            metadata={"critique_for_job_type": job_type_str},
            allow_multi_model_review=False,
            needs_tools=[],
        )
        
        result = await registry_llm_call(
            provider_id="google",
            model_id=GEMINI_CRITIC_MODEL,
            messages=critique_messages,
            job_envelope=critic_envelope,
        )
        
        if not result.is_success() or not result.content:
            logger.warning("[critic] Gemini critic failed or returned empty")
            return None
        
        print(f"[critic] Gemini 3 Pro critique completed: {len(result.content)} chars")
        logger.info("[critic] Gemini 3 Pro critique completed: %d chars", len(result.content))
        
        # Return as LLMResult for consistency
        return LLMResult(
            content=result.content,
            provider="google",
            model=GEMINI_CRITIC_MODEL,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )
        
    except Exception as exc:
        logger.exception("[critic] Gemini critic call failed: %s", exc)
        return None


async def call_opus_revision(
    original_task: LLMTask,
    draft_result: LLMResult,
    critique_result: LLMResult,
    opus_model_id: str,
) -> Optional[LLMResult]:
    """
    Call Opus to revise its draft based on Gemini's critique.
    
    Args:
        original_task: The original LLMTask
        draft_result: The Opus draft response
        critique_result: The Gemini critique
        opus_model_id: Opus model identifier
    
    Returns:
        LLMResult with revised answer, or None on failure
    """
    # Extract original user request
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""
    
    # Build revision prompt (all in user message - Anthropic doesn't accept system role)
    revision_prompt = f"""You are revising your own previous answer based on external critique. Address the valid concerns raised while maintaining your technical accuracy. Produce an improved final answer.

ORIGINAL REQUEST:
{original_request}

YOUR DRAFT ANSWER:
{draft_result.content}

INDEPENDENT CRITIQUE:
{critique_result.content}

YOUR TASK:
Revise your draft answer by:
1. Addressing valid concerns from the critique
2. Fixing any identified errors or oversights
3. Improving clarity and completeness
4. Maintaining technical accuracy

Provide your revised, final answer."""

    revision_messages = [
        {"role": "user", "content": revision_prompt}
    ]
    
    print("[critic] Calling Opus for revision using Gemini critique")
    logger.info("[critic] Calling Opus for revision using Gemini critique")
    
    try:
        # Create envelope for revision call
        revision_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=f"revision-{uuid4()}",
            project_id=1,
            job_type=Phase4JobType.APP_ARCHITECTURE,
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=OPUS_REVISION_MAX_TOKENS,
                max_cost_estimate=0.10,
                max_wall_time_seconds=60,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=revision_messages,
            metadata={"revision_of_draft": True},
            allow_multi_model_review=False,
            needs_tools=[],
        )
        
        result = await registry_llm_call(
            provider_id="anthropic",
            model_id=opus_model_id,
            messages=revision_messages,
            job_envelope=revision_envelope,
        )
        
        if not result.is_success() or not result.content:
            logger.warning("[critic] Opus revision failed or returned empty")
            return None
        
        print(f"[critic] Opus revision complete: {len(result.content)} chars")
        logger.info("[critic] Opus revision complete: %d chars", len(result.content))
        
        # Return as LLMResult
        return LLMResult(
            content=result.content,
            provider="anthropic",
            model=opus_model_id,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )
        
    except Exception as exc:
        logger.exception("[critic] Opus revision call failed: %s", exc)
        return None


async def run_high_stakes_with_critique(
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
) -> LLMResult:
    """
    Run 3-step critique pipeline for high-stakes Opus work.
    
    Flow:
    1. Opus generates draft
    2. Check length (skip critique if too short)
    3. Gemini 3 Pro critiques draft (using job-specific prompt with env context)
    4. Opus revises based on critique
    
    Graceful degradation: Returns original draft on any failure.
    
    v0.13.7: Critique prompt now customized by job_type with environment context
    
    Args:
        task: Original LLMTask
        provider_id: "anthropic"
        model_id: Opus model identifier
        envelope: JobEnvelope
        job_type_str: Fine-grained job type string
    
    Returns:
        LLMResult with revised answer (or draft on failure)
    """
    print(f"[critic] High-stakes pipeline enabled: job_type={job_type_str} model={model_id}")
    logger.info(
        "[critic] High-stakes pipeline enabled: job_type=%s model=%s",
        job_type_str, model_id
    )
    
    # ==========================================================================
    # STEP 1: Generate Opus Draft
    # ==========================================================================
    
    try:
        draft_result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            job_envelope=envelope,
        )
        
        if not draft_result.is_success():
            logger.error("[critic] Opus draft failed: %s", draft_result.error_message)
            return LLMResult(
                content=draft_result.error_message or "",
                provider=provider_id,
                model=model_id,
                finish_reason="error",
                error_message=draft_result.error_message,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                raw_response=None,
            )
        
        # Convert to LLMResult format
        draft = LLMResult(
            content=draft_result.content,
            provider=provider_id,
            model=model_id,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=draft_result.usage.prompt_tokens,
            completion_tokens=draft_result.usage.completion_tokens,
            total_tokens=draft_result.usage.total_tokens,
            cost_usd=draft_result.usage.cost_estimate,
            raw_response=draft_result.raw_response,
        )
        
        print(f"[critic] Opus draft complete: {len(draft.content)} chars")
        
    except Exception as exc:
        logger.exception("[critic] Opus draft call failed: %s", exc)
        return LLMResult(
            content="",
            provider=provider_id,
            model=model_id,
            finish_reason="error",
            error_message=str(exc),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )
    
    # ==========================================================================
    # STEP 2: Check Draft Length
    # ==========================================================================
    
    if not is_long_enough_for_critique(draft.content):
        print(f"[critic] Draft too short for critique ({len(draft.content or '')} chars < {MIN_CRITIQUE_CHARS}), returning draft")
        logger.info(
            "[critic] Draft too short for critique (%d chars < %d), returning draft",
            len(draft.content or ""), MIN_CRITIQUE_CHARS
        )
        return draft
    
    # ==========================================================================
    # STEP 3: Call Gemini Critic (v0.13.7 - job-specific prompt + env context)
    # ==========================================================================
    
    critique = await call_gemini_critic(
        original_task=task,
        draft_result=draft,
        job_type_str=job_type_str,
    )
    
    if not critique or not critique.content:
        print("[critic] Gemini critic failed; returning original Opus draft")
        logger.warning("[critic] Gemini critic failed; returning original Opus draft")
        return draft
    
    # ==========================================================================
    # STEP 4: Call Opus for Revision
    # ==========================================================================
    
    revision = await call_opus_revision(
        original_task=task,
        draft_result=draft,
        critique_result=critique,
        opus_model_id=model_id,
    )
    
    if not revision or not revision.content:
        print("[critic] Opus revision failed; returning original Opus draft")
        logger.warning("[critic] Opus revision failed; returning original Opus draft")
        return draft
    
    # ==========================================================================
    # SUCCESS: Return Revised Answer
    # ==========================================================================
    
    # Add metadata about critique pipeline
    revision.routing_decision = {
        "job_type": job_type_str,
        "provider": provider_id,
        "model": model_id,
        "reason": "High-stakes pipeline: Opus draft → Gemini critique → Opus revision",
        "critique_pipeline": {
            "draft_tokens": draft.total_tokens,
            "critique_tokens": critique.total_tokens,
            "revision_tokens": revision.total_tokens,
            "total_cost": draft.cost_usd + critique.cost_usd + revision.cost_usd,
        }
    }
    
    return revision


# =============================================================================
# CORE CALL FUNCTION (Async)
# =============================================================================

async def call_llm_async(task: LLMTask) -> LLMResult:
    """
    Primary async LLM call entry point.
    
    Provider selection priority:
    1. task.force_provider (explicit override)
    2. Job classifier (8-route system with HARD RULES)
    3. Attachment safety rule enforcement
    4. High-stakes critique pipeline check (v0.13.7)
    5. task.provider (legacy hint)
    6. Fallback to CHAT_LIGHT routing
    
    HARD RULES ENFORCED:
    - Images/video NEVER go to Claude
    - PDFs NEVER go to Claude
    
    ATTACHMENT SAFETY RULE:
    - If attachments present AND no job_type specified:
      - Do NOT route to Claude unless classifier returns code.medium/orchestrator
    
    HIGH-STAKES CRITIQUE PIPELINE:
    - When Anthropic + Opus + high-stakes job + response >= 1500 chars:
      - Run 3-step pipeline: Opus draft → Gemini critique → Opus revision
      - v0.13.7: Critique prompt customized by job_type with environment context
    """
    session_id = getattr(task, "session_id", None)
    project_id = getattr(task, "project_id", 1) or 1

    # Track if job_type was explicitly specified
    job_type_specified = task.job_type is not None and task.job_type != JobType.UNKNOWN
    
    # Track if attachments are present
    has_attachments = bool(task.attachments and len(task.attachments) > 0)

    # Synthesize envelope
    try:
        envelope = synthesize_envelope_from_task(
            task=task,
            session_id=session_id,
            project_id=project_id,
        )
    except (ValidationError, Exception) as exc:
        logger.warning("[router] Envelope synthesis failed: %s", exc)
        return LLMResult(
            content="",
            provider=task.provider.value if task.provider else Provider.OPENAI.value,
            model=task.model or "",
            finish_reason="validation_error",
            error_message=f"Envelope synthesis failed: {exc}",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )

    # ==========================================================================
    # PROVIDER/MODEL SELECTION (with HARD RULES and ATTACHMENT SAFETY)
    # ==========================================================================
    
    if ROUTER_DEBUG:
        _debug_log("=" * 70)
        _debug_log("ROUTER START")
        _debug_log("=" * 70)
        _debug_log(f"Task job_type: {task.job_type.value if task.job_type else 'None'}")
        _debug_log(f"Task attachments: {len(task.attachments) if task.attachments else 0}")
        _debug_log(f"Task force_provider: {task.force_provider.value if task.force_provider else 'None'}")
    
    # Priority 1: Explicit force_provider override
    if task.force_provider is not None:
        provider_id = task.force_provider.value
        model_id = task.model or DEFAULT_MODELS.get(provider_id, "")
        reason = "force_provider override"
        classified_type = task.job_type
        logger.info("[router] Using force_provider: %s / %s", provider_id, model_id)
        if ROUTER_DEBUG:
            _debug_log(f"Priority 1: force_provider override")
            _debug_log(f"  Provider: {provider_id}")
            _debug_log(f"  Model: {model_id}")
    
    # Priority 2: Job classifier (8-route system)
    else:
        if ROUTER_DEBUG:
            _debug_log(f"Priority 2: Using job classifier")
        
        # Check if task already has a valid job_type set (from main.py classification)
        if task.job_type and task.job_type != JobType.UNKNOWN:
            # Respect the pre-classification instead of re-classifying
            classified_type = task.job_type
            
            if ROUTER_DEBUG:
                _debug_log(f"  Task has pre-set job_type: {classified_type.value}")
            
            # Force CHAT_LIGHT to GPT mini immediately
            if classified_type == JobType.CHAT_LIGHT or classified_type.value in ["chat.light", "chat_light", "casual_chat"]:
                provider = Provider.OPENAI
                provider_id = "openai"
                model_id = os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini")
                reason = "Pre-classified as CHAT_LIGHT, forced to GPT mini"
                print(f"[router] RESPECTING PRE-CLASSIFICATION: {classified_type.value} → FORCED to {provider_id}/{model_id}")
                if ROUTER_DEBUG:
                    _debug_log(f"  CHAT_LIGHT OVERRIDE triggered")
                    _debug_log(f"    Provider: {provider_id}")
                    _debug_log(f"    Model: {model_id}")
            else:
                # Get routing for the pre-classified type
                decision_temp = get_routing_for_job_type(classified_type.value)
                provider = decision_temp.provider
                provider_id = provider.value
                model_id = decision_temp.model
                reason = f"Pre-classified as {classified_type.value}"
                print(f"[router] RESPECTING PRE-CLASSIFICATION: {classified_type.value} → {provider_id}/{model_id}")
                if ROUTER_DEBUG:
                    _debug_log(f"  Using routing for pre-classified type")
                    _debug_log(f"    Provider: {provider_id}")
                    _debug_log(f"    Model: {model_id}")
        else:
            if ROUTER_DEBUG:
                _debug_log(f"  No valid pre-classification, calling classifier...")
            
            # No job_type set, use classifier to determine routing
            provider, model_id, classified_type, reason = classify_and_route(task)
            provider_id = provider.value
            
            if ROUTER_DEBUG:
                _debug_log(f"  Classifier returned:")
                _debug_log(f"    Job Type: {classified_type.value}")
                _debug_log(f"    Provider: {provider_id}")
                _debug_log(f"    Model: {model_id}")
                _debug_log(f"    Reason: {reason}")
            
            # Force CHAT_LIGHT to GPT mini
            if classified_type == JobType.CHAT_LIGHT or classified_type.value in ["chat.light", "chat_light", "casual_chat"]:
                provider = Provider.OPENAI
                provider_id = "openai"
                model_id = os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini")
                reason = "HARD OVERRIDE: CHAT_LIGHT forced to GPT mini"
                print("[router] OVERRIDE: Forcing CHAT_LIGHT → openai/gpt-4.1-mini")
                if ROUTER_DEBUG:
                    _debug_log(f"  CHAT_LIGHT OVERRIDE triggered")
                    _debug_log(f"    Provider: {provider_id}")
                    _debug_log(f"    Model: {model_id}")
        
        # Allow task.model to override model choice
        if task.model:
            if ROUTER_DEBUG:
                _debug_log(f"  Task.model override: {task.model}")
            model_id = task.model
        
        print(f"[router] Final routing: {classified_type.value} → {provider_id}/{model_id}")
        logger.info(
            "[router] Classifier: %s → %s / %s (%s)",
            classified_type.value, provider_id, model_id, reason,
        )
        
        if ROUTER_DEBUG:
            _debug_log("=" * 70)
            _debug_log("ROUTING DECISION FINAL")
            _debug_log(f"  Job Type: {classified_type.value}")
            _debug_log(f"  Provider: {provider_id}")
            _debug_log(f"  Model: {model_id}")
            _debug_log(f"  Reason: {reason}")
            _debug_log("=" * 70)
        
        # Create a RoutingDecision for safety check
        decision = RoutingDecision(
            job_type=classified_type,
            provider=provider,
            model=model_id,
            reason=reason,
        )
        
        # ======================================================================
        # ATTACHMENT SAFETY RULE ENFORCEMENT
        # ======================================================================
        
        provider_id, model_id, reason = _check_attachment_safety(
            task=task,
            decision=decision,
            has_attachments=has_attachments,
            job_type_specified=job_type_specified,
        )
    
    # ==========================================================================
    # HARD RULE ENFORCEMENT (vision jobs cannot go to Claude)
    # ==========================================================================
    
    if is_claude_forbidden(classified_type):
        if provider_id == "anthropic":
            logger.error(
                "[router] BLOCKED: Attempted to route %s to Claude - forcing to Gemini",
                classified_type.value,
            )
            provider_id = "google"
            model_id = os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash")
            reason = f"FORCED: {classified_type.value} cannot go to Claude"
    
    # ==========================================================================
    # HIGH-STAKES CRITIQUE PIPELINE CHECK (v0.13.7)
    # ==========================================================================
    
    # v0.13.6: Normalize job type for high-stakes matching
    # Maps 'orchestrator' + architecture reason → 'architecture_design'
    normalized_job_type = normalize_job_type_for_high_stakes(
        classified_type.value,
        reason
    )
    
    # Check if critique pipeline should be triggered
    should_run_critique = (
        provider_id == "anthropic" and
        is_opus_model(model_id) and
        is_high_stakes_job(normalized_job_type)
    )
    
    if should_run_critique:
        if ROUTER_DEBUG:
            _debug_log("=" * 70)
            _debug_log("HIGH-STAKES CRITIQUE PIPELINE TRIGGERED")
            _debug_log(f"  Original Job Type: {classified_type.value}")
            _debug_log(f"  Normalized Job Type: {normalized_job_type}")
            _debug_log(f"  Model: {model_id}")
            _debug_log("=" * 70)
        
        print(f"[router] HIGH-STAKES PIPELINE: {classified_type.value} → {normalized_job_type}")
        
        return await run_high_stakes_with_critique(
            task=task,
            provider_id=provider_id,
            model_id=model_id,
            envelope=envelope,
            job_type_str=normalized_job_type,
        )
    
    # ==========================================================================
    # NORMAL LLM CALL (No Critique Pipeline)
    # ==========================================================================

    try:
        result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            job_envelope=envelope,
        )
    except Exception as exc:
        logger.exception("[router] llm_call failed: %s", exc)
        return LLMResult(
            content="",
            provider=provider_id,
            model=model_id,
            finish_reason="error",
            error_message=str(exc),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )

    if not result.is_success():
        return LLMResult(
            content=result.error_message or "",
            provider=provider_id,
            model=model_id,
            finish_reason="error",
            error_message=result.error_message,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )

    return LLMResult(
        content=result.content,
        provider=provider_id,
        model=model_id,
        finish_reason="stop",
        error_message=None,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        cost_usd=result.usage.cost_estimate,
        raw_response=result.raw_response,
        job_type=classified_type,
        routing_decision={
            "job_type": classified_type.value,
            "provider": provider_id,
            "model": model_id,
            "reason": reason,
        },
    )


# =============================================================================
# HIGH-LEVEL HELPERS (Async)
# =============================================================================

async def quick_chat_async(
    message: str,
    context: Optional[str] = None,
) -> LLMResult:
    """Simple async chat helper."""
    messages: List[Dict[str, str]] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})

    task = LLMTask(
        job_type=JobType.CHAT_LIGHT,
        messages=messages,
        routing=RoutingOptions(),
        project_context=context,
    )
    return await call_llm_async(task)


async def request_code_async(
    message: str,
    context: Optional[str] = None,
    high_stakes: bool = False,
) -> LLMResult:
    """Async helper for code tasks."""
    messages: List[Dict[str, str]] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})

    job_type = JobType.ORCHESTRATOR if high_stakes else JobType.CODE_MEDIUM

    task = LLMTask(
        job_type=job_type,
        messages=messages,
        routing=RoutingOptions(),
        project_context=context,
    )
    return await call_llm_async(task)


async def review_work_async(
    message: str,
    context: Optional[str] = None,
) -> LLMResult:
    """Async helper for review/critique."""
    messages: List[Dict[str, str]] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})

    task = LLMTask(
        job_type=JobType.ORCHESTRATOR,
        messages=messages,
        routing=RoutingOptions(),
        project_context=context,
    )
    return await call_llm_async(task)


# =============================================================================
# SYNC WRAPPERS (CLI/Testing only)
# =============================================================================

def call_llm(task: LLMTask) -> LLMResult:
    """Sync wrapper for call_llm_async."""
    import asyncio
    return asyncio.run(call_llm_async(task))


def quick_chat(message: str, context: Optional[str] = None) -> LLMResult:
    """Sync wrapper for quick_chat_async."""
    import asyncio
    return asyncio.run(quick_chat_async(message=message, context=context))


def request_code(
    message: str,
    context: Optional[str] = None,
    high_stakes: bool = False,
) -> LLMResult:
    """Sync wrapper for request_code_async."""
    import asyncio
    return asyncio.run(request_code_async(message, context, high_stakes))


def review_work(message: str, context: Optional[str] = None) -> LLMResult:
    """Sync wrapper for review_work_async."""
    import asyncio
    return asyncio.run(review_work_async(message=message, context=context))


# =============================================================================
# COMPATIBILITY HELPERS
# =============================================================================

def analyze_with_vision(
    prompt: str,
    image_description: Optional[str] = None,
    context: Optional[str] = None,
) -> LLMResult:
    """Compatibility wrapper for legacy vision calls."""
    parts: List[str] = [prompt]
    if image_description:
        parts.append("\n\n[Image description]\n" + image_description)
    if context:
        parts.append("\n\n[Context]\n" + context)
    return quick_chat(message="".join(parts), context=None)


def web_search_query(query: str, context: Optional[str] = None) -> LLMResult:
    """Compatibility wrapper for legacy web search."""
    parts = [f"[WEB SEARCH STYLE QUERY]\n{query}"]
    if context:
        parts.append("\n\n[ADDITIONAL CONTEXT]\n" + context)
    return quick_chat(message="".join(parts), context=None)


def list_job_types() -> List[str]:
    """List available job types."""
    return [jt.value for jt in JobType]


def get_routing_info() -> Dict[str, Any]:
    """Get routing configuration info."""
    models = get_model_config()
    return {
        "routing_version": "0.13.7",
        "job_types": {
            "CHAT_LIGHT": {"provider": "openai", "model": models["openai"]},
            "TEXT_HEAVY": {"provider": "openai", "model": models["openai_heavy"]},
            "CODE_MEDIUM": {"provider": "anthropic", "model": models["anthropic_sonnet"]},
            "ORCHESTRATOR": {"provider": "anthropic", "model": models["anthropic_opus"]},
            "IMAGE_SIMPLE": {"provider": "google", "model": models["gemini_fast"]},
            "IMAGE_COMPLEX": {"provider": "google", "model": models["gemini_complex"]},
            "VIDEO_HEAVY": {"provider": "google", "model": models["gemini_video"]},
            "OPUS_CRITIC": {"provider": "google", "model": models["gemini_critic"]},
        },
        "default_models": DEFAULT_MODELS,
        "hard_rules": [
            "Images/video NEVER go to Claude",
            "PDFs NEVER go to Claude",
            "opus.critic is EXPLICIT ONLY",
            "Attachments without job_type do NOT default to Claude",
        ],
        "critique_pipeline": {
            "enabled": True,
            "high_stakes_types": list(HIGH_STAKES_JOB_TYPES),
            "min_length_chars": MIN_CRITIQUE_CHARS,
            "critic_model": GEMINI_CRITIC_MODEL,
            "critic_max_tokens": GEMINI_CRITIC_MAX_TOKENS,
            "revision_max_tokens": OPUS_REVISION_MAX_TOKENS,
            "architecture_critique": "HARD-CODED constraints: single-host, local repos, solo dev, no K8s/VLANs/containers",
            "security_critique": "Aggressive hardening focus",
            "environment_context": "Passed to architecture critique",
        },
    }


def is_policy_routing_enabled() -> bool:
    """Always True - we use job_classifier."""
    return True


def enable_policy_routing() -> None:
    """No-op for compatibility."""
    logger.info("[router] Policy routing is always enabled (job_classifier)")


def disable_policy_routing() -> None:
    """No-op for compatibility."""
    logger.info("[router] Policy routing cannot be disabled")


def reload_routing_policy() -> None:
    """No-op for compatibility."""
    logger.info("[router] Routing policy reload requested (no-op)")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Async API
    "call_llm_async",
    "quick_chat_async",
    "request_code_async",
    "review_work_async",
    
    # Sync wrappers
    "call_llm",
    "quick_chat",
    "request_code",
    "review_work",
    
    # Classification
    "classify_and_route",
    "normalize_job_type_for_high_stakes",
    
    # High-stakes pipeline
    "run_high_stakes_with_critique",
    "is_high_stakes_job",
    "is_opus_model",
    "HIGH_STAKES_JOB_TYPES",
    "get_environment_context",
    
    # Compatibility
    "analyze_with_vision",
    "web_search_query",
    "list_job_types",
    "get_routing_info",
    "is_policy_routing_enabled",
    "enable_policy_routing",
    "disable_policy_routing",
    "reload_routing_policy",
    "synthesize_envelope_from_task",
]