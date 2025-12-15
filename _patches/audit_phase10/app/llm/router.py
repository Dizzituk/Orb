# FILE: app/llm/router.py
"""
LLM Router (PHASE 4 - v0.15.1)

Version: 0.15.1 - Simplified OVERRIDE → Frontier Model Routing

v0.15.1 Changes:
- NEW: Simplified OVERRIDE mechanism for frontier model routing
- OVERRIDE (default) → Gemini 3 Pro Preview (GEMINI_FRONTIER_MODEL_ID)
- OVERRIDE CLAUDE/OPUS → Anthropic frontier (ANTHROPIC_FRONTIER_MODEL_ID)
- OVERRIDE CHATGPT/GPT/OPENAI → OpenAI frontier (OPENAI_FRONTIER_MODEL_ID)
- OVERRIDE line stripped from prompt before sending to LLM
- Override detection happens BEFORE normal classification
- All other pipeline logic (critique, video+code) still runs after override

v0.15.0 Changes:
- Integrated file_classifier.py for MIXED_FILE detection and stable [FILE_X] naming
- Integrated audit_logger.py for structured routing decision logging
- Integrated relationship_detector.py for pairwise modality relationship detection
- Integrated token_budgeting.py for context budget management
- Integrated task_extractor.py for multi-task parsing (future use)
- Integrated fallbacks.py for structured failure handling
- File map injection into prompts for all multimodal jobs
- Enhanced debug logging with audit trail

CRITICAL PIPELINE SPEC COMPLIANCE:
- §1 File Classification: MIXED_FILE detection for PDFs/DOCX with images
- §2 Stable Naming: [FILE_1], [FILE_2], etc. via build_file_map()
- §3 Relationship Detection: Pairwise modality relationships
- §7 Token Budgeting: Context allocation by content type
- §11 Fallbacks: Structured fallback chains
- §12 Audit Logging: Full routing decision trace

8-ROUTE CLASSIFICATION SYSTEM:
- CHAT_LIGHT → OpenAI (gpt-4.1-mini) - casual chat
- TEXT_HEAVY → OpenAI (gpt-4.1) - heavy text, text-only PDFs
- CODE_MEDIUM → Anthropic Sonnet - scoped code (1-3 files)
- ORCHESTRATOR → Anthropic Opus - architecture, multi-file
- IMAGE_SIMPLE → Gemini Flash - LEGACY ONLY (never auto-selected)
- IMAGE_COMPLEX → Gemini 2.5 Pro - ALL images, PDFs with images, MIXED_FILE
- VIDEO_HEAVY → Gemini 3.0 Pro - ALL videos
- OPUS_CRITIC → Gemini 3.0 Pro - explicit Opus review only
- VIDEO_CODE_DEBUG → 2-step pipeline: Gemini3 transcribe → Sonnet code

HARD RULES:
- Images/video NEVER go to Claude
- PDFs NEVER go to Claude
- MIXED_FILE (docs with images) → Gemini
- opus.critic is EXPLICIT ONLY

HIGH-STAKES CRITIQUE PIPELINE:
- Opus draft → Gemini 3 critique → Opus revision
- Triggers: Anthropic + Opus + high-stakes job + response >= 1500 chars
- Environment context for architecture jobs

VIDEO+CODE DEBUG PIPELINE:
- Gemini 3 Pro transcribes videos → Claude Sonnet receives code + transcripts
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

# Job classifier (8-route system with MIXED_FILE detection)
from app.llm.job_classifier import (
    classify_job,
    classify_and_route as classifier_classify_and_route,
    get_routing_for_job_type,
    get_model_config,
    is_claude_forbidden,
    is_claude_allowed,
    prepare_attachments,
    compute_modality_flags,
    # v0.15.1: Frontier override detection
    detect_frontier_override,
    GEMINI_FRONTIER_MODEL_ID,
    ANTHROPIC_FRONTIER_MODEL_ID,
    OPENAI_FRONTIER_MODEL_ID,
)

# Video transcription for pipelines
from app.llm.gemini_vision import transcribe_video_for_context

# =============================================================================
# v0.15.0: CRITICAL PIPELINE SPEC MODULE IMPORTS
# =============================================================================

# Audit logging (Spec §12)
try:
    from app.llm.audit_logger import (
        get_audit_logger,
        RoutingTrace,
        AuditEventType,
    )
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    def get_audit_logger():
        return None

# File classifier (Spec §1 & §2)
try:
    from app.llm.file_classifier import (
        classify_attachments,
        build_file_map,
        ClassificationResult,
    )
    FILE_CLASSIFIER_AVAILABLE = True
except ImportError:
    FILE_CLASSIFIER_AVAILABLE = False

# Relationship detector (Spec §3)
try:
    from app.llm.relationship_detector import (
        detect_relationships,
        RelationshipResult,
        RelationshipType,
    )
    RELATIONSHIP_DETECTOR_AVAILABLE = True
except ImportError:
    RELATIONSHIP_DETECTOR_AVAILABLE = False

# Token budgeting (Spec §7)
try:
    from app.llm.token_budgeting import (
        allocate_budget,
        create_budget_for_model,
        TokenBudget,
    )
    TOKEN_BUDGETING_AVAILABLE = True
except ImportError:
    TOKEN_BUDGETING_AVAILABLE = False

# Fallback handler (Spec §11)
try:
    from app.llm.fallbacks import (
        FallbackHandler,
        handle_video_failure,
        handle_vision_failure,
        handle_overwatcher_failure,
        get_fallback_chain,
        FailureType,
        FallbackAction,
    )
    FALLBACKS_AVAILABLE = True
except ImportError:
    FALLBACKS_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER DEBUG MODE
# =============================================================================
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"
AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

def _debug_log(msg: str):
    """Print debug message if ROUTER_DEBUG is enabled."""
    if ROUTER_DEBUG:
        print(f"[router-debug] {msg}")


# =============================================================================
# HIGH-STAKES CRITIQUE PIPELINE CONFIGURATION
# =============================================================================

HIGH_STAKES_JOB_TYPES = {
    "architecture_design",
    "big_architecture",
    "high_stakes_infra",
    "security_review",
    "privacy_sensitive_change",
    "security_sensitive_change",
    "complex_code_change",
    "implementation_plan",
    "spec_review",
    "architecture",
    "deep_planning",
    "orchestrator",
}

MIN_CRITIQUE_CHARS = int(os.getenv("ORB_MIN_CRITIQUE_CHARS", "1500"))
GEMINI_CRITIC_MODEL = os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3-pro")
GEMINI_CRITIC_MAX_TOKENS = int(os.getenv("GEMINI_CRITIC_MAX_TOKENS", "1024"))
OPUS_REVISION_MAX_TOKENS = int(os.getenv("OPUS_REVISION_MAX_TOKENS", "2048"))


# =============================================================================
# ENVIRONMENT CONTEXT
# =============================================================================

def get_environment_context() -> Dict[str, Any]:
    """Get current environment context for architecture critique."""
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
    """Normalize job type to specific high-stakes string."""
    if job_type_str in HIGH_STAKES_JOB_TYPES and job_type_str != "orchestrator":
        return job_type_str
    
    if job_type_str == "orchestrator":
        reason_lower = reason.lower()
        
        architecture_keywords = ["architecture", "system design", "architect", "system architecture"]
        if any(kw in reason_lower for kw in architecture_keywords):
            print(f"[router] Normalized orchestrator → architecture_design (reason: {reason[:80]})")
            return "architecture_design"
        
        security_keywords = ["security", "security review", "security audit"]
        if any(kw in reason_lower for kw in security_keywords):
            print(f"[router] Normalized orchestrator → security_review (reason: {reason[:80]})")
            return "security_review"
        
        infra_keywords = ["infrastructure", "infra", "deployment"]
        if any(kw in reason_lower for kw in infra_keywords):
            print(f"[router] Normalized orchestrator → high_stakes_infra (reason: {reason[:80]})")
            return "high_stakes_infra"
        
        print(f"[router] Normalized orchestrator → architecture_design (default)")
        return "architecture_design"
    
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
# CRITIQUE PROMPT TEMPLATES
# =============================================================================

def build_critique_prompt_for_architecture(
    draft_text: str, 
    original_request: str,
    env_context: Dict[str, Any]
) -> str:
    """Build critique prompt for architecture design tasks."""
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

2. **ENVIRONMENT FIT**
   - Can this be implemented by one person in 2-4 weeks?
   - Does it respect the single-host deployment context?

3. **TECHNICAL CORRECTNESS**
   - Are there actual errors in the design logic?
   - Are proposed patterns sound?

4. **PRAGMATIC SAFETY**
   - Are security invariants appropriate for local deployment?
   - Can goals be met with simpler solutions?

===== OUTPUT RULES =====

If the design is OVER-ENGINEERED:
  → State it directly: "This design is over-engineered for your single-host setup."
  → Suggest simpler alternatives using local controls

Keep your critique under 800 words. Be direct and specific."""


def build_critique_prompt_for_security(draft_text: str, original_request: str) -> str:
    """Build critique prompt for security review tasks."""
    return f"""You are performing a security review.

ORIGINAL REQUEST:
{original_request}

SECURITY ANALYSIS:
{draft_text}

YOUR TASK:
Provide a critical security review focusing on:

1. **Threat Model Completeness** - Are all attack vectors identified?
2. **Security Controls** - Are proposed controls sufficient?
3. **Implementation Risks** - Are there security pitfalls?
4. **Defense in Depth** - Are there sufficient layers?

Be aggressive in identifying security issues.
Keep your critique under 800 words."""


def build_critique_prompt_for_general(draft_text: str, original_request: str, job_type_str: str) -> str:
    """Build critique prompt for general high-stakes tasks."""
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

Keep your critique under 800 words."""


def build_critique_prompt(
    draft_text: str, 
    original_request: str, 
    job_type_str: str,
    env_context: Optional[Dict[str, Any]] = None
) -> str:
    """Build critique prompt based on job type."""
    if job_type_str in ["architecture_design", "big_architecture", "high_stakes_infra", 
                         "architecture", "orchestrator"]:
        if not env_context:
            env_context = get_environment_context()
        return build_critique_prompt_for_architecture(draft_text, original_request, env_context)
    
    elif job_type_str in ["security_review", "security_sensitive_change", 
                           "privacy_sensitive_change"]:
        return build_critique_prompt_for_security(draft_text, original_request)
    
    else:
        return build_critique_prompt_for_general(draft_text, original_request, job_type_str)


# =============================================================================
# DEFAULT MODELS PER PROVIDER
# =============================================================================

DEFAULT_MODELS: Dict[str, str] = {
    "openai": os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini"),
    "openai_heavy": os.getenv("OPENAI_MODEL_HEAVY_TEXT", "gpt-4.1"),
    "anthropic": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
    "anthropic_opus": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20250514"),
    "google": os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro"),
    "google_complex": os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro"),
    "google_video": os.getenv("GEMINI_VIDEO_HEAVY_MODEL", "gemini-3.0-pro-preview"),
    "google_critic": os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3-pro"),
    # v0.15.1: Frontier models
    "gemini_frontier": GEMINI_FRONTIER_MODEL_ID,
    "anthropic_frontier": ANTHROPIC_FRONTIER_MODEL_ID,
    "openai_frontier": OPENAI_FRONTIER_MODEL_ID,
}


# =============================================================================
# JOB TYPE MAPPING (Legacy → Phase 4)
# =============================================================================

_LEGACY_TO_PHASE4_JOB_TYPE: Dict[str, Phase4JobType] = {
    JobType.TEXT_ADMIN.value: Phase4JobType.CHAT_SIMPLE,
    JobType.CASUAL_CHAT.value: Phase4JobType.CHAT_SIMPLE,
    JobType.CHAT_LIGHT.value: Phase4JobType.CHAT_SIMPLE,
    JobType.TEXT_HEAVY.value: Phase4JobType.CHAT_SIMPLE,
    JobType.QUICK_QUESTION.value: Phase4JobType.CHAT_RESEARCH,
    JobType.PROMPT_SHAPING.value: Phase4JobType.CHAT_SIMPLE,
    JobType.SUMMARY.value: Phase4JobType.CHAT_SIMPLE,
    JobType.EXPLANATION.value: Phase4JobType.CHAT_SIMPLE,
    JobType.SMALL_CODE.value: Phase4JobType.CODE_SMALL,
    JobType.CODE_MEDIUM.value: Phase4JobType.CODE_SMALL,
    JobType.SIMPLE_CODE_CHANGE.value: Phase4JobType.CODE_SMALL,
    JobType.SMALL_BUGFIX.value: Phase4JobType.CODE_SMALL,
    JobType.BIG_ARCHITECTURE.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.ORCHESTRATOR.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.COMPLEX_CODE_CHANGE.value: Phase4JobType.CODE_REPO,
    JobType.CODEGEN_FULL_FILE.value: Phase4JobType.CODE_REPO,
    JobType.ARCHITECTURE_DESIGN.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.CODE_REVIEW.value: Phase4JobType.CRITIQUE_REVIEW,
    JobType.SPEC_REVIEW.value: Phase4JobType.CRITIQUE_REVIEW,
    JobType.HIGH_STAKES_INFRA.value: Phase4JobType.APP_ARCHITECTURE,
    JobType.OPUS_CRITIC.value: Phase4JobType.CRITIQUE_REVIEW,
    JobType.SIMPLE_VISION.value: Phase4JobType.VISION_SIMPLE,
    JobType.IMAGE_SIMPLE.value: Phase4JobType.VISION_SIMPLE,
    JobType.HEAVY_MULTIMODAL_CRITIQUE.value: Phase4JobType.VISION_COMPLEX,
    JobType.IMAGE_COMPLEX.value: Phase4JobType.VISION_COMPLEX,
    JobType.VIDEO_HEAVY.value: Phase4JobType.VISION_COMPLEX,
}


def _map_to_phase4_job_type(job_type: JobType) -> Phase4JobType:
    """Map our JobType to Phase 4 JobType."""
    return _LEGACY_TO_PHASE4_JOB_TYPE.get(job_type.value, Phase4JobType.CHAT_SIMPLE)


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
# v0.15.0: FILE MAP INJECTION
# =============================================================================

def inject_file_map_into_messages(
    messages: List[Dict[str, str]],
    file_map: str,
) -> List[Dict[str, str]]:
    """
    Inject file map into messages for stable file referencing.
    
    Adds file map to system message or creates one if needed.
    """
    if not file_map:
        return messages
    
    file_map_instruction = f"""
{file_map}

IMPORTANT: When referring to files in your response, always use the [FILE_X] identifiers shown above.
"""
    
    # Check if there's already a system message
    new_messages = []
    system_found = False
    
    for msg in messages:
        if msg.get("role") == "system":
            # Append file map to existing system message
            new_content = msg.get("content", "") + "\n\n" + file_map_instruction
            new_messages.append({"role": "system", "content": new_content})
            system_found = True
        else:
            new_messages.append(msg)
    
    # If no system message, add one at the beginning
    if not system_found:
        new_messages.insert(0, {"role": "system", "content": file_map_instruction})
    
    return new_messages


# =============================================================================
# CLASSIFY AND ROUTE
# =============================================================================

def classify_and_route(task: LLMTask) -> Tuple[Provider, str, JobType, str]:
    """Classify task and determine routing using job_classifier."""
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    message_text = user_messages[-1].get("content", "") if user_messages else ""
    
    requested_type = task.job_type.value if task.job_type else None
    metadata = task.metadata or {}
    
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
    file_map: Optional[str] = None,
    cleaned_message: Optional[str] = None,
) -> JobEnvelope:
    """
    Synthesize a JobEnvelope from LLMTask.
    
    v0.15.1: Added cleaned_message parameter for OVERRIDE line removal.
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

    # Build messages with system/project context
    final_messages = []
    
    system_parts = []
    if task.system_prompt:
        system_parts.append(task.system_prompt)
    if task.project_context:
        system_parts.append(task.project_context)
    
    if system_parts:
        system_content = "\n\n".join(system_parts)
        final_messages.append({"role": "system", "content": system_content})
    
    # v0.15.1: Replace user message content if cleaned_message provided
    if cleaned_message is not None:
        for msg in task.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Simple string message - replace entirely
                    final_messages.append({"role": "user", "content": cleaned_message})
                elif isinstance(content, list):
                    # Multimodal message - replace only the text part
                    new_content = []
                    text_replaced = False
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text" and not text_replaced:
                            # Replace first text part with cleaned message
                            new_content.append({"type": "text", "text": cleaned_message})
                            text_replaced = True
                        else:
                            new_content.append(part)
                    # If no text part was found, append the cleaned message
                    if not text_replaced and cleaned_message:
                        new_content.append({"type": "text", "text": cleaned_message})
                    final_messages.append({"role": "user", "content": new_content})
                else:
                    final_messages.append(msg)
            else:
                final_messages.append(msg)
    else:
        final_messages.extend(task.messages)
    
    # v0.15.0: Inject file map if available
    if file_map:
        final_messages = inject_file_map_into_messages(final_messages, file_map)

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
        messages=final_messages,
        metadata={
            "legacy_provider_hint": task.provider.value if task.provider else None,
            "legacy_routing": routing.model_dump() if routing else None,
            "legacy_context": task.project_context,
            "file_map": file_map,
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
    """Enforce attachment safety rule."""
    provider_id = decision.provider.value
    model_id = decision.model
    reason = decision.reason
    
    if has_attachments and not job_type_specified and provider_id == "anthropic":
        if not is_claude_allowed(decision.job_type):
            logger.warning(
                "[router] Attachment safety: Blocked Claude route for %s with attachments",
                decision.job_type.value,
            )
            provider_id = "openai"
            model_id = DEFAULT_MODELS["openai_heavy"]
            reason = f"Attachment safety: {decision.job_type.value} not allowed on Claude"
    
    return (provider_id, model_id, reason)


# =============================================================================
# HIGH-STAKES CRITIQUE PIPELINE
# =============================================================================

async def call_gemini_critic(
    original_task: LLMTask,
    draft_result: LLMResult,
    job_type_str: str,
) -> Optional[LLMResult]:
    """Call Gemini 3 Pro to critique the Opus draft."""
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""
    
    env_context = None
    if job_type_str in ["architecture_design", "big_architecture", "high_stakes_infra", 
                         "architecture", "orchestrator"]:
        env_context = get_environment_context()
        print(f"[critic] Passing environment context to architecture critique")
    
    critique_prompt = build_critique_prompt(
        draft_text=draft_result.content,
        original_request=original_request,
        job_type_str=job_type_str,
        env_context=env_context
    )

    critique_messages = [{"role": "user", "content": critique_prompt}]
    
    print(f"[critic] Calling Gemini 3 Pro for critique of {job_type_str} task")
    
    try:
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
    """Call Opus to revise its draft based on Gemini's critique."""
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""
    
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

    revision_messages = [{"role": "user", "content": revision_prompt}]
    
    print("[critic] Calling Opus for revision using Gemini critique")
    
    try:
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


# =============================================================================
# VIDEO+CODE DEBUG PIPELINE
# =============================================================================

async def run_video_code_debug_pipeline(
    task: LLMTask,
    envelope: JobEnvelope,
    file_map: Optional[str] = None,
) -> LLMResult:
    """2-step pipeline for Video+Code debug jobs."""
    print("[video-code] Starting Video+Code debug pipeline")
    
    attachments = task.attachments or []
    flags = compute_modality_flags(attachments)
    
    video_attachments = flags.get("video_attachments", [])
    code_attachments = flags.get("code_attachments", [])
    
    print(f"[video-code] Found {len(video_attachments)} video(s), {len(code_attachments)} code file(s)")
    
    # Step 1: Transcribe videos
    video_transcripts = []
    
    for video_att in video_attachments:
        video_path = getattr(video_att, 'file_path', None)
        
        if not video_path:
            file_id = getattr(video_att, 'file_id', None)
            if file_id:
                project_id = getattr(task, 'project_id', 1) or 1
                video_path = f"data/files/{project_id}/{video_att.filename}"
        
        if video_path:
            print(f"[video-code] Step 1: Transcribing video: {video_att.filename}")
            try:
                transcript = await transcribe_video_for_context(video_path)
                video_transcripts.append({
                    "filename": video_att.filename,
                    "transcript": transcript,
                })
            except Exception as e:
                print(f"[video-code] Transcription failed for {video_att.filename}: {e}")
                # v0.15.0: Use fallback handler if available
                if FALLBACKS_AVAILABLE:
                    action, event = handle_video_failure(
                        str(e),
                        has_code=len(code_attachments) > 0,
                        task_id=envelope.job_id,
                    )
                    if action == FallbackAction.SKIP_STEP:
                        video_transcripts.append({
                            "filename": video_att.filename,
                            "transcript": f"[Video transcription failed: {e}]",
                        })
        else:
            video_transcripts.append({
                "filename": video_att.filename,
                "transcript": f"[Video file path not available for: {video_att.filename}]",
            })
    
    print(f"[video-code] Step 1 complete: {len(video_transcripts)} video transcript(s)")
    
    # Step 2: Build context for Sonnet
    transcripts_text = ""
    for vt in video_transcripts:
        transcripts_text += f"\n\n=== Video: {vt['filename']} ===\n{vt['transcript']}"
    
    original_user_message = ""
    for msg in envelope.messages:
        if msg.get("role") == "user":
            original_user_message = msg.get("content", "")
            break
    
    # Build system context with file map
    system_content = f"""You are debugging code based on video recordings of the issue.

VIDEO TRANSCRIPTS (generated by AI vision model):
{transcripts_text}

Use the video context above to understand what happened and help debug/fix the code.
Focus on:
- Any errors or issues visible in the video
- User actions that led to the problem
- Log output or console messages
- UI state changes"""

    if file_map:
        system_content += f"\n\n{file_map}\n\nIMPORTANT: When referring to files, use the [FILE_X] identifiers above."

    enhanced_messages = [{"role": "system", "content": system_content}]
    
    for msg in envelope.messages:
        if msg.get("role") != "system":
            enhanced_messages.append(msg)
    
    envelope.messages = enhanced_messages
    
    # Step 3: Call Sonnet
    sonnet_model = os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929")
    print(f"[video-code] Step 2: Calling Sonnet ({sonnet_model}) with code + transcripts")
    
    try:
        result = await registry_llm_call(
            provider_id="anthropic",
            model_id=sonnet_model,
            messages=envelope.messages,
            job_envelope=envelope,
        )
        
        if not result.is_success():
            print(f"[video-code] Sonnet call failed: {result.error_message}")
            return LLMResult(
                content=result.error_message or "Video+Code pipeline failed",
                provider="anthropic",
                model=sonnet_model,
                finish_reason="error",
                error_message=result.error_message,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                raw_response=None,
            )
        
        print(f"[video-code] Pipeline complete: {len(result.content)} chars")
        
        llm_result = LLMResult(
            content=result.content,
            provider="anthropic",
            model=sonnet_model,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )
        
        llm_result.routing_decision = {
            "job_type": "video.code.debug",
            "provider": "anthropic",
            "model": sonnet_model,
            "reason": "Video+Code debug pipeline: Gemini3 transcription → Sonnet coding",
            "pipeline": {
                "video_count": len(video_transcripts),
                "code_count": len(code_attachments),
                "transcript_chars": len(transcripts_text),
            }
        }
        
        return llm_result
        
    except Exception as exc:
        logger.exception("[video-code] Sonnet call failed: %s", exc)
        return LLMResult(
            content="",
            provider="anthropic",
            model=sonnet_model,
            finish_reason="error",
            error_message=str(exc),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )


async def run_high_stakes_with_critique(
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
    file_map: Optional[str] = None,
) -> LLMResult:
    """Run 3-step critique pipeline for high-stakes Opus work."""
    print(f"[critic] High-stakes pipeline enabled: job_type={job_type_str} model={model_id}")
    
    # v0.15.0: Start audit trace if available
    trace = None
    if AUDIT_AVAILABLE and AUDIT_ENABLED:
        audit_logger = get_audit_logger()
        if audit_logger:
            trace = audit_logger.start_trace(
                session_id=envelope.session_id,
                project_id=envelope.project_id,
                user_text=None,
                is_critical=True,
                sandbox_mode=False,
                request_id=getattr(envelope, "job_id", None),
            )
            trace.log_request_start(
                job_type=job_type_str,
                attachments=task.attachments,
                frontier_override=False,
                file_map_injected=bool(file_map),
                reason="high-stakes critique pipeline",
            )
            trace.log_routing_decision(
                job_type=job_type_str,
                provider=provider_id,
                model=model_id,
                reason="Opus draft + Gemini critic + Opus revision",
                frontier_override=False,
                file_map_injected=bool(file_map),
            )
    
    # Pre-step: Transcribe video attachments if present
    attachments = task.attachments or []
    if attachments:
        flags = compute_modality_flags(attachments)
        video_attachments = flags.get("video_attachments", [])
        
        if video_attachments:
            print(f"[critic] Pre-step: Transcribing {len(video_attachments)} video(s)")
            
            video_transcripts = []
            for video_att in video_attachments:
                video_path = getattr(video_att, 'file_path', None)
                
                if not video_path:
                    file_id = getattr(video_att, 'file_id', None)
                    if file_id:
                        project_id = getattr(task, 'project_id', 1) or 1
                        video_path = f"data/files/{project_id}/{video_att.filename}"
                
                if video_path:
                    print(f"[critic] Pre-step: Transcribing {video_att.filename}")
                    try:
                        transcript = await transcribe_video_for_context(video_path)
                        video_transcripts.append({
                            "filename": video_att.filename,
                            "transcript": transcript,
                        })
                    except Exception as e:
                        print(f"[critic] Pre-step: Transcription failed: {e}")
            
            if video_transcripts:
                transcripts_text = ""
                for vt in video_transcripts:
                    transcripts_text += f"\n\n=== Video: {vt['filename']} ===\n{vt['transcript']}"
                
                print(f"[critic] Pre-step complete: Injected {len(transcripts_text)} chars")
                
                video_system_msg = {
                    "role": "system",
                    "content": f"""VIDEO CONTEXT (transcribed for this high-stakes review):
{transcripts_text}

Use the video context above to inform your analysis."""
                }
                
                envelope.messages = [video_system_msg] + list(envelope.messages)
    
    # v0.15.0: Inject file map if not already present
    if file_map:
        envelope.messages = inject_file_map_into_messages(envelope.messages, file_map)
    
    # Step 1: Generate Opus Draft
    try:
        draft_result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            job_envelope=envelope,
        )
        
        if not draft_result.is_success():
            logger.error("[critic] Opus draft failed: %s", draft_result.error_message)
            if trace:
                trace.log_error("opus_draft", "OPUS_DRAFT_FAILED", str(draft_result.error_message))
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
        
        if trace:
            trace.log_model_call(
                "opus_draft",
                provider_id,
                model_id,
                draft_result.usage.prompt_tokens,
                draft_result.usage.completion_tokens,
                draft_result.usage.cost_estimate,
            )
        
    except Exception as exc:
        logger.exception("[critic] Opus draft call failed: %s", exc)
        if trace:
            trace.log_error("opus_draft", "OPUS_DRAFT_EXCEPTION", str(exc))
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
    
    # Step 2: Check Draft Length
    if not is_long_enough_for_critique(draft.content):
        print(f"[critic] Draft too short for critique ({len(draft.content or '')} chars)")
        if trace:
            trace.finalize(success=True)
        return draft
    
    # Step 3: Call Gemini Critic
    critique = await call_gemini_critic(
        original_task=task,
        draft_result=draft,
        job_type_str=job_type_str,
    )
    
    if not critique or not critique.content:
        print("[critic] Gemini critic failed; returning original Opus draft")
        if trace:
            trace.log_warning("CRITIQUE_FAILED", "Returning original draft")
            trace.finalize(success=True)
        return draft
    
    if trace:
        trace.log_model_call(
            "gemini_critique",
            "google",
            GEMINI_CRITIC_MODEL,
            critique.prompt_tokens,
            critique.completion_tokens,
            critique.cost_usd,
        )
    
    # Step 4: Call Opus for Revision
    revision = await call_opus_revision(
        original_task=task,
        draft_result=draft,
        critique_result=critique,
        opus_model_id=model_id,
    )
    
    if not revision or not revision.content:
        print("[critic] Opus revision failed; returning original Opus draft")
        if trace:
            trace.log_warning("REVISION_FAILED", "Returning original draft")
            trace.finalize(success=True)
        return draft
    
    if trace:
        trace.log_model_call(
            "opus_revision",
            provider_id,
            model_id,
            revision.prompt_tokens,
            revision.completion_tokens,
            revision.cost_usd,
        )
        trace.finalize(success=True)
    
    # Success
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
    """Primary async LLM call entry point."""
    session_id = getattr(task, "session_id", None)
    project_id = getattr(task, "project_id", 1) or 1

    job_type_specified = task.job_type is not None and task.job_type != JobType.UNKNOWN
    has_attachments = bool(task.attachments and len(task.attachments) > 0)

    # ==========================================================================
    # v0.15.1: FRONTIER OVERRIDE CHECK (BEFORE ALL OTHER ROUTING)
    # ==========================================================================
    
    # Extract user message for override detection
    # Handle both simple string and multimodal (list) content formats
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    original_message = ""
    
    if user_messages:
        content = user_messages[-1].get("content", "")
        if isinstance(content, str):
            original_message = content
        elif isinstance(content, list):
            # Multimodal message - extract text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    original_message = part.get("text", "")
                    break
                elif isinstance(part, str):
                    original_message = part
                    break
        
        if ROUTER_DEBUG:
            _debug_log(f"Extracted message for override check: {repr(original_message[:100])}...")
    
    # Check for OVERRIDE command
    override_result = detect_frontier_override(original_message)
    frontier_override_active = override_result is not None
    cleaned_message = None
    
    if frontier_override_active:
        force_provider, force_model_id, cleaned_message = override_result
        
        print(f"[router] FRONTIER OVERRIDE ACTIVE: {force_provider} → {force_model_id}")
        if ROUTER_DEBUG:
            _debug_log("=" * 70)
            _debug_log("FRONTIER OVERRIDE DETECTED")
            _debug_log(f"  Provider: {force_provider}")
            _debug_log(f"  Model: {force_model_id}")
            _debug_log(f"  Original message: {len(original_message)} chars")
            _debug_log(f"  Cleaned message: {len(cleaned_message)} chars")
            _debug_log("=" * 70)

    # v0.15.0: Compute modality flags with file_classifier
    file_map = None
    if has_attachments:
        modality_flags = compute_modality_flags(task.attachments or [])
        file_map = modality_flags.get("file_map")
        
        if ROUTER_DEBUG and file_map:
            _debug_log(f"File map generated ({len(file_map)} chars)")

    # Synthesize envelope with file map (and cleaned message if override active)
    try:
        envelope = synthesize_envelope_from_task(
            task=task,
            session_id=session_id,
            project_id=project_id,
            file_map=file_map,
            cleaned_message=cleaned_message,  # v0.15.1: Pass cleaned message
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
    # PROVIDER/MODEL SELECTION
    # ==========================================================================
    
    if ROUTER_DEBUG:
        _debug_log("=" * 70)
        _debug_log("ROUTER START")
        _debug_log("=" * 70)
        _debug_log(f"Task job_type: {task.job_type.value if task.job_type else 'None'}")
        _debug_log(f"Task attachments: {len(task.attachments) if task.attachments else 0}")
        _debug_log(f"Task force_provider: {task.force_provider.value if task.force_provider else 'None'}")
        _debug_log(f"Frontier override active: {frontier_override_active}")
    
    # ==========================================================================
    # v0.15.1: FRONTIER OVERRIDE TAKES PRIORITY
    # ==========================================================================
    
    if frontier_override_active:
        provider_id = force_provider
        model_id = force_model_id
        reason = f"OVERRIDE → {force_provider} frontier ({force_model_id})"
        
        # Still need a classified_type for pipeline checks
        # Use TEXT_HEAVY as default since OVERRIDE implies serious work
        if force_provider == "anthropic":
            classified_type = JobType.ORCHESTRATOR
        elif force_provider == "google":
            classified_type = JobType.VIDEO_HEAVY  # Use video-capable type for Gemini
        else:
            classified_type = JobType.TEXT_HEAVY
        
        print(f"[router] OVERRIDE routing: {provider_id}/{model_id}")
        
        if ROUTER_DEBUG:
            _debug_log(f"OVERRIDE: Skipping normal classification")
            _debug_log(f"  → Provider: {provider_id}")
            _debug_log(f"  → Model: {model_id}")
            _debug_log(f"  → Implied job type: {classified_type.value}")
    
    # Priority 1: Explicit force_provider override (legacy)
    elif task.force_provider is not None:
        provider_id = task.force_provider.value
        model_id = task.model or DEFAULT_MODELS.get(provider_id, "")
        reason = "force_provider override"
        classified_type = task.job_type
        logger.info("[router] Using force_provider: %s / %s", provider_id, model_id)
        if ROUTER_DEBUG:
            _debug_log(f"Priority 1: force_provider override → {provider_id}/{model_id}")
    
    # Priority 2: Job classifier
    else:
        if ROUTER_DEBUG:
            _debug_log(f"Priority 2: Using job classifier")
        
        if task.job_type and task.job_type != JobType.UNKNOWN:
            classified_type = task.job_type
            
            if ROUTER_DEBUG:
                _debug_log(f"  Task has pre-set job_type: {classified_type.value}")
            
            if classified_type == JobType.CHAT_LIGHT or classified_type.value in ["chat.light", "chat_light", "casual_chat"]:
                provider = Provider.OPENAI
                provider_id = "openai"
                model_id = os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini")
                reason = "Pre-classified as CHAT_LIGHT, forced to GPT mini"
                print(f"[router] RESPECTING PRE-CLASSIFICATION: {classified_type.value} → FORCED to {provider_id}/{model_id}")
            else:
                decision_temp = get_routing_for_job_type(classified_type.value)
                provider = decision_temp.provider
                provider_id = provider.value
                model_id = decision_temp.model
                reason = f"Pre-classified as {classified_type.value}"
                print(f"[router] RESPECTING PRE-CLASSIFICATION: {classified_type.value} → {provider_id}/{model_id}")
        else:
            if ROUTER_DEBUG:
                _debug_log(f"  No valid pre-classification, calling classifier...")
            
            provider, model_id, classified_type, reason = classify_and_route(task)
            provider_id = provider.value
            
            if ROUTER_DEBUG:
                _debug_log(f"  Classifier returned: {classified_type.value} → {provider_id}/{model_id}")
            
            if classified_type == JobType.CHAT_LIGHT or classified_type.value in ["chat.light", "chat_light", "casual_chat"]:
                provider = Provider.OPENAI
                provider_id = "openai"
                model_id = os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini")
                reason = "HARD OVERRIDE: CHAT_LIGHT forced to GPT mini"
                print("[router] OVERRIDE: Forcing CHAT_LIGHT → openai/gpt-4.1-mini")
        
        if task.model:
            if ROUTER_DEBUG:
                _debug_log(f"  Task.model override: {task.model}")
            model_id = task.model
        
        print(f"[router] Final routing: {classified_type.value} → {provider_id}/{model_id}")
        
        if ROUTER_DEBUG:
            _debug_log("=" * 70)
            _debug_log(f"ROUTING DECISION: {classified_type.value} → {provider_id}/{model_id}")
            _debug_log("=" * 70)
        
        decision = RoutingDecision(
            job_type=classified_type,
            provider=provider,
            model=model_id,
            reason=reason,
        )
        
        # Attachment safety check (skip if frontier override is active)
        provider_id, model_id, reason = _check_attachment_safety(
            task=task,
            decision=decision,
            has_attachments=has_attachments,
            job_type_specified=job_type_specified,
        )
    
    # ==========================================================================
    # HARD RULE ENFORCEMENT (skip for frontier override)
    # ==========================================================================
    
    if not frontier_override_active and is_claude_forbidden(classified_type):
        if provider_id == "anthropic":
            logger.error(
                "[router] BLOCKED: Attempted to route %s to Claude - forcing to Gemini",
                classified_type.value,
            )
            provider_id = "google"
            model_id = os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro")
            reason = f"FORCED: {classified_type.value} cannot go to Claude"
    
    # ==========================================================================
    # VIDEO+CODE DEBUG PIPELINE CHECK (still runs with override)
    # ==========================================================================
    
    if classified_type == JobType.VIDEO_CODE_DEBUG:
        if ROUTER_DEBUG:
            _debug_log("VIDEO+CODE DEBUG PIPELINE TRIGGERED")
        
        print(f"[router] VIDEO+CODE PIPELINE: {classified_type.value}")
        
        return await run_video_code_debug_pipeline(
            task=task,
            envelope=envelope,
            file_map=file_map,
        )
    
    # ==========================================================================
    # HIGH-STAKES CRITIQUE PIPELINE CHECK (still runs with override if Opus)
    # ==========================================================================
    
    normalized_job_type = normalize_job_type_for_high_stakes(classified_type.value, reason)
    
    should_run_critique = (
        provider_id == "anthropic" and
        is_opus_model(model_id) and
        is_high_stakes_job(normalized_job_type)
    )
    
    if should_run_critique:
        if ROUTER_DEBUG:
            _debug_log("HIGH-STAKES CRITIQUE PIPELINE TRIGGERED")
            _debug_log(f"  Normalized Job Type: {normalized_job_type}")
        
        print(f"[router] HIGH-STAKES PIPELINE: {classified_type.value} → {normalized_job_type}")
        
        return await run_high_stakes_with_critique(
            task=task,
            provider_id=provider_id,
            model_id=model_id,
            envelope=envelope,
            job_type_str=normalized_job_type,
            file_map=file_map,
        )
    
    # ==========================================================================
    # NORMAL LLM CALL
    # ==========================================================================

    # v0.15.2: Non-sensitive audit trace for normal requests
    trace = None
    audit_logger = None
    if AUDIT_AVAILABLE and AUDIT_ENABLED:
        audit_logger = get_audit_logger()
        if audit_logger:
            trace = audit_logger.start_trace(
                session_id=envelope.session_id,
                project_id=envelope.project_id,
                user_text=None,
                is_critical=False,
                sandbox_mode=False,
                request_id=getattr(envelope, 'job_id', None),
            )
            # No prompt content. Only counts + decision.
            trace.log_request_start(
                job_type=(task.job_type.value if task.job_type else ''),
                attachments=task.attachments,
                frontier_override=frontier_override_active,
                file_map_injected=bool(file_map),
                reason=reason,
            )
            trace.log_routing_decision(
                job_type=classified_type.value,
                provider=provider_id,
                model=model_id,
                reason=reason,
                frontier_override=frontier_override_active,
                file_map_injected=bool(file_map),
            )

    _t0_ms = int(__import__('time').time() * 1000)

    try:
        result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            job_envelope=envelope,
        )

        _dur_ms = int(__import__('time').time() * 1000) - _t0_ms
    except Exception as exc:
        logger.exception("[router] llm_call failed: %s", exc)
        if trace:
            trace.log_error('primary', 'LLM_CALL_EXCEPTION', str(exc))
        if audit_logger and trace:
            audit_logger.complete_trace(trace, success=False, error_message=str(exc))
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
        if trace:
            trace.log_model_call(
                'primary',
                provider_id,
                model_id,
                'primary',
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
                duration_ms=_dur_ms,
                success=False,
                error=result.error_message or 'error',
                cost_usd=result.usage.cost_estimate,
            )
        if audit_logger and trace:
            audit_logger.complete_trace(trace, success=False, error_message=result.error_message or '')
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

    if trace:
        trace.log_model_call(
            'primary',
            provider_id,
            model_id,
            'primary',
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
            duration_ms=_dur_ms,
            success=True,
            error=None,
            cost_usd=result.usage.cost_estimate,
        )
    if audit_logger and trace:
        audit_logger.complete_trace(trace, success=True)

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
            "file_map_injected": file_map is not None,
            "frontier_override": frontier_override_active,
        },
    )


# =============================================================================
# HIGH-LEVEL HELPERS (Async)
# =============================================================================

async def quick_chat_async(message: str, context: Optional[str] = None) -> LLMResult:
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


async def request_code_async(message: str, context: Optional[str] = None, high_stakes: bool = False) -> LLMResult:
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


async def review_work_async(message: str, context: Optional[str] = None) -> LLMResult:
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
# SYNC WRAPPERS
# =============================================================================

def call_llm(task: LLMTask) -> LLMResult:
    """Sync wrapper for call_llm_async."""
    import asyncio
    return asyncio.run(call_llm_async(task))


def quick_chat(message: str, context: Optional[str] = None) -> LLMResult:
    """Sync wrapper for quick_chat_async."""
    import asyncio
    return asyncio.run(quick_chat_async(message=message, context=context))


def request_code(message: str, context: Optional[str] = None, high_stakes: bool = False) -> LLMResult:
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

def analyze_with_vision(prompt: str, image_description: Optional[str] = None, context: Optional[str] = None) -> LLMResult:
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
        "routing_version": "0.15.1",
        "spec_compliance": "Critical Pipeline Spec v1.0",
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
        "frontier_models": {
            "gemini": GEMINI_FRONTIER_MODEL_ID,
            "anthropic": ANTHROPIC_FRONTIER_MODEL_ID,
            "openai": OPENAI_FRONTIER_MODEL_ID,
        },
        "hard_rules": [
            "Images/video NEVER go to Claude",
            "PDFs NEVER go to Claude",
            "MIXED_FILE (docs with images) → Gemini",
            "opus.critic is EXPLICIT ONLY",
            "OVERRIDE → Frontier model (skips normal routing)",
        ],
        "modules_integrated": {
            "file_classifier": FILE_CLASSIFIER_AVAILABLE,
            "audit_logger": AUDIT_AVAILABLE,
            "relationship_detector": RELATIONSHIP_DETECTOR_AVAILABLE,
            "token_budgeting": TOKEN_BUDGETING_AVAILABLE,
            "fallbacks": FALLBACKS_AVAILABLE,
        },
        "critique_pipeline": {
            "enabled": True,
            "high_stakes_types": list(HIGH_STAKES_JOB_TYPES),
            "min_length_chars": MIN_CRITIQUE_CHARS,
            "critic_model": GEMINI_CRITIC_MODEL,
        },
    }


def is_policy_routing_enabled() -> bool:
    """Always True - we use job_classifier."""
    return True


def enable_policy_routing() -> None:
    """No-op for compatibility."""
    logger.info("[router] Policy routing is always enabled")


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
    
    # v0.15.0: File map injection
    "inject_file_map_into_messages",
    
    # v0.15.1: Frontier override
    "detect_frontier_override",
    "GEMINI_FRONTIER_MODEL_ID",
    "ANTHROPIC_FRONTIER_MODEL_ID", 
    "OPENAI_FRONTIER_MODEL_ID",
    
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