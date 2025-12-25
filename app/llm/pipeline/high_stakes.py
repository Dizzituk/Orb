# FILE: app/llm/pipeline/high_stakes.py
"""High-stakes critique pipeline configuration and prompt builders.

Extracted from app.llm.router to keep the router thin and make sanity checks easier.
"""

from typing import Dict, Any, Optional
import os
import textwrap

from app.llm.schemas import JobType
from app.jobs.schemas import JobType as Phase4JobType
from app.llm.job_classifier import (
    GEMINI_FRONTIER_MODEL_ID,
    ANTHROPIC_FRONTIER_MODEL_ID,
    OPENAI_FRONTIER_MODEL_ID,
)

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
