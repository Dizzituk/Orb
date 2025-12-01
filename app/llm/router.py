# FILE: app/llm/router.py
"""
LLM Router: The single entry point for all LLM calls in Orb.

All other parts of Orb MUST call through this router.
Never call the raw provider clients directly.

Routing rules:
- Low-stakes text jobs → GPT only
- Medium dev work → GPT or Claude (configurable via SMART_PROVIDER)
- Heavy dev / architecture → Claude (primary)
- High-stakes / critical work → Claude (primary) + Gemini (critic review)
- Vision/analysis → Gemini
- Unknown → GPT for text-like, Claude for dev-like

NEW: Policy-based routing available via use_policy=True or by setting
     ORB_USE_POLICY_ROUTING=true environment variable.
"""
import os
import logging
from typing import Optional

from app.llm.schemas import (
    LLMTask,
    LLMResult,
    JobType,
    Provider,
    RoutingConfig,
)
from app.llm.clients import (
    call_openai,
    call_anthropic,
    call_google,
)

logger = logging.getLogger(__name__)

# Check if policy-based routing is enabled
_USE_POLICY_ROUTING = os.getenv("ORB_USE_POLICY_ROUTING", "false").lower() == "true"

# Try to import policy module (optional)
_policy_available = False
try:
    from app.llm.policy import (
        load_routing_policy,
        make_routing_decision,
        resolve_job_type,
        Provider as PolicyProvider,
        AttachmentMode,
        RoutingDecision,
        RoutingPolicy,
        PolicyError,
    )
    _policy_available = True
except ImportError:
    pass


# ============== SYSTEM PROMPT BUILDERS ==============

def _build_gpt_system_prompt(task: LLMTask) -> str:
    """Build system prompt for GPT (conversational/lightweight tasks)."""
    base = """You are Orb, a fast and helpful assistant.

Your role: Handle conversational tasks, summaries, explanations, and lightweight text work.

Be concise, clear, and direct. Get to the point quickly."""

    if task.project_context:
        base += f"\n\nPROJECT CONTEXT:\n{task.project_context}"
    
    if task.system_prompt:
        base += f"\n\nADDITIONAL INSTRUCTIONS:\n{task.system_prompt}"
    
    return base


def _build_claude_system_prompt(task: LLMTask) -> str:
    """Build system prompt for Claude (engineering/architecture tasks)."""
    base = """You are Orb's engineering brain — a senior backend architect and implementer.

Your role: Handle complex code, architecture design, full-file generation, and technical planning.

CRITICAL RULES:
1. When modifying existing files: ALWAYS ask for the full current file content first, then return the COMPLETE updated file.
2. NEVER return partial files, diffs, or snippets. Always return complete, runnable code.
3. Include all imports, all functions, all boilerplate — the user should be able to copy-paste directly.
4. Write clear comments explaining non-obvious decisions.
5. Think through edge cases before writing code.

Be precise, technical, and thorough."""

    if task.project_context:
        base += f"\n\nPROJECT CONTEXT:\n{task.project_context}"
    
    if task.system_prompt:
        base += f"\n\nADDITIONAL INSTRUCTIONS:\n{task.system_prompt}"
    
    return base


def _build_gemini_system_prompt(task: LLMTask, is_critic: bool = False) -> str:
    """Build system prompt for Gemini (review/analysis tasks)."""
    if is_critic:
        base = """You are Orb's critic — a code reviewer and quality analyst.

Your role: Review the provided work from Claude (Orb's engineer) and identify:
1. Logical inconsistencies or bugs
2. Security concerns
3. Edge cases not handled
4. Over-complication or unnecessary complexity
5. Missing error handling
6. Potential performance issues

Be constructive but thorough. Prioritize issues by severity (CRITICAL / HIGH / MEDIUM / LOW).

Format your review as:
## Summary
(1-2 sentence overall assessment)

## Issues Found
### [SEVERITY] Issue Title
Description and suggested fix.

## Recommendations
(Optional improvements that aren't bugs)"""
    else:
        base = """You are Orb's analyst — a reviewer and vision specialist.

Your role: Analyze content, review work, identify patterns, and provide structured feedback.

Be analytical, precise, and actionable."""

    if task.project_context:
        base += f"\n\nPROJECT CONTEXT:\n{task.project_context}"
    
    if task.system_prompt:
        base += f"\n\nADDITIONAL INSTRUCTIONS:\n{task.system_prompt}"
    
    return base


# ============== ORIGINAL ROUTING LOGIC ==============

def _determine_provider(task: LLMTask) -> tuple[Provider, bool]:
    """
    Determine which provider(s) to use for a task.
    
    Returns:
        Tuple of (primary_provider, needs_critic_review)
    """
    job_type = task.job_type
    
    # Check for forced provider
    if task.force_provider:
        return task.force_provider, False
    
    # GPT-only jobs
    if job_type in RoutingConfig.GPT_ONLY_JOBS:
        return Provider.OPENAI, False
    
    # Medium dev jobs → use configured "smart" provider
    if job_type in RoutingConfig.MEDIUM_DEV_JOBS:
        return RoutingConfig.SMART_PROVIDER, False
    
    # Claude primary jobs (heavy dev)
    if job_type in RoutingConfig.CLAUDE_PRIMARY_JOBS:
        return Provider.ANTHROPIC, False
    
    # High-stakes jobs → Claude + Gemini critic
    if job_type in RoutingConfig.HIGH_STAKES_JOBS:
        return Provider.ANTHROPIC, True
    
    # Gemini jobs (vision/analysis)
    if job_type in RoutingConfig.GEMINI_JOBS:
        return Provider.GOOGLE, False
    
    # Unknown job type → guess based on content
    return _guess_provider_for_unknown(task)


def _guess_provider_for_unknown(task: LLMTask) -> tuple[Provider, bool]:
    """
    For unknown job types, guess the best provider based on content.
    
    Heuristic:
    - If messages contain code-related keywords → Claude
    - Otherwise → GPT (safe default for text)
    """
    code_keywords = [
        "code", "function", "class", "def ", "import ", "return ",
        "bug", "fix", "refactor", "implement", "architecture",
        "api", "endpoint", "database", "schema", "model",
        ".py", ".js", ".ts", ".html", ".css", ".sql",
    ]
    
    # Check last message content
    if task.messages:
        last_content = task.messages[-1].get("content", "").lower()
        for keyword in code_keywords:
            if keyword in last_content:
                return Provider.ANTHROPIC, False
    
    # Default to GPT for general text
    return Provider.OPENAI, False


def _call_provider(
    provider: Provider,
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    attachments: Optional[list[dict]] = None,
    enable_web_search: bool = False,
) -> tuple[str, Optional[dict]]:
    """
    Call the specified provider.
    
    Returns:
        Tuple of (content, usage_dict)
    """
    if provider == Provider.OPENAI:
        return call_openai(system_prompt, messages, temperature=temperature)
    elif provider == Provider.ANTHROPIC:
        return call_anthropic(system_prompt, messages, temperature=temperature)
    elif provider == Provider.GOOGLE:
        return call_google(
            system_prompt, 
            messages, 
            temperature=temperature,
            attachments=attachments,
            enable_web_search=enable_web_search,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============== MAIN ROUTER FUNCTION ==============

def call_llm(
    task: LLMTask,
    semantic_context: Optional[str] = None,
    use_policy: Optional[bool] = None,
) -> LLMResult:
    """
    Route an LLM task to the appropriate provider(s).
    
    This is the ONLY function that should be called from other parts of Orb.
    
    Args:
        task: LLMTask containing job_type, messages, and optional context
        semantic_context: Optional RAG context to inject (NEW)
        use_policy: Force policy-based routing on/off (NEW)
    
    Returns:
        LLMResult with response content and optional critic review
    
    Example:
        from app.llm import call_llm, LLMTask, JobType
        
        result = call_llm(LLMTask(
            job_type=JobType.CASUAL_CHAT,
            messages=[{"role": "user", "content": "Hello!"}],
        ))
        print(result.content)
    """
    # Determine if we should use policy-based routing
    should_use_policy = use_policy if use_policy is not None else _USE_POLICY_ROUTING
    
    if should_use_policy and _policy_available:
        return _call_llm_with_policy(task, semantic_context)
    else:
        return _call_llm_original(task, semantic_context)


def _call_llm_original(
    task: LLMTask,
    semantic_context: Optional[str] = None,
) -> LLMResult:
    """Original routing implementation."""
    # Determine routing
    primary_provider, needs_critic = _determine_provider(task)
    
    # Inject semantic context if provided
    messages = list(task.messages)
    if semantic_context:
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                original = msg.get("content", "")
                messages[i] = {
                    "role": "user",
                    "content": f"RELEVANT CONTEXT:\n{semantic_context}\n\n---\n\nUSER REQUEST:\n{original}"
                }
                break
    
    # Build system prompt for primary provider
    if primary_provider == Provider.OPENAI:
        system_prompt = _build_gpt_system_prompt(task)
    elif primary_provider == Provider.ANTHROPIC:
        system_prompt = _build_claude_system_prompt(task)
    else:
        system_prompt = _build_gemini_system_prompt(task, is_critic=False)
    
    # Get attachments if present
    attachments = getattr(task, 'attachments', None)
    
    # Call primary provider
    primary_content, primary_usage = _call_provider(
        primary_provider,
        system_prompt,
        messages,
        attachments=attachments,
    )
    
    # Initialize result
    result = LLMResult(
        provider=primary_provider,
        content=primary_content,
        job_type=task.job_type,
        was_reviewed=False,
        usage=primary_usage,
    )
    
    # If high-stakes, get Gemini critic review
    if needs_critic:
        critic_system_prompt = _build_gemini_system_prompt(task, is_critic=True)
        
        # Build critic messages: show the original request + Claude's response
        original_request = task.messages[-1].get("content", "") if task.messages else ""
        critic_messages = [
            {
                "role": "user",
                "content": f"""Review the following work from Claude (Orb's engineer).

ORIGINAL REQUEST:
{original_request}

CLAUDE'S RESPONSE:
{primary_content}

Please review this work for issues, bugs, security concerns, and areas for improvement."""
            }
        ]
        
        critic_content, critic_usage = _call_provider(
            Provider.GOOGLE,
            critic_system_prompt,
            critic_messages,
        )
        
        result.critic_provider = Provider.GOOGLE
        result.critic_review = critic_content
        result.was_reviewed = True
    
    return result


def _call_llm_with_policy(
    task: LLMTask,
    semantic_context: Optional[str] = None,
) -> LLMResult:
    """Policy-based routing implementation."""
    policy = load_routing_policy()
    
    # Get job type as string
    job_type_str = task.job_type.value if isinstance(task.job_type, JobType) else str(task.job_type)
    
    # Check for forced provider
    if task.force_provider:
        primary_provider = task.force_provider
        needs_critic = False
        temperature = 0.7
        logger.info(f"Force override: {job_type_str} → {primary_provider.value}")
    else:
        # Get policy-based routing decision
        try:
            content = task.messages[-1].get("content", "") if task.messages else ""
            attachments = getattr(task, 'attachments', None)
            
            decision = make_routing_decision(
                job_type=job_type_str,
                content=content,
                attachments=attachments,
                policy=policy,
            )
            
            # Map policy provider to our Provider enum
            provider_map = {
                "openai": Provider.OPENAI,
                "anthropic": Provider.ANTHROPIC,
                "gemini": Provider.GOOGLE,
            }
            primary_provider = provider_map.get(decision.primary_provider.value, Provider.OPENAI)
            needs_critic = decision.is_high_stakes and decision.review_provider is not None
            temperature = decision.temperature
            
            logger.info(
                f"Policy routing: {job_type_str} → {primary_provider.value} "
                f"(review: {needs_critic})"
            )
        except PolicyError as e:
            logger.warning(f"Policy error, falling back to original routing: {e}")
            return _call_llm_original(task, semantic_context)
    
    # Inject semantic context if provided
    messages = list(task.messages)
    if semantic_context:
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                original = msg.get("content", "")
                messages[i] = {
                    "role": "user",
                    "content": f"RELEVANT CONTEXT:\n{semantic_context}\n\n---\n\nUSER REQUEST:\n{original}"
                }
                break
    
    # Build system prompt
    if primary_provider == Provider.OPENAI:
        system_prompt = _build_gpt_system_prompt(task)
    elif primary_provider == Provider.ANTHROPIC:
        system_prompt = _build_claude_system_prompt(task)
    else:
        system_prompt = _build_gemini_system_prompt(task, is_critic=False)
    
    # Get attachments
    attachments = getattr(task, 'attachments', None)
    
    # Call primary provider
    try:
        primary_content, primary_usage = _call_provider(
            primary_provider,
            system_prompt,
            messages,
            temperature=temperature,
            attachments=attachments,
        )
    except Exception as e:
        logger.error(f"Primary provider failed: {e}")
        return LLMResult(
            provider=primary_provider,
            content=f"Error: {e}",
            job_type=task.job_type,
            was_reviewed=False,
            error=str(e),
        )
    
    # Initialize result
    result = LLMResult(
        provider=primary_provider,
        content=primary_content,
        job_type=task.job_type,
        was_reviewed=False,
        usage=primary_usage,
    )
    
    # If high-stakes, get critic review
    if needs_critic:
        critic_system_prompt = _build_gemini_system_prompt(task, is_critic=True)
        
        original_request = task.messages[-1].get("content", "") if task.messages else ""
        critic_messages = [
            {
                "role": "user",
                "content": f"""Review the following work from Claude (Orb's engineer).

ORIGINAL REQUEST:
{original_request}

CLAUDE'S RESPONSE:
{primary_content}

Please review this work for issues, bugs, security concerns, and areas for improvement."""
            }
        ]
        
        try:
            critic_content, critic_usage = _call_provider(
                Provider.GOOGLE,
                critic_system_prompt,
                critic_messages,
                temperature=0.5,
            )
            
            result.critic_provider = Provider.GOOGLE
            result.critic_review = critic_content
            result.was_reviewed = True
            result.critic_usage = critic_usage
        except Exception as e:
            logger.error(f"Critic review failed: {e}")
            result.critic_error = str(e)
    
    return result


# ============== CONVENIENCE FUNCTIONS ==============

def quick_chat(message: str, context: Optional[str] = None) -> str:
    """
    Quick helper for casual chat. Always routes to GPT.
    
    Args:
        message: User message
        context: Optional project context
    
    Returns:
        Response string
    """
    result = call_llm(LLMTask(
        job_type=JobType.CASUAL_CHAT,
        messages=[{"role": "user", "content": message}],
        project_context=context,
    ))
    return result.content


def request_code(
    message: str,
    context: Optional[str] = None,
    high_stakes: bool = False,
) -> LLMResult:
    """
    Request code generation or modification. Routes to Claude.
    
    Args:
        message: Code request
        context: Optional project context
        high_stakes: If True, also gets Gemini review
    
    Returns:
        LLMResult (check .critic_review if high_stakes=True)
    """
    job_type = JobType.HIGH_STAKES_INFRA if high_stakes else JobType.CODEGEN_FULL_FILE
    
    return call_llm(LLMTask(
        job_type=job_type,
        messages=[{"role": "user", "content": message}],
        project_context=context,
    ))


def review_work(content: str, context: Optional[str] = None) -> str:
    """
    Get Gemini to review/critique provided content.
    
    Args:
        content: Content to review
        context: Optional project context
    
    Returns:
        Review/critique string
    """
    result = call_llm(LLMTask(
        job_type=JobType.CODE_REVIEW,
        messages=[{"role": "user", "content": f"Review this:\n\n{content}"}],
        project_context=context,
        force_provider=Provider.GOOGLE,  # Force Gemini for pure review
    ))
    return result.content


# ============== NEW CONVENIENCE FUNCTIONS ==============

def analyze_with_vision(
    prompt: str,
    attachments: list[dict],
    context: Optional[str] = None,
) -> str:
    """
    Analyze images/documents using Gemini vision.
    
    Args:
        prompt: Analysis prompt
        attachments: List of {mime_type, data} dicts
        context: Optional project context
    
    Returns:
        Analysis string
    """
    task = LLMTask(
        job_type=JobType.VISION,
        messages=[{"role": "user", "content": prompt}],
        project_context=context,
        attachments=attachments,
    )
    result = call_llm(task)
    return result.content


def web_search_query(query: str, context: Optional[str] = None) -> str:
    """
    Perform web search with Gemini grounding.
    
    Args:
        query: Search query
        context: Optional project context
    
    Returns:
        Search results with sources
    """
    result = call_llm(LLMTask(
        job_type=JobType.WEB_SEARCH,
        messages=[{"role": "user", "content": query}],
        project_context=context,
    ))
    return result.content


# ============== INTROSPECTION ==============

def list_job_types() -> list[str]:
    """List all available job types."""
    return [jt.value for jt in JobType]


def get_routing_info(job_type: JobType) -> dict:
    """Get routing info for a job type."""
    if job_type in RoutingConfig.GPT_ONLY_JOBS:
        return {"provider": "openai", "needs_review": False}
    elif job_type in RoutingConfig.MEDIUM_DEV_JOBS:
        return {"provider": RoutingConfig.SMART_PROVIDER.value, "needs_review": False}
    elif job_type in RoutingConfig.CLAUDE_PRIMARY_JOBS:
        return {"provider": "anthropic", "needs_review": False}
    elif job_type in RoutingConfig.HIGH_STAKES_JOBS:
        return {"provider": "anthropic", "needs_review": True, "reviewer": "google"}
    elif job_type in RoutingConfig.GEMINI_JOBS:
        return {"provider": "google", "needs_review": False}
    else:
        return {"provider": "unknown", "needs_review": False}


def is_policy_routing_enabled() -> bool:
    """Check if policy-based routing is enabled."""
    return _USE_POLICY_ROUTING and _policy_available


def enable_policy_routing(enabled: bool = True) -> None:
    """Enable or disable policy-based routing at runtime."""
    global _USE_POLICY_ROUTING
    _USE_POLICY_ROUTING = enabled and _policy_available