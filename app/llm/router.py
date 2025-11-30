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
"""
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


# ============== ROUTING LOGIC ==============

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
) -> tuple[str, Optional[dict]]:
    """
    Call the specified provider.
    
    Returns:
        Tuple of (content, usage_dict)
    """
    if provider == Provider.OPENAI:
        return call_openai(system_prompt, messages)
    elif provider == Provider.ANTHROPIC:
        return call_anthropic(system_prompt, messages)
    elif provider == Provider.GOOGLE:
        return call_google(system_prompt, messages)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============== MAIN ROUTER FUNCTION ==============

def call_llm(task: LLMTask) -> LLMResult:
    """
    Route an LLM task to the appropriate provider(s).
    
    This is the ONLY function that should be called from other parts of Orb.
    
    Args:
        task: LLMTask containing job_type, messages, and optional context
    
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
    # Determine routing
    primary_provider, needs_critic = _determine_provider(task)
    
    # Build system prompt for primary provider
    if primary_provider == Provider.OPENAI:
        system_prompt = _build_gpt_system_prompt(task)
    elif primary_provider == Provider.ANTHROPIC:
        system_prompt = _build_claude_system_prompt(task)
    else:
        system_prompt = _build_gemini_system_prompt(task, is_critic=False)
    
    # Call primary provider
    primary_content, primary_usage = _call_provider(
        primary_provider,
        system_prompt,
        task.messages,
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