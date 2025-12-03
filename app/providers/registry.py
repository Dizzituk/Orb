# FILE: app/providers/registry.py
"""
Phase 4 Provider Registry - Single Source of Truth for LLM Calls

This is the ONLY module that:
- Decrypts and manages API keys
- Constructs HTTP clients
- Makes outbound LLM API calls
- Handles timeouts, retries, rate limiting
- Logs usage with job/session/project tracking

ALL LLM calls (from /chat, /stream/chat, /jobs) MUST go through this registry.

PHASE 4 FIXES:
- Fixed memory leak: _usage_log now uses collections.deque with max size
- Implemented retry logic with exponential backoff
- Added configurable retry settings per provider

NOTE: Streaming is NOT yet unified through this registry. Streaming endpoints
continue using their existing path until streaming is properly implemented.
"""
from __future__ import annotations

import os
import time
import asyncio
import logging
from collections import deque
from typing import Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.crypto import decrypt_string, is_encryption_ready

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum number of usage records to keep in memory
# Prevents unbounded memory growth over time
USAGE_LOG_MAX_SIZE = int(os.getenv("ORB_USAGE_LOG_MAX_SIZE", "5000"))

# Default retry settings
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 30.0  # seconds


# =============================================================================
# RESULT TYPES
# =============================================================================

class LlmCallStatus(str, Enum):
    """Status of an LLM call."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class UsageMetrics:
    """
    Per-call token usage and cost tracking.
    
    NOTE: This is low-level per-call metrics. For job-level aggregated
    usage, see app.jobs.schemas.UsageMetrics (Pydantic model).
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0


@dataclass
class LlmCallResult:
    """
    Result from an LLM API call.
    
    Contains:
    - Status (success/error/timeout)
    - Content (response text)
    - Usage metrics (tokens, cost)
    - Metadata (model, provider, timing)
    - Error information (if failed)
    """
    status: LlmCallStatus
    content: str
    usage: UsageMetrics
    
    # Identifiers
    provider_id: str
    model_id: str
    job_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[int] = None
    
    # Timing
    started_at: datetime = None
    completed_at: datetime = None
    duration_seconds: float = 0.0
    
    # Error tracking
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Raw response for debugging
    raw_response: Optional[Any] = None
    
    def is_success(self) -> bool:
        """Check if call succeeded."""
        return self.status == LlmCallStatus.SUCCESS
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "content": self.content,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "cost_estimate": self.usage.cost_estimate,
            },
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "duration_seconds": self.duration_seconds,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    provider_id: str
    display_name: str
    api_key_env_var: str
    timeout_seconds: int = 120
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY
    rate_limit_per_minute: Optional[int] = None
    
    # Errors that should trigger retry
    retryable_errors: tuple = field(default_factory=lambda: (
        "rate_limit",
        "429",
        "503",
        "502",
        "500",
        "timeout",
        "connection",
        "overloaded",
    ))


# Provider configurations
PROVIDERS = {
    "openai": ProviderConfig(
        provider_id="openai",
        display_name="OpenAI (GPT)",
        api_key_env_var="OPENAI_API_KEY",
        timeout_seconds=120,
        max_retries=2,
        rate_limit_per_minute=60,
    ),
    "anthropic": ProviderConfig(
        provider_id="anthropic",
        display_name="Anthropic (Claude)",
        api_key_env_var="ANTHROPIC_API_KEY",
        timeout_seconds=180,
        max_retries=2,
        rate_limit_per_minute=50,
    ),
    "google": ProviderConfig(
        provider_id="google",
        display_name="Google (Gemini)",
        api_key_env_var="GOOGLE_API_KEY",
        timeout_seconds=180,
        max_retries=2,
        rate_limit_per_minute=60,
    ),
}


# =============================================================================
# PROVIDER REGISTRY - SINGLE SOURCE OF TRUTH
# =============================================================================

class ProviderRegistry:
    """
    The ONLY class that makes outbound LLM API calls.
    
    Responsibilities:
    - Load and decrypt API keys from environment
    - Construct HTTP clients for each provider
    - Execute LLM calls with timeout/retry/rate limit enforcement
    - Log all usage with job/session/project tracking
    - Map errors to Phase 4 error taxonomy
    
    ALL code in Orb (existing /chat, /stream/chat, new /jobs) must
    call through this registry. No exceptions.
    """
    
    def __init__(self):
        self._configs = PROVIDERS.copy()
        self._encrypted_keys: dict[str, str] = {}
        # FIXED: Use deque with maxlen to prevent memory leak
        self._usage_log: deque[LlmCallResult] = deque(maxlen=USAGE_LOG_MAX_SIZE)
        self._rate_limit_trackers: dict[str, list[float]] = {}
        
        # Load and encrypt API keys
        self._load_keys_from_env()
    
    def _load_keys_from_env(self) -> None:
        """
        Load API keys from environment and encrypt for in-memory storage.
        
        This is the ONLY place in Phase 4 where raw API keys are read.
        """
        if not is_encryption_ready():
            logger.warning("[registry] Encryption not ready, keys stored unencrypted in memory")
        
        for provider_id, config in self._configs.items():
            key = os.getenv(config.api_key_env_var)
            if key:
                # Encrypt for in-memory storage
                from app.crypto import encrypt_string
                encrypted = encrypt_string(key) if is_encryption_ready() else key
                self._encrypted_keys[provider_id] = encrypted
                logger.info(f"[registry] Loaded API key for {provider_id}")
            else:
                logger.warning(f"[registry] No API key for {provider_id} (env: {config.api_key_env_var})")
    
    def is_provider_available(self, provider_id: str) -> bool:
        """Check if provider is available (has API key)."""
        return provider_id in self._encrypted_keys
    
    def get_provider_config(self, provider_id: str) -> ProviderConfig:
        """Get configuration for a provider."""
        if provider_id not in self._configs:
            raise ValueError(f"Unknown provider: {provider_id}")
        return self._configs[provider_id]
    
    def _get_decrypted_key(self, provider_id: str) -> str:
        """
        Get decrypted API key.
        
        SECURITY: This is the ONLY place where raw keys are exposed.
        Keys are immediately used for HTTP client construction and
        never stored or logged in plaintext.
        """
        if provider_id not in self._encrypted_keys:
            raise ValueError(f"No API key for provider: {provider_id}")
        
        encrypted = self._encrypted_keys[provider_id]
        return decrypt_string(encrypted)
    
    def _check_rate_limit(self, provider_id: str) -> bool:
        """Check if provider is within rate limit."""
        config = self.get_provider_config(provider_id)
        if config.rate_limit_per_minute is None:
            return True
        
        now = time.time()
        window_start = now - 60.0
        
        if provider_id not in self._rate_limit_trackers:
            self._rate_limit_trackers[provider_id] = []
        
        tracker = self._rate_limit_trackers[provider_id]
        tracker[:] = [ts for ts in tracker if ts > window_start]
        
        if len(tracker) >= config.rate_limit_per_minute:
            return False
        
        tracker.append(now)
        return True
    
    def _is_retryable_error(self, error: Exception, config: ProviderConfig) -> bool:
        """Check if an error should trigger a retry."""
        error_str = str(error).lower()
        return any(keyword in error_str for keyword in config.retryable_errors)
    
    def _calculate_retry_delay(self, attempt: int, config: ProviderConfig) -> float:
        """Calculate delay before next retry using exponential backoff."""
        delay = config.retry_base_delay * (2 ** attempt)
        return min(delay, config.retry_max_delay)
    
    async def llm_call(
        self,
        provider_id: str,
        model_id: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        job_envelope: Optional[Any] = None,  # JobEnvelope type (avoid circular import)
        temperature: float = 0.7,
        max_tokens: int = 100000,
        timeout_seconds: Optional[int] = None,
        attachments: Optional[list[dict]] = None,
        enable_web_search: bool = False,
        stream: bool = False,
    ) -> LlmCallResult:
        """
        Make an LLM API call with retry logic.
        
        This is the SINGLE ENTRY POINT for all LLM calls in Orb.
        
        Both /chat (synthesized envelope) and /jobs (real envelope) paths
        call this function.
        
        Args:
            provider_id: "openai" | "anthropic" | "google"
            model_id: Specific model string
            messages: List of message dicts
            system_prompt: Optional system message
            job_envelope: JobEnvelope (for tracking) or None (for legacy /chat)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens (TODO: should come from envelope.budget)
            timeout_seconds: Override default timeout (TODO: should come from envelope.budget)
            attachments: For vision tasks
            enable_web_search: Enable Gemini grounding
            stream: Whether to stream response (NOT YET IMPLEMENTED)
        
        Returns:
            LlmCallResult with status, content, usage, and error info.
            
            IMPORTANT: This function returns an LlmCallResult with error status
            instead of raising exceptions for provider/timeout errors. Check
            result.is_success() before using result.content.
        
        Raises:
            ValueError: Only for invalid arguments (e.g., unknown provider)
            
        NOTE: Streaming is NOT yet unified through this registry. When stream=True,
        this function raises NotImplementedError. Streaming endpoints continue using
        their existing path until streaming is properly implemented in a later branch.
        """
        config = self.get_provider_config(provider_id)
        
        # Extract tracking IDs from envelope
        job_id = job_envelope.job_id if job_envelope else None
        session_id = job_envelope.session_id if job_envelope else None
        project_id = job_envelope.project_id if job_envelope else None
        
        # Streaming not yet unified
        if stream:
            raise NotImplementedError(
                "Streaming is not yet unified through provider registry. "
                "Streaming endpoints must continue using their existing path for now."
            )
        
        # Check rate limit
        if not self._check_rate_limit(provider_id):
            logger.warning(f"[registry] Rate limit exceeded: {provider_id}")
            result = LlmCallResult(
                status=LlmCallStatus.RATE_LIMITED,
                content="",
                usage=UsageMetrics(),
                provider_id=provider_id,
                model_id=model_id,
                job_id=job_id,
                session_id=session_id,
                project_id=project_id,
                error_code="RATE_LIMITED",
                error_message=f"Rate limit exceeded for {provider_id}",
            )
            self._usage_log.append(result)
            return result
        
        # Use configured or override timeout
        timeout = timeout_seconds or config.timeout_seconds
        
        started_at = datetime.utcnow()
        last_error: Optional[Exception] = None
        retry_count = 0
        
        # FIXED: Implement retry logic with exponential backoff
        for attempt in range(config.max_retries + 1):
            try:
                # Call provider-specific implementation
                content, usage_dict = await self._call_provider_sync(
                    provider_id=provider_id,
                    model_id=model_id,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    attachments=attachments,
                    enable_web_search=enable_web_search,
                )
                
                completed_at = datetime.utcnow()
                duration = (completed_at - started_at).total_seconds()
                
                # Parse usage metrics
                usage = UsageMetrics(
                    prompt_tokens=usage_dict.get("prompt_tokens", 0) if usage_dict else 0,
                    completion_tokens=usage_dict.get("completion_tokens", 0) if usage_dict else 0,
                    total_tokens=usage_dict.get("total_tokens", 0) if usage_dict else 0,
                    cost_estimate=usage_dict.get("cost_estimate", 0.0) if usage_dict else 0.0,
                )
                
                result = LlmCallResult(
                    status=LlmCallStatus.SUCCESS,
                    content=content,
                    usage=usage,
                    provider_id=provider_id,
                    model_id=model_id,
                    job_id=job_id,
                    session_id=session_id,
                    project_id=project_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    retry_count=retry_count,
                )
                
                # Log successful call
                self._usage_log.append(result)
                logger.info(
                    f"[registry] SUCCESS: {provider_id}/{model_id} "
                    f"job={job_id} tokens={usage.total_tokens} "
                    f"cost=${usage.cost_estimate:.4f} duration={duration:.2f}s"
                    f"{f' (after {retry_count} retries)' if retry_count > 0 else ''}"
                )
                
                return result
            
            except asyncio.TimeoutError as e:
                last_error = e
                retry_count = attempt
                
                # Timeout is not retryable by default
                logger.warning(
                    f"[registry] TIMEOUT on attempt {attempt + 1}/{config.max_retries + 1}: "
                    f"{provider_id}/{model_id} job={job_id}"
                )
                break  # Don't retry timeouts
            
            except Exception as e:
                last_error = e
                retry_count = attempt
                
                # Check if error is retryable
                if attempt < config.max_retries and self._is_retryable_error(e, config):
                    delay = self._calculate_retry_delay(attempt, config)
                    logger.warning(
                        f"[registry] Retryable error on attempt {attempt + 1}/{config.max_retries + 1}: "
                        f"{provider_id}/{model_id} job={job_id} error={str(e)[:100]} "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Not retryable or max retries reached
                    logger.error(
                        f"[registry] ERROR on attempt {attempt + 1}/{config.max_retries + 1}: "
                        f"{provider_id}/{model_id} job={job_id} error={str(e)[:200]}"
                    )
                    break
        
        # All retries exhausted or non-retryable error
        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()
        
        if isinstance(last_error, asyncio.TimeoutError):
            result = LlmCallResult(
                status=LlmCallStatus.TIMEOUT,
                content="",
                usage=UsageMetrics(),
                provider_id=provider_id,
                model_id=model_id,
                job_id=job_id,
                session_id=session_id,
                project_id=project_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                error_code="TIMEOUT",
                error_message=f"Call exceeded timeout of {timeout}s",
                retry_count=retry_count,
            )
        else:
            # Map exception to error code
            error_code = self._map_error_to_code(last_error)
            
            result = LlmCallResult(
                status=LlmCallStatus.ERROR,
                content="",
                usage=UsageMetrics(),
                provider_id=provider_id,
                model_id=model_id,
                job_id=job_id,
                session_id=session_id,
                project_id=project_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                error_code=error_code,
                error_message=str(last_error),
                retry_count=retry_count,
            )
        
        self._usage_log.append(result)
        logger.error(
            f"[registry] FAILED after {retry_count + 1} attempts: {provider_id}/{model_id} "
            f"job={job_id} error={result.error_code}"
        )
        
        return result
    
    async def _call_provider_sync(
        self,
        provider_id: str,
        model_id: str,
        system_prompt: Optional[str],
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout: int,
        attachments: Optional[list[dict]],
        enable_web_search: bool,
    ) -> tuple[str, Optional[dict]]:
        """
        Call provider API synchronously.
        
        This method owns HTTP client construction and API interaction.
        No other code should make raw API calls.
        """
        api_key = self._get_decrypted_key(provider_id)
        
        if provider_id == "openai":
            return await self._call_openai(
                api_key=api_key,
                model_id=model_id,
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider_id == "anthropic":
            return await self._call_anthropic(
                api_key=api_key,
                model_id=model_id,
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider_id == "google":
            return await self._call_google(
                api_key=api_key,
                model_id=model_id,
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                attachments=attachments,
                enable_web_search=enable_web_search,
            )
        else:
            raise ValueError(f"Unknown provider: {provider_id}")
    
    async def _call_openai(
        self,
        api_key: str,
        model_id: str,
        system_prompt: Optional[str],
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> tuple[str, dict]:
        """
        Call OpenAI API.
        
        TODO: max_tokens and timeout should come from JobEnvelope.budget
        when available (envelope.budget.max_tokens, envelope.budget.max_wall_time_seconds).
        """
        import openai
        
        client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)
        
        # Build messages with system prompt
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)
        
        response = await client.chat.completions.create(
            model=model_id,
            messages=api_messages,
            temperature=temperature,
            max_tokens=min(max_tokens, 16384),  # OpenAI limit
        )
        
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost_estimate": self._estimate_cost("openai", model_id, response.usage),
        }
        
        return content, usage
    
    async def _call_anthropic(
        self,
        api_key: str,
        model_id: str,
        system_prompt: Optional[str],
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> tuple[str, dict]:
        """
        Call Anthropic API.
        
        TODO: max_tokens and timeout should come from JobEnvelope.budget
        when available (envelope.budget.max_tokens, envelope.budget.max_wall_time_seconds).
        """
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        
        response = await client.messages.create(
            model=model_id,
            system=system_prompt or "",
            messages=messages,
            temperature=temperature,
            max_tokens=min(max_tokens, 8192),  # Anthropic default limit
        )
        
        content = response.content[0].text
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            "cost_estimate": self._estimate_cost("anthropic", model_id, response.usage),
        }
        
        return content, usage
    
    async def _call_google(
        self,
        api_key: str,
        model_id: str,
        system_prompt: Optional[str],
        messages: list[dict],
        temperature: float,
        timeout: int,
        attachments: Optional[list[dict]],
        enable_web_search: bool,
    ) -> tuple[str, dict]:
        """
        Call Google Gemini API.
        
        TODO: timeout should come from JobEnvelope.budget.max_wall_time_seconds
        when available.
        """
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        
        # Build prompt
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
        
        # Generate (sync call wrapped in async - Gemini SDK doesn't have proper async yet)
        import asyncio
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                ),
            )
        )
        
        content = response.text
        usage = {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
            "cost_estimate": 0.0,  # TODO: Add Gemini cost estimation
        }
        
        return content, usage
    
    def _estimate_cost(self, provider_id: str, model_id: str, usage: Any) -> float:
        """
        Estimate API call cost.
        
        NOTE: These are approximate costs and may not reflect current pricing.
        Consider moving to a config file for easier updates.
        """
        # Updated cost estimation - prices as of late 2024
        if provider_id == "openai":
            if "gpt-4o" in model_id:
                # GPT-4o pricing: $2.50/1M input, $10/1M output
                return (usage.prompt_tokens * 0.0025 + usage.completion_tokens * 0.01) / 1000
            elif "gpt-4" in model_id:
                # GPT-4 Turbo pricing: $10/1M input, $30/1M output
                return (usage.prompt_tokens * 0.01 + usage.completion_tokens * 0.03) / 1000
            else:
                # GPT-3.5 Turbo pricing: $0.50/1M input, $1.50/1M output
                return (usage.prompt_tokens * 0.0005 + usage.completion_tokens * 0.0015) / 1000
        elif provider_id == "anthropic":
            if "opus" in model_id.lower():
                # Claude Opus: $15/1M input, $75/1M output
                return (usage.input_tokens * 0.015 + usage.output_tokens * 0.075) / 1000
            elif "sonnet" in model_id.lower():
                # Claude Sonnet: $3/1M input, $15/1M output
                return (usage.input_tokens * 0.003 + usage.output_tokens * 0.015) / 1000
            else:
                # Claude Haiku: $0.25/1M input, $1.25/1M output
                return (usage.input_tokens * 0.00025 + usage.output_tokens * 0.00125) / 1000
        else:
            return 0.0
    
    def _map_error_to_code(self, error: Exception) -> str:
        """Map exception to Phase 4 error taxonomy."""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            return "RATE_LIMITED"
        elif "timeout" in error_str:
            return "TIMEOUT"
        elif "authentication" in error_str or "401" in error_str:
            return "AUTHENTICATION_ERROR"
        elif "not found" in error_str or "404" in error_str:
            return "MODEL_NOT_FOUND"
        elif "invalid" in error_str:
            return "INVALID_REQUEST"
        elif "overloaded" in error_str or "503" in error_str:
            return "SERVICE_OVERLOADED"
        else:
            return "MODEL_ERROR"
    
    def get_usage_for_job(self, job_id: str) -> list[LlmCallResult]:
        """Get all LLM calls for a job."""
        return [r for r in self._usage_log if r.job_id == job_id]
    
    def get_usage_for_session(self, session_id: str) -> list[LlmCallResult]:
        """Get all LLM calls for a session."""
        return [r for r in self._usage_log if r.session_id == session_id]
    
    def get_total_cost(self, job_id: Optional[str] = None, 
                       session_id: Optional[str] = None) -> float:
        """Calculate total cost for scope."""
        usage = list(self._usage_log)
        
        if job_id:
            usage = [r for r in usage if r.job_id == job_id]
        elif session_id:
            usage = [r for r in usage if r.session_id == session_id]
        
        return sum(r.usage.cost_estimate for r in usage)
    
    def get_usage_log_size(self) -> int:
        """Get current size of usage log (for monitoring)."""
        return len(self._usage_log)
    
    def clear_usage_log(self) -> int:
        """
        Clear usage log and return number of entries cleared.
        
        Use sparingly - mainly for testing or manual cleanup.
        """
        count = len(self._usage_log)
        self._usage_log.clear()
        logger.info(f"[registry] Cleared {count} usage log entries")
        return count


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def llm_call(
    provider_id: str,
    model_id: str,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    job_envelope: Optional[Any] = None,
    **kwargs
) -> LlmCallResult:
    """
    Async convenience function for LLM calls.
    
    This is the function that ALL async Orb code should use for LLM calls.
    """
    return await get_provider_registry().llm_call(
        provider_id=provider_id,
        model_id=model_id,
        messages=messages,
        system_prompt=system_prompt,
        job_envelope=job_envelope,
        **kwargs
    )


def is_provider_available(provider_id: str) -> bool:
    """Check if provider is available."""
    return get_provider_registry().is_provider_available(provider_id)


__all__ = [
    "LlmCallStatus",
    "UsageMetrics",
    "LlmCallResult",
    "ProviderConfig",
    "ProviderRegistry",
    "get_provider_registry",
    "llm_call",
    "is_provider_available",
]