# FILE: app/llm/stream_utils.py
"""
Stream utilities - helper functions for stream_router.py

v3.3 (2026-01): Fixed DEFAULT_MODELS to use runtime env lookup instead of import-time frozen values.
"""

import os
import re
import uuid
from typing import Optional, List, Tuple

from sqlalchemy.orm import Session

from app.auth.middleware import AuthResult
from app.memory import service as memory_service
from app.llm.schemas import JobType, RoutingConfig


# =============================================================================
# Model Configuration - RUNTIME LOOKUP (not frozen at import)
# =============================================================================

# Hard-coded fallbacks ONLY used if env vars are completely unset
_HARDCODED_FALLBACKS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5-20250929",
    "anthropic_opus": "claude-opus-4-5-20251101",
    "gemini": "gemini-2.0-flash",
}


def get_default_model(provider_key: str) -> str:
    """Get default model for a provider at RUNTIME (reads env on each call).
    
    This ensures env var changes take effect without restarting the server.
    """
    env_map = {
        "openai": "OPENAI_DEFAULT_MODEL",
        "anthropic": "ANTHROPIC_SONNET_MODEL",
        "anthropic_opus": "ANTHROPIC_OPUS_MODEL",
        "gemini": "GEMINI_DEFAULT_MODEL",
    }
    env_var = env_map.get(provider_key)
    if env_var:
        val = os.getenv(env_var)
        if val:
            return val
    return _HARDCODED_FALLBACKS.get(provider_key, _HARDCODED_FALLBACKS["openai"])


def get_spec_gate_model() -> str:
    """Get SpecGate model at RUNTIME with proper precedence.
    
    Precedence:
    1. OPENAI_SPEC_GATE_MODEL (if set)
    2. OPENAI_DEFAULT_MODEL (if set)
    3. Hard-coded fallback
    """
    spec_gate_model = os.getenv("OPENAI_SPEC_GATE_MODEL")
    if spec_gate_model:
        return spec_gate_model
    default_model = os.getenv("OPENAI_DEFAULT_MODEL")
    if default_model:
        return default_model
    return _HARDCODED_FALLBACKS["openai"]


def get_spec_gate_provider() -> str:
    """Get SpecGate provider at RUNTIME."""
    return os.getenv("SPEC_GATE_PROVIDER", "openai")


# DEPRECATED: Kept for backwards compatibility but now delegates to runtime lookup
# New code should use get_default_model() directly
class _DefaultModelsProxy:
    """Proxy that looks up env vars at access time, not import time."""
    
    def __getitem__(self, key: str) -> str:
        return get_default_model(key)
    
    def get(self, key: str, default: Optional[str] = None) -> str:
        result = get_default_model(key)
        # If we got the hardcoded fallback and a different default was requested, use that
        if result == _HARDCODED_FALLBACKS.get(key) and default is not None:
            return default
        return result


DEFAULT_MODELS = _DefaultModelsProxy()


# =============================================================================
# Utility Functions
# =============================================================================

def chunk_text(s: str, chunk_size: int = 120) -> List[str]:
    if not s:
        return []
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def cap_text(label: str, text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n<<truncated {label}: {len(text)} chars total>>\n"


def parse_reasoning_tags(raw: str) -> Tuple[str, str]:
    """Extract answer and reasoning from tagged content."""
    thinking_match = re.search(r"<THINKING>([\s\S]*?)</THINKING>", raw, re.IGNORECASE)
    answer_match = re.search(r"<ANSWER>([\s\S]*?)</ANSWER>", raw, re.IGNORECASE)

    if thinking_match and answer_match:
        reasoning = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return answer, reasoning

    cleaned = re.sub(r"</?THINKING[^>]*>", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?ANSWER[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    return cleaned if cleaned else raw, ""


def make_session_id(auth: AuthResult) -> str:
    for attr in ("session_id", "sid", "session", "session_token"):
        try:
            v = getattr(auth, attr, None)
            if v:
                return str(v)
        except Exception:
            pass
    try:
        user = getattr(auth, "user", None)
        if isinstance(user, dict):
            return str(user.get("id") or user.get("email") or user.get("username") or "")
    except Exception:
        pass
    return f"legacy-{uuid.uuid4()}"


def coerce_int(v) -> int:
    try:
        if v is None:
            return 0
        return int(v)
    except Exception:
        return 0


def extract_usage_tokens(usage_obj) -> Tuple[int, int]:
    if usage_obj is None:
        return (0, 0)
    if isinstance(usage_obj, dict):
        pt = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or usage_obj.get("prompt")
        ct = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or usage_obj.get("completion")
        return (coerce_int(pt), coerce_int(ct))
    pt = getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", None)
    ct = getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None)
    return (coerce_int(pt), coerce_int(ct))


def build_context_block(db: Session, project_id: int) -> str:
    sections = []
    notes = memory_service.list_notes(db, project_id)[:10]
    if notes:
        notes_text = "\n".join(f"- [{n.id}] {n.title}: {n.content[:200]}..." for n in notes)
        sections.append(f"PROJECT NOTES:\n{notes_text}")

    tasks = memory_service.list_tasks(db, project_id, status="pending")[:10]
    if tasks:
        tasks_text = "\n".join(f"- {t.title}" for t in tasks)
        sections.append(f"PENDING TASKS:\n{tasks_text}")

    return "\n\n".join(sections) if sections else ""


def build_document_context(db: Session, project_id: int) -> str:
    try:
        from app.memory.models import DocumentContent

        recent_docs = (
            db.query(DocumentContent)
            .filter(DocumentContent.project_id == project_id)
            .order_by(DocumentContent.created_at.desc())
            .limit(5)
            .all()
        )

        if not recent_docs:
            return ""

        context_parts = []
        for doc in recent_docs:
            summary = doc.summary[:500] if doc.summary else ""
            raw_preview = doc.raw_text[:1000] if doc.raw_text else ""
            if summary or raw_preview:
                context_parts.append(f"[{doc.filename}]:\nSummary: {summary}\nContent: {raw_preview}...")

        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"[stream_utils] Error building document context: {e}")
        return ""


def get_semantic_context(db: Session, project_id: int, query: str) -> str:
    try:
        from app.embeddings import service as embeddings_service

        results = embeddings_service.search(db=db, project_id=project_id, query=query, top_k=5)
        if not results:
            return ""

        context_parts = ["=== RELEVANT CONTEXT (semantic search) ==="]
        for result in results:
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            context_parts.append(f"\n[Score: {result.similarity:.3f}] {content_preview}")

        return "\n".join(context_parts)
    except Exception as e:
        print(f"[stream_utils] Semantic search failed: {e}")
        return ""


def classify_job_type(message: str, requested_type: str) -> JobType:
    if requested_type and requested_type != "casual_chat":
        try:
            return JobType(requested_type)
        except ValueError:
            pass

    msg_lower = message.lower()
    print(f"[stream_utils] Classifying message (first 200 chars): {repr(message[:200])}")

    security_keywords = [
        "security review", "security audit", "security assessment", "penetration test",
        "pentest", "threat model", "threat modeling", "vulnerability", "vulnerabilities",
        "vulnerability assessment", "exploit", "attack vector", "attack surface",
        "sql injection", "xss", "csrf", "authentication bypass", "privilege escalation",
        "session fixation", "session hijacking", "security analysis", "security check",
        "encryption review", "key management", "secrets management", "authentication security",
        "authorization security", "security hardening", "security posture",
    ]

    arch_keywords = [
        "architect", "architecture", "design a system", "system design", "microservice",
        "micro-service", "infrastructure", "infra", "scalab", "database schema", "db schema",
        "api design", "high-level design", "hld", "distributed system", "design pattern",
        "tech stack", "critical architecture",
    ]

    review_keywords = [
        "review this", "review my", "code review", "check this code", "find bugs",
        "audit this", "critique", "what's wrong with",
    ]

    code_keywords = [
        "write a function", "write code", "implement", "debug", "fix this code",
        "refactor", "def ", "function ", "```",
    ]

    language_keywords = [
        "python", "javascript", "typescript", "java", "c++", "rust", "react", "vue",
        "fastapi", "django",
    ]

    if any(kw in msg_lower for kw in security_keywords):
        print("[stream_utils] Classified: SECURITY_REVIEW")
        return JobType.SECURITY_REVIEW

    if any(kw in msg_lower for kw in arch_keywords):
        print("[stream_utils] Classified: ARCHITECTURE_DESIGN")
        return JobType.ARCHITECTURE_DESIGN

    if any(kw in msg_lower for kw in review_keywords):
        print("[stream_utils] Classified: CODE_REVIEW")
        return JobType.CODE_REVIEW

    is_code_related = any(kw in msg_lower for kw in code_keywords) or any(kw in msg_lower for kw in language_keywords)
    if is_code_related:
        complex_indicators = ["complex", "full file", "entire file", "production"]
        if any(x in msg_lower for x in complex_indicators):
            print("[stream_utils] Classified: COMPLEX_CODE_CHANGE")
            return JobType.COMPLEX_CODE_CHANGE
        print("[stream_utils] Classified: SIMPLE_CODE_CHANGE")
        return JobType.SIMPLE_CODE_CHANGE

    print("[stream_utils] Classified: CASUAL_CHAT (default)")
    return JobType.CASUAL_CHAT


def select_provider_for_job_type(job_type: JobType) -> Tuple[str, str]:
    if job_type in RoutingConfig.GPT_ONLY_JOBS:
        return ("openai", get_default_model("openai"))

    if job_type in RoutingConfig.HIGH_STAKES_JOBS:
        print(f"[stream_utils] High-stakes job '{job_type.value}' → Opus")
        return ("anthropic", get_default_model("anthropic_opus"))

    if job_type in RoutingConfig.CLAUDE_PRIMARY_JOBS:
        return ("anthropic", get_default_model("anthropic"))

    if job_type == JobType.DEEP_RESEARCH:
        return ("gemini", get_default_model("gemini"))

    provider_key = os.getenv("ORB_DEFAULT_PROVIDER", "anthropic")
    return (provider_key, get_default_model(provider_key))