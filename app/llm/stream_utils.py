# FILE: app/llm/stream_utils.py
# Purpose: Stream utilities - helper functions for stream_router.py
# Called-by: app.jobs._engine_helpers, app.jobs.engine, app.llm.high_stakes_stream, app.llm.routing.chat_routing (+2 more)
# Depends-on: app.auth.middleware, app.embeddings, app.embeddings.service, app.llm.schemas (+3 more)
# Last-renovated: 2026-07-02
"""
Stream utilities - helper functions for stream_router.py

v3.3 (2026-01): Fixed DEFAULT_MODELS to use runtime env lookup instead of import-time frozen values.
v3.4 (2026-01): Prevent scan/map ASTRA commands from being misclassified as ARCHITECTURE_DESIGN in fallback routing.
v3.5 (2026-07-02, LANE D): _HARDCODED_FALLBACKS deleted — every provider key
    resolves env-only through app.llm.frontier_models; OPENAI_MODEL_CODE and
    JOB_CONTINUATION_* seeded in .env preserve the old effective values.
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
# LANE D (2026-07-02): zero literal model IDs. Each provider key resolves from
# .env on access — its dedicated env var first (historical precedence kept),
# then the frontier_models {PROVIDER}_DEFAULT_MODEL -> DEFAULT_MODEL chain.
# "anthropic_opus" additionally honours the JOB_CONTINUATION role vars so
# chat_routing's job-continuation path keeps its own .env knob.

from app.llm.frontier_models import get_provider_default_model, get_role_model

# provider key -> (dedicated env var, provider for the default-model chain)
_PROVIDER_ENV = {
    "openai": ("OPENAI_DEFAULT_MODEL", "openai"),
    "openai_code": ("OPENAI_MODEL_CODE", "openai"),
    "anthropic": ("ANTHROPIC_SONNET_MODEL", "anthropic"),
    "anthropic_opus": ("ANTHROPIC_OPUS_MODEL", "anthropic"),
    "gemini": ("GEMINI_DEFAULT_MODEL", "gemini"),
}


def get_default_model(provider_key: str) -> str:
    """Get default model for a provider key at RUNTIME (reads env on each call).

    This ensures env var changes take effect without restarting the server.
    Env-only: a completely unconfigured .env yields "" (with an error log from
    the resolver) and fails loudly downstream — never a rotted literal.
    """
    env_var, provider = _PROVIDER_ENV.get(provider_key, (None, provider_key))
    if env_var:
        val = os.getenv(env_var, "").strip()
        if val:
            return val
    if provider_key == "anthropic_opus":
        try:
            return get_role_model("JOB_CONTINUATION", "ARCHITECT")[1]
        except RuntimeError:
            pass
    return get_provider_default_model(provider, strict=False)


def get_spec_gate_model() -> str:
    """Get SpecGate model at RUNTIME with proper precedence.

    Precedence:
    1. OPENAI_SPEC_GATE_MODEL (if set)
    2. OPENAI_DEFAULT_MODEL, then the provider default chain (env-only)
    """
    spec_gate_model = os.getenv("OPENAI_SPEC_GATE_MODEL")
    if spec_gate_model:
        return spec_gate_model
    return get_default_model("openai")


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
        # Nothing resolved from env and the caller offered a default: use it.
        if not result and default is not None:
            return default
        return result


DEFAULT_MODELS = _DefaultModelsProxy()


# =============================================================================
# Utility Functions
# =============================================================================

def chunk_text(s: str, chunk_size: int = 120) -> List[str]:
    if not s:
        return []
    return [s[i: i + chunk_size] for i in range(0, len(s), chunk_size)]


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


def build_document_context(db: Session, project_id: int, query: str = "") -> str:
    """Build document context from uploaded files.
    
    v0.17: Most recent doc gets full content (up to 50KB).
    Older docs get summary + 1KB preview.
    """
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
        for i, doc in enumerate(recent_docs):
            summary = doc.summary[:500] if doc.summary else ""
            raw_text = doc.raw_text or ""
            
            if i == 0 and len(raw_text) <= 50 * 1024:
                # Most recent doc: include full content (up to 50KB)
                context_parts.append(
                    f"=== LATEST UPLOAD: {doc.filename} ===\n"
                    f"Summary: {summary}\n\n"
                    f"--- FULL CONTENT ---\n{raw_text}"
                )
            elif i == 0:
                # Most recent but too large: first 40KB + last 10KB
                context_parts.append(
                    f"=== LATEST UPLOAD: {doc.filename} (TRUNCATED) ===\n"
                    f"Summary: {summary}\n\n"
                    f"--- CONTENT (first 40KB + last 10KB) ---\n"
                    f"{raw_text[:40*1024]}\n\n... [TRUNCATED] ...\n\n{raw_text[-10*1024:]}"
                )
            else:
                # Older docs: summary + preview
                raw_preview = raw_text[:1000]
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
    # Respect explicit requested type unless casual_chat
    if requested_type and requested_type != "casual_chat":
        try:
            return JobType(requested_type)
        except ValueError:
            pass

    # Normalize message safely
    msg = (message or "").strip()
    msg_lower = msg.lower()

    # -------------------------------------------------------------------------
    # v3.4: HARD GUARD — prevent scan/map ASTRA commands from being classified
    #       as ARCHITECTURE_DESIGN in fallback routing.
    #
    # Why: If translation-layer command execution fails for any reason and the
    # router falls through into normal routing, "architecture" keywords would
    # previously escalate into ARCHITECTURE_DESIGN -> high-stakes -> SpecGate.
    # These specific commands must stay non-governed.
    # -------------------------------------------------------------------------
    if msg_lower.startswith("astra, command:"):
        cmd_text = msg_lower.split("astra, command:", 1)[1].strip()

        # Scan-only (memory only, no out folder, no governance)
        if cmd_text.startswith("update architecture") or cmd_text.startswith("scan system"):
            print("[stream_utils] ASTRA scan command detected; forcing CASUAL_CHAT classification (no high-stakes).")
            return JobType.CASUAL_CHAT

        # Map builder commands (scanner + optional Opus map formatting, but NO governance)
        # Note: We are ONLY preventing high-stakes escalation here. Actual routing to the
        # correct local handler still happens in stream_router.py when translation resolves.
        if cmd_text.startswith("create architecture map") or cmd_text.startswith("create architecture mapping"):
            print("[stream_utils] ASTRA arch-map command detected; forcing CASUAL_CHAT classification (no high-stakes).")
            return JobType.CASUAL_CHAT

    print(f"[stream_utils] Classifying message (first 200 chars): {repr(msg[:200])}")

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
        # v2.3: HTML/CSS/web creation + general creation
        "html", "css", "web page", "webpage", "website", "landing page",
        "create me a", "build me a", "make me a",
        "web app", "webapp", "component", "scaffold",
    ]

    language_keywords = [
        "python", "javascript", "typescript", "java", "c++", "rust", "react", "vue",
        "fastapi", "django", "kotlin", "swift", "angular",
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


# v2.3: Code creation tasks routed to GPT-5.4 (not Claude)
# v2.3: ALL code tasks route to GPT-5.4 (Claude reserved for architecture only)
_CODE_CREATION_JOBS = {
    JobType.CODE_MEDIUM,
    JobType.SIMPLE_CODE_CHANGE,
    JobType.SMALL_CODE,
    JobType.SMALL_BUGFIX,
    JobType.BUG_FIX,
    JobType.COMPLEX_CODE_CHANGE,
    JobType.CODEGEN_FULL_FILE,
    JobType.CODE_REVIEW,
    JobType.REFACTOR,
    JobType.COMPLEX_CODE,
    JobType.BUG_ANALYSIS,
    JobType.REFACTORING,
}


def select_provider_for_job_type(job_type: JobType) -> Tuple[str, str]:
    if job_type in RoutingConfig.GPT_ONLY_JOBS:
        return ("openai", get_default_model("openai"))

    # v2.3: Code creation tasks -> the OPENAI_MODEL_CODE slot (seeded in .env)
    if job_type in _CODE_CREATION_JOBS:
        code_model = get_default_model("openai_code")
        print(f"[stream_utils] Code task '{job_type.value}' -> openai/{code_model}")
        return ("openai", code_model)

    if job_type in RoutingConfig.HIGH_STAKES_JOBS:
        print(f"[stream_utils] High-stakes job '{job_type.value}' -> Opus")
        return ("anthropic", get_default_model("anthropic_opus"))

    if job_type in RoutingConfig.CLAUDE_PRIMARY_JOBS:
        return ("anthropic", get_default_model("anthropic"))

    if job_type == JobType.DEEP_RESEARCH:
        return ("gemini", get_default_model("gemini"))

    provider_key = os.getenv("ORB_DEFAULT_PROVIDER", "anthropic")
    return (provider_key, get_default_model(provider_key))
