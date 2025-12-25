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
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request
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

# High-stakes critique configuration + prompts extracted to keep this router smaller.
from app.llm.pipeline.high_stakes import (
    HIGH_STAKES_JOB_TYPES,
    MIN_CRITIQUE_CHARS,
    GEMINI_CRITIC_MODEL,
    GEMINI_CRITIC_MAX_TOKENS,
    OPUS_REVISION_MAX_TOKENS,
    DEFAULT_MODELS,
    _LEGACY_TO_PHASE4_JOB_TYPE,
    _map_to_phase4_job_type,
    get_environment_context,
    normalize_job_type_for_high_stakes,
    is_high_stakes_job,
    is_opus_model,
    is_long_enough_for_critique,
    build_critique_prompt_for_architecture,
    build_critique_prompt_for_security,
    build_critique_prompt_for_general,
    build_critique_prompt,
)

# Routing helpers extracted to dedicated module for clarity.
from app.llm.routing.job_routing import (
    _default_importance_for_job_type,
    _default_modalities_for_job_type,
    inject_file_map_into_messages,
    classify_and_route,
)


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
# LOCAL ACTION: ZOBIE MAP (Prompt-triggered repo mapper; read-only)
# =============================================================================

_ZOBIE_MAP_TIMEOUT_SECS = int(os.getenv("ZOBIE_MAP_TIMEOUT_SECS", "30"))
_ZOBIE_MAP_DEFAULT_BASE = (os.getenv("ZOBIE_CONTROLLER_BASE") or os.getenv("ZOMBIE_CONTROLLER_BASE") or "http://192.168.250.2:8765").rstrip("/")
_ZOBIE_MAP_MAX_FILES_DEFAULT = int(os.getenv("ZOBIE_MAP_MAX_FILES", "200000"))

# Do NOT pull secrets (controller allows it, so client must refuse)
_ZOBIE_DENY_FILE_PATTERNS = [
    r"(^|/)\.env($|/)",
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
    r"\.p12$",
    r"secrets?",
    r"credentials?",
]

_ZOBIE_ANCHOR_BASENAMES = {
    "package.json", "main.js", "electron.js", "vite.config.js", "vite.config.ts",
    "requirements.txt", "pyproject.toml", "poetry.lock",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "README.md", "README.txt",
}

_ZOBIE_ANCHOR_PATH_HINTS = [
    "Orb-backend/main.py",
    "Orb-backend/router.py",
    "orb-desktop/package.json",
    "orb-desktop/main.js",
]

def _zobie_http_json(url: str) -> Any:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=_ZOBIE_MAP_TIMEOUT_SECS) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))

def _zobie_is_denied_path(p: str) -> bool:
    p2 = p.replace("\\", "/").lower()
    return any(re.search(pat, p2) for pat in _ZOBIE_DENY_FILE_PATTERNS)

def _zobie_pick_anchor_files(tree_paths: List[str]) -> List[str]:
    anchors = set()

    tree_set = set(tree_paths)
    for hp in _ZOBIE_ANCHOR_PATH_HINTS:
        if hp in tree_set and not _zobie_is_denied_path(hp):
            anchors.add(hp)

    for p in tree_paths:
        if _zobie_is_denied_path(p):
            continue
        base = p.split("/")[-1]
        if base in _ZOBIE_ANCHOR_BASENAMES:
            anchors.add(p)

    for p in tree_paths:
        if _zobie_is_denied_path(p):
            continue
        base = p.split("/")[-1].lower()
        if base in {"main.py", "app.py", "server.py", "router.py", "main.ts", "main.jsx", "main.tsx"}:
            anchors.add(p)

    return sorted(list(anchors))[:80]

def _zobie_condensed_tree(paths: List[str], max_depth: int = 3) -> List[str]:
    out = set()
    for p in paths:
        if _zobie_is_denied_path(p):
            continue
        parts = p.split("/")
        for d in range(1, min(max_depth, len(parts)) + 1):
            out.add("/".join(parts[:d]))
    return sorted(out)

def _zobie_extract_signals(content: str) -> List[str]:
    signals: List[str] = []

    for m in re.finditer(r"(?:port\s*[:=]\s*|--port\s+)(\d{2,5})", content, flags=re.IGNORECASE):
        signals.append(f"Port: {m.group(1)}")
    if "0.0.0.0" in content:
        signals.append("Binds to 0.0.0.0")
    if "localhost" in content:
        signals.append("References localhost")

    lc = content.lower()
    if "fastapi" in lc:
        signals.append("FastAPI detected")
    if "uvicorn" in lc:
        signals.append("Uvicorn detected")

    if "electron" in lc:
        signals.append("Electron detected")
    if "spawn(" in content or "child_process" in content:
        signals.append("Spawns subprocess (child_process/spawn)")

    if "keytar" in lc:
        signals.append("Uses keytar / Credential Manager")
    if "credential" in lc:
        signals.append("Credential handling present")

    uniq: List[str] = []
    for s in signals:
        if s not in uniq:
            uniq.append(s)
    return uniq[:12]

def _zobie_default_out_dir() -> str:
    env = os.getenv("ZOBIE_MAP_OUT_DIR") or os.getenv("ORB_ZOBIE_MAP_OUT_DIR")
    if env:
        return env

    # Preferred: outside the main repo
    if os.path.isdir(r"D:\zobie_mapper"):
        return r"D:\zobie_mapper\out"

    # Fallback: still outside repo (cwd)
    return os.path.join(os.getcwd(), "_zobie_maps")

def _zobie_parse_command(message: str) -> Optional[Dict[str, Any]]:
    """
    Supported prompts:
      - ZOBIE MAP
      - ZOMBIE MAP
      - /zobie map
      - ZOBIE MAP http://192.168.250.2:8765
      - ZOBIE MAP 192.168.250.2:8765
      - Optional params: max_files=200000 include_hashes=false
    """
    if not message:
        return None

    m = re.match(r"^\s*(?:/)?zob(?:ie|mbie)\s+map\b(.*)$", message, flags=re.IGNORECASE)
    if not m:
        return None

    rest = (m.group(1) or "").strip()
    base_url = _ZOBIE_MAP_DEFAULT_BASE
    max_files = _ZOBIE_MAP_MAX_FILES_DEFAULT
    include_hashes = False

    if rest:
        parts = rest.split()
        if parts and (
            parts[0].startswith("http://")
            or parts[0].startswith("https://")
            or re.match(r"^\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}$", parts[0])
        ):
            tok = parts.pop(0)
            base_url = tok if tok.startswith("http") else f"http://{tok}"
            base_url = base_url.rstrip("/")

        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip().lower()
            if k == "max_files":
                try:
                    max_files = int(v)
                except Exception:
                    pass
            elif k == "include_hashes":
                include_hashes = v in {"1", "true", "yes", "y"}

    return {"base_url": base_url, "max_files": max_files, "include_hashes": include_hashes}

def _zobie_run_map_sync(base_url: str, out_dir: str, max_files: int, include_hashes: bool) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    health = _zobie_http_json(f"{base_url}/health")
    repo_root = health.get("repo_root", "?")

    tree = _zobie_http_json(
        f"{base_url}/repo/tree?include_hashes={'true' if include_hashes else 'false'}&max_files={max_files}"
    )
    paths = [x.get("path") for x in tree if isinstance(x, dict) and x.get("path")]
    paths = [p for p in paths if isinstance(p, str)]

    safe_paths = [p for p in paths if not _zobie_is_denied_path(p)]

    top: Dict[str, int] = {}
    for p in safe_paths:
        head = p.split("/")[0]
        top[head] = top.get(head, 0) + 1

    anchors = _zobie_pick_anchor_files(safe_paths)

    file_summaries: List[Dict[str, Any]] = []
    for p in anchors:
        data = _zobie_http_json(f"{base_url}/repo/file?path={quote(p)}")
        content = data.get("content", "") or ""
        file_summaries.append(
            {
                "path": p,
                "bytes": data.get("bytes"),
                "signals": _zobie_extract_signals(content),
            }
        )

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tree_txt = out_path / f"REPO_TREE_{stamp}.txt"
    map_md = out_path / f"ARCH_MAP_{stamp}.md"
    index_json = out_path / f"INDEX_{stamp}.json"

    # Write condensed tree
    tree_lines: List[str] = []
    tree_lines.append(f"Repo root (VM): {repo_root}")
    tree_lines.append(f"Controller: {base_url}")
    tree_lines.append("")
    tree_lines.append("Top-level dirs/files (count of files under each):")
    for k in sorted(top.keys()):
        tree_lines.append(f"- {k}: {top[k]}")
    tree_lines.append("")
    tree_lines.append("Condensed tree (depth<=3):")
    tree_lines.extend(_zobie_condensed_tree(safe_paths, max_depth=3))
    tree_lines.append("")
    tree_txt.write_text("\n".join(tree_lines), encoding="utf-8")

    # Write small architecture map (Stage 1)
    md_lines: List[str] = []
    md_lines.append("# ZombieOrb Repo Map (read-only)")
    md_lines.append("")
    md_lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    md_lines.append(f"- Controller: `{base_url}`")
    md_lines.append(f"- VM repo root: `{repo_root}`")
    md_lines.append("")
    md_lines.append("## High-level layout")
    for k in sorted(top.keys()):
        md_lines.append(f"- **{k}** — {top[k]} files")
    md_lines.append("")
    md_lines.append("## Anchor files scanned")
    for s in file_summaries:
        md_lines.append(f"- `{s['path']}` ({s.get('bytes')} bytes)")
        for sig in s.get("signals") or []:
            md_lines.append(f"  - {sig}")
    md_lines.append("")
    md_lines.append("## Notes / guardrails")
    md_lines.append("- This mapper intentionally skips `.env` and key/cert/secret-like files.")
    md_lines.append("- Output is for cross-checking structure + entrypoints; no changes are made to the VM.")
    md_lines.append("")
    map_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Write index
    index_payload = {
        "controller": base_url,
        "repo_root": repo_root,
        "top_level_counts": top,
        "anchors": file_summaries,
        "all_paths_count": len(paths),
        "safe_paths_count": len(safe_paths),
        "outputs": {
            "repo_tree_txt": str(tree_txt),
            "arch_map_md": str(map_md),
            "index_json": str(index_json),
        },
    }
    index_json.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    return index_payload

async def _maybe_handle_zobie_map(task: LLMTask, original_message: str) -> Optional[LLMResult]:
    parsed = _zobie_parse_command(original_message)
    if not parsed:
        return None

    base_url = parsed["base_url"]
    max_files = parsed["max_files"]
    include_hashes = parsed["include_hashes"]
    out_dir = _zobie_default_out_dir()

    if ROUTER_DEBUG:
        _debug_log("=" * 70)
        _debug_log("LOCAL ACTION: ZOBIE MAP")
        _debug_log(f"  Controller: {base_url}")
        _debug_log(f"  Out dir: {out_dir}")
        _debug_log(f"  max_files={max_files} include_hashes={include_hashes}")
        _debug_log("=" * 70)

    try:
        payload = await asyncio.to_thread(_zobie_run_map_sync, base_url, out_dir, max_files, include_hashes)
        outputs = payload.get("outputs") or {}

        content = "\n".join(
            [
                "ZOBIE MAP complete (read-only).",
                f"- Controller: {payload.get('controller')}",
                f"- VM repo root: {payload.get('repo_root')}",
                f"- Files: {payload.get('safe_paths_count')}/{payload.get('all_paths_count')} (safe/total)",
                "",
                "Outputs:",
                f"- {outputs.get('repo_tree_txt')}",
                f"- {outputs.get('arch_map_md')}",
                f"- {outputs.get('index_json')}",
            ]
        )

        return LLMResult(
            content=content,
            provider="local",
            model="zobie_mapper",
            finish_reason="stop",
            error_message=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response={"local_action": "zobie_map", "payload": payload},
            job_type=JobType.TEXT_HEAVY,
            routing_decision={
                "job_type": "local.zobie.map",
                "provider": "local",
                "model": "zobie_mapper",
                "reason": "Prompt trigger: ZOBIE MAP",
            },
        )
    except Exception as exc:
        err = f"ZOBIE MAP failed: {type(exc).__name__}: {exc}"
        return LLMResult(
            content=err,
            provider="local",
            model="zobie_mapper",
            finish_reason="error",
            error_message=err,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response={"local_action": "zobie_map", "error": err},
            job_type=JobType.TEXT_HEAVY,
            routing_decision={
                "job_type": "local.zobie.map",
                "provider": "local",
                "model": "zobie_mapper",
                "reason": "Prompt trigger: ZOBIE MAP (failed)",
            },
        )





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
    # Extract user message text (string or multimodal list)
    # ==========================================================================
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    original_message = ""

    if user_messages:
        content = user_messages[-1].get("content", "")
        if isinstance(content, str):
            original_message = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    original_message = part.get("text", "")
                    break
                elif isinstance(part, str):
                    original_message = part
                    break

        if ROUTER_DEBUG:
            _debug_log(f"Extracted message for override check: {repr(original_message[:100])}...")

    # ==========================================================================
    # Stage 1: Prompt-triggered ZOBIE MAP (local action, read-only)
    # ==========================================================================
    local_action_result = await _maybe_handle_zobie_map(task, original_message)
    if local_action_result is not None:
        return local_action_result

    # ==========================================================================
    # v0.15.1: FRONTIER OVERRIDE CHECK (BEFORE ALL OTHER ROUTING)
    # ==========================================================================
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
