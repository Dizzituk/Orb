# FILE: app/llm/local_tools/zobie_tools.py
"""Streaming local-tool generators for Zobie mapper and CREATE ARCHITECTURE MAP.

Kept out of app.llm.stream_router to keep the router lean and easier to sanity-check.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import contextlib
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from app.llm.local_tools.archmap_helpers import (
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    ARCHMAP_TEMPLATE_PATH,
    ARCHMAP_SECTION_MODE,
    ARCHMAP_SECTION_MAX_CHARS,
    ARCHMAP_SECTION_MAX_SCANNED_FILES,
    ARCHMAP_STREAM_TOKENS_TO_CLIENT,
    ARCHMAP_INGEST_TO_MEMORY,
    ARCHMAP_DOC_RAW_MAX_CHARS,
    ARCHMAP_OPENAI_MAX_COMPLETION_TOKENS,
    ARCHMAP_OPENAI_TEMPERATURE,
    ARCHMAP_OPENAI_TIMEOUT_SEC,
    _controller_fetch_file_content,
    _fmt_scanned_file,
    _line_number,
    _load_index_json,
    _next_archmap_version,
    _pick_scanned_files_for_section,
    _safe_read_text,
    _section_plan,
    _openai_chat_completion_nonstream,
)

logger = logging.getLogger(__name__)

# Zobie mapper settings
ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL", "http://192.168.250.2:8765")
ZOBIE_MAPPER_SCRIPT = os.getenv("ORB_ZOBIE_MAPPER_SCRIPT", r"D:\\tools\\zobie_mapper\\zobie_map.py")
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR", r"D:\\tools\\zobie_mapper\\out")
ZOBIE_MAPPER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_MAPPER_TIMEOUT_SEC", "180"))
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "200000 0 60 120000").strip()
ZOBIE_MAPPER_ARGS: List[str] = [a for a in ZOBIE_MAPPER_ARGS_RAW.split() if a]


def _sse_token(content: str) -> str:
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"


def _sse_error(error: str) -> str:
    return "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"

def _sse_done(
    *,
    provider: str,
    model: str,
    total_length: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "type": "done",
        "provider": provider,
        "model": model,
        "total_length": int(total_length or 0),
        "success": bool(success),
    }
    if error:
        payload["error"] = str(error)
    if meta:
        payload["meta"] = meta
    return "data: " + json.dumps(payload) + "\n\n"


def _chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _cap_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n<<truncated>>\n"


def _parse_reasoning_tags(raw: str) -> Tuple[str, Optional[str]]:
    """Strip optional <reasoning> blocks from model outputs."""
    if not raw:
        return "", None
    m = re.search(r"<reasoning>(.*?)</reasoning>", raw, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return raw, None
    cleaned = raw[: m.start()] + raw[m.end() :]
    return cleaned.strip(), m.group(1).strip()


def _extract_usage_tokens(usage: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    # OpenAI chat-completions usage keys vary; keep it flexible.
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt, completion


def _extract_stamp_from_paths(*paths: str) -> str:
    """Extract YYYY-MM-DD_HHMM stamp from known artifact names."""
    for p in paths:
        base = os.path.basename(p or "")
        for prefix in ("ARCH_MAP_", "INDEX_", "MANIFEST_", "REPO_TREE_"):
            m = re.search(prefix + r"(\d{4}-\d{2}-\d{2}_\d{4})", base)
            if m:
                return m.group(1)
    return ""


def _find_latest_matching(out_dir: str, pattern: str) -> str:
    try:
        best = ""
        best_mtime = -1.0
        for name in os.listdir(out_dir):
            if re.match(pattern, name):
                p = os.path.join(out_dir, name)
                try:
                    mt = os.path.getmtime(p)
                except Exception:
                    mt = -1
                if mt > best_mtime:
                    best_mtime = mt
                    best = p
        return best
    except Exception:
        return ""


async def _run_mapper() -> Tuple[str, str, List[str]]:
    """Run zobie_map.py and return (stdout, stderr, output_paths)."""
    cmd = [sys.executable, ZOBIE_MAPPER_SCRIPT, ZOBIE_CONTROLLER_URL, ZOBIE_MAPPER_OUT_DIR] + ZOBIE_MAPPER_ARGS
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=ZOBIE_MAPPER_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        raise RuntimeError(f"Zobie mapper timed out after {ZOBIE_MAPPER_TIMEOUT_SEC}s")
    stdout = (stdout_b or b"").decode("utf-8", errors="replace")
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")
    output_paths: List[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        # mapper prints absolute paths, one per line
        if os.path.isabs(s) and os.path.exists(s):
            output_paths.append(s)
            continue
        # sometimes a relative path slips through
        candidate = os.path.join(ZOBIE_MAPPER_OUT_DIR, s)
        if os.path.exists(candidate):
            output_paths.append(candidate)
    return stdout, stderr, output_paths


async def generate_local_zobie_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """Run the repo scanner (zobie mapper) and stream a brief progress UI."""
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)

    yield _sse_token("Step 1/1: Running repo scanner...\n")

    try:
        stdout, stderr, output_paths = await _run_mapper()
        if trace is not None:
            duration_ms = max(0, int(loop.time() * 1000) - started_ms)
            trace.log_model_call(
                "local_tool",
                "local",
                "zobie_mapper",
                "zobie_mapper",
                0,
                0,
                duration_ms,
                success=True,
                error=None,
            )
    except Exception as e:
        if trace is not None:
            trace.log_error(f"Zobie mapper failed: {e}")
        yield _sse_error(f"ZOBIE MAP failed: {e}")
        yield _sse_done(provider="local", model="zobie_mapper", success=False, error=str(e))
        return

    # Record a small breadcrumb in memory (optional but useful for debugging)
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content="[zobie_map] Repo scan complete.",
                provider="local",
                model="zobie_mapper",
            ),
        )
    except Exception:
        pass

    if not output_paths:
        # fall back: list newest artifacts
        guess = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, r"^(ARCH_MAP_|INDEX_|MANIFEST_|REPO_TREE_).*\.(md|json|txt)$")
        if guess:
            output_paths = [guess]

    summary = "Repo scan complete.\n\nOutputs:\n" + "\n".join(f"- {p}" for p in output_paths) + "\n"
    yield _sse_token(summary)
    yield _sse_done(provider="local", model="zobie_mapper", total_length=len(summary), meta={"outputs": output_paths, "out_dir": ZOBIE_MAPPER_OUT_DIR})


async def generate_local_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """Run mapper + generate a sectioned architecture map, writing files to ZOBIE_MAPPER_OUT_DIR."""
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)

    # 1) Run mapper
    yield _sse_token("Step 1/4: Running repo scanner...\n")
    try:
        stdout, stderr, output_paths = await _run_mapper()
    except Exception as e:
        if trace is not None:
            trace.log_error(f"Mapper failed: {e}")
        yield _sse_error(f"Mapper failed: {e}")
        yield _sse_done(provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, success=False, error=str(e))
        return

    # Determine key artifacts
    arch_path = next((p for p in output_paths if os.path.basename(p).startswith("ARCH_MAP_") and p.lower().endswith(".md")), "")
    index_path = next((p for p in output_paths if os.path.basename(p).startswith("INDEX_") and p.lower().endswith(".json")), "")
    manifest_path = next((p for p in output_paths if os.path.basename(p).startswith("MANIFEST_") and p.lower().endswith(".json")), "")

    stamp = _extract_stamp_from_paths(arch_path, index_path, manifest_path)
    tree_path = os.path.join(ZOBIE_MAPPER_OUT_DIR, f"REPO_TREE_{stamp}.txt") if stamp else ""
    if tree_path and not os.path.exists(tree_path):
        tree_path = ""

    if not tree_path:
        tree_path = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, r"^REPO_TREE_\d{4}-\d{2}-\d{2}_\d{4}\.txt$")
    if not arch_path:
        arch_path = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, r"^ARCH_MAP_\d{4}-\d{2}-\d{2}_\d{4}\.md$")
    if not index_path:
        index_path = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, r"^INDEX_\d{4}-\d{2}-\d{2}_\d{2}_\d{4}\.json$")
        if not index_path:
            index_path = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, r"^INDEX_\d{4}-\d{2}-\d{2}_\d{4}\.json$")
    if not (arch_path and index_path):
        yield _sse_error("Mapper ran but expected artifacts were not found (ARCH_MAP_*.md and INDEX_*.json).")
        yield _sse_done(provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, success=False, error="artifacts_not_found")
        return

    stamp2 = datetime.now().strftime("%Y-%m-%d_%H%M")
    version_n = _next_archmap_version(ZOBIE_MAPPER_OUT_DIR)

    # 2) Load index + build section plan evidence
    yield _sse_token("Step 2/4: Planning sectioned generation...\n")
    index_data = _load_index_json(index_path)
    if not index_data:
        yield _sse_error(f"Failed to load index JSON: {index_path}")
        return

    sections = _section_plan()

    # Load optional template style
    template_text = ""
    if ARCHMAP_TEMPLATE_PATH:
        template_text = _safe_read_text(ARCHMAP_TEMPLATE_PATH, max_bytes=2_000_000)

    section_files: List[str] = []
    all_section_texts: List[str] = []

    # 3) Generate each section via provider/model
    yield _sse_token("Step 3/4: Generating sections...\n")
    for idx, sec in enumerate(sections, start=1):
        sec_name = sec.get("name", f"Section {idx}")
        scanned_files = _pick_scanned_files_for_section(index_data, sec)

        evidence_parts: List[str] = []
        evidence_parts.append(f"# Evidence: {sec_name}\n")
        if tree_path and os.path.exists(tree_path):
            evidence_parts.append("## Repo tree (excerpt)\n")
            evidence_parts.append(_cap_text(_safe_read_text(tree_path, max_bytes=3_000_000), 120000))
            evidence_parts.append("\n")
        evidence_parts.append("## Scanned file summaries\n")
        for sf in scanned_files[: max(5, min(ARCHMAP_SECTION_MAX_SCANNED_FILES, 120))]:
            try:
                evidence_parts.append(_fmt_scanned_file(sf))
            except Exception:
                continue

        evidence = "\n".join(evidence_parts)
        evidence = _cap_text(evidence, ARCHMAP_SECTION_MAX_CHARS)

        system_prompt = (
            "You are generating a precise software architecture map.\n"
            "Use only the provided evidence; do not invent files or routes.\n"
            "If something is unknown, say it is unknown.\n"
            "Output Markdown.\n"
        )

        user_prompt = (
            f"Project: Orb/Astra\n"
            f"Task: CREATE ARCHITECTURE MAP (sectioned)\n"
            f"Section: {sec_name}\n\n"
        )
        if template_text:
            user_prompt += "Style template (for headings/format):\n" + _cap_text(template_text, 60000) + "\n\n"
        user_prompt += "Evidence:\n" + evidence + "\n\n"
        user_prompt += "Write the section now."

        # Call provider/model (OpenAI direct path for this tool)
        try:
            raw, usage = _openai_chat_completion_nonstream(
                model=ARCHMAP_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_completion_tokens=ARCHMAP_OPENAI_MAX_COMPLETION_TOKENS,
                temperature=ARCHMAP_OPENAI_TEMPERATURE,
                timeout_sec=ARCHMAP_OPENAI_TIMEOUT_SEC,
            )
        except Exception as e:
            if trace is not None:
                trace.log_error(f"Archmap section failed ({sec_name}): {e}")
            yield _sse_error(f"Section generation failed ({sec_name}): {e}")
            yield _sse_done(provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, success=False, error=str(e), meta={"section": sec_name})
            return

        section_text, _reason = _parse_reasoning_tags(raw)

        sec_path = os.path.join(ZOBIE_MAPPER_OUT_DIR, f"ARCH_MAP_SECTION_V{version_n}_{idx:02d}_{stamp2}.md")
        with open(sec_path, "w", encoding="utf-8") as f:
            f.write(f"# Orb Architecture Map v{version_n}\n\n")
            f.write(f"## {sec_name}\n\n")
            f.write(section_text.strip() + "\n")

        section_files.append(sec_path)
        all_section_texts.append(section_text.strip())

        # Stream minimal progress; do not dump content unless enabled
        if ARCHMAP_STREAM_TOKENS_TO_CLIENT:
            for ch in _chunk_text(section_text + "\n", 1000):
                yield _sse_token(ch)
        else:
            yield _sse_token(f"- Completed: {sec_name}\n")

    # Combine into full map
    yield _sse_token("Step 4/4: Saving combined map...\n")
    header = []
    header.append(f"# Orb Architecture Map v{version_n}\n")
    header.append(f"Generated: {stamp2}\n")
    header.append("## Inputs\n")
    header.append(f"- ARCH_MAP evidence pack: {os.path.basename(arch_path)}\n")
    header.append(f"- INDEX artifact: {os.path.basename(index_path)}\n")
    if tree_path:
        header.append(f"- REPO_TREE: {os.path.basename(tree_path)}\n")
    header.append("\n---\n\n")

    full_text = "".join(header)
    for sec_name, sec_text in zip([s.get("name", "") for s in sections], all_section_texts):
        full_text += f"## {sec_name}\n\n{sec_text.strip()}\n\n---\n\n"

    out_full = os.path.join(ZOBIE_MAPPER_OUT_DIR, f"ARCH_MAP_FULL_V{version_n}_{stamp2}.md")
    with open(out_full, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Optional ingest
    if ARCHMAP_INGEST_TO_MEMORY:
        try:
            filename = os.path.basename(out_full)
            file_rec = memory_service.create_file(
                db,
                memory_schemas.FileCreate(
                    project_id=project_id,
                    path=out_full,
                    original_name=filename,
                    file_type="generated/archmap",
                    description=f"Architecture map v{version_n} ({stamp2})",
                ),
            )
            trimmed = full_text[:ARCHMAP_DOC_RAW_MAX_CHARS]
            if len(full_text) > ARCHMAP_DOC_RAW_MAX_CHARS:
                trimmed += "\n\n<<trimmed for DB storage>>\n"
            memory_service.create_document_content(
                db,
                memory_schemas.DocumentContentCreate(
                    project_id=project_id,
                    file_id=file_rec.id,
                    filename=filename,
                    doc_type="archmap",
                    raw_text=trimmed,
                    summary=None,
                    structured_data=None,
                ),
            )
            memory_service.create_message(
                db,
                memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=f"[archmap] Saved + ingested: {out_full}",
                    provider=ARCHMAP_PROVIDER,
                    model=ARCHMAP_MODEL,
                ),
            )
        except Exception as e:
            logger.exception("[archmap] ingest failed: %s", e)

    if trace is not None:
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)
        trace.log_model_call(
            "local_tool",
            ARCHMAP_PROVIDER,
            ARCHMAP_MODEL,
            "create_architecture_map",
            0,
            0,
            duration_ms,
            success=True,
            error=None,
        )

    # Return a short summary
    summary = (
        f"Architecture map v{version_n} generated.\n\n"
        f"Full: {out_full}\n"
        + "\n".join(f"Section: {p}" for p in section_files)
        + "\n"
    )
    yield _sse_token(summary)
    yield _sse_done(provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, total_length=len(summary), meta={"version": version_n, "full": out_full, "sections": section_files, "out_dir": ZOBIE_MAPPER_OUT_DIR})
