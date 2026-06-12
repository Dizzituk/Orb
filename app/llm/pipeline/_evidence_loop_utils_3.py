# Purpose: evidence loop utils 3
# Called-by: app.llm.pipeline.evidence_loop
# Depends-on: app.db, app.embeddings.service, app.llm.local_tools.arch_query, app.llm.local_tools.zobie.sandbox_inspector (+4 more)
# Last-renovated: 2026-06-11
from __future__ import annotations
import logging
import re
import yaml
from app.llm.pipeline._evidence_loop_utils_2 import _block_boundary_pattern, _request_block_pattern, _restructure_flat_evidence_request
from typing import Dict, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _escape_backslashes_for_yaml(text: str) -> str:
    """Escape Windows-style backslashes before YAML parsing.

    YAML treats backslash as an escape character inside double-quoted strings.
    Windows paths like D:\\Orb\\main.py contain \\O and \\m which are invalid
    YAML escape sequences, causing yaml.safe_load() to throw ScannerError.

    Strategy: double any backslash followed by a letter that is NOT a common
    YAML escape (\\n, \\r, \\t). Obscure YAML escapes like \\L, \\N, \\P, \\U
    are almost always accidental Windows path components, so we double those too.
    Also handles trailing backslash before closing quote (\\\" edge case)
    which YAML would interpret as an escaped quote character.
    Already-doubled backslashes (\\\\) are preserved.
    """
    common_escapes = set('nrt')  # \n \r \t — the only ones LLMs use

    def _fix_match(m):
        letter = m.group(1)
        if letter in common_escapes:
            return '\\' + letter
        else:
            return '\\\\' + letter

    # Protect already-doubled backslashes, fix singles, then restore
    text = text.replace('\\\\', '\x00DOUBLE_BS\x00')
    text = re.sub(r'\\([A-Za-z"])', _fix_match, text)
    text = text.replace('\x00DOUBLE_BS\x00', '\\\\')
    return text

def _regex_fallback_extract(block_text: str) -> dict:
    """Last-resort regex extraction when YAML parsing completely fails.

    Extracts key fields using simple regex patterns. Won't get nested
    structures perfectly, but recovers the essentials (id, severity,
    tool_calls) so the evidence loop can still dispatch.
    """
    result = {}

    # Extract simple key: "value" pairs
    for key in ["id", "severity", "need", "why", "success_criteria", "fallback_if_not_found"]:
        # Match: key: "value" or key: value (unquoted)
        pattern = r'^\s*' + re.escape(key) + r':\s*["\']?(.*?)["\']?\s*$'
        match = re.search(pattern, block_text, re.MULTILINE)
        if match:
            result[key] = match.group(1).strip().strip('"\'')

    # Extract tool_calls — look for tool: "name" patterns
    tool_calls = []
    tool_pattern = r'(?:^|\n)\s*-\s*tool:\s*["\']?([\w.]+)["\']?'
    for tool_match in re.finditer(tool_pattern, block_text):
        tool_name = tool_match.group(1)
        tc = {"tool": tool_name, "args": {}}

        # Try to find args on subsequent lines
        args_pattern = (
            r'tool:\s*["\']?' + re.escape(tool_name) + r'["\']?\s*\n'
            r'\s*args:\s*(\{.*?\})'
        )
        args_match = re.search(args_pattern, block_text, re.DOTALL)
        if args_match:
            try:
                args_text = _escape_backslashes_for_yaml(args_match.group(1))
                tc["args"] = yaml.safe_load(args_text) or {}
            except Exception:
                # Extract file_path specifically if present
                fp_pattern = r'file_path:\s*["\']?([\w./\\:_-]+)["\']?'
                fp_match = re.search(fp_pattern, args_match.group(1))
                if fp_match:
                    tc["args"] = {"file_path": fp_match.group(1)}

        tool_calls.append(tc)

    if tool_calls:
        result["tool_calls"] = tool_calls

    # Extract scope.roots
    roots_match = re.search(r'roots:\s*\[(.*?)\]', block_text)
    if roots_match:
        roots_text = roots_match.group(1)
        roots = [r.strip().strip('"\'')
                 for r in roots_text.split(',') if r.strip()]
        max_files_match = re.search(r'max_files:\s*(\d+)', block_text)
        result["scope"] = {
            "roots": roots,
            "max_files": int(max_files_match.group(1)) if max_files_match else 500,
        }

    return result if result.get("id") else None

def _try_parse_block(block_text: str, start_pos: int) -> dict:
    """Attempt to parse an EVIDENCE_REQUEST block with three-layer defense.

    Layer 1: Escape backslashes, then YAML parse + restructure
    Layer 2: Raw YAML parse + restructure (if escaping made it worse)
    Layer 3: Regex fallback extraction

    Returns parsed dict or None if all layers fail.
    """
    # Layer 1: Fix backslashes then YAML parse
    try:
        fixed_text = _escape_backslashes_for_yaml(block_text)
        parsed = yaml.safe_load(fixed_text)
        if isinstance(parsed, dict):
            req = _restructure_flat_evidence_request(parsed)
            if req and req.get("id"):
                logger.debug(
                    "[evidence_loop] v2.1 Parsed ER block at pos %d (YAML+backslash fix): id=%s",
                    start_pos, req.get("id"),
                )
                return req
    except yaml.YAMLError:
        pass

    # Layer 2: Try raw YAML without backslash fix (maybe it was fine)
    try:
        parsed = yaml.safe_load(block_text)
        if isinstance(parsed, dict):
            req = _restructure_flat_evidence_request(parsed)
            if req and req.get("id"):
                logger.debug(
                    "[evidence_loop] v2.1 Parsed ER block at pos %d (raw YAML): id=%s",
                    start_pos, req.get("id"),
                )
                return req
    except yaml.YAMLError:
        pass

    # Layer 3: Regex fallback
    req = _regex_fallback_extract(block_text)
    if req and req.get("id"):
        logger.info(
            "[evidence_loop] v2.1 Parsed ER block at pos %d (regex fallback): id=%s, severity=%s, tools=%d",
            start_pos, req.get("id"), req.get("severity", "?"), len(req.get("tool_calls", [])),
        )
        return req

    logger.warning(
        "[evidence_loop] v2.1 All parse layers failed for ER block at pos %d (block_len=%d)",
        start_pos, len(block_text),
    )
    return None

def parse_evidence_requests(output: str) -> list:
    """Parse EVIDENCE_REQUEST blocks from stage output.

    v2.1: Three-layer defense against LLM formatting variations:
      1. Escape Windows backslashes + YAML parse + flat restructure
      2. Raw YAML parse + flat restructure
      3. Regex fallback extraction

    Uses block-boundary parsing to safely handle multiple EVIDENCE_REQUEST blocks,
    blank lines inside YAML, nested maps, and arbitrary top-level blocks that follow.

    Returns list of dicts: {id, severity, need, why, scope, tool_calls,
    success_criteria, fallback_if_not_found}.
    """
    requests = []
    seen_ids = set()

    # Find each EVIDENCE_REQUEST: marker and capture until the next block
    # boundary or end of string.
    boundary = _block_boundary_pattern()

    # Walk through all EVIDENCE_REQUEST: occurrences
    for match in re.finditer(r'\nEVIDENCE_REQUEST:', output):
        start = match.start() + 1  # skip the leading \n
        # Find the next block boundary AFTER this marker
        end_match = boundary.search(output, pos=start + len("EVIDENCE_REQUEST:"))
        end = end_match.start() if end_match else len(output)
        block_text = output[start:end].strip()

        req = _try_parse_block(block_text, start)
        if req and req.get("id") and req["id"] not in seen_ids:
            requests.append(req)
            seen_ids.add(req["id"])

    # Also check if output STARTS with EVIDENCE_REQUEST: (no leading \n)
    if output.lstrip().startswith("EVIDENCE_REQUEST:"):
        end_match = boundary.search(output, pos=len("EVIDENCE_REQUEST:"))
        end = end_match.start() if end_match else len(output)
        block_text = output[:end].strip()

        req = _try_parse_block(block_text, 0)
        if req and req.get("id") and req["id"] not in seen_ids:
            requests.insert(0, req)
            seen_ids.add(req["id"])

    if requests:
        logger.info(
            "[evidence_loop] v2.1 Parsed %d EVIDENCE_REQUEST(s): %s",
            len(requests), [r.get("id") for r in requests],
        )

    return requests

def strip_forced_stop_requests(output: str, forced_ids: set) -> str:
    """Replace force-stopped EVIDENCE_REQUEST blocks with FORCED_RESOLUTION markers.

    Different from strip_fulfilled_requests: these were NOT successfully fulfilled.
    The stage ran out of loops and these are being converted to HUMAN_REQUIRED.
    Using a distinct stub type prevents lying in the audit trail.
    """
    for req_id in forced_ids:
        pattern = _request_block_pattern(req_id)
        stub = (
            f'FORCED_RESOLUTION:\n'
            f'  id: "{req_id}"\n'
            f'  reason: "force_resolve_only"\n'
            f'  action: "HUMAN_REQUIRED"\n'
        )
        output = pattern.sub(stub, output)
    return output

def validate_output_structure(output: str) -> bool:
    """Ensure CRITICAL_CLAIMS is the last structured block if present.

    Once a stage's output is final, call this and check the return value.
    After validation passes, the orchestrator must NOT modify the output.

    Returns True if valid (or no register present during transition).
    """
    claims_idx = output.rfind("\nCRITICAL_CLAIMS:")
    if claims_idx == -1:
        return True  # No register (allowed during transition)

    after_claims = output[claims_idx:].split("CRITICAL_CLAIMS:", 1)[-1]
    forbidden = [
        "EVIDENCE_REQUEST:", "DECISION:", "HUMAN_REQUIRED:",
        "RESOLVED_REQUEST:", "FORCED_RESOLUTION:",
    ]
    for marker in forbidden:
        if marker in after_claims:
            return False
    return True

def validate_tool_call(
    tool_name: str,
    tool_args: dict,
    stage_name: str,
    severity: str = "NONCRITICAL",
) -> Tuple[bool, Optional[str]]:
    """Validate a tool call: check that the tool exists and is allowed.

    Returns (is_valid, error_message).
    If tool is unknown or args are malformed, returns (False, reason).
    Tweak #5: unknown tool / invalid args -> NONCRITICAL issue + ignore,
    unless the request is CRITICAL, then escalate to HUMAN_REQUIRED after loops.
    """
    from .evidence_loop import TOOL_DISPATCH, tool_allowed
    if not tool_name or not isinstance(tool_name, str):
        return False, f"Invalid tool name: {tool_name!r}"

    if tool_name not in TOOL_DISPATCH:
        return False, f"Unknown tool '{tool_name}' — not in dispatch table"

    if not tool_allowed(tool_name, stage_name):
        return False, (
            f"Tool '{tool_name}' not allowed for stage '{stage_name}'. "
            f"Allowed stages: {TOOL_DISPATCH[tool_name]['allowed_stages']}"
        )

    if not isinstance(tool_args, dict):
        return False, f"Tool args must be a dict, got {type(tool_args).__name__}"

    return True, None

async def execute_tool_call(
    tool_call: Dict,
    *,
    stage_name: str,
    bundle: object,
    max_files: int = 500,
) -> Dict:
    """Dispatch a single tool call from an EVIDENCE_REQUEST.

    Each tool is resolved to a real function imported from ASTRA modules.
    Returns a result dict the orchestrator can feed back to the stage.
    Unknown / disallowed tools return an error payload.
    """
    tool_name = tool_call.get("tool", "")
    args = tool_call.get("args", {})

    ok, err = validate_tool_call(tool_name, args, stage_name)
    if not ok:
        return {"tool": tool_name, "error": err}

    try:
        # ── sandbox_inspector ──
        if tool_name == "sandbox_inspector.read_sandbox_file":
            file_path = args.get("file_path") or args.get("path") or ""
            if not file_path.strip():
                logger.warning("[evidence_loop] v2.3 Empty file_path in read_sandbox_file call — args=%s", args)
                print(f"[evidence_loop] ⚠️ SKIPPED read_sandbox_file: empty file_path (args={args})")
                return {"tool": tool_name, "success": False, "error": "Empty file_path", "content": None}
            from app.pot_spec.grounded.evidence_gathering import sandbox_read_file
            success, content = sandbox_read_file(
                file_path,
                max_chars=args.get("max_chars", 50000),
            )
            return {"tool": tool_name, "success": success, "content": content[:4000] if content else None}

        if tool_name == "sandbox_inspector.file_exists_in_sandbox":
            file_path = args.get("file_path") or args.get("path") or ""
            if not file_path.strip():
                logger.warning("[evidence_loop] v2.3 Empty file_path in file_exists call — args=%s", args)
                return {"tool": tool_name, "exists": False, "error": "Empty file_path"}
            from app.pot_spec.grounded.evidence_gathering import sandbox_path_exists
            exists = sandbox_path_exists(file_path)
            return {"tool": tool_name, "exists": exists}

        if tool_name == "sandbox_inspector.run_sandbox_discovery_chain":
            from app.llm.local_tools.zobie.sandbox_inspector import run_sandbox_discovery_chain
            result = run_sandbox_discovery_chain(
                anchor=args.get("anchor", ""),
                subfolder=args.get("subfolder", ""),
                job_intent=args.get("job_intent", ""),
            )
            return {"tool": tool_name, "result": result}

        # ── evidence_collector ──
        if tool_name == "evidence_collector.load_evidence":
            from app.pot_spec.evidence_collector import load_evidence
            loaded = load_evidence()
            return {"tool": tool_name, "loaded": bool(loaded)}

        if tool_name == "evidence_collector.add_file_read_to_bundle":
            from app.pot_spec.evidence_collector import add_file_read_to_bundle
            content = add_file_read_to_bundle(
                bundle,
                args.get("path", ""),
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
                head_lines=args.get("head_lines"),
            )
            return {"tool": tool_name, "content": content[:4000] if content else None}

        if tool_name == "evidence_collector.add_search_to_bundle":
            from app.pot_spec.evidence_collector import add_search_to_bundle
            hits = add_search_to_bundle(
                bundle,
                args.get("query", ""),
                limit=min(args.get("limit", 10), max_files),
            )
            return {"tool": tool_name, "hits": hits}

        if tool_name == "evidence_collector.find_in_evidence":
            from app.pot_spec.evidence_collector import find_in_evidence
            matches = find_in_evidence(
                bundle,
                args.get("pattern", ""),
                source_type=args.get("source_type"),
            )
            return {"tool": tool_name, "matches": matches}

        if tool_name == "evidence_collector.verify_path_exists":
            from app.pot_spec.evidence_collector import verify_path_exists
            exists = verify_path_exists(bundle, args.get("path", ""))
            return {"tool": tool_name, "exists": exists}

        # ── embeddings / arch_query ──
        if tool_name == "embeddings_service.search_embeddings":
            from app.embeddings.service import search_embeddings
            from app.db import SessionLocal
            _db = SessionLocal()
            try:
                results, total = search_embeddings(
                    db=_db,
                    project_id=args.get("project_id", 0),
                    query=args.get("query", ""),
                    top_k=min(args.get("top_k", 10), max_files),
                )
                return {"tool": tool_name, "results": [r.__dict__ if hasattr(r, '__dict__') else r for r in results], "total_searched": total}
            finally:
                _db.close()

        if tool_name == "arch_query.search_symbols":
            from app.llm.local_tools.arch_query import search_symbols
            results = search_symbols(
                args.get("query", ""),
                limit=min(args.get("limit", 10), max_files),
            )
            return {"tool": tool_name, "results": results}

        if tool_name == "arch_query.get_file_signatures":
            from app.llm.local_tools.arch_query import get_file_signatures
            sigs = get_file_signatures(args.get("file_path", ""))
            return {"tool": tool_name, "signatures": sigs}

        return {"tool": tool_name, "error": f"Tool '{tool_name}' not implemented in dispatcher"}

    except ImportError as exc:
        logger.warning("[evidence_loop] Import failed for tool %s: %s", tool_name, exc)
        return {"tool": tool_name, "error": f"Module not available: {exc}"}
    except Exception as exc:
        logger.warning("[evidence_loop] Tool %s failed: %s", tool_name, exc)
        return {"tool": tool_name, "error": str(exc)}
