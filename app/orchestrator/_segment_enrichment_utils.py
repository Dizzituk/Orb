from __future__ import annotations
import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from app.sandbox_fs import sandbox_read_text as _sbx_read_text
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


ENRICHMENT_PROVIDER = os.getenv("SEGMENT_ENRICHMENT_PROVIDER", "anthropic")

def _pick_primary_source(
    source_evidence: Dict[str, str],
) -> Tuple[str, str]:
    """
    From the source evidence dict, pick the largest Python file as the
    primary monolith.  Returns (path, content) or ("", "") if none found.
    """
    best_path = ""
    best_content = ""
    for path, content in source_evidence.items():
        if not path.endswith(".py"):
            continue
        if len(content) > len(best_content):
            best_path = path
            best_content = content
    return best_path, best_content

def _is_constant_name(name: str) -> bool:
    """
    Check if a name looks like a Python constant (ALL_CAPS or ALL_CAPS_WITH_UNDERSCORES).
    Excludes dunder names like __all__.
    """
    if name.startswith("__") and name.endswith("__"):
        return False
    # Must have at least one uppercase letter and contain only uppercase, digits, underscores
    return bool(re.match(r'^[A-Z][A-Z0-9_]*$', name))

def _get_function_signature(node: ast.FunctionDef, source_code: str) -> str:
    """
    Extract the function signature (def line with parameters and return annotation).
    """
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"

    # Build parameter list from AST
    args = node.args
    params = []

    # positional args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        param = arg.arg
        if arg.annotation:
            ann = ast.get_source_segment(source_code, arg.annotation)
            if ann:
                param += f": {ann}"
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            default = ast.get_source_segment(source_code, args.defaults[default_idx])
            if default:
                param += f" = {default}"
        params.append(param)

    # *args
    if args.vararg:
        va = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            ann = ast.get_source_segment(source_code, args.vararg.annotation)
            if ann:
                va += f": {ann}"
        params.append(va)
    elif args.kwonlyargs:
        params.append("*")

    # keyword-only args
    for i, kwarg in enumerate(args.kwonlyargs):
        param = kwarg.arg
        if kwarg.annotation:
            ann = ast.get_source_segment(source_code, kwarg.annotation)
            if ann:
                param += f": {ann}"
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            default = ast.get_source_segment(source_code, args.kw_defaults[i])
            if default:
                param += f" = {default}"
        params.append(param)

    # **kwargs
    if args.kwarg:
        kw = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            ann = ast.get_source_segment(source_code, args.kwarg.annotation)
            if ann:
                kw += f": {ann}"
        params.append(kw)

    sig = f"{prefix} {node.name}({', '.join(params)})"

    if node.returns:
        ret = ast.get_source_segment(source_code, node.returns)
        if ret:
            sig += f" -> {ret}"

    return sig + ":"

def _get_name(node: ast.expr) -> str:
    """Extract a readable name from an AST name/attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        return f"{_get_name(node.value)}[...]"
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    return "?"

def _extract_names_from_import(import_line: str) -> Set[str]:
    """
    Extract the names imported by an import statement.

    Examples:
        "import os" → {"os"}
        "from typing import Dict, List, Optional" → {"Dict", "List", "Optional"}
        "from app.llm.streaming import call_llm_text" → {"call_llm_text"}
        "import json" → {"json"}
    """
    names: Set[str] = set()
    line = import_line.strip()

    if line.startswith("from "):
        # from X import a, b, c
        match = re.search(r'import\s+(.+)', line)
        if match:
            imports_part = match.group(1)
            # Handle multi-line imports (parenthesised)
            imports_part = imports_part.strip("()")
            for part in imports_part.split(","):
                part = part.strip()
                if " as " in part:
                    # "foo as bar" → use the alias "bar"
                    names.add(part.split(" as ")[-1].strip())
                elif part and part != "*":
                    names.add(part.strip())
    elif line.startswith("import "):
        # import X, Y
        imports_part = line[7:]
        for part in imports_part.split(","):
            part = part.strip()
            if " as " in part:
                names.add(part.split(" as ")[-1].strip())
            elif part:
                # "import os.path" → use "os"
                names.add(part.split(".")[0].strip())

    return names

def _build_enrichment_user_prompt(
    manifest: Any,
    symbol_map: Dict[str, Any],
    extractions: Dict[str, Dict],
    unassigned_symbols: List[Dict[str, Any]],
    experience_patterns: str,
    source_path: str,
) -> str:
    """Build the user prompt for the single LLM intelligence call."""
    parts = []

    # Segments overview
    parts.append("## Segments\n")
    for seg in manifest.segments:
        deps = ", ".join(seg.dependencies) if seg.dependencies else "(none)"
        files = ", ".join(seg.file_scope)
        parts.append(
            f"- **{seg.segment_id}**: {seg.title}\n"
            f"  - Target files: {files}\n"
            f"  - Dependencies: {deps}\n"
        )

    # Exports per segment
    parts.append("\n## Symbol Map\n### Exports per segment:\n")
    for seg in manifest.segments:
        seg_exports = symbol_map["exports"].get(seg.segment_id, set())
        if seg_exports:
            parts.append(f"- **{seg.segment_id}**: {', '.join(sorted(seg_exports))}")
        else:
            parts.append(f"- **{seg.segment_id}**: (no symbols assigned yet)")

    # Cross-segment dependencies
    parts.append("\n### Cross-segment dependencies:\n")
    has_deps = False
    for seg in manifest.segments:
        seg_consumes = symbol_map["consumes"].get(seg.segment_id, {})
        for other_id, symbols in seg_consumes.items():
            parts.append(f"- {seg.segment_id} imports from {other_id}: {', '.join(symbols)}")
            has_deps = True
    if not has_deps:
        parts.append("(none detected yet — will be clearer after symbol assignment)")

    # Unresolved symbols
    parts.append("\n### Unresolved symbols (CRITICAL — these will cause boot failure):\n")
    for u in symbol_map.get("unresolved", []):
        parts.append(f"- {u}")
    if not symbol_map.get("unresolved"):
        parts.append("(none)")

    # Unassigned symbols for LLM to resolve
    if unassigned_symbols:
        parts.append(
            f"\n## Unassigned Symbols ({len(unassigned_symbols)} symbols need assignment)\n"
            "These symbols were extracted from the monolith but could not be "
            "deterministically assigned to any segment.  For each one, decide "
            "which segment it belongs to based on the segment descriptions and "
            "target file names.\n"
        )
        for sym in unassigned_symbols:
            if sym["type"] == "function":
                parts.append(
                    f"- **{sym['name']}** (function): `{sym.get('signature', '')}`\n"
                    f"  Docstring: {sym.get('docstring', '(none)')[:150]}\n"
                    f"  Lines: {sym.get('line_range', '?')}"
                )
            elif sym["type"] == "class":
                parts.append(
                    f"- **{sym['name']}** (class): bases={sym.get('bases', [])}, "
                    f"methods={sym.get('methods', [])}"
                )
            elif sym["type"] == "constant":
                parts.append(
                    f"- **{sym['name']}** (constant): {sym.get('value_preview', '')}"
                )

    # Experience patterns
    if experience_patterns:
        parts.append(f"\n## Experience Patterns (lessons from past runs)\n{experience_patterns}")

    # Instructions
    parts.append("""
## Instructions

Respond in JSON with this exact structure:
{
  "segments": {
    "<segment_id>": {
      "implementation_order": <integer, 1 = implement first>,
      "design_guidance": "<2-3 sentences of specific advice>",
      "risk_level": "<low|medium|high>",
      "risk_notes": "<why this segment is risky, if medium/high>"
    }
  },
  "symbol_assignments": {
    "<symbol_name>": "<segment_id it belongs to>"
  },
  "global_notes": "<any cross-cutting concerns>"
}

Pay special attention to:
- Constants/config modules: EVERY constant must be included (this is the #1 failure mode)
- Facade/init modules: Must re-export exactly the right symbols
- Modules with many cross-segment consumers: High risk if they miss exports
- For symbol_assignments: assign each unassigned symbol to the segment whose TARGET FILE
  will DEFINE (contain) the function implementation — NOT the segment that calls/uses it.
  Example: if 'can_execute_segment' is a dependency-checking helper that will live in
  '_dependencies.py', assign it to the segment targeting '_dependencies.py' even though
  it is called by the main orchestration segment.
  Key principle: each function belongs to the segment responsible for DEFINING it.
  The consuming segment will import it — it does not need to own it.
""")

    return "\n".join(parts)

def load_enrichment(job_dir_path: str, segment_id: str) -> Optional[Dict]:
    """Load cached enrichment for a segment (for resume/retry)."""
    path = os.path.join(job_dir_path, "segments", segment_id, "enrichment.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[SEGMENT_ENRICHMENT] Failed to load %s: %s", path, e)
        return None


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "ENRICHMENT_MAX_TOKENS": "_segment_enrichment_utils_5",
    "ENRICHMENT_MODEL": "_segment_enrichment_utils_5",
    "ENRICHMENT_SYSTEM_PROMPT": "_segment_enrichment_utils_5",
    "ENRICHMENT_TIMEOUT": "_segment_enrichment_utils_5",
    "SegmentEnrichment": "_segment_enrichment_utils_5",
    "_apply_llm_assignments": "_segment_enrichment_utils_5",
    "_build_per_segment_extractions": "_segment_enrichment_utils_5",
    "_load_experience_patterns": "_segment_enrichment_utils_5",
    "_build_symbol_map": "_segment_enrichment_utils_6",
    "_generate_implementation_intelligence": "_segment_enrichment_utils_6",
    "_save_enrichment": "_segment_enrichment_utils_6",
    "BUILD_ID": "_segment_enrichment_utils_7",
    "enrich_segments": "_segment_enrichment_utils_7",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.orchestrator.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
