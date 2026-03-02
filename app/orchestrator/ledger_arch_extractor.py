# FILE: app/orchestrator/ledger_arch_extractor.py
"""
Extract architectural decisions from architecture text and write to evidence ledger.

BUILD_ID: 2026-02-28-v1.0-ledger-arch-extractor

After architecture generation, this module parses the structured output to
extract decisions (API endpoints, file paths, models, patterns) and writes
them as ledger entries. Downstream stages and future passes can then read
these as ground truth without re-discovering from the codebase.

Called from segment_loop.py after architecture is sanitised and saved.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEDGER_ARCH_EXTRACTOR_BUILD_ID = "2026-02-28-v1.0-ledger-arch-extractor"


def extract_and_record_decisions(
    arch_text: str,
    seg_id: str,
    ledger: Any,
    job_dir: str,
    emit: Any = None,
) -> int:
    """Extract decisions from architecture text and write to ledger.

    Returns count of decisions recorded.
    """
    if not ledger or not arch_text:
        return 0

    try:
        from app.orchestrator.evidence_ledger import ledger_append, save_ledger
    except ImportError:
        logger.warning("[ledger_arch_extractor] Cannot import evidence_ledger")
        return 0

    decisions: List[Dict[str, str]] = []

    # --- Extract file inventory (CREATE/MODIFY paths) ---
    decisions.extend(_extract_file_inventory(arch_text))

    # --- Extract API endpoints ---
    decisions.extend(_extract_api_endpoints(arch_text))

    # --- Extract model/schema definitions ---
    decisions.extend(_extract_models(arch_text))

    # --- Extract import relationships ---
    decisions.extend(_extract_imports(arch_text))

    # Write each decision to ledger
    count = 0
    for d in decisions:
        try:
            # Build relevant_to list from decision key (e.g. file path, endpoint)
            _relevant = [d.get("key", seg_id)]
            if seg_id and seg_id not in _relevant:
                _relevant.append(seg_id)

            ledger_append(
                ledger,
                entry_type="decision",
                stage="critical_pipeline",
                relevant_to=_relevant,
                summary=d["summary"],
                key=d.get("key", ""),
                value=d.get("value", ""),
                segment_id=seg_id,
            )
            count += 1
        except Exception as exc:
            logger.debug("[ledger_arch_extractor] Failed to write decision: %s", exc)

    if count > 0:
        try:
            save_ledger(ledger, job_dir)
        except Exception as exc:
            logger.warning("[ledger_arch_extractor] Failed to save ledger: %s", exc)

        if emit:
            emit(f"  📝 Recorded {count} architectural decision(s) to evidence ledger")
        logger.info(
            "[ledger_arch_extractor] Recorded %d decisions for %s", count, seg_id,
        )

    return count


# =============================================================================
# EXTRACTORS — deterministic regex, no LLM
# =============================================================================

def _extract_file_inventory(arch_text: str) -> List[Dict[str, str]]:
    """Extract file paths from File Inventory section."""
    results = []
    in_inventory = False

    for line in arch_text.split("\n"):
        stripped = line.strip()

        # Detect File Inventory section header
        if re.match(r"^#{1,4}\s*File\s+Inventory", stripped, re.IGNORECASE):
            in_inventory = True
            continue

        # Exit on next major header
        if in_inventory and re.match(r"^#{1,3}\s+[A-Z]", stripped):
            break

        if not in_inventory:
            continue

        # Match lines like: - `path/to/file.py` — CREATE or MODIFY
        m = re.match(
            r"^[-*]\s+`([^`]+)`\s*[—-]+\s*(CREATE|MODIFY|READ)",
            stripped, re.IGNORECASE,
        )
        if m:
            path, action = m.group(1), m.group(2).upper()
            results.append({
                "key": f"file:{path}",
                "value": action,
                "summary": f"File {action}: {path}",
            })
            continue

        # Also match: **`path/to/file.py`** (CREATE)
        m2 = re.match(
            r"^[-*]?\s*\*?\*?`([^`]+)`\*?\*?\s*\((CREATE|MODIFY|READ)",
            stripped, re.IGNORECASE,
        )
        if m2:
            path, action = m2.group(1), m2.group(2).upper()
            results.append({
                "key": f"file:{path}",
                "value": action,
                "summary": f"File {action}: {path}",
            })

    return results


def _extract_api_endpoints(arch_text: str) -> List[Dict[str, str]]:
    """Extract API endpoint definitions from architecture text."""
    results = []
    seen = set()

    # Match patterns like: GET /api/courses, POST /api/courses/{id}
    # Also: @router.get("/courses"), @app.post("/api/v1/courses")
    patterns = [
        # Explicit HTTP method + path
        re.compile(
            r"(GET|POST|PUT|DELETE|PATCH)\s+(/[a-zA-Z0-9_/{}\-]+)",
            re.IGNORECASE,
        ),
        # Decorator style
        re.compile(
            r"@(?:router|app)\.(get|post|put|delete|patch)\(\s*[\"']([^\"']+)[\"']",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        for m in pattern.finditer(arch_text):
            method = m.group(1).upper()
            path = m.group(2)
            key = f"endpoint:{method} {path}"
            if key not in seen:
                seen.add(key)
                results.append({
                    "key": key,
                    "value": f"{method} {path}",
                    "summary": f"API endpoint: {method} {path}",
                })

    return results


def _extract_models(arch_text: str) -> List[Dict[str, str]]:
    """Extract Pydantic/dataclass model definitions."""
    results = []
    seen = set()

    # Match: class CourseName(BaseModel): or class Foo(DataClass):
    pattern = re.compile(
        r"class\s+(\w+)\s*\(\s*(BaseModel|BaseSchema|DataClass|TypedDict|Enum)\s*\)",
    )

    for m in pattern.finditer(arch_text):
        name, base = m.group(1), m.group(2)
        key = f"model:{name}"
        if key not in seen:
            seen.add(key)
            results.append({
                "key": key,
                "value": f"{name}({base})",
                "summary": f"Model: {name} extends {base}",
            })

    return results


def _extract_imports(arch_text: str) -> List[Dict[str, str]]:
    """Extract cross-module import declarations."""
    results = []
    seen = set()

    # Match: from app.routers.courses import router as courses_router
    pattern = re.compile(
        r"from\s+(app\.[a-zA-Z0-9_.]+)\s+import\s+(\w+(?:\s+as\s+\w+)?)",
    )

    for m in pattern.finditer(arch_text):
        module, symbol = m.group(1), m.group(2)
        key = f"import:{module}.{symbol.split()[0]}"
        if key not in seen:
            seen.add(key)
            results.append({
                "key": key,
                "value": f"from {module} import {symbol}",
                "summary": f"Import: {symbol.split()[0]} from {module}",
            })

    return results
