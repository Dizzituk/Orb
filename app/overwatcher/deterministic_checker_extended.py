# FILE: app/overwatcher/deterministic_checker_extended.py
"""
Extended Deterministic Checker — Additional post-write checks.

Supplements the core deterministic_checker with checks that eliminate
the need for the LLM job checker fallback.

Checks:
8.  PROHIBITED DEFINITIONS — File doesn't redefine symbols owned by
    other segments (from skeleton contract JSON or markdown).
9.  ARCHITECTURE FILE INVENTORY — Every file listed in the architecture
    was actually produced by the implementer.
10. THIRD-PARTY IMPORT VALIDATION — Imports of third-party packages
    reference known/allowed packages, not hallucinated ones.
11. CLASS COMPLETENESS — Classes declared in architecture have all
    expected methods implemented (not just the class definition).

v1.1 (2026-02-27): Fix — parse skeleton contract JSON, not just markdown.
v1.0 (2026-02-27): Initial implementation — Stage 3 of deterministic
verification migration.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

EXTENDED_CHECKER_BUILD_ID = "2026-02-27-v1.1-extended-det-checker"

# Known standard library modules (subset — covers common ones)
_STDLIB_MODULES: Set[str] = {
    "abc", "argparse", "ast", "asyncio", "base64", "bisect",
    "collections", "concurrent", "contextlib", "copy", "csv",
    "dataclasses", "datetime", "decimal", "difflib", "email",
    "enum", "errno", "fnmatch", "fractions", "functools",
    "getpass", "glob", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "importlib", "inspect", "io", "itertools",
    "json", "keyword", "linecache", "locale", "logging",
    "math", "mimetypes", "multiprocessing", "operator", "os",
    "pathlib", "pdb", "pickle", "platform", "pprint",
    "queue", "random", "re", "shlex", "shutil", "signal",
    "site", "socket", "sqlite3", "ssl", "stat", "statistics",
    "string", "struct", "subprocess", "sys", "tempfile",
    "textwrap", "threading", "time", "timeit", "token",
    "tokenize", "traceback", "types", "typing", "unittest",
    "urllib", "uuid", "venv", "warnings", "weakref", "xml",
    "zipfile", "zlib",
    "typing_extensions", "__future__",
}

# Known third-party packages in the ASTRA ecosystem
_KNOWN_THIRD_PARTY: Set[str] = {
    "fastapi", "uvicorn", "pydantic", "starlette", "httpx",
    "aiohttp", "aiofiles", "sqlalchemy", "alembic",
    "openai", "anthropic", "google", "tiktoken", "tokenizers",
    "numpy", "pandas", "scipy", "sklearn", "torch",
    "PIL", "pillow", "requests", "beautifulsoup4", "bs4",
    "lxml", "yaml", "pyyaml", "toml", "tomli", "tomllib",
    "dotenv", "python_dotenv", "celery", "redis", "boto3",
    "jinja2", "markupsafe", "click", "rich", "colorama",
    "pytest", "coverage", "mypy", "black", "isort", "ruff",
    "websockets", "sse_starlette", "python_multipart",
    "cryptography", "passlib", "jose", "jwt", "bcrypt",
    "apscheduler", "schedule", "watchdog", "psutil",
    "pygments", "markdown", "mdformat",
    "sentence_transformers", "chromadb", "qdrant_client",
    "faster_whisper", "whisper", "gtts", "pyttsx3",
    "openpyxl", "xlsxwriter", "reportlab", "fpdf",
    "playwright", "selenium", "scrapy",
}


# =========================================================================
# Shared helpers
# =========================================================================

class ExtendedCheckIssue:
    """A single extended check issue."""

    def __init__(
        self,
        severity: str,
        category: str,
        description: str,
        line_hint: Optional[str] = None,
    ):
        self.severity = severity
        self.category = category
        self.description = description
        self.line_hint = line_hint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "line_hint": self.line_hint,
        }


# =========================================================================
# CHECK 8: Prohibited Definitions
# =========================================================================

def check_prohibited_definitions(
    file_content: str,
    file_path: str,
    interface_contract: str,
    segment_id: Optional[str] = None,
) -> List[ExtendedCheckIssue]:
    """
    Verify file doesn't redefine symbols owned by other segments.

    The interface_contract can be either:
    - Structured JSON (skeleton contract with "skeletons" array)
    - Markdown text with "DO NOT DEFINE" prose

    For JSON contracts, builds an ownership map from exports of other
    segments and checks this file doesn't redefine those symbols.
    """
    issues: List[ExtendedCheckIssue] = []

    if not interface_contract:
        return issues

    # Build set of symbols owned by OTHER segments
    prohibited: Set[str] = set()

    # Try JSON first (skeleton_contract.json format)
    try:
        contract = json.loads(interface_contract)
        if isinstance(contract, dict) and "skeletons" in contract:
            for skel in contract.get("skeletons", []):
                skel_seg = skel.get("segment_id", "")
                if segment_id and skel_seg == segment_id:
                    continue  # Skip own segment
                for export in skel.get("exports", []):
                    names = export.get("names", [])
                    if isinstance(names, list):
                        prohibited.update(n for n in names if isinstance(n, str))
    except (json.JSONDecodeError, TypeError, ValueError):
        # Not JSON — try markdown "DO NOT DEFINE" pattern
        for m in re.finditer(
            r'(?:DO NOT|MUST NOT|NEVER)\s+(?:DEFINE|IMPLEMENT|REDEFINE)'
            r'[:\s]+([^\n]+)',
            interface_contract,
            re.IGNORECASE,
        ):
            names_str = m.group(1).strip()
            for name in re.findall(r'`(\w+)`', names_str):
                prohibited.add(name)
            for name in re.split(r'[,;]\s*', names_str):
                name = name.strip().strip('`').strip()
                if re.match(r'^[a-zA-Z_]\w*$', name):
                    prohibited.add(name)

    if not prohibited:
        return issues

    # Parse file to find top-level definitions
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return issues  # Syntax check happens elsewhere

    defined: Set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)

    violations = defined & prohibited
    for name in sorted(violations):
        issues.append(ExtendedCheckIssue(
            severity="blocking",
            category="prohibited_definition",
            description=(
                f"'{name}' is defined in {file_path} but is "
                f"owned by another segment in the skeleton contract."
            ),
        ))

    return issues


# =========================================================================
# CHECK 9: Architecture File Inventory Match
# =========================================================================

def check_file_inventory_match(
    architecture_content: str,
    produced_files: Set[str],
    segment_file_scope: Optional[List[str]] = None,
) -> List[ExtendedCheckIssue]:
    """
    Every file in the architecture's File Inventory should have been
    produced by the implementer.
    """
    issues: List[ExtendedCheckIssue] = []

    if not architecture_content:
        return issues

    # Extract File Inventory from architecture
    arch_files: Set[str] = set()
    inv_match = re.search(r'(?:^|\n)#+\s*File Inventory', architecture_content)
    if inv_match:
        inv_start = inv_match.start()
        inv_end = re.search(r'\n(?:##[^#]|---)', architecture_content[inv_start + 20:])
        section = (
            architecture_content[inv_start:inv_start + 20 + inv_end.start()]
            if inv_end
            else architecture_content[inv_start:inv_start + 3000]
        )
        for line in section.split("\n"):
            m = re.search(r'`([\w/\\._-]+\.py)`', line)
            if m:
                arch_files.add(m.group(1).replace("\\", "/").lower())

    if not arch_files:
        return issues

    produced_norm = {f.replace("\\", "/").lower() for f in produced_files}

    for arch_file in sorted(arch_files):
        found = False
        for prod in produced_norm:
            if prod.endswith(arch_file) or arch_file.endswith(prod):
                found = True
                break
            if arch_file.rsplit("/", 1)[-1] == prod.rsplit("/", 1)[-1]:
                found = True
                break
        if not found:
            issues.append(ExtendedCheckIssue(
                severity="warning",
                category="missing_arch_file",
                description=(
                    f"Architecture lists '{arch_file}' in File Inventory "
                    f"but it was not produced by the implementer."
                ),
            ))

    return issues


# =========================================================================
# CHECK 10: Third-Party Import Validation
# =========================================================================

def check_third_party_imports(
    file_content: str,
    file_path: str,
) -> List[ExtendedCheckIssue]:
    """
    Flag imports of packages that aren't stdlib, known third-party,
    or local project imports (app.*).
    """
    issues: List[ExtendedCheckIssue] = []

    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return issues

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _STDLIB_MODULES and root not in _KNOWN_THIRD_PARTY:
                    if root != "app" and not root.startswith("_"):
                        issues.append(ExtendedCheckIssue(
                            severity="warning",
                            category="unknown_import",
                            description=(
                                f"Unknown package '{root}' imported in "
                                f"{file_path} (line {node.lineno}). "
                                f"Verify this is an installed dependency."
                            ),
                            line_hint=f"line {node.lineno}",
                        ))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            if root not in _STDLIB_MODULES and root not in _KNOWN_THIRD_PARTY:
                if root != "app" and not root.startswith("_"):
                    issues.append(ExtendedCheckIssue(
                        severity="warning",
                        category="unknown_import",
                        description=(
                            f"Unknown package '{root}' imported in "
                            f"{file_path} (line {node.lineno}). "
                            f"Verify this is an installed dependency."
                        ),
                        line_hint=f"line {node.lineno}",
                    ))

    return issues


# =========================================================================
# CHECK 11: Class Method Completeness
# =========================================================================

def check_class_completeness(
    file_content: str,
    file_path: str,
    architecture_content: str,
) -> List[ExtendedCheckIssue]:
    """
    Classes declared in architecture have all expected methods
    implemented (not just an empty class or only __init__).
    """
    issues: List[ExtendedCheckIssue] = []

    if not architecture_content:
        return issues

    arch_classes: Dict[str, Set[str]] = {}
    current_class = None
    for line in architecture_content.split("\n"):
        cls_m = re.match(r'\s*class\s+(\w+)', line)
        if cls_m:
            current_class = cls_m.group(1)
            arch_classes[current_class] = set()
        elif current_class:
            meth_m = re.match(r'\s+(?:async\s+)?def\s+(\w+)', line)
            if meth_m:
                mname = meth_m.group(1)
                if not mname.startswith("__") or mname == "__init__":
                    arch_classes[current_class].add(mname)
            elif re.match(r'\S', line) and not line.strip().startswith("#"):
                current_class = None

    if not arch_classes:
        return issues

    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return issues

    impl_classes: Dict[str, Set[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods: Set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
            impl_classes[node.name] = methods

    for cls_name, expected_methods in arch_classes.items():
        if cls_name not in impl_classes:
            continue
        impl_methods = impl_classes[cls_name]
        missing = expected_methods - impl_methods
        if missing:
            issues.append(ExtendedCheckIssue(
                severity="warning",
                category="incomplete_class",
                description=(
                    f"Class '{cls_name}' in {file_path} is missing "
                    f"methods from architecture: {', '.join(sorted(missing))}"
                ),
            ))

    return issues


# =========================================================================
# ORCHESTRATOR
# =========================================================================

def run_extended_det_checks(
    file_content: str,
    file_path: str,
    interface_contract: str = "",
    architecture_content: str = "",
    segment_id: Optional[str] = None,
    produced_files: Optional[Set[str]] = None,
    segment_file_scope: Optional[List[str]] = None,
) -> List[ExtendedCheckIssue]:
    """
    Run all extended deterministic checks.

    Returns list of issues. Empty list = all checks passed.
    """
    all_issues: List[ExtendedCheckIssue] = []

    try:
        all_issues.extend(check_prohibited_definitions(
            file_content, file_path, interface_contract, segment_id,
        ))
    except Exception as e:
        logger.debug("[det_ext] Check 8 (prohibited defs) error: %s", e)

    if produced_files is not None:
        try:
            all_issues.extend(check_file_inventory_match(
                architecture_content, produced_files, segment_file_scope,
            ))
        except Exception as e:
            logger.debug("[det_ext] Check 9 (file inventory) error: %s", e)

    try:
        all_issues.extend(check_third_party_imports(
            file_content, file_path,
        ))
    except Exception as e:
        logger.debug("[det_ext] Check 10 (third-party imports) error: %s", e)

    try:
        all_issues.extend(check_class_completeness(
            file_content, file_path, architecture_content,
        ))
    except Exception as e:
        logger.debug("[det_ext] Check 11 (class completeness) error: %s", e)

    if all_issues:
        blocking = sum(1 for i in all_issues if i.severity == "blocking")
        logger.info(
            "[det_ext] Extended checks for %s: %d issues (%d blocking)",
            file_path, len(all_issues), blocking,
        )

    return all_issues


__all__ = [
    "ExtendedCheckIssue",
    "check_prohibited_definitions",
    "check_file_inventory_match",
    "check_third_party_imports",
    "check_class_completeness",
    "run_extended_det_checks",
    "EXTENDED_CHECKER_BUILD_ID",
]
