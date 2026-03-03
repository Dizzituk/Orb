"""
Post-Implementation Validation Gate (Fix 5).

Runs after all segments are implemented, before cohesion checks.
Two deterministic checks — no LLM cost:

1. SYNTAX VALIDATION
   - TypeScript/TSX: runs `tsc --noEmit` in the sandbox
   - CSS: regex-based brace/rule validation
   - Catches junk arch-doc snippets prepended to files

2. CSS CLASS COHESION
   - Extracts className="..." from all TSX files
   - Extracts .class-name definitions from all CSS files
   - Flags any class used in TSX that has no CSS definition
   - Prevents invisible unstyled components

Both checks run via the sandbox bridge (files live in sandbox, not host).

v1.0 (2026-03-03): Initial implementation
"""

import re
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_CLASS_PREFIXES = (
    "education-", "course-", "investments-", "content-", "social-",
    "finance-", "lifestyle-", "builds-", "debug-", "settings-",
    "ch-",
)

_FRAMEWORK_CLASS_PATTERNS = re.compile(
    r"^(active|selected|disabled|hidden|visible|open|closed|"
    r"loading|error|success|warning|"
    r"flex|grid|block|inline|relative|absolute|"
    r"sr-only|container|wrapper|row|col|"
    r"btn|icon|text|link|badge|card|list|item)$"
)


# --- Sandbox Bridge Helpers ---------------------------------------------------

def _sandbox_read_files(
    paths: List[str], bridge_url: str = "http://192.168.250.2:8765",
) -> Dict[str, str]:
    """Read file contents from sandbox via bridge API."""
    import requests
    body = {"paths": paths}
    resp = requests.post(
        f"{bridge_url}/fs/contents", json=body, timeout=15,
    )
    data = resp.json()
    result = {}
    for f in data.get("files", []):
        if f.get("error"):
            logger.warning("[post_impl] Cannot read %s: %s", f["path"], f["error"])
            continue
        lines = []
        for line in f["content"].split("\n"):
            cleaned = re.sub(r"^\s*\d+:\s?", "", line)
            lines.append(cleaned)
        result[f["path"]] = "\n".join(lines)
    return result


def _sandbox_list_files(
    directory: str, pattern: str = "*",
    bridge_url: str = "http://192.168.250.2:8765",
) -> List[Dict[str, Any]]:
    """List files in a sandbox directory."""
    import requests
    cmd = (
        f"Get-ChildItem '{directory}' -File -Filter '{pattern}' "
        f"-Recurse -ErrorAction SilentlyContinue | "
        f"ForEach-Object {{ $_.FullName + '|' + $_.Length }}"
    )
    body = {"cmd": ["powershell", "-NoProfile", "-Command", cmd], "timeout_sec": 10}
    resp = requests.post(
        f"{bridge_url}/shell/run", json=body, timeout=15,
    )
    data = resp.json()
    files = []
    for line in (data.get("stdout", "") or "").strip().split("\n"):
        line = line.strip()
        if "|" in line:
            path, size = line.rsplit("|", 1)
            files.append({"path": path.strip(), "size": int(size.strip())})
    return files


def _sandbox_run_command(
    cmd: List[str], bridge_url: str = "http://192.168.250.2:8765",
    timeout: int = 30,
) -> Tuple[str, str, int]:
    """Run a command in the sandbox, return (stdout, stderr, exit_code)."""
    import requests
    body = {"cmd": cmd, "timeout_sec": timeout}
    resp = requests.post(
        f"{bridge_url}/shell/run", json=body, timeout=timeout + 5,
    )
    data = resp.json()
    return (
        data.get("stdout", "") or "",
        data.get("stderr", "") or "",
        data.get("exit_code", -1),
    )

# --- Check 1: Syntax Validation -----------------------------------------------

def check_typescript_syntax(
    project_dir: str, files: List[str],
) -> List[Dict[str, Any]]:
    """Run tsc --noEmit on specific files to catch syntax errors."""
    if not files:
        return []
    tsc_bin = os.path.join(project_dir, "node_modules", ".bin", "tsc.cmd")
    tsconfig = os.path.join(project_dir, "tsconfig.json")
    cmd = [
        "powershell", "-NoProfile", "-Command",
        f"& '{tsc_bin}' --noEmit --project '{tsconfig}' 2>&1 | Out-String",
    ]
    stdout, stderr, exit_code = _sandbox_run_command(cmd, timeout=60)
    issues = []
    for line in (stdout + "\n" + stderr).split("\n"):
        match = re.match(
            r"(.+?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)", line.strip(),
        )
        if match:
            filepath = match.group(1).strip()
            if any(filepath.replace("\\", "/").endswith(
                f.replace("\\", "/").split(project_dir.replace("\\", "/"))[-1]
            ) for f in files):
                issues.append({
                    "check": "typescript_syntax", "severity": "error",
                    "file": filepath, "line": int(match.group(2)),
                    "column": int(match.group(3)), "code": match.group(4),
                    "message": match.group(5).strip(),
                })
    logger.info("[post_impl] TypeScript syntax: %d error(s) in %d file(s)", len(issues), len(files))
    return issues


def check_css_syntax(content: str, filepath: str) -> List[Dict[str, Any]]:
    """Basic CSS syntax validation — brace balance + TSX contamination."""
    issues = []
    open_count = content.count("{")
    close_count = content.count("}")
    if open_count != close_count:
        issues.append({
            "check": "css_syntax", "severity": "error", "file": filepath, "line": 0,
            "message": f"Unmatched braces: {open_count} opening vs {close_count} closing",
        })
    tsx_patterns = [
        (r"^\s*import\s+\{", "import statement (TypeScript code in CSS)"),
        (r"^\s*export\s+(function|const|interface)", "export statement (TypeScript code in CSS)"),
        (r"^\s*interface\s+\w+Props", "TypeScript interface in CSS"),
        (r"<\w+\s+className=", "JSX element in CSS"),
        (r"^\s*const\s+\w+\s*[:=]", "const declaration (TypeScript code in CSS)"),
    ]
    for line_num, line in enumerate(content.split("\n"), 1):
        for pattern, description in tsx_patterns:
            if re.match(pattern, line):
                issues.append({
                    "check": "css_syntax", "severity": "error",
                    "file": filepath, "line": line_num, "message": description,
                })
    return issues


def check_tsx_preamble(content: str, filepath: str) -> List[Dict[str, Any]]:
    """Detect arch-doc junk prepended to TSX files."""
    issues = []
    lines = content.split("\n")
    first_import = -1
    first_content = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if first_content == -1:
            first_content = i
        if stripped.startswith("import ") or stripped.startswith("import{"):
            first_import = i
            break
    if first_import > 0 and first_content < first_import:
        pre_import = "\n".join(lines[first_content:first_import])
        if re.search(r"interface\s+\w+", pre_import):
            issues.append({
                "check": "tsx_preamble", "severity": "error", "file": filepath,
                "line": first_content + 1,
                "message": (
                    f"Interface declaration before imports (lines "
                    f"{first_content + 1}-{first_import}). Arch-doc snippet leaked."
                ),
            })
        if re.search(r"\[.*\u00d7.*\]|\[.*x\s+\d+\]", pre_import):
            issues.append({
                "check": "tsx_preamble", "severity": "error", "file": filepath,
                "line": first_content + 1,
                "message": "JSX pseudo-code before imports. Arch-doc outline leaked.",
            })
    interface_names = re.findall(r"(?:export\s+)?interface\s+(\w+)", content)
    seen = {}
    for name in interface_names:
        if name in seen:
            issues.append({
                "check": "tsx_preamble", "severity": "error", "file": filepath,
                "line": 0, "message": f"Duplicate interface declaration: '{name}'",
            })
        seen[name] = True
    export_lines = re.findall(
        r"^(export\s+\{[^}]+\}\s+from\s+['\"][^'\"]+['\"];?\s*)$",
        content, re.MULTILINE,
    )
    export_seen = {}
    for exp in export_lines:
        normalised = re.sub(r"\s+", " ", exp.strip())
        if normalised in export_seen:
            issues.append({
                "check": "tsx_preamble", "severity": "error", "file": filepath,
                "line": 0, "message": f"Duplicate export: '{normalised}'",
            })
        export_seen[normalised] = True
    return issues

# --- Check 2: CSS Class Cohesion ---------------------------------------------

def extract_tsx_class_names(content: str) -> List[str]:
    """Extract all className values from TSX content."""
    classes = set()
    for match in re.finditer(r'className="([^"]+)"', content):
        for cls in match.group(1).split():
            classes.add(cls)
    for match in re.finditer(r"className=\{[`'\"]([^}]+)\}", content):
        raw = match.group(1)
        for part in re.findall(r"([\w-]+)", raw):
            if any(part.startswith(p) for p in _PROJECT_CLASS_PREFIXES):
                classes.add(part)
    for match in re.finditer(r'className=\{`([^`]+)`\}', content):
        raw = match.group(1)
        for part in re.findall(r"([\w][\w-]*)", raw):
            if any(part.startswith(p) for p in _PROJECT_CLASS_PREFIXES):
                classes.add(part)
    return sorted(classes)


def extract_css_class_names(content: str) -> List[str]:
    """Extract all class selectors from CSS content."""
    classes = set()
    for match in re.finditer(r"\.([\w][\w-]*)", content):
        cls = match.group(1)
        if not cls[0].isdigit():
            classes.add(cls)
    return sorted(classes)


def check_css_class_cohesion(
    tsx_files: Dict[str, str], css_files: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Cross-reference className usage in TSX against CSS definitions."""
    all_css_classes = set()
    for css_path, css_content in css_files.items():
        all_css_classes.update(extract_css_class_names(css_content))
    issues = []
    for tsx_path, tsx_content in tsx_files.items():
        tsx_classes = extract_tsx_class_names(tsx_content)
        for cls in tsx_classes:
            if not any(cls.startswith(p) for p in _PROJECT_CLASS_PREFIXES):
                continue
            if _FRAMEWORK_CLASS_PATTERNS.match(cls):
                continue
            if cls not in all_css_classes:
                issues.append({
                    "check": "css_class_cohesion", "severity": "warning",
                    "file": tsx_path, "class_name": cls,
                    "message": f"Class '{cls}' used in component but not defined in any CSS file",
                })
    logger.info("[post_impl] CSS class cohesion: %d unmatched across %d TSX file(s)", len(issues), len(tsx_files))
    return issues

# --- Main Entry Point ---------------------------------------------------------

def run_post_implementation_validation(
    job_dir: str, frontend_dir: str = r"D:\orb-desktop", on_progress=None,
) -> Dict[str, Any]:
    """Run all post-implementation validation checks."""
    emit = on_progress or (lambda msg: None)
    all_issues = []
    emit("\n\U0001f50d Running post-implementation validation...")

    tsx_file_list = _sandbox_list_files(os.path.join(frontend_dir, "src", "components"), "*.tsx")
    ts_file_list = _sandbox_list_files(os.path.join(frontend_dir, "src", "components"), "*.ts")
    css_file_list = _sandbox_list_files(os.path.join(frontend_dir, "src", "styles", "components"), "*.css")

    all_paths = (
        [f["path"] for f in tsx_file_list]
        + [f["path"] for f in ts_file_list]
        + [f["path"] for f in css_file_list]
    )
    if not all_paths:
        emit("  \u26a0\ufe0f No implementation files found in sandbox")
        return {"passed": True, "issues": [], "summary": {}}

    file_contents = _sandbox_read_files(all_paths)
    tsx_files = {p: c for p, c in file_contents.items() if p.endswith(".tsx")}
    ts_files = {p: c for p, c in file_contents.items() if p.endswith(".ts") and not p.endswith(".d.ts")}
    css_files = {p: c for p, c in file_contents.items() if p.endswith(".css")}

    # Check 1a: TSX/TS preamble junk
    preamble_count = 0
    for filepath, content in {**tsx_files, **ts_files}.items():
        preamble_issues = check_tsx_preamble(content, filepath)
        if preamble_issues:
            preamble_count += len(preamble_issues)
            all_issues.extend(preamble_issues)
    if preamble_count:
        emit(f"  \u274c Preamble check: {preamble_count} file(s) have arch-doc junk")
    else:
        emit("  \u2705 Preamble check: clean")

    # Check 1b: CSS syntax
    css_syntax_count = 0
    for filepath, content in css_files.items():
        css_issues = check_css_syntax(content, filepath)
        if css_issues:
            css_syntax_count += len(css_issues)
            all_issues.extend(css_issues)
    if css_syntax_count:
        emit(f"  \u274c CSS syntax: {css_syntax_count} issue(s)")
    else:
        emit(f"  \u2705 CSS syntax: {len(css_files)} file(s) clean")

    # Check 1c: TypeScript syntax (via tsc in sandbox)
    ts_tsx_paths = [f["path"] for f in tsx_file_list + ts_file_list]
    if ts_tsx_paths:
        try:
            ts_issues = check_typescript_syntax(frontend_dir, ts_tsx_paths)
            if ts_issues:
                emit(f"  \u274c TypeScript: {len(ts_issues)} error(s)")
                all_issues.extend(ts_issues)
            else:
                emit(f"  \u2705 TypeScript: {len(ts_tsx_paths)} file(s) clean")
        except Exception as e:
            logger.warning("[post_impl] tsc check failed: %s", e)
            emit(f"  \u26a0\ufe0f TypeScript check skipped: {e}")

    # Check 2: CSS class cohesion
    css_cohesion_issues = check_css_class_cohesion(tsx_files, css_files)
    if css_cohesion_issues:
        by_file = {}
        for issue in css_cohesion_issues:
            by_file.setdefault(issue["file"], []).append(issue["class_name"])
        for filepath, classes in by_file.items():
            short = filepath.split("\\")[-1] if "\\" in filepath else filepath
            emit(f"  \u274c CSS cohesion: {short} uses {len(classes)} undefined class(es)")
        all_issues.extend(css_cohesion_issues)
    else:
        emit("  \u2705 CSS class cohesion: all classes defined")

    # Summary
    syntax_errors = sum(1 for i in all_issues if i["check"] in ("typescript_syntax", "css_syntax"))
    preamble_errors = sum(1 for i in all_issues if i["check"] == "tsx_preamble")
    css_mismatches = sum(1 for i in all_issues if i["check"] == "css_class_cohesion")
    blocking = syntax_errors + preamble_errors

    passed = blocking == 0 and css_mismatches == 0
    summary = {
        "syntax_errors": syntax_errors, "preamble_errors": preamble_errors,
        "css_mismatches": css_mismatches, "total_issues": len(all_issues),
        "blocking": blocking, "passed": passed,
    }
    if passed:
        emit("  \u2705 Post-implementation validation PASSED")
    else:
        emit(f"  \u274c Post-implementation validation FAILED: {blocking} blocking, {css_mismatches} CSS mismatch(es)")
    return {"passed": passed, "issues": all_issues, "summary": summary}

def format_issues_for_regen(
    issues: List[Dict[str, Any]], segment_files: List[str],
) -> str:
    """Format validation issues as feedback text for segment re-generation."""
    relevant = []
    for issue in issues:
        issue_file = issue.get("file", "").replace("\\", "/")
        for seg_file in segment_files:
            if seg_file.replace("\\", "/") in issue_file:
                relevant.append(issue)
                break
    if not relevant:
        return ""
    parts = ["## Post-Implementation Validation Errors\n"]
    parts.append("The following issues were found in the implemented files:\n")
    for issue in relevant:
        check = issue["check"]
        msg = issue["message"]
        filepath = issue.get("file", "unknown")
        short = filepath.split("\\")[-1] if "\\" in filepath else filepath
        if check == "css_class_cohesion":
            cls = issue.get("class_name", "")
            parts.append(f"- **CSS Mismatch**: `{cls}` used in `{short}` but not defined in any CSS file")
        elif check == "tsx_preamble":
            parts.append(f"- **Preamble Error**: {msg} in `{short}`")
        elif check == "typescript_syntax":
            line = issue.get("line", 0)
            code = issue.get("code", "")
            parts.append(f"- **TS Error** {code} at `{short}:{line}`: {msg}")
        elif check == "css_syntax":
            parts.append(f"- **CSS Error**: {msg} in `{short}`")
    parts.append(
        "\nFix all issues above. Ensure class names in components "
        "match exactly what is defined in the CSS files."
    )
    return "\n".join(parts)