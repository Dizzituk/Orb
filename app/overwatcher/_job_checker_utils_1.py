from typing import Dict, List, Optional


JOB_CHECKER_BUILD_ID = "2026-02-17-v2.4-re-export-and-forward-ref-tolerance"

JOB_CHECKER_SYSTEM_PROMPT = """\
You are a post-write code verifier. You have just been given:
1. A file that was just written to disk
2. The architecture specification for that file
3. Optionally, an interface contract listing what this file MUST expose

Your job is to verify the written file matches the specification. Check:

1. EXPORTS: Every class, function, constant, or endpoint listed in the \
architecture spec actually exists in the file with the correct name.

2. SIGNATURES: Function/method signatures match the spec — parameter names, \
types, and return types are correct.

3. IMPORTS: The file's imports reference modules/packages that should exist \
(based on the architecture). Flag imports to clearly non-existent local modules.

4. CONTRACT COMPLIANCE: If an interface contract is provided, every "MUST EXPOSE" \
boundary exists with the exact name, signature, and return type specified. \
A symbol counts as "exported" if it is importable from this file — this includes \
both locally defined functions AND re-exports via `from .other_module import symbol`. \
Do NOT reject a file for re-exporting a symbol that was defined in another module \
within the same package. Re-exporting is a valid and common Python pattern. \
ALSO: if a file imports a symbol from a sibling module AND defines it locally, \
the local definition takes precedence (Python shadowing). This is NOT a conflict — \
the local definition IS the canonical export. Do NOT flag "imports AND defines" as \
a contract violation or namespace collision. The import may be there as a fallback, \
or the code may have been refactored. As long as the symbol IS defined/importable \
from this file, the contract is satisfied. \
Similarly, if a file imports a symbol and re-exports it (without local definition), \
that ALSO satisfies the contract — the symbol is importable from this file. \
NEVER use the words "ambiguity", "dual presence", "namespace collision", or \
"contradicts" when a symbol is importable from the file. The ONLY question is: \
can downstream code do `from this_file import symbol` successfully? If yes, PASS.

5. COMPLETENESS: No TODO, FIXME, NotImplementedError, or "pass" placeholders \
in critical paths. Stub implementations are acceptable ONLY for genuinely \
optional features.

RULES:
- Focus on integration-critical issues that would cause OTHER files to break.
- Don't nitpick style, formatting, or internal implementation details.
- Severity "blocking" = would cause import errors, type errors, or runtime \
failures in OTHER files. Severity "warning" = might cause issues, worth noting.
- Be precise. Quote the exact name/signature that's wrong.
- Forward reference strings in type hints (e.g. `param: "ClassName"`) are valid Python \
and do NOT require the class to be imported at module level. Using `from __future__ import \
annotations` or string annotations avoids circular imports. Do NOT flag forward references \
as blocking issues unless you are certain the referenced type does not exist anywhere.
- If a type annotation references a class from a verified import (in the VERIFIED list), \
do NOT flag the import as missing regardless of whether quotes are used.

OUTPUT FORMAT:
Return ONLY a JSON object:
{
  "passed": true/false,
  "issues": [
    {
      "severity": "blocking" | "warning",
      "category": "missing_export" | "wrong_signature" | "import_error" | \
"missing_implementation" | "contract_violation" | "naming_mismatch",
      "description": "ExactClassName.method_name has wrong return type: \
expected Dict[str, Any] but found None",
      "line_hint": "near line 45"
    }
  ],
  "reasoning": "Brief summary of check"
}

passed = false if ANY blocking issues exist, true otherwise.
If the file looks correct, return {"passed": true, "issues": [], "reasoning": "..."}.
"""

SKIP_PATTERNS = [
    r'__init__\.py$',      # Usually just imports/re-exports
    r'\.env',              # Config files
    r'\.json$',            # Data/config
    r'\.yaml$',
    r'\.yml$',
    r'\.toml$',
    r'\.cfg$',
    r'\.ini$',
    r'\.md$',              # Documentation
    r'\.txt$',
    r'\.gitignore$',
    r'\.dockerignore$',
    r'requirements\.txt$',
    r'Dockerfile$',
]

MIN_CHECK_CHARS = 100

MAX_CHECK_CHARS = 15000

def _build_import_evidence(import_results: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Build a prompt section that tells the LLM which imports are
    filesystem-verified so it does NOT flag them.
    """
    verified = import_results.get("verified", [])
    unresolvable = import_results.get("unresolvable", [])

    if not verified and not unresolvable:
        return ""

    lines = []
    lines.append("\n## Deterministic Import Verification (GROUND TRUTH — do NOT override)")
    lines.append("The following imports have been checked against the actual filesystem.")
    lines.append("DO NOT flag verified imports as errors. They are confirmed correct.\n")

    if verified:
        lines.append("### ✅ VERIFIED (exist on disk — do NOT flag):")
        for v in verified:
            lines.append(f"- `{v['import_ref']}` → `{v['resolved_path']}`")
        lines.append("")

    if unresolvable:
        lines.append("### ❌ NOT FOUND on disk:")
        lines.append("These modules were not found on the filesystem at check time.")
        lines.append("**IMPORTANT**: In segmented execution, later segments' files may")
        lines.append("not exist on disk yet. If the architecture spec explicitly")
        lines.append("prescribes a deferred/local import from a module that will be")
        lines.append("created by a later segment, this is EXPECTED and should be a")
        lines.append("WARNING, not BLOCKING. Only flag as blocking if the import")
        lines.append("references a module name that doesn't appear anywhere in the")
        lines.append("architecture spec or skeleton contract.\n")
        for u in unresolvable:
            lines.append(f"- `{u['import_ref']}` → {u.get('reason', 'not found')}")
        lines.append("")

    return "\n".join(lines)

def _build_check_prompt(
    file_path: str,
    file_content: str,
    arch_section: str,
    interface_contract: str = "",
    import_evidence: str = "",
    previous_strike_errors: Optional[List[str]] = None,
) -> str:
    """Build user prompt for the job checker."""
    # Trim file content if huge
    _content = file_content
    if len(_content) > MAX_CHECK_CHARS:
        _half = MAX_CHECK_CHARS // 2 - 100
        _content = (
            _content[:_half]
            + f"\n\n... ({len(file_content) - MAX_CHECK_CHARS} chars trimmed) ...\n\n"
            + _content[-_half:]
        )

    contract_section = ""
    if interface_contract and interface_contract.strip():
        contract_section = f"""

## Interface Contract
{interface_contract}
"""

    # v2.2: Strike history — show previous checker feedback to prevent contradictions
    strike_history_section = ""
    if previous_strike_errors:
        strike_lines = []
        for i, err in enumerate(previous_strike_errors, 1):
            strike_lines.append(f"- **Strike {i}**: {err}")
        strike_history_section = f"""

## ⚠️ PREVIOUS STRIKE HISTORY (CRITICAL — READ CAREFULLY)
This file has been rejected {len(previous_strike_errors)} time(s) already.
The Implementer rewrote the file after each rejection based on YOUR feedback.

{chr(10).join(strike_lines)}

**IMPORTANT**: If you see that previous strikes gave CONTRADICTORY feedback
(e.g. strike 1 said "make it async" and strike 2 said "make it sync"), then
the spec itself has an ambiguity. In this case you MUST:
1. Accept the current implementation if it is functionally correct
2. Downgrade the contradicted issue from "blocking" to "warning"
3. Do NOT re-raise an issue that contradicts feedback from a previous strike

The goal is forward progress, not perfection. Only flag issues that are
genuinely broken (will cause ImportError, NameError, or logic bugs at runtime).
"""

    return f"""\
## File Path
`{file_path}`

## Architecture Specification For This File
{arch_section}
{contract_section}{import_evidence}{strike_history_section}
## Written File Content
```
{_content}
```

Verify the written file against the architecture spec and contract. Return ONLY the JSON verdict.
"""
