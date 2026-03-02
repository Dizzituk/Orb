"""
Prompt construction for Implementer LLM calls.

Handles contract signature extraction/injection, architecture context
stripping, verbatim code extraction, and prompt assembly for both
CREATE and MODIFY actions.

Extracted from orchestrator.py monolith (v5.23 contract logic + prompt building).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from .execution_state import ExecutionContext
from .constants import MODIFY_EDIT_MODE_THRESHOLD
from .parsing import extract_section_for_file, _extract_verbatim_code_from_architecture
from .context import _extract_existing_imports
from .prompts import (
    IMPLEMENTER_NEW_FILE_SYSTEM,
    IMPLEMENTER_MODIFY_FILE_SYSTEM,
    IMPLEMENTER_MODIFY_EDIT_SYSTEM,
    IMPLEMENTER_VERIFY_FILE_SYSTEM,
)
from .path_resolution import _infer_lang_from_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract signature extraction  (v5.23)
# ---------------------------------------------------------------------------

def extract_contract_block(
    interface_contract: str,
    rel_path: str,
) -> Tuple[str, List[str], List[str]]:
    """Extract mandatory contract signatures for a specific file.

    Returns (contract_block_text, file_contract_sigs, bare_export_names).
    contract_block_text is empty string if no signatures found.
    """
    file_contract_sigs: List[str] = []
    bare_export_names: List[str] = []

    try:
        from app.overwatcher.signature_checker import (
            extract_contract_signatures_for_file as _extract_sigs,
        )
        file_contract_sigs = _extract_sigs(interface_contract, rel_path)
    except ImportError:
        logger.debug("[arch_exec] v5.23 signature_checker not available for contract injection")
        return "", [], []
    except Exception as ci_err:
        logger.warning("[arch_exec] v5.23 Contract injection failed (non-fatal): %s", ci_err)
        return "", [], []

    # v5.23b: Also extract bare export names (no def prefix)
    try:
        bare_export_names = _extract_bare_export_names(interface_contract, rel_path)
    except Exception:
        pass  # Best-effort

    if not file_contract_sigs and not bare_export_names:
        logger.debug("[arch_exec] v5.23 No contract exports for %s", rel_path)
        return "", [], []

    # Build the contract block
    block = _format_contract_block(file_contract_sigs, bare_export_names)

    logger.info(
        "[arch_exec] v5.23 CONTRACT_INJECT for %s: %d signature(s) + %d bare name(s)",
        rel_path, len(file_contract_sigs), len(bare_export_names),
    )
    print(
        f"[ARCH_EXEC] v5.23 CONTRACT_INJECT: {rel_path} — "
        f"{len(file_contract_sigs)} sig(s), {len(bare_export_names)} bare name(s)"
    )
    return block, file_contract_sigs, bare_export_names


def _extract_bare_export_names(
    interface_contract: str,
    rel_path: str,
) -> List[str]:
    """Parse bare export names (non-def) from the interface contract."""
    bare_names: List[str] = []
    lines = interface_contract.split("\n")
    in_file = False
    in_exports = False
    file_norm = rel_path.replace("\\", "/").strip()

    for cl in lines:
        cs = cl.strip()
        indent = len(cl) - len(cl.lstrip())
        cs_norm = cs.replace("\\", "/")

        if f"`{file_norm}`" in cs_norm and indent <= 4:
            in_file = True
            in_exports = False
            continue
        if in_file:
            if "MUST" in cs and "EXPORT" in cs:
                in_exports = True
                continue
            if cs.startswith("###") or cs.startswith("## "):
                in_file = False
                in_exports = False
                continue
            if indent <= 4 and cs.startswith("- `"):
                cm = re.match(r'^-\s*`([^`]+)`', cs)
                if cm:
                    cv = cm.group(1).strip().replace("\\", "/")
                    is_fp = ("/" in cv or cv.endswith(".py"))
                    if is_fp and cv != file_norm:
                        in_file = False
                        in_exports = False
                        continue
            if in_exports and indent >= 4 and cs.startswith("- `"):
                cm = re.match(r'^-\s*`([^`]+)`', cs)
                if cm:
                    cv = cm.group(1).strip()
                    if not (cv.startswith("def ") or cv.startswith("async def ")):
                        if "/" not in cv and not cv.endswith(".py"):
                            bare_names.append(cv)
    return bare_names


def _format_contract_block(
    sigs: List[str],
    bare_names: List[str],
) -> str:
    """Format the MANDATORY CONTRACT block for the Implementer prompt."""
    lines = [
        "## MANDATORY CONTRACT — COPY THESE SIGNATURES EXACTLY",
        "",
        "A deterministic checker will REJECT your output if any signature deviates.",
        "Copy each `def` line below character-for-character. "
        "Do not rename parameters, change types, or reorder arguments.",
        "",
    ]
    if sigs:
        for sig in sigs:
            lines.extend([
                "```python",
                f"{sig}:",
                '    """Implementation required."""',
                "    ...",
                "```",
                "",
            ])
    if bare_names:
        lines.append("Also REQUIRED (must be defined/importable from this file):")
        for bn in bare_names:
            lines.append(f"  - `{bn}`")
        lines.append("")
    lines.extend([
        "DO NOT rename functions or parameters. DO NOT change types. "
        "DO NOT add or remove parameters.",
        "The architecture section below may describe these functions in prose "
        "— when there is ANY conflict, this MANDATORY CONTRACT wins.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Architecture context stripping  (v5.23c)
# ---------------------------------------------------------------------------

def strip_competing_signatures(
    file_context: str,
    contract_sigs: List[str],
) -> str:
    """Remove function signature lines from architecture context
    that compete with the mandatory contract block.
    """
    if not contract_sigs:
        return file_context

    try:
        func_names = set()
        for csig in contract_sigs:
            m = re.match(r'^(?:async\s+)?def\s+(\w+)\s*\(', csig)
            if m:
                func_names.add(m.group(1))

        if not func_names:
            return file_context

        cleaned = []
        for line in file_context.split('\n'):
            skip = False
            for fn in func_names:
                if re.match(rf'^\s*(?:async\s+)?def\s+{re.escape(fn)}\s*\(', line):
                    skip = True
                    break
            cleaned.append(
                line if not skip
                else "# [signature from MANDATORY CONTRACT — see above]"
            )
        result = '\n'.join(cleaned)
        logger.info(
            "[arch_exec] v5.23c Stripped competing signatures from arch context (%s)",
            func_names,
        )
        return result
    except Exception as e:
        logger.warning("[arch_exec] v5.23c Signature stripping failed (non-fatal): %s", e)
        return file_context


# ---------------------------------------------------------------------------
# Strike error formatting  (v5.23c)
# ---------------------------------------------------------------------------

def format_strike_error_block(
    strike: int,
    max_strikes: int,
    last_error: str,
    structured_sig_mismatches: list,
) -> str:
    """Build the retry feedback block prepended to the Implementer prompt."""
    if structured_sig_mismatches:
        parts = [
            f"## STRIKE {strike}/{max_strikes} — YOUR SIGNATURES WERE WRONG\n",
            "Your previous output was REJECTED because function signatures "
            "did not match the contract.\n",
        ]
        for mm in structured_sig_mismatches[:3]:
            parts.append(f"### `{mm.function_name}`\n")
            parts.append(f"You wrote:\n```python\n{mm.actual_signature}\n```\n")
            parts.append(f"Contract REQUIRES:\n```python\n{mm.expected_signature}\n```\n")
            if mm.differences:
                parts.append(f"Differences: {'; '.join(mm.differences[:3])}\n")
        parts.append(
            "\nCopy the REQUIRED signatures from the MANDATORY CONTRACT "
            "section above. Do not modify them in any way.\n"
        )
        parts.append(
            "\nCRITICAL: Output ONLY valid Python source code. No English "
            "explanations, no markdown commentary, no architecture descriptions. "
            "Start with imports or a module docstring.\n"
        )
        return "\n".join(parts)

    capped = last_error[:500] + ("..." if len(last_error) > 500 else "")
    return (
        f"## STRIKE {strike}/{max_strikes} — FIX NOW\n\n"
        f"Your previous output was REJECTED:\n\n"
        f"```\n{capped}\n```\n\n"
        f"Fix the issue above. If a MANDATORY CONTRACT section is present, "
        f"copy its signatures exactly.\n\n"
        f"CRITICAL: Output ONLY valid Python source code. No English "
        f"explanations, no markdown commentary, no architecture descriptions. "
        f"Start with imports or a module docstring.\n\n"
    )


def prepend_strike_error(user_prompt: str, error_block: str) -> str:
    """Insert the strike error block after the first paragraph break."""
    first_break = user_prompt.find("\n\n")
    if first_break > 0:
        return user_prompt[:first_break + 2] + error_block + user_prompt[first_break + 2:]
    return error_block + user_prompt


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_create_prompt(
    rel_path: str,
    file_context_for_prompt: str,
    contract_block: str,
    job_context_section: str,
    available_modules_evidence: str,
) -> Tuple[str, str]:
    """Build (user_prompt, system_prompt) for CREATE action."""
    user_prompt = f"Generate the complete content for a new file: `{rel_path}`\n\n"
    if contract_block:
        user_prompt += f"{contract_block}\n\n"
    user_prompt += f"## Architecture Specification\n\n{file_context_for_prompt}\n\n"
    if job_context_section:
        user_prompt += f"{job_context_section}\n\n"
    # Source context and modules evidence appended by caller if needed
    if available_modules_evidence:
        user_prompt += available_modules_evidence
    user_prompt += "Output ONLY the file content. No markdown fences, no explanations."
    return user_prompt, IMPLEMENTER_NEW_FILE_SYSTEM


def build_modify_prompt(
    rel_path: str,
    existing_content: str,
    file_context_for_prompt: str,
    contract_block: str,
    job_context_section: str,
    available_modules_evidence: str,
    use_edit_mode: bool,
) -> Tuple[str, str]:
    """Build (user_prompt, system_prompt) for MODIFY action."""
    file_char_count = len(existing_content)
    existing_imports = _extract_existing_imports(existing_content, rel_path)

    if use_edit_mode:
        user_prompt = (
            f"Apply the following modifications to `{rel_path}` ({file_char_count:,} chars).\n\n"
            f"## Current File Content\n```\n{existing_content}\n```\n\n"
        )
        if existing_imports:
            user_prompt += (
                f"## Existing Imports\n"
                f"Follow the same import patterns for any new imports.\n"
                f"```\n{existing_imports}\n```\n\n"
            )
        if contract_block:
            user_prompt += f"{contract_block}\n\n"
        user_prompt += f"## Modification Instructions\n\n{file_context_for_prompt}\n\n"
        if job_context_section:
            user_prompt += f"{job_context_section}\n\n"
        user_prompt += (
            "Output ONLY a JSON array of edit objects. "
            'Each object has "old_text" (exact unique snippet from the file) '
            'and "new_text" (replacement). No markdown fences.'
        )
        return user_prompt, IMPLEMENTER_MODIFY_EDIT_SYSTEM

    # Standard full-file rewrite
    user_prompt = (
        f"Apply the following modifications to `{rel_path}`.\n\n"
        f"## Current File Content\n```\n{existing_content}\n```\n\n"
    )
    if existing_imports:
        user_prompt += (
            f"## Existing Imports\n"
            f"The file currently uses these imports. Follow the same "
            f"import patterns and module paths for any new imports you add.\n"
            f"```\n{existing_imports}\n```\n\n"
        )
    if contract_block:
        user_prompt += f"{contract_block}\n\n"
    user_prompt += f"## Modification Instructions\n\n{file_context_for_prompt}\n\n"
    if job_context_section:
        user_prompt += f"{job_context_section}\n\n"
    if available_modules_evidence:
        user_prompt += available_modules_evidence
    user_prompt += "Output the COMPLETE modified file. No markdown fences."
    return user_prompt, IMPLEMENTER_MODIFY_FILE_SYSTEM




# ---------------------------------------------------------------------------
# Verification prompt assembly  (v1.0 code block extraction)
# ---------------------------------------------------------------------------

def build_verify_prompt(
    rel_path: str,
    prefilled_content: str,
    file_context_for_prompt: str,
    contract_block: str,
    job_context_section: str,
    available_modules_evidence: str,
) -> tuple[str, str]:
    """Build (user_prompt, system_prompt) for verification of pre-filled code.

    Used when code has been extracted from the architecture document and
    the LLM's role is to verify and gap-fill rather than generate.

    Args:
        rel_path: Relative file path.
        prefilled_content: Code extracted from the architecture document.
        file_context_for_prompt: Architecture section text for context.
        contract_block: Mandatory contract signatures (if any).
        job_context_section: Cross-file context from other created files.
        available_modules_evidence: Available modules list.

    Returns:
        (user_prompt, system_prompt) tuple.
    """
    user_prompt = (
        f"## Pre-Filled File: `{rel_path}`\n\n"
        f"The following code was extracted directly from the approved "
        f"architecture document. It is the intended implementation.\n\n"
        f"```\n{prefilled_content}\n```\n\n"
    )

    if contract_block:
        user_prompt += f"{contract_block}\n\n"

    user_prompt += (
        f"## Architecture Specification (for reference)\n\n"
        f"{file_context_for_prompt}\n\n"
    )

    if job_context_section:
        user_prompt += f"{job_context_section}\n\n"

    if available_modules_evidence:
        user_prompt += available_modules_evidence

    user_prompt += (
        "Verify the pre-filled code above. If it is complete and correct, "
        "output it UNCHANGED. Only fix genuine issues (missing imports, "
        "syntax errors, incomplete sections). Do NOT rewrite working code. "
        "Output ONLY the file content — no markdown fences, no explanations."
    )

    return user_prompt, IMPLEMENTER_VERIFY_FILE_SYSTEM

# ---------------------------------------------------------------------------
# LLM context injection  (v3.0 experience memory + RAG)
# ---------------------------------------------------------------------------

def inject_experience_and_rag(
    system_prompt: str,
    rel_path: str,
    action: str,
    file_context_snippet: str,
) -> str:
    """Append experience memory and RAG context to system prompt.

    Non-fatal — returns system_prompt unchanged if either fails.
    """
    # Experience memory
    try:
        from app.experience.retrieval import retrieve_for_stage, format_injection
        from app.db import get_db_session
        db = get_db_session()
        patterns = retrieve_for_stage(
            db, stage="implementer",
            context=f"Implementing {rel_path} ({action}): {file_context_snippet[:150]}",
            language=_infer_lang_from_path(rel_path),
            error_signature=None,
            max_results=5,
        )
        if patterns:
            injection = format_injection(patterns, stage="implementer")
            if injection:
                system_prompt += f"\n\n{injection}"
        db.close()
    except Exception:
        pass

    # Codebase RAG
    try:
        from app.rag.vector_store import retrieve_code_context
        from app.db import get_db_session
        db = get_db_session()
        rag_ctx = retrieve_code_context(
            db, stage="implementer",
            context=f"{rel_path}: {file_context_snippet[:200]}",
            file_scope=[rel_path] if action == "modify" else None,
            max_results=3,
        )
        if rag_ctx:
            system_prompt += f"\n\n{rag_ctx}"
        db.close()
    except Exception:
        pass

    return system_prompt


# ---------------------------------------------------------------------------
# Pre-flight gate  (v2.5)
# ---------------------------------------------------------------------------

def run_preflight_gate(
    interface_contract: str,
    rel_path: str,
    user_prompt: str,
    system_prompt: str,
) -> None:
    """Advisory check — warns if required symbols are missing from prompt."""
    try:
        from app.overwatcher.deterministic_checker import extract_required_exports
        required = extract_required_exports(interface_contract, rel_path)
        if required:
            combined = (user_prompt or "") + (system_prompt or "")
            missing = [sym for sym in required if sym not in combined]
            if missing:
                logger.warning(
                    "[arch_exec] v2.5 PRE-FLIGHT: %d required symbols missing "
                    "from prompt for %s: %s",
                    len(missing), rel_path, missing[:5],
                )
    except Exception:
        pass  # Advisory only
