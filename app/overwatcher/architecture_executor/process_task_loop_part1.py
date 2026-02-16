"""
process_task_loop_part1.py

Part 1 of the per-file task processing loop.
Responsibilities:
- Prepare per-task execution context for LLM generation
- Execute the three-strike loop "generation phase":
  - Extract architecture section for the file
  - Decide verbatim extraction vs LLM generation
  - Read existing file for modify operations
  - Decide edit mode vs full rewrite
- Return structured attempt artifacts for Part 2 to write+verify

This module is read-only for sandbox operations and delegates all writes to Part 2.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable

from ..sandbox_client import SandboxClient
from .constants import MAX_STRIKES_PER_TASK, MODIFY_EDIT_MODE_THRESHOLD
from .sandbox_ops import _read_existing_file
from .parsing import extract_section_for_file

logger = logging.getLogger(__name__)


async def run_process_task_loop_part1(
    client: SandboxClient,
    architecture_content: str,
    sandbox_base: str,
    task_info: Dict[str, Any],
    task_action: str,
    job_context: Dict[str, Any],
    router_registrations: List[str],
    available_modules_evidence: str,
    impl_provider: str,
    impl_model: str,
    impl_max_tokens: int,
    llm_call_fn: Callable[[str, str, str, str, int], Awaitable[str]],
    strip_markdown_fences_fn: Callable[[str], str],
    parse_edit_pairs_fn: Callable[[str], List[tuple]],
    format_job_context_fn: Callable[[Dict], str],
    resolve_multi_root_path_fn: Callable[[SandboxClient, str, str], str],
    file_prompt_builder_fn: Callable[[str, str, str, bool, Optional[str], str, List[str], str], tuple],
    source_files_context: str = "",
    files_already_created: str = "",
    quarantine_skip_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute Part 1 of the file task processing loop.
    
    Returns a dict containing:
    - task_success: bool (always False in Part1, Part2 decides final success)
    - file_content: Optional[str] (candidate content from last strike)
    - use_edit_mode: bool
    - verbatim_used: bool
    - existing_content: Optional[str]
    - last_error: Optional[str]
    - strike_count: int
    - file_path: str
    - file_context: str (architecture section for this file)
    - user_prompt: str (last generated prompt)
    - system_prompt: str (last used system prompt)
    """
    
    file_path = task_info["file"]
    strike_count = 0
    last_error: Optional[str] = None
    file_content: Optional[str] = None
    use_edit_mode = False
    verbatim_used = False
    existing_content: Optional[str] = None
    file_context = ""
    user_prompt = ""
    system_prompt = ""
    
    # Early skip for quarantined files
    if quarantine_skip_list and file_path in quarantine_skip_list:
        logger.info(f"[Part1] Skipping quarantined file: {file_path}")
        return {
            "task_success": False,
            "file_content": None,
            "use_edit_mode": False,
            "verbatim_used": False,
            "existing_content": None,
            "last_error": f"Quarantined file skipped: {file_path}",
            "strike_count": 0,
            "file_path": file_path,
            "file_context": "",
            "user_prompt": "",
            "system_prompt": "",
        }
    
    # Resolve absolute sandbox path
    try:
        absolute_file_path = resolve_multi_root_path_fn(client, sandbox_base, file_path)
        logger.debug(f"[Part1] Resolved {file_path} -> {absolute_file_path}")
    except Exception as e:
        logger.error(f"[Part1] Failed to resolve path for {file_path}: {e}")
        return {
            "task_success": False,
            "file_content": None,
            "use_edit_mode": False,
            "verbatim_used": False,
            "existing_content": None,
            "last_error": f"Path resolution error: {e}",
            "strike_count": 0,
            "file_path": file_path,
            "file_context": "",
            "user_prompt": "",
            "system_prompt": "",
        }
    
    # Extract architecture section for this file
    file_context = extract_section_for_file(architecture_content, file_path)
    if not file_context:
        logger.warning(f"[Part1] No architecture section found for {file_path}")
        return {
            "task_success": False,
            "file_content": None,
            "use_edit_mode": False,
            "verbatim_used": False,
            "existing_content": None,
            "last_error": f"No architecture section found for {file_path}",
            "strike_count": 0,
            "file_path": file_path,
            "file_context": "",
            "user_prompt": "",
            "system_prompt": "",
        }
    
    # Decide verbatim extraction vs LLM generation
    # Check if SOURCE FILES section exists in file_context
    verbatim_used = "SOURCE FILES" in file_context
    
    # For modify actions, read existing file
    if task_action == "modify":
        existing_content = await _read_existing_file(client, absolute_file_path)
        if existing_content is None:
            logger.warning(f"[Part1] Modify action but file does not exist: {file_path}")
            # Treat as create
            task_action = "create"
    
    # Decide edit mode for modify actions
    if task_action == "modify" and existing_content:
        existing_lines = len(existing_content.splitlines())
        use_edit_mode = existing_lines >= MODIFY_EDIT_MODE_THRESHOLD
        logger.info(f"[Part1] File {file_path} has {existing_lines} lines, edit_mode={use_edit_mode}")
    
    # Three-strike loop for generation
    for strike in range(MAX_STRIKES_PER_TASK):
        strike_count = strike + 1
        logger.info(f"[Part1] Strike {strike_count}/{MAX_STRIKES_PER_TASK} for {file_path}")
        
        try:
            # Build prompts
            system_prompt, user_prompt = file_prompt_builder_fn(
                file_path,
                file_context,
                format_job_context_fn(job_context),
                use_edit_mode,
                existing_content if task_action == "modify" else None,
                available_modules_evidence,
                router_registrations,
                files_already_created,
            )
            
            # Add source files context if provided
            if source_files_context:
                user_prompt = f"{user_prompt}\n\n## SOURCE FILES\n{source_files_context}"
            
            # Add last error context for retry strikes
            if strike > 0 and last_error:
                user_prompt = f"{user_prompt}\n\n## PREVIOUS ATTEMPT ERROR\n{last_error}\n\nPlease fix the issues and regenerate the complete file."
            
            # Call LLM
            logger.debug(f"[Part1] Calling LLM for {file_path} (strike {strike_count})")
            raw_response = await llm_call_fn(
                system_prompt,
                user_prompt,
                impl_provider,
                impl_model,
                impl_max_tokens,
            )
            
            # Strip markdown fences
            file_content = strip_markdown_fences_fn(raw_response)
            
            # If verbatim extraction, extract from file_context
            if verbatim_used:
                # Extract code between SOURCE FILES markers
                if "SOURCE FILES" in file_context:
                    parts = file_context.split("SOURCE FILES", 1)
                    if len(parts) > 1:
                        source_section = parts[1]
                        # Find first code block
                        if "```" in source_section:
                            code_blocks = source_section.split("```")
                            if len(code_blocks) > 1:
                                # Take first code block, strip language identifier
                                code = code_blocks[1]
                                if "\n" in code:
                                    lines = code.split("\n", 1)
                                    if len(lines) > 1:
                                        file_content = lines[1]
                                    else:
                                        file_content = code
                                else:
                                    file_content = code
                                logger.info(f"[Part1] Extracted verbatim content for {file_path}")
            
            # Validate basic structure
            if not file_content or len(file_content.strip()) == 0:
                last_error = "Generated content is empty"
                logger.warning(f"[Part1] Strike {strike_count}: {last_error}")
                continue
            
            # If edit mode, parse edit pairs
            if use_edit_mode:
                try:
                    edit_pairs = parse_edit_pairs_fn(file_content)
                    if not edit_pairs:
                        last_error = "Edit mode but no valid SEARCH/REPLACE blocks found"
                        logger.warning(f"[Part1] Strike {strike_count}: {last_error}")
                        continue
                    logger.info(f"[Part1] Parsed {len(edit_pairs)} edit pairs for {file_path}")
                except Exception as e:
                    last_error = f"Failed to parse edit pairs: {e}"
                    logger.warning(f"[Part1] Strike {strike_count}: {last_error}")
                    continue
            
            # Success - content generated
            logger.info(f"[Part1] Successfully generated content for {file_path} on strike {strike_count}")
            last_error = None
            break
            
        except Exception as e:
            last_error = f"Strike {strike_count} generation error: {e}"
            logger.error(f"[Part1] {last_error}")
            continue
    
    # Return artifacts for Part 2
    return {
        "task_success": False,  # Part2 will decide final success
        "file_content": file_content,
        "use_edit_mode": use_edit_mode,
        "verbatim_used": verbatim_used,
        "existing_content": existing_content,
        "last_error": last_error,
        "strike_count": strike_count,
        "file_path": file_path,
        "file_context": file_context,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "absolute_file_path": absolute_file_path,
        "task_action": task_action,
    }