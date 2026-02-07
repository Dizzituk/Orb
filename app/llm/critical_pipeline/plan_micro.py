# FILE: app/llm/critical_pipeline/plan_micro.py
"""
Micro-execution plan generation.

Generates a minimal, mode-aware execution plan for MICRO jobs that
Overwatcher can execute directly (no architecture design needed).
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_micro_execution_plan(spec_data: Dict[str, Any], job_id: str) -> str:
    """
    Generate a minimal execution plan for MICRO jobs.

    Mode-aware plan generation:
    - CHAT_ONLY:  no write steps, no output verification
    - REWRITE_IN_PLACE / APPEND_IN_PLACE:  write to same file
    - SEPARATE_REPLY_FILE:  write to output path

    Handles single-file and multi-target read operations.
    """
    output_mode = (spec_data.get("sandbox_output_mode") or "").strip().lower()
    is_multi_target_read = spec_data.get("is_multi_target_read", False)
    multi_target_files = spec_data.get("multi_target_files", [])

    logger.info(
        "[plan_micro] output_mode=%r, is_multi_target_read=%s, multi_target_count=%d",
        output_mode, is_multi_target_read, len(multi_target_files),
    )

    # Resolve input/output paths
    if is_multi_target_read and multi_target_files:
        input_path = f"(multi-target read: {len(multi_target_files)} files)"
    else:
        input_path = (
            spec_data.get("sandbox_input_path")
            or spec_data.get("input_file_path")
            or "(input path not resolved)"
        )

    output_path = (
        spec_data.get("sandbox_output_path")
        or spec_data.get("output_file_path")
        or spec_data.get("planned_output_path")
        or "(output path not resolved)"
    )

    input_excerpt = spec_data.get("sandbox_input_excerpt", "")
    reply_content = spec_data.get("sandbox_generated_reply", "")
    content_type = spec_data.get("sandbox_selected_type", "unknown")
    summary = spec_data.get(
        "goal",
        spec_data.get("summary", spec_data.get("objective", "Execute task per spec")),
    )

    content_type_line = ""
    if content_type and content_type.lower() != "unknown":
        content_type_line = f"- **Content Type:** {content_type}\n"

    # Dispatch to mode-specific plan builder
    if output_mode == "chat_only":
        return _plan_chat_only(
            job_id, summary, input_path, content_type_line,
            input_excerpt, reply_content,
        )
    elif output_mode in ("rewrite_in_place", "append_in_place"):
        return _plan_in_place(
            job_id, summary, input_path, content_type_line,
            input_excerpt, reply_content, output_mode,
        )
    else:
        return _plan_separate_file(
            job_id, summary, input_path, output_path, content_type_line,
            input_excerpt, reply_content, output_mode,
        )


# ---------------------------------------------------------------------------
# Plan templates
# ---------------------------------------------------------------------------

def _plan_chat_only(job_id, summary, input_path, ct_line, excerpt, reply):
    plan = f"""# Micro-Execution Plan

**Job ID:** {job_id}
**Type:** MICRO_EXECUTION (no architecture required)
**Output Mode:** CHAT_ONLY ⚠️ NO FILE WRITES

## Task Summary
{summary}

## Resolved Paths (by SpecGate)
- **Input:** `{input_path}`
- **Output:** (none - CHAT_ONLY mode)
{ct_line}

## Execution Steps

1. **Read Input File**
   - Path: `{input_path}`
   - Action: Read file contents

2. **Generate Response**
   - Parse the content
   - Generate response based on file content

3. **Return Response in Chat**
   - ⚠️ **NO FILE WRITE** - Response is returned in chat only
   - The input file will NOT be modified
   - No output file will be created

## Verification
- ✅ Input file was read
- ✅ Response was generated
- ⚠️ **NO OUTPUT FILE** (CHAT_ONLY mode - this is intentional)
"""
    plan += _append_preview_and_reply(excerpt, reply, is_chat_only=True)
    plan += """
## Notes
- ⚠️ **CHAT_ONLY MODE ACTIVE**
- The user explicitly requested NO file modifications
- Response will be returned in chat only
- Input file remains unchanged
- No output file will be created

---
✅ **Ready for Overwatcher** - CHAT_ONLY mode: Response will be returned in chat, NO file will be modified.
"""
    return plan


def _plan_in_place(job_id, summary, input_path, ct_line, excerpt, reply, mode):
    mode_desc = "rewrite content" if mode == "rewrite_in_place" else "append to file"
    plan = f"""# Micro-Execution Plan

**Job ID:** {job_id}
**Type:** MICRO_EXECUTION (no architecture required)
**Output Mode:** {mode.upper()}

## Task Summary
{summary}

## Resolved Paths (by SpecGate)
- **Input:** `{input_path}`
- **Output:** `{input_path}` (same file - {mode})
{ct_line}

## Execution Steps

1. **Read Input File**
   - Path: `{input_path}`
   - Action: Read file contents

2. **Process Content**
   - Parse the content
   - Generate response based on file content

3. **Write to Same File ({mode.upper()})**
   - Path: `{input_path}`
   - Mode: {mode}
   - Action: {mode_desc}

4. **Verify**
   - Confirm file was updated
   - Validate content is correct
"""
    plan += _append_preview_and_reply(excerpt, reply, is_chat_only=False)
    plan += f"""
## Notes
- This is a simple file operation task
- Mode: {mode.upper()} - writing to same file
- All paths are pre-resolved by SpecGate
- Overwatcher can execute directly

---
✅ **Ready for Overwatcher** - Say 'Astra, command: send to overwatcher' to execute.
"""
    return plan


def _plan_separate_file(
    job_id, summary, input_path, output_path, ct_line,
    excerpt, reply, mode,
):
    plan = f"""# Micro-Execution Plan

**Job ID:** {job_id}
**Type:** MICRO_EXECUTION (no architecture required)
**Output Mode:** {mode.upper() if mode else 'SEPARATE_REPLY_FILE'}

## Task Summary
{summary}

## Resolved Paths (by SpecGate)
- **Input:** `{input_path}`
- **Output:** `{output_path}`
{ct_line}

## Execution Steps

1. **Read Input File**
   - Path: `{input_path}`
   - Action: Read file contents

2. **Process Content**
   - Parse the content
   - Generate response based on file content

3. **Write Output File**
   - Path: `{output_path}`
   - Action: Write generated reply

4. **Verify**
   - Confirm output file exists
   - Validate content is correct
"""
    plan += _append_preview_and_reply(excerpt, reply, is_chat_only=False)
    plan += """
## Notes
- This is a simple file operation task
- No architectural changes required
- All paths are pre-resolved by SpecGate
- Overwatcher can execute directly

---
✅ **Ready for Overwatcher** - Say 'Astra, command: send to overwatcher' to execute.
"""
    return plan


def _append_preview_and_reply(excerpt, reply, is_chat_only):
    """Append input preview and generated reply sections."""
    parts = []
    if excerpt:
        parts.append(f"""
## Input Preview
```
{excerpt[:500]}
```
""")
    if reply:
        if is_chat_only:
            parts.append(f"""
## Generated Reply (from SpecGate) - WILL BE RETURNED IN CHAT
```
{reply}
```

**Note:** This reply will be displayed in chat. NO file will be modified.
""")
        else:
            parts.append(f"""
## Generated Reply (from SpecGate)
```
{reply}
```

**Note:** SpecGate has already generated this reply. Overwatcher just needs to write it.
""")
    elif not is_chat_only:
        parts.append("""
## Expected Output
(to be generated by Overwatcher based on input content)
""")
    return "".join(parts)
