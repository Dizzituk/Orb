# FILE: app/llm/critical_pipeline/plan_scan.py
"""
Scan-only plan generation.

Generates a minimal execution plan for SCAN_ONLY jobs — read-only
filesystem scan/search/enumerate operations returned via chat.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_scan_execution_plan(spec_data: Dict[str, Any], job_id: str) -> str:
    """
    Generate a minimal execution plan for SCAN_ONLY jobs.

    Uses the scan fields populated by SpecGate:
    scan_roots, scan_terms, scan_targets, scan_case_mode,
    scan_exclusions, scan_content_mode.
    """
    scan_roots = spec_data.get("scan_roots", [])
    scan_terms = spec_data.get("scan_terms", [])
    scan_targets = spec_data.get("scan_targets", ["names", "contents"])
    scan_case_mode = spec_data.get("scan_case_mode", "case_insensitive")
    scan_exclusions = spec_data.get("scan_exclusions", [
        ".git", "node_modules", "dist", "build", ".venv", "__pycache__",
    ])
    scan_content_mode = spec_data.get("scan_content_mode", "text_only")

    goal = spec_data.get(
        "goal",
        spec_data.get("summary", spec_data.get("objective", "Scan and enumerate matching items")),
    )

    roots_display = (
        "\n".join(f"  - `{r}`" for r in scan_roots) if scan_roots else "  - (none specified)"
    )
    terms_display = ", ".join(f"`{t}`" for t in scan_terms) if scan_terms else "(none specified)"
    targets_display = ", ".join(scan_targets) if scan_targets else "names, contents"

    exclusions_display = ", ".join(f"`{e}`" for e in scan_exclusions[:5]) if scan_exclusions else "(none)"
    if len(scan_exclusions) > 5:
        exclusions_display += f" + {len(scan_exclusions) - 5} more"

    return f"""# Scan Execution Plan

**Job ID:** {job_id}
**Type:** SCAN_ONLY (read-only, no file writes)
**Output Mode:** CHAT_ONLY ⚠️ NO FILE WRITES

## Task Summary
{goal}

## Scan Parameters (from SpecGate)

### Scan Roots
{roots_display}

### Search Terms
{terms_display}

### Search Targets
{targets_display}

### Case Mode
{scan_case_mode}

### Content Mode
{scan_content_mode}

### Exclusions
{exclusions_display}

## Execution Steps

1. **Enumerate Directories**
   - Walk directory tree from each scan root
   - Skip excluded directories ({exclusions_display})
   - Collect folder and file names

2. **Match Folder/File Names**
   - Check each folder/file name against search terms
   - Mode: {scan_case_mode}
   - Record name hits with full path

3. **Scan File Contents** (if "contents" in targets)
   - Read text/code files only (skip binaries)
   - Search for term occurrences
   - Record content hits with path, line number, snippet

4. **Compile Results**
   - Group hits by type (name vs content)
   - Group by path/folder
   - Include context for each hit

5. **Return Report in Chat**
   - ⚠️ **NO FILE WRITE** - Results returned in chat only
   - Format as structured report with full paths
   - Explain why each hit exists

## Expected Output Shape

```
SCAN_REPORT:
├── Name Hits:
│   ├── D:\\path\\to\\folder\\orb-file.txt  (matched: "orb" in filename)
│   └── ...
├── Content Hits:
│   ├── D:\\path\\to\\file.py:42  (snippet: "import orb...")
│   └── ...
└── Summary: X name hits, Y content hits across Z files
```

## Verification
- ✅ All scan roots were traversed
- ✅ Search terms were applied
- ✅ Results were compiled
- ⚠️ **NO OUTPUT FILE** (SCAN_ONLY mode - results in chat only)

## Notes
- ⚠️ **SCAN_ONLY MODE ACTIVE**
- This is a read-only operation
- No files will be created or modified
- Results will be returned in chat only
- Scan does NOT require Overwatcher for execution

---
✅ **Ready for Execution** - This scan job can be executed directly without Overwatcher.
"""
