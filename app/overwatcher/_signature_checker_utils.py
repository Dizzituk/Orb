import logging
import re
from typing import List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


SIGNATURE_CHECKER_BUILD_ID = "2026-02-17-v1.2-backslash-path-normalisation"

_TYPE_ALIASES = {
    "dict": "Dict",
    "list": "List",
    "tuple": "Tuple",
    "set": "Set",
    "frozenset": "FrozenSet",
    "type": "Type",
}

_REVERSE_ALIASES = {v: k for k, v in _TYPE_ALIASES.items()}

def _normalise_type(type_str: Optional[str]) -> Optional[str]:
    """
    Normalise a type annotation string for comparison.

    Handles:
      - Dict vs dict, List vs list, etc.
      - Optional[X] vs X | None vs Union[X, None]
      - Whitespace differences
      - Quoting differences

    Returns normalised string, or None if input is None/empty.
    """
    if not type_str or not type_str.strip():
        return None

    t = type_str.strip()

    # Strip quotes (forward references)
    t = t.strip("'\"")

    # Normalise whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    # Normalise Optional[X] -> X | None
    optional_match = re.match(r'^Optional\[(.+)\]$', t)
    if optional_match:
        inner = optional_match.group(1).strip()
        t = f"{inner} | None"

    # Normalise Union[X, None] -> X | None
    union_match = re.match(r'^Union\[(.+),\s*None\]$', t)
    if union_match:
        inner = union_match.group(1).strip()
        t = f"{inner} | None"

    # Normalise Dict -> dict, List -> list, etc. (case-sensitive mapping)
    for typing_name, builtin_name in _REVERSE_ALIASES.items():
        # Replace at word boundary: Dict[str, Any] -> dict[str, Any]
        t = re.sub(rf'\b{typing_name}\b', builtin_name, t)

    # Also handle lowercase -> lowercase (already normalised, but ensure)
    # No-op since we already mapped to lowercase

    return t

def _extract_base_type(normalised: Optional[str]) -> Optional[str]:
    """Extract the base type, stripping generic parameters.

    ``dict[str, Any]`` → ``dict``, ``list[int]`` → ``list``.

    v1.1: Prevents false-positive mismatches for ``Dict[str, Any]`` vs ``dict``.
    """
    if not normalised:
        return None
    idx = normalised.find('[')
    return normalised[:idx].strip() if idx > 0 else normalised

_CONTAINER_BUILTINS = {'dict', 'list', 'tuple', 'set', 'frozenset', 'type'}

def _types_match(req_norm: Optional[str], act_norm: Optional[str]) -> bool:
    """Compare two normalised type strings with base-type leniency.

    Rules:
      - Exact match after normalisation → pass
      - Both resolve to the same base builtin container → pass
        (e.g. ``dict`` vs ``dict[str, Any]``)
      - Different base types → fail (e.g. ``str`` vs ``Path``)
    """
    if not req_norm or not act_norm:
        return True  # One side unannotated
    if req_norm == act_norm:
        return True
    req_base = _extract_base_type(req_norm)
    act_base = _extract_base_type(act_norm)
    if req_base == act_base and req_base in _CONTAINER_BUILTINS:
        return True
    return False

def extract_contract_signatures_for_file(
    interface_contract: str,
    file_path: str,
) -> List[str]:
    """
    Parse the interface_contract markdown (from format_contract_for_segment())
    to find signature strings for a specific file.

    The contract format has:
        - `path/to/file.py` → consumed by `seg-05`, `seg-06`
          **MUST EXPORT these symbols** (downstream segments depend on them):
            - `def func_name(params) -> ret`
            - `bare_name`

    Returns list of signature strings (only the "def ..." ones, not bare names).
    """
    if not interface_contract or not file_path:
        return []

    signatures = []
    file_path_norm = file_path.replace("\\", "/").strip()

    # State machine: find the file path line, then collect signatures.
    # v1.3 (Fix 23): The file path can appear MULTIPLE times in the contract
    # markdown — once in "File Scope Constraint", once in "This Segment EXPORTS",
    # and again in "Package Module Map". The extractor must NOT give up after
    # the first occurrence exits without finding signatures. Instead, when a
    # file section ends, we keep scanning so subsequent occurrences are found.
    lines = interface_contract.split("\n")
    in_file_section = False
    in_exports = False

    for line in lines:
        stripped = line.strip()

        # Detect file path line: "  - `path/to/file.py` → consumed by ..."
        # v1.2: Normalise backslashes in contract line too (skeleton stores Windows paths)
        stripped_norm = stripped.replace("\\", "/")
        if f"`{file_path_norm}`" in stripped_norm:
            # v1.3: Re-enter file section even if we already exited a previous one.
            # This handles the file appearing in File Scope first, then Exports later.
            in_file_section = True
            in_exports = False
            continue

        if in_file_section:
            # Detect MUST EXPORT header
            # v1.3: Match all variants: "MUST EXPORT", "MUST DEFINE AND EXPORT",
            # "MUST RE-EXPORT". The common signal is both "MUST" and "EXPORT" present.
            if "MUST" in stripped and "EXPORT" in stripped:
                in_exports = True
                continue

            # Detect end of this file's section
            # New section header always ends the file section
            if stripped.startswith("###") or stripped.startswith("## "):
                in_file_section = False
                in_exports = False
                continue  # v1.3: keep scanning — file may appear in a later section

            # New file entry: starts with "- `", contains a file path (has /)
            # but is NOT a function signature (doesn't contain 'def ')
            if stripped.startswith("- `") and "`" in stripped[3:]:
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    candidate = match.group(1).strip().replace("\\", "/")
                    # File paths contain '/' or end with '.py'; signatures start with 'def '/'async def '
                    is_file_path = ("/" in candidate or candidate.endswith(".py"))
                    is_signature = candidate.startswith("def ") or candidate.startswith("async def ")
                    if is_file_path and not is_signature:
                        if candidate != file_path_norm:
                            in_file_section = False
                            in_exports = False
                            continue  # v1.3: keep scanning

            # Collect signature lines
            if in_exports and stripped.startswith("- `"):
                # Extract content between backticks
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    sig = match.group(1).strip()
                    # Only include actual function signatures (def ... or async def ...)
                    if sig.startswith("def ") or sig.startswith("async def "):
                        signatures.append(sig)

    if signatures:
        logger.debug(
            "[sig_checker] Found %d contract signature(s) for %s",
            len(signatures), file_path,
        )

    return signatures
