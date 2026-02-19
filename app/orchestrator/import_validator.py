"""
Import Validator — Deterministic Pre-Cohesion Gate (Fix 11).

Runs immediately after architecture generation, per-segment, BEFORE
the architecture is approved. Zero LLM cost — pure dictionary lookup.

Checks every cross-segment import in the architecture against the
enrichment export map. If an import references a symbol that doesn't
exist in any sibling segment's enrichment, it is flagged immediately
with a precise error listing the real available symbols.

v1.0 (2026-02-19): Initial implementation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

IMPORT_VALIDATOR_BUILD_ID = "2026-02-19-v1.1-positive-symbol-map"
print(f"[IMPORT_VALIDATOR_LOADED] BUILD_ID={IMPORT_VALIDATOR_BUILD_ID}")


@dataclass
class ImportViolation:
    """A single invalid cross-segment import."""
    symbol_name: str
    referenced_module: str  # e.g. "_dependency"
    consuming_segment: str  # the segment whose architecture has this import
    error_type: str  # "phantom" or "wrong_segment"
    available_symbols: List[str] = field(default_factory=list)
    correct_source: str = ""  # if wrong_segment, where it actually lives
    message: str = ""


@dataclass
class ValidationResult:
    """Result of the import validation."""
    passed: bool
    violations: List[ImportViolation] = field(default_factory=list)
    symbols_checked: int = 0
    segments_scanned: int = 0
    # v1.1 (Fix 19): Full module export map for positive guidance
    module_export_map: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def format_feedback(self) -> str:
        """Format violations as actionable feedback for LLM re-generation.

        v1.1 (Fix 19): Now includes the complete positive symbol map so the
        LLM knows exactly what IS available, not just what's wrong.
        """
        if self.passed:
            return ""
        lines = [
            "## ❌ Import Validation Failed (Deterministic Check)",
            "",
            f"Checked {self.symbols_checked} cross-segment import(s) against "
            f"{self.segments_scanned} sibling segment(s).",
            "",
            "The following imports reference symbols that do not exist in the "
            "specified source module. Fix each one:",
            "",
        ]
        for v in self.violations:
            lines.append(f"### {v.symbol_name}")
            lines.append(f"- **Error**: {v.message}")
            if v.available_symbols:
                sym_list = ", ".join(f"`{s}`" for s in sorted(v.available_symbols))
                lines.append(f"- **Available from `{v.referenced_module}`**: {sym_list}")
            else:
                lines.append(f"- **`{v.referenced_module}` exports**: (nothing)")
            if v.correct_source:
                lines.append(f"- **Hint**: This symbol is actually exported by `{v.correct_source}`")
            lines.append("")
        lines.append(
            "Either import from the correct module, use an available symbol, "
            "or define the function yourself in this segment."
        )

        # v1.1 (Fix 19): Append complete positive symbol map
        if self.module_export_map:
            lines.append("")
            lines.append("## ✅ Complete Module Export Map (ground truth)")
            lines.append("")
            lines.append("Use ONLY these symbols for cross-segment imports:")
            lines.append("")
            for mod_name in sorted(self.module_export_map.keys()):
                seg_syms = self.module_export_map[mod_name]
                all_syms: Set[str] = set()
                for syms in seg_syms.values():
                    all_syms.update(syms)
                if all_syms:
                    sym_list = ", ".join(f"`{s}`" for s in sorted(all_syms))
                    lines.append(f"- **`{mod_name}`**: {sym_list}")
                else:
                    lines.append(f"- **`{mod_name}`**: (no exports)")
            lines.append("")
            lines.append(
                "Any symbol NOT in the list above will fail validation. "
                "Do not invent new symbols. Pick from this list or define "
                "the function yourself within this segment."
            )

        return "\n".join(lines)


def _load_sibling_exports(
    parent_job_dir: str,
    current_segment_id: str,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, str]]:
    """Load enrichment data from all sibling segments.

    Returns:
        module_exports: {module_name: {segment_id: [symbol_names]}}
            Maps module filenames (without .py) to the symbols they export.
        symbol_to_segment: {symbol_name: segment_id}
            Maps every known symbol to which segment exports it.
    """
    segments_dir = os.path.join(parent_job_dir, "segments")
    module_exports: Dict[str, Dict[str, List[str]]] = {}
    symbol_to_segment: Dict[str, str] = {}

    if not os.path.isdir(segments_dir):
        logger.debug(
            "[import_validator] No segments dir: %s", segments_dir
        )
        return module_exports, symbol_to_segment

    for seg_dir_name in sorted(os.listdir(segments_dir)):
        seg_dir_path = os.path.join(segments_dir, seg_dir_name)
        enrichment_path = os.path.join(seg_dir_path, "enrichment.json")

        if not os.path.isfile(enrichment_path):
            continue

        # Skip self
        if seg_dir_name == current_segment_id:
            continue

        try:
            with open(enrichment_path, "r", encoding="utf-8") as f:
                enrichment = json.load(f)
        except Exception as e:
            logger.warning(
                "[import_validator] Failed to load %s: %s",
                enrichment_path, e
            )
            continue

        # Collect all symbols from this sibling
        symbols: Set[str] = set()

        # From explicit exports list
        for exp in enrichment.get("exports", []):
            if isinstance(exp, str):
                symbols.add(exp)
            elif isinstance(exp, dict):
                name = exp.get("name", "")
                if name:
                    symbols.add(name)

        # From AST-extracted functions
        for func in enrichment.get("functions", []):
            if isinstance(func, dict):
                name = func.get("name", "")
                if name:
                    symbols.add(name)
            elif isinstance(func, str):
                symbols.add(func)

        # From AST-extracted constants
        for const in enrichment.get("constants", []):
            if isinstance(const, dict):
                name = const.get("name", "")
                if name:
                    symbols.add(name)
            elif isinstance(const, str):
                symbols.add(const)

        # From AST-extracted classes
        for cls in enrichment.get("classes", []):
            if isinstance(cls, dict):
                name = cls.get("name", "")
                if name:
                    symbols.add(name)
            elif isinstance(cls, str):
                symbols.add(cls)

        # Determine the module name for this segment from file_scope
        # e.g. seg-04-dependency-checking -> _dependency.py -> "_dependency"
        module_name = ""
        file_scope = enrichment.get("file_scope", [])
        if file_scope:
            # Take the first .py file that isn't __init__.py
            for fs in file_scope:
                basename = os.path.basename(fs)
                if basename != "__init__.py" and basename.endswith(".py"):
                    module_name = basename[:-3]  # strip .py
                    break

        if not module_name:
            # Fall back: derive from segment name
            # e.g. "seg-04-dependency-checking" -> try to find in arch
            pass

        if symbols:
            for sym in symbols:
                symbol_to_segment[sym] = seg_dir_name

            if module_name:
                module_exports.setdefault(module_name, {})[seg_dir_name] = sorted(symbols)

            logger.debug(
                "[import_validator] %s (%s): %d symbols",
                seg_dir_name, module_name or "?", len(symbols)
            )

    return module_exports, symbol_to_segment


def _extract_cross_segment_imports(
    arch_text: str,
) -> List[Tuple[str, str]]:
    """Extract cross-segment imports from architecture text.

    Looks for patterns like:
        from ._constants import SEGMENT_LOOP_BUILD_ID
        from ._dependency import can_execute_segment, is_segment_blocked

    Returns:
        List of (module_name, symbol_name) tuples.
        e.g. [("_constants", "SEGMENT_LOOP_BUILD_ID"),
              ("_dependency", "can_execute_segment")]
    """
    imports: List[Tuple[str, str]] = []

    # Pattern: from ._module import sym1, sym2, ...
    # Also matches: from ._module import (sym1, sym2, ...)
    pattern = re.compile(
        r'from\s+\._(\w+)\s+import\s+'
        r'(?:\(([^)]+)\)|([^\n#]+))',
        re.MULTILINE,
    )

    for m in pattern.finditer(arch_text):
        module_name = f"_{m.group(1)}"
        # Get the import list from either parenthesized or single-line form
        import_str = m.group(2) or m.group(3) or ""
        # Split by comma and clean up
        for sym in import_str.split(","):
            sym = sym.strip().rstrip("\\").strip()
            # Skip empty, comments, and continuation lines
            if not sym or sym.startswith("#"):
                continue
            # Handle "symbol as alias" — we want the original name
            if " as " in sym:
                sym = sym.split(" as ")[0].strip()
            # Skip non-identifiers (sometimes architecture has prose)
            if not re.match(r'^[a-zA-Z_]\w*$', sym):
                continue
            imports.append((module_name, sym))

    return imports


def validate_architecture_imports(
    arch_text: str,
    segment_id: str,
    parent_job_dir: str,
) -> ValidationResult:
    """Validate that all cross-segment imports reference real symbols.

    This is a deterministic, zero-LLM-cost check. It:
    1. Extracts all `from ._module import symbol` from the architecture
    2. Loads enrichment data from all sibling segments
    3. Checks each imported symbol exists in the referenced module's exports
    4. Returns precise errors for any violations

    Args:
        arch_text: The generated architecture markdown text.
        segment_id: The current segment being validated.
        parent_job_dir: Path to the parent job directory (contains segments/).

    Returns:
        ValidationResult with pass/fail and any violations.
    """
    result = ValidationResult(passed=True)

    # Load sibling exports
    module_exports, symbol_to_segment = _load_sibling_exports(
        parent_job_dir, segment_id
    )
    result.segments_scanned = len(set(
        seg for segs in module_exports.values() for seg in segs
    ))
    # v1.1 (Fix 19): Store full export map for positive guidance in feedback
    result.module_export_map = module_exports

    if not module_exports and not symbol_to_segment:
        # No enrichment data available — pass through (non-blocking)
        logger.debug(
            "[import_validator] No sibling enrichment data for %s — skipping",
            segment_id,
        )
        return result

    # Extract cross-segment imports from architecture
    cross_imports = _extract_cross_segment_imports(arch_text)
    result.symbols_checked = len(cross_imports)

    if not cross_imports:
        logger.debug(
            "[import_validator] No cross-segment imports in %s", segment_id
        )
        return result

    # Build a flat set of all known symbols per module
    module_symbol_sets: Dict[str, Set[str]] = {}
    for mod_name, seg_syms in module_exports.items():
        all_syms: Set[str] = set()
        for syms in seg_syms.values():
            all_syms.update(syms)
        module_symbol_sets[mod_name] = all_syms

    # Also build a global set of ALL known symbols (for wrong_segment detection)
    all_known_symbols: Set[str] = set(symbol_to_segment.keys())

    # v1.1 (Fix 22): Pre-check enrichment completeness.
    # Modules with zero exports in enrichment are data gaps, not architecture
    # errors. Track them so we don't false-positive every import from them.
    _incomplete_modules: Set[str] = set()
    for mod_name, syms_set in module_symbol_sets.items():
        if not syms_set:
            _incomplete_modules.add(mod_name)
    # Also check: referenced modules with no enrichment at all
    _referenced_modules = set(mod for mod, _ in cross_imports)
    _missing_enrichment = _referenced_modules - set(module_symbol_sets.keys())

    if _incomplete_modules or _missing_enrichment:
        _all_gaps = _incomplete_modules | _missing_enrichment
        logger.info(
            "[import_validator] v1.1 Enrichment gaps for %s: %s (imports from these modules not validated)",
            segment_id, ", ".join(sorted(_all_gaps)),
        )

    # Validate each import
    for module_name, symbol_name in cross_imports:
        # v1.1 (Fix 22): Skip validation for modules with no enrichment data
        # These are data gaps, not architecture errors
        if module_name in _missing_enrichment:
            logger.debug(
                "[import_validator] Skipping %s from %s (no enrichment for module)",
                symbol_name, module_name,
            )
            continue

        # Check if we know this module at all
        if module_name not in module_symbol_sets:
            # Module not in enrichment — could be stdlib or unknown segment
            # Only flag if the symbol isn't known anywhere
            if symbol_name not in all_known_symbols:
                # Completely unknown — potential phantom
                # But only flag if it looks like a sibling import (starts with _)
                # and not a well-known import
                violation = ImportViolation(
                    symbol_name=symbol_name,
                    referenced_module=module_name,
                    consuming_segment=segment_id,
                    error_type="phantom",
                    message=(
                        f"'{symbol_name}' imported from '{module_name}' but "
                        f"no sibling segment exports this symbol, and module "
                        f"'{module_name}' has no enrichment data."
                    ),
                )
                result.violations.append(violation)
                result.passed = False
            continue

        available = module_symbol_sets[module_name]

        if symbol_name in available:
            # Valid import — symbol exists in the referenced module
            continue

        # Symbol NOT in this module. Is it anywhere else?
        if symbol_name in all_known_symbols:
            # Wrong module — it exists but in a different segment
            correct_seg = symbol_to_segment[symbol_name]
            # Find the module name for the correct segment
            correct_mod = ""
            for mod, segs in module_exports.items():
                if correct_seg in segs and symbol_name in segs[correct_seg]:
                    correct_mod = mod
                    break

            violation = ImportViolation(
                symbol_name=symbol_name,
                referenced_module=module_name,
                consuming_segment=segment_id,
                error_type="wrong_segment",
                available_symbols=sorted(available),
                correct_source=correct_mod or correct_seg,
                message=(
                    f"'{symbol_name}' imported from '{module_name}' but it "
                    f"is not exported by that module. It is exported by "
                    f"'{correct_mod or correct_seg}'."
                ),
            )
        else:
            # Phantom — doesn't exist anywhere
            violation = ImportViolation(
                symbol_name=symbol_name,
                referenced_module=module_name,
                consuming_segment=segment_id,
                error_type="phantom",
                available_symbols=sorted(available),
                message=(
                    f"'{symbol_name}' imported from '{module_name}' but this "
                    f"symbol does not exist in any sibling segment's exports. "
                    f"It appears to be hallucinated."
                ),
            )

        result.violations.append(violation)
        result.passed = False

    # Log results
    if result.passed:
        logger.info(
            "[import_validator] ✅ %s: %d cross-segment import(s) validated "
            "against %d sibling(s)",
            segment_id, result.symbols_checked, result.segments_scanned,
        )
    else:
        logger.warning(
            "[import_validator] ❌ %s: %d violation(s) in %d cross-segment "
            "import(s) — %s",
            segment_id,
            len(result.violations),
            result.symbols_checked,
            "; ".join(v.symbol_name for v in result.violations),
        )

    return result
