# FILE: tests/test_no_hardcoded_models.py
# Purpose: Lane A + Lane D regression guard — no literal model IDs
#          (gpt-/claude-/gemini-) in app/llm or app/memory source. Models
#          live in .env; the resolver is app.llm.frontier_models.
# Called-by: pytest
# Depends-on: stdlib only (AST scan of source files)
# Last-renovated: 2026-07-02
"""
Lane A Task 5 (2026-07-01) established the guard for the routing modules.
LANE D (2026-07-02) widened it to the WHOLE of app/llm and app/memory —
stream_utils, routing/core + job_classifier, complexity, model_families,
schemas, fallbacks, the image/vision stack, everything.

Scans every .py file for STRING LITERALS containing a model-ID prefix
(gpt- / claude- / gemini-). Exclusions, each deliberate and narrow:

* Docstrings AND bare string-expression statements (prose blocks) — an
  unassigned string literal cannot select a model.
* EXEMPT_FILES — whole files whose model strings are table KEYS, not routing
  selections (pricing table, context-window capability table) or prompt
  example filenames.
* TOKEN-level allowlists — matching is per model-shaped token: a literal
  passes iff EVERY token it contains is allowed. Family-prefix guards
  ("gpt-5" for startswith checks) and feature names in BUILD_IDs pass;
  a full model ID ("gpt-5.4", "claude-opus-4-8") extracts as its own token
  and always fails. Caveat: a bare legacy id equal to a family prefix
  (exactly "gpt-5") would pass — current-generation pins cannot.

Deliberately adding e.g. "claude-opus-4-6" to any scanned file fails with
the offending file, line and literal in the message.

app/rag is guarded by Lane C's tests/test_rag_no_hardcoded_models.py.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Whole trees under guard (acceptance: grep across app/llm, app/memory returns
# only whitelisted keys, docstrings and .env seeds).
SCAN_ROOTS = ("app/llm", "app/memory")

# Lane A's original file list — kept as an existence tripwire so a rename in
# the routing core is noticed here too.
LANE_A_FILES = [
    "app/llm/frontier_models.py",
    "app/llm/routing/chat_model_selection.py",
    "app/llm/routing/cognitive_escalation.py",
    "app/llm/routing/rag_fallback.py",
    "app/llm/routing/tier_momentum.py",
    "app/llm/routing/chat_routing.py",
    "app/llm/routing/conversational_gate.py",
    "app/llm/routing/codebase_context_bridge.py",
    # Lane D additions
    "app/llm/stream_utils.py",
    "app/llm/routing/core.py",
    "app/llm/job_classifier.py",
    "app/memory/complexity.py",
    "app/llm/model_families.py",
    "app/llm/model_roles.py",
]

# file (repo-relative, forward slashes) -> reason. The whole file is exempt.
EXEMPT_FILES: dict[str, str] = {
    # Pricing-table KEYS: they label prices for known models, never select a
    # model. Unknown models price at 0.0. (Outside the scan roots; listed for
    # documentation and because test_pricing_exemption_documented checks it.)
    "app/cost/cost_pricing.py":
        "pricing-table keys label costs; they never select a model",
    # Context-window capability table: family substrings map to token limits,
    # with a safe default for unknown families. Same class as pricing keys.
    "app/llm/token_budgeting.py":
        "context-window capability keys; unknown families get a safe default",
    # Prompt examples show a generated FILENAME shaped like gpt-<hash>.png —
    # not a model id at all.
    "app/llm/astra_filesystem_block.py":
        "prompt example filenames (gpt-<hash>.png), not model ids",
}

# Tokens allowed anywhere: family-prefix capability guards and non-model id
# prefixes. A full model ID never equals these.
GLOBAL_TOKEN_OK = {
    "gpt-",         # provider-audit family filter (frontier_models CLI) + filename prefixes
    "gpt-5",        # tool-eligibility prefix guard + prose ("Claude or GPT-5")
    "claude-opus",  # family guard (_job_classifier_classify source-model check)
    "gemini-fc-",   # gemini tool-call id prefix (chat_tool_loop, _streaming_utils_3)
}

# file -> extra allowed tokens, each with a documented reason.
TOKEN_WHITELIST: dict[str, set[str]] = {
    # BUILD_ID version tag names the gemini-thought-signatures FEATURE.
    "app/llm/chat_tool_loop.py": {"gemini-thought-signatures"},
    # Provider-family PREFIX guard, not a model selection: any claude-* model
    # is tool-eligible (2026-07-03 live incident — a frontier model missing
    # from the trust list ran toolless). Mirrors the "gpt-5" prefix guard,
    # which only passes unlisted because it carries no hyphen.
    "app/llm/chat_tool_registry.py": {"claude-"},
}

_MODEL_PREFIX = re.compile(r"\b(?:gpt-|claude-|gemini-)", re.IGNORECASE)
_MODEL_TOKEN = re.compile(r"\b((?:gpt|claude|gemini)-[A-Za-z0-9._\-]*)", re.IGNORECASE)


def _model_tokens(literal: str) -> set[str]:
    """Model-shaped tokens inside a string, lowercased, trailing dots/commas
    normalised off. A full ID extracts as its own token, so allowing the
    "gpt-5" prefix never hides "gpt-5.4"."""
    tokens: set[str] = set()
    for m in _MODEL_TOKEN.finditer(literal):
        tokens.add(m.group(1).lower().rstrip(".,"))
    return tokens


def _prose_nodes(tree: ast.AST) -> set[int]:
    """id()s of Constant nodes that are inert prose: docstrings and ANY bare
    string-expression statement. Unassigned strings cannot select a model."""
    prose_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                prose_ids.add(id(value))
    return prose_ids


def find_model_literals(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, literal) for every non-prose string literal in
    ``path`` that contains a model-ID prefix."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    prose_ids = _prose_nodes(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in prose_ids
            and _MODEL_PREFIX.search(node.value)
        ):
            hits.append((node.lineno, node.value))
    return hits


def _scan_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(sorted((REPO_ROOT / root).rglob("*.py")))
    return files


def test_lane_a_files_exist():
    """The scan list must track reality — a moved/renamed file fails here."""
    missing = [f for f in LANE_A_FILES if not (REPO_ROOT / f).exists()]
    assert not missing, f"Guard scan list is stale, files missing: {missing}"


def test_no_hardcoded_model_ids_in_llm_and_memory_trees():
    offenders: list[str] = []
    for path in _scan_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in EXEMPT_FILES:
            continue
        allowed = GLOBAL_TOKEN_OK | TOKEN_WHITELIST.get(rel, set())
        for lineno, literal in find_model_literals(path):
            bad = _model_tokens(literal) - allowed
            if not bad:
                continue
            snippet = literal if len(literal) <= 120 else literal[:117] + "..."
            offenders.append(f"  {rel}:{lineno}: tokens {sorted(bad)} in {snippet!r}")
    assert not offenders, (
        "Hardcoded model ID literal(s) found — models must resolve from .env "
        "(add to .env + read via app.llm.frontier_models, or whitelist here "
        "with justification):\n" + "\n".join(offenders)
    )


def test_pricing_exemption_documented():
    """cost_pricing.py must carry its exemption note in the header (Task 1f)."""
    path = REPO_ROOT / "app/cost/cost_pricing.py"
    head = path.read_text(encoding="utf-8")[:1500]
    assert "LANE D EXEMPTION" in head and "KEYS" in head, (
        "app/cost/cost_pricing.py lost its documented pricing-keys exemption header"
    )


def test_exempt_files_exist():
    """Exempt entries must not go stale."""
    missing = [f for f in EXEMPT_FILES if not (REPO_ROOT / f).exists()]
    assert not missing, f"EXEMPT_FILES entries missing on disk: {missing}"


def test_scanner_catches_a_deliberate_literal(tmp_path):
    """The acceptance case: adding \"claude-opus-4-6\" to a routing file must
    fail with file+line in the message."""
    bad = tmp_path / "bad_routing_module.py"
    bad.write_text(
        '"""Docstring mentioning claude-opus-4-6 is fine."""\n'
        "import os\n"
        "\n"
        "def pick():\n"
        "    return os.getenv('ARCHITECT_MODEL', 'claude-opus-4-6')\n",
        encoding="utf-8",
    )
    hits = find_model_literals(bad)
    assert hits == [(5, "claude-opus-4-6")], (
        f"Scanner must report exactly the non-docstring literal with its "
        f"line number; got {hits!r}"
    )
    # And the token layer must NOT excuse it via the claude-opus prefix.
    assert _model_tokens("claude-opus-4-6") - GLOBAL_TOKEN_OK, (
        "Token allowlist must never cover a full model id"
    )


def test_scanner_ignores_docstrings_and_non_model_strings(tmp_path):
    ok = tmp_path / "ok_module.py"
    ok.write_text(
        '"""Talks about gpt-5.4 and claude-opus-4-8 in prose only."""\n'
        "PROVIDER = 'openai'\n"
        "KEY = 'anthropic_opus'\n",
        encoding="utf-8",
    )
    assert find_model_literals(ok) == []


def test_family_prefix_tokens_do_not_hide_full_ids():
    """Allowed tokens are prefixes/feature names, never routable model IDs —
    and a full ID always extracts as its own (disallowed) token."""
    for allowed in GLOBAL_TOKEN_OK:
        assert not re.search(r"\d\.\d|-\d-|-\d{8}|preview|latest", allowed), (
            f"GLOBAL_TOKEN_OK entry {allowed!r} looks like a full model id"
        )
    for full_id in ("gpt-5.4", "gpt-5.4-mini", "claude-opus-4-8",
                    "claude-sonnet-5", "gemini-2.5-flash"):
        assert _model_tokens(full_id) - GLOBAL_TOKEN_OK, (
            f"{full_id!r} must not be covered by the token allowlist"
        )
