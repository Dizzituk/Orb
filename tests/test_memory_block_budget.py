# FILE: tests/test_memory_block_budget.py
# Purpose: Lock in the per-source caps + hard total budget that bound the
#          per-turn memory block (was ~127KB live, dominated by uncapped
#          preferences_text). Proves the block never blows past budget and that
#          a small block is left byte-identical.
# Called-by: pytest discovery
# Depends-on: app.llm.routing.memory_injection, app.llm.routing.memory_injection_sources
# Last-renovated: 2026-06-25 (new — memory-block size budget)
"""
Memory-block size budget tests.

Live measurement (2026-06-24) found the injected memory block at ~127KB, almost
all of it preferences_text: get_applicable_preferences returns EVERY matching
preference row (1,100+, ~96% auto-extracted doc_extract fact-dumps) and the
formatter rendered them all uncapped. The fix is two-layer:

  1. per-source char caps (here: _format_preferences max_chars), priority-ordered
     so the genuine behavioural rules survive and the extracted-fact tail is shed;
  2. a hard TOTAL budget in assemble_block (ASTRA_MEMORY_BLOCK_MAX_CHARS) as the
     backstop that bounds the assembled whole, trimming the low-priority tail.

These tests assert both, plus the invariant that a block already within budget is
returned byte-identical (so normal turns are unchanged).

Run with: pytest tests/test_memory_block_budget.py -v
"""

from types import SimpleNamespace

from app.llm.routing.memory_injection import _format_preferences
from app.llm.routing import memory_injection_sources as src


def _pref(key, value="true", strength="soft", status="active"):
    return SimpleNamespace(
        preference_key=key, preference_value=value,
        strength=strength, status=status, confidence=1.0,
    )


# =========================================================================
# Per-source cap: preferences
# =========================================================================

class TestPreferencesCap:
    def test_uncapped_renders_everything(self):
        prefs = [_pref(f"doc_extract:fact_{i}", "x" * 100) for i in range(50)]
        out = _format_preferences(prefs, max_chars=0)
        # 0 == unbounded: every preference is present, no omission marker.
        assert out.count("\n") == 49
        assert "omitted to fit memory budget" not in out

    def test_cap_bounds_size(self):
        prefs = [_pref(f"doc_extract:fact_{i}", "x" * 100) for i in range(200)]
        out = _format_preferences(prefs, max_chars=1000)
        # Bounded to ~the cap (one short omission-marker line of slack allowed).
        assert len(out) <= 1000 + 120
        assert "omitted to fit memory budget" in out

    def test_hard_rules_survive_the_cap(self):
        # One genuine curated hard rule buried after a wall of extracted facts.
        prefs = [_pref(f"doc_extract:fact_{i}", "x" * 300) for i in range(200)]
        prefs.append(_pref("no_git_commands", "true", strength="hard_rule"))
        out = _format_preferences(prefs, max_chars=600)
        # Priority ordering floats the hard rule to the front so the cap can't
        # drop it in favour of the bulk extracted-fact tail.
        assert "no_git_commands" in out
        assert "[REQUIRED]" in out

    def test_active_preferred_over_stale(self):
        prefs = [
            _pref("topic", "stale_value", strength="soft", status="superseded"),
            _pref("topic_active", "live_value", strength="soft", status="active"),
        ]
        # Tiny cap that fits only one line → the ACTIVE one must win.
        out = _format_preferences(prefs, max_chars=40)
        assert "live_value" in out


# =========================================================================
# Hard total budget: assemble_block
# =========================================================================

class TestAssembleBudget:
    def _big_facts(self, n=400):
        # Many short unique lines so dedup keeps them and the clip finds a
        # newline boundary to cut on.
        return "\n".join(f"- fact line number {i} about something" for i in range(n))

    def test_block_respects_total_budget(self, monkeypatch):
        monkeypatch.setenv("ASTRA_MEMORY_BLOCK_MAX_CHARS", "1500")
        block = src.assemble_block(
            preferences_text="• no_git_commands: true [REQUIRED]",
            facts_text=self._big_facts(),
        )
        assert len(block) <= 1500
        # Highest-priority source (preferences) is kept; the oversized tail is
        # clipped with the budget marker.
        assert "no_git_commands" in block
        assert "memory trimmed to fit budget" in block

    def test_small_block_is_byte_identical(self, monkeypatch):
        # A block well under budget must be returned unchanged (no trim marker),
        # so ordinary turns are untouched.
        monkeypatch.setenv("ASTRA_MEMORY_BLOCK_MAX_CHARS", "24000")
        small = src.assemble_block(
            preferences_text="• no_git_commands: true [REQUIRED]",
            facts_text="- a single small fact",
        )
        monkeypatch.setenv("ASTRA_MEMORY_BLOCK_MAX_CHARS", "0")  # disabled
        disabled = src.assemble_block(
            preferences_text="• no_git_commands: true [REQUIRED]",
            facts_text="- a single small fact",
        )
        assert small == disabled
        assert "memory trimmed to fit budget" not in small

    def test_budget_zero_disables_trim(self, monkeypatch):
        monkeypatch.setenv("ASTRA_MEMORY_BLOCK_MAX_CHARS", "0")
        block = src.assemble_block(facts_text=self._big_facts())
        # No budget → nothing trimmed, the whole (large) block survives.
        assert "memory trimmed to fit budget" not in block
        assert len(block) > 5000

    def test_lower_priority_tail_dropped_first(self, monkeypatch):
        # Budget fits the high-priority preferences segment but not the router
        # tail behind it → the tail is dropped, preferences kept.
        monkeypatch.setenv("ASTRA_MEMORY_BLOCK_MAX_CHARS", "90")
        block = src.assemble_block(
            preferences_text="• keep_me: yes [REQUIRED]",
            router_text="[MEMORY CONTEXT]\n  drop this low-priority tail\n[/MEMORY CONTEXT]",
        )
        assert len(block) <= 90
        assert "keep_me" in block
        assert "drop this low-priority tail" not in block
