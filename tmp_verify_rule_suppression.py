"""Verify Fix 2 — rejection feedback loop & rule suppression."""
import sys
import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, r"D:\Orb")

print("=" * 60)
print("Imports")
print("=" * 60)

from app.translation import rule_suppression, confirmation_log
from app.translation.rule_suppression import (
    is_rule_suppressed, get_rule_stats, list_suppressed_rules,
    force_refresh, NET_REJECTION_THRESHOLD, WINDOW_DAYS,
)
from app.translation.confirmation_log import log_confirmation_event
from app.translation.confidence_graduation import run_graduation, regenerate_learned_rules
from app.translation.schemas import CanonicalIntent
print("[OK] All imports succeed (run_graduation alias works)")

print()
print("=" * 60)
print("Step 1: alias check")
print("=" * 60)
assert run_graduation is regenerate_learned_rules, "alias not wired"
print(f"[OK] run_graduation IS regenerate_learned_rules")

# ---------------------------------------------------------------------------
# Sandbox the log file so we don't pollute D:\Orb\metrics
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    test_dir = Path(td)
    test_log = test_dir / "confirmation_events.jsonl"

    # Patch both modules to read/write the temp file
    rule_suppression._LOG_FILE = test_log
    confirmation_log._LOG_DIR = test_dir
    confirmation_log._LOG_FILE = test_log
    force_refresh()

    print()
    print("=" * 60)
    print("Step 2: rule_name flows through log_confirmation_event")
    print("=" * 60)

    log_confirmation_event(
        intent=CanonicalIntent.WEB_SEARCH,
        user_message_excerpt="What's the latest on Iran?",
        confirmed=False,
        confidence=0.75,
        conversation_id="test-1",
        rule_name="web_search_staleness_auto",
    )
    # Verify it landed in the log with rule_name
    with open(test_log, "r", encoding="utf-8") as f:
        line = f.readline()
        event = json.loads(line)
    assert event["rule_name"] == "web_search_staleness_auto"
    assert event["confirmed"] is False
    print("[OK] rule_name persisted in JSONL")

    print()
    print("=" * 60)
    print("Step 3: Suppression threshold")
    print("=" * 60)

    # 1 rejection — not yet suppressed
    force_refresh()
    assert not is_rule_suppressed("web_search_staleness_auto")
    stats = get_rule_stats("web_search_staleness_auto")
    print(f"  After 1 reject: rejects={stats['rejects']} confirms={stats['confirms']} net={stats['net']}")
    assert stats["net"] == 1
    print("[OK] 1 rejection — not suppressed (net=1, threshold=3)")

    # 2 more rejections — net=3, suppressed
    for i in range(2):
        log_confirmation_event(
            intent=CanonicalIntent.WEB_SEARCH,
            user_message_excerpt=f"another tutoring message {i}",
            confirmed=False,
            confidence=0.75,
            conversation_id=f"test-{i+2}",
            rule_name="web_search_staleness_auto",
        )
    force_refresh()
    stats = get_rule_stats("web_search_staleness_auto")
    print(f"  After 3 rejects: rejects={stats['rejects']} confirms={stats['confirms']} net={stats['net']}")
    assert stats["net"] == 3
    assert is_rule_suppressed("web_search_staleness_auto")
    print(f"[OK] 3 rejections, net={stats['net']} >= {NET_REJECTION_THRESHOLD} — SUPPRESSED")

    print()
    print("=" * 60)
    print("Step 4: Confirmations cancel rejections (net rejections)")
    print("=" * 60)

    # 1 confirmation — net drops to 2, no longer suppressed
    log_confirmation_event(
        intent=CanonicalIntent.WEB_SEARCH,
        user_message_excerpt="legit search query",
        confirmed=True,
        confidence=0.75,
        conversation_id="test-conf",
        rule_name="web_search_staleness_auto",
    )
    force_refresh()
    stats = get_rule_stats("web_search_staleness_auto")
    print(f"  After 3 rejects + 1 confirm: rejects={stats['rejects']} confirms={stats['confirms']} net={stats['net']}")
    assert stats["net"] == 2
    assert not is_rule_suppressed("web_search_staleness_auto")
    print("[OK] Confirmation reduced net below threshold — UNSUPPRESSED")

    print()
    print("=" * 60)
    print("Step 5: Other rules unaffected")
    print("=" * 60)

    assert not is_rule_suppressed("web_search_search_web_for")
    assert not is_rule_suppressed("look_up")
    assert not is_rule_suppressed(None)
    assert not is_rule_suppressed("")
    print("[OK] Suppression is per-rule — other rules and empty inputs unaffected")

    print()
    print("=" * 60)
    print("Step 6: Window expiry — old rejections age out")
    print("=" * 60)

    # Write a rejection with an old timestamp (40 days ago)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    new_event = {
        "timestamp": old_ts,
        "intent": "WEB_SEARCH",
        "user_message_excerpt": "ancient rejection",
        "confirmed": False,
        "confidence": 0.75,
        "conversation_id": "old-test",
        "rule_name": "old_rule_to_be_ignored",
    }
    with open(test_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(new_event) + "\n")

    # Add 2 more old rejections for the same rule — total 3 old rejections
    for i in range(2):
        old_event = dict(new_event)
        old_event["user_message_excerpt"] = f"ancient {i}"
        with open(test_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(old_event) + "\n")

    force_refresh()
    stats = get_rule_stats("old_rule_to_be_ignored")
    print(f"  3 rejections from 40 days ago: rejects={stats['rejects']} confirms={stats['confirms']} net={stats['net']}")
    assert stats["net"] == 0  # Outside 30-day window
    assert not is_rule_suppressed("old_rule_to_be_ignored")
    print(f"[OK] Rejections older than {WINDOW_DAYS} days don't count")

    print()
    print("=" * 60)
    print("Step 7: list_suppressed_rules diagnostic")
    print("=" * 60)

    # Add a fresh rule with 4 net rejects
    for i in range(4):
        log_confirmation_event(
            intent=CanonicalIntent.WEB_SEARCH,
            user_message_excerpt=f"misfire {i}",
            confirmed=False,
            confidence=0.75,
            conversation_id=f"new-{i}",
            rule_name="some_misbehaving_rule",
        )
    force_refresh()
    suppressed = list_suppressed_rules()
    print(f"  Currently suppressed rules: {list(suppressed.keys())}")
    assert "some_misbehaving_rule" in suppressed
    assert suppressed["some_misbehaving_rule"]["net"] == 4
    print("[OK] list_suppressed_rules returns {rule: stats} for diagnostics")

    print()
    print("=" * 60)
    print("Step 8: Mtime caching — repeated checks are cheap")
    print("=" * 60)

    # First call refreshes; subsequent calls hit the cache.
    t0 = time.perf_counter()
    for _ in range(1000):
        is_rule_suppressed("some_misbehaving_rule")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  1000 lookups: {elapsed_ms:.2f}ms total ({elapsed_ms/1000*1000:.2f}us each)")
    assert elapsed_ms < 200, f"Hot path too slow: {elapsed_ms}ms for 1000 lookups"
    print("[OK] Cache hot-path under 200us per lookup")

print()
print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
