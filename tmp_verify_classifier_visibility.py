"""End-to-end verify: Fix 1 + Fix 3 wiring."""
import sys
sys.path.insert(0, r"D:\Orb")

print("=" * 60)
print("Imports")
print("=" * 60)

from app.translation._tier0_web_search import _check_staleness, check_web_search_trigger
from app.translation.recent_decisions import (
    record_decision, mark_status, get_recent_decisions,
    build_decisions_block, clear_conversation,
    STATUS_PENDING, STATUS_CONFIRMED, STATUS_REJECTED, STATUS_AUTO,
)
from app.translation.translator import Translator
from app.llm.routing.prompt_builders import build_full_context, _CONVERSATIONAL_GUIDELINES
from app.llm import stream_router  # parse check
print("[OK] All target modules import")

print()
print("=" * 60)
print("Fix 1: Staleness no longer fires on bug message")
print("=" * 60)
bug_msg = "Well, I think if it's been fed today's news via retrieval, then the answer is true."
result = check_web_search_trigger(bug_msg)
print(f"  matched: {result.matched}")
assert not result.matched, "Bug message still triggers — FAILED"
print("[OK] Bug message no longer classified as web search")

print()
print("=" * 60)
print("Fix 3: Recent decisions roundtrip")
print("=" * 60)
test_conv = "test_conv_999"

# Clean slate
clear_conversation(test_conv)
assert get_recent_decisions(test_conv) == []
print("[OK] Clean state")

# Record a pending decision (gate fired)
record_decision(
    conversation_id=test_conv,
    intent="WEB_SEARCH",
    rule_name="web_search_staleness_auto",
    reason="Auto web search (stale topic: temporal_signal:today)",
    message_excerpt=bug_msg,
    confidence=0.75,
    status=STATUS_PENDING,
)
decisions = get_recent_decisions(test_conv)
assert len(decisions) == 1
assert decisions[0].status == STATUS_PENDING
assert decisions[0].intent == "WEB_SEARCH"
assert decisions[0].rule_name == "web_search_staleness_auto"
print("[OK] PENDING decision recorded")

# Mark as rejected (user clicked No)
mark_status(test_conv, STATUS_REJECTED, message_excerpt=bug_msg)
decisions = get_recent_decisions(test_conv)
assert decisions[0].status == STATUS_REJECTED
print("[OK] Status marked REJECTED")

# Build the injection block
block = build_decisions_block(test_conv)
print()
print("Injection block produced:")
print("-" * 40)
print(block)
print("-" * 40)
assert "WEB_SEARCH" in block
assert "web_search_staleness_auto" in block
assert "rejected" in block
assert "temporal_signal:today" in block
print("[OK] Block contains expected metadata")

# Idempotency: recording the same message_excerpt updates rather than dups
record_decision(
    conversation_id=test_conv,
    intent="WEB_SEARCH",
    rule_name="web_search_staleness_auto",
    reason="(retry)",
    message_excerpt=bug_msg,
    confidence=0.75,
    status=STATUS_PENDING,
)
decisions = get_recent_decisions(test_conv)
assert len(decisions) == 1, f"Expected 1 decision after idempotent re-record, got {len(decisions)}"
print("[OK] Idempotent on identical message excerpt")

# Cap at 3
clear_conversation(test_conv)
for i in range(5):
    record_decision(
        conversation_id=test_conv,
        intent=f"INTENT_{i}",
        rule_name=f"rule_{i}",
        reason=f"reason_{i}",
        message_excerpt=f"message {i}",
        confidence=0.5,
    )
decisions = get_recent_decisions(test_conv)
assert len(decisions) == 3, f"Expected cap of 3, got {len(decisions)}"
assert decisions[0].intent == "INTENT_2"
assert decisions[2].intent == "INTENT_4"
print("[OK] Cap of 3 most-recent decisions enforced")

# Empty conversation produces empty block (not None, not crash)
clear_conversation(test_conv)
assert build_decisions_block(test_conv) == ""
assert build_decisions_block(None) == ""
assert build_decisions_block("") == ""
print("[OK] Empty / missing conversation handled cleanly")

print()
print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
