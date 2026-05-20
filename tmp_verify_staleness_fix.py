"""Verify Fix 1 — staleness detector tightening."""
import sys
sys.path.insert(0, r"D:\Orb")

from app.translation._tier0_web_search import (
    _check_staleness,
    check_web_search_trigger,
)

# Cases that SHOULD NOT trigger staleness (the bug we're fixing)
should_not_trigger = [
    # The actual bug — Taz's tutoring answer
    "Well, I think if it's been fed today's news via retrieval, then the answer is true. Because of that exact reason, that's how rag stands for retrieve, augment, generate.",
    # Common conversational uses of removed keywords
    "I know what you mean now, that makes sense",
    "Are you still working on it?",
    "I haven't done it yet",
    "The status update from yesterday",
    "I think the answer is to use retrieval",
    "Well, the reason is that the model can't know it directly",
    "Actually, I would say that's a good point",
    # "currents" should not trigger via "current"
    "The currents at Porthtowan were strong today",  # "today" still triggers but length-wise OK; let's see
    # "know" should not trigger via "now"
    "I don't know if that's right",
]

# Cases that SHOULD still trigger (legitimate web search via staleness)
should_trigger = [
    "What's the latest on the Iran situation?",
    "Bitcoin price today",
    "What's happening in Ukraine",
    "Trending topics on Twitter",
    "What's currently happening in parliament",
]

print("=" * 60)
print("Cases that should NOT trigger staleness:")
print("=" * 60)
all_pass_neg = True
for msg in should_not_trigger:
    result = _check_staleness(msg)
    status = "[PASS]" if result is None else f"[FAIL] (triggered: {result})"
    if result is not None:
        all_pass_neg = False
    print(f"{status} {msg[:80]}")

print()
print("=" * 60)
print("Cases that SHOULD still trigger staleness:")
print("=" * 60)
all_pass_pos = True
for msg in should_trigger:
    result = _check_staleness(msg)
    status = f"[PASS] (triggered: {result})" if result else "[FAIL] (did not trigger)"
    if result is None:
        all_pass_pos = False
    print(f"{status} {msg[:80]}")

print()
print("=" * 60)
print("Full check_web_search_trigger on the bug message:")
print("=" * 60)
bug_msg = "Well, I think if it's been fed today's news via retrieval, then the answer is true. Because of that exact reason, that's how rag stands for retrieve, augment, generate."
result = check_web_search_trigger(bug_msg)
print(f"matched: {result.matched}")
print(f"intent: {result.intent}")
print(f"rule_name: {result.rule_name}")
if result.matched:
    print("[FAIL] — bug message STILL classified as web search!")
else:
    print("[PASS] — bug message no longer classified as web search.")

print()
print(f"Negatives: {'ALL PASS' if all_pass_neg else 'SOME FAILED'}")
print(f"Positives: {'ALL PASS' if all_pass_pos else 'SOME FAILED'}")
