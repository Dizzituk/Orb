"""Comprehensive over-match audit across all topic domains."""
import sys
sys.path.insert(0, r'D:\Orb')
from app.astra_memory.topic_tagger import extract_tags

# Decoy phrases — innocent English that historically over-matched.
# Each entry: (sentence, MUST-NOT-tag)
decoys = [
    # politics false positives
    ("It's a labour-intensive process.", "politics"),
    ("Reform the schema next week.", "politics"),
    ("The party is green-themed.", "politics"),
    ("Let me focus on the policy file structure.", "politics"),
    ("It's a major Conservative estimate for that.", "politics"),
    # ai_society false positives
    ("Show me the displacement of the cursor.", "ai_society"),
    ("AGI Industries was a 90s rock band.", "ai_society"),
    # work false positives
    ("My route through the codebase starts here.", "work"),
    ("Ming the merciless was the Flash Gordon villain.", "work"),
    ("Parcel out the work between two threads.", "work"),
    # legal false positives
    ("My claim is that the bug is in line 24.", "legal"),
    ("Section 230 of the design doc covers this.", "legal"),
    ("Let me make a claim about the test result.", "legal"),
    # portugal false positives
    ("Portuguese man-of-war jellyfish.", "portugal"),
    ("This script targets the algarve namespace.", "portugal"),
    # content false positives
    ("Add a thumbnail to the README.", "content"),
    ("The audience for this doc is engineers.", "content"),
    # finance false positives
    ("Track this expense in the test report.", "finance"),
    ("Budget the time properly.", "finance"),
    # lifestyle false positives
    ("PT routine: parse, transform.", "lifestyle"),
    ("I had a sugar craving for the design.", "lifestyle"),
    ("My fitness function is too slow.", "lifestyle"),
    # astra_dev false positives
    ("Implementer pattern is from the GoF book.", "astra_dev"),
]

# Real queries — MUST tag with the given domain
realq = [
    ("Reform UK has gained seats in the local election.", "politics"),
    ("Zack Polanski leads the Green Party now.", "politics"),
    ("How is the AGI displacement timeline looking?", "ai_society"),
    ("Alex Imas wrote about AGI economics.", "ai_society"),
    ("My Yodel route TO09 was 11 hours today.", "work"),
    ("InPost delivery work in Cornwall.", "work"),
    ("The Leigh Day claim under ERA 1996.", "legal"),
    ("Worker status under s.230(3)(b).", "legal"),
    ("The D7 visa requires passive income.", "portugal"),
    ("We're planning to move to Lagos, Algarve.", "portugal"),
    ("My Facebook content strategy.", "content"),
    ("Working on the Man in the Van TikTok.", "content"),
    ("My self-assessment tax return is due.", "finance"),
    ("Making Tax Digital affects sole traders.", "finance"),
    ("Surf session at Porthtowan today.", "lifestyle"),
    ("My ADHD meds are due for a refill.", "lifestyle"),
    ("The Scaffold Engine writes deterministic files.", "astra_dev"),
    ("HotIndex retrieval cost is TINY.", "astra_dev"),
]

print("=== DECOYS (must NOT tag with given domain) ===")
fp = 0
for sentence, must_not in decoys:
    tags = extract_tags(sentence)
    bad = must_not in tags
    if bad:
        fp += 1
        print(f"  !!OVERMATCH '{must_not}'  {tags}  :: {sentence}")
    else:
        print(f"  ok                       {tags}  :: {sentence}")

print()
print("=== REAL QUERIES (must tag with given domain) ===")
fn = 0
for sentence, must in realq:
    tags = extract_tags(sentence)
    ok = must in tags
    if not ok:
        fn += 1
        print(f"  !!MISS '{must}'  {tags}  :: {sentence}")
    else:
        print(f"  ok            {tags}  :: {sentence}")

print()
print(f"Over-matches: {fp} / {len(decoys)}")
print(f"Misses:       {fn} / {len(realq)}")
