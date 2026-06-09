"""Check the over-matching risk for new investment-related keywords."""
import sys
sys.path.insert(0, r'D:\Orb')
from app.astra_memory.topic_tagger import extract_tags

# Test phrases that COULD over-match the investment keywords but shouldn't
non_invest = [
    "Let me share my thoughts on this design.",
    "We have a stake in making this work.",
    "Hold on, let me check that position.",
    "I'll allocate some time to this.",
    "The stock answer is X.",
    "What's the share count for the table?",
    "The yield strength of the bridge...",
    "Position the cursor at the start.",
    "Take a look at the dividend distribution algorithm in line 24.",  # code
]
for q in non_invest:
    tags = extract_tags(q)
    flagged = 'investments' in tags
    marker = '!!!OVERMATCH' if flagged else 'ok'
    print(f"  {marker:13s} {tags!r:50s} :: {q}")

# Real investment queries — must tag
invest = [
    "Should I rebalance my investments?",
    "What is the current allocation?",
    "I want to add to my TAO position.",
    "My portfolio is down 5% this week.",
]
print()
print("Real investment queries — these MUST tag as 'investments':")
for q in invest:
    tags = extract_tags(q)
    ok = 'investments' in tags
    marker = 'ok' if ok else '!!!MISS'
    print(f"  {marker:13s} {tags!r:50s} :: {q}")
