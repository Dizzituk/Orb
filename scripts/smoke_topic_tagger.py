"""Smoke test for the topic tagger."""
import sys
sys.path.insert(0, r'D:\Orb')
from app.astra_memory.topic_tagger import extract_tags, extract_entities

samples = [
    ('TAO investment',
     'TAO looks like one of the more serious AI-crypto bets, but it has high '
     'subnet concentration risk.'),
    ('Polanski politics',
     'Zack Polanski leading the Green party is interesting given the post-May '
     '2026 surge.'),
    ('Yodel work',
     'Route TO09 took 11.5 hours today, parcel count 184 from Redruth depot.'),
    ('Leigh Day legal',
     'The Leigh Day claim hinges on s.230(3)(b) worker status, not pay-level.'),
    ('surfing lifestyle',
     'Surfed twice this weekend at Porthtowan, longboard.'),
    ('code',
     'def upsert_hot_index(db, record_type, record_id): pass # python module'),
    ('Portugal D7',
     'D7 visa requires ~870 EUR/month passive income. Algarve looks doable.'),
    ('AGI economics',
     'Alex Imas DeepMind hire matters because AGI economics changes the '
     'displacement timeline.'),
    ('content',
     'Working on the Man in the Van TikTok thumbnail and Facebook caption.'),
    ('mixed',
     'How does TAO valuation respond under different AGI displacement '
     'scenarios? Should I rebalance the portfolio?'),
]

for label, text in samples:
    print(f'\n[{label}]')
    print(f'  tags:     {extract_tags(text)}')
    print(f'  entities: {extract_entities(text)}')
