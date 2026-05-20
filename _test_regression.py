"""Regression sanity: verify Fix 2 didn't break any existing target resolution cases."""
import sys
sys.path.insert(0, '.')

from app.pipeline_v2.target_registry import (
    resolve_project_from_message,
    detect_all_projects_from_message,
)

print('Regression tests:')
cases = [
    ('Working on Astra Bridge TTS playback and wake word', 'astra-bridge'),
    ('Fix bug in app/pipeline_v2/spec_runner.py', 'astra-backend'),
    ('Update the React component in orb-desktop', 'astra-frontend'),
    ('Build target: driver-copilot\nNew feature X', 'driver-copilot'),
    ('Let\'s work on Astra itself - improve the overwatcher', 'astra-backend'),
    ('Driver CoPilot needs a Yodel app scraper for round planner', 'driver-copilot'),
    ('Astra Bridge ExoPlayer chatterbox audio pipeline', 'astra-bridge'),
]
all_pass = True
for text, expected in cases:
    result = resolve_project_from_message(text)
    actual = result.project_id if result else None
    status = 'OK ' if actual == expected else 'FAIL'
    print(f'  [{status}] "{text[:50]}" -> {actual} (expected {expected})')
    if actual != expected:
        all_pass = False

print('\nAmbiguity tests (should return None or plausible):')
ambig = [
    'Hello',
    'Can you help me?',
    'Fix the thing we discussed',
]
for text in ambig:
    result = resolve_project_from_message(text)
    actual = result.project_id if result else 'None'
    print(f'  "{text}" -> {actual}')

print('\nALL REGRESSIONS PASS' if all_pass else '\nREGRESSION FAILURES DETECTED')
