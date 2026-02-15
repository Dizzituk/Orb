import json

state_path = r'D:\Orb\jobs\jobs\sg-c5c1368b\state.json'

with open(state_path, 'r', encoding='utf-8') as f:
    state = json.load(f)

print('BEFORE overall_status:', state.get('overall_status'))

for seg_id in state['segments']:
    state['segments'][seg_id]['status'] = 'approved'
    state['segments'][seg_id]['started_at'] = None
    state['segments'][seg_id]['completed_at'] = None
    state['segments'][seg_id]['error'] = None
    state['segments'][seg_id]['output_files'] = []
    state['segments'][seg_id]['evidence_provided_to'] = []

state['overall_status'] = 'running'
for key in ['integration_check', 'phase_checkout_boot', 'phase_checkout_error']:
    if key in state:
        state[key] = None

with open(state_path, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)

# Verify
with open(state_path, 'r', encoding='utf-8') as f:
    v = json.load(f)
print('AFTER overall_status:', v.get('overall_status'))
for sid in v['segments']:
    print(f'  {sid}: {v["segments"][sid]["status"]}')
