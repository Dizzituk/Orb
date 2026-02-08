"""Test _extract_file_scope_from_spec with various path formats."""
import sys, os
sys.path.insert(0, r"D:\Orb")
from app.pot_spec.grounded.spec_runner import _extract_file_scope_from_spec

# Test 1: Relative paths (existing behaviour) — no backticks, plain markdown
spec1 = """## Files to Modify
- app/services/transcription_service.py — Add endpoint
- src/components/VoiceInput.tsx — New component
"""
r1 = _extract_file_scope_from_spec(spec1)
print(f"Test 1 (relative): {len(r1)} paths: {r1}")
assert len(r1) >= 2, f"Expected >=2, got {r1}"

# Test 1b: Backtick-wrapped relative paths
spec1b = """## Files
- `app/services/transcription_service.py` — modify
- `src/components/VoiceInput.tsx` — create
"""
r1b = _extract_file_scope_from_spec(spec1b)
print(f"Test 1b (backtick rel): {len(r1b)} paths: {r1b}")
assert len(r1b) >= 2, f"Expected >=2, got {r1b}"

# Test 2: Absolute Windows paths (THE FIX)
spec2 = r"""## Integration Points
- D:\Orb\app\services\transcription_service.py — modify
- D:\Orb\app\routers\transcribe.py — create
- D:\orb-desktop\src\components\VoiceInput.tsx — create
"""
r2 = _extract_file_scope_from_spec(spec2)
print(f"Test 2 (absolute): {len(r2)} paths: {r2}")
has_abs = any("D:" in p or "d:" in p for p in r2)
has_rel = any(p.startswith("app") or p.startswith("src") for p in r2)
assert has_abs, f"No absolute paths found: {r2}"
assert has_rel, f"No relative paths extracted: {r2}"

# Test 3: Mixed absolute + relative  
spec3 = r"""- D:\Orb\app\config\local_ai_config.py
- app/routers/voice_status.py
- src/hooks/useVoiceRecorder.ts
"""
r3 = _extract_file_scope_from_spec(spec3)
print(f"Test 3 (mixed): {len(r3)} paths: {r3}")
assert len(r3) >= 3, f"Expected >=3, got {r3}"

# Test 4: Forward-slash absolute paths
spec4 = "D:/Orb/app/services/model_manager.py is the target"
r4 = _extract_file_scope_from_spec(spec4)
print(f"Test 4 (forward slash): {len(r4)} paths: {r4}")
assert len(r4) >= 1, f"Expected >=1, got {r4}"

# Test 5: Empty spec returns empty
r5 = _extract_file_scope_from_spec(None)
assert r5 == [], f"Expected empty, got {r5}"
r6 = _extract_file_scope_from_spec("")
assert r6 == [], f"Expected empty, got {r6}"
print("Test 5 (empty): OK")

# Test 6: No source file paths → empty
spec6 = "Just some text about the project with no file paths."
r6 = _extract_file_scope_from_spec(spec6)
assert r6 == [], f"Expected empty for no-path text, got {r6}"
print("Test 6 (no paths): OK")

print()
print("=" * 50)
print("ALL Gap 4 tests PASSED")
print("=" * 50)
