# Purpose: test phase1 verdict smoke
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.pipeline_v2.final_verdict, app.pipeline_v2.spec_review.context
# Last-renovated: 2026-06-11
import sys, asyncio
sys.path.insert(0, "D:/Orb")

# --- Job 3: final verdict ---
from app.pipeline_v2.final_verdict import build_final_verdict, VERIFIED, FAILED, CONFLICTING

class FakeReview:
    def __init__(self, crit=0, major=0, unmet=None):
        class S:
            def __init__(self, v): self.value = v
        class F:
            def __init__(self, v): self.severity = S(v)
        self.findings = [F("critical")]*crit + [F("major")]*major
        self.requirements_unmet = unmet or []
        self.summary = "fake"

class FakeContract:
    def __init__(self, ok): self._ok = ok
    def is_passing(self): return self._ok
    def summary(self): return "targets=1/1" if self._ok else "targets=0/1"

class FakeBVL:
    def __init__(self, ok):
        self.all_signed_off = ok
        self.blocked_components = []

class FakeResult:
    def __init__(self, bvl, review):
        self.bvl_report = bvl
        self.spec_review_report = review

# conflict: contract pass vs review criticals
v = build_final_verdict(FakeResult(None, FakeReview(crit=7)), FakeContract(True), {"acceptance_criteria": ["chat prompt opens", "private logs nothing"]})
assert v.overall == CONFLICTING, v.overall
assert "spec_review" in v.conflict_note and "contract" in v.conflict_note, v.conflict_note

# all green
v2 = build_final_verdict(FakeResult(FakeBVL(True), FakeReview()), FakeContract(True), {})
assert v2.overall == VERIFIED, v2.overall

# bvl fail, no passes -> FAILED
v3 = build_final_verdict(FakeResult(FakeBVL(False), None), None, {})
assert v3.overall == FAILED, v3.overall

# unmet criterion matching
v4 = build_final_verdict(FakeResult(None, FakeReview(major=1, unmet=["private logs nothing"])), None, {"acceptance_criteria": ["chat prompt opens", "private logs nothing"]})
rows = {c.criterion: c.status for c in v4.criteria}
assert rows["private logs nothing"] == "FAILED", rows
assert rows["chat prompt opens"] == "UNTESTED", rows
print("final_verdict OK")
for line in v.render_lines(): print(line)

# --- Job 2: file collection ---
from app.pipeline_v2.spec_review.context import _collect_feature_files
spec = {"segments": [{"file_scope": ["a/B.kt", "c/d.kt"]}]}
manifest = {"segments": [{"file_scope": ["A/b.kt", "e/f.kt"]}]}
out = _collect_feature_files(spec, manifest, ["x/y.xml", "e\\f.kt"])
assert out[0] == "x/y.xml" and "c/d.kt" in out, out
assert len([p for p in out if p.replace("\\","/").lower()=="e/f.kt"]) == 1, out
assert len([p for p in out if p.lower().replace("\\","/")=="a/b.kt"]) == 1, out
print("collect_feature_files OK:", out)
print("PHASE 1 SMOKE: ALL PASS")
