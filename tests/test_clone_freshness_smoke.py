import sys, os, asyncio
os.chdir("D:/Orb"); sys.path.insert(0, "D:/Orb")
from app.pipeline_v2 import clone_freshness as cf

# is_self_build routing
class P:
    def __init__(self, pid, root): self.project_id, self.project_root = pid, root
assert cf.is_self_build(P("astra-backend", "D:/Orb"))
assert cf.is_self_build(P("custom", "D:\\orb-desktop\\"))
assert not cf.is_self_build(P("astra-bridge", "D:/Astra Android Folder/Astra-Bridge"))
assert not cf.is_self_build(None)

HOST = {}
CLONE = {}
def fake_host(root): return HOST[root]
async def fake_clone(root): return CLONE[root]
cf._host_git = fake_host
cf._clone_git = fake_clone

def setup(h_head, h_dirty, c_head, c_dirty, c_err=""):
    for _, root in cf.REPO_PAIRS:
        HOST[root] = (h_head, h_dirty, "")
        CLONE[root] = (c_head, c_dirty, c_err)

H = "a"*40; B = "b"*40

# 1 fresh everywhere -> ok
setup(H, 0, H, 0)
r = asyncio.run(cf.check_clone_freshness())
assert r.ok, r.reason

# 2 head mismatch -> block
setup(H, 0, B, 0)
r = asyncio.run(cf.check_clone_freshness())
assert not r.ok and "different commit" in r.reason

# 3 host dirty -> block
setup(H, 186, H, 0)
r = asyncio.run(cf.check_clone_freshness())
assert not r.ok and "uncommitted" in r.reason

# 4 clone dirty -> block
setup(H, 0, H, 81)
r = asyncio.run(cf.check_clone_freshness())
assert not r.ok and "debris" in r.reason

# 5 sandbox down -> block with reason
setup(H, 0, "", -1, "sandbox unreachable: timeout")
r = asyncio.run(cf.check_clone_freshness())
assert not r.ok and "unreachable" in r.reason

# 6 env skip -> ok
os.environ["ASTRA_SKIP_CLONE_FRESHNESS"] = "1"
r = asyncio.run(cf.check_clone_freshness())
assert r.ok and "skipped" in r.reason
del os.environ["ASTRA_SKIP_CLONE_FRESHNESS"]

# render shape
setup(H, 0, B, 3)
r = asyncio.run(cf.check_clone_freshness())
lines = r.render_lines()
assert any("CLONE FRESHNESS GATE" in l for l in lines) and any("refused" in l for l in lines)
print("FRESHNESS GATE SMOKE: ALL 6 SCENARIOS + RENDER PASS")
