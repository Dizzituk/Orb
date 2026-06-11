import sys, os, asyncio
os.chdir("D:/Orb"); sys.path.insert(0, "D:/Orb")
from app.pipeline_v2.promote import _is_allowed, _host_abs, _manifest_paths_for_job, build_promote_plan
assert _is_allowed("D:/Orb/app/x.py") and _is_allowed("d:\\orb-desktop\\src\\a.tsx".replace("\\","/"))
assert not _is_allowed("C:/Windows/system32/evil.dll") and not _is_allowed("D:/Astra Android Folder/x.kt")
assert _host_abs("app/pipeline_v2/promote.py").lower().startswith("d:/orb/")
files = _manifest_paths_for_job("sg-f2a46eb5")
print("manifest scope files:", len(files), files[:4])
plan = asyncio.run(build_promote_plan("sg-f2a46eb5"))
print("plan ok:", plan["ok"], "| error:", plan.get("error", "none")[:80], "| files:", len(plan.get("files", [])))
plan2 = asyncio.run(build_promote_plan("no-such-job"))
assert plan2["ok"] is False and "No manifest" in plan2["error"]
print("J15 SMOKE PASS")
