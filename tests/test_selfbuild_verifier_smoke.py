# Purpose: test selfbuild verifier smoke
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.pipeline_v2.sandbox_tools, app.pipeline_v2.verifier_agent, app.pipeline_v2.verifier_agent.agent, app.pipeline_v2.verifier_agent.backend_tools
# Last-renovated: 2026-06-11
import sys, os, asyncio
os.chdir("D:/Orb"); sys.path.insert(0, "D:/Orb")
from app.pipeline_v2.verifier_agent import backend_tools as bt

names = [t["function"]["name"] for t in bt.BACKEND_TOOLS]
assert names == ["backend_start", "backend_stop", "http_request", "backend_logs", "wait"], names

# script construction sanity
s = bt._start_script(8000)
assert "ORB_MASTER_KEY" in s and bt.THROWAWAY_CLONE_KEY in s and "'8000'" in s and "8765" in s
h = bt._http_script("POST", "/system/uptime", '{"a":"it''s"}', 8000)
assert "http://127.0.0.1:8000/system/uptime" in h and "-Method POST" in h and "ContentType" in h

CAPTURED = {}
async def fake_run_shell(cmd, timeout_sec=30, profile=None):
    CAPTURED["cmd"], CAPTURED["timeout"], CAPTURED["profile"] = cmd, timeout_sec, profile
    return {"stdout": "STATUS 200\n{\"uptime\": 42}", "stderr": "", "returncode": 0}

import app.pipeline_v2.sandbox_tools as st
st.run_shell = fake_run_shell

class P: project_id, project_root, language = "astra-backend", "D:/Orb", "python"

r = asyncio.run(bt.execute_backend_tool("http_request", {"method": "get", "path": "system/uptime"}, P()))
assert "STATUS 200" in r and CAPTURED["profile"].__class__ is P and "/system/uptime" in CAPTURED["cmd"]

r2 = asyncio.run(bt.execute_backend_tool("http_request", {"method": "GET", "path": "/x", "port": 8765}, P()))
assert "off limits" in r2
r3 = asyncio.run(bt.execute_backend_tool("nonsense", {}, P()))
assert "unknown" in r3
r4 = asyncio.run(bt.execute_backend_tool("wait", {"seconds": 0.05}, P()))
assert "waited" in r4
r5 = asyncio.run(bt.execute_backend_tool("backend_start", {}, P()))
assert CAPTURED["timeout"] == 45

# prompt + toolset dispatch
from app.pipeline_v2.verifier_agent.agent import _system_prompt
p_self = _system_prompt(P())
assert "CLONE OF YOURSELF" in p_self and "CHECKOUT VERDICT" in p_self
class A: project_id, project_root, language, package_name = "astra-bridge", "D:/Astra Android Folder/Astra-Bridge", "kotlin", "com.astra.astrabridge"
p_droid = _system_prompt(A())
assert "EMULATOR" in p_droid and "CHECKOUT VERDICT" in p_droid
print("SELF-BUILD VERIFIER SMOKE: ALL PASS (5 tools, scripts, routing, guards, prompt dispatch)")
