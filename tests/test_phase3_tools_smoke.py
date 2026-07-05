# Purpose: test phase3 tools smoke
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.pipeline_v2, app.pipeline_v2.llm_tool_defs, app.pipeline_v2.llm_tools_anthropic, app.pipeline_v2.sandbox_tools
# Last-renovated: 2026-06-11
import sys, asyncio
sys.path.insert(0, "D:/Orb")

# --- Job 9: edit_file uniqueness + backup logic (IO monkeypatched) ---
from app.pipeline_v2 import sandbox_tools

FAKE_FS = {}

async def fake_read(path, profile=None):
    return FAKE_FS.get(path)

async def fake_write(path, content, profile=None):
    FAKE_FS[path] = content
    return True

async def fake_exists(path, profile=None):
    return path in FAKE_FS

# Derek p5 (2026-07-04): capture originals so this module can RESTORE them.
# These patches used to leak for the rest of the pytest process, silently
# redirecting every later test's file IO into FAKE_FS.
_orig_read, _orig_write, _orig_exists = (
    sandbox_tools.read_file, sandbox_tools.write_file, sandbox_tools.file_exists,
)
sandbox_tools.read_file = fake_read
sandbox_tools.write_file = fake_write
sandbox_tools.file_exists = fake_exists

async def main():
    FAKE_FS.clear()
    FAKE_FS["app/x.py"] = "alpha\nbeta\ngamma\nbeta-tail\n"

    # unique match -> ok, backup written
    ok, msg = await sandbox_tools.edit_file("app/x.py", "gamma", "GAMMA")
    assert ok, msg
    assert FAKE_FS["app/x.py"] == "alpha\nbeta\nGAMMA\nbeta-tail\n", FAKE_FS["app/x.py"]
    baks = [k for k in FAKE_FS if k.startswith("app/x.py.bak-")]
    assert len(baks) == 1, baks
    assert FAKE_FS[baks[0]] == "alpha\nbeta\ngamma\nbeta-tail\n"

    # zero matches -> refused
    ok2, msg2 = await sandbox_tools.edit_file("app/x.py", "delta", "D")
    assert not ok2 and "not found" in msg2, (ok2, msg2)

    # multiple matches -> refused with count
    ok3, msg3 = await sandbox_tools.edit_file("app/x.py", "beta", "B")
    assert not ok3 and "2 times" in msg3, (ok3, msg3)

    # file content unchanged by the two refusals
    assert FAKE_FS["app/x.py"] == "alpha\nbeta\nGAMMA\nbeta-tail\n"

    # second edit same day -> backup NOT overwritten (preserves pre-session state)
    ok4, msg4 = await sandbox_tools.edit_file("app/x.py", "alpha", "ALPHA")
    assert ok4, msg4
    assert FAKE_FS[baks[0]] == "alpha\nbeta\ngamma\nbeta-tail\n", "backup must keep original"

    # empty old_str refused; identical strings refused; missing file refused
    ok5, _ = await sandbox_tools.edit_file("app/x.py", "", "x")
    ok6, _ = await sandbox_tools.edit_file("app/x.py", "same", "same")
    ok7, msg7 = await sandbox_tools.edit_file("app/none.py", "a", "b")
    assert not ok5 and not ok6 and not ok7 and "not found" in msg7.lower()

    print("edit_file OK (7 paths)")

asyncio.run(main())

# --- Job 8: tool-schema conversion + result-content building (no API) ---
from app.pipeline_v2.llm_tools_anthropic import to_anthropic_tools, _tool_result_content
from app.pipeline_v2.llm_tool_defs import TOOLS

at = to_anthropic_tools(TOOLS)
names = [t["name"] for t in at]
assert names == ["read_file", "write_file", "edit_file", "run_shell"], names
assert all("input_schema" in t for t in at)
assert at[2]["input_schema"]["required"] == ["path", "old_str", "new_str"]
# already-anthropic passthrough
again = to_anthropic_tools(at)
assert again == at

# result content: plain string
rc1 = _tool_result_content("hello")
assert rc1 == "hello"
# text + image dict
rc2 = _tool_result_content({"text": "shot attached", "image": {"data": "QUJD", "media_type": "image/png"}})
assert isinstance(rc2, list) and rc2[0]["type"] == "text" and rc2[1]["type"] == "image", rc2
assert rc2[1]["source"]["data"] == "QUJD"
# empty dict -> placeholder
rc3 = _tool_result_content({})
assert rc3[0]["text"] == "(empty result)"
print("anthropic adapter conversion OK")
print("PHASE 3 (J8-J9) SMOKE: ALL PASS")

# Derek p5: restore the real IO functions for the rest of the test session.
sandbox_tools.read_file = _orig_read
sandbox_tools.write_file = _orig_write
sandbox_tools.file_exists = _orig_exists
