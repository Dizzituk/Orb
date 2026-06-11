"""Live self-test for the Anthropic tool-loop adapter (Job 8) + fable-5
access check (Job 12). One trivial agentic task: read a planted buggy file,
fix one line surgically, run a shell check, report DONE.

Budget: max 8 iterations, 2000 output tokens/call, small thinking budget.
"""
import sys, os, asyncio, json
os.chdir("D:/Orb")  # relative sqlite path in app.db resolves against CWD
sys.path.insert(0, "D:/Orb")

# Load .env for ANTHROPIC_API_KEY
def load_env(path="D:/Orb/.env"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print("env load failed:", e)

load_env()

# Keys live in the encrypted settings store. Decryption needs ORB_MASTER_KEY,
# which deliberately exists only in the running backend's memory. To run this
# test standalone, launch it with the master key in the shell:
#   $env:ORB_MASTER_KEY="<key>" ; & D:\Orb\.venv\Scripts\python.exe D:\Orb\tests\test_j8_live_anthropic_loop.py
if not os.environ.get("ANTHROPIC_API_KEY"):
    if os.environ.get("ORB_MASTER_KEY"):
        from app.crypto.encryption import init_master_key_from_env
        init_master_key_from_env()
        from app.db import SessionLocal
        from app.settings.service import sync_all_to_env
        _db = SessionLocal()
        try:
            n = sync_all_to_env(_db)
            print(f"synced {n} keys from encrypted store")
        finally:
            _db.close()
    else:
        print("\nNO KEY AVAILABLE: set ORB_MASTER_KEY in this shell first (see comment above).")
        sys.exit(2)
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing from .env AND settings store"

PLAY = r"D:\Orb\jobs\verifier_selftest"
os.makedirs(PLAY, exist_ok=True)
with open(os.path.join(PLAY, "bug.py"), "w", encoding="utf-8") as f:
    f.write("def add(a, b):\n    return a - b  # BUG: should add\n")
with open(os.path.join(PLAY, "bug_check.py"), "w", encoding="utf-8") as f:
    f.write("from bug import add\nassert add(2, 3) == 5, 'still broken'\nprint('CHECK_PASS')\n")

from app.pipeline_v2 import sandbox_tools

async def executor(name, args):
    """Host-scoped executor for the playground only."""
    import subprocess
    if name == "read_file":
        p = os.path.join(PLAY, os.path.basename(args.get("path", "")))
        if not os.path.isfile(p):
            return "ERROR: not found: " + p
        return open(p, "r", encoding="utf-8").read()
    if name == "edit_file":
        p = os.path.join(PLAY, os.path.basename(args.get("path", "")))
        # reuse the real sandbox_tools logic via monkeypatched IO
        orig_read, orig_write, orig_exists = sandbox_tools.read_file, sandbox_tools.write_file, sandbox_tools.file_exists
        async def r(path, profile=None):
            return open(p, "r", encoding="utf-8").read() if os.path.isfile(p) else None
        async def w(path, content, profile=None):
            tgt = p if path == p or os.path.basename(path) == os.path.basename(p) else os.path.join(PLAY, os.path.basename(path))
            open(tgt, "w", encoding="utf-8", newline="\n").write(content)
            return True
        async def e(path, profile=None):
            return os.path.isfile(os.path.join(PLAY, os.path.basename(path)))
        sandbox_tools.read_file, sandbox_tools.write_file, sandbox_tools.file_exists = r, w, e
        try:
            ok, msg = await sandbox_tools.edit_file(p, args.get("old_str",""), args.get("new_str",""))
        finally:
            sandbox_tools.read_file, sandbox_tools.write_file, sandbox_tools.file_exists = orig_read, orig_write, orig_exists
        return ("OK: " + msg) if ok else ("EDIT REFUSED: " + msg)
    if name == "run_shell":
        try:
            r = subprocess.run(
                ["powershell", "-Command", f"cd '{PLAY}'; & 'D:\\Orb\\.venv\\Scripts\\python.exe' bug_check.py"],
                capture_output=True, text=True, timeout=30,
            )
            return f"exit_code={r.returncode}\nSTDOUT:\n{r.stdout[:500]}\nSTDERR:\n{r.stderr[:300]}"
        except Exception as exc:
            return f"ERROR: shell failed: {exc}"
    return "ERROR: unknown tool " + name

SYS = ("You are a precise code-fixing agent. Fix the bug using edit_file with an "
       "EXACT unique old_str, verify with run_shell (it runs bug_check.py), then "
       "reply with exactly DONE and a one-line summary. No other commentary.")
USER = ("bug.py contains a one-line bug (add() subtracts). Read bug.py, fix that "
        "one line surgically with edit_file, run the shell check, and finish.")

from app.pipeline_v2.llm_tools_anthropic import run_anthropic_tool_loop
from app.pipeline_v2.llm_tool_defs import TOOLS

calls = []
def on_tool(name, args):
    calls.append(name)
    print(f"  tool: {name} {json.dumps({k: str(v)[:60] for k, v in args.items()})}")

async def try_model(model):
    print(f"\n=== {model} ===")
    msgs, itok, otok = await run_anthropic_tool_loop(
        system_prompt=SYS, initial_user_message=USER, model=model,
        max_iterations=8, max_tokens=2000,
        on_tool_call=on_tool, on_text=lambda t: print("  text:", t[:200]),
        tools=TOOLS, tool_executor=executor,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    fixed = open(os.path.join(PLAY, "bug.py"), encoding="utf-8").read()
    ok = "a + b" in fixed.replace("a+b", "a + b")
    print(f"  tokens: {itok} in / {otok} out, tool calls: {calls}")
    print(f"  file fixed: {ok}")
    return ok and len(msgs) > 2

async def main():
    ok = await try_model("claude-fable-5")
    if ok:
        print("\nRESULT: J8 LIVE PASS on claude-fable-5 (access confirmed)")
        return
    print("\nfable-5 attempt incomplete - trying fallback claude-opus-4-8")
    calls.clear()
    with open(os.path.join(PLAY, "bug.py"), "w", encoding="utf-8") as f:
        f.write("def add(a, b):\n    return a - b  # BUG: should add\n")
    ok2 = await try_model("claude-opus-4-8")
    print("\nRESULT:", "J8 LIVE PASS on fallback opus-4-8 (fable-5 NOT accessible)" if ok2 else "J8 LIVE FAIL on both models")

asyncio.run(main())
