# FILE: tests/test_debug_console.py
# Purpose: Unit tests for the visible "ASTRA console" plumbing (host + sandbox).
#          Host: proves the wrapper is TRANSPARENT (returns the underlying result)
#          and echoes command + output to a transcript. Sandbox: proves the
#          LIVE-STREAMING wrapper (docs/debug-visibility Part A) is transparent,
#          embeds the original command, streams each output line into the visible
#          transcript, and preserves the exit code -- without popping real windows
#          or needing a live sandbox.
import asyncio
import base64
import re


# ------------------------------ host console -------------------------------

def test_host_format_block_has_command_and_output():
    from app.debug import host_console
    block = host_console._format_block("rg foo", "Exit code: 0\nSTDOUT:\nbar")
    assert "HOST PS> rg foo" in block
    assert "Exit code: 0" in block
    assert "bar" in block


def test_host_run_command_visible_is_transparent_and_logs(tmp_path, monkeypatch):
    from app.debug import host_console

    log = tmp_path / "host_console.log"
    monkeypatch.setattr(host_console, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(host_console, "_LOG_PATH", str(log))
    # never pop a real console window in tests
    monkeypatch.setattr(host_console, "_ensure_window_sync", lambda: None)

    async def fake_exec(params):
        return "Exit code: 0\nSTDOUT:\nhello-world"

    monkeypatch.setattr(
        "app.debug.executors.filesystem.execute_run_command", fake_exec
    )

    out = asyncio.run(
        host_console.run_command_visible({"command": "echo hello-world"})
    )
    # transparent: returns exactly what the underlying executor returned
    assert out == "Exit code: 0\nSTDOUT:\nhello-world"
    # and the command + output were echoed to the transcript
    txt = log.read_text(encoding="utf-8")
    assert "HOST PS> echo hello-world" in txt
    assert "hello-world" in txt


def test_host_console_failure_never_breaks_command(monkeypatch):
    from app.debug import host_console

    monkeypatch.setattr(host_console, "_ensure_window_sync", lambda: None)

    async def fake_exec(params):
        return "OK"

    monkeypatch.setattr(
        "app.debug.executors.filesystem.execute_run_command", fake_exec
    )

    def boom(*a, **k):
        raise RuntimeError("disk gone")

    # appending blows up; the command's result must STILL come through unchanged
    monkeypatch.setattr(host_console, "_append_block_sync", boom)
    out = asyncio.run(host_console.run_command_visible({"command": "x"}))
    assert out == "OK"


# ----------------------------- sandbox console -----------------------------

class _FakeResult:
    def __init__(self):
        self.ok = True
        self.exit_code = 0
        self.duration_ms = 5
        self.stdout = "did the thing"
        self.stderr = ""


class _FakeClient:
    def __init__(self):
        self.calls = []

    def shell_run(self, command, cwd_target="REPO", timeout_seconds=60):
        self.calls.append((command, cwd_target, timeout_seconds))
        return _FakeResult()


def test_sandbox_visible_shell_run_streams_and_is_transparent(monkeypatch):
    from app.debug import sandbox_console

    monkeypatch.setattr(sandbox_console, "_window_up", False)
    fc = _FakeClient()
    res = sandbox_console.visible_shell_run(
        "npm install @univerjs/presets", client=fc
    )

    # transparent: returns the ShellResult unchanged (exit code preserved upstream)
    assert res.stdout == "did the thing"

    # the command actually executed in the sandbox is the streaming wrapper
    exec_cmds = [
        c[0] for c in fc.calls
        if "FromBase64String" in c[0] and "ForEach-Object" in c[0]
    ]
    assert exec_cmds, "expected a streaming wrapper to run in the sandbox"
    wrapper = exec_cmds[-1]

    # it streams each line into the visible transcript AND passes it through
    assert "Add-Content" in wrapper
    assert "ForEach-Object" in wrapper
    # the command line is written to the window BEFORE it runs (appears immediately)
    assert "SANDBOX PS> " in wrapper
    assert wrapper.index("SANDBOX PS> ") < wrapper.index("Invoke-Expression")
    # exit code is preserved so ShellResult.exit_code stays correct
    assert "$LASTEXITCODE" in wrapper
    assert "exit $code" in wrapper

    # the embedded base64 decodes back to the ORIGINAL command (quoting-proof)
    m = re.search(r"FromBase64String\('([^']+)'\)", wrapper)
    decoded = base64.b64decode(m.group(1)).decode("utf-8")
    assert decoded == "npm install @univerjs/presets"


def test_sandbox_window_launched_once(monkeypatch):
    from app.debug import sandbox_console

    monkeypatch.setattr(sandbox_console, "_window_up", False)
    fc = _FakeClient()
    sandbox_console.visible_shell_run("echo a", client=fc)
    first_count = len(fc.calls)          # 2 window-setup + 1 wrapper = 3
    assert first_count == 3
    sandbox_console.visible_shell_run("echo b", client=fc)
    # 2nd call must NOT re-run the window setup: only the wrapper (+1)
    assert len(fc.calls) == first_count + 1
