# FILE: tests/test_sandbox_visual_selfheal.py
# Purpose: Unit tests for WI-3 (sandbox screenshot + vision/OCR) and WI-4
#          (inspect_sandbox_boot = backend readout + visual). All mocked -- no live
#          sandbox, no vision API, no real D:\ writes.
import base64


# ----------------------------- WI-3: screenshot ----------------------------

def test_capture_snippet_substitutes_and_grabs_screen():
    from app.debug import sandbox_screenshot as ss
    snip = ss._build_capture_snippet(r"D:\Orb\_astra_selfheal\sandbox_screen.b64")
    # placeholders are all replaced
    assert "__MAXEDGE__" not in snip and "__QUALITY__" not in snip and "__B64PATH__" not in snip
    # it actually grabs the screen + encodes JPEG to the requested path
    assert "CopyFromScreen" in snip
    assert "ImageFormat]::Jpeg" in snip
    assert r"D:\Orb\_astra_selfheal\sandbox_screen.b64" in snip
    assert "SCREENSHOT_OK" in snip and "SCREENSHOT_ERR" in snip


def test_capture_snippet_is_amsi_safe():
    """AMSI/Defender blocks the capture script if it carries an inline DllImport
    SetForegroundWindow P/Invoke (the window-injection signature). Foreground must be
    brought via COM AppActivate instead -- regression guard for the live AV block."""
    from app.debug import sandbox_screenshot as ss
    snip = ss._build_capture_snippet(r"D:\Orb\_astra_selfheal\sandbox_screen.b64")
    assert "DllImport" not in snip
    assert "user32" not in snip
    assert "SetForegroundWindow" not in snip
    assert "AppActivate" in snip          # benign COM foreground instead
    assert "CopyFromScreen" in snip       # still actually captures the screen


def test_sandbox_abs_b64_path():
    from app.debug import sandbox_screenshot as ss
    assert ss._sandbox_abs_b64_path(r"D:\Orb") == r"D:\Orb\_astra_selfheal\sandbox_screen.b64"
    # trailing slash tolerated
    assert ss._sandbox_abs_b64_path("D:\\Orb\\") == r"D:\Orb\_astra_selfheal\sandbox_screen.b64"


class _Health:
    def __init__(self, repo_root=r"D:\Orb"):
        self.repo_root = repo_root


class _FileContent:
    def __init__(self, content):
        self.content = content


class _ShellResult:
    def __init__(self, stdout="", stderr="", exit_code=0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.ok = exit_code == 0


class _FakeClient:
    """Stands in for SandboxClient: health() + repo_file() only."""
    def __init__(self, repo_file_content=None, repo_file_exc=None, health_exc=None):
        self._content = repo_file_content
        self._repo_exc = repo_file_exc
        self._health_exc = health_exc
        self.repo_file_calls = []

    def health(self):
        if self._health_exc:
            raise self._health_exc
        return _Health()

    def repo_file(self, rel):
        self.repo_file_calls.append(rel)
        if self._repo_exc:
            raise self._repo_exc
        return _FileContent(self._content)


def _patch_host_paths(monkeypatch, tmp_path):
    from app.debug import sandbox_screenshot as ss
    monkeypatch.setattr(ss, "_HOST_IMG_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "_HOST_IMG_PATH", str(tmp_path / "sandbox_screen.jpg"))


def test_capture_and_read_happy_path(monkeypatch, tmp_path):
    from app.debug import sandbox_screenshot as ss

    _patch_host_paths(monkeypatch, tmp_path)
    fake_jpeg = b"\xff\xd8\xff\xe0FAKEJPEGBYTES\xff\xd9"
    b64 = base64.b64encode(fake_jpeg).decode("ascii") + "\r\n"  # trailing CRLF tolerated
    client = _FakeClient(repo_file_content=b64)

    sent = {}

    def fake_visible(command, cwd_target="REPO", timeout_seconds=60, client=None):
        sent["command"] = command
        return _ShellResult(stdout="SCREENSHOT_OK full=1920x1080 scaled=1280x720 jpeg_bytes=99 b64_chars=132")

    monkeypatch.setattr("app.debug.sandbox_console.visible_shell_run", fake_visible)
    monkeypatch.setattr(
        "app.llm.gemini_vision.ask_about_image",
        lambda image_source, user_question: {
            "answer": 'Vite error overlay: Failed to resolve import "@univerjs/presets"',
            "provider": "google",
            "model": "gemini-2.5-pro",
        },
    )

    out = ss.capture_and_read_sandbox_screen(client=client)

    # the capture command actually ran (visibly) and targeted the b64 drop path
    assert "CopyFromScreen" in sent["command"]
    assert r"D:\Orb\_astra_selfheal\sandbox_screen.b64" in sent["command"]
    # the image bytes were decoded + written for the vision call
    assert (tmp_path / "sandbox_screen.jpg").read_bytes() == fake_jpeg
    # the vision answer (the demo error) is in the report
    assert "@univerjs/presets" in out
    assert "google/gemini-2.5-pro" in out
    assert "VISUAL READOUT" in out


def test_capture_reports_capture_error(monkeypatch, tmp_path):
    from app.debug import sandbox_screenshot as ss
    _patch_host_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.debug.sandbox_console.visible_shell_run",
        lambda *a, **k: _ShellResult(stdout="SCREENSHOT_ERR GDI+ generic error"),
    )
    out = ss.capture_and_read_sandbox_screen(client=_FakeClient())
    assert "did not succeed" in out
    assert "GDI+ generic error" in out


def test_capture_reports_controller_unreachable(monkeypatch, tmp_path):
    from app.debug import sandbox_screenshot as ss
    _patch_host_paths(monkeypatch, tmp_path)
    client = _FakeClient(health_exc=RuntimeError("connection refused"))
    out = ss.capture_and_read_sandbox_screen(client=client)
    assert "could not reach the sandbox controller" in out
    assert "connection refused" in out


def test_capture_reports_repo_file_failure(monkeypatch, tmp_path):
    from app.debug import sandbox_screenshot as ss
    _patch_host_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.debug.sandbox_console.visible_shell_run",
        lambda *a, **k: _ShellResult(stdout="SCREENSHOT_OK full=800x600 scaled=800x600 jpeg_bytes=10 b64_chars=16"),
    )
    client = _FakeClient(repo_file_exc=RuntimeError("HTTP 413 too large"))
    out = ss.capture_and_read_sandbox_screen(client=client)
    assert "could not retrieve the image bytes" in out
    assert "HTTP 413 too large" in out


# --------------------------- WI-4: inspect loop ----------------------------

def test_clone_log_abs_path():
    from app.debug import sandbox_boot_tool as sbt
    assert sbt._clone_log_abs_path(r"D:\Orb", "logs/astra.log") == r"D:\Orb\logs\astra.log"
    assert sbt._clone_log_abs_path("D:\\Orb\\", "logs/astra.log") == r"D:\Orb\logs\astra.log"


class _Mgr:
    def __init__(self, client):
        self.client = client


def test_read_clone_log_falls_back_to_visible_console(monkeypatch):
    """The controller denies 'logs/' to repo_file -> the helper must fall back to a
    visible Get-Content and return the real log text."""
    from app.debug import sandbox_boot_tool as sbt

    client = _FakeClient(repo_file_exc=RuntimeError("HTTP 403 Denied path"))
    seen = {}

    def fake_visible(command, cwd_target="REPO", timeout_seconds=60, client=None):
        seen["command"] = command
        return _ShellResult(stdout="2026 INFO ok\n2026 ERROR boom\n2026 INFO done")

    monkeypatch.setattr("app.debug.sandbox_console.visible_shell_run", fake_visible)
    text, source = sbt._read_clone_log("logs/astra.log", 200, _Mgr(client))
    assert source == "console"
    assert "ERROR boom" in text
    # it read the real clone log path via Get-Content
    assert r"D:\Orb\logs\astra.log" in seen["command"]
    assert "Get-Content" in seen["command"]


def test_inspect_combines_backend_and_visual(monkeypatch):
    from app.debug import sandbox_boot_tool as sbt

    monkeypatch.setattr(sbt, "read_sandbox_boot", lambda params: "CLONE STATUS: backend_ok=True\nBACKEND_FAKE")
    monkeypatch.setattr(
        "app.debug.frontend_boot_check.probe_frontend_boot",
        lambda *a, **k: "FRONTEND TEXT PROBE FAKE",
    )
    monkeypatch.setattr(
        "app.debug.sandbox_screenshot.capture_and_read_sandbox_screen",
        lambda *a, **k: "VISUAL READOUT: @univerjs/presets overlay",
    )
    # delay=0 so the test does not actually sleep
    out = sbt.inspect_sandbox_boot({"screenshot_delay_seconds": 0})
    assert "BACKEND_FAKE" in out
    assert "@univerjs/presets" in out
    assert "FRONTEND TEXT PROBE FAKE" in out      # the text probe is included
    assert "BACKEND" in out and "FRONTEND" in out
    assert ":8000" in out


def test_inspect_never_raises_on_visual_failure(monkeypatch):
    from app.debug import sandbox_boot_tool as sbt

    monkeypatch.setattr(sbt, "read_sandbox_boot", lambda params: "backend ok")
    monkeypatch.setattr("app.debug.frontend_boot_check.probe_frontend_boot", lambda *a, **k: "FE OK")

    def boom(*a, **k):
        raise RuntimeError("vision exploded")

    monkeypatch.setattr("app.debug.sandbox_screenshot.capture_and_read_sandbox_screen", boom)
    out = sbt.inspect_sandbox_boot({"screenshot_delay_seconds": 0})
    assert "backend ok" in out
    assert "inspection failed" in out and "vision exploded" in out
