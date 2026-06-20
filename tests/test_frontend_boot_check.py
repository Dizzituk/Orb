# FILE: tests/test_frontend_boot_check.py
# Purpose: Tests for the text-based frontend boot probe (Job 03d) -- the channel that
#          catches the @univerjs "Failed to resolve import" failure as TEXT (declared dep
#          not installed in the clone), independent of the AV-fragile screenshot. All
#          mocked: no live sandbox.
# Last-renovated: 2026-06-17 (created -- Job 03d frontend text probe)
from app.debug import frontend_boot_check as fb
from app.debug import boot_failure_classifier as clf


class _Health:
    def __init__(self, repo_root=r"D:\Orb"):
        self.repo_root = repo_root


class _ShellResult:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.exit_code = 0
        self.ok = True


class _FakeClient:
    def __init__(self, repo_root=r"D:\Orb", health_exc=None):
        self._rr = repo_root
        self._exc = health_exc

    def health(self):
        if self._exc:
            raise self._exc
        return _Health(self._rr)


# ----------------------------- snippet -----------------------------

def test_probe_snippet_is_amsi_safe_and_substituted():
    snip = fb._build_probe_snippet(r"D:\orb-desktop", 5173)
    assert "__FE__" not in snip and "__PORT__" not in snip
    assert r"D:\orb-desktop" in snip
    assert "ConvertFrom-Json" in snip
    assert "node_modules" in snip
    assert "localhost:5173" in snip
    # no native interop -> AMSI-safe
    assert "DllImport" not in snip and "user32" not in snip


def test_probe_snippet_escapes_single_quotes():
    snip = fb._build_probe_snippet("D:\\o'rb-desktop", 5173)
    assert "D:\\o''rb-desktop" in snip


def test_frontend_dir_from_controller():
    fd, err = fb._frontend_dir_from_controller(_FakeClient(r"D:\Orb"))
    assert err == ""
    assert fd.endswith("orb-desktop")


# ----------------------------- parsing -----------------------------

def test_format_missing_dep_emits_classifiable_line():
    out = "FE_PROBE missing @univerjs/presets,@foo/bar\nFE_PROBE dev down\n"
    report = fb._format(out, r"D:\orb-desktop")
    assert "declared dependencies NOT installed" in report
    assert "@univerjs/presets" in report
    assert 'Failed to resolve import "@univerjs/presets"' in report
    # and the boot classifier maps that text to an npm install
    c = clf.classify(report)
    assert c.kind == "frontend_missing_module"
    assert c.remediation.action == clf.ACTION_NPM_INSTALL
    assert c.remediation.package == "@univerjs/presets"


def test_format_node_modules_missing():
    report = fb._format("FE_PROBE nm_missing\nFE_PROBE dev down\n", r"D:\orb-desktop")
    assert "node_modules: MISSING" in report
    assert "Cannot find module" in report
    c = clf.classify(report)
    assert c.kind == "frontend_missing_module"  # Cannot find module -> npm install


def test_format_all_good_is_clean():
    report = fb._format("FE_PROBE deps_ok\nFE_PROBE dev 200\n", r"D:\orb-desktop")
    assert "all present in node_modules" in report
    assert "UP (HTTP 200)" in report
    # nothing for the classifier to act on -> unknown (i.e. healthy frontend)
    assert clf.classify(report).kind == "unknown"


def test_format_no_package_json():
    report = fb._format("FE_PROBE nopkg\n", r"D:\orb-desktop")
    assert "no package.json" in report


# ----------------------------- end to end (mocked) -----------------------------

def test_probe_happy_path(monkeypatch):
    sent = {}

    def fake_visible(command, cwd_target="REPO", timeout_seconds=60, client=None):
        sent["command"] = command
        return _ShellResult(stdout="FE_PROBE missing @univerjs/presets\nFE_PROBE dev down\n")

    monkeypatch.setattr("app.debug.sandbox_console.visible_shell_run", fake_visible)
    out = fb.probe_frontend_boot(client=_FakeClient(r"D:\Orb"))
    # ran the probe against the clone frontend dir
    assert "orb-desktop" in sent["command"]
    assert 'Failed to resolve import "@univerjs/presets"' in out


def test_probe_controller_unreachable_is_graceful():
    out = fb.probe_frontend_boot(client=_FakeClient(health_exc=RuntimeError("refused")))
    assert "FRONTEND (text probe)" in out
    assert "unreachable" in out
