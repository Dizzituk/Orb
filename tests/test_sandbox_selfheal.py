# FILE: tests/test_sandbox_selfheal.py
# Purpose: Unit tests for Job 03 (sandbox boot self-heal). Covers the config-driven
#          failure classifier mapping (the required test), the boot health assessment,
#          the detect->classify->remediate->reboot->re-verify loop (fully mocked -- no
#          live sandbox, no real boots, no installs), and the clone_freshness deps check.
# Last-renovated: 2026-06-17 (created -- Job 03 sandbox-selfheal)
import asyncio

from app.debug import boot_failure_classifier as clf
from app.debug import sandbox_selfheal as sh


# ───────────────────────── canned boot reports ─────────────────────────

_CLEAN = (
    "=== SANDBOX BOOT INSPECTION (backend + frontend) ===\n"
    "--- BACKEND ---\n"
    "CLONE STATUS: controller_ok=True backend_ok=True\n"
    "--- scanned last 50 lines of logs/astra.log (via console): 0 error line(s) ---\n"
    "No ERROR/CRITICAL/Traceback in the recent boot log -- boot looks clean.\n"
    "--- FRONTEND ---\n"
    "VISUAL READOUT of the sandbox Electron app:\nThe app booted fine; dashboards are visible."
)

_UNIVERJS = (
    "=== SANDBOX BOOT INSPECTION (backend + frontend) ===\n"
    "CLONE STATUS: controller_ok=True backend_ok=True\n"
    "No ERROR/CRITICAL/Traceback in the recent boot log -- boot looks clean.\n"
    "--- FRONTEND ---\n"
    "VISUAL READOUT: Vite error overlay -- "
    'Failed to resolve import "@univerjs/presets" from "src/main.tsx".'
)

_BACKEND_MISSING = (
    "CLONE STATUS: controller_ok=True backend_ok=False\n"
    "BOOT ERRORS:\n"
    "2026-06-17 ERROR ModuleNotFoundError: No module named 'httpx_sse'\n"
)

_PORT_IN_USE = (
    "CLONE STATUS: controller_ok=True backend_ok=False\n"
    "BOOT ERRORS:\n"
    "2026-06-17 ERROR [WinError 10048] only one usage of each socket address is normally permitted\n"
)

_UNKNOWN = (
    "CLONE STATUS: controller_ok=True backend_ok=False\n"
    "BOOT ERRORS:\n"
    "2026-06-17 ERROR something entirely novel went wrong in the reactor core\n"
)

_CONTROLLER_DOWN = "Sandbox controller is UNREACHABLE -- start the Windows Sandbox first."


# ───────────────────── classifier mapping (the required test) ─────────────────────

def test_classify_frontend_missing_module():
    c = clf.classify(_UNIVERJS)
    assert c.kind == "frontend_missing_module"
    assert c.auto_fixable is True
    assert c.remediation.action == clf.ACTION_NPM_INSTALL
    assert c.remediation.target == "frontend"
    assert c.remediation.package == "@univerjs/presets"


def test_classify_backend_missing_module():
    c = clf.classify(_BACKEND_MISSING)
    assert c.kind == "backend_missing_module"
    assert c.auto_fixable is True
    assert c.remediation.action == clf.ACTION_PIP_INSTALL
    assert c.remediation.target == "backend"
    assert c.remediation.package == "httpx_sse"


def test_classify_port_in_use():
    c = clf.classify(_PORT_IN_USE)
    assert c.kind == "port_in_use"
    assert c.auto_fixable is True
    assert c.remediation.action == clf.ACTION_KILL_STALE


def test_classify_unknown_is_not_auto_fixable():
    c = clf.classify(_UNKNOWN)
    assert c.kind == "unknown"
    assert c.auto_fixable is False
    assert c.remediation.action == clf.ACTION_NONE


def test_backend_module_error_not_misread_as_frontend():
    """A python ModuleNotFoundError must hit the pip rule, not the broad node rule."""
    c = clf.classify("ModuleNotFoundError: No module named 'pydantic'")
    assert c.kind == "backend_missing_module"
    assert c.remediation.action == clf.ACTION_PIP_INSTALL


def test_classify_cannot_find_module_node():
    c = clf.classify('Error: Cannot find module "@reduxjs/toolkit"')
    assert c.kind == "frontend_missing_module"
    assert c.remediation.action == clf.ACTION_NPM_INSTALL
    assert c.remediation.package == "@reduxjs/toolkit"


# ───────────────────────── boot assessment ─────────────────────────

def test_assess_clean_is_healthy():
    a = clf.assess_boot(_CLEAN)
    assert a.healthy is True
    assert a.backend_ok is True
    assert a.controller_down is False


def test_assess_controller_down():
    a = clf.assess_boot(_CONTROLLER_DOWN)
    assert a.controller_down is True
    assert a.healthy is False


def test_assess_univerjs_unhealthy_autofixable():
    a = clf.assess_boot(_UNIVERJS)
    assert a.healthy is False
    assert a.classification.kind == "frontend_missing_module"
    assert a.classification.auto_fixable is True


def test_assess_unknown_unhealthy_unfixable():
    a = clf.assess_boot(_UNKNOWN)
    assert a.healthy is False
    assert a.classification.kind == "unknown"


def test_assess_negative_screen_blocks_false_success():
    """Backend clean but the screen shows an unclassified error overlay -> NOT healthy."""
    report = (
        "CLONE STATUS: controller_ok=True backend_ok=True\n"
        "No ERROR/CRITICAL/Traceback in the recent boot log -- boot looks clean.\n"
        "VISUAL READOUT: there is a red error screen with a stack trace I cannot parse."
    )
    a = clf.assess_boot(report)
    assert a.healthy is False
    assert a.classification.kind == "unknown"  # no rule matched, but screen is bad


def test_assess_negated_error_terms_stay_healthy():
    """A clean vision answer that echoes the prompt's error terms negatively must NOT
    read as broken (the vision prompt literally lists 'error overlay', 'crash dialog')."""
    report = (
        "CLONE STATUS: controller_ok=True backend_ok=True\n"
        "No ERROR/CRITICAL/Traceback in the recent boot log -- boot looks clean.\n"
        "VISUAL READOUT: The app booted fine. There is no error overlay, no crash dialog, "
        "and no stack trace -- dashboards render normally."
    )
    a = clf.assess_boot(report)
    assert a.healthy is True


# ───────────────────────── the self-heal loop (mocked) ─────────────────────────

class _FakeHealth:
    def __init__(self, repo_root="D:\\Orb"):
        self.repo_root = repo_root


class _FakeClient:
    def __init__(self, repo_root="D:\\Orb", health_exc=None):
        self._rr = repo_root
        self._exc = health_exc

    def health(self):
        if self._exc:
            raise self._exc
        return _FakeHealth(self._rr)


class _FakeMgr:
    def __init__(self, backend_up=False, **kw):
        self.client = _FakeClient(**kw)
        self._backend_up = backend_up

    def check_backend(self):
        return (self._backend_up, "up" if self._backend_up else "down")


def _wire_loop(monkeypatch, reports, *, mgr=None, remediation_result=(True, "fix ran")):
    """Patch the manager + inspect + remediation. `reports` is consumed one per inspect;
    the last entry repeats if the loop inspects more times than provided."""
    mgr = mgr or _FakeMgr()
    starts, stops, fixes = [], [], []

    monkeypatch.setattr("app.sandbox.manager.get_sandbox_manager", lambda: mgr)
    monkeypatch.setattr("app.sandbox.manager.start_zombie_orb",
                        lambda full=True: (starts.append(full) or (True, "started")))
    # The loop reboots via the controller-safe _safe_stop_clone (NOT manager.stop_clone).
    monkeypatch.setattr(sh, "_safe_stop_clone",
                        lambda mgr_: (stops.append(True) or (True, "stopped (controller spared)")))

    seq = list(reports)
    inspects = []

    def fake_inspect(params):
        inspects.append(dict(params or {}))
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr("app.debug.sandbox_boot_tool.inspect_sandbox_boot", fake_inspect)

    def fake_remediate(rem, repo_root, frontend_dir, mgr_):
        fixes.append((rem.action, rem.target, rem.package, frontend_dir))
        return remediation_result

    monkeypatch.setattr(sh, "_apply_remediation", fake_remediate)
    return {"starts": starts, "stops": stops, "fixes": fixes, "inspects": inspects}


def test_loop_clean_first_boot(monkeypatch):
    rec = _wire_loop(monkeypatch, [_CLEAN])
    out = sh.selfheal_sandbox_boot({"wait_seconds": 0})
    assert "SELF-HEAL: CLEAN" in out
    assert len(rec["starts"]) == 1   # booted once
    assert rec["fixes"] == []        # nothing to fix


def test_loop_univerjs_then_clean(monkeypatch):
    rec = _wire_loop(monkeypatch, [_UNIVERJS, _CLEAN])
    out = sh.selfheal_sandbox_boot({"wait_seconds": 0})
    assert "SELF-HEAL: CLEAN" in out
    # exactly one npm fix in the frontend dir, then a reboot
    assert len(rec["fixes"]) == 1
    action, target, package, frontend_dir = rec["fixes"][0]
    assert action == clf.ACTION_NPM_INSTALL and target == "frontend"
    assert package == "@univerjs/presets"
    assert frontend_dir.endswith("orb-desktop")   # derived from REPO_ROOT, not hardcoded
    assert len(rec["stops"]) == 1 and len(rec["starts"]) == 2  # initial boot + reboot


def test_loop_cycle_cap(monkeypatch):
    rec = _wire_loop(monkeypatch, [_UNIVERJS])  # never goes clean
    out = sh.selfheal_sandbox_boot({"wait_seconds": 0, "max_cycles_per_class": 1})
    assert "SELF-HEAL: CYCLE_CAP" in out
    assert len(rec["fixes"]) == 1   # one fix attempt, then cap hit


def test_loop_unknown_failure_reports_verbatim(monkeypatch):
    rec = _wire_loop(monkeypatch, [_UNKNOWN])
    out = sh.selfheal_sandbox_boot({"wait_seconds": 0})
    assert "SELF-HEAL: UNFIXABLE" in out
    assert "reactor core" in out    # the verbatim error is surfaced
    assert rec["fixes"] == []       # never tried to fix an unknown


def test_loop_controller_unreachable_aborts(monkeypatch):
    mgr = _FakeMgr(health_exc=RuntimeError("connection refused"))
    rec = _wire_loop(monkeypatch, [_CLEAN], mgr=mgr)
    out = sh.selfheal_sandbox_boot({"wait_seconds": 0})
    assert "SELF-HEAL: ABORTED" in out
    assert "connection refused" in out
    assert rec["starts"] == []      # never even tried to boot


def test_boot_waits_at_least_60s_even_if_zero_requested(monkeypatch):
    """The slow boot must be given time: a tiny wait_seconds is floored to 60."""
    rec = _wire_loop(monkeypatch, [_CLEAN])
    sh.selfheal_sandbox_boot({"wait_seconds": 0})
    assert rec["inspects"], "inspect was never called"
    assert rec["inspects"][0]["wait_seconds"] >= 60


def test_boot_default_wait_is_90(monkeypatch):
    rec = _wire_loop(monkeypatch, [_CLEAN])
    sh.selfheal_sandbox_boot({})       # no wait_seconds given
    assert rec["inspects"][0]["wait_seconds"] == 90


def test_auto_fix_false_reports_without_fixing(monkeypatch):
    """'boot and tell me how it went': boots, waits, inspects, REPORTS the known fix, fixes NOTHING."""
    rec = _wire_loop(monkeypatch, [_UNIVERJS])  # a fixable failure is present
    out = sh.selfheal_sandbox_boot({"wait_seconds": 0, "auto_fix": False})
    assert "SELF-HEAL: REPORT" in out
    assert "@univerjs/presets" in out          # surfaced the concrete error
    assert "frontend_missing_module" in out     # and the classification / would-be fix
    assert rec["fixes"] == []                    # but changed nothing
    assert rec["stops"] == []                    # and did NOT reboot


def test_auto_fix_false_clean_boot_reports_clean(monkeypatch):
    _wire_loop(monkeypatch, [_CLEAN])
    out = sh.selfheal_sandbox_boot({"auto_fix": False})
    assert "SELF-HEAL: CLEAN" in out


def test_already_running_fix_mode_stops_then_reboots(monkeypatch):
    """Can't boot over a running clone -- fix mode stops it first, then boots fresh."""
    mgr = _FakeMgr(backend_up=True)
    rec = _wire_loop(monkeypatch, [_CLEAN], mgr=mgr)
    sh.selfheal_sandbox_boot({"wait_seconds": 0})   # auto_fix defaults True
    assert len(rec["stops"]) >= 1                    # stopped the running instance
    assert len(rec["starts"]) >= 1                   # then booted fresh


def test_safe_stop_clone_targets_clone_not_controller(monkeypatch):
    """The reboot stop must kill ONLY the clone (ports 8000/5173 + electron/node) and NEVER
    blanket-kill python -- the manager's broad filter killed the controller (WinError 10054)."""
    class _Res:
        stdout = "SAFE_STOP_DONE"
        stderr = ""
        exit_code = 0
        ok = True

    sent = {}

    def fake_visible(command, cwd_target="REPO", timeout_seconds=60, client=None):
        sent["command"] = command
        return _Res()

    monkeypatch.setattr("app.debug.sandbox_console.visible_shell_run", fake_visible)
    ok, msg = sh._safe_stop_clone(_FakeMgr())
    assert ok is True
    cmd = sent["command"]
    assert "8000" in cmd and "5173" in cmd          # stops the clone by port
    assert "electron" in cmd                          # and the Electron app
    assert "Get-Process python" not in cmd            # NEVER blanket-kills python
    assert "sandbox_controller" not in cmd            # not the manager's misfiring filter


def test_loop_reboot_uses_safe_stop_not_manager(monkeypatch):
    """A remediation reboot must go through _safe_stop_clone, never manager.stop_zombie_orb
    (which kills the controller)."""
    import app.sandbox.manager as mgr_mod
    called = {"manager_stop": 0}
    monkeypatch.setattr(mgr_mod, "stop_zombie_orb",
                        lambda: (called.__setitem__("manager_stop", called["manager_stop"] + 1) or (True, "x")))
    rec = _wire_loop(monkeypatch, [_UNIVERJS, _CLEAN])   # fix -> reboot -> clean
    sh.selfheal_sandbox_boot({"wait_seconds": 0})
    assert called["manager_stop"] == 0                 # manager.stop_clone NEVER called
    assert len(rec["stops"]) >= 1                      # the safe stop was used instead


def test_already_running_report_mode_inspects_as_is(monkeypatch):
    """Report-only on an already-running clone inspects it as-is -- no disruptive reboot."""
    mgr = _FakeMgr(backend_up=True)
    rec = _wire_loop(monkeypatch, [_CLEAN], mgr=mgr)
    out = sh.selfheal_sandbox_boot({"wait_seconds": 0, "auto_fix": False})
    assert "SELF-HEAL: CLEAN" in out
    assert rec["starts"] == [] and rec["stops"] == []   # never rebooted


# ───────────────────── remediation command construction ─────────────────────

def test_pip_command_renders_correct_path_no_cr():
    """`\\\\requirements.txt` was a worry -> confirm the path renders correctly with NO
    stray carriage return and prefers requirements.txt."""
    cmd = sh._pip_command("D:\\Orb", "numpy")
    assert "D:\\Orb\\requirements.txt" in cmd
    assert "\r" not in cmd
    assert "pip install -r" in cmd
    assert "pip install 'numpy'" in cmd  # fallback present


def test_pip_command_escapes_single_quotes():
    cmd = sh._pip_command("D:\\O'rb", "numpy")
    assert "D:\\O''rb" in cmd          # quote doubled for PS -LiteralPath safety
    assert "-LiteralPath 'D:\\O''rb'" in cmd


def test_sq_doubles_quotes():
    assert sh._sq("a'b") == "a''b"
    assert sh._sq("plain") == "plain"


# ───────────────────────── clone_freshness deps check ─────────────────────────

def _run(coro):
    return asyncio.run(coro)


def test_clone_deps_parses_stale(monkeypatch):
    from app.pipeline_v2 import clone_freshness as cf

    async def fake_run_shell(cmd, timeout_sec=30, profile=None):
        return {"stdout": "DEPS_STALE node_modules missing (npm install needed)\n", "stderr": "", "returncode": 0}

    monkeypatch.setattr("app.pipeline_v2.sandbox_tools.run_shell", fake_run_shell)
    stale, reason, err = _run(cf._clone_deps("D:/orb-desktop"))
    assert stale == 1 and err == ""
    assert "node_modules missing" in reason


def test_clone_deps_parses_fresh(monkeypatch):
    from app.pipeline_v2 import clone_freshness as cf

    async def fake_run_shell(cmd, timeout_sec=30, profile=None):
        return {"stdout": "DEPS_OK node\n", "stderr": "", "returncode": 0}

    monkeypatch.setattr("app.pipeline_v2.sandbox_tools.run_shell", fake_run_shell)
    stale, reason, err = _run(cf._clone_deps("D:/orb-desktop"))
    assert stale == 0 and err == ""


def test_clone_deps_handles_no_marker(monkeypatch):
    from app.pipeline_v2 import clone_freshness as cf

    async def fake_run_shell(cmd, timeout_sec=30, profile=None):
        return {"stdout": "", "stderr": "VM down", "returncode": -1}

    monkeypatch.setattr("app.pipeline_v2.sandbox_tools.run_shell", fake_run_shell)
    stale, reason, err = _run(cf._clone_deps("D:/orb-desktop"))
    assert stale == -1 and "VM down" in err


def _wire_freshness(monkeypatch, deps):
    """Fresh git everywhere; deps result is `deps` (stale_flag, reason, err)."""
    from app.pipeline_v2 import clone_freshness as cf
    H = "a" * 40

    def fake_host(root):
        return (H, 0, "")

    async def fake_clone(root):
        return (H, 0, "")

    async def fake_deps(root):
        return deps

    monkeypatch.setattr(cf, "_host_git", fake_host)
    monkeypatch.setattr(cf, "_clone_git", fake_clone)
    monkeypatch.setattr(cf, "_clone_deps", fake_deps)
    return cf


def test_freshness_deps_advisory_does_not_block(monkeypatch):
    monkeypatch.delenv("ASTRA_CLONE_DEPS_GATE", raising=False)
    cf = _wire_freshness(monkeypatch, (1, "node_modules missing (npm install needed)", ""))
    r = _run(cf.check_clone_freshness())
    assert r.ok is True                       # advisory: git is fresh, so gate passes
    assert any(repo.deps_stale == 1 for repo in r.repos)
    assert any("deps stale" in l for l in r.render_lines())


def test_freshness_deps_gate_blocks_when_enabled(monkeypatch):
    monkeypatch.setenv("ASTRA_CLONE_DEPS_GATE", "1")
    cf = _wire_freshness(monkeypatch, (1, "node_modules missing (npm install needed)", ""))
    r = _run(cf.check_clone_freshness())
    assert r.ok is False                      # gating on: staleness blocks the self-build
    assert "deps stale" in r.reason


def test_freshness_deps_check_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ASTRA_CLONE_DEPS_CHECK", "0")
    from app.pipeline_v2 import clone_freshness as cf
    H = "a" * 40
    monkeypatch.setattr(cf, "_host_git", lambda root: (H, 0, ""))

    async def fake_clone(root):
        return (H, 0, "")

    called = {"deps": False}

    async def fake_deps(root):
        called["deps"] = True
        return (1, "should not run", "")

    monkeypatch.setattr(cf, "_clone_git", fake_clone)
    monkeypatch.setattr(cf, "_clone_deps", fake_deps)
    r = _run(cf.check_clone_freshness())
    assert r.ok is True
    assert called["deps"] is False            # probe skipped when disabled
