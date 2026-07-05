# FILE: tests/test_host_shell_guard.py
# Purpose: live14 — model-issued shell commands can never manage host process/system lifecycle.
# Called-by: pytest
# Depends-on: app.pipeline_v2.sandbox_tools
# Last-renovated: 2026-07-05
"""2026-07-05 11:04: a checkout-repair worker 'cleaned up' its timed-out game
test by killing python processes — the ASTRA backend and Chatterbox both died
with it (simultaneous rc=-1). This deny-list is the whole policy surface for
LLM-issued run_shell commands; system code paths never cross it."""

from app.pipeline_v2.sandbox_tools import is_forbidden_host_command


KILLERS = [
    "Stop-Process -Name python -Force",
    "Get-Process python | Stop-Process -Force",
    "taskkill /F /IM python.exe",
    "taskkill /f /im python.exe /t",
    "tskill python",
    "stop-computer",
    "Restart-Computer -Force",
    "shutdown /r /t 0",
    "Stop-Service AstraSentinelAgent",
    "$p.Kill()",
    "wmic process where name='python.exe' delete",
    'cd "C:/Games/T"; $p = Start-Process -FilePath "D:\\Orb\\.venv\\Scripts\\python.exe" src\\main.py',
]

BENIGN = [
    'cd D:\\Orb && .venv\\Scripts\\python.exe -m py_compile "C:/Games/T/src/main.py"',
    ".venv\\Scripts\\python.exe -m pytest tests -q",
    'python -c "import src.main"',
    "Get-ChildItem src | Select-Object Name",
    "Set-Location D:\\Orb; .\\gradlew assembleDebug",
    "Get-Process python | Select-Object Id, ProcessName",  # looking is fine
    "Write-Output 'killstreak highscore'",                  # substring, not a verb
]


def test_every_killer_is_blocked():
    for cmd in KILLERS:
        reason = is_forbidden_host_command(cmd)
        assert reason, f"NOT blocked: {cmd}"
        assert "BLOCKED" in reason
        assert "eyes own" in reason  # tells the worker the right way


def test_benign_builder_commands_pass():
    for cmd in BENIGN:
        assert is_forbidden_host_command(cmd) == "", f"False positive: {cmd}"


def test_empty_and_none_pass():
    assert is_forbidden_host_command("") == ""
    assert is_forbidden_host_command(None) == ""
