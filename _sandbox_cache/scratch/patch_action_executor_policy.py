from pathlib import Path
path = Path("D:/Orb/app/debug/action_executor.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
'''_HOST_ONLY_PREFIXES = [
    # Read-only reference data on host
    "D:/Orb/.architecture",
    "D:\\Orb\\.architecture",
    "D:/Orb/logs",
    "D:\\Orb\\logs",
    # Android project — host only (not in sandbox)
    "D:/Astra Android Folder",
    "D:\\Astra Android Folder",
    # Desktop frontend — host only (not in sandbox)
    "D:/orb-desktop",
    "D:\\orb-desktop",
    # NOTE: D:\Orb\app, D:\Orb\main.py, D:\Orb\config, D:\Orb\docs
    # are INTENTIONALLY NOT HERE. These go through the sandbox.
]
''',
'''_HOST_ONLY_PREFIXES = [
    # Read-only reference data on host
    "D:/Orb/.architecture",
    "D:\\Orb\\.architecture",
    "D:/Orb/logs",
    "D:\\Orb\\logs",
    # Android project — host only (not in sandbox)
    "D:/Astra Android Folder",
    "D:\\Astra Android Folder",
    # NOTE: desktop and ASTRA repos are protected and must not be edited on host.
]

_PROTECTED_HOST_WRITE_PREFIXES = [
    "D:/Orb",
    "D:\\Orb",
    "D:/orb-desktop",
    "D:\\orb-desktop",
    "D:/orb-electron",
    "D:\\orb-electron",
]
'''
)
text = text.replace(
'''def _is_host_only(path: str) -> bool:
    """Check if path is host-only data (architecture maps, logs)."""
    for prefix in _HOST_ONLY_PREFIXES:
        if path.startswith(prefix):
            return True
    return False
''',
'''def _is_host_only(path: str) -> bool:
    """Check if path is host-only data (architecture maps, logs)."""
    for prefix in _HOST_ONLY_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def _is_protected_host_write(path: str) -> bool:
    """Return True when a host path must never be edited directly."""
    for prefix in _PROTECTED_HOST_WRITE_PREFIXES:
        if path.startswith(prefix):
            return True
    return False
'''
)
text = text.replace(
'''async def execute_write_file(params: Dict[str, Any]) -> str:
    """Write a file — SANDBOX for ASTRA code, host-direct only for Android/desktop."""
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return "Error: path is required."

    path = _resolve_sandbox_path(path)

    # v0.14.0: BLOCK host writes to ASTRA backend code
    _astra_code_prefixes = ["D:/Orb/app", "D:\\Orb\\app", "D:/Orb/main.py", "D:\\Orb\\main.py",
                             "D:/Orb/config", "D:\\Orb\\config", "D:/Orb/docs", "D:\\Orb\\docs"]
    for _acp in _astra_code_prefixes:
        if path.startswith(_acp):
            logger.info("[action_executor] ASTRA code write routed to SANDBOX: %s", path)
            break  # Fall through to sandbox write below
    else:
        # Host-direct write for known non-ASTRA paths (Android, desktop)
        if _is_host_only(path):
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                logger.info("[action_executor] Host write: %s (%d chars)", path, len(content))
                return f"Successfully wrote {len(content)} chars to {path}"
            except Exception as e:
                return f"Host write error: {e}"

    # Sandbox write for everything else
''',
'''async def execute_write_file(params: Dict[str, Any]) -> str:
    """Write a file — sandbox for code, host-direct only for user-approved non-project outputs."""
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return "Error: path is required."

    path = _resolve_sandbox_path(path)

    if _is_protected_host_write(path) and _is_host_only(path):
        return (
            f"Error: direct host edits are blocked for protected project path {path}. "
            "Use the sandbox mirror and promote after review."
        )

    # Sandbox write for protected project code paths
    if _is_protected_host_write(path):
        logger.info("[action_executor] Protected project write routed to SANDBOX: %s", path)
    elif _is_host_only(path):
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            logger.info("[action_executor] Host write: %s (%d chars)", path, len(content))
            return f"Successfully wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"Host write error: {e}"

    # Sandbox write for everything else
'''
)
text = text.replace(
'''async def execute_run_command(params: Dict[str, Any]) -> str:
    """Run a command — host-direct via asyncio subprocess.

    Debug lock mode needs to run commands on the host (e.g. Gradle builds
    for Android projects, Python syntax checks). The sandbox may not have
    the required SDKs or project files.
    """
''',
'''async def execute_run_command(params: Dict[str, Any]) -> str:
    """Run a command in the sandbox by default.

    Host-side command execution is reserved for explicit host diagnostics or
    deployment work outside protected project mirrors.
    """
'''
)
text = text.replace(
'''    try:
        import base64
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        proc = await asyncio.create_subprocess_exec(
            "powershell.exe", "-NoProfile", "-EncodedCommand", encoded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd if Path(cwd).exists() else None,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

        result = f"Exit code: {proc.returncode}\n"
        if stdout:
            result += f"\nSTDOUT:\n{stdout.decode('utf-8', errors='replace')[:5000]}"
        if stderr:
            result += f"\nSTDERR:\n{stderr.decode('utf-8', errors='replace')[:2000]}"
        return result
''',
'''    protected_cwd = _resolve_sandbox_path(cwd)
    if _is_protected_host_write(protected_cwd) and not _is_host_only(protected_cwd):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=float(timeout_sec) + 5.0) as client:
                resp = await client.post(
                    f"{SANDBOX_CONTROLLER_URL}/shell/run",
                    json={
                        "cmd": ["powershell.exe", "-NoProfile", "-Command", command],
                        "cwd": protected_cwd,
                        "timeout_sec": timeout_sec,
                    },
                )
            if resp.status_code == 200:
                data = resp.json()
                result = f"Exit code: {data.get('exit_code', '?')}\n"
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                if stdout:
                    result += f"\nSTDOUT:\n{stdout[:5000]}"
                if stderr:
                    result += f"\nSTDERR:\n{stderr[:2000]}"
                return result
            return f"Sandbox command failed ({resp.status_code}): {resp.text}"
        except Exception as e:
            return (
                "Sandbox command execution unavailable. Please start the sandbox before "
                f"running code commands. Details: {e}"
            )

    try:
        import base64
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        proc = await asyncio.create_subprocess_exec(
            "powershell.exe", "-NoProfile", "-EncodedCommand", encoded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd if Path(cwd).exists() else None,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

        result = f"Exit code: {proc.returncode}\n"
        if stdout:
            result += f"\nSTDOUT:\n{stdout.decode('utf-8', errors='replace')[:5000]}"
        if stderr:
            result += f"\nSTDERR:\n{stderr.decode('utf-8', errors='replace')[:2000]}"
        return result
'''
)
path.write_text(text, encoding="utf-8")
print('patched action_executor.py')
