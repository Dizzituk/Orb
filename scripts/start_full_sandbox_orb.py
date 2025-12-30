# FILE: scripts/start_full_sandbox_orb.py
"""Start full Orb (backend + frontend) in Windows Sandbox with visible windows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.overwatcher.sandbox_client import get_sandbox_client


def main():
    print("=" * 60)
    print("Starting Full Orb in Sandbox (Backend + Frontend)")
    print("=" * 60)
    
    client = get_sandbox_client()
    
    # Check connection
    health = client.health()
    print(f"Sandbox connected: {health.status}")
    
    # Write full startup script
    script = r'''# Orb Full Startup Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Orb Backend..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Start backend in separate window
$env:ORB_MASTER_KEY = 'MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTIzNDU2Nzg5MDE='
Start-Process powershell -ArgumentList '-NoExit', '-Command', "cd C:\Orb\Orb; `$env:ORB_MASTER_KEY='MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTIzNDU2Nzg5MDE='; python -m uvicorn main:app --host 0.0.0.0 --port 8000"

Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 4

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Orb Frontend..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Set-Location C:\Orb\orb-desktop
npm run electron:dev
'''
    
    print("\n[1] Writing start_full_orb.ps1 to sandbox...")
    result = client.write_file(
        target="REPO",
        filename="start_full_orb.ps1",
        content=script,
        overwrite=True,
    )
    print(f"    Written: {result.path}")
    
    print("\n[2] Launching with visible window...")
    result = client.shell_run(
        command='cmd /c start powershell -NoExit -File C:\\Orb\\start_full_orb.ps1',
        timeout_seconds=10,
    )
    print(f"    Launched: {result.ok}")
    if result.stderr:
        print(f"    Error: {result.stderr[:200]}")
    
    print("\n" + "=" * 60)
    print("Check the sandbox - you should see:")
    print("  1. PowerShell window running uvicorn (backend)")
    print("  2. Electron window (frontend)")
    print("=" * 60)


if __name__ == "__main__":
    main()
