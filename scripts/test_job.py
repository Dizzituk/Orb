# FILE: scripts/test_job.py
"""Test script for Overwatcher job runner.

Usage:
    python scripts/test_job.py
    python scripts/test_job.py --sandbox
    python scripts/test_job.py --shell "dir"
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import asyncio
import argparse


async def test_write_job(use_sandbox: bool = True):
    """Test writing a file via the job runner."""
    from app.overwatcher.job_runner import run_simple_job
    
    print("=" * 60)
    print("Overwatcher Job Runner Test")
    print("=" * 60)
    print()
    
    result = await run_simple_job(
        task="Write a hello world test file to verify the pipeline",
        output_target="DESKTOP",
        output_filename="orb_hello.txt",
        use_sandbox=use_sandbox,
    )
    
    print(f"Success: {result.success}")
    print(f"Sandbox used: {result.sandbox_used}")
    print(f"Output path: {result.output_path}")
    print(f"SHA256: {result.sha256}")
    print(f"Duration: {result.duration_ms}ms")
    print()
    
    if result.success:
        print("✓ Job completed successfully!")
        print(f"  Check: {result.output_path}")
    else:
        print("✗ Job failed!")
        print(f"  Error: {result.job.error}")
    
    return result.success


async def test_shell_job():
    """Test running a shell command via sandbox."""
    from app.overwatcher.job_runner import run_shell_job
    
    print("=" * 60)
    print("Overwatcher Shell Job Test")
    print("=" * 60)
    print()
    
    # Simple directory listing
    result = await run_shell_job("Get-ChildItem -Name | Select-Object -First 10")
    
    print(f"Success: {result.get('success')}")
    print(f"Sandbox used: {result.get('sandbox_used')}")
    print(f"Exit code: {result.get('exit_code')}")
    print(f"Duration: {result.get('duration_ms')}ms")
    print()
    
    if result.get('stdout'):
        print("Output:")
        print("-" * 40)
        print(result.get('stdout'))
    
    if result.get('stderr'):
        print("Stderr:")
        print("-" * 40)
        print(result.get('stderr'))
    
    return result.get('success', False)


async def test_sandbox_connection():
    """Test if sandbox is reachable."""
    from app.overwatcher.sandbox_client import get_sandbox_client
    
    print("=" * 60)
    print("Sandbox Connection Test")
    print("=" * 60)
    print()
    
    client = get_sandbox_client()
    
    print(f"Sandbox URL: {client.base_url}")
    
    try:
        health = client.health()
        print(f"Status: {health.status}")
        print(f"Version: {health.version}")
        print(f"Repo root: {health.repo_root}")
        print(f"Cache root: {health.cache_root}")
        print()
        print("✓ Sandbox is connected!")
        return True
    except Exception as e:
        print(f"✗ Sandbox not reachable: {e}")
        print()
        print("To start the sandbox:")
        print("  1. Open Windows Sandbox")
        print("  2. Run: C:\\Orb\\sandbox_controller\\start_controller.ps1")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Overwatcher job runner")
    parser.add_argument("--sandbox", action="store_true", help="Test with sandbox (default: auto-detect)")
    parser.add_argument("--local", action="store_true", help="Force local execution (no sandbox)")
    parser.add_argument("--shell", action="store_true", help="Test shell command execution")
    parser.add_argument("--check", action="store_true", help="Just check sandbox connection")
    
    args = parser.parse_args()
    
    async def run():
        if args.check:
            return await test_sandbox_connection()
        
        # First check sandbox
        connected = await test_sandbox_connection()
        print()
        
        if args.shell:
            if not connected:
                print("Shell test requires sandbox connection")
                return False
            return await test_shell_job()
        else:
            use_sandbox = connected and not args.local
            return await test_write_job(use_sandbox=use_sandbox)
    
    success = asyncio.run(run())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
