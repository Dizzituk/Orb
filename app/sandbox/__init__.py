# FILE: app/sandbox/__init__.py
"""Sandbox Module: Control the zombie/clone Orb in Windows Sandbox.

This module enables Main Orb to control its sandbox clone for:
- Self-testing: Run tests in isolated environment
- Self-improvement: Test changes before applying to production
- Stress testing: Exercise the clone without affecting main

Components:
- manager.py: Core sandbox management (start/stop/status)
- tools.py: LLM-callable tools and intent detection
- router.py: FastAPI endpoints for API access

Usage:
    # Direct Python usage
    from app.sandbox import start_zombie_orb, stop_zombie_orb, check_zombie_status
    
    success, msg = start_zombie_orb()
    print(msg)  # "🧟 Zombie Orb is now running!"
    
    # LLM tool handling
    from app.sandbox import handle_sandbox_prompt
    
    response = handle_sandbox_prompt("start your zombie")
    if response:
        print(response)  # Handled as sandbox command
    
    # API endpoints (add to main.py)
    from app.sandbox.router import router as sandbox_router
    app.include_router(sandbox_router)
"""

from app.sandbox.manager import (
    SandboxStatus,
    SandboxHealth,
    SandboxManager,
    get_sandbox_manager,
    start_zombie_orb,
    stop_zombie_orb,
    check_zombie_status,
)

from app.sandbox.tools import (
    SANDBOX_TOOLS,
    execute_sandbox_tool,
    detect_sandbox_intent,
    handle_sandbox_prompt,
)

__all__ = [
    # Manager
    "SandboxStatus",
    "SandboxHealth",
    "SandboxManager",
    "get_sandbox_manager",
    "start_zombie_orb",
    "stop_zombie_orb",
    "check_zombie_status",
    # Tools
    "SANDBOX_TOOLS",
    "execute_sandbox_tool",
    "detect_sandbox_intent",
    "handle_sandbox_prompt",
]
