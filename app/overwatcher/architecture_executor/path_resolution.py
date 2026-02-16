import logging
from typing import Dict, List, Optional

from .constants import FRONTEND_PREFIX, FRONTEND_ROOT
from ..sandbox_client import SandboxClient

logger = logging.getLogger(__name__)


def _resolve_multi_root_path(rel_path: str) -> str:
    """
    Convert architecture relative path to absolute Windows path.
    
    Rules:
    - Paths starting with 'orb-desktop/' -> D:\orb-desktop\...
    - All other paths -> D:\Orb\...
    
    Args:
        rel_path: Architecture-relative path (e.g. "app/models/user.py" or "orb-desktop/src/App.tsx")
    
    Returns:
        Absolute Windows path
    """
    if rel_path.startswith(FRONTEND_PREFIX):
        # Strip prefix and join with frontend root
        sub_path = rel_path[len(FRONTEND_PREFIX):]
        return f"{FRONTEND_ROOT}\\{sub_path.replace('/', '\\')}"
    else:
        # Backend path - use D:\Orb
        return f"D:\\Orb\\{rel_path.replace('/', '\\')}"


def _ensure_python_init_files(abs_path: str, sandbox: SandboxClient) -> List[str]:
    """
    Determine which __init__.py files are missing for a new Python file.
    
    This function performs read-only existence checks in the sandbox to identify
    which __init__.py files need to be created. It does NOT perform any writes.
    
    Args:
        abs_path: Absolute Windows path to new Python file
        sandbox: SandboxClient instance for existence checks
    
    Returns:
        List of absolute paths to __init__.py files that need to be created
    """
    print(f"[ARCH_EXEC] v2.6 _ensure_python_init_files called for: {abs_path}")
    
    if not abs_path.endswith(".py"):
        print(f"[ARCH_EXEC] v2.6 Not a .py file, skipping __init__ check")
        return []
    
    # Split path into parts
    parts = abs_path.split("\\")
    
    # Find the index where 'Orb' appears (the project root)
    try:
        orb_index = parts.index("Orb")
    except ValueError:
        print(f"[ARCH_EXEC] v2.6 'Orb' not found in path, skipping __init__ check")
        return []
    
    needed: List[str] = []
    
    # Check each directory from Orb down to the file's parent
    for i in range(orb_index + 1, len(parts) - 1):
        dir_path = "\\".join(parts[:i + 1])
        init_path = f"{dir_path}\\__init__.py"
        
        # Check if __init__.py exists via sandbox
        check_cmd = f'if exist "{init_path}" (echo EXISTS) else (echo MISSING)'
        result = sandbox.shell_run(check_cmd)
        
        if result and "MISSING" in result:
            print(f"[ARCH_EXEC] v2.6 __init__.py missing at: {init_path}")
            needed.append(init_path)
        else:
            print(f"[ARCH_EXEC] v2.6 __init__.py exists at: {init_path}")
    
    return needed


def _infer_lang_from_path(path: str) -> Optional[str]:
    """
    Infer programming language from file extension.
    
    Args:
        path: File path (relative or absolute)
    
    Returns:
        Language string ("python", "typescript", "javascript") or None if unknown
    """
    path_lower = path.lower()
    
    if path_lower.endswith(".py"):
        return "python"
    elif path_lower.endswith((".ts", ".tsx")):
        return "typescript"
    elif path_lower.endswith((".js", ".jsx")):
        return "javascript"
    
    return None