"""
Sandbox Read-Only Operations

Provides foundational sandbox verification and file reading operations.
These are imported by multiple architecture executor segments and must maintain
stable signatures and semantics.
"""

import logging
from typing import Any, Dict, Optional

from ..sandbox_client import SandboxClient
from .constants import VERIFY_READ_TIMEOUT

logger = logging.getLogger(__name__)

__all__ = ["_verify_file_via_sandbox", "_resolve_sandbox_base", "_read_existing_file"]


def _verify_file_via_sandbox(
    client: SandboxClient,
    path: str,
    expected_min_chars: int = 10
) -> Dict[str, Any]:
    """
    Verify a file exists in the sandbox and meets minimum character requirements.
    
    Uses PowerShell Get-Content with UTF-8 encoding to read the file and validate
    its existence and content length.
    
    Args:
        client: Sandbox client instance
        path: Path to file in sandbox
        expected_min_chars: Minimum expected character count (default: 10)
        
    Returns:
        Dict with keys:
            - exists (bool): Whether file exists
            - chars (int): Character count (0 if file doesn't exist)
            - valid (bool): Whether file meets minimum requirements
            - error (str or None): Error message if validation failed
    """
    script = f"""
    if (Test-Path "{path}") {{
        $content = Get-Content -Path "{path}" -Raw -Encoding UTF8
        $chars = $content.Length
        Write-Output "EXISTS:$chars"
    }} else {{
        Write-Output "MISSING"
    }}
    """
    
    try:
        result = client.execute_script(script, timeout=VERIFY_READ_TIMEOUT)
        output = result.get("output", "").strip()
        
        if output.startswith("EXISTS:"):
            char_count_str = output.split(":", 1)[1]
            char_count = int(char_count_str)
            
            if char_count < expected_min_chars:
                return {
                    "exists": True,
                    "chars": char_count,
                    "valid": False,
                    "error": f"File too short: {char_count} chars < {expected_min_chars}"
                }
            
            return {
                "exists": True,
                "chars": char_count,
                "valid": True,
                "error": None
            }
        elif output == "MISSING":
            return {
                "exists": False,
                "chars": 0,
                "valid": False,
                "error": "File does not exist"
            }
        else:
            return {
                "exists": False,
                "chars": 0,
                "valid": False,
                "error": f"Unexpected output: {output}"
            }
            
    except Exception as e:
        logger.error(f"Verification failed for {path}: {e}")
        return {
            "exists": False,
            "chars": 0,
            "valid": False,
            "error": str(e)
        }


def _resolve_sandbox_base(client: SandboxClient) -> str:
    """
    Resolve the sandbox base directory by checking candidate locations.
    
    Probes multiple candidate directories and checks for the existence of main.py
    to determine the correct sandbox base path.
    
    Args:
        client: Sandbox client instance
        
    Returns:
        str: Resolved sandbox base directory path (defaults to D:\\Orb if not found)
    """
    candidates = [
        r"C:\Orb\Orb",
        r"C:\Orb",
        r"D:\Orb"
    ]
    
    for candidate in candidates:
        script = f'Test-Path "{candidate}\\main.py"'
        try:
            result = client.execute_script(script, timeout=10)
            output = result.get("output", "").strip()
            
            if output.lower() == "true":
                logger.info(f"Resolved sandbox base: {candidate}")
                return candidate
        except Exception as e:
            logger.debug(f"Could not check {candidate}: {e}")
            continue
    
    default = r"D:\Orb"
    logger.warning(f"Could not resolve sandbox base, defaulting to {default}")
    return default


def _read_existing_file(client: SandboxClient, path: str) -> Optional[str]:
    """
    Read an existing file from the sandbox.
    
    Uses PowerShell Get-Content with UTF-8 encoding to read file content.
    
    Args:
        client: Sandbox client instance
        path: Path to file in sandbox
        
    Returns:
        str: File content if successful, None on error
    """
    script = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'
    
    try:
        result = client.execute_script(script, timeout=30)
        content = result.get("output", "")
        
        if result.get("exit_code") == 0:
            return content
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Failed to read {path}: {error_msg}")
            return None
            
    except Exception as e:
        logger.error(f"Exception reading {path}: {e}")
        return None