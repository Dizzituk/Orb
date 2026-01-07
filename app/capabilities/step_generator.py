# FILE: app/pot_spec/step_generator.py
"""
Auto-Generate Steps from Capabilities

Instead of asking users "what steps should the system take?", this module
generates steps automatically based on:
1. The task objective
2. Available ASTRA capabilities
3. The target outputs

Users provide WHAT they want, the system figures out HOW.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Standard step sequences for common task types
# =============================================================================

TASK_TEMPLATES = {
    "create_file_in_folder": [
        "Scan sandbox to locate target folder: {folder}",
        "Navigate to {folder} in sandbox",
        "Create/overwrite file {filename}",
        "Write content: {content_preview}",
        "Verify file exists with correct content",
    ],
    "modify_existing_file": [
        "Scan sandbox to locate file: {filename}",
        "Read current file contents",
        "Apply modifications as specified",
        "Save updated file",
        "Verify changes applied correctly",
    ],
    "execute_and_verify": [
        "Prepare execution environment in sandbox",
        "Execute specified operation",
        "Capture output/results",
        "Verify against acceptance criteria",
    ],
}


# =============================================================================
# Step Generation Logic
# =============================================================================

def generate_steps_from_task(
    objective: str,
    outputs: List[Dict[str, Any]],
    content_verbatim: Optional[str] = None,
    location: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Auto-generate execution steps based on task requirements and capabilities.
    
    Args:
        objective: What the user wants to accomplish
        outputs: Expected output artifacts
        content_verbatim: Exact content to write (if any)
        location: Target location (if any)
        capabilities: Available ASTRA capabilities (optional)
        
    Returns:
        List of execution steps
    """
    steps = []
    objective_lower = objective.lower()
    
    # Detect task type from objective and outputs
    task_type = _detect_task_type(objective_lower, outputs)
    
    # Extract key entities
    entities = _extract_entities(objective, outputs, content_verbatim, location)
    
    # Generate steps based on task type
    if task_type == "create_file":
        steps = _generate_file_creation_steps(entities)
    elif task_type == "modify_file":
        steps = _generate_file_modification_steps(entities)
    elif task_type == "execute_script":
        steps = _generate_script_execution_steps(entities)
    elif task_type == "locate_and_report":
        steps = _generate_locate_steps(entities)
    else:
        # Generic task - build from patterns
        steps = _generate_generic_steps(objective, entities)
    
    # Always add verification step if we have outputs
    if outputs and not any("verify" in s.lower() for s in steps):
        steps.append("Verify all outputs exist with correct content")
    
    logger.info(f"[step_generator] Generated {len(steps)} steps for task type: {task_type}")
    return steps


def _detect_task_type(objective: str, outputs: List[Dict[str, Any]]) -> str:
    """Detect the type of task from objective and outputs."""
    
    # File creation indicators
    if any(kw in objective for kw in ["create", "write", "make", "new file"]):
        return "create_file"
    
    # File modification indicators
    if any(kw in objective for kw in ["modify", "edit", "update", "change", "overwrite"]):
        return "modify_file"
    
    # Script execution indicators
    if any(kw in objective for kw in ["run", "execute", "script"]):
        return "execute_script"
    
    # Location/search indicators
    if any(kw in objective for kw in ["find", "locate", "where", "search"]):
        return "locate_and_report"
    
    # Check outputs for hints
    if outputs:
        output_types = [o.get("type", "file") for o in outputs]
        if all(t == "file" for t in output_types):
            actions = [o.get("action", "add") for o in outputs]
            if all(a == "add" for a in actions):
                return "create_file"
            else:
                return "modify_file"
    
    return "generic"


def _extract_entities(
    objective: str,
    outputs: List[Dict[str, Any]],
    content_verbatim: Optional[str],
    location: Optional[str],
) -> Dict[str, Any]:
    """Extract key entities (files, folders, content) from task."""
    
    entities = {
        "filename": None,
        "folder": None,
        "content": content_verbatim,
        "location": location,
        "outputs": outputs,
    }
    
    # Extract filename from outputs
    if outputs:
        first_output = outputs[0]
        entities["filename"] = first_output.get("name")
        entities["folder"] = first_output.get("path") or location
        entities["content"] = first_output.get("content") or content_verbatim
    
    # Try to extract folder from location string
    if location and not entities["folder"]:
        entities["folder"] = _parse_location(location)
    
    return entities


def _parse_location(location: str) -> str:
    """Parse natural language location to path."""
    location_lower = location.lower()
    
    # Common patterns
    if "desktop" in location_lower:
        path = "Desktop"
        # Check for subfolder like "desktop/test" or "desktop\test"
        match = re.search(r'desktop[/\\](\w+)', location_lower)
        if match:
            path = f"Desktop\\{match.group(1)}"
        # Also check "folder called X"
        match = re.search(r'folder\s+(?:called\s+)?(\w+)', location_lower)
        if match:
            path = f"Desktop\\{match.group(1)}"
        return path
    
    return location


def _generate_file_creation_steps(entities: Dict[str, Any]) -> List[str]:
    """Generate steps for file creation task."""
    steps = []
    
    folder = entities.get("folder") or "target location"
    filename = entities.get("filename") or "output file"
    content = entities.get("content")
    
    # Step 1: Locate/verify folder exists
    steps.append(f"Locate target folder in sandbox: {folder}")
    
    # Step 2: Navigate
    steps.append(f"Navigate to {folder}")
    
    # Step 3: Check if file exists (for overwrite)
    steps.append(f"Check if {filename} already exists (handle overwrite)")
    
    # Step 4: Create file with content
    if content:
        content_preview = content[:30] + "..." if len(content) > 30 else content
        steps.append(f"Create {filename} with content: \"{content_preview}\"")
    else:
        steps.append(f"Create {filename}")
    
    # Step 5: Verify
    steps.append(f"Verify {filename} exists with correct content")
    
    return steps


def _generate_file_modification_steps(entities: Dict[str, Any]) -> List[str]:
    """Generate steps for file modification task."""
    steps = []
    
    filename = entities.get("filename") or "target file"
    folder = entities.get("folder")
    
    if folder:
        steps.append(f"Locate {filename} in {folder}")
    else:
        steps.append(f"Scan sandbox to locate {filename}")
    
    steps.append(f"Read current contents of {filename}")
    steps.append("Apply specified modifications")
    steps.append(f"Save updated {filename}")
    steps.append("Verify modifications applied correctly")
    
    return steps


def _generate_script_execution_steps(entities: Dict[str, Any]) -> List[str]:
    """Generate steps for script execution task."""
    return [
        "Prepare sandbox execution environment",
        "Load/create script to execute",
        "Execute script in sandbox",
        "Capture execution output",
        "Verify expected results",
    ]


def _generate_locate_steps(entities: Dict[str, Any]) -> List[str]:
    """Generate steps for locate/find task."""
    target = entities.get("filename") or entities.get("folder") or "target"
    return [
        f"Scan sandbox filesystem for {target}",
        "Report location and metadata",
        "Verify accessibility",
    ]


def _generate_generic_steps(objective: str, entities: Dict[str, Any]) -> List[str]:
    """Generate generic steps when task type is unclear."""
    steps = []
    
    # Always start with scanning/understanding the environment
    steps.append("Scan sandbox to understand current state")
    
    objective_lower = objective.lower()
    
    if any(kw in objective_lower for kw in ["go", "navigate", "open"]):
        steps.append(f"Navigate to specified location: {entities.get('folder', 'target')}")
    
    if any(kw in objective_lower for kw in ["create", "write", "make"]):
        steps.append(f"Create specified output: {entities.get('filename', 'file')}")
    
    if any(kw in objective_lower for kw in ["read", "check", "view"]):
        steps.append("Read and verify contents")
    
    # Fallback if nothing matched
    if len(steps) == 1:
        steps.append("Execute primary task operation")
        steps.append("Verify operation completed successfully")
    
    return steps


# =============================================================================
# Integration with Spec Gate
# =============================================================================

def should_ask_for_steps(
    objective: str,
    outputs: List[Dict[str, Any]],
    location: Optional[str] = None,
) -> bool:
    """
    Determine if we need to ask user for steps or can auto-generate.
    
    Returns True only if the task is too ambiguous to auto-generate steps.
    Most tasks should return False (we can figure it out).
    """
    # If we have clear outputs with filenames, we can generate steps
    if outputs and any(o.get("name") for o in outputs):
        return False
    
    # If we have objective + location, we can usually figure it out
    if objective and location:
        return False
    
    # If objective contains clear action verbs, we can generate
    action_verbs = ["create", "write", "modify", "delete", "run", "execute", "find", "locate"]
    if any(verb in objective.lower() for verb in action_verbs):
        return False
    
    # Only ask if truly ambiguous
    return len(objective.split()) < 5 and not outputs


def enhance_spec_with_auto_steps(
    spec_dict: Dict[str, Any],
    capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enhance a spec dictionary with auto-generated steps if missing.
    
    This is called by Spec Gate when steps are empty/missing.
    """
    if spec_dict.get("steps"):
        # Already has steps, don't override
        return spec_dict
    
    # Generate steps
    steps = generate_steps_from_task(
        objective=spec_dict.get("objective", ""),
        outputs=spec_dict.get("outputs", []),
        content_verbatim=spec_dict.get("content_verbatim"),
        location=spec_dict.get("location"),
        capabilities=capabilities,
    )
    
    if steps:
        spec_dict["steps"] = steps
        logger.info(f"[step_generator] Auto-generated {len(steps)} steps for spec")
    
    return spec_dict


__all__ = [
    "generate_steps_from_task",
    "should_ask_for_steps",
    "enhance_spec_with_auto_steps",
]
