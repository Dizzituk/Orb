# FILE: app/llm/task_extractor.py
"""
Task Extractor for Orb Routing Pipeline.

Version: 1.0.0 - Critical Pipeline Spec Implementation

Implements Spec §4 (Task Extraction):
- Extracts multiple distinct tasks from a single user message
- Assigns files to tasks
- Determines task priority

TASK FORMAT:
{
    "task_id": "TASK_1",
    "description": "Summarise the PDF [FILE_3].",
    "target_files": ["[FILE_3]"],
    "primary_modalities": ["TEXT", "IMAGE"],
    "priority": 1
}

RULES (Spec §4):
- If user clearly separates tasks ("first do X, then do Y") → multiple tasks
- If user is vague but asks about all files together → single task with multiple target_files
- Each task will run through routing logic independently
- Tasks can be executed sequentially and aggregated

Usage:
    from app.llm.task_extractor import extract_tasks, Task
    
    tasks = await extract_tasks(
        user_text=message,
        classification=classification_result,
        file_map=file_map_string,
        llm_call=cheap_llm_call,
    )
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Enable multi-task extraction (can be disabled for simple use cases)
TASK_EXTRACTION_ENABLED = os.getenv("ORB_TASK_EXTRACTION", "1") == "1"

# Maximum tasks to extract from a single message
MAX_TASKS = 5

# Router debug mode
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"


# =============================================================================
# MODALITY ENUM
# =============================================================================

class Modality(str, Enum):
    """Content modalities."""
    TEXT = "TEXT"
    CODE = "CODE"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    MIXED = "MIXED"


# =============================================================================
# TASK DATA CLASS
# =============================================================================

@dataclass
class Task:
    """
    Extracted task from user message.
    
    Represents a single distinct operation to be performed.
    """
    task_id: str
    description: str
    target_files: List[str]  # e.g., ["[FILE_1]", "[FILE_3]"]
    primary_modalities: List[Modality]
    priority: int = 1  # 1 = highest priority
    
    # Optional: specific user text fragment for this task
    user_text_fragment: str = ""
    
    # Routing hints (filled by router)
    suggested_lane: str = ""
    is_critical: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "target_files": self.target_files,
            "primary_modalities": [m.value for m in self.primary_modalities],
            "priority": self.priority,
            "user_text_fragment": self.user_text_fragment[:200],
            "suggested_lane": self.suggested_lane,
            "is_critical": self.is_critical,
        }


@dataclass
class TaskExtractionResult:
    """Result of task extraction."""
    tasks: List[Task]
    extraction_method: str  # "llm" or "heuristic"
    is_multi_task: bool
    total_files_covered: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_count": len(self.tasks),
            "extraction_method": self.extraction_method,
            "is_multi_task": self.is_multi_task,
            "total_files_covered": self.total_files_covered,
            "tasks": [t.to_dict() for t in self.tasks],
        }


# =============================================================================
# MULTI-TASK DETECTION PATTERNS
# =============================================================================

# Patterns that indicate multiple distinct tasks
MULTI_TASK_PATTERNS = [
    r'\bfirst\b.*\bthen\b',
    r'\b(?:1\)|1\.)\s+.*\b(?:2\)|2\.)\s+',
    r'\band also\b',
    r'\badditionally\b',
    r'\bseparately\b',
    r'\bafter that\b',
    r'\bnext\b.*\bthen\b',
    r'\bfollowed by\b',
]

# Patterns that indicate a single unified task
SINGLE_TASK_PATTERNS = [
    r'\ball (?:of )?(?:the|these|my)\b',
    r'\beverything\b',
    r'\btogether\b',
    r'\bcombined?\b',
    r'\bas a whole\b',
    r'\bin total\b',
]


def detect_multi_task_heuristic(user_text: str) -> bool:
    """
    Detect if message likely contains multiple tasks using heuristics.
    
    Args:
        user_text: User's message
    
    Returns:
        True if multiple tasks likely
    """
    user_lower = user_text.lower()
    
    # Check for explicit single-task patterns
    for pattern in SINGLE_TASK_PATTERNS:
        if re.search(pattern, user_lower):
            return False
    
    # Check for multi-task patterns
    for pattern in MULTI_TASK_PATTERNS:
        if re.search(pattern, user_lower):
            return True
    
    # Check for numbered lists
    if re.search(r'\b[1-5]\.\s+\w', user_text):
        return True
    
    # Check for bullet points
    if user_text.count('\n-') >= 2 or user_text.count('\n*') >= 2:
        return True
    
    return False


# =============================================================================
# FILE REFERENCE EXTRACTION
# =============================================================================

def extract_file_references(text: str) -> List[str]:
    """
    Extract [FILE_X] references from text.
    
    Args:
        text: Text to search
    
    Returns:
        List of file IDs (e.g., ["[FILE_1]", "[FILE_3]"])
    """
    # Match [FILE_X] pattern
    matches = re.findall(r'\[FILE_\d+\]', text)
    return list(dict.fromkeys(matches))  # Dedupe, preserve order


def extract_file_mentions(text: str, file_names: List[str]) -> List[str]:
    """
    Find file names mentioned in text.
    
    Args:
        text: Text to search
        file_names: List of (file_id, filename) to look for
    
    Returns:
        List of file IDs that were mentioned
    """
    mentioned = []
    text_lower = text.lower()
    
    for file_id, filename in file_names:
        # Check for filename mention
        if filename.lower() in text_lower:
            mentioned.append(file_id)
        # Check for [FILE_X] reference
        if file_id in text:
            mentioned.append(file_id)
    
    return list(dict.fromkeys(mentioned))


# =============================================================================
# MODALITY DETECTION
# =============================================================================

def detect_modalities_for_files(
    file_ids: List[str],
    classification: Any,
) -> List[Modality]:
    """
    Detect primary modalities for a set of files.
    
    Args:
        file_ids: List of file IDs
        classification: ClassificationResult
    
    Returns:
        List of Modality enums
    """
    modalities = set()
    
    classified_files = getattr(classification, "classified_files", [])
    
    for cf in classified_files:
        cf_id = getattr(cf, "file_id", "")
        if cf_id not in file_ids:
            continue
        
        file_type = getattr(cf, "file_type", None)
        file_type_value = file_type.value if hasattr(file_type, "value") else str(file_type)
        
        if file_type_value == "VIDEO_FILE":
            modalities.add(Modality.VIDEO)
        elif file_type_value == "IMAGE_FILE":
            modalities.add(Modality.IMAGE)
        elif file_type_value == "CODE_FILE":
            modalities.add(Modality.CODE)
        elif file_type_value == "MIXED_FILE":
            modalities.add(Modality.MIXED)
            modalities.add(Modality.IMAGE)  # Mixed includes images
            modalities.add(Modality.TEXT)
        else:  # TEXT_FILE
            modalities.add(Modality.TEXT)
    
    return list(modalities) or [Modality.TEXT]


# =============================================================================
# HEURISTIC TASK EXTRACTION
# =============================================================================

def extract_tasks_heuristic(
    user_text: str,
    classification: Any,
    file_map: str,
) -> TaskExtractionResult:
    """
    Extract tasks using heuristics (no LLM call).
    
    Rules:
    - If user references specific files → assign those files to task
    - If no specific references → assign all files to single task
    - If multi-task patterns detected → try to split
    
    Args:
        user_text: User's message
        classification: ClassificationResult
        file_map: File map string
    
    Returns:
        TaskExtractionResult
    """
    tasks = []
    classified_files = getattr(classification, "classified_files", [])
    all_file_ids = [getattr(cf, "file_id", "") for cf in classified_files]
    file_names = [(getattr(cf, "file_id", ""), getattr(cf, "original_name", "")) 
                  for cf in classified_files]
    
    # Check for multi-task indicators
    is_multi_task = detect_multi_task_heuristic(user_text)
    
    if is_multi_task:
        # Try to split by numbered items or sentences
        parts = _split_by_task_markers(user_text)
        
        for idx, part in enumerate(parts[:MAX_TASKS]):
            # Find files mentioned in this part
            file_refs = extract_file_references(part)
            mentioned_files = extract_file_mentions(part, file_names)
            target_files = list(dict.fromkeys(file_refs + mentioned_files)) or all_file_ids
            
            modalities = detect_modalities_for_files(target_files, classification)
            
            task = Task(
                task_id=f"TASK_{idx + 1}",
                description=part.strip()[:200],
                target_files=target_files,
                primary_modalities=modalities,
                priority=idx + 1,
                user_text_fragment=part.strip(),
            )
            tasks.append(task)
    
    # If no tasks extracted or single task, create one task for everything
    if not tasks:
        # Check for explicit file references
        file_refs = extract_file_references(user_text)
        mentioned_files = extract_file_mentions(user_text, file_names)
        target_files = list(dict.fromkeys(file_refs + mentioned_files)) or all_file_ids
        
        modalities = detect_modalities_for_files(target_files, classification)
        
        task = Task(
            task_id="TASK_1",
            description=user_text[:200],
            target_files=target_files,
            primary_modalities=modalities,
            priority=1,
            user_text_fragment=user_text,
        )
        tasks.append(task)
    
    # Calculate files covered
    covered = set()
    for task in tasks:
        covered.update(task.target_files)
    
    return TaskExtractionResult(
        tasks=tasks,
        extraction_method="heuristic",
        is_multi_task=len(tasks) > 1,
        total_files_covered=len(covered),
    )


def _split_by_task_markers(text: str) -> List[str]:
    """Split text by task markers (numbers, bullets, 'then')."""
    parts = []
    
    # Try numbered split
    numbered = re.split(r'\n?\s*\d+[.\)]\s+', text)
    if len(numbered) > 1:
        return [p.strip() for p in numbered if p.strip()]
    
    # Try bullet split
    bulleted = re.split(r'\n\s*[-*]\s+', text)
    if len(bulleted) > 1:
        return [p.strip() for p in bulleted if p.strip()]
    
    # Try "then" split
    then_split = re.split(r'\bthen\b', text, flags=re.IGNORECASE)
    if len(then_split) > 1:
        return [p.strip() for p in then_split if p.strip()]
    
    # No split found
    return [text]


# =============================================================================
# LLM-BASED TASK EXTRACTION
# =============================================================================

TASK_EXTRACTION_PROMPT = """Analyze this user request and extract distinct tasks.

USER REQUEST:
{user_text}

FILE MAP:
{file_map}

INSTRUCTIONS:
1. If the user is asking for multiple separate operations (e.g., "summarize X and then debug Y"), extract each as a separate task.
2. If the user is asking about all files together as one operation, return a single task.
3. Assign relevant [FILE_X] references to each task.
4. Priority 1 = most important/first task.

Respond ONLY with JSON (no other text):
{{
    "is_multi_task": true/false,
    "tasks": [
        {{
            "task_id": "TASK_1",
            "description": "Brief description",
            "target_files": ["[FILE_1]", "[FILE_2]"],
            "primary_modalities": ["TEXT", "CODE", "IMAGE", "VIDEO"],
            "priority": 1
        }}
    ]
}}"""


async def extract_tasks_llm(
    user_text: str,
    classification: Any,
    file_map: str,
    llm_call: Callable[[str], Awaitable[str]],
) -> TaskExtractionResult:
    """
    Extract tasks using LLM call.
    
    Args:
        user_text: User's message
        classification: ClassificationResult
        file_map: File map string
        llm_call: Async callable for LLM call
    
    Returns:
        TaskExtractionResult
    """
    prompt = TASK_EXTRACTION_PROMPT.format(
        user_text=user_text[:2000],
        file_map=file_map[:1500],
    )
    
    try:
        response = await llm_call(prompt)
        
        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            
            tasks = []
            for task_data in data.get("tasks", [])[:MAX_TASKS]:
                # Parse modalities
                modalities = []
                for m in task_data.get("primary_modalities", ["TEXT"]):
                    try:
                        modalities.append(Modality(m.upper()))
                    except ValueError:
                        modalities.append(Modality.TEXT)
                
                task = Task(
                    task_id=task_data.get("task_id", f"TASK_{len(tasks)+1}"),
                    description=task_data.get("description", "")[:200],
                    target_files=task_data.get("target_files", []),
                    primary_modalities=modalities or [Modality.TEXT],
                    priority=task_data.get("priority", len(tasks) + 1),
                )
                tasks.append(task)
            
            if tasks:
                # Calculate files covered
                covered = set()
                for task in tasks:
                    covered.update(task.target_files)
                
                return TaskExtractionResult(
                    tasks=tasks,
                    extraction_method="llm",
                    is_multi_task=data.get("is_multi_task", len(tasks) > 1),
                    total_files_covered=len(covered),
                )
        
        logger.warning("[task_extractor] Could not parse LLM response, falling back to heuristic")
        
    except Exception as e:
        logger.warning(f"[task_extractor] LLM extraction failed: {e}, falling back to heuristic")
    
    # Fallback to heuristic
    return extract_tasks_heuristic(user_text, classification, file_map)


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

async def extract_tasks(
    user_text: str,
    classification: Any,
    file_map: str,
    llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
    use_llm: bool = True,
) -> TaskExtractionResult:
    """
    Extract tasks from user message.
    
    This is the main entry point for task extraction.
    
    Args:
        user_text: User's message
        classification: ClassificationResult from file_classifier
        file_map: File map string from build_file_map
        llm_call: Optional async callable for LLM-based extraction
        use_llm: Whether to use LLM extraction
    
    Returns:
        TaskExtractionResult with extracted tasks
    """
    if not TASK_EXTRACTION_ENABLED:
        # Task extraction disabled - return single task
        all_files = [
            getattr(cf, "file_id", "")
            for cf in getattr(classification, "classified_files", [])
        ]
        modalities = detect_modalities_for_files(all_files, classification)
        
        return TaskExtractionResult(
            tasks=[Task(
                task_id="TASK_1",
                description=user_text[:200],
                target_files=all_files,
                primary_modalities=modalities,
                priority=1,
                user_text_fragment=user_text,
            )],
            extraction_method="disabled",
            is_multi_task=False,
            total_files_covered=len(all_files),
        )
    
    # Check if multi-task extraction makes sense
    if use_llm and llm_call and detect_multi_task_heuristic(user_text):
        logger.debug("[task_extractor] Using LLM extraction (multi-task detected)")
        return await extract_tasks_llm(user_text, classification, file_map, llm_call)
    
    # Use heuristic extraction
    logger.debug("[task_extractor] Using heuristic extraction")
    return extract_tasks_heuristic(user_text, classification, file_map)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "Modality",
    
    # Data classes
    "Task",
    "TaskExtractionResult",
    
    # Functions
    "extract_tasks",
    "extract_tasks_heuristic",
    "extract_tasks_llm",
    "detect_multi_task_heuristic",
    "extract_file_references",
    "detect_modalities_for_files",
    
    # Configuration
    "TASK_EXTRACTION_ENABLED",
    "MAX_TASKS",
]