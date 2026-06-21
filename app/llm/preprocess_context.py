# FILE: app/llm/preprocess_context.py
# Purpose: Combined-context block builders (build_task_context / build_critical_context).
# Called-by: app.llm.preprocessor (shim)
# Depends-on: app.llm.preprocess_results
# Last-renovated: 2026-06-21
"""Combined-context string builders for the Orb routing pipeline.

Split out of preprocessor.py (batch 3, 2026-06-21). Pure formatters over a
TaskPreprocessResult; no config or modality coupling.
"""
from app.llm.preprocess_results import TaskPreprocessResult


def build_task_context(
    preprocess_result: TaskPreprocessResult,
    user_text: str,
    file_map: str,
    task_id: str = "TASK_1",
    include_full_code: bool = True,
    include_full_text: bool = False,
) -> str:
    """
    Build combined context block for a task (Spec §6).
    
    Format:
        USER REQUEST FOR TASK_X:
        {user_text}

        FILE MAP:
        [FILE_1] ...

        CONTEXT:
        [TEXT SUMMARY]
        ...
        [CODE SUMMARY]
        ...
        [IMAGE SUMMARY]
        ...
        [VIDEO SUMMARY]
        ...
    
    Args:
        preprocess_result: TaskPreprocessResult from preprocess_task
        user_text: User's message (or relevant fragment)
        file_map: File map string for this task
        task_id: Task identifier
        include_full_code: Include full code content (not just summary)
        include_full_text: Include full text content (not just summary)
    
    Returns:
        Combined context string
    """
    sections = []
    
    # User request
    sections.append(f"USER REQUEST FOR {task_id}:")
    sections.append(user_text)
    sections.append("")
    
    # File map
    sections.append("FILE MAP:")
    sections.append(file_map)
    sections.append("")
    
    # Context sections
    sections.append("CONTEXT:")
    
    # Text summaries
    if preprocess_result.text_results:
        sections.append("")
        sections.append("=== TEXT DOCUMENTS ===")
        for tr in preprocess_result.text_results:
            sections.append(f"[{tr.filename}]:")
            if include_full_text and tr.full_content:
                sections.append(tr.full_content[:5000])
            else:
                sections.append(tr.summary[:2000])
    
    # Code summaries
    if preprocess_result.code_results:
        sections.append("")
        sections.append("=== CODE FILES ===")
        for cr in preprocess_result.code_results:
            sections.append(f"[{cr.filename}]:")
            sections.append(cr.summary)
            if include_full_code and cr.full_content:
                sections.append("--- Code Content ---")
                sections.append(cr.full_content[:10000])
    
    # Image descriptions
    if preprocess_result.image_results:
        sections.append("")
        sections.append("=== IMAGES ===")
        for ir in preprocess_result.image_results:
            sections.append(f"[{ir.filename}]: {ir.description}")
    
    # Video summaries
    if preprocess_result.video_results:
        sections.append("")
        sections.append("=== VIDEO CONTENT ===")
        for vr in preprocess_result.video_results:
            sections.append(f"[{vr.filename}]:")
            sections.append(vr.summary or vr.transcript[:3000])
    
    combined = "\n".join(sections)
    
    # Estimate tokens
    preprocess_result.combined_context = combined
    preprocess_result.combined_context_tokens = len(combined) // 4
    
    return combined


def build_critical_context(
    preprocess_result: TaskPreprocessResult,
    user_text: str,
    file_map: str,
    task_id: str = "TASK_1",
) -> str:
    """
    Build context block for critical pipeline (Spec §8.1).
    
    Adjusted budget for critical ops:
    - 20% user + file map + instructions
    - 45% code (higher weight)
    - 20% video summary/transcript
    - 10% text docs
    - 5% images
    
    Args:
        preprocess_result: TaskPreprocessResult
        user_text: User's message
        file_map: File map string
        task_id: Task identifier
    
    Returns:
        Critical context string
    """
    sections = []
    
    sections.append(f"CRITICAL TASK CONTEXT ({task_id}):")
    sections.append("")
    sections.append("USER REQUEST:")
    sections.append(user_text)
    sections.append("")
    sections.append("FILE MAP:")
    sections.append(file_map)
    sections.append("")
    
    # Text docs (10% budget - shorter)
    if preprocess_result.text_results:
        sections.append("TEXT DOC SUMMARY:")
        for tr in preprocess_result.text_results:
            sections.append(f"[{tr.filename}]: {tr.summary[:1000]}")
        sections.append("")
    
    # Code (45% budget - most space)
    if preprocess_result.code_results:
        sections.append("CODE CONTEXT:")
        for cr in preprocess_result.code_results:
            sections.append(f"[{cr.filename}]:")
            sections.append(cr.summary)
            if cr.full_content:
                sections.append("--- Full Code ---")
                sections.append(cr.full_content[:15000])  # More space for code
            sections.append("")
    
    # Images (5% budget - minimal)
    if preprocess_result.image_results:
        sections.append("IMAGE SUMMARY:")
        for ir in preprocess_result.image_results:
            sections.append(f"[{ir.filename}]: {ir.description[:300]}")
        sections.append("")
    
    # Video (20% budget)
    if preprocess_result.video_results:
        sections.append("VIDEO SUMMARY:")
        for vr in preprocess_result.video_results:
            sections.append(f"[{vr.filename}]:")
            sections.append(vr.summary[:4000] or vr.transcript[:4000])
        sections.append("")
    
    return "\n".join(sections)
