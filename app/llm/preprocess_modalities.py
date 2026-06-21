# FILE: app/llm/preprocess_modalities.py
# Purpose: Per-modality preprocessors (video/image/code/text) + preprocess_task fan-out + config.
# Called-by: app.llm.preprocessor (shim)
# Depends-on: app.llm.preprocess_results, app.llm.file_classifier (lazy)
# Last-renovated: 2026-06-21
"""Per-modality content preprocessors and the preprocess_task orchestrator.

Split out of preprocessor.py (batch 3, 2026-06-21). The config constants are
default-arg values for the modality functions, so they move here with them.
"""
import os
import logging
from typing import Optional, List, Dict, Any, Callable, Awaitable

from app.llm.preprocess_results import (
    VideoPreprocessResult,
    ImagePreprocessResult,
    CodePreprocessResult,
    TextPreprocessResult,
    TaskPreprocessResult,
)

logger = logging.getLogger(__name__)


# Maximum tokens for video transcript (Spec §5.1: 40-50% of max context)
VIDEO_TRANSCRIPT_MAX_TOKENS = int(os.getenv("ORB_VIDEO_TRANSCRIPT_MAX_TOKENS", "80000"))

# Summary lengths
VIDEO_SUMMARY_MAX_CHARS = int(os.getenv("ORB_VIDEO_SUMMARY_MAX_CHARS", "2000"))
IMAGE_SUMMARY_MAX_CHARS = int(os.getenv("ORB_IMAGE_SUMMARY_MAX_CHARS", "1000"))
CODE_SUMMARY_MAX_CHARS = int(os.getenv("ORB_CODE_SUMMARY_MAX_CHARS", "4000"))
TEXT_SUMMARY_MAX_CHARS = int(os.getenv("ORB_TEXT_SUMMARY_MAX_CHARS", "2000"))

# Router debug mode
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"


async def preprocess_video(
    video_file: Any,
    transcribe_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    max_transcript_tokens: int = VIDEO_TRANSCRIPT_MAX_TOKENS,
) -> VideoPreprocessResult:
    """
    Preprocess a video file.
    
    Steps:
    1. Transcribe with Gemini 3 Pro
    2. Generate summary if transcript is long
    
    Args:
        video_file: ClassifiedFile with file_type == VIDEO_FILE
        transcribe_fn: Async function to transcribe video (path -> transcript)
        max_transcript_tokens: Maximum tokens for transcript
    
    Returns:
        VideoPreprocessResult
    """
    import time
    start_time = time.time()
    
    file_id = getattr(video_file, "file_id", "[UNKNOWN]")
    filename = getattr(video_file, "original_name", "unknown")
    file_path = getattr(video_file, "file_path", None)
    
    result = VideoPreprocessResult(file_id=file_id, filename=filename)
    
    if not file_path:
        result.error = "No file path available for video"
        logger.warning(f"[preprocess] Video {filename}: {result.error}")
        return result
    
    try:
        if transcribe_fn:
            # Use provided transcription function
            transcript = await transcribe_fn(file_path)
            result.transcript = transcript
            result.transcript_tokens = len(transcript) // 4  # Rough estimate
            result.model_used = "gemini-3.0-pro-preview"  # Assumed
            
            # Generate summary if transcript is long
            if result.transcript_tokens > 2000:
                # Truncate and summarize
                result.summary = transcript[:VIDEO_SUMMARY_MAX_CHARS]
                result.summary_tokens = len(result.summary) // 4
            else:
                result.summary = transcript
                result.summary_tokens = result.transcript_tokens
                
            logger.debug(f"[preprocess] Video {filename}: {result.transcript_tokens} tokens transcribed")
        else:
            # No transcription function - use stub
            result.transcript = f"[Video: {filename} - transcription not available]"
            result.summary = result.transcript
            logger.warning(f"[preprocess] Video {filename}: No transcription function provided")
            
    except Exception as e:
        result.error = str(e)
        logger.error(f"[preprocess] Video {filename} failed: {e}")
    
    result.duration_ms = int((time.time() - start_time) * 1000)
    return result


# =============================================================================
# IMAGE PREPROCESSING (Spec §5.2)
# =============================================================================

async def preprocess_image(
    image_file: Any,
    describe_fn: Optional[Callable[[str], Awaitable[str]]] = None,
) -> ImagePreprocessResult:
    """
    Preprocess an image file.
    
    Steps:
    1. OCR for text extraction
    2. Semantic description for content understanding
    
    Args:
        image_file: ClassifiedFile with file_type == IMAGE_FILE
        describe_fn: Async function to describe image (path -> description)
    
    Returns:
        ImagePreprocessResult
    """
    import time
    start_time = time.time()
    
    file_id = getattr(image_file, "file_id", "[UNKNOWN]")
    filename = getattr(image_file, "original_name", "unknown")
    file_path = getattr(image_file, "file_path", None)
    
    result = ImagePreprocessResult(file_id=file_id, filename=filename)
    
    if not file_path:
        result.error = "No file path available for image"
        logger.warning(f"[preprocess] Image {filename}: {result.error}")
        return result
    
    try:
        if describe_fn:
            description = await describe_fn(file_path)
            result.description = description[:IMAGE_SUMMARY_MAX_CHARS]
            result.model_used = "gemini-2.5-pro"  # Assumed
            logger.debug(f"[preprocess] Image {filename}: {len(result.description)} chars description")
        else:
            result.description = f"[Image: {filename}]"
            logger.warning(f"[preprocess] Image {filename}: No describe function provided")
            
    except Exception as e:
        result.error = str(e)
        logger.error(f"[preprocess] Image {filename} failed: {e}")
    
    result.duration_ms = int((time.time() - start_time) * 1000)
    return result


# =============================================================================
# CODE PREPROCESSING (Spec §5.3)
# =============================================================================

import re

def preprocess_code_sync(
    code_file: Any,
    max_summary_chars: int = CODE_SUMMARY_MAX_CHARS,
) -> CodePreprocessResult:
    """
    Preprocess a code file (synchronous).
    
    Extracts:
    - Classes and methods
    - Functions
    - Imports
    - Error locations (if referenced)
    - Key snippets
    
    Args:
        code_file: ClassifiedFile with file_type == CODE_FILE
        max_summary_chars: Maximum characters for summary
    
    Returns:
        CodePreprocessResult
    """
    file_id = getattr(code_file, "file_id", "[UNKNOWN]")
    filename = getattr(code_file, "original_name", "unknown")
    content = getattr(code_file, "extracted_text", "") or ""
    
    result = CodePreprocessResult(
        file_id=file_id,
        filename=filename,
        full_content=content,
        full_content_chars=len(content),
    )
    
    if not content:
        result.error = "No content extracted from code file"
        return result
    
    try:
        # Extract classes
        classes = re.findall(r'class\s+(\w+)', content)
        result.classes = classes[:20]
        
        # Extract functions/methods
        functions = re.findall(r'(?:def|function|async function|const|let|var)\s+(\w+)\s*[=\(]', content)
        result.functions = list(dict.fromkeys(functions))[:30]  # Dedupe, limit
        
        # Extract imports
        imports = re.findall(r'(?:import|from)\s+([^\s;]+)', content[:3000])
        result.imports = list(set(imports))[:20]
        
        # Build summary
        summary_parts = [f"File: {filename}"]
        
        if result.classes:
            summary_parts.append(f"Classes: {', '.join(result.classes[:10])}")
        
        if result.functions:
            summary_parts.append(f"Functions: {', '.join(result.functions[:15])}")
        
        if result.imports:
            summary_parts.append(f"Imports: {', '.join(result.imports[:10])}")
        
        # Add line count
        line_count = content.count('\n') + 1
        summary_parts.append(f"Lines: {line_count}")
        
        result.summary = "\n".join(summary_parts)[:max_summary_chars]
        
        # Extract key snippets (first 2000 chars as fallback)
        result.key_snippets = [content[:2000]]
        result.key_snippets_chars = min(2000, len(content))
        
        logger.debug(f"[preprocess] Code {filename}: {len(result.classes)} classes, {len(result.functions)} functions")
        
    except Exception as e:
        result.error = str(e)
        logger.error(f"[preprocess] Code {filename} failed: {e}")
    
    return result


async def preprocess_code(
    code_file: Any,
    max_summary_chars: int = CODE_SUMMARY_MAX_CHARS,
) -> CodePreprocessResult:
    """Async wrapper for code preprocessing."""
    return preprocess_code_sync(code_file, max_summary_chars)


# =============================================================================
# TEXT PREPROCESSING (Spec §5.4)
# =============================================================================

def preprocess_text_sync(
    text_file: Any,
    max_summary_chars: int = TEXT_SUMMARY_MAX_CHARS,
) -> TextPreprocessResult:
    """
    Preprocess a text file (synchronous).
    
    Extracts:
    - High-level summary
    - Key excerpts
    
    Args:
        text_file: ClassifiedFile with file_type == TEXT_FILE
        max_summary_chars: Maximum characters for summary
    
    Returns:
        TextPreprocessResult
    """
    file_id = getattr(text_file, "file_id", "[UNKNOWN]")
    filename = getattr(text_file, "original_name", "unknown")
    content = getattr(text_file, "extracted_text", "") or ""
    
    result = TextPreprocessResult(
        file_id=file_id,
        filename=filename,
        full_content=content,
        full_content_chars=len(content),
    )
    
    if not content:
        result.error = "No content extracted from text file"
        return result
    
    try:
        # Simple summary: first N characters
        result.summary = content[:max_summary_chars]
        
        # Key excerpts: split into paragraphs, take first few
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        result.key_excerpts = paragraphs[:5]
        
        logger.debug(f"[preprocess] Text {filename}: {len(content)} chars, {len(paragraphs)} paragraphs")
        
    except Exception as e:
        result.error = str(e)
        logger.error(f"[preprocess] Text {filename} failed: {e}")
    
    return result


async def preprocess_text(
    text_file: Any,
    max_summary_chars: int = TEXT_SUMMARY_MAX_CHARS,
) -> TextPreprocessResult:
    """Async wrapper for text preprocessing."""
    return preprocess_text_sync(text_file, max_summary_chars)


# =============================================================================
# TASK PREPROCESSING
# =============================================================================

async def preprocess_task(
    classification: Any,
    task_file_ids: List[str],
    transcribe_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    describe_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    task_id: str = "TASK_1",
) -> TaskPreprocessResult:
    """
    Preprocess all files for a task.
    
    Args:
        classification: ClassificationResult from file_classifier
        task_file_ids: List of file IDs for this task
        transcribe_fn: Async function for video transcription
        describe_fn: Async function for image description
        task_id: Task identifier
    
    Returns:
        TaskPreprocessResult with all preprocessing done
    """
    from datetime import datetime
    result = TaskPreprocessResult(
        task_id=task_id,
        file_ids=task_file_ids,
        started_at=datetime.utcnow(),
    )
    
    # Get files by ID
    all_files = getattr(classification, "classified_files", [])
    task_files = [f for f in all_files if getattr(f, "file_id", "") in task_file_ids]
    
    # Import FileType if available
    try:
        from app.llm.file_classifier import FileType
    except ImportError:
        # Fallback: use string comparison
        class FileType:
            VIDEO_FILE = "VIDEO_FILE"
            IMAGE_FILE = "IMAGE_FILE"
            CODE_FILE = "CODE_FILE"
            TEXT_FILE = "TEXT_FILE"
            MIXED_FILE = "MIXED_FILE"
    
    # Process each file by type
    for f in task_files:
        file_type = getattr(f, "file_type", None)
        file_type_value = file_type.value if hasattr(file_type, "value") else str(file_type)
        
        try:
            if file_type_value == "VIDEO_FILE":
                video_result = await preprocess_video(f, transcribe_fn)
                result.video_results.append(video_result)
                if video_result.error:
                    result.errors.append(f"Video {video_result.filename}: {video_result.error}")
                    
            elif file_type_value == "IMAGE_FILE":
                image_result = await preprocess_image(f, describe_fn)
                result.image_results.append(image_result)
                if image_result.error:
                    result.errors.append(f"Image {image_result.filename}: {image_result.error}")
                    
            elif file_type_value == "CODE_FILE":
                code_result = await preprocess_code(f)
                result.code_results.append(code_result)
                if code_result.error:
                    result.errors.append(f"Code {code_result.filename}: {code_result.error}")
                    
            elif file_type_value in ("TEXT_FILE", "MIXED_FILE"):
                text_result = await preprocess_text(f)
                result.text_results.append(text_result)
                if text_result.error:
                    result.errors.append(f"Text {text_result.filename}: {text_result.error}")
                    
        except Exception as e:
            filename = getattr(f, "original_name", "unknown")
            result.errors.append(f"{filename}: {str(e)}")
            logger.error(f"[preprocess] Error processing {filename}: {e}")
    
    result.completed_at = datetime.utcnow()
    if result.started_at:
        delta = result.completed_at - result.started_at
        result.total_duration_ms = int(delta.total_seconds() * 1000)
    
    logger.info(
        f"[preprocess] Task {task_id} complete: "
        f"videos={len(result.video_results)}, images={len(result.image_results)}, "
        f"code={len(result.code_results)}, text={len(result.text_results)}, "
        f"errors={len(result.errors)}"
    )
    
    return result
