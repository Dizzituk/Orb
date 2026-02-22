from typing import Any, List, Optional


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".m4v", ".wmv", ".flv", ".mpeg", ".mpg",
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".flac", ".m4a",
    ".aac", ".wma", ".aiff",
}

def build_file_map_for_task(
    classified_files: List[ClassifiedFile],
    target_file_ids: List[str],
) -> str:
    """
    Build file map for a specific task (subset of files).
    
    Args:
        classified_files: All classified files
        target_file_ids: List of file IDs for this task (e.g., ["[FILE_1]", "[FILE_3]"])
    
    Returns:
        Formatted file map string with only relevant files
    """
    relevant = [cf for cf in classified_files if cf.file_id in target_file_ids]
    return build_file_map(relevant)

def get_file_by_id(
    classified_files: List[ClassifiedFile],
    file_id: str,
) -> Optional[ClassifiedFile]:
    """Get a classified file by its stable ID."""
    for cf in classified_files:
        if cf.file_id == file_id:
            return cf
    return None

def get_files_by_type(
    classified_files: List[ClassifiedFile],
    file_type: FileType,
) -> List[ClassifiedFile]:
    """Get all files of a specific type."""
    return [cf for cf in classified_files if cf.file_type == file_type]

def has_any_media(result: ClassificationResult) -> bool:
    """Check if classification has any media (images or video)."""
    return result.has_image or result.has_video

def has_vision_content(result: ClassificationResult) -> bool:
    """Check if classification requires vision model (images, video, or mixed)."""
    return result.has_image or result.has_video or result.has_mixed

def classify_from_attachment_info(
    attachment_infos: List[Any],
) -> ClassificationResult:
    """
    Convert from existing AttachmentInfo objects to classification result.
    
    This bridges the old AttachmentInfo schema to the new file classification system.
    
    Args:
        attachment_infos: List of AttachmentInfo objects (from app.llm.schemas)
    
    Returns:
        ClassificationResult
    """
    attachments = []
    
    for att in attachment_infos:
        attachments.append({
            "filename": getattr(att, "filename", "unknown"),
            "mime_type": getattr(att, "mime_type", None),
            "size_bytes": getattr(att, "size_bytes", 0),
            "pdf_image_count": getattr(att, "pdf_image_count", None),
            "pdf_text_chars": getattr(att, "pdf_text_chars", None),
            "pdf_page_count": getattr(att, "pdf_page_count", None),
        })
    
    return classify_attachments(attachments)
