# FILE: app/llm/file_analyzer.py
"""
File analysis utilities for document extraction and parsing.
Handles DOCX, PDF, TXT, and other text-based files.
"""
import os
import json
from pathlib import Path
from typing import Optional, Callable

# Supported text-based file extensions
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".java", ".kt", ".swift", ".go", ".rs", ".c", ".cpp", ".h",
    ".sql", ".sh", ".bat", ".ps1", ".log", ".ini", ".cfg", ".conf",
}


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """
    Extract text from a DOCX file using python-docx.
    
    Args:
        file_path: Path to the DOCX file
    
    Returns:
        Extracted text or None if failed
    """
    try:
        from docx import Document
        doc = Document(file_path)
        
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        
        # Also extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        
        content = "\n\n".join(paragraphs)
        print(f"[file_analyzer] Extracted {len(content)} chars from DOCX: {file_path}")
        return content if content else None
        
    except ImportError:
        print("[file_analyzer] ERROR: python-docx not installed. Run: pip install python-docx")
        return None
    except Exception as e:
        print(f"[file_analyzer] Error reading DOCX {file_path}: {e}")
        return None


def extract_text_from_pdf(file_path: str, max_chars: int = 50000) -> Optional[str]:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        max_chars: Maximum characters to extract
    
    Returns:
        Extracted text or None if failed
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        
        text_parts = []
        total_len = 0
        
        for page in doc:
            page_text = page.get_text()
            text_parts.append(page_text)
            total_len += len(page_text)
            if total_len > max_chars:
                break
        
        doc.close()
        content = "\n\n".join(text_parts)[:max_chars]
        print(f"[file_analyzer] Extracted {len(content)} chars from PDF: {file_path}")
        return content if content else None
        
    except ImportError:
        print("[file_analyzer] PyMuPDF not installed. Run: pip install pymupdf")
        return None
    except Exception as e:
        print(f"[file_analyzer] Error reading PDF {file_path}: {e}")
        return None


def extract_text_from_txt(file_path: str, max_chars: int = 50000) -> Optional[str]:
    """
    Extract text from a plain text file.
    
    Args:
        file_path: Path to the text file
        max_chars: Maximum characters to extract
    
    Returns:
        Extracted text or None if failed
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_chars)
        print(f"[file_analyzer] Extracted {len(content)} chars from TXT: {file_path}")
        return content if content else None
    except Exception as e:
        print(f"[file_analyzer] Error reading TXT {file_path}: {e}")
        return None


def extract_text_content(file_path: str, max_chars: int = 50000) -> Optional[str]:
    """
    Extract text content from a file based on its extension.
    
    Args:
        file_path: Path to the file
        max_chars: Maximum characters to extract
    
    Returns:
        Extracted text or None if not readable
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    print(f"[file_analyzer] Extracting from: {file_path} (type: {suffix})")
    
    # DOCX files
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    
    # PDF files
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path, max_chars)
    
    # Plain text files
    if suffix in TEXT_EXTENSIONS or suffix in {".doc", ".rtf"}:
        return extract_text_from_txt(file_path, max_chars)
    
    # Try to read as text anyway
    try:
        return extract_text_from_txt(file_path, max_chars)
    except Exception:
        print(f"[file_analyzer] Cannot extract text from: {file_path}")
        return None


def detect_document_type(filename: str, content: str) -> str:
    """
    Detect the type of document based on filename and content.
    
    Returns: cv, resume, report, notes, code, config, data, document
    """
    filename_lower = filename.lower()
    content_lower = content.lower()[:2000] if content else ""
    
    # CV/Resume detection
    cv_keywords = ["cv", "curriculum vitae", "resume", "résumé"]
    if any(kw in filename_lower for kw in cv_keywords):
        return "cv"
    
    resume_content_keywords = [
        "work experience", "employment history", "professional experience",
        "education", "qualifications", "skills", "references",
        "career objective", "work history"
    ]
    if sum(1 for kw in resume_content_keywords if kw in content_lower) >= 3:
        return "cv"
    
    # Code detection
    code_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}
    if Path(filename).suffix.lower() in code_extensions:
        return "code"
    
    # Config detection
    config_extensions = {".json", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".env"}
    if Path(filename).suffix.lower() in config_extensions:
        return "config"
    
    # Data detection
    if Path(filename).suffix.lower() in {".csv", ".xml"}:
        return "data"
    
    return "document"


def parse_cv_with_llm(content: str, filename: str, llm_call_func: Callable[[str], str]) -> dict:
    """
    Parse a CV/resume using an LLM to extract structured data.
    
    Args:
        content: Raw text content of the CV
        filename: Original filename
        llm_call_func: Function to call LLM (takes prompt, returns response)
    
    Returns:
        dict with parsed CV data
    """
    prompt = f"""Analyze this CV/resume and extract structured information.

FILENAME: {filename}

CV CONTENT:
{content[:8000]}

Extract and return a JSON object with:
{{
    "name": "Full name of the person",
    "summary": "A 2-3 sentence professional summary",
    "roles": [
        {{
            "title": "Job title",
            "employer": "Company/Organization name",
            "location": "City/Country if mentioned",
            "start_date": "YYYY-MM or YYYY or approximate",
            "end_date": "YYYY-MM or YYYY or 'present' or null",
            "description": "Brief description of role"
        }}
    ],
    "skills": ["skill1", "skill2", ...],
    "education": [
        {{
            "institution": "School/University name",
            "qualification": "Degree/Certificate",
            "year": "Year completed or range"
        }}
    ]
}}

IMPORTANT:
- Include ALL roles/jobs mentioned, even part-time or casual work
- For dates, use whatever format is in the CV
- If information is missing, use null
- Return ONLY valid JSON, no other text"""

    try:
        response = llm_call_func(prompt)
        
        # Clean up response
        response_text = response.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines if they're markdown code fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        parsed = json.loads(response_text)
        print(f"[file_analyzer] Parsed CV with {len(parsed.get('roles', []))} roles")
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"[file_analyzer] Failed to parse CV JSON: {e}")
        print(f"[file_analyzer] Raw response: {response[:500]}")
        return {
            "name": None,
            "summary": f"CV document: {filename}",
            "roles": [],
            "skills": [],
            "education": [],
            "parse_error": str(e)
        }
    except Exception as e:
        print(f"[file_analyzer] Error parsing CV: {e}")
        return {
            "name": None,
            "summary": f"CV document: {filename}",
            "roles": [],
            "skills": [],
            "education": [],
            "parse_error": str(e)
        }


def generate_document_summary(
    content: str, 
    filename: str, 
    doc_type: str,
    llm_call_func: Callable[[str], str]
) -> str:
    """
    Generate a summary of a document using an LLM.
    
    Args:
        content: Raw text content
        filename: Original filename
        doc_type: Type of document (cv, report, etc.)
        llm_call_func: Function to call LLM
    
    Returns:
        Summary string
    """
    if doc_type == "cv":
        prompt = f"""Summarize this CV in 2-3 sentences, mentioning:
- The person's name (if visible)
- Their main profession/field
- Number of roles/years of experience
- Key skills or notable employers

CV CONTENT:
{content[:5000]}

Respond with ONLY the summary, no other text."""
    else:
        prompt = f"""Summarize this {doc_type} document in 2-3 sentences.

FILENAME: {filename}

CONTENT:
{content[:5000]}

Respond with ONLY the summary, no other text."""

    try:
        summary = llm_call_func(prompt)
        return summary.strip()
    except Exception as e:
        print(f"[file_analyzer] Error generating summary: {e}")
        return f"Document: {filename}"