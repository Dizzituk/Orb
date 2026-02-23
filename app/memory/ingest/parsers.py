# FILE: app/memory/ingest/parsers.py
"""
Format-specific document parsers (Spec Section 9.2, Stage 1).

Each parser extracts raw text and structural metadata from a
specific file format. All parsers return a list of ParsedChunk
objects — uniform data regardless of source format.

Supported formats:
    JSON  — OpenAI conversation exports, structured data files
    PDF   — Text extraction (pdfplumber or fallback)
    MD    — Markdown with section-aware chunking
    CSV   — Row-based with header preservation
    TXT   — Plain text with paragraph splitting

Usage:
    from app.memory.ingest.parsers import parse_file

    chunks = parse_file("conversations.json")
    chunks = parse_file("document.pdf")
"""

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Parsed chunk
# =========================================================================

@dataclass
class ParsedChunk:
    """A single chunk of extracted content."""
    text: str
    source_file: str
    chunk_index: int
    format: str                     # json, pdf, md, csv, txt
    metadata: dict = field(default_factory=dict)
    # metadata examples:
    #   json: {"conversation_id": "...", "role": "assistant"}
    #   pdf:  {"page": 3}
    #   md:   {"section": "## Architecture", "heading_level": 2}
    #   csv:  {"row": 5, "headers": [...]}


# =========================================================================
# Dispatcher
# =========================================================================

def parse_file(path: str) -> list[ParsedChunk]:
    """
    Parse a file into chunks based on its extension.

    Args:
        path: File path to parse.

    Returns:
        List of ParsedChunk objects.

    Raises:
        ValueError: If format is unsupported.
        FileNotFoundError: If file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    parsers = {
        ".json": _parse_json,
        ".pdf": _parse_pdf,
        ".md": _parse_markdown,
        ".markdown": _parse_markdown,
        ".csv": _parse_csv,
        ".tsv": _parse_csv,
        ".txt": _parse_text,
    }

    parser = parsers.get(ext)
    if not parser:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported: {', '.join(parsers.keys())}"
        )

    chunks = parser(p)
    logger.info(
        "[parsers] Parsed %s: %d chunks from %s",
        ext, len(chunks), p.name,
    )
    return chunks


# =========================================================================
# JSON parser (including GPT export)
# =========================================================================

def _parse_json(path: Path) -> list[ParsedChunk]:
    """Parse JSON file — handles GPT exports and generic JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect GPT conversation export format
    if isinstance(data, list) and data and _is_gpt_export(data[0]):
        return _parse_gpt_export(data, str(path))

    # Generic JSON — stringify with context
    if isinstance(data, list):
        return [
            ParsedChunk(
                text=json.dumps(item, indent=2, default=str),
                source_file=str(path),
                chunk_index=i,
                format="json",
                metadata={"type": "json_array_item", "index": i},
            )
            for i, item in enumerate(data)
            if _has_content(item)
        ]

    # Single object
    return [ParsedChunk(
        text=json.dumps(data, indent=2, default=str),
        source_file=str(path),
        chunk_index=0,
        format="json",
        metadata={"type": "json_object"},
    )]


def _is_gpt_export(item: dict) -> bool:
    """Check if a JSON item looks like a GPT conversation export."""
    return (
        isinstance(item, dict)
        and "mapping" in item
        and "title" in item
    )


def _parse_gpt_export(conversations: list, source: str) -> list[ParsedChunk]:
    """Parse OpenAI's conversation export format."""
    chunks = []
    chunk_idx = 0

    for conv in conversations:
        if not isinstance(conv, dict):
            continue

        conv_id = conv.get("id", "unknown")
        title = conv.get("title", "Untitled")
        mapping = conv.get("mapping", {})

        for node_id, node in mapping.items():
            msg = node.get("message")
            if not msg:
                continue

            role = msg.get("author", {}).get("role", "unknown")
            parts = msg.get("content", {}).get("parts", [])

            text_parts = [
                str(p) for p in parts
                if isinstance(p, str) and p.strip()
            ]
            if not text_parts:
                continue

            text = "\n".join(text_parts)
            if len(text.strip()) < 10:
                continue

            chunks.append(ParsedChunk(
                text=text,
                source_file=source,
                chunk_index=chunk_idx,
                format="json",
                metadata={
                    "type": "gpt_export",
                    "conversation_id": conv_id,
                    "conversation_title": title,
                    "role": role,
                    "node_id": node_id,
                },
            ))
            chunk_idx += 1

    return chunks


# =========================================================================
# PDF parser
# =========================================================================

def _parse_pdf(path: Path) -> list[ParsedChunk]:
    """Parse PDF — uses pdfplumber if available, falls back to PyPDF2."""
    try:
        return _parse_pdf_pdfplumber(path)
    except ImportError:
        pass

    try:
        return _parse_pdf_pypdf(path)
    except ImportError:
        pass

    logger.warning(
        "[parsers] No PDF library available. "
        "Install pdfplumber or PyPDF2."
    )
    return []


def _parse_pdf_pdfplumber(path: Path) -> list[ParsedChunk]:
    """Parse PDF using pdfplumber."""
    import pdfplumber

    chunks = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                chunks.append(ParsedChunk(
                    text=text.strip(),
                    source_file=str(path),
                    chunk_index=i,
                    format="pdf",
                    metadata={"page": i + 1, "total_pages": len(pdf.pages)},
                ))
    return chunks


def _parse_pdf_pypdf(path: Path) -> list[ParsedChunk]:
    """Parse PDF using PyPDF2."""
    from PyPDF2 import PdfReader

    chunks = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append(ParsedChunk(
                text=text.strip(),
                source_file=str(path),
                chunk_index=i,
                format="pdf",
                metadata={"page": i + 1, "total_pages": len(reader.pages)},
            ))
    return chunks


# =========================================================================
# Markdown parser
# =========================================================================

def _parse_markdown(path: Path) -> list[ParsedChunk]:
    """Parse Markdown with section-aware chunking."""
    text = path.read_text(encoding="utf-8")
    sections = _split_markdown_sections(text)

    chunks = []
    for i, (heading, body) in enumerate(sections):
        content = body.strip()
        if not content:
            continue

        level = 0
        if heading:
            level = len(heading) - len(heading.lstrip("#"))
            content = f"{heading}\n{content}"

        chunks.append(ParsedChunk(
            text=content,
            source_file=str(path),
            chunk_index=i,
            format="md",
            metadata={
                "section": heading or "(preamble)",
                "heading_level": level,
            },
        ))
    return chunks


def _split_markdown_sections(text: str) -> list[tuple]:
    """Split markdown into (heading, body) tuples."""
    lines = text.split("\n")
    sections = []
    current_heading = None
    current_body = []

    for line in lines:
        if re.match(r"^#{1,6}\s+", line):
            # New section
            if current_body or current_heading:
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)

    # Last section
    if current_body or current_heading:
        sections.append((current_heading, "\n".join(current_body)))

    return sections


# =========================================================================
# CSV parser
# =========================================================================

def _parse_csv(path: Path) -> list[ParsedChunk]:
    """Parse CSV/TSV — one chunk per row with header context."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames or []

        chunks = []
        for i, row in enumerate(reader):
            # Build readable text from row
            parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
            if not parts:
                continue

            chunks.append(ParsedChunk(
                text="; ".join(parts),
                source_file=str(path),
                chunk_index=i,
                format="csv",
                metadata={
                    "row": i + 1,
                    "headers": headers,
                },
            ))
    return chunks


# =========================================================================
# Text parser
# =========================================================================

def _parse_text(path: Path) -> list[ParsedChunk]:
    """Parse plain text — split by double newlines (paragraphs)."""
    text = path.read_text(encoding="utf-8")
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para or len(para) < 10:
            continue

        chunks.append(ParsedChunk(
            text=para,
            source_file=str(path),
            chunk_index=i,
            format="txt",
            metadata={"paragraph": i + 1},
        ))
    return chunks


# =========================================================================
# Helpers
# =========================================================================

def _has_content(item) -> bool:
    """Check if a JSON item has meaningful content."""
    if isinstance(item, str):
        return len(item.strip()) > 10
    if isinstance(item, dict):
        return bool(item)
    if isinstance(item, list):
        return bool(item)
    return False
