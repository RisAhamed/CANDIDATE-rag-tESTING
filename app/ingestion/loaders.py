"""Document loaders for various file types."""

import io
import re
from pathlib import Path

import pdfplumber
from docx import Document

from app.core.logging import logger


def load_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF."""
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise
    return "\n".join(text_parts)


def load_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX."""
    doc = Document(io.BytesIO(file_bytes))
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                text_parts.append(row_text)
    return "\n".join(text_parts)


def load_txt(file_bytes: bytes) -> str:
    """Load plain text."""
    return file_bytes.decode("utf-8", errors="replace")


def load_srt(file_bytes: bytes) -> str:
    """Load SRT subtitle file, preserving timestamps."""
    content = file_bytes.decode("utf-8", errors="replace")
    lines = content.split("\n")
    result = []
    current_block = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_block:
                result.append(" ".join(current_block))
                current_block = []
            continue
        # Skip sequence numbers
        if line.isdigit():
            continue
        # Timestamp line
        if "-->" in line:
            current_block.append(f"[{line}]")
            continue
        current_block.append(line)

    if current_block:
        result.append(" ".join(current_block))

    return "\n".join(result)


def load_vtt(file_bytes: bytes) -> str:
    """Load VTT subtitle file, preserving timestamps."""
    content = file_bytes.decode("utf-8", errors="replace")
    lines = content.split("\n")
    result = []
    current_block = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            if current_block:
                result.append(" ".join(current_block))
                current_block = []
            continue
        if "-->" in line:
            current_block.append(f"[{line}]")
            continue
        current_block.append(line)

    if current_block:
        result.append(" ".join(current_block))

    return "\n".join(result)


def detect_file_type(filename: str) -> str:
    """Detect file type from extension."""
    ext = Path(filename).suffix.lower()
    type_map = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "txt",
        ".md": "md",
        ".srt": "srt",
        ".vtt": "vtt",
    }
    return type_map.get(ext, "unknown")


def load_document(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Load a document and return (text, file_type).

    Args:
        file_bytes: Raw file content
        filename: Original filename

    Returns:
        Tuple of (extracted_text, file_type)
    """
    file_type = detect_file_type(filename)
    logger.info(f"Loading document: {filename} (type: {file_type})")

    loaders = {
        "pdf": load_pdf,
        "docx": load_docx,
        "txt": load_txt,
        "md": load_txt,
        "srt": load_srt,
        "vtt": load_vtt,
    }

    loader = loaders.get(file_type)
    if not loader:
        raise ValueError(f"Unsupported file type: {file_type}")

    text = loader(file_bytes)
    logger.info(f"Extracted {len(text)} characters from {filename}")
    return text, file_type
