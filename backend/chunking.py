"""Text chunking utilities for Offline Study AI.

This module provides functions to split text into overlapping chunks
while preserving document metadata such as page numbers and sequential chunk numbers.
"""

from typing import Any

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Divides text into smaller overlapping chunks.

    Args:
        text: The string to be divided into chunks.
        chunk_size: Maximum number of characters per chunk (default: 1200).
        chunk_overlap: Overlapping character count between consecutive chunks
          (default: 200).

    Returns:
        A list of text chunk strings.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >=
        chunk_size.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    text_len = len(text)
    start = 0

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks


def chunk_pdf_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Chunks text page-by-page while preserving 1-indexed page numbers

    and assigning sequential chunk numbers across the document.

    Args:
        pages: List of (page_number, page_text) tuples where page_number is
          1-indexed.
        chunk_size: Maximum characters per chunk (default: 1200).
        chunk_overlap: Overlapping character count between consecutive chunks
          (default: 200).

    Returns:
        A list of dictionaries with structure:
            {
                "chunk_number": int,
                "page_number": int,
                "text": str
            }
    """
    chunks: list[dict[str, Any]] = []
    chunk_number = 1

    for page_number, page_text in pages:
        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk_content in page_chunks:
            chunks.append({
                "chunk_number": chunk_number,
                "page_number": page_number,
                "text": chunk_content,
            })
            chunk_number += 1

    return chunks
