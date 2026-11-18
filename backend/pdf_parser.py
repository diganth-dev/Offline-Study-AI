"""PDF parsing and text extraction utilities for Offline Study AI.

Reuses PyMuPDF extraction and the project's chunking functions to produce
structured chunks with page numbers and sequential chunk numbers.
"""

from typing import Any
from fastapi import HTTPException

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    from backend.chunking import (
        chunk_pdf_pages,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
    )
except ImportError:
    from chunking import (
        chunk_pdf_pages,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
    )


def extract_and_chunk_pdf(
    content: bytes,
    filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[int, list[dict[str, Any]]]:
    """Extracts text page-by-page from PDF bytes and divides it into chunks.

    Args:
        content: Raw bytes of the uploaded PDF file.
        filename: Original filename of the PDF for validation.
        chunk_size: Maximum characters per chunk (default: 1200).
        chunk_overlap: Overlapping characters between consecutive chunks
          (default: 200).

    Returns:
        A tuple of (total_pages, chunks), where chunks is a list of dicts:
            [{"chunk_number": int, "page_number": int, "text": str}, ...]

    Raises:
        HTTPException: If file is invalid, encrypted, corrupt, has 0 pages, or
        has no text.
    """
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF file is empty."
        )

    try:
        pdf = fitz.open(stream=content, filetype="pdf")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read this PDF."
        )

    try:
        if pdf.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported."
            )

        total_pages = len(pdf)
        if total_pages == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no pages."
            )

        pages_text: list[tuple[int, str]] = []
        for page_idx in range(total_pages):
            page = pdf[page_idx]
            text = page.get_text()
            pages_text.append((page_idx + 1, text))

        chunks = chunk_pdf_pages(
            pages_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no extractable text."
            )

        return total_pages, chunks

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to process this PDF."
        )
    finally:
        pdf.close()
