"""Document Summarization Service for Offline Study AI.

Processes indexed ChromaDB chunks in manageable batches (map-reduce style)
to generate complete, student-friendly document summaries using the local
Ollama model (qwen2.5:1.5b), avoiding single-prompt context overflow.
"""

import re
from typing import Any, Optional
from fastapi import HTTPException

try:
    from backend.vector_store import (
        get_document_chunks,
        get_indexed_documents,
        DEFAULT_COLLECTION_NAME,
    )
    from backend.ollama_client import (
        check_ollama_availability,
        generate_rag_answer,
        DEFAULT_OLLAMA_MODEL,
        OLLAMA_BASE_URL,
    )
except ImportError:
    from vector_store import (
        get_document_chunks,
        get_indexed_documents,
        DEFAULT_COLLECTION_NAME,
    )
    from ollama_client import (
        check_ollama_availability,
        generate_rag_answer,
        DEFAULT_OLLAMA_MODEL,
        OLLAMA_BASE_URL,
    )

# Maximum chunks per batch for long PDFs to respect 8GB RAM and CPU limits
SUMMARY_BATCH_SIZE = 3


def extract_chapters_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identifies chapter boundaries across sequential document chunks.

    Returns a list of chapter groups if 2 or more chapters are identified:
    [
        {
            "chapter_title": str,
            "page_start": int,
            "page_end": int,
            "page_range": str,
            "chunks": list[dict]
        },
        ...
    ]
    """
    chapter_pattern = re.compile(
        r"(?:^|\n)\s*(chapter\s+[0-9ivxlcdm]+(?:[:\.\s\-]+[^\n\r]+)?|section\s+[0-9ivxlcdm]+(?:[:\.\s\-]+[^\n\r]+)?)",
        re.IGNORECASE,
    )

    detected_chapters: list[dict[str, Any]] = []
    current_chapter: Optional[dict[str, Any]] = None

    for chunk in chunks:
        text = chunk.get("text", "")
        match = chapter_pattern.search(text)
        if match:
            title = match.group(1).strip()
            # Clean trailing punctuation
            title = re.sub(r"[\s\:\-\.]+$", "", title).strip()

            if current_chapter is not None:
                current_chapter["page_end"] = current_chapter["chunks"][-1]["page_number"]
                start_p = current_chapter["page_start"]
                end_p = current_chapter["page_end"]
                current_chapter["page_range"] = f"Page {start_p}" if start_p == end_p else f"Pages {start_p}–{end_p}"
                detected_chapters.append(current_chapter)

            current_chapter = {
                "chapter_title": title,
                "page_start": chunk["page_number"],
                "page_end": chunk["page_number"],
                "chunks": [chunk],
            }
        else:
            if current_chapter is not None:
                current_chapter["chunks"].append(chunk)

    if current_chapter is not None:
        current_chapter["page_end"] = current_chapter["chunks"][-1]["page_number"]
        start_p = current_chapter["page_start"]
        end_p = current_chapter["page_end"]
        current_chapter["page_range"] = f"Page {start_p}" if start_p == end_p else f"Pages {start_p}–{end_p}"
        detected_chapters.append(current_chapter)

    # Only return chapters if at least 2 distinct chapter boundaries were identified
    return detected_chapters if len(detected_chapters) >= 2 else []


def parse_summary_sections(text: str) -> dict[str, Any]:
    """Extracts structured sections from the LLM summary output.

    Handles markdown headers:
    - Document Overview
    - Main Topics
    - Important Points
    - Key Takeaways
    """
    sections: dict[str, Any] = {
        "overview": "",
        "main_topics": [],
        "important_points": [],
        "key_takeaways": [],
    }

    # 1. Document Overview
    m_over = re.search(
        r"###\s*Document Overview\s*\n(.*?)(?=\n###|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m_over:
        sections["overview"] = m_over.group(1).strip()

    # 2. Main Topics
    m_top = re.search(
        r"###\s*Main Topics\s*\n(.*?)(?=\n###|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m_top:
        raw_top = m_top.group(1).strip()
        lines = [
            line.strip().lstrip("-*• ").strip()
            for line in raw_top.split("\n")
            if line.strip() and line.strip()[0] in ("-", "*", "•", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        ]
        sections["main_topics"] = [l for l in lines if l]

    # 3. Important Points
    m_pts = re.search(
        r"###\s*Important Points\s*\n(.*?)(?=\n###|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m_pts:
        raw_pts = m_pts.group(1).strip()
        lines = [
            line.strip().lstrip("-*• ").strip()
            for line in raw_pts.split("\n")
            if line.strip() and line.strip()[0] in ("-", "*", "•", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        ]
        sections["important_points"] = [l for l in lines if l]

    # 4. Key Takeaways
    m_tak = re.search(
        r"###\s*Key Takeaways\s*\n(.*?)(?=\n###|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m_tak:
        raw_tak = m_tak.group(1).strip()
        lines = [
            line.strip().lstrip("-*• ").strip()
            for line in raw_tak.split("\n")
            if line.strip() and line.strip()[0] in ("-", "*", "•", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        ]
        sections["key_takeaways"] = [l for l in lines if l]

    # Fallbacks in case LLM deviated from exact markdown header syntax
    if not sections["overview"]:
        # Use first non-empty paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and not p.strip().startswith("#")]
        sections["overview"] = paragraphs[0] if paragraphs else text.strip()

    if not sections["important_points"]:
        bullet_lines = [
            line.strip().lstrip("-*• ").strip()
            for line in text.split("\n")
            if line.strip().startswith(("-", "*", "•"))
        ]
        if bullet_lines:
            sections["important_points"] = bullet_lines[:5]

    if not sections["main_topics"]:
        bullet_lines = [
            line.strip().lstrip("-*• ").strip()
            for line in text.split("\n")
            if line.strip().startswith(("-", "*", "•"))
        ]
        if bullet_lines:
            sections["main_topics"] = bullet_lines[:4]
        elif sections["important_points"]:
            sections["main_topics"] = sections["important_points"][:3]

    if not sections["key_takeaways"]:
        if sections["important_points"] and len(sections["important_points"]) > 1:
            sections["key_takeaways"] = [sections["important_points"][-1]]

    return sections


def summarize_document(
    document_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    include_chapters: bool = True,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Generates a complete, structured summary of a specific indexed document.

    Workflow:
    1. Validates document existence and ownership in ChromaDB for the user.
    2. Retrieves all chunks sequentially for the document.
    3. Handles empty or non-extractable documents gracefully.
    4. For long PDFs, batches chunks sequentially to avoid single-prompt context overflow.
    5. Detects chapter boundaries if identifiable and generates chapter breakdowns.
    6. Formats structured output: Overview, Main Topics, Important Points, Key Takeaways.
    7. Tracks exact source page numbers.

    Returns:
        Dict matching SummarizeResponse schema.
    """
    clean_doc_id = document_id.strip()
    if not clean_doc_id:
        raise HTTPException(status_code=400, detail="Document ID cannot be empty.")

    # 1. Fetch document metadata summary (scoped strictly to user_id)
    indexed_docs = get_indexed_documents(
        collection_name=collection_name,
        persist_directory=persist_directory,
        user_id=user_id,
    )
    doc_meta = next((d for d in indexed_docs if d.get("document_id") == clean_doc_id), None)
    if not doc_meta:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{clean_doc_id}' was not found in your study library.",
        )

    filename = doc_meta["filename"]

    # 2. Retrieve all chunks belonging to this document (scoped strictly to user_id)
    all_chunks = get_document_chunks(
        document_id=clean_doc_id,
        collection_name=collection_name,
        persist_directory=persist_directory,
        user_id=user_id,
    )

    if not all_chunks or not any(c.get("text", "").strip() for c in all_chunks):
        raise HTTPException(
            status_code=400,
            detail=f"The document '{filename}' contains no extractable text (it may be a scanned image or empty PDF).",
        )

    # 3. Verify Ollama service and model
    check_ollama_availability(base_url=ollama_base_url, model=ollama_model)

    source_pages = sorted(list(set(c["page_number"] for c in all_chunks)))

    # 4. Batch processing for long documents (Map-Reduce)
    intermediate_summaries: list[str] = []
    if len(all_chunks) <= 4:
        # Small document: can process all chunks directly
        combined_text = "\n\n".join(
            f"[Page {c['page_number']}]\n{c['text']}" for c in all_chunks
        )
    else:
        # Long document: process in batches of SUMMARY_BATCH_SIZE chunks
        batches = [
            all_chunks[i : i + SUMMARY_BATCH_SIZE]
            for i in range(0, len(all_chunks), SUMMARY_BATCH_SIZE)
        ]
        for batch_idx, batch in enumerate(batches, start=1):
            batch_pages = f"Pages {batch[0]['page_number']}–{batch[-1]['page_number']}"
            batch_text = "\n\n".join(
                f"[Page {c['page_number']}]\n{c['text']}" for c in batch
            )
            batch_prompt = (
                f"Document: {filename} ({batch_pages})\n"
                f"Context excerpts:\n"
                f"{batch_text}\n\n"
                f"Task: Factually summarize the main concepts, definitions, and key points from these pages in 2-3 concise paragraphs.\n"
                f"Section Summary:"
            )
            batch_summary = generate_rag_answer(
                prompt=batch_prompt,
                system_prompt="You are a precise study assistant. Summarize the provided document pages factually and concisely.",
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=0.1,
                max_tokens=180,
            )
            intermediate_summaries.append(f"### Section {batch_idx} ({batch_pages})\n{batch_summary.strip()}")

        combined_text = "\n\n".join(intermediate_summaries)

    # 5. Final synthesis prompt into the required sections
    system_prompt = (
        "You are Offline Study AI, an expert, factual personal learning assistant for college students. "
        f"You are summarizing the study document: '{filename}'. "
        "Rules:\n"
        "1. Base your summary STRICTLY and ONLY on the provided context excerpts from this document.\n"
        "2. Do NOT invent, assume, or bring in outside facts.\n"
        "3. Provide clear, student-friendly, and educational explanations.\n"
        "4. Format your output with the EXACT markdown headers below:\n\n"
        "### Document Overview\n"
        "[1-2 structured, clear paragraphs explaining the document purpose, subject, and scope]\n\n"
        "### Main Topics\n"
        "- **Topic Name**: Brief description\n"
        "- **Topic Name**: Brief description\n\n"
        "### Important Points\n"
        "- Core concept, definition, or key principle\n"
        "- Core concept, definition, or key principle\n\n"
        "### Key Takeaways\n"
        "- Essential takeaway students need to remember\n"
        "- Essential takeaway students need to remember"
    )

    synthesis_prompt = (
        f"Document: {filename}\n"
        f"Document Content Excerpts:\n"
        f"{combined_text}\n\n"
        f"Task: Produce a complete, student-friendly study summary following the exact required sections.\n"
        f"Summary:"
    )

    full_summary = generate_rag_answer(
        prompt=synthesis_prompt,
        system_prompt=system_prompt,
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=0.1,
        max_tokens=500,
    )
    if full_summary.startswith('"') and full_summary.endswith('"') and len(full_summary) > 2:
        full_summary = full_summary[1:-1].strip()

    parsed = parse_summary_sections(full_summary)

    # 6. Chapter-wise Summaries (when chapter boundaries are identifiable)
    chapter_summaries: list[dict[str, str]] = []
    if include_chapters:
        detected_chapters = extract_chapters_from_chunks(all_chunks)
        for chap in detected_chapters:
            chap_title = chap["chapter_title"]
            chap_range = chap["page_range"]
            chap_text = "\n\n".join(c["text"] for c in chap["chunks"])
            chap_prompt = (
                f"Chapter: {chap_title} ({chap_range})\n"
                f"Text:\n{chap_text}\n\n"
                f"Task: Provide a concise 2-sentence summary of this chapter's primary focus and conclusion.\n"
                f"Chapter Summary:"
            )
            chap_sum = generate_rag_answer(
                prompt=chap_prompt,
                system_prompt="You are a precise study assistant. Summarize the chapter concisely.",
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=0.1,
                max_tokens=120,
            )
            chapter_summaries.append({
                "chapter_title": chap_title,
                "page_range": chap_range,
                "summary": chap_sum.strip(),
            })

    return {
        "document_id": clean_doc_id,
        "filename": filename,
        "total_pages": doc_meta.get("total_pages", len(source_pages)),
        "total_chunks": len(all_chunks),
        "overview": parsed["overview"],
        "main_topics": parsed["main_topics"],
        "important_points": parsed["important_points"],
        "key_takeaways": parsed["key_takeaways"],
        "chapters": chapter_summaries,
        "full_summary": full_summary,
        "source_pages": source_pages,
    }
