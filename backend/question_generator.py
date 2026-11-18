"""Important Questions Generator Service for Offline Study AI.

Generates exam-oriented study questions covering the complete selected document,
categorized into:
1. Short-answer questions (definitions, basic concepts, 2-5 marks)
2. Long-answer questions (comprehensive explanations, mechanisms, comparisons, 10-15 marks)
3. Important conceptual questions (underlying principles, theory, critical thinking)

Also provides individual on-demand exam-friendly answer generation grounded strictly in the PDF.
Uses local Ollama (qwen2.5:1.5b) and ChromaDB document storage.
"""

import re
from typing import Any, Optional
from fastapi import HTTPException

try:
    from backend.vector_store import (
        get_document_chunks,
        get_indexed_documents,
        search_documents,
        select_representative_chunks,
        DEFAULT_COLLECTION_NAME,
    )
    from backend.ollama_client import (
        check_ollama_availability,
        generate_rag_answer,
        DEFAULT_OLLAMA_MODEL,
        OLLAMA_BASE_URL,
    )
    from backend.metadata_service import (
        is_comparison_question,
        extract_comparison_topics,
        format_comparison_answer,
    )
except ImportError:
    from vector_store import (
        get_document_chunks,
        get_indexed_documents,
        search_documents,
        select_representative_chunks,
        DEFAULT_COLLECTION_NAME,
    )
    from ollama_client import (
        check_ollama_availability,
        generate_rag_answer,
        DEFAULT_OLLAMA_MODEL,
        OLLAMA_BASE_URL,
    )
    from metadata_service import (
        is_comparison_question,
        extract_comparison_topics,
        format_comparison_answer,
    )


def is_metadata_line(line: str) -> bool:
    """Checks if a line is a citation, source note, or metadata rather than a question."""
    cleaned = re.sub(r"^(?:\d+[\.\)\-]|[-*•])\s*", "", line).strip()
    if not cleaned:
        return True
    # Matches "Source: ...", "Ref: ...", "Page 2", "(Page 2)", etc.
    if re.match(
        r"^(?:sources?|references?|refs?|pages?|citations?|answers?|ans|notes?)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    if re.match(r"^[\(\[\{]\s*(?:pages?|p\.?)\s*\d+.*[\)\]\}]$", cleaned, re.IGNORECASE):
        return True
    return False


def is_valid_question_line(line: str) -> bool:
    """Checks if a line contains a legitimate question rather than notes or placeholders."""
    cleaned = re.sub(r"^(?:\d+[\.\)\-]|[-*•])\s*", "", line).strip()
    if not cleaned or len(cleaned) < 12:
        return False
    if len(cleaned.split()) < 3:
        return False
    if is_metadata_line(cleaned):
        return False
    if re.search(r"\[?\bquestion\s+text\b\]?", cleaned, re.IGNORECASE):
        return False
    return True


def extract_section_questions(sec_body: str) -> list[str]:
    """Extracts valid question lines from a section body, filtering out metadata/source lines and attaching citations."""
    lines = [l.strip() for l in sec_body.split("\n") if l.strip()]
    q_lines: list[str] = []

    for l in lines:
        stripped = re.sub(r"^(?:\d+[\.\)\-]|[-*•])\s*", "", l).strip()
        # If it's a metadata/source line (e.g. "- Source: Page 3" or "Page 2")
        if is_metadata_line(l) or is_metadata_line(stripped):
            # If the previous question does not yet have an explicit page reference, attach it
            if q_lines and not re.search(r"[\(\[\{]\s*pages?\s*\d+", q_lines[-1], re.IGNORECASE):
                page_match = re.search(r"\bpages?\s*(\d+(?:\s*[-–]\s*\d+)?)", stripped, re.IGNORECASE)
                if page_match:
                    q_lines[-1] = f"{q_lines[-1]} (Page {page_match.group(1)})"
            continue

        is_bulleted = bool(re.match(r"^(?:\d+[\.\)\-]|[-*•])\s+", l))
        if is_bulleted or "?" in l or len(stripped) > 20:
            if is_valid_question_line(l):
                q_lines.append(stripped)
            elif q_lines and len(stripped) > 5 and not is_metadata_line(stripped):
                # Continuation line of previous question
                q_lines[-1] = f"{q_lines[-1]} {stripped}"

    return q_lines


def extract_page_reference(text: str, fallback_chunks: list[dict[str, Any]]) -> tuple[str, list[int], str]:
    """Extracts page reference from question text or identifies it from matching document chunks.

    Returns:
        tuple of (cleaned_question, list_of_page_numbers, formatted_page_reference)
    """
    cleaned = text.strip()

    # Pattern 1: (Page 2) or [Page 2] or (Pages 1-3) or [Pages 1–3] or (p. 2)
    page_match = re.search(
        r"[\(\[\{]\s*(?:pages?|p\.?)\s*(\d+(?:\s*[-–]\s*\d+)?)\s*[\)\]\}]",
        cleaned,
        re.IGNORECASE,
    )
    if page_match:
        page_str = page_match.group(1).replace(" ", "")
        # Remove the page marker from the question string
        cleaned = (cleaned[:page_match.start()] + cleaned[page_match.end():]).strip()
        cleaned = re.sub(r"[\s\:\-\.]+$", "", cleaned).strip()
        if not cleaned.endswith(("?", ".", "!")):
            cleaned += "?"

        if "-" in page_str or "–" in page_str:
            parts = re.split(r"[-–]", page_str)
            try:
                p_start, p_end = int(parts[0]), int(parts[1])
                pages = list(range(p_start, p_end + 1))
                return cleaned, pages, f"Pages {p_start}–{p_end}"
            except (ValueError, IndexError):
                pass
        else:
            try:
                p_num = int(page_str)
                return cleaned, [p_num], f"Page {p_num}"
            except ValueError:
                pass

    # Pattern 2: Trailing " - Page 2" or " - Page 1-2"
    trailing_match = re.search(
        r"[-–—:]\s*(?:pages?|p\.?)\s*(\d+(?:\s*[-–]\s*\d+)?)\s*$",
        cleaned,
        re.IGNORECASE,
    )
    if trailing_match:
        page_str = trailing_match.group(1).replace(" ", "")
        cleaned = cleaned[:trailing_match.start()].strip()
        cleaned = re.sub(r"[\s\:\-\.]+$", "", cleaned).strip()
        if not cleaned.endswith(("?", ".", "!")):
            cleaned += "?"

        if "-" in page_str or "–" in page_str:
            parts = re.split(r"[-–]", page_str)
            try:
                p_start, p_end = int(parts[0]), int(parts[1])
                pages = list(range(p_start, p_end + 1))
                return cleaned, pages, f"Pages {p_start}–{p_end}"
            except (ValueError, IndexError):
                pass
        else:
            try:
                p_num = int(page_str)
                return cleaned, [p_num], f"Page {p_num}"
            except ValueError:
                pass

    # Fallback: Locate page reference by content token overlap in fallback_chunks
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", cleaned)]
    best_chunk: Optional[dict[str, Any]] = None
    best_overlap = 0

    for chunk in fallback_chunks:
        chunk_text = chunk.get("text", "").lower()
        overlap = sum(1 for w in words if w in chunk_text)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk

    if best_chunk and "page_number" in best_chunk:
        p_num = int(best_chunk["page_number"])
        return cleaned, [p_num], f"Page {p_num}"

    # Default fallback to page 1
    return cleaned, [1], "Page 1"


def parse_questions_output(
    raw_text: str,
    all_chunks: list[dict[str, Any]],
    total_pages: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parses raw LLM markdown output into structured categories and question items.

    Returns:
        tuple of (categories_list, flat_questions_list)
    """
    text = raw_text.strip()
    if text.startswith('"') and text.endswith('"') and len(text) > 2:
        text = text[1:-1].strip()

    # Define sections
    short_raw: list[str] = []
    long_raw: list[str] = []
    conceptual_raw: list[str] = []

    # Section regex boundaries
    short_pattern = re.compile(
        r"(?:###|\*\*)\s*(?:short[- ]answer\s*(?:questions?)?)(?:###|\*\*|:|\n)",
        re.IGNORECASE,
    )
    long_pattern = re.compile(
        r"(?:###|\*\*)\s*(?:long[- ]answer\s*(?:questions?)?)(?:###|\*\*|:|\n)",
        re.IGNORECASE,
    )
    conceptual_pattern = re.compile(
        r"(?:###|\*\*)\s*(?:(?:important\s+)?conceptual\s*(?:questions?)?)(?:###|\*\*|:|\n)",
        re.IGNORECASE,
    )

    short_match = short_pattern.search(text)
    long_match = long_pattern.search(text)
    conceptual_match = conceptual_pattern.search(text)

    # If headers are found, slice accordingly
    sections_found = [
        ("short", short_match.start() if short_match else -1, short_match.end() if short_match else -1),
        ("long", long_match.start() if long_match else -1, long_match.end() if long_match else -1),
        ("conceptual", conceptual_match.start() if conceptual_match else -1, conceptual_match.end() if conceptual_match else -1),
    ]
    # Filter only found sections and sort by their position in text
    valid_sections = sorted([s for s in sections_found if s[1] != -1], key=lambda x: x[1])

    if valid_sections:
        for idx, (sec_name, start_pos, end_pos) in enumerate(valid_sections):
            next_start = valid_sections[idx + 1][1] if idx + 1 < len(valid_sections) else len(text)
            sec_body = text[end_pos:next_start].strip()
            q_lines = extract_section_questions(sec_body)

            if sec_name == "short":
                short_raw.extend(q_lines)
            elif sec_name == "long":
                long_raw.extend(q_lines)
            elif sec_name == "conceptual":
                conceptual_raw.extend(q_lines)

    # If parsing found few or no questions, fall back to parsing all numbered questions across entire text
    total_found = len(short_raw) + len(long_raw) + len(conceptual_raw)
    if total_found < 6:
        all_lines = [l.strip() for l in text.split("\n") if l.strip()]
        generic_questions = []
        for l in all_lines:
            if is_metadata_line(l):
                continue
            m = re.match(r"^(?:\d+[\.\)\-]|[-*•])\s*(.+)$", l)
            if m:
                cand = m.group(1).strip()
                if is_valid_question_line(cand):
                    generic_questions.append(cand)
            elif is_valid_question_line(l):
                generic_questions.append(l)

        if generic_questions:
            # Distribute into 3 categories
            k = len(generic_questions)
            s_end = max(1, k // 3)
            l_end = max(s_end + 1, (2 * k) // 3)
            short_raw = generic_questions[:s_end]
            long_raw = generic_questions[s_end:l_end]
            conceptual_raw = generic_questions[l_end:]

    # Ensure at least some questions exist in each list if text allowed
    if not short_raw and not long_raw and not conceptual_raw:
        # Emergency fallback question from first chunk
        if all_chunks:
            short_raw = [f"What is the core subject discussed in {all_chunks[0].get('filename', 'this document')}?"]

    # Build final Question items with sequential global IDs
    global_id = 1
    short_items: list[dict[str, Any]] = []
    for q in short_raw:
        cleaned_q, pages, page_ref = extract_page_reference(q, all_chunks)
        short_items.append({
            "id": global_id,
            "question": cleaned_q,
            "question_type": "short",
            "page_reference": page_ref,
            "page_numbers": pages,
            "answer": None,
        })
        global_id += 1

    long_items: list[dict[str, Any]] = []
    for q in long_raw:
        cleaned_q, pages, page_ref = extract_page_reference(q, all_chunks)
        long_items.append({
            "id": global_id,
            "question": cleaned_q,
            "question_type": "long",
            "page_reference": page_ref,
            "page_numbers": pages,
            "answer": None,
        })
        global_id += 1

    conceptual_items: list[dict[str, Any]] = []
    for q in conceptual_raw:
        cleaned_q, pages, page_ref = extract_page_reference(q, all_chunks)
        conceptual_items.append({
            "id": global_id,
            "question": cleaned_q,
            "question_type": "conceptual",
            "page_reference": page_ref,
            "page_numbers": pages,
            "answer": None,
        })
        global_id += 1

    categories = [
        {
            "category": "Short-Answer Questions",
            "category_id": "short",
            "description": "Definitions, concise facts, and fundamental concept checks (2–5 marks).",
            "questions": short_items,
        },
        {
            "category": "Long-Answer Questions",
            "category_id": "long",
            "description": "Comprehensive explanations, structured processes, and analytical comparisons (10–15 marks).",
            "questions": long_items,
        },
        {
            "category": "Important Conceptual Questions",
            "category_id": "conceptual",
            "description": "Core principles, underlying theoretical mechanisms, and critical thinking questions.",
            "questions": conceptual_items,
        },
    ]

    flat_questions = short_items + long_items + conceptual_items
    return categories, flat_questions


def generate_important_questions(
    document_id: str,
    count: int = 12,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
) -> dict[str, Any]:
    """Generates 10-15 important study questions based on the complete selected document.

    Workflow:
    1. Validates document existence in ChromaDB.
    2. Retrieves all indexed chunks sequentially for the document.
    3. Handles empty or non-extractable documents gracefully.
    4. Gathers full-document context across all pages (using representative sampling for long documents).
    5. Prompts local Ollama (qwen2.5:1.5b) to formulate structured questions divided into
       Short-answer, Long-answer, and Conceptual categories with page citations.
    6. Parses and validates the questions into a clean, numbered format.
    7. Preserves document isolation and source references.

    Args:
        document_id: Document identifier in ChromaDB.
        count: Desired total question count (default: 12).
        collection_name: ChromaDB collection name.
        persist_directory: Custom ChromaDB directory if applicable.
        ollama_model: Ollama model name.
        ollama_base_url: Ollama daemon URL.

    Returns:
        Dict matching ImportantQuestionsResponse schema.
    """
    clean_doc_id = document_id.strip()
    if not clean_doc_id:
        raise HTTPException(status_code=400, detail="Document ID cannot be empty.")

    # 1. Fetch document metadata summary
    indexed_docs = get_indexed_documents(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    doc_meta = next((d for d in indexed_docs if d.get("document_id") == clean_doc_id), None)
    if not doc_meta:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{clean_doc_id}' was not found in the study library.",
        )

    filename = doc_meta["filename"]

    # 2. Retrieve all chunks belonging to this document
    all_chunks = get_document_chunks(
        document_id=clean_doc_id,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    if not all_chunks or not any(c.get("text", "").strip() for c in all_chunks):
        raise HTTPException(
            status_code=400,
            detail=f"The document '{filename}' contains no extractable text (it may be a scanned image or empty PDF).",
        )

    # 3. Verify Ollama availability
    check_ollama_availability(base_url=ollama_base_url, model=ollama_model)

    # 4. Prepare context covering the complete document
    # If the document has <= 8 chunks, include all of them.
    # If more than 8 chunks, sample representative chunks evenly across all pages.
    if len(all_chunks) <= 8:
        selected_chunks = all_chunks
    else:
        selected_chunks = select_representative_chunks(all_chunks, max_chunks=8)

    context_blocks = []
    for c in selected_chunks:
        pn = c.get("page_number", 1)
        txt = c.get("text", "").strip()
        context_blocks.append(f"[Source Page {pn}]\n{txt}")

    context_str = "\n\n".join(context_blocks)
    total_pages = doc_meta.get("total_pages", max((c.get("page_number", 1) for c in all_chunks), default=1))

    # 5. Compose system prompt and generation prompt
    system_prompt = (
        "You are Offline Study AI, an expert examination tutor and academic question paper setter. "
        f"You are formulating important study and exam questions for students based STRICTLY and ONLY "
        f"on the provided excerpts from '{filename}'.\n\n"
        "Rules:\n"
        "1. Ground every question strictly and exclusively in the provided document text. "
        "Do NOT invent, assume, or ask about topics not present in the excerpts.\n"
        "2. Formulate realistic, specific exam questions divided into three categories:\n"
        "   - Short-Answer Questions: definitions, direct concepts, terminology (2-5 marks)\n"
        "   - Long-Answer Questions: comprehensive explanations, detailed processes, comparisons (10-15 marks)\n"
        "   - Conceptual Questions: core principles, theoretical foundations, 'why' and 'how' theory\n"
        "3. Generate around 10 to 15 real, distinct questions in total (4-5 short-answer, 3-5 long-answer, 3-5 conceptual).\n"
        "4. For EVERY question, write a complete, actual question and cite its source page number from the provided context (e.g. Page 1 or Page 2).\n"
        "5. Do NOT output placeholder text. Write actual questions derived from the text.\n"
        "6. Format your response strictly using the markdown section headers below:\n\n"
        "### Short-Answer Questions\n"
        "1. What is ...? (Page 1)\n"
        "2. Define ... (Page 2)\n\n"
        "### Long-Answer Questions\n"
        "1. Explain the mechanism of ... in detail. (Page 1)\n"
        "2. Compare and contrast ... and ... (Page 3)\n\n"
        "### Conceptual Questions\n"
        "1. Why does ... occur? (Page 2)\n"
        "2. What is the fundamental principle behind ...? (Page 3)"
    )

    prompt = (
        f"Document: {filename} (Total Pages: {total_pages})\n"
        f"Context excerpts across the document:\n"
        f"{context_str}\n\n"
        f"Task: Read the context excerpts and write approximately {count} real, specific exam questions "
        f"divided into Short-Answer, Long-Answer, and Conceptual questions with source page citations.\n"
        f"Exam Questions:"
    )

    raw_response = generate_rag_answer(
        prompt=prompt,
        system_prompt=system_prompt,
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=0.2,
        max_tokens=650,
    )

    categories, flat_questions = parse_questions_output(
        raw_text=raw_response,
        all_chunks=all_chunks,
        total_pages=total_pages,
    )

    return {
        "document_id": clean_doc_id,
        "filename": filename,
        "total_pages": total_pages,
        "total_questions": len(flat_questions),
        "categories": categories,
        "questions": flat_questions,
    }


def generate_question_answer(
    document_id: str,
    question: str,
    question_type: str = "short",
    page_numbers: Optional[list[int]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
) -> dict[str, Any]:
    """Generates a clear, exam-friendly answer for an individual study question.

    Args:
        document_id: Document identifier.
        question: The question text to answer.
        question_type: Category of question ('short', 'long', 'conceptual').
        page_numbers: Optional list of supporting page numbers.
        collection_name: ChromaDB collection.
        persist_directory: ChromaDB directory.
        ollama_model: Ollama model.
        ollama_base_url: Ollama URL.

    Returns:
        Dict matching QuestionAnswerResponse schema.
    """
    clean_doc_id = document_id.strip()
    clean_q = question.strip()
    if not clean_doc_id:
        raise HTTPException(status_code=400, detail="Document ID cannot be empty.")
    if not clean_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Fetch document metadata
    indexed_docs = get_indexed_documents(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    doc_meta = next((d for d in indexed_docs if d.get("document_id") == clean_doc_id), None)
    if not doc_meta:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{clean_doc_id}' was not found in the study library.",
        )

    filename = doc_meta["filename"]

    # 2. Verify Ollama
    check_ollama_availability(base_url=ollama_base_url, model=ollama_model)

    # 3. Retrieve relevant chunks scoped strictly to document_id
    is_comp = is_comparison_question(clean_q)
    search_k = 6 if is_comp else (4 if question_type == "long" else 3)
    chunks = search_documents(
        query=clean_q,
        top_k=search_k,
        collection_name=collection_name,
        persist_directory=persist_directory,
        document_id=clean_doc_id,
    )

    # For comparison questions, ensure chunks covering both topics are present
    if is_comp and chunks:
        comp_topics = extract_comparison_topics(clean_q)
        if len(comp_topics) == 2:
            t2 = comp_topics[1]
            has_t2 = any(t2.lower() in c.get("text", "").lower() for c in chunks)
            if not has_t2:
                extra_chunks = search_documents(
                    query=t2,
                    top_k=3,
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                    document_id=clean_doc_id,
                )
                seen_chunk_ids = {c.get("chunk_id") for c in chunks}
                for ec in extra_chunks:
                    if ec.get("chunk_id") not in seen_chunk_ids:
                        chunks.append(ec)
                        seen_chunk_ids.add(ec.get("chunk_id"))

    # Filter chunks: only retain relevant chunks close to the top match
    if len(chunks) > 1 and not is_comp:
        top_sim = chunks[0].get("similarity_score", 0.0)
        filtered = [c for c in chunks if c.get("similarity_score", 0.0) >= max(0.20, top_sim - 0.15)]
        if filtered:
            chunks = filtered

    # If page_numbers are provided, also ensure chunks on those pages are prioritized
    if page_numbers and len(chunks) < 3:
        doc_chunks = get_document_chunks(
            document_id=clean_doc_id,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        page_chunks = [c for c in doc_chunks if c.get("page_number") in page_numbers]
        for pc in page_chunks:
            if not any(c.get("chunk_id") == pc.get("chunk_id") for c in chunks):
                chunks.append(pc)

    if not chunks:
        return {
            "document_id": clean_doc_id,
            "question": clean_q,
            "question_type": question_type,
            "answer": f"The document '{filename}' does not contain sufficient information to answer this question.",
            "page_reference": "N/A",
            "sources": [],
        }

    # Format context blocks
    sorted_chunks = sorted(chunks, key=lambda c: (int(c.get("page_number", 1)), int(c.get("chunk_number", 1))))
    context_blocks = []
    seen_pages: set[int] = set()
    sources: list[dict[str, Any]] = []

    for c in sorted_chunks:
        fn = c.get("filename", filename)
        pn = int(c.get("page_number", 1))
        if pn not in seen_pages:
            seen_pages.add(pn)
            sources.append({"filename": fn, "page_number": pn})
        context_blocks.append(f"[Source Page {pn}]\n{c.get('text', '')}")

    context_str = "\n\n".join(context_blocks)
    
    if page_numbers:
        p_sorted = sorted(page_numbers)
        page_ref = f"Pages {p_sorted[0]}–{p_sorted[-1]}" if len(p_sorted) > 1 else f"Page {p_sorted[0]}"
    else:
        page_nums_list = sorted(list(seen_pages))
        page_ref = (
            f"Pages {page_nums_list[0]}–{page_nums_list[-1]}"
            if len(page_nums_list) > 1
            else f"Page {page_nums_list[0]}"
            if page_nums_list
            else "Page 1"
        )

    # Tailor system prompt based on comparison request or question_type
    if is_comp:
        type_instructions = (
            "The question asks to compare, differentiate, or contrast concepts. "
            "You MUST structure your response strictly in clean Markdown with:\n"
            "### Introduction\n"
            "A short 1 to 2 sentence student-friendly introduction defining both concepts based strictly on the document.\n\n"
            "### Comparison Table\n"
            "Display the key differences in a clean, readable Markdown table with explicit columns:\n"
            "| Basis of Comparison | Concept A | Concept B |\n"
            "| :--- | :--- | :--- |\n"
            "| [Parameter 1] | [Aspect for A] | [Aspect for B] |\n"
            "| [Parameter 2] | [Aspect for A] | [Aspect for B] |\n\n"
            "### Important Points\n"
            "Explain key points and differences in simple, student-friendly bullet points (- ).\n\n"
            "### Examples from Document\n"
            "Provide concrete examples cited directly from the PDF context excerpts for each concept.\n\n"
            "### Related Concepts\n"
            "(ONLY include this section if other related concepts are explicitly mentioned and supported in the provided context excerpts. If none are supported, omit this section completely).\n\n"
            "### Conclusion\n"
            "Provide a short 1 to 2 sentence exam-ready conclusion summarizing the key takeaway."
        )
        max_toks = 700
    elif question_type == "short":
        type_instructions = (
            "Provide a concise, simple, exam-friendly answer in 2 to 4 sentences or brief bullet points. "
            "Define terms directly and focus on core facts."
        )
        max_toks = 200
    elif question_type == "long":
        type_instructions = (
            "Provide a well-structured, comprehensive exam answer with clear headings or bullet points. "
            "Explain mechanisms, steps, or comparisons thoroughly using simple English."
        )
        max_toks = 350
    else:  # conceptual
        type_instructions = (
            "Provide an intuitive, conceptual explanation focusing on the underlying 'why' and 'how'. "
            "Make it easy for students to grasp the fundamental principle."
        )
        max_toks = 250

    system_prompt = (
        "You are Offline Study AI, an expert exam preparation tutor for college students. "
        f"Answer the exam question based STRICTLY and ONLY on the provided context excerpts from '{filename}'.\n\n"
        "Rules:\n"
        "1. Base your answer strictly and completely on the provided context.\n"
        "2. Do NOT invent, assume, or bring in outside facts.\n"
        "3. Use simple, clear, exam-friendly English.\n"
        f"4. {type_instructions}\n"
        "5. Be direct, clear, and educational."
    )

    prompt = (
        f"Document: {filename}\n"
        f"Context excerpts:\n"
        f"{context_str}\n\n"
        f"Question: {clean_q}\n"
        f"Question Type: {question_type}\n\n"
        f"Answer:"
    )

    raw_answer = generate_rag_answer(
        prompt=prompt,
        system_prompt=system_prompt,
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=0.1,
        max_tokens=max_toks,
    )

    if raw_answer.startswith('"') and raw_answer.endswith('"') and len(raw_answer) > 2:
        raw_answer = raw_answer[1:-1].strip()

    if is_comp:
        raw_answer = format_comparison_answer(raw_answer, clean_q, context_str)

    return {
        "document_id": clean_doc_id,
        "question": clean_q,
        "question_type": question_type,
        "answer": raw_answer.strip(),
        "page_reference": page_ref,
        "sources": sources,
    }
