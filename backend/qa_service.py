"""RAG Question Answering Service for Offline Study AI.

Combines ChromaDB vector retrieval with the local Ollama LLM (qwen2.5:1.5b)
to answer study questions based strictly on uploaded documents.
Also resolves document metadata questions (pages, chunks, documents count, filenames)
directly from ChromaDB metadata without calling Ollama.
"""

import re
from typing import Any, Optional

try:
    from backend.vector_store import (
        search_documents,
        get_indexed_documents,
        get_document_chunks,
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
        handle_metadata_question,
        classify_contextual_instruction,
        get_instruction_guidance,
        find_matching_document,
        is_short_summary_request,
        is_detailed_explanation_request,
        is_comparison_question,
        extract_comparison_topics,
        format_comparison_answer,
    )
    from backend.hybrid_retrieval import is_short_or_factual_query, get_content_tokens
except ImportError:
    from vector_store import (
        search_documents,
        get_indexed_documents,
        get_document_chunks,
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
        handle_metadata_question,
        classify_contextual_instruction,
        get_instruction_guidance,
        find_matching_document,
        is_short_summary_request,
        is_detailed_explanation_request,
        is_comparison_question,
        extract_comparison_topics,
        format_comparison_answer,
    )
    from hybrid_retrieval import is_short_or_factual_query, get_content_tokens

# Minimum cosine similarity required to consider retrieved context relevant.
# Cosine similarity is in [0, 1]. For all-MiniLM-L6-v2, completely unrelated queries score < 0.15.
MIN_RELEVANCE_THRESHOLD = 0.20


def format_short_description_answer(raw_answer: str) -> str:
    """Formats and standardizes the AI response into a clean, concise point-wise short description.

    Ensures the response begins with '**Short Description**' and is composed of 4-6 concise,
    student-friendly bullet points without rambling preambles or unstructured paragraphs.
    """
    ans = raw_answer.strip()
    if ans.startswith('"') and ans.endswith('"') and len(ans) > 2:
        ans = ans[1:-1].strip()

    header_match = re.search(r"(?:^|\n)\s*(?:[#*_`\s]*)(?:short\s+description)(?:[#*_`:\s]*)", ans, re.IGNORECASE)
    if header_match:
        content_after = ans[header_match.end():].strip()
    else:
        content_after = re.sub(
            r"^(?:(?:Here\s+(?:is|are)|Below\s+is|Sure[!,.]?|Certainly[!,.]?)[^\n]*\n+)",
            "",
            ans,
            flags=re.IGNORECASE,
        ).strip()

    lines = [line.strip() for line in content_after.split("\n") if line.strip()]

    # Filter out empty or conversational chatter lines
    cleaned_lines = []
    for line in lines:
        if re.match(r"^(?:here\s+(?:is|are)|in\s+summary|to\s+summarize|hope\s+this\s+helps)[^\n]*:?$", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    has_bullets = any(re.match(r"^[-*•]\s+|\d+\.\s+", line) for line in cleaned_lines)

    if not has_bullets and cleaned_lines:
        # Convert plain paragraphs to 4-5 concise bullet points
        raw_text = " ".join(cleaned_lines)
        raw_text = re.sub(r"^(?:Sure|Certainly|Of course|Here is the summary)[!,.]?\s*", "", raw_text, flags=re.IGNORECASE)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]
        if len(sentences) >= 2:
            bullets = []
            chunk_size = max(1, len(sentences) // 4)
            for i in range(0, len(sentences), chunk_size):
                bullet_text = " ".join(sentences[i:i + chunk_size])
                bullets.append(f"- {bullet_text}")
                if len(bullets) == 5:
                    if i + chunk_size < len(sentences):
                        bullets[-1] += " " + " ".join(sentences[i + chunk_size:])
                    break
            return "**Short Description**\n" + "\n".join(bullets)

    # Standardize existing bullets to "- "
    formatted_bullets = []
    for line in cleaned_lines:
        standardized = re.sub(r"^[-*•]\s+|\d+\.\s+", "- ", line)
        if not standardized.startswith("- "):
            standardized = f"- {standardized}"
        formatted_bullets.append(standardized)

    if not formatted_bullets:
        return f"**Short Description**\n- {content_after}"

    return "**Short Description**\n" + "\n".join(formatted_bullets)


def answer_study_question(
    question: str,
    top_k: int = 4,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Answers a question based strictly on uploaded document content or metadata.

    Workflow:
    1. Validates the question.
    2. If document_id is provided, verifies that the document exists and scopes
       all metadata questions and vector searches to that document only.
    3. Checks if question asks about document metadata (page count, total pages,
       document count, filenames, chunk count). If so, answers directly from ChromaDB
       metadata without invoking Ollama.
    4. For normal study questions: verifies Ollama availability and retrieves top_k chunks
       (filtered by user_id and document_id when specified).
    5. Evaluates relevance threshold: if no relevant info found, clearly notifies user.
    6. Formats prompt and invokes Ollama to synthesize a grounded answer.
    7. Returns answer, source metadata (filename, page numbers), and retrieved chunks.

    Args:
        question: User's question text.
        top_k: Number of relevant chunks to retrieve (default: 4).
        collection_name: ChromaDB collection to search.
        persist_directory: Custom ChromaDB path if applicable.
        ollama_model: Model name for Ollama (default: qwen2.5:1.5b).
        ollama_base_url: Base URL for Ollama daemon.
        document_id: Optional document ID to restrict search and metadata to a single document.
        user_id: Optional user ID to isolate search to the authenticated user.

    Returns:
        Dict with keys:
            - question: str
            - answer: str
            - sources: list[dict] (unique filename and page_number pairs)
            - context_used: list[dict] (retrieved chunk details)
            - document_id: Optional[str] (present when document-scoped)
    """
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty.")

    # 1. Fetch indexed documents to inspect metadata (scoped to user_id)
    indexed_docs = get_indexed_documents(
        collection_name=collection_name,
        persist_directory=persist_directory,
        user_id=user_id,
    )

    selected_doc: Optional[dict[str, Any]] = None
    if document_id:
        selected_doc = next((d for d in indexed_docs if d.get("document_id") == document_id), None)
        if not selected_doc:
            return {
                "question": clean_question,
                "answer": f"Selected document with ID '{document_id}' was not found in the study library. It may have been removed.",
                "sources": [],
                "context_used": [],
                "document_id": document_id,
            }

    # 2. Resolve document-related metadata questions directly without calling Ollama
    target_docs = [selected_doc] if selected_doc else indexed_docs
    metadata_response = handle_metadata_question(
        clean_question,
        target_docs,
        selected_document=selected_doc,
    )
    if metadata_response is not None:
        if document_id:
            metadata_response["document_id"] = document_id
        return metadata_response

    # 3. Handle broad contextual instructions (e.g. 'Explain briefly', 'Summarize this document',
    # 'What are the main topics?', 'Give important points', 'Explain the complete document')
    instruction_type = classify_contextual_instruction(clean_question)
    if instruction_type is not None:
        # Determine the target document for the instruction
        target_doc = selected_doc
        if not target_doc:
            matched_doc = find_matching_document(clean_question, indexed_docs)
            if matched_doc:
                target_doc = matched_doc
            elif len(indexed_docs) == 1:
                target_doc = indexed_docs[0]
            elif len(indexed_docs) > 1:
                lines = [f"You have {len(indexed_docs)} uploaded documents. Which document would you like to explore?"]
                for idx, d in enumerate(indexed_docs, start=1):
                    lines.append(f"{idx}. **{d['filename']}** ({d['total_pages']} pages)")
                lines.append("\nPlease select a document or specify its name in your question.")
                return {
                    "question": clean_question,
                    "answer": "\n".join(lines),
                    "sources": [{"filename": d["filename"], "page_number": 1} for d in indexed_docs],
                    "context_used": [],
                }
            else:
                return {
                    "question": clean_question,
                    "answer": "No study documents have been uploaded yet. Please upload a PDF first to explore its content.",
                    "sources": [],
                    "context_used": [],
                }

        # Check if the document contains extractable text
        all_chunks = get_document_chunks(
            document_id=target_doc["document_id"],
            collection_name=collection_name,
            persist_directory=persist_directory,
            user_id=user_id,
        )
        if not all_chunks or not any(c.get("text", "").strip() for c in all_chunks):
            return {
                "question": clean_question,
                "answer": (
                    f"The document '{target_doc['filename']}' contains no extractable text "
                    f"(it may be a scanned image or empty PDF). Please upload a document with readable text."
                ),
                "sources": [],
                "context_used": [],
                "document_id": target_doc["document_id"],
            }

        # Sample representative chunks distributed evenly across the document
        summary_chunks = select_representative_chunks(all_chunks, max_chunks=8)

        # Verify Ollama availability
        check_ollama_availability(base_url=ollama_base_url, model=ollama_model)

        # Extract unique sources and format context blocks
        summary_sources: list[dict[str, Any]] = []
        seen_pages: set[int] = set()
        context_blocks: list[str] = []
        for c in summary_chunks:
            fn = c.get("filename", target_doc["filename"])
            pn = int(c.get("page_number", 1))
            if pn not in seen_pages:
                seen_pages.add(pn)
                summary_sources.append({"filename": fn, "page_number": pn})
            context_blocks.append(f"[Source: {fn} | Page: {pn}]\n{c.get('text', '')}")

        summary_sources.sort(key=lambda s: s["page_number"])
        context_str = "\n\n".join(context_blocks)

        if instruction_type in ("short_summary", "summary", "brief_explanation"):
            system_prompt = (
                "You are Offline Study AI, an expert, factual, and student-friendly learning assistant. "
                f"The student has selected the document: '{target_doc['filename']}'. "
                "Your task is to generate a SHORT, SIMPLE, POINT-WISE summary based STRICTLY and ONLY on the provided context excerpts from this document.\n"
                "Rules:\n"
                "1. Ground your response completely and strictly in the provided document text.\n"
                "2. Use simple English that students can easily understand.\n"
                "3. Format your response strictly starting with '**Short Description**' followed by 4 to 6 concise bullet points (- ).\n"
                "4. Keep each point to 1 to 2 short sentences.\n"
                "5. Avoid unnecessary technical explanations.\n"
                "6. Do not repeat the same information across bullets.\n"
                "7. Do not invent, extrapolate, or bring outside facts not mentioned in the context excerpts.\n"
                "8. Start directly with '**Short Description**' without any conversational filler or introductory sentences."
            )
        elif instruction_type == "detailed_explanation":
            system_prompt = (
                "You are Offline Study AI, an expert, factual, and student-friendly learning assistant. "
                f"The student has selected the document: '{target_doc['filename']}'. "
                "Your task is to provide a comprehensive, detailed, and in-depth explanation based STRICTLY and ONLY on the provided context excerpts from this document.\n"
                "Rules:\n"
                "1. Ground your response completely and strictly in the provided document text.\n"
                "2. Explain concepts in clear, student-friendly language suitable for college study with thorough coverage of details, mechanisms, and background.\n"
                "3. Do not invent, extrapolate, or bring outside facts not mentioned in the context excerpts.\n"
                "4. Provide detailed, well-structured paragraphs and sections as requested."
            )
        else:
            system_prompt = (
                "You are Offline Study AI, an expert, factual, and student-friendly learning assistant. "
                f"The student has selected the document: '{target_doc['filename']}'. "
                "Your task is to respond accurately based STRICTLY and ONLY on the provided context excerpts from this document. "
                "Rules:\n"
                "1. Ground your response completely in the provided document text.\n"
                "2. Explain concepts in clear, student-friendly language suitable for college study.\n"
                "3. Do not invent, extrapolate, or bring outside facts not mentioned in the context excerpts.\n"
                "4. Follow the specific format requested (e.g. key bullet points, main topics, or summary)."
            )

        guidance = get_instruction_guidance(instruction_type, target_doc["filename"])
        prompt = (
            f"Document: {target_doc['filename']}\n"
            f"Context excerpts across the document:\n"
            f"{context_str}\n\n"
            f"Student Request: {clean_question}\n"
            f"Guidance: {guidance}\n\n"
            f"Answer:"
        )

        answer = generate_rag_answer(
            prompt=prompt,
            system_prompt=system_prompt,
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.1,
        )
        if answer.startswith('"') and answer.endswith('"') and len(answer) > 2:
            answer = answer[1:-1].strip()

        if instruction_type in ("short_summary", "summary", "brief_explanation"):
            answer = format_short_description_answer(answer)

        return {
            "question": clean_question,
            "answer": answer,
            "sources": summary_sources,
            "context_used": summary_chunks,
            "document_id": target_doc["document_id"],
        }

    # 4. Verify Ollama service and model availability for RAG study questions
    check_ollama_availability(base_url=ollama_base_url, model=ollama_model)

    # 4. Determine retrieval depth: retrieve additional candidate chunks for short factual queries and comparison questions
    is_short = is_short_or_factual_query(clean_question)
    is_comp = is_comparison_question(clean_question)
    search_k = max(top_k, 6) if (is_short or is_comp) else top_k

    # Retrieve relevant chunks via hybrid search (scoped to document_id and user_id if provided)
    chunks = search_documents(
        query=clean_question,
        top_k=search_k,
        collection_name=collection_name,
        persist_directory=persist_directory,
        document_id=document_id,
        user_id=user_id,
    )

    # For comparison questions, ensure chunks covering both topics are present
    if is_comp and chunks:
        comp_topics = extract_comparison_topics(clean_question)
        if len(comp_topics) == 2:
            t2 = comp_topics[1]
            has_t2 = any(t2.lower() in c.get("text", "").lower() for c in chunks)
            if not has_t2:
                extra_chunks = search_documents(
                    query=t2,
                    top_k=3,
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                    document_id=document_id,
                    user_id=user_id,
                )
                seen_chunk_ids = {c.get("chunk_id") for c in chunks}
                for ec in extra_chunks:
                    if ec.get("chunk_id") not in seen_chunk_ids:
                        chunks.append(ec)
                        seen_chunk_ids.add(ec.get("chunk_id"))

    # 5. Check if any context exists or if top match is below relevance threshold
    if not chunks:
        not_found_msg = (
            f"The selected document ({selected_doc['filename']}) does not contain information to answer this question."
            if selected_doc else
            "The provided documents do not contain information to answer this question. Please upload and index relevant study materials first."
        )
        res_empty: dict[str, Any] = {
            "question": clean_question,
            "answer": not_found_msg,
            "sources": [],
            "context_used": [],
        }
        if document_id:
            res_empty["document_id"] = document_id
        return res_empty

    max_sim = max((c.get("similarity_score", 0.0) for c in chunks), default=0.0)
    if max_sim < MIN_RELEVANCE_THRESHOLD:
        not_found_msg = (
            f"The selected document ({selected_doc['filename']}) does not contain information to answer this question."
            if selected_doc else
            "The provided documents do not contain information to answer this question."
        )
        res_below: dict[str, Any] = {
            "question": clean_question,
            "answer": not_found_msg,
            "sources": [],
            "context_used": chunks,
        }
        if document_id:
            res_below["document_id"] = document_id
        return res_below

    # For short factual / entity inquiries, ensure that at least one content keyword or synonym
    # has lexical evidence in the retrieved context before calling Ollama.
    # This prevents the LLM from hallucinating when an entity (e.g., 'principal') is completely absent
    # but generic academic terms yield a borderline dense embedding score.
    if is_short:
        content_tokens = get_content_tokens(clean_question)
        if content_tokens:
            max_lex = max((c.get("lexical_score", 0.0) for c in chunks), default=0.0)
            if max_lex <= 0.0 and max_sim < 0.40:
                not_found_msg = (
                    f"The selected document ({selected_doc['filename']}) does not contain information to answer this question."
                    if selected_doc else
                    "The provided documents do not contain information to answer this question."
                )
                res_missing: dict[str, Any] = {
                    "question": clean_question,
                    "answer": not_found_msg,
                    "sources": [],
                    "context_used": chunks,
                }
                if document_id:
                    res_missing["document_id"] = document_id
                return res_missing

    # Filter out weak trailing chunks if expanded search_k was used, while ensuring top matches remain
    if is_short and len(chunks) > top_k:
        # Keep chunks with similarity >= 0.15 or the top 3
        filtered_chunks = [c for c in chunks if c.get("similarity_score", 0.0) >= 0.15]
        active_chunks = filtered_chunks if filtered_chunks else chunks[:top_k]
    else:
        active_chunks = chunks

    # Sort context chunks in natural document sequence (by page_number and chunk_number)
    sorted_chunks = sorted(
        active_chunks,
        key=lambda c: (int(c.get("page_number", 1)), int(c.get("chunk_number", 1)))
    )

    # 5. Extract unique source references (filename and page_number)
    sources: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, int]] = set()
    for c in sorted_chunks:
        fn = c.get("filename", "")
        pn = int(c.get("page_number", 1))
        key = (fn, pn)
        if key not in seen_sources and fn:
            seen_sources.add(key)
            sources.append({"filename": fn, "page_number": pn})

    # 6. Format context excerpts for the prompt
    context_blocks: list[str] = []
    for c in sorted_chunks:
        fn = c.get("filename", "Document")
        pn = c.get("page_number", 1)
        txt = c.get("text", "")
        context_blocks.append(f"[Source: {fn} | Page: {pn}]\n{txt}")

    context_str = "\n\n".join(context_blocks)

    # 7. Compose strict system prompt and user prompt
    is_short_brevity = is_short_summary_request(clean_question) and not is_comp
    is_detail = is_detailed_explanation_request(clean_question) and not is_comp

    formatting_rules = ""
    if is_comp:
        formatting_rules = (
            "\n7. Formatting: The student requested a structured comparison/differentiation answer. "
            "You MUST structure your response strictly in clean Markdown with the following sections:\n\n"
            "### Introduction\n"
            "Provide a short 1 to 2 sentence student-friendly introduction defining both concepts based strictly on the document.\n\n"
            "### Comparison Table\n"
            "Display the key differences in a clean, readable Markdown table with explicit columns:\n"
            "| Basis of Comparison | Concept A | Concept B |\n"
            "| :--- | :--- | :--- |\n"
            "| [Parameter 1] | [Aspect for A] | [Aspect for B] |\n"
            "| [Parameter 2] | [Aspect for A] | [Aspect for B] |\n"
            "| [Parameter 3] | [Aspect for A] | [Aspect for B] |\n\n"
            "### Important Points\n"
            "Explain key points and differences in simple, student-friendly bullet points (- ).\n\n"
            "### Examples from Document\n"
            "Provide concrete examples cited directly from the PDF context excerpts for each concept.\n\n"
            "### Related Concepts\n"
            "(ONLY include this section if other related concepts are explicitly mentioned and supported in the provided context excerpts. If none are supported in the context, do NOT include this section).\n\n"
            "### Conclusion\n"
            "Provide a short 1 to 2 sentence exam-ready conclusion summarizing the key takeaway."
        )
    elif is_short_brevity:
        formatting_rules = (
            "\n7. Formatting: The student requested a short, simple explanation. "
            "Provide a concise, simple, point-wise response in 4 to 6 clear bullet points using plain, student-friendly English. "
            "Keep each point to 1-2 short sentences. Avoid unnecessary technical explanations and do not repeat information."
        )
    elif is_detail:
        formatting_rules = (
            "\n7. Formatting: The student requested a detailed explanation. "
            "Provide a comprehensive, detailed, and in-depth explanation covering all concepts, mechanisms, and background thoroughly with complete detail."
        )

    system_prompt = (
        "You are Offline Study AI, a strict and factual personal learning assistant. "
        "Answer the user's question accurately based STRICTLY and ONLY on the provided context excerpts from their study materials. "
        "Rules:\n"
        "1. Base your answer primarily and strictly on the provided context.\n"
        "2. Do NOT invent, assume, extrapolate, or bring outside facts not mentioned in the context.\n"
        "3. If the context does not contain enough information to answer the question, state clearly:\n"
        "'The provided documents do not contain information to answer this question.'\n"
        "4. For factual questions (such as department name, mentor name, student names, college or institute name, contact details):\n"
        "   - State the requested information directly and clearly from the context.\n"
        "   - If multiple entries or members are listed (such as student team members), list all of them.\n"
        "5. Strict Role & Designation Accuracy: Do NOT assume, infer, or substitute different roles, titles, or designations. "
        "For example, do NOT claim a Mentor, HOD, Teacher, or Professor is a Principal, Dean, Chancellor, or President unless the text explicitly states that exact title. "
        "If the requested role, person, or fact is not explicitly stated in the context, state clearly:\n"
        "'The provided documents do not contain information to answer this question.'\n"
        "6. Be direct, clear, and educational."
        f"{formatting_rules}"
    )

    prompt = (
        f"Context excerpts from uploaded study materials:\n"
        f"{context_str}\n\n"
        f"Question: {clean_question}\n\n"
        f"Answer:"
    )

    # 8. Generate answer via Ollama
    answer = generate_rag_answer(
        prompt=prompt,
        system_prompt=system_prompt,
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=0.1,
    )

    # Strip surrounding quotes if the model wrapped its output in quotes
    if answer.startswith('"') and answer.endswith('"') and len(answer) > 2:
        answer = answer[1:-1].strip()

    if is_comp:
        answer = format_comparison_answer(answer, clean_question, context_str)
    elif is_short_brevity and not any(phrase in clean_question.lower() for phrase in ["mentor", "department", "student", "college", "phone", "email"]):
        answer = format_short_description_answer(answer)

    res: dict[str, Any] = {
        "question": clean_question,
        "answer": answer,
        "sources": sources,
        "context_used": chunks,
    }
    if document_id:
        res["document_id"] = document_id
    return res
