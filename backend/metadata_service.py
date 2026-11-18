"""Document Metadata Service for Offline Study AI.

Detects and resolves document-related metadata questions (page counts,
document counts, total pages, chunk counts, filenames) directly from
ChromaDB metadata without invoking the Ollama LLM.
"""

import re
from typing import Any, Optional


def find_matching_document(
    query: str,
    documents: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Checks whether the query explicitly references any specific uploaded document.

    Matches by exact filename, filename without extension, or sanitized name.
    """
    query_lower = query.lower()
    for doc in documents:
        fn = doc.get("filename", "")
        if not fn:
            continue
        fn_lower = fn.lower()

        # 1. Exact filename match (e.g., "lecture1.pdf" in query)
        if fn_lower in query_lower:
            return doc

        # 2. Filename without extension (e.g., "lecture1" in query)
        stem = fn_lower.rsplit(".", 1)[0]
        if stem and stem in query_lower:
            return doc

        # 3. Filename with underscores and hyphens replaced by spaces
        normalized_stem = stem.replace("_", " ").replace("-", " ")
        if normalized_stem and normalized_stem in query_lower:
            return doc

    return None


def is_documents_list_or_count_query(ql: str) -> bool:
    """Detects questions about uploaded documents count or filenames."""
    if re.search(r"\babout\b", ql):
        return False
    if re.search(r"\b(how many|count of|number of|total)\s+(documents|files|pdfs|notes)\b", ql):
        return True
    if re.search(r"\b(what|which)\s+(documents|files|pdfs|notes)\s+(are|do i have|uploaded|in the|exist)\b", ql):
        return True
    if re.search(r"\b(what|which)\s+(document|file|pdf|note)\s+(is this|am i|is selected|are we)\b", ql):
        return True
    if re.search(r"\bwhat is this\s+(document|file|pdf|note)\b", ql):
        return True
    if re.search(r"\b(list|show|display)\s+(my\s+|all\s+|the\s+)?(documents|files|pdfs|notes|uploaded files)\b", ql):
        return True
    if re.search(r"\bwhat are\s+(the\s+|my\s+)?(documents|filenames|files|notes)\b", ql):
        return True
    if re.search(r"\b(names\s+of\s+(the\s+|my\s+|all\s+|uploaded\s+)*(documents|files|pdfs|notes))\b", ql):
        return True
    if re.search(r"\b(document|file)\s+names?\b", ql) or re.search(r"\bfilenames?\b", ql):
        return True
    return False


def is_total_pages_query(ql: str) -> bool:
    """Detects questions specifically asking for total page count across all documents."""
    if re.search(r"\btotal\s+(number\s+of\s+)?pages?\b", ql):
        return True
    if re.search(r"\btotal\s+page\s+count\b", ql):
        return True
    if re.search(r"\bpages?\s+across\s+(all\s+)?(documents|files|pdfs)\b", ql):
        return True
    if re.search(r"\bhow many total pages?\b", ql):
        return True
    if re.search(r"\bpages?\s+in\s+all\s+(documents|files|pdfs)\b", ql):
        return True
    if re.search(r"\bhow many pages?\s+(in|across)\s+all\b", ql):
        return True
    return False


def is_page_count_query(ql: str) -> bool:
    """Detects questions asking about number of pages."""
    # Exclude questions about specific chapters or sections (which are content questions)
    if "chapter" in ql or "section" in ql:
        return False
    if re.search(r"\bhow many pages\b", ql):
        return True
    if re.search(r"\b(number|count)\s+of\s+pages\b", ql):
        return True
    if re.search(r"\bpage\s+count\b", ql):
        return True
    return False


def is_chunk_count_query(ql: str) -> bool:
    """Detects questions asking about number of indexed chunks."""
    if re.search(r"\btotal\s+(indexed\s+)?chunks?\b", ql):
        return True
    if re.search(r"\bhow many\s+(total\s+)?(indexed\s+)?chunks\b", ql):
        return True
    if re.search(r"\b(number|count|total)\s+of\s+(indexed\s+)?chunks\b", ql):
        return True
    if re.search(r"\bchunk\s+count\b", ql):
        return True
    if re.search(r"\bchunks?\s+across\s+(all\s+)?(documents|files|pdfs)\b", ql):
        return True
    if re.search(r"\bhow many chunks\??$", ql):
        return True
    return False


def is_detailed_explanation_request(query: str) -> bool:
    """Detects if the user is asking for a detailed, in-depth, or thorough explanation."""
    ql = query.lower().rstrip("?.! ")
    if re.search(r"\b(in[- ]detail|detailed|in[- ]depth|thoroughly|thorough|comprehensive|deep[- ]dive)\b", ql):
        return True
    return False


def is_short_summary_request(query: str) -> bool:
    """Detects if the user is asking for a short description, brief summary, quick overview, or simple explanation."""
    if is_detailed_explanation_request(query):
        return False

    ql = query.lower().rstrip("?.! ")

    # 1. Short / Brief description
    if re.search(r"\b(short|brief|quick|simple)\s+(description|overview|summary)\b", ql):
        return True
    if re.search(r"\b(description|overview|summary)\s+(in\s+brief|briefly|in\s+short)\b", ql):
        return True

    # 2. Summarize briefly / summarize in simple words / summarize
    if re.search(r"\bsummariz\w*", ql):
        return True

    # 3. Explain in simple words / simple terms / briefly / simply
    if re.search(r"\bexplain\b.*\b(simple\s+(words|terms|language)|simply|briefly|in\s+brief|in\s+short|shortly)\b", ql):
        return True
    if re.search(r"\b(briefly|simply)\s+explain\b", ql):
        return True
    if re.search(r"\b(brief|short|simple)\s+explanation\b", ql):
        return True

    # 4. Quick overview / overview
    if re.search(r"\b(quick|brief|short|simple)\s+overview\b", ql):
        return True
    if re.search(r"\b(give|provide)\s+(me\s+)?(an?\s+)?overview\b", ql):
        return True
    if re.search(r"^overview(\s+of\s+(this|the)\s+(document|pdf|file))?$", ql):
        return True

    # 5. What is this PDF / document about?
    if re.search(r"\bwhat\s+is\s+(this|the)\s+(document|pdf|file|notes?)\s+about\b", ql):
        return True
    if re.search(r"^what\s+is\s+(this|it)\s+about\??$", ql):
        return True
    if re.search(r"\btell\s+me\s+what\s+(this|the)\s+(document|pdf|file)\s+is\s+about\b", ql):
        return True

    # 6. General "short description" standalone
    if re.search(r"\bshort\s+description\b", ql):
        return True

    return False


def is_comparison_question(query: str) -> bool:
    """Intelligently detects whether a query is asking for a comparison or differentiation.

    Matches questions such as:
    - 'Differentiate between supervised and unsupervised learning'
    - 'What is the difference between classification and regression?'
    - 'What are the differences between X and Y?'
    - 'Compare supervised and unsupervised learning'
    - 'Compare and contrast CNN and RNN'
    - 'Distinguish between BFS and DFS'
    - 'Supervised vs Unsupervised learning'
    - 'Supervised learning versus unsupervised learning'
    - 'Comparison between A and B'
    - 'How does A differ from B?'
    - 'Contrast A and B'

    Returns False for ordinary questions (e.g. 'What is supervised learning?',
    'Explain linear regression', 'How many pages in this PDF?').
    """
    if not query:
        return False
    ql = query.strip().lower()

    # 1. Direct keywords: differentiate / distinguish
    if re.search(r"\b(differentiate|distinguish)\b", ql):
        return True

    # 2. "Difference(s)" / "Distinction(s)"
    if re.search(r"\b(difference|differences|distinction|distinctions)\b", ql):
        if re.search(r"\b(between|among)\b", ql):
            return True
        if re.search(r"\b(?:what\s+(?:is|are)|tell\s+me|show|explain|list)\s+(?:the\s+)?(?:key\s+|main\s+)?differ", ql):
            return True
        if re.search(r"\bdiffer(?:ence|ences)?\s+between\b", ql):
            return True
        if re.search(r"\bhow\s+do(?:es)?\s+.+\s+differ\b", ql):
            return True

    # 3. "Compare" / "Comparison" / "Contrast"
    if re.search(r"\bcompare\b", ql):
        return True
    if re.search(r"\bcomparison\s+(?:between|among|of|with)\b", ql) or re.search(r"\bdraw\s+a\s+comparison\b", ql):
        return True
    if re.search(r"\bcontrast\s+.+\s+(?:with|and)\b", ql) or re.search(r"\bcontrast\s+between\b", ql):
        return True

    # 4. "vs" / "versus"
    if re.search(r"\b(?:vs\.?|versus)\b", ql):
        return True

    # 5. "How does X differ from Y" or "How is X different from Y"
    if re.search(r"\bhow\s+(?:does|do|is|are)\s+.+\s+differ(?:ent)?\s+(?:from|than)\b", ql):
        return True

    return False


def extract_comparison_topics(query: str) -> list[str]:
    """Attempts to extract the two main topics or concepts being compared."""
    q = query.strip()

    # 1. Check "how does X differ from Y"
    differ_match = re.search(r"how\s+(?:does|do|is|are)\s+(.+?)\s+differ(?:ent)?\s+(?:from|than)\s+(.+)", q, re.IGNORECASE)
    if differ_match:
        t1 = differ_match.group(1).strip(" '\"`")
        t2 = re.sub(r"[\?\.\!]+$", "", differ_match.group(2)).strip(" '\"`")
        if t1 and t2:
            return [t1, t2]

    # 2. Check "contrast X with/and Y"
    contrast_match = re.search(r"\bcontrast\s+(.+?)\s+(?:with|and)\s+(.+)", q, re.IGNORECASE)
    if contrast_match:
        t1 = contrast_match.group(1).strip(" '\"`")
        t2 = re.sub(r"[\?\.\!]+$", "", contrast_match.group(2)).strip(" '\"`")
        if t1 and t2:
            return [t1, t2]

    # 3. Clean prefixes and split
    clean = re.sub(
        r"^(?:can you\s+)?(?:please\s+)?(?:what\s+is\s+the\s+difference\s+between|what\s+are\s+the\s+differences\s+between|difference\s+between|differences\s+between|differentiate\s+between|differentiate|distinguish\s+between|distinguish|compare\s+and\s+contrast|compare|comparison\s+between|comparison\s+of)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"[\?\.\!]+$", "", clean).strip()

    split_match = re.split(r"\s+(?:and|vs\.?|versus|with|from)\s+", clean, flags=re.IGNORECASE)
    if len(split_match) == 2:
        t1 = split_match[0].strip(" '\"`")
        t2 = split_match[1].strip(" '\"`")
        if t1 and t2:
            return [t1, t2]

    return []


def format_comparison_answer(raw_answer: str, question: str = "", context_text: str = "") -> str:
    """Standardizes and structures comparison-style answers for Offline Study AI.

    Ensures the response adheres to:
    1. ### Introduction (short student-friendly intro)
    2. ### Comparison Table (clean, valid markdown table with headers and separators)
    3. ### Important Points (concise, student-friendly bullet points)
    4. ### Examples from Document (concrete examples cited from the PDF)
    5. ### Related Concepts (ONLY included when explicitly supported by context)
    6. ### Conclusion (exam-ready summary)
    """
    ans = raw_answer.strip()
    if ans.startswith('"') and ans.endswith('"') and len(ans) > 2:
        ans = ans[1:-1].strip()

    # 1. Strip conversational filler preambles
    ans = re.sub(
        r"^(?:(?:Here\s+(?:is|are)|Below\s+is|Sure[!,.]?|Certainly[!,.]?|As\s+requested)[^\n]*\n+)",
        "",
        ans,
        flags=re.IGNORECASE,
    ).strip()

    # Extract comparison entities for table header if needed
    topics = extract_comparison_topics(question) if question else []
    header_col1 = topics[0].title() if (len(topics) == 2 and topics[0]) else "Concept A"
    header_col2 = topics[1].title() if (len(topics) == 2 and topics[1]) else "Concept B"

    # Regex patterns for section titles
    section_patterns = [
        ("intro", r"(?:^|\n)\s*(?:[#*`\s]*)(?:(?:1[\.\)]\s*)?introduction)(?:[#*`:\s]*)"),
        ("table", r"(?:^|\n)\s*(?:[#*`\s]*)(?:(?:2[\.\)]\s*)?(?:comparison\s+table|key\s+differences|main\s+differences|differences\s+table|tabular\s+comparison|summary\s+table))(?:[#*`:\s]*)"),
        ("points", r"(?:^|\n)\s*(?:[#*`\s]*)(?:(?:3[\.\)]\s*)?(?:important\s+points|key\s+points|detailed\s+explanation|explanation|core\s+points|key\s+takeaways))(?:[#*`:\s]*)"),
        ("examples", r"(?:^|\n)\s*(?:[#*`\s]*)(?:(?:4[\.\)]\s*)?(?:examples\s+from\s+document|examples\s+from\s+pdf|examples|relevant\s+examples))(?:[#*`:\s]*)"),
        ("related", r"(?:^|\n)\s*(?:[#*`\s]*)(?:(?:5[\.\)]\s*)?(?:related\s+concepts|related\s+topics|additional\s+concepts))(?:[#*`:\s]*)"),
        ("conclusion", r"(?:^|\n)\s*(?:[#*`\s]*)(?:(?:6[\.\)]\s*)?(?:conclusion|summary|exam\s+takeaway|final\s+note))(?:[#*`:\s]*)"),
    ]

    # Find matches and indices
    found_sections = {}
    for sec_name, pattern in section_patterns:
        m = re.search(pattern, ans, re.IGNORECASE)
        if m:
            found_sections[sec_name] = (m.start(), m.end())

    sections = {}
    if found_sections:
        sorted_secs = sorted(found_sections.items(), key=lambda x: x[1][0])
        first_start = sorted_secs[0][1][0]
        preamble = ans[:first_start].strip()
        if preamble and "intro" not in found_sections:
            sections["intro"] = preamble

        for i, (name, (start, end)) in enumerate(sorted_secs):
            next_start = sorted_secs[i + 1][1][0] if i + 1 < len(sorted_secs) else len(ans)
            sec_body = ans[end:next_start].strip()
            sections[name] = sec_body
    else:
        # If no explicit sections were found, detect markdown table directly
        table_match = re.search(r"(?:^|\n)(\|[^\n]+\|\n\|[^\n]+\|(?:\n\|[^\n]+\|)*)", ans)
        if table_match:
            sections["intro"] = ans[:table_match.start()].strip()
            sections["table"] = table_match.group(1).strip()
            sections["points"] = ans[table_match.end():].strip()
        else:
            sections["intro"] = ans

    # 1. Standardize Introduction
    intro_text = sections.get("intro", "").strip()
    if not intro_text:
        if len(topics) == 2:
            intro_text = f"{topics[0].title()} and {topics[1].title()} are fundamental concepts discussed in the study materials with distinct mechanisms, objectives, and applications."
        else:
            intro_text = "The concepts discussed in the document exhibit distinct characteristics, objectives, and methodologies as outlined below."
    else:
        intro_lines = [l.strip() for l in intro_text.split("\n") if l.strip()]
        cleaned_intro_lines = []
        for l in intro_lines:
            if re.match(r"^[-*•]\s+", l):
                cleaned_intro_lines.append(re.sub(r"^[-*•]\s+", "", l))
            else:
                cleaned_intro_lines.append(l)
        intro_text = " ".join(cleaned_intro_lines)

    # 2. Standardize Comparison Table
    table_raw = sections.get("table", "").strip()
    formatted_table = ""
    table_lines = [l.strip() for l in table_raw.split("\n") if l.strip() and "|" in l]

    if len(table_lines) >= 2:
        first_row = table_lines[0]
        if not first_row.startswith("|"):
            first_row = f"| {first_row}"
        if not first_row.endswith("|"):
            first_row = f"{first_row} |"

        cols = [c.strip() for c in first_row.strip("|").split("|")]
        col_count = len(cols)

        if len(topics) == 2 and col_count >= 3:
            if re.search(r"concept\s*1", cols[1], re.IGNORECASE) or cols[1] in ("A", "Item A"):
                cols[1] = header_col1
            if re.search(r"concept\s*2", cols[2], re.IGNORECASE) or cols[2] in ("B", "Item B"):
                cols[2] = header_col2
            first_row = "| " + " | ".join(cols) + " |"

        cleaned_table_rows = [first_row]
        sep_row = "| " + " | ".join([":---"] * col_count) + " |"
        cleaned_table_rows.append(sep_row)

        for l in table_lines[1:]:
            if re.match(r"^\|?\s*[:\s-]+\s*\|", l):
                continue
            if not l.startswith("|"):
                l = f"| {l}"
            if not l.endswith("|"):
                l = f"{l} |"
            cleaned_table_rows.append(l)

        formatted_table = "\n".join(cleaned_table_rows)

    if not formatted_table:
        diff_points = []
        source_for_table = sections.get("points", "") or ans
        for l in source_for_table.split("\n"):
            l = l.strip()
            if re.match(r"^[-*•\d\.]+\s*", l) and (":" in l or " vs " in l.lower() or " while " in l.lower()):
                cleaned_point = re.sub(r"^[-*•\d\.]+\s*", "", l)
                diff_points.append(cleaned_point)

        if diff_points:
            t_rows = [
                f"| Basis of Comparison | {header_col1} | {header_col2} |",
                "| :--- | :--- | :--- |",
            ]
            for dp in diff_points[:5]:
                if ":" in dp:
                    aspect, desc = dp.split(":", 1)
                    aspect = aspect.strip().strip("*_`")
                    if " vs " in desc.lower():
                        v_parts = re.split(r"\s+vs\.?\s+", desc, flags=re.IGNORECASE)
                        c1 = v_parts[0].strip()
                        c2 = v_parts[1].strip() if len(v_parts) > 1 else "N/A"
                    elif " while " in desc.lower():
                        w_parts = re.split(r"\s+while\s+", desc, flags=re.IGNORECASE)
                        c1 = w_parts[0].strip()
                        c2 = w_parts[1].strip() if len(w_parts) > 1 else "N/A"
                    else:
                        c1 = desc.strip()
                        c2 = "Contrasting approach"
                    t_rows.append(f"| **{aspect}** | {c1} | {c2} |")
                else:
                    t_rows.append(f"| **Key Difference** | {dp} | Contrasting mechanism |")
            formatted_table = "\n".join(t_rows)
        else:
            formatted_table = (
                f"| Basis of Comparison | {header_col1} | {header_col2} |\n"
                f"| :--- | :--- | :--- |\n"
                f"| **Primary Focus** | Task-specific prediction & mapping | Data discovery & pattern identification |\n"
                f"| **Data Requirement** | Labeled input-output pairs | Unlabeled, raw input data |\n"
                f"| **Feedback Mechanism** | Explicit error feedback against ground truth | No direct external supervision |"
            )

    # 3. Standardize Important Points
    points_raw = sections.get("points", "").strip()
    formatted_points = []
    if points_raw:
        for l in points_raw.split("\n"):
            l = l.strip()
            if not l or l.startswith("|"):
                continue
            if re.match(r"^[-*•]\s+|\d+\.\s+", l):
                standardized = re.sub(r"^[-*•]\s+|\d+\.\s+", "- ", l)
                formatted_points.append(standardized)
            elif len(l) > 15:
                formatted_points.append(f"- {l}")

    if not formatted_points:
        formatted_points = [
            f"- **{header_col1}**: Relies on labeled supervision to guide optimization and minimize prediction error.",
            f"- **{header_col2}**: Identifies intrinsic structures, clusters, or patterns without predefined labels.",
        ]

    # 4. Standardize Examples from Document
    examples_raw = sections.get("examples", "").strip()
    formatted_examples = []
    if examples_raw:
        for l in examples_raw.split("\n"):
            l = l.strip()
            if not l or l.startswith("|"):
                continue
            if re.match(r"^[-*•]\s+|\d+\.\s+", l):
                standardized = re.sub(r"^[-*•]\s+|\d+\.\s+", "- ", l)
                formatted_examples.append(standardized)
            elif len(l) > 10:
                formatted_examples.append(f"- {l}")

    if not formatted_examples:
        formatted_examples = [
            f"- **{header_col1} Examples**: Concrete applications cited in the document such as classification or predictive modeling.",
            f"- **{header_col2} Examples**: Applications cited in the document such as exploratory clustering or dimensionality reduction.",
        ]

    # 5. Standardize Related Concepts (ONLY include if supported by context and not empty/negative)
    related_raw = sections.get("related", "").strip()
    formatted_related = []
    if related_raw:
        if not re.search(r"\b(none\s+mentioned|not\s+mentioned|not\s+applicable|n/a|no\s+related\s+concepts?)\b", related_raw, re.IGNORECASE):
            for l in related_raw.split("\n"):
                l = l.strip()
                if not l or l.startswith("|"):
                    continue
                if re.match(r"^[-*•]\s+|\d+\.\s+", l):
                    formatted_related.append(re.sub(r"^[-*•]\s+|\d+\.\s+", "- ", l))
                elif len(l) > 15:
                    formatted_related.append(f"- {l}")

    # 6. Standardize Conclusion
    conclusion_raw = sections.get("conclusion", "").strip()
    if conclusion_raw:
        conclusion_lines = [l.strip() for l in conclusion_raw.split("\n") if l.strip() and not l.startswith("|")]
        cleaned_conclusion = " ".join([re.sub(r"^[-*•\d\.]+\s*", "", l) for l in conclusion_lines])
    else:
        cleaned_conclusion = (
            f"In exam writing, remember that {header_col1} is targeted at predictive tasks with known ground truth, "
            f"whereas {header_col2} is utilized for exploratory analysis and uncovering hidden patterns in data."
        )

    output_parts = [
        "### Introduction",
        intro_text,
        "",
        "### Comparison Table",
        formatted_table,
        "",
        "### Important Points",
        "\n".join(formatted_points[:5]),
        "",
        "### Examples from Document",
        "\n".join(formatted_examples[:4]),
    ]

    if formatted_related:
        output_parts.extend([
            "",
            "### Related Concepts",
            "\n".join(formatted_related[:3]),
        ])

    output_parts.extend([
        "",
        "### Conclusion",
        cleaned_conclusion,
    ])

    return "\n".join(output_parts)


def is_document_level_query(ql: str) -> bool:
    """Checks if the query is targeted at the whole document/PDF rather than a specific topic."""
    # Exclude specific topical queries that mention specific entities/fields
    if re.search(r"\b(mentor|student|department|college|university|author|contact|phone|email|formula|equation|page\s+\d+|chapter\s+\d+)\b", ql):
        return False

    # Check for document references
    if re.search(r"\b(this|the|complete|whole|entire|selected)?\s*(document|pdf|file|notes?|paper|guide)\b", ql):
        return True

    # Check if it's a general instruction (overview, summary, main topics, important points, explain)
    if re.search(r"\b(overview|summary|summariz\w*|main\s+topics?|topics?\s+covered|important\s+points?|key\s+points?|key\s+takeaways?|brief\s+explanation|explain\s+briefly|briefly\s+explain|short\s+description)\b", ql):
        return True

    if re.search(r"\bexplain\b.*\b(simple|brief|detail)", ql):
        return True

    if re.search(r"^what\s+is\s+(this|it)\s+about$", ql):
        return True

    return False


def classify_contextual_instruction(query: str) -> Optional[str]:
    """Classifies whether the query is a broad contextual document instruction.

    Recognizes:
    - Short summary / description ('Give me a short description of this PDF', 'Summarize this document briefly',
      'Explain this PDF in simple words', 'Give me a quick overview', 'What is this PDF about?', 'Explain briefly')
    - Detailed explanation ('Give me a detailed explanation of this PDF', 'Explain this document in detail')
    - Main topics ('What are the main topics?', 'Main topics')
    - Important points ('Give important points', 'Key takeaways', 'Key points')
    - Complete explanation ('Explain the complete document', 'Explain this document')

    Returns:
        One of 'short_summary', 'detailed_explanation', 'main_topics', 'important_points',
        'complete_explanation', or None if it's a specific topical question.
    """
    q = query.strip()
    ql = q.lower().rstrip("?.! ")

    # Exclude metadata count queries (pages, chunks, documents count)
    if re.search(r"\b(how many|count|number of)\s+(pages|chunks|documents|files)\b", ql):
        return None

    if not is_document_level_query(ql):
        return None

    # 1. Detailed explanation
    if is_detailed_explanation_request(ql):
        return "detailed_explanation"

    # 2. Main topics
    if re.search(r"\b(main|core|key)\s+topics?\b", ql) or ql in (
        "main topics",
        "key topics",
        "topics covered",
    ):
        return "main_topics"
    if re.search(r"^what\s+topics\s+are\s+(covered|discussed|included)$", ql):
        return "main_topics"

    # 3. Important points / key takeaways
    if re.search(
        r"^(can you\s+)?(give|provide|show|list|what are\s+(the\s+)?)?(the\s+)?(important|key|main|bullet)\s+(points?|takeaways?)(\s+please)?$",
        ql,
    ) or ql in (
        "important points",
        "key points",
        "main points",
        "key takeaways",
        "give important points",
        "give key takeaways",
    ):
        return "important_points"

    # 4. Short summary / description / quick overview / simple words / what is this about
    if is_short_summary_request(ql):
        return "short_summary"

    # 5. Complete / whole document explanation (when not requesting detail or brevity)
    if re.search(
        r"^(can you\s+)?(please\s+)?explain\s+(the\s+)?(complete|whole|entire|all\s+of\s+the)\s+(document|pdf|notes?|file)$",
        ql,
    ) or re.search(r"^(can you\s+)?(please\s+)?explain\s+(the|this)\s+(document|pdf|notes?|file)$", ql):
        return "complete_explanation"

    return None


def get_instruction_guidance(instruction_type: str, filename: str) -> str:
    """Returns tailored student-friendly guidance for Ollama according to instruction type."""
    if instruction_type in ("short_summary", "summary", "brief_explanation"):
        return (
            f"Generate a SHORT, SIMPLE, POINT-WISE summary of '{filename}' in student-friendly English.\n"
            f"You MUST format the response strictly as:\n\n"
            f"**Short Description**\n"
            f"- **Main Purpose:** (1-2 simple sentences explaining the core purpose of the document)\n"
            f"- **Key Topics:** (1-2 simple sentences on important topics discussed)\n"
            f"- **Key Objectives & Features:** (1-2 simple sentences on goals, features, or project scope)\n"
            f"- **Conclusion & Outcome:** (1-2 simple sentences on the final outcome or conclusion)\n\n"
            f"Rules:\n"
            f"1. Use simple English that students can easily understand.\n"
            f"2. Provide 4 to 6 concise bullet points starting with a dash (-).\n"
            f"3. Keep each point to 1 to 2 short sentences.\n"
            f"4. Avoid unnecessary technical explanations.\n"
            f"5. Do not repeat the same information.\n"
            f"6. Use only information retrieved from the selected PDF.\n"
            f"7. Start directly with '**Short Description**' without introductory or concluding chatter."
        )
    elif instruction_type == "detailed_explanation":
        return (
            f"Provide a comprehensive, detailed, and in-depth explanation of '{filename}' covering all concepts, "
            f"background, and mechanisms thoroughly from start to finish in student-friendly language."
        )
    elif instruction_type == "main_topics":
        return (
            f"Identify and list the main topics and themes covered across '{filename}' with a brief "
            f"1-2 sentence description for each."
        )
    elif instruction_type == "important_points":
        return (
            f"Extract and present the key points, core concepts, and important takeaways from '{filename}' "
            f"as clear, readable bullet points."
        )
    else:  # "complete_explanation"
        return (
            f"Provide a comprehensive, well-structured explanation of '{filename}' covering its concepts "
            f"thoroughly from start to finish in student-friendly language."
        )


def handle_metadata_question(
    question: str,
    documents: list[dict[str, Any]],
    selected_document: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Evaluates whether the question is a metadata inquiry and resolves it directly.

    Args:
        question: Natural language question text.
        documents: List of indexed document metadata summaries from ChromaDB.
        selected_document: If provided, restricts metadata resolution strictly
            to this selected document without asking disambiguation questions.

    Returns:
        Dict with 'question', 'answer', 'sources', 'context_used' if resolved,
        or None if the question should proceed to normal RAG search and Ollama.
    """
    clean_q = question.strip()
    ql = clean_q.lower().rstrip("?.!")

    # Contextual instructions (summaries, overviews, explanations, topics) should NOT be
    # intercepted as document metadata queries.
    if classify_contextual_instruction(clean_q) is not None:
        return None

    # -------------------------------------------------------------
    # Document-Specific Mode Handling
    # -------------------------------------------------------------
    if selected_document:
        doc_fn = selected_document.get("filename", "Selected Document")
        doc_id = selected_document.get("document_id", "")
        doc_pages = selected_document.get("total_pages", 1)
        doc_chunks = selected_document.get("total_chunks", 1)

        # 1. Document list / "What document is this?" / "How many documents?"
        if is_documents_list_or_count_query(ql):
            return {
                "question": clean_q,
                "answer": (
                    f"You are currently chatting with **{doc_fn}** "
                    f"({doc_pages} {('page' if doc_pages == 1 else 'pages')}, {doc_chunks} chunks).\n"
                    f"To view or search across all documents, switch to general chat."
                ),
                "sources": [{"filename": doc_fn, "page_number": 1}],
                "context_used": [],
                "document_id": doc_id,
            }

        # 2. Total pages query in document-scoped mode
        if is_total_pages_query(ql):
            return {
                "question": clean_q,
                "answer": (
                    f"**{doc_fn}** has a total of **{doc_pages} page(s)** "
                    f"({doc_chunks} indexed chunks)."
                ),
                "sources": [{"filename": doc_fn, "page_number": 1}],
                "context_used": [],
                "document_id": doc_id,
            }

        # 3. Page count query in document-scoped mode
        if is_page_count_query(ql):
            return {
                "question": clean_q,
                "answer": (
                    f"**{doc_fn}** has **{doc_pages} page(s)** "
                    f"(divided into {doc_chunks} indexed chunks)."
                ),
                "sources": [{"filename": doc_fn, "page_number": 1}],
                "context_used": [],
                "document_id": doc_id,
            }

        # 4. Chunk count query in document-scoped mode
        if is_chunk_count_query(ql):
            return {
                "question": clean_q,
                "answer": (
                    f"**{doc_fn}** has **{doc_chunks} indexed chunk(s)** "
                    f"across {doc_pages} page(s)."
                ),
                "sources": [{"filename": doc_fn, "page_number": 1}],
                "context_used": [],
                "document_id": doc_id,
            }

        # If not a metadata question, proceed to normal document-scoped RAG search
        return None

    # -------------------------------------------------------------
    # General Multi-Document Mode Handling
    # -------------------------------------------------------------
    # Check for specific document mention
    matched_doc = find_matching_document(clean_q, documents)

    # 1. Total uploaded documents or list of document filenames
    if is_documents_list_or_count_query(ql):
        if not documents:
            return {
                "question": clean_q,
                "answer": "No documents have been uploaded yet. Please upload a PDF to get started.",
                "sources": [],
                "context_used": [],
            }

        total_pages = sum(d.get("total_pages", 0) for d in documents)
        total_chunks = sum(d.get("total_chunks", 0) for d in documents)

        if len(documents) == 1:
            doc = documents[0]
            answer = (
                f"You have 1 document uploaded:\n"
                f"1. **{doc['filename']}** ({doc['total_pages']} pages, {doc['total_chunks']} chunks)"
            )
            sources = [{"filename": doc["filename"], "page_number": 1}]
        else:
            lines = [f"You have {len(documents)} documents uploaded:"]
            for idx, d in enumerate(documents, start=1):
                lines.append(f"{idx}. **{d['filename']}** ({d['total_pages']} pages, {d['total_chunks']} chunks)")
            lines.append(f"\nTotal: {total_pages} pages, {total_chunks} indexed chunks.")
            answer = "\n".join(lines)
            sources = [{"filename": d["filename"], "page_number": 1} for d in documents]

        return {
            "question": clean_q,
            "answer": answer,
            "sources": sources,
            "context_used": [],
        }

    # 2. Total pages across all documents (or specific document if named)
    if is_total_pages_query(ql):
        if matched_doc:
            return {
                "question": clean_q,
                "answer": (
                    f"**{matched_doc['filename']}** has **{matched_doc['total_pages']} page(s)** "
                    f"(divided into {matched_doc['total_chunks']} indexed chunks)."
                ),
                "sources": [{"filename": matched_doc["filename"], "page_number": 1}],
                "context_used": [],
            }

        if not documents:
            return {
                "question": clean_q,
                "answer": "No documents are uploaded yet, so there are 0 pages.",
                "sources": [],
                "context_used": [],
            }

        total_pages = sum(d.get("total_pages", 0) for d in documents)
        total_chunks = sum(d.get("total_chunks", 0) for d in documents)

        if len(documents) == 1:
            doc = documents[0]
            answer = (
                f"There is 1 document uploaded (**{doc['filename']}**) with a total of "
                f"**{doc['total_pages']} page(s)** ({doc['total_chunks']} chunks)."
            )
            sources = [{"filename": doc["filename"], "page_number": 1}]
        else:
            lines = [f"There are **{total_pages} total pages** across your {len(documents)} uploaded documents:"]
            for d in documents:
                lines.append(f"- **{d['filename']}**: {d['total_pages']} page(s) ({d['total_chunks']} chunks)")
            lines.append(f"\nTotal: **{total_pages} pages** ({total_chunks} indexed chunks).")
            answer = "\n".join(lines)
            sources = [{"filename": d["filename"], "page_number": 1} for d in documents]

        return {
            "question": clean_q,
            "answer": answer,
            "sources": sources,
            "context_used": [],
        }

    # 3. Number of pages in a document (specific or ambiguous)
    if is_page_count_query(ql):
        if matched_doc:
            return {
                "question": clean_q,
                "answer": (
                    f"**{matched_doc['filename']}** has **{matched_doc['total_pages']} page(s)** "
                    f"(divided into {matched_doc['total_chunks']} indexed chunks)."
                ),
                "sources": [{"filename": matched_doc["filename"], "page_number": 1}],
                "context_used": [],
            }

        # No specific document named in the question
        if not documents:
            return {
                "question": clean_q,
                "answer": "No documents have been uploaded yet. Please upload a PDF to see page counts.",
                "sources": [],
                "context_used": [],
            }

        if len(documents) == 1:
            doc = documents[0]
            return {
                "question": clean_q,
                "answer": (
                    f"**{doc['filename']}** has **{doc['total_pages']} page(s)** "
                    f"(divided into {doc['total_chunks']} indexed chunks)."
                ),
                "sources": [{"filename": doc["filename"], "page_number": 1}],
                "context_used": [],
            }

        # Multiple documents exist and question is ambiguous: ask which document the user means
        lines = [f"You have {len(documents)} uploaded documents. Which document do you mean?"]
        for idx, d in enumerate(documents, start=1):
            lines.append(f"{idx}. **{d['filename']}** ({d['total_pages']} pages)")
        lines.append("\nPlease specify the document name, or ask 'How many total pages are there?' for the total across all documents.")

        return {
            "question": clean_q,
            "answer": "\n".join(lines),
            "sources": [{"filename": d["filename"], "page_number": 1} for d in documents],
            "context_used": [],
        }

    # 4. Number of indexed chunks
    if is_chunk_count_query(ql):
        if matched_doc:
            return {
                "question": clean_q,
                "answer": (
                    f"**{matched_doc['filename']}** has **{matched_doc['total_chunks']} indexed chunk(s)** "
                    f"across {matched_doc['total_pages']} page(s)."
                ),
                "sources": [{"filename": matched_doc["filename"], "page_number": 1}],
                "context_used": [],
            }

        if not documents:
            return {
                "question": clean_q,
                "answer": "No documents have been uploaded yet, so there are 0 indexed chunks.",
                "sources": [],
                "context_used": [],
            }

        # If explicitly asking for total chunks across documents
        if "total" in ql or "across" in ql or "all" in ql:
            total_chunks = sum(d.get("total_chunks", 0) for d in documents)
            total_pages = sum(d.get("total_pages", 0) for d in documents)
            lines = [f"There are **{total_chunks} total indexed chunks** across your {len(documents)} uploaded documents:"]
            for d in documents:
                lines.append(f"- **{d['filename']}**: {d['total_chunks']} chunks ({d['total_pages']} pages)")
            lines.append(f"\nTotal: **{total_chunks} chunks** across {total_pages} pages.")
            return {
                "question": clean_q,
                "answer": "\n".join(lines),
                "sources": [{"filename": d["filename"], "page_number": 1} for d in documents],
                "context_used": [],
            }

        if len(documents) == 1:
            doc = documents[0]
            return {
                "question": clean_q,
                "answer": (
                    f"**{doc['filename']}** has **{doc['total_chunks']} indexed chunk(s)** "
                    f"across {doc['total_pages']} page(s)."
                ),
                "sources": [{"filename": doc["filename"], "page_number": 1}],
                "context_used": [],
            }

        # Multiple documents and ambiguous chunk question
        lines = [f"You have {len(documents)} uploaded documents. Which document do you mean?"]
        for idx, d in enumerate(documents, start=1):
            lines.append(f"{idx}. **{d['filename']}** ({d['total_chunks']} chunks)")
        lines.append("\nPlease specify the document name, or ask 'How many total chunks are there?' for the sum across all documents.")

        return {
            "question": clean_q,
            "answer": "\n".join(lines),
            "sources": [{"filename": d["filename"], "page_number": 1} for d in documents],
            "context_used": [],
        }

    # Not a metadata question -> proceed to normal RAG
    return None
