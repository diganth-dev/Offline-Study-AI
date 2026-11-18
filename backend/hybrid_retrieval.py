"""Hybrid retrieval module for Offline Study AI.

Combines dense semantic vector search (ChromaDB + Hugging Face embeddings)
with lexical BM25Okapi scoring, exact phrase matching, field header matching,
contact pattern recognition, and domain-general synonym expansion for accurate
information retrieval.
"""

from collections import Counter
import math
import re
from typing import Any, Optional


# Standard English stopwords to prevent common words from biasing lexical scores
STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "tell", "give", "show", "find", "please",
}

# Domain-general academic and document field synonyms (not hardcoded to any document)
FIELD_SYNONYMS: dict[str, list[str]] = {
    "college": ["institute", "institution", "university", "academy", "school", "campus"],
    "university": ["college", "institute", "institution", "academy", "school", "campus"],
    "institute": ["college", "institution", "university", "academy", "school", "campus"],
    "institution": ["college", "institute", "university", "academy", "school"],
    "department": ["dept", "branch", "discipline", "division", "program", "stream"],
    "dept": ["department", "branch", "discipline"],
    "mentor": ["guide", "advisor", "supervisor", "faculty", "professor", "teacher", "hod", "head"],
    "guide": ["mentor", "advisor", "supervisor", "faculty", "professor", "teacher"],
    "advisor": ["mentor", "guide", "supervisor", "faculty", "professor"],
    "faculty": ["mentor", "guide", "advisor", "professor", "teacher", "instructor"],
    "student": ["learner", "candidate", "member", "pupil", "scholar", "team"],
    "students": ["learners", "candidates", "members", "pupils", "scholars", "team"],
    "contact": ["email", "phone", "mobile", "whatsapp", "call", "mail", "tel", "details"],
    "email": ["contact", "mail"],
    "phone": ["contact", "mobile", "whatsapp", "call", "tel"],
    "mobile": ["phone", "contact", "whatsapp", "cell"],
    "head": ["hod", "mentor", "lead", "director", "chair"],
    "lead": ["leader", "head", "team lead"],
}

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}|\b\d{10}\b")


def tokenize(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    if not text:
        return []
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9_]+", text)]


def get_content_tokens(text: str) -> list[str]:
    """Extracts non-stopword content tokens."""
    tokens = tokenize(text)
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return content if content else tokens


def is_short_or_factual_query(query: str) -> bool:
    """Detects whether a query is a short question or factual inquiry.

    Identifies:
    - Terse questions (<= 5 words, e.g., 'Department', 'Department name', 'Student name')
    - Factual inquiry patterns ('Who is the mentor?', 'What is the college name?', 'Contact details')
    """
    clean = query.strip()
    words = tokenize(clean)
    if not words:
        return False

    if len(words) <= 5:
        return True

    ql = clean.lower().rstrip("?.! ")
    factual_starters = (
        "who is", "who are", "who was", "who were",
        "what is", "what are", "what was", "what were",
        "which department", "which college", "which institute",
        "name of", "names of", "contact details", "contact info",
        "phone number", "email address",
    )
    if any(ql.startswith(s) for s in factual_starters):
        return True

    if any(w in ("department", "mentor", "student", "college", "institute", "contact") for w in words[:3]):
        return True

    return False


def expand_query_terms(tokens: list[str]) -> list[str]:
    """Expands query content tokens with domain-general synonyms."""
    expanded: list[str] = list(tokens)
    for t in tokens:
        if t in FIELD_SYNONYMS:
            expanded.extend(FIELD_SYNONYMS[t])
    # Preserve order while removing duplicates
    seen: set[str] = set()
    result: list[str] = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class BM25Okapi:
    """BM25Okapi ranker for text chunks."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(documents)
        self.doc_token_counts: list[Counter[str]] = [Counter(tokenize(doc)) for doc in documents]
        self.doc_lens: list[int] = [sum(c.values()) for c in self.doc_token_counts]
        self.avgdl = (sum(self.doc_lens) / max(1, self.corpus_size)) if self.corpus_size > 0 else 1.0

        # Precompute document frequencies df(t)
        self.df: dict[str, int] = {}
        for counts in self.doc_token_counts:
            for term in counts.keys():
                self.df[term] = self.df.get(term, 0) + 1

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        """Computes BM25 score for a specific document against query terms."""
        if doc_idx < 0 or doc_idx >= self.corpus_size:
            return 0.0

        counts = self.doc_token_counts[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0

        for term in query_terms:
            f = counts.get(term, 0)
            if f <= 0:
                continue

            n_docs = self.df.get(term, 0)
            # Robertson-Spärck Jones IDF
            idf = math.log((self.corpus_size - n_docs + 0.5) / (n_docs + 0.5) + 1.0)
            numerator = f * (self.k1 + 1.0)
            denominator = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            score += idf * (numerator / denominator)

        return score


def compute_lexical_score(
    query: str,
    chunks: list[dict[str, Any]],
) -> list[float]:
    """Computes lexical relevance scores combining BM25, exact phrase matching,

    contact pattern recognition, and field-header alignment across chunks.
    """
    if not chunks or not query.strip():
        return [0.0] * len(chunks)

    doc_texts = [c.get("text", "") for c in chunks]
    bm25 = BM25Okapi(doc_texts)

    content_tokens = get_content_tokens(query)
    expanded_tokens = expand_query_terms(content_tokens)
    clean_query = " ".join(content_tokens).lower()

    lexical_scores: list[float] = []

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        text_lower = text.lower()

        # 1. Base BM25 score with synonym expansion
        bm25_val = bm25.score(expanded_tokens, idx)

        # 2. Exact phrase bonus (e.g. 'department name' appearing verbatim or synonym phrases)
        phrase_bonus = 0.0
        if clean_query and clean_query in text_lower:
            phrase_bonus += 2.5
        elif len(content_tokens) >= 2:
            # Check 2-term sub-phrases
            for i in range(len(content_tokens) - 1):
                sub = f"{content_tokens[i]} {content_tokens[i+1]}"
                if sub in text_lower:
                    phrase_bonus += 1.2
                    break

        # Check synonym phrase matches (e.g., 'institute name', 'faculty mentor')
        for orig in content_tokens:
            if orig in FIELD_SYNONYMS:
                for syn in FIELD_SYNONYMS[orig]:
                    if f"{syn} name" in text_lower or f"{syn}:" in text_lower or f"{syn} information" in text_lower:
                        phrase_bonus += 2.0
                        break

        # 3. Contact patterns (emails, phone numbers) when contact info is queried
        has_email = bool(EMAIL_REGEX.search(text))
        has_phone = bool(PHONE_REGEX.search(text))
        if any(t in ("phone", "mobile", "call", "whatsapp") for t in content_tokens) and has_phone:
            phrase_bonus += 2.5
        if any(t in ("email", "mail") for t in content_tokens) and has_email:
            phrase_bonus += 2.5
        if any(t in ("contact", "details") for t in content_tokens) and (has_email or has_phone):
            phrase_bonus += 2.0

        # 4. Field header match bonus (e.g. 'Department:', 'Mentor Name\n', 'Institute Name:', 'Contact Details:')
        header_bonus = 0.0
        for token in expanded_tokens:
            pattern = r"(?:^|\n|\r)\s*" + re.escape(token) + r"(\s+name)?\s*[:\n\r\-]"
            if re.search(pattern, text_lower):
                header_bonus += 1.5
                break

        # 5. Keyword coverage
        matched_content = sum(1 for t in content_tokens if t in text_lower)
        coverage = (matched_content / len(content_tokens)) if content_tokens else 0.0

        # Check whether any content token, expanded synonym, or recognized pattern matched
        matched_any = (
            any(t in text_lower for t in expanded_tokens)
            or (has_email and any(t in ("email", "mail", "contact") for t in content_tokens))
            or (has_phone and any(t in ("phone", "mobile", "contact") for t in content_tokens))
        )
        if not matched_any:
            lexical_scores.append(0.0)
            continue

        total_lexical = bm25_val + phrase_bonus + header_bonus + (coverage * 2.0)
        lexical_scores.append(round(total_lexical, 4))

    return lexical_scores


def combine_hybrid_scores(
    dense_similarity: float,
    lexical_score: float,
) -> float:
    """Fuses dense semantic cosine similarity with lexical score.

    Properties:
    - If lexical_score is 0 (no keywords present): returns dense_similarity.
      Ensures off-topic queries remain below MIN_RELEVANCE_THRESHOLD (0.20).
    - If dense_similarity is high (e.g. >= 0.20) and lexical is 0: preserves dense_similarity.
    - If lexical_score > 0 (strong keyword / phrase / pattern match): calibrates the score so
      accurate matches reliably rank high and surpass the relevance threshold.
    - The output is bounded in [0.0, 1.0].
    """
    if lexical_score <= 0.0:
        return round(max(0.0, min(1.0, dense_similarity)), 4)

    # Soft saturation for lexical score into [0, 1)
    norm_lexical = lexical_score / (lexical_score + 1.5)

    # Hybrid combination: ensures verified lexical hits receive strong confidence
    hybrid_score = max(dense_similarity, norm_lexical, 0.3 * dense_similarity + 0.7 * norm_lexical)
    return round(min(1.0, max(0.0, hybrid_score)), 4)
