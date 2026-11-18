"""Persistent ChromaDB vector store for Offline Study AI.

Manages persistent storage in backend/storage/chroma_db/, indexes document chunks
with metadata (document ID, filename, page number, chunk number), and performs
similarity search using Hugging Face embeddings.
"""

import hashlib
import os
from typing import Any, Optional
import chromadb
from chromadb.config import Settings

try:
    from backend.embeddings import generate_embeddings, generate_query_embedding
    from backend.hybrid_retrieval import compute_lexical_score, combine_hybrid_scores
except ImportError:
    from embeddings import generate_embeddings, generate_query_embedding
    from hybrid_retrieval import compute_lexical_score, combine_hybrid_scores

# Default persistent storage path inside backend/storage/chroma_db/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PERSIST_DIR = os.path.join(BASE_DIR, "storage", "chroma_db")
DEFAULT_COLLECTION_NAME = "study_documents"

_client: Optional[chromadb.PersistentClient] = None


def get_chroma_path() -> str:
    """Returns the absolute path to the ChromaDB persistent directory."""
    return DEFAULT_PERSIST_DIR


def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client pointing to the storage directory.

    Reuses existing client singleton when using default directory.
    """
    global _client
    target_dir = persist_directory or DEFAULT_PERSIST_DIR
    os.makedirs(target_dir, exist_ok=True)

    if persist_directory is not None:
        return chromadb.PersistentClient(path=target_dir)

    if _client is None:
        _client = chromadb.PersistentClient(path=target_dir)
    return _client


def get_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None
) -> chromadb.Collection:
    """Gets or creates a Chroma collection with cosine distance space."""
    client = get_chroma_client(persist_directory=persist_directory)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def generate_document_id(filename: str) -> str:
    """Generates a stable, unique document ID based on the filename."""
    hash_digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]
    clean_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_")).lower()
    return f"doc_{clean_name}_{hash_digest}"


def index_document(
    filename: str,
    total_pages: int,
    chunks: list[dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None
) -> dict[str, Any]:
    """Indexes document chunks into ChromaDB with Hugging Face embeddings and metadata.

    Removes any existing chunks for this filename to prevent duplicate entries
    when the same PDF is re-indexed.

    Args:
        filename: PDF filename.
        total_pages: Total pages in the PDF document.
        chunks: List of chunk dictionaries containing 'chunk_number', 'page_number', 'text'.
        collection_name: Name of the Chroma collection.
        persist_directory: Optional directory path for ChromaDB.

    Returns:
        Summary dict containing document_id, filename, total_pages, and indexed_chunks count.
    """
    if not chunks:
        raise ValueError("Cannot index an empty list of chunks.")

    collection = get_collection(collection_name=collection_name, persist_directory=persist_directory)
    doc_id = generate_document_id(filename)

    # Prevent duplicate document entries: remove prior chunks for this filename
    try:
        collection.delete(where={"filename": filename})
    except Exception:
        pass

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_num = int(chunk["chunk_number"])
        page_num = int(chunk["page_number"])
        chunk_text = chunk["text"]

        chunk_id = f"{doc_id}_c{chunk_num}"
        ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append({
            "document_id": doc_id,
            "filename": filename,
            "page_number": page_num,
            "chunk_number": chunk_num,
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return {
        "document_id": doc_id,
        "filename": filename,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "indexed_chunks": len(ids),
    }


def search_documents(
    query: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
    document_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Performs hybrid retrieval (semantic vector search + BM25 keyword matching)
    across indexed document chunks.

    Combines dense cosine similarity from Hugging Face embeddings with BM25Okapi,
    exact phrase matching, and domain-general synonym expansion while strictly
    enforcing document-specific filtering when document_id is provided.

    Args:
        query: Search query text.
        top_k: Maximum number of relevant chunks to return (default: 5).
        collection_name: Name of the Chroma collection.
        persist_directory: Optional directory path for ChromaDB.
        document_id: Optional document ID to restrict search to a specific document.

    Returns:
        List of result dictionaries containing filename, page_number, chunk_number,
        document_id, chunk_id, text, distance, and similarity_score.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return []

    collection = get_collection(collection_name=collection_name, persist_directory=persist_directory)
    count = collection.count()
    if count == 0:
        return []

    # 1. Fetch dense vector search candidates
    query_emb = generate_query_embedding(clean_query)
    candidate_k = min(count, max(top_k * 4, 25))

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_emb],
        "n_results": candidate_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if document_id:
        query_kwargs["where"] = {"document_id": document_id}

    dense_res = collection.query(**query_kwargs)

    # Map candidate chunks by chunk_id
    candidates_map: dict[str, dict[str, Any]] = {}

    if dense_res and dense_res.get("documents") and dense_res["documents"][0]:
        docs = dense_res["documents"][0]
        metas = dense_res["metadatas"][0] if dense_res.get("metadatas") else []
        dists = dense_res["distances"][0] if dense_res.get("distances") else []
        ids = dense_res["ids"][0] if dense_res.get("ids") else []

        for i in range(len(docs)):
            c_id = ids[i] if i < len(ids) else ""
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            dense_sim = round(max(0.0, 1.0 - dist), 4)

            candidates_map[c_id] = {
                "chunk_id": c_id,
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "page_number": int(meta.get("page_number", 1)),
                "chunk_number": int(meta.get("chunk_number", 1)),
                "text": docs[i],
                "dense_distance": round(dist, 4),
                "dense_similarity": dense_sim,
            }

    # 2. In document-scoped mode or for small/medium collections, also fetch all chunks
    # belonging to the target scope to evaluate keyword BM25 matches that dense search might miss.
    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if document_id:
        get_kwargs["where"] = {"document_id": document_id}

    # For scoped search, always get all document chunks (fast and guarantees full document coverage).
    # For global search, get all chunks if collection count <= 300.
    if document_id or count <= 300:
        doc_items = collection.get(**get_kwargs)
        all_docs = doc_items.get("documents", [])
        all_metas = doc_items.get("metadatas", [])
        all_ids = doc_items.get("ids", [])

        for j in range(len(all_docs)):
            c_id = all_ids[j] if j < len(all_ids) else ""
            if c_id not in candidates_map:
                meta = all_metas[j] if j < len(all_metas) else {}
                candidates_map[c_id] = {
                    "chunk_id": c_id,
                    "document_id": meta.get("document_id", ""),
                    "filename": meta.get("filename", ""),
                    "page_number": int(meta.get("page_number", 1)),
                    "chunk_number": int(meta.get("chunk_number", 1)),
                    "text": all_docs[j],
                    "dense_distance": 1.0,
                    "dense_similarity": 0.0,
                }

    candidate_list = list(candidates_map.values())
    if not candidate_list:
        return []

    # 3. Compute lexical scores across candidate chunks
    lex_scores = compute_lexical_score(clean_query, candidate_list)

    # 4. Combine semantic and lexical scores
    scored_results: list[dict[str, Any]] = []
    for chunk, lex_score in zip(candidate_list, lex_scores):
        dense_sim = chunk.get("dense_similarity", 0.0)
        combined_sim = combine_hybrid_scores(dense_sim, lex_score)
        combined_dist = round(max(0.0, 1.0 - combined_sim), 4)

        scored_results.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"],
            "distance": combined_dist,
            "similarity_score": combined_sim,
            "lexical_score": lex_score,
            "dense_similarity": dense_sim,
        })

    # 5. Rank by similarity_score descending, with lexical and dense scores as tie-breakers
    scored_results.sort(
        key=lambda r: (r["similarity_score"], r["lexical_score"], r["dense_similarity"]),
        reverse=True
    )

    return scored_results[:top_k]


def get_indexed_documents(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None
) -> list[dict[str, Any]]:
    """Retrieves metadata summary for all unique documents indexed in ChromaDB."""
    collection = get_collection(collection_name=collection_name, persist_directory=persist_directory)
    if collection.count() == 0:
        return []

    items = collection.get(include=["metadatas"])
    metas = items.get("metadatas", [])
    docs_map: dict[str, dict[str, Any]] = {}

    for meta in metas:
        fname = meta.get("filename", "")
        doc_id = meta.get("document_id", "")
        page_num = meta.get("page_number", 1)
        if not fname:
            continue
        if fname not in docs_map:
            docs_map[fname] = {
                "document_id": doc_id,
                "filename": fname,
                "pages": set(),
                "chunk_count": 0,
            }
        docs_map[fname]["pages"].add(page_num)
        docs_map[fname]["chunk_count"] += 1

    result: list[dict[str, Any]] = []
    for fname, data in docs_map.items():
        result.append({
            "document_id": data["document_id"],
            "filename": fname,
            "total_pages": len(data["pages"]),
            "total_chunks": data["chunk_count"],
        })
    return result


def get_document_chunks(
    document_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None
) -> list[dict[str, Any]]:
    """Retrieves all indexed text chunks belonging to a document from ChromaDB.

    Returns the chunks sorted by page_number and chunk_number.
    """
    collection = get_collection(collection_name=collection_name, persist_directory=persist_directory)
    if collection.count() == 0:
        return []

    items = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"]
    )
    docs = items.get("documents", [])
    metas = items.get("metadatas", [])
    ids = items.get("ids", [])

    chunks: list[dict[str, Any]] = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        chunk_id = ids[i] if i < len(ids) else ""
        chunks.append({
            "chunk_id": chunk_id,
            "document_id": meta.get("document_id", document_id),
            "filename": meta.get("filename", ""),
            "page_number": int(meta.get("page_number", 1)),
            "chunk_number": int(meta.get("chunk_number", 1)),
            "text": docs[i],
            "distance": 0.0,
            "similarity_score": 1.0,
        })

    # Sort sequentially in document order
    chunks.sort(key=lambda c: (c["page_number"], c["chunk_number"]))
    return chunks


def select_representative_chunks(
    chunks: list[dict[str, Any]],
    max_chunks: int = 8,
) -> list[dict[str, Any]]:
    """Selects chunks distributed evenly across the entire document.

    If total chunks <= max_chunks, returns all chunks.
    Otherwise, samples chunks evenly across the document from start to finish
    (including the beginning, middle sections, and conclusion).
    """
    if len(chunks) <= max_chunks:
        return chunks

    indices: list[int] = []
    step = (len(chunks) - 1) / (max_chunks - 1)
    for i in range(max_chunks):
        idx = int(round(i * step))
        if idx not in indices and 0 <= idx < len(chunks):
            indices.append(idx)

    indices = sorted(set(indices))
    return [chunks[idx] for idx in indices]


def delete_document(
    filename: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None
) -> int:
    """Deletes all chunks belonging to the given filename from ChromaDB."""
    collection = get_collection(collection_name=collection_name, persist_directory=persist_directory)
    existing = collection.get(where={"filename": filename})
    count = len(existing.get("ids", []))
    if count > 0:
        collection.delete(where={"filename": filename})
    return count
