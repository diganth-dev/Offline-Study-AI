"""Hugging Face Embedding Service for Offline Study AI.

Loads sentence-transformers/all-MiniLM-L6-v2 on CPU and generates
normalized embeddings for document chunks and search queries.
"""

from typing import Optional
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Returns the cached SentenceTransformer model instance on CPU.

    Loads the model on first call and reuses it for subsequent calls
    to keep memory footprint minimal on 8 GB RAM systems.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model


def generate_embeddings(
    texts: list[str],
    batch_size: int = 32
) -> list[list[float]]:
    """Generates normalized vector embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for CPU inference.

    Returns:
        List of 384-dimensional float embeddings.
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def generate_query_embedding(query: str) -> list[float]:
    """Generates a normalized vector embedding for a single search query string.

    Args:
        query: Search query text.

    Returns:
        384-dimensional float embedding.
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()
