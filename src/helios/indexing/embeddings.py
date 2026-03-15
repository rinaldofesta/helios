"""Embedding generation via Ollama and vector similarity helpers."""

from __future__ import annotations

import logging
import math
import struct

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 50


# --------------------------------------------------------------------------- #
#  Serialization
# --------------------------------------------------------------------------- #


def serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float32 vector to bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_f32(data: bytes) -> list[float]:
    """Deserialize bytes to a float32 vector."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# --------------------------------------------------------------------------- #
#  Similarity
# --------------------------------------------------------------------------- #


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------- #
#  Embedding generation
# --------------------------------------------------------------------------- #


async def is_ollama_available(host: str | None = None) -> bool:
    """Check if Ollama is running and reachable."""
    try:
        import ollama as ollama_lib
        client = ollama_lib.AsyncClient(host=host) if host else ollama_lib.AsyncClient()
        await client.list()
        return True
    except Exception:
        return False


async def ensure_embed_model(
    model: str = DEFAULT_EMBED_MODEL,
    host: str | None = None,
) -> bool:
    """Ensure the embedding model is available, pulling if needed.

    On first use, will auto-pull the model (~274MB for nomic-embed-text).
    Logs a clear message so the user knows what's happening.
    """
    try:
        import ollama as ollama_lib
        client = ollama_lib.AsyncClient(host=host) if host else ollama_lib.AsyncClient()
        try:
            await client.show(model)
            return True
        except Exception:
            logger.warning(
                "Embedding model '%s' not found locally. "
                "Pulling now (~274MB for nomic-embed-text). "
                "This is a one-time download.",
                model,
            )
            await client.pull(model)
            logger.info("Embedding model '%s' ready.", model)
            return True
    except Exception as e:
        logger.warning("Failed to ensure embedding model %s: %s", model, e)
        return False


async def generate_embeddings(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
    host: str | None = None,
) -> list[list[float]] | None:
    """Generate embeddings for a list of texts using Ollama.

    Returns None if Ollama is not available or embedding fails.
    Processes in batches for efficiency.
    """
    if not texts:
        return []

    try:
        import ollama as ollama_lib
    except ImportError:
        logger.warning("ollama package not installed, skipping embeddings")
        return None

    client = ollama_lib.AsyncClient(host=host) if host else ollama_lib.AsyncClient()

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = await client.embed(model=model, input=batch)
            all_embeddings.extend(response.embeddings)
        except Exception as e:
            logger.warning("Embedding batch %d failed: %s", i // batch_size, e)
            return None

    return all_embeddings


async def get_query_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    host: str | None = None,
) -> list[float] | None:
    """Generate a single embedding for a search query."""
    result = await generate_embeddings([text], model=model, host=host)
    if result and len(result) == 1:
        return result[0]
    return None
