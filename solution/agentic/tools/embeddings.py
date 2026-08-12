"""Shared embedding + similarity helpers.

Used by both long-term memory (`memory_tools.py`) and knowledge-base search
(`knowledge_tools.py`) so there's exactly one place that talks to the
embeddings API and one place that implements cosine-similarity ranking.

Every function accepts an `embed_fn` override so callers -- and tests --
never *need* a live OpenAI API key: the default client is constructed lazily,
only the first time an embedding is actually requested with no override.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Sequence

# A batch-embedding function: list of texts in, one vector per text out.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]

DEFAULT_MODEL = "text-embedding-3-small"


@lru_cache(maxsize=1)
def _default_embed_fn() -> EmbedFn:
    """Build (once) the real OpenAI embeddings client.

    Deferred so importing this module, or anything built on it, never
    requires OPENAI_API_KEY unless an embedding is actually requested.
    """
    from langchain_openai import OpenAIEmbeddings

    client = OpenAIEmbeddings(model=DEFAULT_MODEL)
    return client.embed_documents


def embed_texts(texts: Sequence[str], embed_fn: EmbedFn | None = None) -> list[list[float]]:
    """Embed a batch of texts. Empty input returns an empty list."""
    if not texts:
        return []
    fn = embed_fn or _default_embed_fn()
    return fn(list(texts))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, in [-1, 1].

    Returns 0.0 if either vector has zero magnitude, rather than raising a
    divide-by-zero error.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(
    query_vector: Sequence[float],
    candidates: Sequence[tuple[Any, Sequence[float]]],
    top_k: int = 3,
) -> list[tuple[Any, float]]:
    """Rank (item, vector) candidates by cosine similarity to query_vector.

    Args:
        query_vector: The embedded query.
        candidates: (item, vector) pairs -- `item` can be anything (an ORM
            row, a dict, an id); it's returned as-is alongside its score.
        top_k: Max number of results to return.

    Returns:
        Up to `top_k` (item, score) pairs, highest score first.
    """
    scored = [(item, cosine_similarity(query_vector, vector)) for item, vector in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
