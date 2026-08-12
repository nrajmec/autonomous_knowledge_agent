"""Knowledge-base retrieval (RAG) over UDA-Hub's `Knowledge` articles.

How it works: every article's `title + content + tags` is embedded once
(text-embedding-3-small, via `embeddings.embed_texts`) and cached in memory,
keyed by `account_id` -- the CultPass articles are effectively static
reference content, so re-embedding all of them on every single search would
just be wasted latency and API cost. A search embeds the query and ranks
cached article vectors by cosine similarity (`embeddings.rank_by_similarity`).

The cache is process-lifetime only (a plain module-level dict, no
persistence) and rebuilds itself automatically the first time any account is
searched after startup. Call `invalidate_knowledge_cache()` after writing
new/edited articles, or the next search will keep returning stale ones.

Relevance gating: alongside the ranked results, the response includes a
`relevant` flag -- False when even the top match's score falls below
`RELEVANCE_THRESHOLD`. This is deliberately separate from a resolver
agent's own self-reported confidence: it's meant to be the search tool's
own, non-negotiable signal that "no genuinely applicable article exists",
which the calling resolver should treat as a strong escalation trigger
regardless of how confident its drafted answer sounds. The threshold value
is a starting point, not an empirically calibrated one -- there's no
OPENAI_API_KEY in this dev environment to tune it against real embeddings,
so revisit it once real usage/API access is available.
"""
from __future__ import annotations

from typing import Any

from data.models import udahub
from utils import get_session

from agentic.tools.db import get_udahub_engine
from agentic.tools.embeddings import embed_texts, rank_by_similarity

RELEVANCE_THRESHOLD = 0.35

# account_id -> [(article_dict, embedding_vector), ...]
_cache: dict[str, list[tuple[dict[str, Any], list[float]]]] = {}


def invalidate_knowledge_cache(account_id: str | None = None) -> None:
    """Drop cached article embeddings so the next search re-embeds them.

    Args:
        account_id: Clear just this account's cache entry, or every account
            if omitted (also useful for isolating tests from each other).
    """
    if account_id is None:
        _cache.clear()
    else:
        _cache.pop(account_id, None)


def _load_articles(account_id: str) -> list[tuple[dict[str, Any], list[float]]]:
    if account_id in _cache:
        return _cache[account_id]

    with get_session(get_udahub_engine()) as session:
        articles = session.query(udahub.Knowledge).filter_by(account_id=account_id).all()
        article_dicts = [
            {
                "article_id": a.article_id,
                "title": a.title,
                "content": a.content,
                "tags": a.tags,
            }
            for a in articles
        ]

    if not article_dicts:
        _cache[account_id] = []
        return _cache[account_id]

    texts = [f"{a['title']}\n\n{a['content']}\n\nTags: {a['tags']}" for a in article_dicts]
    vectors = embed_texts(texts)
    _cache[account_id] = list(zip(article_dicts, vectors))
    return _cache[account_id]


def search_knowledge_base(query: str, account_id: str, top_k: int = 3) -> dict[str, Any]:
    """Semantically search an account's knowledge base articles.

    Args:
        query: The customer's issue/question text to search with.
        account_id: UDA-Hub `Account.account_id` (e.g. "cultpass").
        top_k: Max number of articles to return.

    Returns:
        {"ok": True, "data": [{"article_id", "title", "content", "tags",
        "score"}, ...], "relevant": bool} on success. An account with no
        articles yet returns {"ok": True, "data": [], "relevant": False}.
        {"ok": False, "error": "<reason>"} on validation failure.
    """
    if not query or not query.strip():
        return {"ok": False, "error": "query is required"}
    if not account_id:
        return {"ok": False, "error": "account_id is required"}

    candidates = _load_articles(account_id)
    if not candidates:
        return {"ok": True, "data": [], "relevant": False}

    [query_vector] = embed_texts([query])
    ranked = rank_by_similarity(query_vector, candidates, top_k=top_k)

    results = [{**article, "score": round(score, 4)} for article, score in ranked]
    relevant = results[0]["score"] >= RELEVANCE_THRESHOLD

    return {"ok": True, "data": results, "relevant": relevant}
