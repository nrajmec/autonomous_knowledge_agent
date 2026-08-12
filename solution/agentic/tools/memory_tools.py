"""Long-term (cross-session) memory: customer preferences and resolved-issue
summaries that must survive process/kernel restarts and outlive any single
ticket's `thread_id`.

This is deliberately separate from LangGraph's own checkpointer, which is
session-scoped (see `chat_interface` in `utils.py`, keyed by `thread_id`).
Long-term memory here is keyed by UDA-Hub's internal `user_id` and backed by
the `customer_memory` table. Retrieval is semantic: each entry is embedded
once when saved (`embeddings.embed_texts`), and recall ranks stored entries
by cosine similarity to the query (`embeddings.rank_by_similarity`).
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from data.models import udahub
from utils import get_session

from agentic.tools.db import get_udahub_engine
from agentic.tools.embeddings import embed_texts, rank_by_similarity

_VALID_MEMORY_TYPES = {"preference", "resolution_summary"}


def save_customer_memory(
    user_id: str, account_id: str, memory_type: str, content: str
) -> dict[str, Any]:
    """Persist a long-term memory entry for a customer.

    Args:
        user_id: UDA-Hub `User.user_id` (the internal id -- not the CultPass
            `external_user_id`).
        account_id: UDA-Hub `Account.account_id`.
        memory_type: "preference" or "resolution_summary".
        content: Free-text memory, e.g. "Prefers email over chat" or
            "Resolved: reset password via emailed reset link".

    Returns:
        {"ok": True, "data": {"memory_id"}} on success, or
        {"ok": False, "error": "<reason>"} on validation failure.
    """
    if not user_id or not account_id:
        return {"ok": False, "error": "user_id and account_id are required"}
    if memory_type not in _VALID_MEMORY_TYPES:
        return {
            "ok": False,
            "error": f"memory_type must be one of {sorted(_VALID_MEMORY_TYPES)}, got '{memory_type}'",
        }
    if not content or not content.strip():
        return {"ok": False, "error": "content is required"}

    [vector] = embed_texts([content])

    with get_session(get_udahub_engine()) as session:
        memory = udahub.CustomerMemory(
            memory_id=str(uuid.uuid4()),
            account_id=account_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=json.dumps(vector),
        )
        session.add(memory)
        session.flush()
        return {"ok": True, "data": {"memory_id": memory.memory_id}}


def recall_customer_memory(user_id: str, query: str, top_k: int = 3) -> dict[str, Any]:
    """Semantically search a customer's long-term memory.

    Args:
        user_id: UDA-Hub `User.user_id`.
        query: Free-text query, e.g. the current ticket's subject.
        top_k: Max number of memories to return.

    Returns:
        {"ok": True, "data": [{"memory_id", "memory_type", "content",
        "created_at", "score"}, ...]}, ranked by similarity, highest first.
        An unknown customer or one with no memories yet is not an error --
        {"ok": True, "data": []} is returned.
    """
    if not user_id:
        return {"ok": False, "error": "user_id is required"}
    if not query or not query.strip():
        return {"ok": False, "error": "query is required"}

    with get_session(get_udahub_engine()) as session:
        memories = session.query(udahub.CustomerMemory).filter_by(user_id=user_id).all()
        if not memories:
            return {"ok": True, "data": []}

        [query_vector] = embed_texts([query])
        candidates = [(m, json.loads(m.embedding)) for m in memories]
        ranked = rank_by_similarity(query_vector, candidates, top_k=top_k)

        return {
            "ok": True,
            "data": [
                {
                    "memory_id": m.memory_id,
                    "memory_type": m.memory_type,
                    "content": m.content,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "score": round(score, 4),
                }
                for m, score in ranked
            ],
        }
