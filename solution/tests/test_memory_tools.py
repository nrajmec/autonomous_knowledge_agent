"""Tests for agentic/tools/memory_tools.py.

Runs against a throwaway SQLite DB (via `temp_engine`, not the real
solution/data/core/udahub.db) and a deterministic fake embedder (via
`fake_embeddings`), so these tests need neither the seeded project data nor
an OPENAI_API_KEY.
"""
import pytest
from sqlalchemy import create_engine

import agentic.tools.memory_tools as memory_tools
from data.models import udahub
from utils import get_session

_VOCAB = ["refund", "login", "password", "email", "premium", "cancel"]


def _fake_embed_texts(texts, embed_fn=None):
    return [[1.0 if word in t.lower() else 0.0 for word in _VOCAB] for t in texts]


@pytest.fixture
def temp_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_udahub.db'}")
    udahub.Base.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def _patch_engine_and_embeddings(monkeypatch, temp_engine):
    monkeypatch.setattr(memory_tools, "get_udahub_engine", lambda: temp_engine)
    monkeypatch.setattr(memory_tools, "embed_texts", _fake_embed_texts)


@pytest.fixture
def seeded_user(temp_engine):
    with get_session(temp_engine) as session:
        session.add(udahub.Account(account_id="acc1", account_name="Test Account"))
        session.add(
            udahub.User(
                user_id="user1", account_id="acc1", external_user_id="ext1", user_name="Test User"
            )
        )
    return "user1", "acc1"


def test_save_customer_memory_returns_memory_id(seeded_user):
    user_id, account_id = seeded_user

    result = memory_tools.save_customer_memory(
        user_id, account_id, "preference", "Prefers email over chat"
    )

    assert result["ok"] is True
    assert result["data"]["memory_id"]


def test_recall_customer_memory_ranks_relevant_entry_first(seeded_user):
    user_id, account_id = seeded_user
    memory_tools.save_customer_memory(user_id, account_id, "preference", "Prefers email over chat")
    memory_tools.save_customer_memory(
        user_id, account_id, "resolution_summary", "Resolved a login password reset"
    )

    result = memory_tools.recall_customer_memory(user_id, "What is their email preference?")

    assert result["ok"] is True
    assert result["data"][0]["content"] == "Prefers email over chat"
    assert result["data"][0]["score"] == 1.0


def test_recall_customer_memory_respects_top_k(seeded_user):
    user_id, account_id = seeded_user
    for i in range(5):
        memory_tools.save_customer_memory(user_id, account_id, "preference", f"note about login {i}")

    result = memory_tools.recall_customer_memory(user_id, "login", top_k=2)

    assert len(result["data"]) == 2


def test_recall_customer_memory_unknown_user_returns_empty_list_not_error():
    result = memory_tools.recall_customer_memory("no-such-user", "anything")

    assert result == {"ok": True, "data": []}


def test_save_customer_memory_requires_user_and_account_id():
    result = memory_tools.save_customer_memory("", "acc1", "preference", "content")

    assert result == {"ok": False, "error": "user_id and account_id are required"}


def test_save_customer_memory_rejects_invalid_memory_type(seeded_user):
    user_id, account_id = seeded_user

    result = memory_tools.save_customer_memory(user_id, account_id, "bogus_type", "content")

    assert result["ok"] is False
    assert "memory_type must be one of" in result["error"]


def test_save_customer_memory_requires_content(seeded_user):
    user_id, account_id = seeded_user

    result = memory_tools.save_customer_memory(user_id, account_id, "preference", "   ")

    assert result == {"ok": False, "error": "content is required"}


def test_recall_customer_memory_requires_query():
    result = memory_tools.recall_customer_memory("user1", "")

    assert result == {"ok": False, "error": "query is required"}
