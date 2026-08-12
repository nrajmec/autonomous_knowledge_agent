"""Tests for agentic/tools/knowledge_tools.py.

Same isolation pattern as test_memory_tools.py: a throwaway SQLite DB and a
deterministic fake embedder, so these need neither the seeded project data
nor OPENAI_API_KEY.
"""
import pytest
from sqlalchemy import create_engine

import agentic.tools.knowledge_tools as knowledge_tools
from data.models import udahub
from utils import get_session

_VOCAB = ["refund", "login", "password", "email", "premium", "cancel", "reservation"]


def _fake_embed_texts(texts):
    return [[1.0 if word in t.lower() else 0.0 for word in _VOCAB] for t in texts]


@pytest.fixture
def temp_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_udahub.db'}")
    udahub.Base.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def _patch_engine_and_embeddings(monkeypatch, temp_engine):
    monkeypatch.setattr(knowledge_tools, "get_udahub_engine", lambda: temp_engine)
    monkeypatch.setattr(knowledge_tools, "embed_texts", _fake_embed_texts)
    knowledge_tools.invalidate_knowledge_cache()
    yield
    knowledge_tools.invalidate_knowledge_cache()


@pytest.fixture
def seeded_articles(temp_engine):
    with get_session(temp_engine) as session:
        session.add(udahub.Account(account_id="acc1", account_name="Test Account"))
        session.add(
            udahub.Knowledge(
                article_id="a-login",
                account_id="acc1",
                title="How to Reset Your Password",
                content="If you can't log in, use the password reset link on the login page.",
                tags="login, password, access",
            )
        )
        session.add(
            udahub.Knowledge(
                article_id="a-refund",
                account_id="acc1",
                title="Requesting a Refund",
                content="Refunds are issued for cancelled events within 14 days.",
                tags="refund, cancel, billing",
            )
        )
    return "acc1"


def test_search_returns_relevant_article_first(seeded_articles):
    result = knowledge_tools.search_knowledge_base(
        "I forgot my password and can't login", seeded_articles
    )

    assert result["ok"] is True
    assert result["data"][0]["article_id"] == "a-login"
    assert result["relevant"] is True


def test_search_respects_top_k(seeded_articles):
    result = knowledge_tools.search_knowledge_base("login", seeded_articles, top_k=1)

    assert len(result["data"]) == 1


def test_search_caches_article_embeddings_across_calls(seeded_articles, monkeypatch):
    calls = []

    def counting_embed_texts(texts):
        calls.append(list(texts))
        return _fake_embed_texts(texts)

    monkeypatch.setattr(knowledge_tools, "embed_texts", counting_embed_texts)

    knowledge_tools.search_knowledge_base("login", seeded_articles)
    knowledge_tools.search_knowledge_base("password", seeded_articles)

    # The 2-article batch should only be embedded once; each search also
    # embeds its 1-word query, but that shouldn't re-trigger the batch.
    article_batch_calls = [c for c in calls if len(c) > 1]
    assert len(article_batch_calls) == 1


def test_search_flags_not_relevant_when_no_article_matches(seeded_articles):
    result = knowledge_tools.search_knowledge_base("something about premium upgrades", seeded_articles)

    assert result["ok"] is True
    assert result["relevant"] is False


def test_search_unknown_account_returns_empty_not_relevant():
    result = knowledge_tools.search_knowledge_base("login", "no-such-account")

    assert result == {"ok": True, "data": [], "relevant": False}


def test_search_requires_query(seeded_articles):
    result = knowledge_tools.search_knowledge_base("", seeded_articles)

    assert result == {"ok": False, "error": "query is required"}


def test_search_requires_account_id():
    result = knowledge_tools.search_knowledge_base("login", "")

    assert result == {"ok": False, "error": "account_id is required"}


def test_invalidate_knowledge_cache_forces_refetch(seeded_articles, temp_engine):
    knowledge_tools.search_knowledge_base("login", seeded_articles)  # warms the cache

    with get_session(temp_engine) as session:
        session.add(
            udahub.Knowledge(
                article_id="a-new",
                account_id=seeded_articles,
                title="Managing Email Preferences",
                content="Update your notification email preferences in account settings.",
                tags="email, notifications",
            )
        )

    stale = knowledge_tools.search_knowledge_base("email preferences", seeded_articles)
    assert all(r["article_id"] != "a-new" for r in stale["data"])

    knowledge_tools.invalidate_knowledge_cache(seeded_articles)
    fresh = knowledge_tools.search_knowledge_base("email preferences", seeded_articles)
    assert fresh["data"][0]["article_id"] == "a-new"
