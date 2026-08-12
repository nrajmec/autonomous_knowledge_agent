"""Tests for agentic/agents/context_loader.py.

The underlying tool functions are monkeypatched (this node's job is just to
assemble their results into `user_context`, not to re-verify the tools
themselves -- those have their own test files).
"""
import agentic.agents.context_loader as context_loader


def test_context_loader_assembles_user_context(monkeypatch):
    monkeypatch.setattr(
        context_loader,
        "get_customer_profile",
        lambda uid: {"ok": True, "data": {"user_id": uid, "is_blocked": False}},
    )
    monkeypatch.setattr(
        context_loader,
        "get_subscription_status",
        lambda uid: {"ok": True, "data": {"tier": "premium"}},
    )
    monkeypatch.setattr(
        context_loader,
        "get_ticket_history",
        lambda acc, ext, exclude_ticket_id=None: {"ok": True, "data": [{"ticket_id": "old1"}]},
    )
    monkeypatch.setattr(
        context_loader, "get_internal_user_id", lambda acc, ext: {"ok": True, "data": "user1"}
    )
    monkeypatch.setattr(
        context_loader,
        "recall_customer_memory",
        lambda uid, query, top_k=3: {"ok": True, "data": [{"content": "Prefers email"}]},
    )

    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "external_user_id": "ext1",
        "ticket_text": "I can't log in",
    }

    result = context_loader.context_loader_node(state)

    ctx = result["user_context"]
    assert ctx["profile"]["is_blocked"] is False
    assert ctx["subscription"]["tier"] == "premium"
    assert ctx["ticket_history"] == [{"ticket_id": "old1"}]
    assert ctx["long_term_memories"] == [{"content": "Prefers email"}]
    assert ctx["internal_user_id"] == "user1"
    assert len(result["trace"]) == 1
    assert result["trace"][0]["node"] == "context_loader"


def test_context_loader_handles_unknown_customer_gracefully(monkeypatch):
    monkeypatch.setattr(
        context_loader, "get_customer_profile", lambda uid: {"ok": False, "error": "not found"}
    )
    monkeypatch.setattr(
        context_loader, "get_subscription_status", lambda uid: {"ok": False, "error": "not found"}
    )
    monkeypatch.setattr(
        context_loader,
        "get_ticket_history",
        lambda acc, ext, exclude_ticket_id=None: {"ok": True, "data": []},
    )
    monkeypatch.setattr(
        context_loader, "get_internal_user_id", lambda acc, ext: {"ok": True, "data": None}
    )
    called = {"recall": False}

    def fake_recall(*args, **kwargs):
        called["recall"] = True
        return {"ok": True, "data": []}

    monkeypatch.setattr(context_loader, "recall_customer_memory", fake_recall)

    state = {"ticket_id": "t1", "account_id": "acc1", "external_user_id": "ext1", "ticket_text": "hi"}

    result = context_loader.context_loader_node(state)

    ctx = result["user_context"]
    assert ctx["profile"] is None
    assert ctx["subscription"] is None
    assert ctx["internal_user_id"] is None
    # No internal_user_id (brand-new customer) -> nothing to recall from.
    assert called["recall"] is False


def test_context_loader_skips_recall_when_no_ticket_text(monkeypatch):
    monkeypatch.setattr(context_loader, "get_customer_profile", lambda uid: {"ok": True, "data": {}})
    monkeypatch.setattr(context_loader, "get_subscription_status", lambda uid: {"ok": True, "data": None})
    monkeypatch.setattr(
        context_loader,
        "get_ticket_history",
        lambda acc, ext, exclude_ticket_id=None: {"ok": True, "data": []},
    )
    monkeypatch.setattr(
        context_loader, "get_internal_user_id", lambda acc, ext: {"ok": True, "data": "user1"}
    )
    called = {"recall": False}

    def fake_recall(*args, **kwargs):
        called["recall"] = True
        return {"ok": True, "data": []}

    monkeypatch.setattr(context_loader, "recall_customer_memory", fake_recall)

    state = {"ticket_id": "t1", "account_id": "acc1", "external_user_id": "ext1", "ticket_text": "   "}

    context_loader.context_loader_node(state)

    assert called["recall"] is False
