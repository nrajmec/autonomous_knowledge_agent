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


def test_context_loader_resets_stale_per_turn_state_from_a_previous_turn(monkeypatch):
    """Regression test: within one thread_id, LangGraph's checkpointer
    carries every TicketState field over to the next invoke() call verbatim
    unless a node overwrites it -- only messages/trace have reducers. A
    second turn that inherited a prior turn's leftover confidence would make
    Supervisor think a resolver already ran for THIS turn and skip straight
    to Finalize/Escalation. Context Loader runs first on every turn, so it's
    responsible for resetting all of that per-turn working state."""
    monkeypatch.setattr(context_loader, "get_customer_profile", lambda uid: {"ok": True, "data": {}})
    monkeypatch.setattr(context_loader, "get_subscription_status", lambda uid: {"ok": True, "data": None})
    monkeypatch.setattr(
        context_loader,
        "get_ticket_history",
        lambda acc, ext, exclude_ticket_id=None: {"ok": True, "data": []},
    )
    monkeypatch.setattr(context_loader, "get_internal_user_id", lambda acc, ext: {"ok": True, "data": None})
    monkeypatch.setattr(context_loader, "recall_customer_memory", lambda *a, **k: {"ok": True, "data": []})

    # Simulates state as LangGraph would hand it to context_loader on a
    # SECOND turn: everything left over from turn 1's finalize.
    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "external_user_id": "ext1",
        "ticket_text": "a new message this turn",
        "classification": {"category": "billing", "hard_escalate": False},
        "confidence": 0.9,
        "draft_response": "turn 1's stale answer",
        "cited_article_ids": ["a-old"],
        "escalation_needed": False,
        "escalation_reason": None,
        "escalation_summary": None,
        "detected_preference": "turn 1's stale preference",
        "final_status": "resolved",
        "route": "finalize",
    }

    result = context_loader.context_loader_node(state)

    assert result["confidence"] is None
    assert result["draft_response"] is None
    assert result["cited_article_ids"] == []
    assert result["escalation_needed"] is False
    assert result["escalation_reason"] is None
    assert result["detected_preference"] is None
    assert result["classification"] == {}
    assert result["final_status"] is None
    assert result["route"] is None


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
