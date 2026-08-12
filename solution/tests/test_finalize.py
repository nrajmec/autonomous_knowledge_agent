"""Tests for agentic/agents/finalize.py."""
import agentic.agents.finalize as finalize


def test_finalize_resolved_persists_message_and_saves_memory(monkeypatch):
    calls = {}

    def fake_update(ticket_id, **kwargs):
        calls["update"] = {"ticket_id": ticket_id, **kwargs}
        return {"ok": True, "data": {}}

    def fake_save(user_id, account_id, memory_type, content):
        calls["save"] = (user_id, account_id, memory_type, content)
        return {"ok": True, "data": {"memory_id": "m1"}}

    monkeypatch.setattr(finalize, "update_ticket_record", fake_update)
    monkeypatch.setattr(finalize, "save_customer_memory", fake_save)

    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "escalation_needed": False,
        "draft_response": "Try resetting your password.",
        "user_context": {"internal_user_id": "user1"},
    }

    result = finalize.finalize_node(state)

    assert result["final_status"] == "resolved"
    assert calls["update"] == {
        "ticket_id": "t1",
        "status": "resolved",
        "message_role": "ai",
        "message_content": "Try resetting your password.",
    }
    assert calls["save"][0] == "user1"
    assert calls["save"][2] == "resolution_summary"
    assert result["trace"][0]["memory_saved"] is True


def test_finalize_escalated_does_not_save_memory(monkeypatch):
    calls = {}
    save_called = {"called": False}

    def fake_update(ticket_id, **kwargs):
        calls["update"] = {"ticket_id": ticket_id, **kwargs}
        return {"ok": True, "data": {}}

    def fake_save(*args, **kwargs):
        save_called["called"] = True
        return {"ok": True, "data": {}}

    monkeypatch.setattr(finalize, "update_ticket_record", fake_update)
    monkeypatch.setattr(finalize, "save_customer_memory", fake_save)

    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "escalation_needed": True,
        "escalation_summary": "Escalated: blocked account.",
        "user_context": {"internal_user_id": "user1"},
    }

    result = finalize.finalize_node(state)

    assert result["final_status"] == "escalated"
    assert calls["update"]["status"] == "escalated"
    assert calls["update"]["message_content"] == "Escalated: blocked account."
    assert save_called["called"] is False


def test_finalize_without_internal_user_id_skips_memory_save(monkeypatch):
    save_called = {"called": False}

    def fake_save(*args, **kwargs):
        save_called["called"] = True
        return {"ok": True, "data": {}}

    monkeypatch.setattr(finalize, "update_ticket_record", lambda ticket_id, **kwargs: {"ok": True, "data": {}})
    monkeypatch.setattr(finalize, "save_customer_memory", fake_save)

    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "escalation_needed": False,
        "draft_response": "Resolved.",
        "user_context": {},
    }

    finalize.finalize_node(state)

    assert save_called["called"] is False


def test_finalize_saves_detected_preference_alongside_resolution_summary(monkeypatch):
    saved = []

    monkeypatch.setattr(finalize, "update_ticket_record", lambda ticket_id, **kwargs: {"ok": True, "data": {}})
    monkeypatch.setattr(
        finalize,
        "save_customer_memory",
        lambda user_id, account_id, memory_type, content: saved.append((user_id, memory_type, content))
        or {"ok": True, "data": {}},
    )

    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "escalation_needed": False,
        "draft_response": "Upgraded your plan.",
        "detected_preference": "Prefers email over phone contact",
        "user_context": {"internal_user_id": "user1"},
    }

    result = finalize.finalize_node(state)

    memory_types_saved = {call[1] for call in saved}
    assert memory_types_saved == {"resolution_summary", "preference"}
    assert ("user1", "preference", "Prefers email over phone contact") in saved
    assert result["trace"][0]["preference_saved"] is True


def test_finalize_saves_preference_even_when_ticket_is_escalated(monkeypatch):
    saved = []

    monkeypatch.setattr(finalize, "update_ticket_record", lambda ticket_id, **kwargs: {"ok": True, "data": {}})
    monkeypatch.setattr(
        finalize,
        "save_customer_memory",
        lambda user_id, account_id, memory_type, content: saved.append((user_id, memory_type, content))
        or {"ok": True, "data": {}},
    )

    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "escalation_needed": True,
        "escalation_summary": "Escalated: needs manual review.",
        "detected_preference": "Always wants the cheapest tier",
        "user_context": {"internal_user_id": "user1"},
    }

    result = finalize.finalize_node(state)

    # No resolution_summary (ticket wasn't resolved), but the preference is
    # about the customer, not this ticket's outcome, so it's still saved.
    assert saved == [("user1", "preference", "Always wants the cheapest tier")]
    assert result["trace"][0]["memory_saved"] is False
    assert result["trace"][0]["preference_saved"] is True


def test_finalize_skips_preference_save_when_none_detected(monkeypatch):
    saved = []

    monkeypatch.setattr(finalize, "update_ticket_record", lambda ticket_id, **kwargs: {"ok": True, "data": {}})
    monkeypatch.setattr(
        finalize,
        "save_customer_memory",
        lambda user_id, account_id, memory_type, content: saved.append(memory_type) or {"ok": True, "data": {}},
    )

    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "escalation_needed": False,
        "draft_response": "Resolved.",
        "user_context": {"internal_user_id": "user1"},
    }

    finalize.finalize_node(state)

    assert "preference" not in saved


def test_finalize_escalated_falls_back_to_default_message_when_no_reason_given(monkeypatch):
    calls = {}

    def fake_update(ticket_id, **kwargs):
        calls["update"] = kwargs
        return {"ok": True, "data": {}}

    monkeypatch.setattr(finalize, "update_ticket_record", fake_update)
    monkeypatch.setattr(finalize, "save_customer_memory", lambda *a, **k: {"ok": True, "data": {}})

    state = {"ticket_id": "t1", "account_id": "acc1", "escalation_needed": True, "user_context": {}}

    finalize.finalize_node(state)

    assert calls["update"]["message_content"] == "This ticket has been escalated to a human agent."
