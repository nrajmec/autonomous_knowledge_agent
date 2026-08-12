"""Tests for agentic/agents/supervisor.py -- pure rule-based routing, no LLM."""
import agentic.agents.supervisor as supervisor


def test_first_pass_routes_by_category():
    state = {"ticket_id": "t1", "classification": {"category": "billing", "hard_escalate": False}}

    result = supervisor.supervisor_node(state)

    assert result["route"] == "billing_resolver"
    assert supervisor.route_from_supervisor({**state, **result}) == "billing_resolver"


def test_first_pass_falls_back_to_general_resolver_for_unknown_category():
    state = {"ticket_id": "t1", "classification": {"category": "something-else", "hard_escalate": False}}

    result = supervisor.supervisor_node(state)

    assert result["route"] == "general_resolver"


def test_first_pass_bypasses_to_escalation_on_hard_escalate():
    state = {
        "ticket_id": "t1",
        "classification": {"category": "account", "hard_escalate": True, "hard_escalate_reason": "Account blocked"},
    }

    result = supervisor.supervisor_node(state)

    assert result["route"] == "escalation"
    assert result["escalation_needed"] is True
    assert result["escalation_reason"] == "Account blocked"


def test_second_pass_finalizes_on_high_confidence():
    state = {
        "ticket_id": "t1",
        "classification": {"category": "technical"},
        "confidence": 0.9,
        "escalation_needed": False,
    }

    result = supervisor.supervisor_node(state)

    assert result["route"] == "finalize"


def test_second_pass_escalates_on_low_confidence():
    state = {
        "ticket_id": "t1",
        "classification": {"category": "technical"},
        "confidence": 0.3,
        "escalation_needed": False,
    }

    result = supervisor.supervisor_node(state)

    assert result["route"] == "escalation"
    assert result["escalation_needed"] is True


def test_second_pass_escalates_when_resolver_flagged_it_even_with_ok_confidence():
    state = {
        "ticket_id": "t1",
        "classification": {"category": "technical"},
        "confidence": 0.8,
        "escalation_needed": True,
        "escalation_reason": "Outside what I can verify",
    }

    result = supervisor.supervisor_node(state)

    assert result["route"] == "escalation"
    # escalation_needed/escalation_reason were already set by the resolver
    # itself, so supervisor shouldn't need to (re-)set or overwrite them.
    assert "escalation_needed" not in result
    assert "escalation_reason" not in result


def test_route_from_supervisor_reads_state():
    assert supervisor.route_from_supervisor({"route": "finalize"}) == "finalize"
