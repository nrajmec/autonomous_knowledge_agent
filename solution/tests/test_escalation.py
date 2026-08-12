"""Tests for agentic/agents/escalation.py, using a FakeChatModel so no
OPENAI_API_KEY is needed."""
import agentic.agents.escalation as escalation
from tests.fakes import FakeChatModel


def test_escalation_produces_customer_message_and_marks_escalated():
    fake_llm = FakeChatModel(
        structured_responses=[
            escalation.EscalationOutput(
                internal_summary="Customer's account is blocked; needs manual unblock.",
                customer_message="Your request has been passed to a specialist who will follow up.",
            )
        ]
    )
    state = {
        "ticket_id": "t1",
        "ticket_text": "Why is my account blocked?!",
        "classification": {"category": "account", "hard_escalate": True},
        "escalation_reason": "Account is blocked",
        "draft_response": None,
    }

    result = escalation.escalation_node(state, llm=fake_llm)

    assert result["escalation_needed"] is True
    assert result["escalation_summary"] == "Your request has been passed to a specialist who will follow up."
    assert result["trace"][0]["node"] == "escalation"
    assert result["trace"][0]["reason"] == "Account is blocked"


def test_escalation_includes_resolver_draft_in_prompt_context():
    seen_prompts = []

    class RecordingStructured:
        def invoke(self, messages):
            seen_prompts.append(messages)
            return escalation.EscalationOutput(internal_summary="s", customer_message="c")

    class RecordingModel:
        def with_structured_output(self, schema):
            return RecordingStructured()

    state = {
        "ticket_id": "t1",
        "ticket_text": "It's still broken",
        "classification": {"category": "technical"},
        "escalation_reason": "Low confidence",
        "draft_response": "Have you tried restarting the app?",
    }

    escalation.escalation_node(state, llm=RecordingModel())

    prompt_text = str(seen_prompts[0])
    assert "Have you tried restarting the app?" in prompt_text
    assert "Low confidence" in prompt_text
