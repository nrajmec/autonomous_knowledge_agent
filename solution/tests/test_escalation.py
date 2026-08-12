"""Tests for agentic/agents/escalation.py, using a FakeChatModel so no
OPENAI_API_KEY is needed."""
from langchain_core.messages import AIMessage, HumanMessage

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
    # The raw reason string isn't logged (it can carry ticket-specific
    # text) -- only a coarse, safe category.
    assert result["trace"][0]["reason_category"] == "blocked_account"
    assert "reason" not in result["trace"][0]


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


def test_escalation_does_not_log_raw_internal_summary():
    fake_llm = FakeChatModel(
        structured_responses=[
            escalation.EscalationOutput(
                internal_summary="Customer Jane Doe at 123 Main St is upset about order #556.",
                customer_message="Thanks, a specialist will follow up.",
            )
        ]
    )
    state = {
        "ticket_id": "t1",
        "ticket_text": "help",
        "classification": {"category": "account"},
        "escalation_reason": "Low confidence",
        "draft_response": None,
    }

    result = escalation.escalation_node(state, llm=fake_llm)

    entry = result["trace"][0]
    assert entry["has_internal_summary"] is True
    assert "Jane Doe" not in str(entry)
    assert "internal_summary" not in entry


def test_escalation_folds_prior_session_messages_into_prompt():
    fake_llm = FakeChatModel(
        structured_responses=[
            escalation.EscalationOutput(internal_summary="s", customer_message="c"),
        ]
    )
    state = {
        "ticket_id": "t1",
        "ticket_text": "still not working",
        "classification": {"category": "technical"},
        "escalation_reason": "Low confidence",
        "draft_response": None,
        "messages": [
            HumanMessage(content="I tried resetting my password like you said"),
            AIMessage(content="Great, did that resolve the login issue?"),
            HumanMessage(content="still not working"),
        ],
    }

    escalation.escalation_node(state, llm=fake_llm)

    prompt_text = str(fake_llm.captured_structured_messages[0])
    assert "I tried resetting my password like you said" in prompt_text
