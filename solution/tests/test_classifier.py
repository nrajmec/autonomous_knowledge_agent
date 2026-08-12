"""Tests for agentic/agents/classifier.py, using a FakeChatModel so no
OPENAI_API_KEY is needed."""
from langchain_core.messages import AIMessage, HumanMessage

import agentic.agents.classifier as classifier
from tests.fakes import FakeChatModel


def test_classifier_returns_structured_classification():
    fake_llm = FakeChatModel(
        structured_responses=[
            classifier.ClassificationSchema(
                category="technical",
                urgency="medium",
                sentiment="neutral",
                complexity="simple",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    state = {
        "ticket_id": "t1",
        "channel": "chat",
        "reported_urgency": "medium",
        "ticket_text": "I can't log in to my account.",
        "user_context": {"profile": {"is_blocked": False}, "subscription": None, "ticket_history": []},
    }

    result = classifier.classifier_node(state, llm=fake_llm)

    assert result["classification"]["category"] == "technical"
    assert result["classification"]["hard_escalate"] is False
    assert len(result["trace"]) == 1
    assert result["trace"][0]["node"] == "classifier"


def test_classifier_falls_back_to_general_for_unrecognized_category():
    fake_llm = FakeChatModel(
        structured_responses=[
            classifier.ClassificationSchema(
                category="not-a-real-category",
                urgency="low",
                sentiment="neutral",
                complexity="simple",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    state = {"ticket_id": "t1", "ticket_text": "hello", "user_context": {}}

    result = classifier.classifier_node(state, llm=fake_llm)

    assert result["classification"]["category"] == "general"


def test_classifier_surfaces_hard_escalate_flag():
    fake_llm = FakeChatModel(
        structured_responses=[
            classifier.ClassificationSchema(
                category="account",
                urgency="high",
                sentiment="negative",
                complexity="moderate",
                is_repeat_issue=False,
                hard_escalate=True,
                hard_escalate_reason="Account is blocked",
            )
        ]
    )
    state = {
        "ticket_id": "t1",
        "ticket_text": "Why is my account blocked?!",
        "user_context": {"profile": {"is_blocked": True}},
    }

    result = classifier.classifier_node(state, llm=fake_llm)

    assert result["classification"]["hard_escalate"] is True
    assert result["classification"]["hard_escalate_reason"] == "Account is blocked"


def test_classifier_folds_prior_session_messages_into_prompt():
    fake_llm = FakeChatModel(
        structured_responses=[
            classifier.ClassificationSchema(
                category="billing",
                urgency="low",
                sentiment="neutral",
                complexity="simple",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    state = {
        "ticket_id": "t1",
        "ticket_text": "does that include the one I asked about?",
        "user_context": {},
        "messages": [
            HumanMessage(content="What experiences are included this month?"),
            AIMessage(content="This month includes the Carnival History Tour and 6 others."),
            HumanMessage(content="does that include the one I asked about?"),
        ],
    }

    classifier.classifier_node(state, llm=fake_llm)

    prompt_text = str(fake_llm.captured_structured_messages[0])
    assert "Carnival History Tour" in prompt_text
    assert "What experiences are included this month?" in prompt_text


def test_classifier_omits_history_block_on_first_turn():
    fake_llm = FakeChatModel(
        structured_responses=[
            classifier.ClassificationSchema(
                category="general",
                urgency="low",
                sentiment="neutral",
                complexity="simple",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    state = {
        "ticket_id": "t1",
        "ticket_text": "Hi there",
        "user_context": {},
        "messages": [HumanMessage(content="Hi there")],
    }

    classifier.classifier_node(state, llm=fake_llm)

    prompt_text = str(fake_llm.captured_structured_messages[0])
    assert "Conversation so far" not in prompt_text
