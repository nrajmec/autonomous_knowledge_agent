"""Tests for the resolver factory in agentic/agents/resolver.py.

Uses FakeChatModel (no OPENAI_API_KEY needed) and monkeypatches the
underlying tool functions the built-in tool builders wrap, so these never
touch a real database or embeddings call either.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agentic.agents.resolver as resolver_module
from agentic.agents.resolver import ResolverOutput, create_resolver_node
from tests.fakes import FakeChatModel


def _tool_call_message(name, args, call_id="call1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _stop_message(content="Here is my answer."):
    return AIMessage(content=content, tool_calls=[])


def _base_state(**overrides):
    state = {
        "ticket_id": "t1",
        "account_id": "acc1",
        "external_user_id": "ext1",
        "ticket_text": "I can't log in.",
        "user_context": {},
    }
    state.update(overrides)
    return state


def test_resolver_calls_tool_then_returns_structured_output(monkeypatch):
    monkeypatch.setattr(
        resolver_module,
        "search_knowledge_base",
        lambda query, account_id, top_k=3: {
            "ok": True,
            "data": [{"article_id": "a-login", "title": "Login Help", "content": "...", "score": 0.9}],
            "relevant": True,
        },
    )

    node = create_resolver_node("technical", "handle login issues", ["search_knowledge_base"])
    fake_llm = FakeChatModel(
        tool_loop_responses=[
            _tool_call_message("search_knowledge_base", {"query": "login issue"}),
            _stop_message(),
        ],
        structured_responses=[
            ResolverOutput(
                response="Try resetting your password via the login page.",
                cited_article_ids=["a-login"],
                confidence=0.9,
                escalate=False,
            )
        ],
    )

    result = node(_base_state(), llm=fake_llm)

    assert result["draft_response"] == "Try resetting your password via the login page."
    assert result["cited_article_ids"] == ["a-login"]
    assert result["confidence"] == 0.9
    assert result["escalation_needed"] is False
    assert result["trace"][0]["node"] == "technical_resolver"
    assert result["trace"][0]["tool_calls"][0]["tool"] == "search_knowledge_base"


def test_resolver_handles_llm_answering_without_any_tool_call(monkeypatch):
    monkeypatch.setattr(resolver_module, "search_knowledge_base", lambda *a, **k: {"ok": True, "data": []})

    node = create_resolver_node("general", "handle general questions", ["search_knowledge_base"])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[
            ResolverOutput(response="I don't have info on that.", confidence=0.2, escalate=True, escalation_reason="No article found")
        ],
    )

    result = node(_base_state(), llm=fake_llm)

    assert result["escalation_needed"] is True
    assert result["trace"][0]["tool_calls"] == []


def test_resolver_stops_after_max_iterations_and_still_answers(monkeypatch):
    monkeypatch.setattr(resolver_module, "search_knowledge_base", lambda *a, **k: {"ok": True, "data": []})

    node = create_resolver_node("general", "handle general questions", ["search_knowledge_base"])
    # The fake LLM keeps requesting tool calls forever -- the loop must
    # still bound itself at MAX_TOOL_ITERATIONS and produce a final answer.
    looping_responses = [
        _tool_call_message("search_knowledge_base", {"query": "x"}, call_id=f"call{i}")
        for i in range(resolver_module.MAX_TOOL_ITERATIONS)
    ]
    fake_llm = FakeChatModel(
        tool_loop_responses=looping_responses,
        structured_responses=[ResolverOutput(response="Best effort answer.", confidence=0.4, escalate=True)],
    )

    result = node(_base_state(), llm=fake_llm)

    assert result["draft_response"] == "Best effort answer."
    assert len(result["trace"][0]["tool_calls"]) == resolver_module.MAX_TOOL_ITERATIONS


def test_resolver_records_error_for_unknown_tool_call_without_crashing(monkeypatch):
    monkeypatch.setattr(resolver_module, "search_knowledge_base", lambda *a, **k: {"ok": True, "data": []})

    node = create_resolver_node("general", "handle general questions", ["search_knowledge_base"])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_tool_call_message("some_tool_not_bound_here", {}), _stop_message()],
        structured_responses=[ResolverOutput(response="Answer anyway.", confidence=0.3, escalate=True)],
    )

    result = node(_base_state(), llm=fake_llm)

    logged_call = result["trace"][0]["tool_calls"][0]
    assert logged_call["tool"] == "some_tool_not_bound_here"
    assert logged_call["ok"] is False
    assert logged_call["error_category"] == "unknown_tool"


@pytest.mark.parametrize(
    "channel,expected_snippet",
    [
        ("email", "complete email reply"),
        ("chat", "short and conversational"),
        ("social_media", "Never include account-specific details"),
        ("carrier_pigeon", resolver_module.DEFAULT_CHANNEL_GUIDANCE),
    ],
)
def test_resolver_prompt_adapts_to_channel(monkeypatch, channel, expected_snippet):
    node = create_resolver_node("general", "handle general questions", [])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[ResolverOutput(response="Answer.", confidence=0.8, escalate=False)],
    )

    node(_base_state(channel=channel), llm=fake_llm)

    system_message = fake_llm.captured_tool_loop_messages[0][0]
    assert expected_snippet in system_message.content


def test_resolver_prompt_uses_default_guidance_when_channel_missing():
    node = create_resolver_node("general", "handle general questions", [])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[ResolverOutput(response="Answer.", confidence=0.8, escalate=False)],
    )

    node(_base_state(), llm=fake_llm)  # no "channel" key at all

    system_message = fake_llm.captured_tool_loop_messages[0][0]
    assert resolver_module.DEFAULT_CHANNEL_GUIDANCE in system_message.content


def test_create_resolver_node_rejects_unknown_tool_names():
    with pytest.raises(ValueError):
        create_resolver_node("technical", "...", ["not_a_real_tool"])


def test_resolver_tool_call_log_never_carries_raw_args_or_results(monkeypatch):
    monkeypatch.setattr(
        resolver_module,
        "get_customer_profile",
        lambda external_user_id: {
            "ok": True,
            "data": {"user_id": "ext1", "full_name": "Alice Kingsley", "email": "alice@wonderland.com"},
        },
    )

    node = create_resolver_node("account", "handle account questions", ["get_customer_profile"])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_tool_call_message("get_customer_profile", {}), _stop_message()],
        structured_responses=[ResolverOutput(response="Here you go.", confidence=0.9, escalate=False)],
    )

    result = node(_base_state(), llm=fake_llm)

    logged_call = result["trace"][0]["tool_calls"][0]
    assert logged_call == {"tool": "get_customer_profile", "ok": True, "result_count": 1, "error_category": None}
    assert "Alice Kingsley" not in str(result["trace"])
    assert "args" not in logged_call
    assert "result" not in logged_call


def test_resolver_tool_call_log_counts_list_results_and_classifies_errors(monkeypatch):
    monkeypatch.setattr(
        resolver_module,
        "search_knowledge_base",
        lambda query, account_id, top_k=3: {
            "ok": True,
            "data": [{"article_id": "a1"}, {"article_id": "a2"}],
            "relevant": True,
        },
    )
    monkeypatch.setattr(
        resolver_module,
        "manage_subscription",
        lambda external_user_id, action, tier=None: {"ok": False, "error": "Account is blocked; cannot proceed"},
    )

    node = create_resolver_node(
        "billing", "handle billing questions", ["search_knowledge_base", "manage_subscription"]
    )
    fake_llm = FakeChatModel(
        tool_loop_responses=[
            _tool_call_message("search_knowledge_base", {"query": "refund"}, call_id="c1"),
            _tool_call_message("manage_subscription", {"action": "cancel"}, call_id="c2"),
            _stop_message(),
        ],
        structured_responses=[ResolverOutput(response="See above.", confidence=0.4, escalate=True)],
    )

    result = node(_base_state(), llm=fake_llm)

    calls = result["trace"][0]["tool_calls"]
    assert calls[0] == {
        "tool": "search_knowledge_base",
        "ok": True,
        "result_count": 2,
        "error_category": None,
        "relevant": True,
    }
    assert calls[1] == {
        "tool": "manage_subscription",
        "ok": False,
        "result_count": 0,
        "error_category": "blocked_account",
    }


def test_resolver_folds_prior_session_messages_into_prompt():
    node = create_resolver_node("billing", "handle billing questions", [])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[ResolverOutput(response="Answer.", confidence=0.8, escalate=False)],
    )

    state = _base_state(
        ticket_text="can I get it cheaper?",
        messages=[
            HumanMessage(content="I'm on the premium plan"),
            AIMessage(content="Got it, how can I help with your premium plan?"),
            HumanMessage(content="can I get it cheaper?"),
        ],
    )
    node(state, llm=fake_llm)

    human_message = fake_llm.captured_tool_loop_messages[0][1]
    assert "I'm on the premium plan" in human_message.content


def test_resolver_omits_history_block_on_first_turn():
    node = create_resolver_node("billing", "handle billing questions", [])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[ResolverOutput(response="Answer.", confidence=0.8, escalate=False)],
    )

    node(_base_state(messages=[HumanMessage(content="I can't log in.")]), llm=fake_llm)

    human_message = fake_llm.captured_tool_loop_messages[0][1]
    assert "Conversation so far" not in human_message.content


def test_resolver_passes_through_detected_preference():
    node = create_resolver_node("billing", "handle billing questions", [])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[
            ResolverOutput(
                response="Sure, I'll note that.",
                confidence=0.9,
                escalate=False,
                detected_preference="Prefers email over phone contact",
            )
        ],
    )

    result = node(_base_state(), llm=fake_llm)

    assert result["detected_preference"] == "Prefers email over phone contact"


def test_resolver_detected_preference_defaults_to_none():
    node = create_resolver_node("billing", "handle billing questions", [])
    fake_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[ResolverOutput(response="Answer.", confidence=0.9, escalate=False)],
    )

    result = node(_base_state(), llm=fake_llm)

    assert result["detected_preference"] is None
