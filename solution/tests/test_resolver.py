"""Tests for the resolver factory in agentic/agents/resolver.py.

Uses FakeChatModel (no OPENAI_API_KEY needed) and monkeypatches the
underlying tool functions the built-in tool builders wrap, so these never
touch a real database or embeddings call either.
"""
import pytest
from langchain_core.messages import AIMessage

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
    assert logged_call["result"]["ok"] is False


def test_create_resolver_node_rejects_unknown_tool_names():
    with pytest.raises(ValueError):
        create_resolver_node("technical", "...", ["not_a_real_tool"])
