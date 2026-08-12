"""Test doubles for LangChain chat models.

There's no OPENAI_API_KEY in this dev environment, so agent nodes
(classifier, resolvers, escalation) accept an injectable `llm` parameter
that defaults to a lazily-built real `ChatOpenAI` -- tests instead pass a
`FakeChatModel`, which duck-types only the two methods our hand-written
agent loops actually call (`bind_tools`, `with_structured_output`), each
returning scripted, pre-built responses instead of calling any API.
"""
from __future__ import annotations

from typing import Any


class _ScriptedRunnable:
    def __init__(self, pop_fn, on_invoke=None):
        self._pop_fn = pop_fn
        self._on_invoke = on_invoke

    def invoke(self, messages: Any):
        if self._on_invoke is not None:
            self._on_invoke(messages)
        return self._pop_fn()


class FakeChatModel:
    """Not a real BaseChatModel -- just enough surface area for our nodes.

    `tool_loop_responses`: a queue of AIMessage (or similar) objects
    returned in order by the tool-calling loop's `.invoke()` calls (the
    `bind_tools(...)`-returned runnable).

    `structured_responses`: a queue of Pydantic model instances returned in
    order by the final `.with_structured_output(...)`-returned runnable's
    `.invoke()` calls.
    """

    def __init__(self, tool_loop_responses=None, structured_responses=None):
        self._tool_loop_responses = list(tool_loop_responses or [])
        self._structured_responses = list(structured_responses or [])
        # Every `messages` list passed to the bind_tools()-returned
        # runnable's .invoke(), in call order -- lets a test assert on the
        # prompt actually built (e.g. per-channel system prompt wording).
        self.captured_tool_loop_messages: list[Any] = []

    def _pop_tool_loop(self):
        if not self._tool_loop_responses:
            raise AssertionError("FakeChatModel ran out of scripted tool_loop_responses")
        return self._tool_loop_responses.pop(0)

    def _pop_structured(self):
        if not self._structured_responses:
            raise AssertionError("FakeChatModel ran out of scripted structured_responses")
        return self._structured_responses.pop(0)

    def bind_tools(self, tools):
        return _ScriptedRunnable(self._pop_tool_loop, on_invoke=self.captured_tool_loop_messages.append)

    def with_structured_output(self, schema):
        return _ScriptedRunnable(self._pop_structured)
