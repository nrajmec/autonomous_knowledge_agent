"""Shared helper: fold accumulated session messages into agent prompts.

`state["messages"]` is LangGraph's session memory (see `agentic/state.py`),
persisted across turns by the compiled graph's `MemorySaver` checkpointer,
keyed by `thread_id = ticket_id` (see `utils.chat_interface`). Without this
helper, that history sat in state but never reached an agent's actual
reasoning: every LLM-calling node built its prompt from `state["ticket_text"]`
alone, so a follow-up turn couldn't depend on anything said earlier in the
same session. Every LLM-calling node folds this transcript into its prompt.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

MAX_HISTORY_MESSAGES = 8
MAX_MESSAGE_CHARS = 400


def format_recent_messages(state: dict[str, Any]) -> str:
    """Render the last few turns of `state["messages"]` as a small transcript.

    Returns "" when there's nothing yet worth including (first turn, or no
    prior messages at all) so a first-turn prompt is unchanged.
    """
    messages = state.get("messages") or []
    if len(messages) <= 1:
        return ""

    lines: list[str] = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        if isinstance(message, HumanMessage):
            role = "Customer"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        else:
            continue

        content = str(message.content or "").strip()
        if not content:
            continue
        if len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS] + "..."
        lines.append(f"{role}: {content}")

    if not lines:
        return ""

    return "Conversation so far this session (oldest first):\n" + "\n".join(lines)
