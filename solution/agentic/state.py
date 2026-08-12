"""Shared LangGraph state schema for the UDA-Hub ticket-resolution graph.

One TypedDict, threaded through every node. `messages` is the short-term/
session memory LangGraph itself manages (scoped by `thread_id = ticket_id`
via the graph's checkpointer, per `utils.chat_interface`) -- everything
else is ticket-specific working state accumulated as the graph runs.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Category = Literal["technical", "billing", "account", "booking", "general"]


class Classification(TypedDict, total=False):
    category: Category
    urgency: Literal["low", "medium", "high"]
    sentiment: Literal["positive", "neutral", "negative"]
    complexity: Literal["simple", "moderate", "complex"]
    is_repeat_issue: bool
    # A hard business-rule bypass straight to Escalation (blocked account,
    # explicit request for a human, etc.) -- independent of any resolver's
    # own confidence score.
    hard_escalate: bool
    hard_escalate_reason: str | None


class TicketState(TypedDict, total=False):
    # Short-term/session memory (see module docstring).
    messages: Annotated[list[BaseMessage], add_messages]

    # Ticket identity + input metadata, seeded once at graph invocation.
    ticket_id: str
    account_id: str
    external_user_id: str
    channel: str
    reported_urgency: str | None
    ticket_text: str

    # Set once by the Context Loader system node.
    user_context: dict[str, Any]

    # Set by the Classifier agent.
    classification: Classification

    # Set by whichever resolver ran (left as defaults if routed straight to
    # Escalation on a hard business rule, before any resolver runs).
    draft_response: str | None
    cited_article_ids: list[str]
    confidence: float | None
    escalation_needed: bool
    escalation_reason: str | None

    # Set by the Escalation agent.
    escalation_summary: str | None

    # Set by the Finalize system node.
    final_status: Literal["resolved", "escalated"]

    # Set by the Supervisor node each time it runs; read by the matching
    # conditional-edge function to decide the next hop (see
    # agentic/agents/supervisor.py).
    route: str

    # Structured audit trail (see agentic/tracing.py). `operator.add`
    # concatenates lists, so every node's `{"trace": [entry]}` return
    # value appends rather than overwriting.
    trace: Annotated[list[dict[str, Any]], operator.add]
