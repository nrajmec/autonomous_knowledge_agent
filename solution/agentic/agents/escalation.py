"""Escalation -- LLM agent.

Reached either directly from the Supervisor (a hard business-rule bypass:
blocked account, explicit request for a human, safety/legal concern) or
after a resolver's own confidence came back too low. Summarizes the ticket
(and any resolver attempt) for the human agent who picks it up, and drafts
the customer-facing handoff message -- it does not attempt to resolve the
issue itself.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.tracing import log_event

SYSTEM_PROMPT = """You are the Escalation agent for UDA-Hub support. A ticket \
has been routed to you instead of being resolved automatically. Write two \
things:
1. A brief internal summary for the human agent who will pick this up: what \
the customer needs, what has already been tried (if anything), and why this \
needed escalation.
2. A short, empathetic customer-facing message telling them their request \
has been passed to a specialist. Do not promise a specific timeline, and do \
not attempt to resolve the issue yourself."""


class EscalationOutput(BaseModel):
    internal_summary: str = Field(description="Brief summary for the human agent picking this up")
    customer_message: str = Field(description="Short, empathetic customer-facing handoff message")


_default_llm: BaseChatModel | None = None


def _get_default_llm() -> BaseChatModel:
    global _default_llm
    if _default_llm is None:
        from langchain_openai import ChatOpenAI

        _default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _default_llm


def escalation_node(state: dict[str, Any], llm: BaseChatModel | None = None) -> dict[str, Any]:
    model = llm or _get_default_llm()
    structured_model = model.with_structured_output(EscalationOutput)

    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Ticket text:\n{state.get('ticket_text', '')}\n\n"
                f"Classification: {state.get('classification')}\n"
                f"Escalation reason so far: {state.get('escalation_reason')}\n"
                f"Resolver's draft attempt (if any): {state.get('draft_response')}"
            )
        ),
    ]

    result: EscalationOutput = structured_model.invoke(prompt)

    entry = log_event(
        state,
        node="escalation",
        event="escalated",
        reason=state.get("escalation_reason"),
        internal_summary=result.internal_summary,
    )

    return {
        "escalation_needed": True,
        "escalation_summary": result.customer_message,
        "trace": [entry],
    }
