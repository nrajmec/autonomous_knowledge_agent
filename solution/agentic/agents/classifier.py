"""Classifier -- LLM agent, no tools.

Reads the ticket text + metadata + loaded `user_context` and produces a
structured classification: category, urgency, sentiment, complexity,
whether this looks like a repeat issue, and a `hard_escalate` bypass for
cases that must go to a human no matter what a resolver could say. The
Supervisor (agentic/agents/supervisor.py) turns this into an actual routing
decision -- the Classifier only observes and reports.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.history import format_recent_messages
from agentic.tracing import log_event

CATEGORIES = ("technical", "billing", "account", "booking", "general")

SYSTEM_PROMPT = """You are the Classifier for UDA-Hub, a support-ticket triage \
system for CultPass (a subscription events platform). Read the customer's \
ticket and their account context, then classify it. Be decisive: pick \
exactly one category even if the ticket touches more than one topic -- pick \
the customer's primary intent.

If this session has earlier turns, use them as context -- a follow-up like \
"does that include the one I asked about?" only makes sense in light of what \
was said before it.

Categories:
- technical: login/access issues, app bugs or crashes
- billing: subscription tier/quota/payment questions, cancel/reactivate/upgrade
- account: profile changes, blocked-account questions, account access/data
- booking: booking or cancelling experience reservations, availability questions
- general: anything else / FAQ that doesn't fit the above

Flag hard_escalate=True only for cases that must go to a human regardless of \
what a resolver could say: the customer's account is blocked, they \
explicitly ask for a human or a manager, or the message describes a safety, \
legal, or abuse concern. Do not hard-escalate just because a ticket sounds \
difficult or the customer is upset -- that is a normal resolver's job."""


class ClassificationSchema(BaseModel):
    category: str = Field(description=f"Exactly one of: {', '.join(CATEGORIES)}")
    urgency: str = Field(description="One of: low, medium, high")
    sentiment: str = Field(description="One of: positive, neutral, negative")
    complexity: str = Field(description="One of: simple, moderate, complex")
    is_repeat_issue: bool = Field(description="True if user_context shows a similar past ticket")
    hard_escalate: bool = Field(
        description="True only for a blocked account, explicit human request, or safety/legal concern"
    )
    hard_escalate_reason: str | None = Field(default=None, description="Why, if hard_escalate is True")


_default_llm: BaseChatModel | None = None


def _get_default_llm() -> BaseChatModel:
    global _default_llm
    if _default_llm is None:
        from langchain_openai import ChatOpenAI

        _default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _default_llm


def _summarize_context(user_context: dict[str, Any]) -> str:
    profile = user_context.get("profile") or {}
    subscription = user_context.get("subscription")
    history = user_context.get("ticket_history") or []

    subscription_line = (
        f"Subscription: {subscription.get('tier')} / {subscription.get('status')}"
        if subscription
        else "Subscription: none on file"
    )
    if history:
        recent = "; ".join(
            f"[{t.get('status')}] {t.get('main_issue_type') or t.get('tags')}" for t in history[:5]
        )
        history_line = f"Past tickets ({len(history)}): {recent}"
    else:
        history_line = "Past tickets: none"

    return "\n".join(
        [
            f"Customer blocked: {profile.get('is_blocked', 'unknown')}",
            subscription_line,
            history_line,
        ]
    )


def classifier_node(state: dict[str, Any], llm: BaseChatModel | None = None) -> dict[str, Any]:
    """Classify the ticket. `llm` is an injection point for tests -- callers
    (the compiled graph) leave it as None to use the real default model."""
    model = llm or _get_default_llm()
    structured_model = model.with_structured_output(ClassificationSchema)

    history_block = format_recent_messages(state)
    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Ticket channel: {state.get('channel', 'unknown')}\n"
                f"Customer-reported urgency: {state.get('reported_urgency', 'not specified')}\n\n"
                f"Account context:\n{_summarize_context(state.get('user_context', {}))}\n\n"
                + (f"{history_block}\n\n" if history_block else "")
                + f"Ticket text:\n{state.get('ticket_text', '')}"
            )
        ),
    ]

    result: ClassificationSchema = structured_model.invoke(prompt)

    classification = result.model_dump()
    if classification["category"] not in CATEGORIES:
        classification["category"] = "general"

    entry = log_event(
        state,
        node="classifier",
        event="classified",
        category=classification["category"],
        urgency=classification["urgency"],
        hard_escalate=classification["hard_escalate"],
    )

    return {"classification": classification, "trace": [entry]}
