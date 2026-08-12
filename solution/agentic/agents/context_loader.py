"""Context Loader -- deterministic system node (not an LLM agent).

Runs first, before any reasoning happens: pulls together the customer's
CultPass profile + subscription, their UDA-Hub ticket history, and any
relevant long-term memories into `state["user_context"]`, so classification
and resolution both start with personalized context already available
instead of each agent re-fetching pieces of it independently.

It also resets every *per-turn* working-state field (see `_FRESH_TURN_STATE`
below). Only `messages` and `trace` have reducers that accumulate across
turns within the same `thread_id` -- everything else (`confidence`,
`draft_response`, `escalation_needed`, ...) is plain last-write-wins state
that the checkpointer would otherwise carry over verbatim from the previous
turn's final values. Left unreset, a second turn in the same session could
see a stale `confidence` from turn 1 and short-circuit straight to
Finalize/Escalation without a resolver ever running for its own ticket_text
-- Supervisor's `resolver_has_run = state.get("confidence") is not None`
check can't otherwise tell "second pass of this turn" apart from "second
pass of the previous turn, still sitting in state."
"""
from __future__ import annotations

from typing import Any

from agentic.tools.cultpass_tools import get_customer_profile, get_subscription_status
from agentic.tools.memory_tools import recall_customer_memory
from agentic.tools.udahub_tools import get_internal_user_id, get_ticket_history
from agentic.tracing import log_event

# Reset at the start of every invocation (see module docstring) so a new
# turn never inherits working state left over from a previous one.
_FRESH_TURN_STATE: dict[str, Any] = {
    "classification": {},
    "draft_response": None,
    "cited_article_ids": [],
    "confidence": None,
    "escalation_needed": False,
    "escalation_reason": None,
    "escalation_summary": None,
    "detected_preference": None,
    "final_status": None,
    "route": None,
}


def context_loader_node(state: dict[str, Any]) -> dict[str, Any]:
    """Populate `user_context`. See module docstring for what it gathers."""
    account_id = state["account_id"]
    external_user_id = state["external_user_id"]
    ticket_id = state.get("ticket_id")
    ticket_text = state.get("ticket_text", "")

    profile = get_customer_profile(external_user_id)
    subscription = get_subscription_status(external_user_id)
    ticket_history = get_ticket_history(account_id, external_user_id, exclude_ticket_id=ticket_id)

    # Long-term memory is keyed by UDA-Hub's *internal* user_id, which is
    # different from external_user_id -- resolve it first. A brand-new
    # customer with no UDA-Hub User record yet simply has no memories,
    # which is not an error.
    internal_user_id_result = get_internal_user_id(account_id, external_user_id)
    internal_user_id = internal_user_id_result.get("data")

    memories: list[dict[str, Any]] = []
    if internal_user_id and ticket_text.strip():
        recall_result = recall_customer_memory(internal_user_id, ticket_text)
        if recall_result.get("ok"):
            memories = recall_result["data"]

    user_context = {
        "profile": profile["data"] if profile.get("ok") else None,
        "subscription": subscription["data"] if subscription.get("ok") else None,
        "ticket_history": ticket_history["data"] if ticket_history.get("ok") else [],
        "long_term_memories": memories,
        "internal_user_id": internal_user_id,
    }

    entry = log_event(
        state,
        node="context_loader",
        event="loaded",
        found_profile=user_context["profile"] is not None,
        is_blocked=(user_context["profile"] or {}).get("is_blocked", False),
        past_ticket_count=len(user_context["ticket_history"]),
        memory_count=len(user_context["long_term_memories"]),
    )

    return {"user_context": user_context, "trace": [entry], **_FRESH_TURN_STATE}
