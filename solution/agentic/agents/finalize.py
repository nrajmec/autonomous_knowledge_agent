"""Finalize -- deterministic system node (not an LLM agent).

Runs last: persists whichever outcome the graph reached (resolved via a
resolver, or escalated) back to `udahub.db` as a new `TicketMessage` plus
the ticket's final status. This is the durable system-of-record write --
independent of, and in addition to, LangGraph's own session checkpointing.
For resolved tickets, it also saves a resolution summary to long-term
memory so a future ticket from the same customer can recall it.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agentic.tools.memory_tools import save_customer_memory
from agentic.tools.udahub_tools import update_ticket_record
from agentic.tracing import log_event


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    """Persist the final outcome and, if resolved, update long-term memory."""
    ticket_id = state["ticket_id"]
    account_id = state["account_id"]
    escalation_needed = state.get("escalation_needed", False)
    internal_user_id = (state.get("user_context") or {}).get("internal_user_id")

    if escalation_needed:
        final_status = "escalated"
        message_content = (
            state.get("escalation_summary")
            or state.get("escalation_reason")
            or "This ticket has been escalated to a human agent."
        )
    else:
        final_status = "resolved"
        message_content = state.get("draft_response") or ""

    update_result = update_ticket_record(
        ticket_id,
        status=final_status,
        message_role="ai",
        message_content=message_content,
    )

    memory_saved = False
    if final_status == "resolved" and internal_user_id and message_content:
        summary = f"Resolved ticket {ticket_id}: {message_content}"
        save_result = save_customer_memory(internal_user_id, account_id, "resolution_summary", summary)
        memory_saved = save_result.get("ok", False)

    # A stated preference is worth keeping regardless of how this particular
    # ticket turned out -- it's about the customer, not this resolution.
    preference_saved = False
    detected_preference = state.get("detected_preference")
    if detected_preference and internal_user_id:
        pref_result = save_customer_memory(internal_user_id, account_id, "preference", detected_preference)
        preference_saved = pref_result.get("ok", False)

    entry = log_event(
        state,
        node="finalize",
        event="persisted",
        final_status=final_status,
        ticket_update_ok=update_result.get("ok", False),
        memory_saved=memory_saved,
        preference_saved=preference_saved,
    )

    return {
        "final_status": final_status,
        # Appended (via the `messages` reducer) so utils.chat_interface can
        # print the final answer -- draft_response/escalation_summary alone
        # aren't part of the LangChain message history.
        "messages": [AIMessage(content=message_content)],
        "trace": [entry],
    }
