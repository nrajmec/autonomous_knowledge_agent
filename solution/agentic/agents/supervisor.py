"""Supervisor -- central router node of the hub-and-spoke graph.

Re-entered after every specialist step (see agentic/workflow.py): on first
entry, right after classification and before any resolver has run, it picks
which resolver to send the ticket to, or bypasses straight to Escalation on
a hard business-rule flag from the Classifier. On re-entry -- a resolver
just ran -- it evaluates that resolver's confidence/escalation flag and
decides Finalize vs. Escalation.

This is deliberately rule-based, not an LLM call: the Classifier has
already been forced (via structured output) to commit to exactly one of
five categories, so there's no remaining ambiguity for an LLM to resolve
here -- routing is just a table lookup plus a threshold check. See
agentic/agents/classifier.py for where the actual judgment call happens.
"""
from __future__ import annotations

from typing import Any

from agentic.tracing import log_event

CATEGORY_TO_RESOLVER = {
    "technical": "technical_resolver",
    "billing": "billing_resolver",
    "account": "account_resolver",
    "booking": "booking_resolver",
    "general": "general_resolver",
}

# Below this, a resolver's own answer isn't trusted enough to send as-is.
CONFIDENCE_THRESHOLD = 0.6


def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Decide the next hop and record it as state["route"] for the
    matching conditional edge (`route_from_supervisor`) to read."""
    classification = state.get("classification") or {}
    resolver_has_run = state.get("confidence") is not None

    if not resolver_has_run:
        if classification.get("hard_escalate"):
            route = "escalation"
            reason = classification.get("hard_escalate_reason") or "Flagged by classifier for mandatory human review"
        else:
            category = classification.get("category", "general")
            route = CATEGORY_TO_RESOLVER.get(category, "general_resolver")
            reason = f"Routed by category: {category}"
    else:
        confidence = state.get("confidence") or 0.0
        resolver_flagged_escalation = state.get("escalation_needed", False)
        if resolver_flagged_escalation or confidence < CONFIDENCE_THRESHOLD:
            route = "escalation"
            reason = state.get("escalation_reason") or f"Resolver confidence {confidence} below threshold"
        else:
            route = "finalize"
            reason = f"Resolver confidence {confidence} met threshold"

    entry = log_event(state, node="supervisor", event="routed", route=route, reason=reason)

    update: dict[str, Any] = {"route": route, "trace": [entry]}
    # Make sure escalation_needed/escalation_reason are set whenever we
    # route to escalation, even on the hard-business-rule bypass path
    # where no resolver has run yet to set them itself.
    if route == "escalation" and not state.get("escalation_needed"):
        update["escalation_needed"] = True
        update["escalation_reason"] = state.get("escalation_reason") or reason

    return update


def route_from_supervisor(state: dict[str, Any]) -> str:
    """Conditional-edge lookup: read the route Supervisor just decided."""
    return state["route"]
