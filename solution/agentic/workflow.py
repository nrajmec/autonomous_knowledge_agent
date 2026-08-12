"""LangGraph orchestration for UDA-Hub.

Hand-built `StateGraph` -- no `langgraph.prebuilt.create_react_agent`, no
`langgraph_supervisor`. Every node and edge is wired explicitly below so the
routing/decision logic is visible end-to-end. See
`agentic/design/architecture.md` for the full architecture writeup and
diagram; this module is the implementation of that design.

Graph shape (hub-and-spoke / Supervisor pattern):

    START -> context_loader -> classifier -> supervisor
                                                  |
                          (routes by category, or bypasses on hard_escalate)
                                                  |
                +---------------+---------------+---------------+---------------+
                |               |               |               |               |
        technical_resolver billing_resolver account_resolver booking_resolver general_resolver
                |               |               |               |               |
                +---------------+---------------+---------------+---------------+
                                                  |
                                            supervisor (re-entered; checks
                                            confidence / escalation_needed)
                                                  |
                                    +-------------+-------------+
                                    |                           |
                                escalation                  finalize -> END
                                    |
                                    +----------------------> finalize -> END

Short-term (session) memory is the compiled graph's own checkpointer,
scoped by `thread_id = ticket_id` (see `utils.chat_interface`). Long-term
(cross-session) memory and the durable interaction history are handled
inside the `context_loader` / `finalize` nodes via `agentic/tools/*`, not by
the checkpointer.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic.agents.classifier import classifier_node
from agentic.agents.context_loader import context_loader_node
from agentic.agents.escalation import escalation_node
from agentic.agents.finalize import finalize_node
from agentic.agents.resolvers import (
    account_resolver_node,
    billing_resolver_node,
    booking_resolver_node,
    general_resolver_node,
    technical_resolver_node,
)
from agentic.agents.supervisor import route_from_supervisor, supervisor_node
from agentic.state import TicketState

RESOLVER_NODES = {
    "technical_resolver": technical_resolver_node,
    "billing_resolver": billing_resolver_node,
    "account_resolver": account_resolver_node,
    "booking_resolver": booking_resolver_node,
    "general_resolver": general_resolver_node,
}


def build_graph() -> StateGraph:
    """Assemble the UDA-Hub ticket-resolution graph (uncompiled)."""
    graph = StateGraph(TicketState)

    graph.add_node("context_loader", context_loader_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("supervisor", supervisor_node)
    for name, node in RESOLVER_NODES.items():
        graph.add_node(name, node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "classifier")
    graph.add_edge("classifier", "supervisor")

    # One conditional edge function serves both of supervisor's passes: it
    # just reads whatever destination supervisor_node already decided and
    # wrote into state["route"] (see agentic/agents/supervisor.py).
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            **{name: name for name in RESOLVER_NODES},
            "escalation": "escalation",
            "finalize": "finalize",
        },
    )

    # Every resolver hands control back to supervisor rather than finishing
    # directly, so the same confidence/escalation check applies no matter
    # which category handled the ticket.
    for name in RESOLVER_NODES:
        graph.add_edge(name, "supervisor")

    graph.add_edge("escalation", "finalize")
    graph.add_edge("finalize", END)

    return graph


def build_orchestrator() -> CompiledStateGraph:
    """Compile the graph with an in-memory checkpointer for short-term/
    session memory, keyed by `thread_id` (== `ticket_id`, see
    `utils.chat_interface`)."""
    return build_graph().compile(checkpointer=MemorySaver())


# IDEALLY YOUR ONLY IMPORT HERE IS:
# from agentic.workflow import orchestrator
orchestrator = build_orchestrator()
