"""The five category resolver nodes, instantiated from the shared factory
in resolver.py. Same execution shape (see resolver.py), different prompt +
tool subset per category.
"""
from __future__ import annotations

from agentic.agents.resolver import create_resolver_node

technical_resolver_node = create_resolver_node(
    category="technical",
    category_instructions=(
        "You handle login/access issues and app bugs or crashes. Check the "
        "customer's profile for a blocked account before troubleshooting "
        "anything else -- a blocked account is not a technical bug and "
        "cannot be fixed by these steps; escalate it instead. Ground "
        "troubleshooting steps in a knowledge base article whenever one exists."
    ),
    tool_names=["search_knowledge_base", "get_customer_profile", "recall_customer_memory"],
)

billing_resolver_node = create_resolver_node(
    category="billing",
    category_instructions=(
        "You handle subscription tier/quota/payment questions and "
        "cancel/reactivate/upgrade requests. Look up the current subscription "
        "before making any change, and only call manage_subscription once you "
        "know what the customer actually wants -- confirm the action and tier "
        "match their request."
    ),
    tool_names=[
        "search_knowledge_base",
        "get_subscription_status",
        "manage_subscription",
        "recall_customer_memory",
    ],
)

account_resolver_node = create_resolver_node(
    category="account",
    category_instructions=(
        "You handle profile questions, blocked-account questions, and "
        "account access/data requests. If the account is blocked, explain "
        "what that means using a knowledge base article if one exists, but "
        "you cannot unblock an account yourself -- that always needs "
        "escalation."
    ),
    tool_names=["search_knowledge_base", "get_customer_profile", "recall_customer_memory"],
)

booking_resolver_node = create_resolver_node(
    category="booking",
    category_instructions=(
        "You handle booking or cancelling CultPass experience reservations "
        "and availability questions. Use search_experiences to find options "
        "and list_reservations before cancelling anything -- get real "
        "experience_id/reservation_id values from those tools rather than "
        "guessing them."
    ),
    tool_names=[
        "search_knowledge_base",
        "search_experiences",
        "list_reservations",
        "manage_reservation",
        "recall_customer_memory",
    ],
)

general_resolver_node = create_resolver_node(
    category="general",
    category_instructions=(
        "You handle anything that doesn't fit technical, billing, account, "
        "or booking -- general FAQ-style questions. Answer only from a "
        "retrieved knowledge base article; if nothing relevant is found, "
        "escalate rather than guessing."
    ),
    tool_names=["search_knowledge_base", "recall_customer_memory"],
)
