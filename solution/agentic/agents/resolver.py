"""Resolver agent factory.

Each category (technical/billing/account/booking/general) gets its own
resolver node from `create_resolver_node`, differing only in system prompt
and which tools it's allowed to call -- the execution shape is identical,
so this file has exactly one implementation shared by all five (see
agentic/agents/resolvers.py for the five instances).

Deliberately not `langgraph.prebuilt.create_react_agent`: the project asks
for each node's control flow to be hand-built rather than delegated to a
prebuilt helper, so the tool-calling loop below is written out explicitly --
bind tools, invoke, execute any requested tool calls, feed results back,
repeat (bounded by MAX_TOOL_ITERATIONS) -- then make one final
structured-output call to produce the resolver's answer contract.

Tool binding: a handful of tools need identity context (external_user_id,
the internal user_id for memory) that must come from graph state, not from
the LLM -- an LLM should never be trusted to supply someone's internal ids.
`_TOOL_BUILDERS` maps a logical tool name to a function that, given the
current `state`, returns a ready-to-call LangChain tool with that context
already bound via closure; the LLM only ever sees the parameters that are
actually its choice to make (query text, action, tier, experience_id, ...).
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic.tools.cultpass_tools import (
    get_customer_profile,
    get_subscription_status,
    list_reservations,
    manage_reservation,
    manage_subscription,
    search_experiences,
)
from agentic.agents.history import format_recent_messages
from agentic.tools.knowledge_tools import search_knowledge_base
from agentic.tools.memory_tools import recall_customer_memory
from agentic.tracing import log_event

MAX_TOOL_ITERATIONS = 4

_default_llm: BaseChatModel | None = None


def _get_default_llm() -> BaseChatModel:
    global _default_llm
    if _default_llm is None:
        from langchain_openai import ChatOpenAI

        _default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _default_llm


class ResolverOutput(BaseModel):
    response: str = Field(description="Customer-facing draft response")
    cited_article_ids: list[str] = Field(
        default_factory=list, description="Knowledge article ids the response is grounded in, if any"
    )
    confidence: float = Field(
        description="0.0-1.0: how directly the response is grounded in retrieved knowledge/tool results"
    )
    escalate: bool = Field(description="True if this should be handed to a human instead of sent as-is")
    escalation_reason: str | None = Field(default=None, description="Why, if escalate is True")
    detected_preference: str | None = Field(
        default=None,
        description=(
            "A durable customer preference explicitly stated in this ticket "
            "(e.g. 'prefers email over phone', 'always wants the cheapest "
            "tier'), worth remembering for future tickets. Leave null unless "
            "the customer actually stated one -- never invent one."
        ),
    )


BASE_RESOLVER_INSTRUCTIONS = """You are the {category} support resolver for \
UDA-Hub (CultPass customer support). Use your tools to look up whatever you \
need -- knowledge base articles, account/subscription/reservation details \
-- before answering. Never invent information: only state facts you \
retrieved via a tool or that were already given to you in the ticket/\
account context.

Confidence guidance: give confidence >= 0.75 only when your answer is \
directly grounded in a retrieved knowledge article or a successful tool \
call. Give confidence < 0.5 and set escalate=True if: no relevant knowledge \
article was found, a tool call failed or returned an error, the request is \
outside what you're able to do, or you cannot verify what the customer is \
asking about.

If the customer explicitly states a durable preference about how they want \
to be helped or contacted (e.g. "please only email me", "I always want the \
cheapest option"), capture it in detected_preference so it can be recalled \
on a future ticket. Leave it null for anything that isn't a real, \
restatable preference.

If this session has earlier turns, use them as context for what the \
customer is actually asking now -- don't ask them to repeat themselves.

{category_instructions}"""

# Multi-channel support: the same resolver logic/tools handle every channel,
# but the customer-facing draft is styled to fit where it's actually going.
# Keyed by `state["channel"]` (see `agentic/state.py`), case-insensitive.
CHANNEL_GUIDANCE = {
    "email": (
        "This ticket came in over email. Write a complete email reply: open "
        "with a brief greeting using the customer's name if known, and close "
        "with a short, professional sign-off."
    ),
    "chat": (
        "This ticket came in over live chat. Keep the reply short and "
        "conversational -- no greeting or sign-off needed, get straight to "
        "the answer."
    ),
    "social_media": (
        "This ticket came in over a public social media channel. Keep the "
        "reply brief and generic. Never include account-specific details "
        "(subscription tier, reservation ids, personal data) in a reply that "
        "may be publicly visible -- if specifics are needed, ask the "
        "customer to continue over a private channel instead."
    ),
}
DEFAULT_CHANNEL_GUIDANCE = "Keep the reply clear and appropriately concise for the customer's channel."


def _channel_guidance(channel: str | None) -> str:
    return CHANNEL_GUIDANCE.get((channel or "").strip().lower(), DEFAULT_CHANNEL_GUIDANCE)


# Coarse, non-leaking error categories for the trace log -- checked in order,
# first match wins. Derived from the actual error strings the tool layer
# returns (see cultpass_tools.py / udahub_tools.py), without repeating the
# raw message (which can embed ids or other customer-specific text).
_ERROR_CATEGORY_RULES = (
    ("unknown tool", "unknown_tool"),
    ("blocked", "blocked_account"),
    ("no slots available", "unavailable"),
    ("found", "not_found"),
    ("already", "conflict"),
    ("required", "validation"),
    ("must be", "validation"),
)


def _classify_error(message: str) -> str:
    lowered = message.lower()
    for keyword, category in _ERROR_CATEGORY_RULES:
        if keyword in lowered:
            return category
    return "other"


def _redact_tool_call(name: str, result: Any) -> dict[str, Any]:
    """Safe metadata about one tool call for the shared trace log.

    Deliberately excludes the raw call arguments and result payload -- both
    can carry customer data (profile fields, subscription/reservation
    details, free-text error messages). Only tool name, success, a result
    count, and a coarse error category are recorded; the full result is
    still available in-memory to the LLM for its own reasoning via the
    ToolMessage appended right after this.
    """
    ok = bool(isinstance(result, dict) and result.get("ok"))
    result_count: int | None = None
    error_category: str | None = None
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, list):
            result_count = len(data)
        elif data not in (None,):
            result_count = 1
        else:
            result_count = 0
        if not ok:
            error_category = _classify_error(str(result.get("error", "")))
    entry: dict[str, Any] = {"tool": name, "ok": ok, "result_count": result_count, "error_category": error_category}
    # search_knowledge_base's `relevant` flag isn't customer data -- it's a
    # signal about the knowledge base itself -- and it's exactly what a
    # retrieval-success-rate metric needs, so it's the one field kept as-is.
    if name == "search_knowledge_base" and isinstance(result, dict) and "relevant" in result:
        entry["relevant"] = bool(result["relevant"])
    return entry


def _build_search_knowledge_base_tool(state: dict[str, Any]) -> StructuredTool:
    account_id = state["account_id"]

    def _search(query: str, top_k: int = 3) -> dict:
        """Search this account's knowledge base articles for information relevant to a query."""
        return search_knowledge_base(query, account_id=account_id, top_k=top_k)

    return StructuredTool.from_function(_search, name="search_knowledge_base")


def _build_get_customer_profile_tool(state: dict[str, Any]) -> StructuredTool:
    external_user_id = state["external_user_id"]

    def _profile() -> dict:
        """Look up the current customer's CultPass profile (name, email, blocked status)."""
        return get_customer_profile(external_user_id)

    return StructuredTool.from_function(_profile, name="get_customer_profile")


def _build_get_subscription_status_tool(state: dict[str, Any]) -> StructuredTool:
    external_user_id = state["external_user_id"]

    def _status() -> dict:
        """Look up the current customer's CultPass subscription (tier, status, quota)."""
        return get_subscription_status(external_user_id)

    return StructuredTool.from_function(_status, name="get_subscription_status")


def _build_manage_subscription_tool(state: dict[str, Any]) -> StructuredTool:
    external_user_id = state["external_user_id"]

    def _manage(action: str, tier: str | None = None) -> dict:
        """Cancel, reactivate, or change the tier of the current customer's subscription.
        action: one of "cancel", "reactivate", "change_tier". tier (required
        for change_tier): "basic" or "premium"."""
        return manage_subscription(external_user_id, action, tier=tier)

    return StructuredTool.from_function(_manage, name="manage_subscription")


def _build_list_reservations_tool(state: dict[str, Any]) -> StructuredTool:
    external_user_id = state["external_user_id"]

    def _list() -> dict:
        """List the current customer's CultPass experience reservations."""
        return list_reservations(external_user_id)

    return StructuredTool.from_function(_list, name="list_reservations")


def _build_manage_reservation_tool(state: dict[str, Any]) -> StructuredTool:
    external_user_id = state["external_user_id"]

    def _manage(action: str, experience_id: str | None = None, reservation_id: str | None = None) -> dict:
        """Book or cancel a CultPass experience reservation for the current customer.
        action: "book" (requires experience_id, get one from search_experiences)
        or "cancel" (requires reservation_id, get one from list_reservations)."""
        return manage_reservation(
            external_user_id, action, experience_id=experience_id, reservation_id=reservation_id
        )

    return StructuredTool.from_function(_manage, name="manage_reservation")


def _build_search_experiences_tool(state: dict[str, Any]) -> StructuredTool:
    def _search(query: str = "", upcoming_only: bool = True) -> dict:
        """Search upcoming CultPass experiences by keyword (title/description/location)."""
        return search_experiences(query=query, upcoming_only=upcoming_only)

    return StructuredTool.from_function(_search, name="search_experiences")


def _build_recall_customer_memory_tool(state: dict[str, Any]) -> StructuredTool:
    internal_user_id = (state.get("user_context") or {}).get("internal_user_id")

    def _recall(query: str, top_k: int = 3) -> dict:
        """Recall the current customer's saved long-term memory (preferences,
        past resolution summaries) relevant to a query."""
        if not internal_user_id:
            return {"ok": True, "data": []}
        return recall_customer_memory(internal_user_id, query, top_k=top_k)

    return StructuredTool.from_function(_recall, name="recall_customer_memory")


_TOOL_BUILDERS: dict[str, Callable[[dict[str, Any]], StructuredTool]] = {
    "search_knowledge_base": _build_search_knowledge_base_tool,
    "get_customer_profile": _build_get_customer_profile_tool,
    "get_subscription_status": _build_get_subscription_status_tool,
    "manage_subscription": _build_manage_subscription_tool,
    "list_reservations": _build_list_reservations_tool,
    "manage_reservation": _build_manage_reservation_tool,
    "search_experiences": _build_search_experiences_tool,
    "recall_customer_memory": _build_recall_customer_memory_tool,
}


def create_resolver_node(
    category: str, category_instructions: str, tool_names: list[str]
) -> Callable[..., dict[str, Any]]:
    """Build a resolver node for one category.

    Args:
        category: e.g. "technical" -- used in the prompt and trace entries.
        category_instructions: Category-specific guidance appended to the
            shared base prompt (what this resolver handles).
        tool_names: Which of `_TOOL_BUILDERS`' logical tools this resolver
            may call.

    Returns:
        A LangGraph node function: (state, llm=None) -> partial state update.
    """
    unknown = set(tool_names) - set(_TOOL_BUILDERS)
    if unknown:
        raise ValueError(f"Unknown tool name(s) for resolver '{category}': {sorted(unknown)}")

    system_prompt = BASE_RESOLVER_INSTRUCTIONS.format(
        category=category, category_instructions=category_instructions
    )

    def resolver_node(state: dict[str, Any], llm: BaseChatModel | None = None) -> dict[str, Any]:
        model = llm or _get_default_llm()

        built_tools = {name: _TOOL_BUILDERS[name](state) for name in tool_names}
        tool_model = model.bind_tools(list(built_tools.values()))

        channel_prompt = f"{system_prompt}\n\nChannel guidance: {_channel_guidance(state.get('channel'))}"
        history_block = format_recent_messages(state)
        messages: list[Any] = [
            SystemMessage(content=channel_prompt),
            HumanMessage(
                content=(
                    f"Account context: {state.get('user_context', {})}\n\n"
                    + (f"{history_block}\n\n" if history_block else "")
                    + f"Ticket text:\n{state.get('ticket_text', '')}"
                )
            ),
        ]

        tool_call_log: list[dict[str, Any]] = []
        for _ in range(MAX_TOOL_ITERATIONS):
            ai_message: AIMessage = tool_model.invoke(messages)
            messages.append(ai_message)

            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                break

            for call in tool_calls:
                lc_tool = built_tools.get(call["name"])
                if lc_tool is None:
                    tool_result: Any = {"ok": False, "error": f"Unknown tool '{call['name']}'"}
                else:
                    try:
                        tool_result = lc_tool.invoke(call["args"])
                    except Exception as exc:  # tool functions validate their own input; this
                        # only guards against something unexpected so one bad call can't crash the graph.
                        tool_result = {"ok": False, "error": str(exc)}

                tool_call_log.append(_redact_tool_call(call["name"], tool_result))
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
        else:
            messages.append(
                HumanMessage(content="Please provide your final answer now based on what you've gathered.")
            )

        structured_model = model.with_structured_output(ResolverOutput)
        result: ResolverOutput = structured_model.invoke(messages)

        entry = log_event(
            state,
            node=f"{category}_resolver",
            event="resolved",
            confidence=result.confidence,
            escalate=result.escalate,
            tool_calls=tool_call_log,
        )

        return {
            "draft_response": result.response,
            "cited_article_ids": result.cited_article_ids,
            "confidence": result.confidence,
            "escalation_needed": result.escalate,
            "escalation_reason": result.escalation_reason,
            "detected_preference": result.detected_preference,
            "trace": [entry],
        }

    resolver_node.__name__ = f"{category}_resolver_node"
    return resolver_node
