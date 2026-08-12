"""Structured, searchable logging for the multi-agent graph.

Every node emits one record per decision point via `log_event`, which does
two things with it:

1. Appends it to `state["trace"]` -- in-band, inspectable per-ticket via
   LangGraph's own `orchestrator.get_state_history()` without touching the
   filesystem (this is what `state.py`'s `trace` field, with an
   append-only reducer, is for).
2. Writes it as one JSON line to `logs/uda_hub_trace.jsonl` -- out-of-band,
   greppable/parseable across *all* tickets and sessions, independent of
   LangGraph state.

Log destination is a `logs/` directory alongside `agentic/`, i.e. under
`solution/`, not `starter/`.

This file is shared and greppable across every ticket and session, so
`categorize_reason()` below exists to keep nodes from writing customer- or
ticket-specific free text into it (a routing "reason" or an escalation
reason can be LLM-authored and reference ticket specifics) -- callers log a
coarse category instead of the raw string. See `agentic/agents/resolver.py`
for the equivalent treatment of tool call arguments/results.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_PATH = LOG_DIR / "uda_hub_trace.jsonl"

# Checked in order, first match wins. Covers both the deterministic reason
# strings this project's own nodes generate (category routing, confidence
# threshold results) and the freeform ones an LLM can author (a classifier's
# hard_escalate_reason, a resolver's escalation_reason).
_REASON_CATEGORY_RULES = (
    ("routed by category", "category_routed"),
    ("met threshold", "confidence_ok"),
    ("below threshold", "confidence_low"),
    ("blocked", "blocked_account"),
    ("human", "human_requested"),
    ("manager", "human_requested"),
    ("safety", "safety_or_legal"),
    ("legal", "safety_or_legal"),
    ("abuse", "safety_or_legal"),
    ("confidence", "low_confidence"),
    ("no relevant", "no_knowledge_match"),
    ("error", "tool_failure"),
)


def categorize_reason(reason: str | None) -> str:
    """Map a routing/escalation reason string to a coarse, safe category."""
    if not reason:
        return "unspecified"
    lowered = reason.lower()
    for keyword, category in _REASON_CATEGORY_RULES:
        if keyword in lowered:
            return category
    return "other"

_logger = logging.getLogger("uda_hub.trace")


def _ensure_configured() -> None:
    if _logger.handlers:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log_event(state: dict[str, Any], node: str, event: str, **details: Any) -> dict[str, Any]:
    """Build and record one structured trace entry.

    Args:
        state: The current graph state (only `ticket_id` is read from it).
        node: Which node/agent produced this entry, e.g. "classifier".
        event: What kind of entry this is, e.g. "classified", "routed",
            "tool_call", "escalated".
        **details: Anything else worth recording (routing decision,
            confidence score, tool name/result, etc.) -- must be
            JSON-serializable (or convertible via `str()`).

    Returns:
        The entry dict. Does NOT mutate `state` -- callers should merge it
        into their node's return value as `{"trace": [entry], ...}`, same
        as any other LangGraph state update.
    """
    _ensure_configured()

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_id": state.get("ticket_id"),
        "node": node,
        "event": event,
        **details,
    }
    _logger.info(json.dumps(entry, default=str))
    return entry
