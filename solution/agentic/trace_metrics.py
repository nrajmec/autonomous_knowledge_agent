"""Reporting layer over the structured trace log (logs/uda_hub_trace.jsonl).

Computes the aggregate metrics a reviewer needs to inspect at a glance:
knowledge-retrieval success rate, escalation frequency, and tool-usage
patterns. This is a pure query layer over entries `agentic/tracing.py`'s
`log_event()` already writes -- no new data collection, and nothing here
reads a raw tool payload, since `agentic/agents/resolver.py` never logs one
in the first place (see `_redact_tool_call`).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentic.tracing import LOG_PATH


def load_trace_entries(log_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read every JSON line from the trace log.

    A missing file just means nothing has run yet -- returns an empty list,
    not an error.
    """
    path = Path(log_path) if log_path is not None else LOG_PATH
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics over a list of trace entries (see `load_trace_entries`)."""
    finalize_entries = [e for e in entries if e.get("node") == "finalize" and e.get("event") == "persisted"]
    total_tickets = len(finalize_entries)
    resolved = sum(1 for e in finalize_entries if e.get("final_status") == "resolved")
    escalated = sum(1 for e in finalize_entries if e.get("final_status") == "escalated")
    escalation_frequency = (escalated / total_tickets) if total_tickets else None

    tool_counts: Counter[str] = Counter()
    tool_successes: Counter[str] = Counter()
    kb_searches = 0
    kb_relevant = 0

    resolver_entries = (e for e in entries if e.get("event") == "resolved" and str(e.get("node", "")).endswith("_resolver"))
    for entry in resolver_entries:
        for call in entry.get("tool_calls", []) or []:
            name = call.get("tool")
            if not name:
                continue
            tool_counts[name] += 1
            if call.get("ok"):
                tool_successes[name] += 1
            if name == "search_knowledge_base":
                kb_searches += 1
                if call.get("relevant") if "relevant" in call else (call.get("result_count") or 0) > 0:
                    kb_relevant += 1

    retrieval_success_rate = (kb_relevant / kb_searches) if kb_searches else None

    tool_usage = {
        name: {
            "calls": count,
            "successes": tool_successes.get(name, 0),
            "success_rate": round(tool_successes.get(name, 0) / count, 4) if count else None,
        }
        for name, count in sorted(tool_counts.items())
    }

    return {
        "total_tickets": total_tickets,
        "resolved": resolved,
        "escalated": escalated,
        "escalation_frequency": round(escalation_frequency, 4) if escalation_frequency is not None else None,
        "knowledge_retrieval": {
            "searches": kb_searches,
            "relevant_matches": kb_relevant,
            "success_rate": round(retrieval_success_rate, 4) if retrieval_success_rate is not None else None,
        },
        "tool_usage": tool_usage,
    }


def report(log_path: Path | str | None = None) -> dict[str, Any]:
    """Convenience: load the log and compute metrics in one call."""
    return compute_metrics(load_trace_entries(log_path))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_report(metrics: dict[str, Any]) -> str:
    """Human-readable rendering of `compute_metrics()`'s output."""
    kb = metrics["knowledge_retrieval"]
    lines = [
        f"Tickets processed: {metrics['total_tickets']} "
        f"({metrics['resolved']} resolved, {metrics['escalated']} escalated)",
        f"Escalation frequency: {_pct(metrics['escalation_frequency'])}",
        f"Knowledge-retrieval success rate: {_pct(kb['success_rate'])} "
        f"({kb['relevant_matches']}/{kb['searches']} searches found a relevant article)",
        "Tool usage:",
    ]
    if not metrics["tool_usage"]:
        lines.append("  (no tool calls recorded)")
    for name, stats in metrics["tool_usage"].items():
        lines.append(f"  {name}: {stats['calls']} calls, {_pct(stats['success_rate'])} success rate")
    return "\n".join(lines)
