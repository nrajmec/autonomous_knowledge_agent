"""Tests for agentic/trace_metrics.py -- the reporting layer over
logs/uda_hub_trace.jsonl. Uses synthetic trace entries shaped exactly like
what log_event() actually writes, so this never touches the real log file.
"""
import json

from agentic.trace_metrics import compute_metrics, format_report, load_trace_entries


def _finalize(final_status):
    return {"node": "finalize", "event": "persisted", "final_status": final_status}


def _resolved(node, tool_calls):
    return {"node": node, "event": "resolved", "tool_calls": tool_calls}


def test_compute_metrics_on_empty_log():
    metrics = compute_metrics([])

    assert metrics["total_tickets"] == 0
    assert metrics["escalation_frequency"] is None
    assert metrics["knowledge_retrieval"]["success_rate"] is None
    assert metrics["tool_usage"] == {}


def test_compute_metrics_counts_resolved_and_escalated_tickets():
    entries = [_finalize("resolved"), _finalize("resolved"), _finalize("escalated"), _finalize("resolved")]

    metrics = compute_metrics(entries)

    assert metrics["total_tickets"] == 4
    assert metrics["resolved"] == 3
    assert metrics["escalated"] == 1
    assert metrics["escalation_frequency"] == 0.25


def test_compute_metrics_knowledge_retrieval_success_rate_uses_relevant_flag():
    entries = [
        _resolved(
            "technical_resolver",
            [{"tool": "search_knowledge_base", "ok": True, "result_count": 3, "relevant": True, "error_category": None}],
        ),
        _resolved(
            "general_resolver",
            [{"tool": "search_knowledge_base", "ok": True, "result_count": 2, "relevant": False, "error_category": None}],
        ),
    ]

    metrics = compute_metrics(entries)

    assert metrics["knowledge_retrieval"]["searches"] == 2
    assert metrics["knowledge_retrieval"]["relevant_matches"] == 1
    assert metrics["knowledge_retrieval"]["success_rate"] == 0.5


def test_compute_metrics_tool_usage_patterns():
    entries = [
        _resolved(
            "billing_resolver",
            [
                {"tool": "manage_subscription", "ok": True, "result_count": 1, "error_category": None},
                {"tool": "manage_subscription", "ok": False, "result_count": 0, "error_category": "blocked_account"},
                {"tool": "get_subscription_status", "ok": True, "result_count": 1, "error_category": None},
            ],
        ),
    ]

    metrics = compute_metrics(entries)

    assert metrics["tool_usage"]["manage_subscription"] == {"calls": 2, "successes": 1, "success_rate": 0.5}
    assert metrics["tool_usage"]["get_subscription_status"] == {"calls": 1, "successes": 1, "success_rate": 1.0}


def test_compute_metrics_ignores_non_resolver_and_non_finalize_entries():
    entries = [
        {"node": "classifier", "event": "classified", "category": "billing"},
        {"node": "context_loader", "event": "loaded"},
        _finalize("resolved"),
    ]

    metrics = compute_metrics(entries)

    assert metrics["total_tickets"] == 1
    assert metrics["tool_usage"] == {}


def test_load_trace_entries_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"

    assert load_trace_entries(missing) == []


def test_load_trace_entries_reads_real_jsonl(tmp_path):
    log_path = tmp_path / "trace.jsonl"
    log_path.write_text(
        json.dumps({"node": "finalize", "event": "persisted", "final_status": "resolved"}) + "\n"
        + "\n"  # blank lines are skipped
        + json.dumps({"node": "finalize", "event": "persisted", "final_status": "escalated"}) + "\n",
        encoding="utf-8",
    )

    entries = load_trace_entries(log_path)

    assert len(entries) == 2
    assert entries[0]["final_status"] == "resolved"
    assert entries[1]["final_status"] == "escalated"


def test_format_report_is_human_readable():
    entries = [
        _finalize("resolved"),
        _finalize("escalated"),
        _resolved(
            "technical_resolver",
            [{"tool": "search_knowledge_base", "ok": True, "result_count": 1, "relevant": True, "error_category": None}],
        ),
    ]

    text = format_report(compute_metrics(entries))

    assert "Tickets processed: 2" in text
    assert "Escalation frequency: 50.0%" in text
    assert "search_knowledge_base" in text
