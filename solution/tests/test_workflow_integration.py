"""End-to-end tests for the compiled `agentic.workflow.orchestrator` graph.

Runs the *real* graph wiring, the *real* tool logic (RAG search, DB reads/
writes, long-term memory), against throwaway temp SQLite DBs -- only the
LLM calls are faked (via each agent module's lazy `_default_llm` cache),
since there's no OPENAI_API_KEY in this dev environment. This is the
closest thing to a real run this environment can produce: it proves the
graph is wired correctly end-to-end (context loading, classification,
routing, tool use, resolution/escalation, persistence), even though the
LLMs' actual judgment isn't being exercised.
"""
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine

import agentic.agents.classifier as classifier_module
import agentic.agents.escalation as escalation_module
import agentic.agents.resolver as resolver_module
import agentic.tools.cultpass_tools as cultpass_tools
import agentic.tools.knowledge_tools as knowledge_tools
import agentic.tools.memory_tools as memory_tools
import agentic.tools.udahub_tools as udahub_tools
from agentic.agents.classifier import ClassificationSchema
from agentic.agents.escalation import EscalationOutput
from agentic.agents.resolver import ResolverOutput
from agentic.workflow import orchestrator
from data.models import cultpass, udahub
from tests.fakes import FakeChatModel
from utils import get_session

_VOCAB = ["refund", "login", "password", "email", "premium", "cancel", "reservation"]


def _fake_embed_texts(texts):
    return [[1.0 if word in t.lower() else 0.0 for word in _VOCAB] for t in texts]


def _tool_call_message(name, args, call_id="call1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _stop_message():
    return AIMessage(content="done", tool_calls=[])


@pytest.fixture(autouse=True)
def _isolated_databases(monkeypatch, tmp_path):
    cultpass_engine = create_engine(f"sqlite:///{tmp_path / 'test_cultpass.db'}")
    cultpass.Base.metadata.create_all(cultpass_engine)
    monkeypatch.setattr(cultpass_tools, "get_cultpass_engine", lambda: cultpass_engine)

    udahub_engine = create_engine(f"sqlite:///{tmp_path / 'test_udahub.db'}")
    udahub.Base.metadata.create_all(udahub_engine)
    monkeypatch.setattr(udahub_tools, "get_udahub_engine", lambda: udahub_engine)
    monkeypatch.setattr(knowledge_tools, "get_udahub_engine", lambda: udahub_engine)
    monkeypatch.setattr(memory_tools, "get_udahub_engine", lambda: udahub_engine)

    monkeypatch.setattr(knowledge_tools, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(memory_tools, "embed_texts", _fake_embed_texts)
    knowledge_tools.invalidate_knowledge_cache()

    with get_session(udahub_engine) as session:
        session.add(udahub.Account(account_id="acc1", account_name="Test Account"))
        session.add(
            udahub.Knowledge(
                article_id="a-login",
                account_id="acc1",
                title="How to Reset Your Password",
                content="If you can't log in, use the password reset link on the login page.",
                tags="login, password, access",
            )
        )

    yield {"cultpass_engine": cultpass_engine, "udahub_engine": udahub_engine}

    knowledge_tools.invalidate_knowledge_cache()


def _seed_customer(udahub_engine, cultpass_engine, *, external_user_id, is_blocked, ticket_id):
    with get_session(cultpass_engine) as session:
        session.add(
            cultpass.User(
                user_id=external_user_id,
                full_name="Test Customer",
                email=f"{external_user_id}@example.com",
                is_blocked=is_blocked,
            )
        )

    internal_user_id = f"user-{external_user_id}"
    with get_session(udahub_engine) as session:
        session.add(
            udahub.User(
                user_id=internal_user_id,
                account_id="acc1",
                external_user_id=external_user_id,
                user_name="Test Customer",
            )
        )
        session.add(
            udahub.Ticket(ticket_id=ticket_id, account_id="acc1", user_id=internal_user_id, channel="chat")
        )
        session.add(udahub.TicketMetadata(ticket_id=ticket_id, status="open", tags="login"))

    return internal_user_id


def test_full_graph_resolves_ticket_via_technical_resolver(_isolated_databases):
    engines = _isolated_databases
    _seed_customer(
        engines["udahub_engine"],
        engines["cultpass_engine"],
        external_user_id="ext-happy",
        is_blocked=False,
        ticket_id="ticket-happy",
    )

    classifier_module._default_llm = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="technical",
                urgency="medium",
                sentiment="neutral",
                complexity="simple",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    resolver_module._default_llm = FakeChatModel(
        tool_loop_responses=[
            _tool_call_message("search_knowledge_base", {"query": "can't log in"}),
            _stop_message(),
        ],
        structured_responses=[
            ResolverOutput(
                response="Please use the password reset link on the login page.",
                cited_article_ids=["a-login"],
                confidence=0.9,
                escalate=False,
            )
        ],
    )

    trigger = {
        "ticket_id": "ticket-happy",
        "account_id": "acc1",
        "external_user_id": "ext-happy",
        "channel": "chat",
        "ticket_text": "I can't log in to my account.",
        "messages": [],
    }
    result = orchestrator.invoke(trigger, config={"configurable": {"thread_id": "ticket-happy"}})

    assert result["final_status"] == "resolved"
    assert result["confidence"] == 0.9
    assert "password reset link" in result["messages"][-1].content

    with get_session(engines["udahub_engine"]) as session:
        meta = session.get(udahub.TicketMetadata, "ticket-happy")
        assert meta.status == "resolved"
        messages = session.query(udahub.TicketMessage).filter_by(ticket_id="ticket-happy").all()
        assert len(messages) == 1
        memories = session.query(udahub.CustomerMemory).filter_by(user_id="user-ext-happy").all()
        assert len(memories) == 1


def test_full_graph_escalates_blocked_account(_isolated_databases):
    engines = _isolated_databases
    _seed_customer(
        engines["udahub_engine"],
        engines["cultpass_engine"],
        external_user_id="ext-blocked",
        is_blocked=True,
        ticket_id="ticket-blocked",
    )

    classifier_module._default_llm = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="account",
                urgency="high",
                sentiment="negative",
                complexity="simple",
                is_repeat_issue=False,
                hard_escalate=True,
                hard_escalate_reason="Account is blocked",
            )
        ]
    )
    escalation_module._default_llm = FakeChatModel(
        structured_responses=[
            EscalationOutput(
                internal_summary="Customer's account is blocked; needs manual review.",
                customer_message="Your request has been passed to a specialist who will follow up shortly.",
            )
        ]
    )

    trigger = {
        "ticket_id": "ticket-blocked",
        "account_id": "acc1",
        "external_user_id": "ext-blocked",
        "channel": "chat",
        "ticket_text": "Why is my account blocked?! Let me speak to a person.",
        "messages": [],
    }
    result = orchestrator.invoke(trigger, config={"configurable": {"thread_id": "ticket-blocked"}})

    assert result["final_status"] == "escalated"
    assert "specialist" in result["messages"][-1].content

    with get_session(engines["udahub_engine"]) as session:
        meta = session.get(udahub.TicketMetadata, "ticket-blocked")
        assert meta.status == "escalated"
        # Escalated tickets shouldn't get a resolution-summary memory entry.
        memories = session.query(udahub.CustomerMemory).filter_by(user_id="user-ext-blocked").all()
        assert len(memories) == 0
