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
from langchain_core.messages import AIMessage, HumanMessage
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

_VOCAB = [
    "refund", "login", "password", "email", "premium", "cancel", "reservation",
    "subscription", "included", "benefits",
]


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
        session.add(
            udahub.Knowledge(
                article_id="a-subscription",
                account_id="acc1",
                title="What's Included in a CultPass Subscription",
                content="A CultPass subscription includes access to curated cultural experiences each month.",
                tags="subscription, benefits, included",
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


def _seed_additional_ticket(udahub_engine, *, internal_user_id, ticket_id):
    """A second ticket for a customer who already has a User row (see
    _seed_customer) -- used to test cross-session behavior for the same
    customer without re-creating the User."""
    with get_session(udahub_engine) as session:
        session.add(
            udahub.Ticket(ticket_id=ticket_id, account_id="acc1", user_id=internal_user_id, channel="chat")
        )
        session.add(udahub.TicketMetadata(ticket_id=ticket_id, status="open", tags=""))


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


def test_full_graph_resolves_normal_faq_via_general_resolver(_isolated_databases):
    """Varied-scenario #1: a plain FAQ question, grounded in a real
    knowledge-base search against the real (temp) database."""
    engines = _isolated_databases
    _seed_customer(
        engines["udahub_engine"],
        engines["cultpass_engine"],
        external_user_id="ext-faq",
        is_blocked=False,
        ticket_id="ticket-faq",
    )

    classifier_module._default_llm = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="general",
                urgency="low",
                sentiment="neutral",
                complexity="simple",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    resolver_module._default_llm = FakeChatModel(
        tool_loop_responses=[
            _tool_call_message("search_knowledge_base", {"query": "what's included in my subscription"}),
            _stop_message(),
        ],
        structured_responses=[
            ResolverOutput(
                response="A CultPass subscription includes access to curated cultural experiences each month.",
                cited_article_ids=["a-subscription"],
                confidence=0.85,
                escalate=False,
            )
        ],
    )

    trigger = {
        "ticket_id": "ticket-faq",
        "account_id": "acc1",
        "external_user_id": "ext-faq",
        "channel": "chat",
        "ticket_text": "What's included in my subscription?",
        "messages": [],
    }
    result = orchestrator.invoke(trigger, config={"configurable": {"thread_id": "ticket-faq"}})

    assert result["final_status"] == "resolved"
    assert result["classification"]["category"] == "general"
    assert "curated cultural experiences" in result["messages"][-1].content

    with get_session(engines["udahub_engine"]) as session:
        meta = session.get(udahub.TicketMetadata, "ticket-faq")
        assert meta.status == "resolved"


def test_full_graph_urgent_booking_performs_real_tool_driven_action(_isolated_databases):
    """Varied-scenario #2: an urgent, tool-driven action -- the resolver
    searches real experiences then books a real reservation, and the
    booking is verified directly against the (temp) CultPass database."""
    engines = _isolated_databases
    _seed_customer(
        engines["udahub_engine"],
        engines["cultpass_engine"],
        external_user_id="ext-urgent",
        is_blocked=False,
        ticket_id="ticket-urgent",
    )
    with get_session(engines["cultpass_engine"]) as session:
        session.add(
            cultpass.Experience(
                experience_id="exp-carnival",
                title="Carnival Night",
                description="A late-night carnival celebration.",
                location="Rio de Janeiro",
                when=datetime.now() + timedelta(hours=6),
                slots_available=5,
                is_premium=False,
            )
        )

    classifier_module._default_llm = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="booking",
                urgency="high",
                sentiment="neutral",
                complexity="complex",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    resolver_module._default_llm = FakeChatModel(
        tool_loop_responses=[
            _tool_call_message("search_experiences", {"query": "carnival"}),
            _tool_call_message("manage_reservation", {"action": "book", "experience_id": "exp-carnival"}),
            _stop_message(),
        ],
        structured_responses=[
            ResolverOutput(
                response="You're booked for Carnival Night tonight!",
                confidence=0.95,
                escalate=False,
            )
        ],
    )

    trigger = {
        "ticket_id": "ticket-urgent",
        "account_id": "acc1",
        "external_user_id": "ext-urgent",
        "channel": "chat",
        "reported_urgency": "high",
        "ticket_text": "I need to book tonight's carnival event urgently, it's the last night!",
        "messages": [],
    }
    result = orchestrator.invoke(trigger, config={"configurable": {"thread_id": "ticket-urgent"}})

    assert result["final_status"] == "resolved"
    assert result["classification"]["urgency"] == "high"

    with get_session(engines["cultpass_engine"]) as session:
        reservations = session.query(cultpass.Reservation).filter_by(user_id="ext-urgent").all()
        assert len(reservations) == 1
        assert reservations[0].experience_id == "exp-carnival"
        experience = session.get(cultpass.Experience, "exp-carnival")
        assert experience.slots_available == 4  # decremented by the real booking


def test_full_graph_resolver_triggered_escalation_on_missing_knowledge(_isolated_databases):
    """Varied-scenario #3: distinct from the hard-escalate bypass test above
    -- here the Classifier does NOT flag hard_escalate, a resolver actually
    runs, finds no relevant knowledge (a real search against the real KB,
    which genuinely has nothing on this topic), and escalates itself on low
    confidence. Exercises supervisor's *second*-pass escalation branch."""
    engines = _isolated_databases
    _seed_customer(
        engines["udahub_engine"],
        engines["cultpass_engine"],
        external_user_id="ext-noinfo",
        is_blocked=False,
        ticket_id="ticket-noinfo",
    )

    classifier_module._default_llm = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="general",
                urgency="medium",
                sentiment="neutral",
                complexity="moderate",
                is_repeat_issue=False,
                hard_escalate=False,
            )
        ]
    )
    resolver_module._default_llm = FakeChatModel(
        tool_loop_responses=[
            _tool_call_message("search_knowledge_base", {"query": "unrelated obscure topic"}),
            _stop_message(),
        ],
        structured_responses=[
            ResolverOutput(
                response="I'm not able to find information on that.",
                confidence=0.3,
                escalate=True,
                escalation_reason="No relevant knowledge article found",
            )
        ],
    )
    escalation_module._default_llm = FakeChatModel(
        structured_responses=[
            EscalationOutput(
                internal_summary="No matching knowledge article; needs a human to look into this.",
                customer_message="Your request has been passed to a specialist who will follow up shortly.",
            )
        ]
    )

    trigger = {
        "ticket_id": "ticket-noinfo",
        "account_id": "acc1",
        "external_user_id": "ext-noinfo",
        "channel": "chat",
        "ticket_text": "I have a strange issue nobody seems able to explain.",
        "messages": [],
    }
    result = orchestrator.invoke(trigger, config={"configurable": {"thread_id": "ticket-noinfo"}})

    assert result["final_status"] == "escalated"
    assert result["route"] == "escalation"

    with get_session(engines["udahub_engine"]) as session:
        meta = session.get(udahub.TicketMetadata, "ticket-noinfo")
        assert meta.status == "escalated"
        memories = session.query(udahub.CustomerMemory).filter_by(user_id="user-ext-noinfo").all()
        assert len(memories) == 0


def test_same_session_second_turn_depends_on_first_turn(_isolated_databases):
    """Short-term/session memory actually reaching agent reasoning: two
    turns, same thread_id, and the second turn's prompt genuinely contains
    what was said in the first -- not just re-run from ticket_text alone."""
    engines = _isolated_databases
    _seed_customer(
        engines["udahub_engine"],
        engines["cultpass_engine"],
        external_user_id="ext-multiturn",
        is_blocked=False,
        ticket_id="ticket-multiturn",
    )

    classifier_fake = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="billing", urgency="medium", sentiment="neutral", complexity="simple",
                is_repeat_issue=False, hard_escalate=False,
            ),
            ClassificationSchema(
                category="billing", urgency="medium", sentiment="neutral", complexity="simple",
                is_repeat_issue=False, hard_escalate=False,
            ),
        ]
    )
    resolver_fake = FakeChatModel(
        tool_loop_responses=[_stop_message(), _stop_message()],
        structured_responses=[
            ResolverOutput(
                response="Got it, I've noted your premium plan and the declined card.",
                confidence=0.8,
                escalate=False,
            ),
            ResolverOutput(
                response="I'll get the declined card on your premium plan sorted out now.",
                confidence=0.85,
                escalate=False,
            ),
        ],
    )
    classifier_module._default_llm = classifier_fake
    resolver_module._default_llm = resolver_fake

    config = {"configurable": {"thread_id": "ticket-multiturn"}}
    base_trigger = {
        "ticket_id": "ticket-multiturn",
        "account_id": "acc1",
        "external_user_id": "ext-multiturn",
        "channel": "chat",
    }

    # messages carries a real HumanMessage here (not the empty [] the other
    # tests use), matching how utils.chat_interface() actually invokes the
    # graph each turn -- needed so the customer's own words are part of the
    # session history this test is checking.
    turn1_text = "I'm on the premium plan and my card was declined."
    orchestrator.invoke(
        {**base_trigger, "ticket_text": turn1_text, "messages": [HumanMessage(content=turn1_text)]}, config=config
    )

    turn2_text = "can you fix the issue we discussed?"
    result2 = orchestrator.invoke(
        {**base_trigger, "ticket_text": turn2_text, "messages": [HumanMessage(content=turn2_text)]}, config=config
    )

    assert result2["final_status"] == "resolved"

    # Regression guard: both turns must have genuinely gone all the way
    # through Classifier and the Resolver a *second* time, not just once
    # with turn 2 short-circuiting straight to Finalize on a stale
    # confidence left over in state from turn 1. FakeChatModel only pops
    # from these queues on a real .invoke() call, so an empty queue proves
    # both fakes were actually called twice.
    assert classifier_fake._structured_responses == []
    assert resolver_fake._structured_responses == []
    assert resolver_fake._tool_loop_responses == []
    assert len(resolver_fake.captured_tool_loop_messages) == 2

    # The resolver's second prompt must actually carry turn 1's content --
    # this is the whole point: a follow-up that only makes sense in light
    # of what was said before it.
    second_call_human_message = resolver_fake.captured_tool_loop_messages[-1][1]
    assert "premium plan" in second_call_human_message.content
    assert "declined" in second_call_human_message.content

    # And LangGraph's own session memory (the checkpointer) really did
    # accumulate both turns under the one thread_id -- invoke() returns the
    # full current state, not just this turn's delta.
    human_texts = [m.content for m in result2["messages"] if isinstance(m, HumanMessage)]
    assert turn1_text in human_texts
    assert turn2_text in human_texts


def test_cross_session_preference_saved_then_recalled_on_a_different_ticket(_isolated_databases):
    """Long-term memory actually reaching agent reasoning: a preference
    stated on one ticket is saved via save_customer_memory(..., "preference",
    ...), and a *different* ticket/session for the same customer recalls it
    via context_loader -- real DB round-trip, two separate thread_ids."""
    engines = _isolated_databases
    internal_user_id = _seed_customer(
        engines["udahub_engine"],
        engines["cultpass_engine"],
        external_user_id="ext-prefs",
        is_blocked=False,
        ticket_id="ticket-prefs-1",
    )

    classifier_module._default_llm = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="account", urgency="low", sentiment="neutral", complexity="simple",
                is_repeat_issue=False, hard_escalate=False,
            )
        ]
    )
    resolver_module._default_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[
            ResolverOutput(
                response="Understood, I've noted you'd prefer email over phone contact.",
                confidence=0.9,
                escalate=False,
                detected_preference="Prefers email contact over phone calls",
            )
        ],
    )

    session1_trigger = {
        "ticket_id": "ticket-prefs-1",
        "account_id": "acc1",
        "external_user_id": "ext-prefs",
        "channel": "chat",
        "ticket_text": "Please only contact me by email from now on, not phone.",
        "messages": [],
    }
    result1 = orchestrator.invoke(session1_trigger, config={"configurable": {"thread_id": "ticket-prefs-1"}})
    assert result1["final_status"] == "resolved"

    with get_session(engines["udahub_engine"]) as session:
        prefs = (
            session.query(udahub.CustomerMemory)
            .filter_by(user_id=internal_user_id, memory_type="preference")
            .all()
        )
        assert len(prefs) == 1
        assert "email" in prefs[0].content

    # A second, independent ticket/session for the SAME customer.
    _seed_additional_ticket(engines["udahub_engine"], internal_user_id=internal_user_id, ticket_id="ticket-prefs-2")

    classifier_module._default_llm = FakeChatModel(
        structured_responses=[
            ClassificationSchema(
                category="account", urgency="low", sentiment="neutral", complexity="simple",
                is_repeat_issue=False, hard_escalate=False,
            )
        ]
    )
    resolver_module._default_llm = FakeChatModel(
        tool_loop_responses=[_stop_message()],
        structured_responses=[
            ResolverOutput(response="Sure, I can help with that.", confidence=0.8, escalate=False)
        ],
    )

    session2_trigger = {
        "ticket_id": "ticket-prefs-2",
        "account_id": "acc1",
        "external_user_id": "ext-prefs",
        "channel": "email",
        "ticket_text": "How do I update my contact preferences?",
        "messages": [],
    }
    result2 = orchestrator.invoke(session2_trigger, config={"configurable": {"thread_id": "ticket-prefs-2"}})

    # The Context Loader on this brand-new ticket/thread recalled the
    # preference saved during the *other* session -- proof long-term memory
    # crosses sessions, not just persists within one.
    recalled = result2["user_context"]["long_term_memories"]
    assert any(m.get("memory_type") == "preference" and "email" in m.get("content", "") for m in recalled)
