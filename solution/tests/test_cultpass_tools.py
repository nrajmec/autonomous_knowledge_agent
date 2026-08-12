"""Tests for agentic/tools/cultpass_tools.py.

Runs against a throwaway SQLite DB (via `temp_engine`), never the real
solution/data/external/cultpass.db.
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine

import agentic.tools.cultpass_tools as cultpass_tools
from data.models import cultpass
from utils import get_session


@pytest.fixture
def temp_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_cultpass.db'}")
    cultpass.Base.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch, temp_engine):
    monkeypatch.setattr(cultpass_tools, "get_cultpass_engine", lambda: temp_engine)


@pytest.fixture
def seeded_data(temp_engine):
    with get_session(temp_engine) as session:
        session.add_all(
            [
                cultpass.User(
                    user_id="blocked1",
                    full_name="Blocked User",
                    email="blocked@example.com",
                    is_blocked=True,
                ),
                cultpass.User(
                    user_id="active1",
                    full_name="Active User",
                    email="active@example.com",
                    is_blocked=False,
                ),
            ]
        )
        session.add(
            cultpass.Subscription(
                subscription_id="sub1",
                user_id="active1",
                status="active",
                tier="basic",
                monthly_quota=5,
                started_at=datetime.now(),
            )
        )
        session.add_all(
            [
                cultpass.Experience(
                    experience_id="exp-open",
                    title="Museum Tour",
                    description="A guided museum tour",
                    location="City",
                    when=datetime.now() + timedelta(days=5),
                    slots_available=3,
                    is_premium=False,
                ),
                cultpass.Experience(
                    experience_id="exp-full",
                    title="Sold Out Concert",
                    description="A concert",
                    location="City",
                    when=datetime.now() + timedelta(days=5),
                    slots_available=0,
                    is_premium=True,
                ),
                cultpass.Experience(
                    experience_id="exp-past",
                    title="Past Event",
                    description="Already happened",
                    location="City",
                    when=datetime.now() - timedelta(days=5),
                    slots_available=3,
                    is_premium=False,
                ),
            ]
        )
    return {"blocked_user": "blocked1", "active_user": "active1"}


# -- get_customer_profile ----------------------------------------------------


def test_get_customer_profile_known_user(seeded_data):
    result = cultpass_tools.get_customer_profile("active1")

    assert result["ok"] is True
    assert result["data"]["email"] == "active@example.com"
    assert result["data"]["is_blocked"] is False


def test_get_customer_profile_unknown_user(seeded_data):
    result = cultpass_tools.get_customer_profile("no-such-user")

    assert result["ok"] is False
    assert "No CultPass user found" in result["error"]


def test_get_customer_profile_requires_id():
    result = cultpass_tools.get_customer_profile("")

    assert result == {"ok": False, "error": "external_user_id is required"}


# -- get_subscription_status --------------------------------------------------


def test_get_subscription_status_with_subscription(seeded_data):
    result = cultpass_tools.get_subscription_status("active1")

    assert result["ok"] is True
    assert result["data"]["tier"] == "basic"


def test_get_subscription_status_no_subscription_on_file(seeded_data):
    result = cultpass_tools.get_subscription_status("blocked1")

    assert result == {"ok": True, "data": None}


def test_get_subscription_status_unknown_user(seeded_data):
    result = cultpass_tools.get_subscription_status("no-such-user")

    assert result["ok"] is False


# -- manage_subscription -------------------------------------------------------


def test_manage_subscription_change_tier(seeded_data):
    result = cultpass_tools.manage_subscription("active1", "change_tier", tier="premium")

    assert result["ok"] is True
    assert result["data"]["tier"] == "premium"


def test_manage_subscription_cancel_sets_ended_at(seeded_data):
    result = cultpass_tools.manage_subscription("active1", "cancel")

    assert result["ok"] is True
    assert result["data"]["status"] == "cancelled"
    assert result["data"]["ended_at"] is not None


def test_manage_subscription_reactivate_clears_ended_at(seeded_data):
    cultpass_tools.manage_subscription("active1", "cancel")

    result = cultpass_tools.manage_subscription("active1", "reactivate")

    assert result["ok"] is True
    assert result["data"]["status"] == "active"
    assert result["data"]["ended_at"] is None


def test_manage_subscription_invalid_action(seeded_data):
    result = cultpass_tools.manage_subscription("active1", "bogus")

    assert result["ok"] is False
    assert "action must be one of" in result["error"]


def test_manage_subscription_change_tier_requires_valid_tier(seeded_data):
    result = cultpass_tools.manage_subscription("active1", "change_tier", tier="ultra")

    assert result["ok"] is False
    assert "tier must be one of" in result["error"]


def test_manage_subscription_user_without_subscription(seeded_data):
    result = cultpass_tools.manage_subscription("blocked1", "cancel")

    assert result["ok"] is False
    assert "no subscription on file" in result["error"]


def test_manage_subscription_unknown_user(seeded_data):
    result = cultpass_tools.manage_subscription("no-such-user", "cancel")

    assert result["ok"] is False


# -- list_reservations ---------------------------------------------------------


def test_list_reservations_empty_for_new_user(seeded_data):
    result = cultpass_tools.list_reservations("active1")

    assert result == {"ok": True, "data": []}


def test_list_reservations_unknown_user(seeded_data):
    result = cultpass_tools.list_reservations("no-such-user")

    assert result["ok"] is False


# -- manage_reservation ---------------------------------------------------------


def test_manage_reservation_book_decrements_slots(seeded_data):
    result = cultpass_tools.manage_reservation("active1", "book", experience_id="exp-open")

    assert result["ok"] is True
    assert result["data"]["experience_id"] == "exp-open"

    remaining = cultpass_tools.search_experiences(upcoming_only=False)
    exp = next(e for e in remaining["data"] if e["experience_id"] == "exp-open")
    assert exp["slots_available"] == 2


def test_manage_reservation_blocked_user_cannot_book(seeded_data):
    result = cultpass_tools.manage_reservation("blocked1", "book", experience_id="exp-open")

    assert result == {"ok": False, "error": "Account is blocked; cannot book a reservation"}


def test_manage_reservation_book_full_experience_fails(seeded_data):
    result = cultpass_tools.manage_reservation("active1", "book", experience_id="exp-full")

    assert result["ok"] is False
    assert "no slots available" in result["error"]


def test_manage_reservation_book_unknown_experience(seeded_data):
    result = cultpass_tools.manage_reservation("active1", "book", experience_id="no-such-exp")

    assert result["ok"] is False


def test_manage_reservation_cancel_round_trip_restores_slots(seeded_data):
    booked = cultpass_tools.manage_reservation("active1", "book", experience_id="exp-open")

    cancelled = cultpass_tools.manage_reservation(
        "active1", "cancel", reservation_id=booked["data"]["reservation_id"]
    )

    assert cancelled["ok"] is True
    assert cancelled["data"]["status"] == "cancelled"

    remaining = cultpass_tools.search_experiences(upcoming_only=False)
    exp = next(e for e in remaining["data"] if e["experience_id"] == "exp-open")
    assert exp["slots_available"] == 3


def test_manage_reservation_cancel_unknown_reservation(seeded_data):
    result = cultpass_tools.manage_reservation("active1", "cancel", reservation_id="no-such-res")

    assert result["ok"] is False


def test_manage_reservation_validation_errors(seeded_data):
    assert cultpass_tools.manage_reservation("active1", "bogus")["ok"] is False
    assert cultpass_tools.manage_reservation("active1", "book")["ok"] is False
    assert cultpass_tools.manage_reservation("active1", "cancel")["ok"] is False


# -- search_experiences ---------------------------------------------------------


def test_search_experiences_upcoming_only_excludes_full_and_past(seeded_data):
    result = cultpass_tools.search_experiences()

    ids = {e["experience_id"] for e in result["data"]}
    assert ids == {"exp-open"}


def test_search_experiences_query_filters_by_keyword(seeded_data):
    result = cultpass_tools.search_experiences(query="museum")

    assert len(result["data"]) == 1
    assert result["data"][0]["experience_id"] == "exp-open"


def test_search_experiences_upcoming_only_false_returns_everything(seeded_data):
    result = cultpass_tools.search_experiences(upcoming_only=False)

    ids = {e["experience_id"] for e in result["data"]}
    assert ids == {"exp-open", "exp-full", "exp-past"}
