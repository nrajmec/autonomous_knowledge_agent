"""Tools that abstract read access to the CultPass (external) database.

CultPass is UDA-Hub's customer, not UDA-Hub itself: `cultpass.db` belongs to
their system, and this module is the *only* place agent code is allowed to
touch it directly. Every function here returns a plain, structured dict
(``{"ok": True, "data": ...}`` or ``{"ok": False, "error": "..."}``) rather
than raising, so agents can inspect the outcome and decide whether to lower
their confidence / escalate instead of crashing the graph on a bad lookup.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from data.models import cultpass
from utils import get_session

from agentic.tools.db import get_cultpass_engine


def get_customer_profile(external_user_id: str) -> dict[str, Any]:
    """Look up a CultPass customer's profile.

    Args:
        external_user_id: The CultPass `user_id` (this is the same value
            stored as `external_user_id` on the UDA-Hub `User` record).

    Returns:
        On success: {"ok": True, "data": {"user_id", "full_name", "email",
        "is_blocked"}}.
        On failure: {"ok": False, "error": "<human-readable reason>"}.
    """
    if not external_user_id:
        return {"ok": False, "error": "external_user_id is required"}

    with get_session(get_cultpass_engine()) as session:
        user = session.get(cultpass.User, external_user_id)
        if user is None:
            return {
                "ok": False,
                "error": f"No CultPass user found for id '{external_user_id}'",
            }

        return {
            "ok": True,
            "data": {
                "user_id": user.user_id,
                "full_name": user.full_name,
                "email": user.email,
                "is_blocked": user.is_blocked,
            },
        }


def get_subscription_status(external_user_id: str) -> dict[str, Any]:
    """Look up a CultPass customer's current subscription.

    Args:
        external_user_id: The CultPass `user_id`.

    Returns:
        On success with a subscription: {"ok": True, "data": {"subscription_id",
        "status", "tier", "monthly_quota", "started_at", "ended_at"}}.
        On success with no subscription on file: {"ok": True, "data": None}.
        On failure (unknown user): {"ok": False, "error": "<reason>"}.
    """
    if not external_user_id:
        return {"ok": False, "error": "external_user_id is required"}

    with get_session(get_cultpass_engine()) as session:
        user = session.get(cultpass.User, external_user_id)
        if user is None:
            return {
                "ok": False,
                "error": f"No CultPass user found for id '{external_user_id}'",
            }

        sub = user.subscription
        if sub is None:
            return {"ok": True, "data": None}

        return {
            "ok": True,
            "data": {
                "subscription_id": sub.subscription_id,
                "status": sub.status,
                "tier": sub.tier,
                "monthly_quota": sub.monthly_quota,
                "started_at": sub.started_at.isoformat() if sub.started_at else None,
                "ended_at": sub.ended_at.isoformat() if sub.ended_at else None,
            },
        }


_VALID_SUBSCRIPTION_ACTIONS = {"cancel", "reactivate", "change_tier"}
_VALID_TIERS = {"basic", "premium"}


def manage_subscription(
    external_user_id: str, action: str, tier: str | None = None
) -> dict[str, Any]:
    """Cancel, reactivate, or change the tier of a CultPass subscription.

    Args:
        external_user_id: The CultPass `user_id`.
        action: One of "cancel", "reactivate", "change_tier".
        tier: Required when action == "change_tier". One of "basic", "premium".

    Returns:
        {"ok": True, "data": {...updated subscription...}} on success, or
        {"ok": False, "error": "<reason>"} on validation failure, or if the
        user/subscription does not exist.
    """
    if not external_user_id:
        return {"ok": False, "error": "external_user_id is required"}
    if action not in _VALID_SUBSCRIPTION_ACTIONS:
        return {
            "ok": False,
            "error": f"action must be one of {sorted(_VALID_SUBSCRIPTION_ACTIONS)}, got '{action}'",
        }
    if action == "change_tier" and tier not in _VALID_TIERS:
        return {
            "ok": False,
            "error": f"tier must be one of {sorted(_VALID_TIERS)} for change_tier, got '{tier}'",
        }

    with get_session(get_cultpass_engine()) as session:
        user = session.get(cultpass.User, external_user_id)
        if user is None:
            return {"ok": False, "error": f"No CultPass user found for id '{external_user_id}'"}

        sub = user.subscription
        if sub is None:
            return {"ok": False, "error": f"User '{external_user_id}' has no subscription on file"}

        if action == "cancel":
            sub.status = "cancelled"
            sub.ended_at = datetime.now()
        elif action == "reactivate":
            sub.status = "active"
            sub.ended_at = None
        elif action == "change_tier":
            sub.tier = tier

        session.flush()
        return {
            "ok": True,
            "data": {
                "subscription_id": sub.subscription_id,
                "status": sub.status,
                "tier": sub.tier,
                "monthly_quota": sub.monthly_quota,
                "started_at": sub.started_at.isoformat() if sub.started_at else None,
                "ended_at": sub.ended_at.isoformat() if sub.ended_at else None,
            },
        }


def list_reservations(external_user_id: str) -> dict[str, Any]:
    """List a CultPass customer's reservations, soonest experience first.

    Args:
        external_user_id: The CultPass `user_id`.

    Returns:
        {"ok": True, "data": [{"reservation_id", "status", "experience_id",
        "experience_title", "when"}, ...]} on success, or
        {"ok": False, "error": "<reason>"} if the user does not exist.
    """
    if not external_user_id:
        return {"ok": False, "error": "external_user_id is required"}

    with get_session(get_cultpass_engine()) as session:
        user = session.get(cultpass.User, external_user_id)
        if user is None:
            return {"ok": False, "error": f"No CultPass user found for id '{external_user_id}'"}

        reservations = sorted(
            user.reservations, key=lambda r: r.experience.when if r.experience else datetime.max
        )
        return {
            "ok": True,
            "data": [
                {
                    "reservation_id": r.reservation_id,
                    "status": r.status,
                    "experience_id": r.experience_id,
                    "experience_title": r.experience.title if r.experience else None,
                    "when": r.experience.when.isoformat() if r.experience and r.experience.when else None,
                }
                for r in reservations
            ],
        }


_VALID_RESERVATION_ACTIONS = {"book", "cancel"}


def manage_reservation(
    external_user_id: str,
    action: str,
    experience_id: str | None = None,
    reservation_id: str | None = None,
) -> dict[str, Any]:
    """Book or cancel a CultPass experience reservation.

    Args:
        external_user_id: The CultPass `user_id`.
        action: "book" (requires experience_id) or "cancel" (requires reservation_id).
        experience_id: Required when action == "book".
        reservation_id: Required when action == "cancel".

    Returns:
        {"ok": True, "data": {...reservation...}} on success, or
        {"ok": False, "error": "<reason>"} on validation failure, a blocked
        account, a full/unknown experience, or an unknown reservation.
    """
    if not external_user_id:
        return {"ok": False, "error": "external_user_id is required"}
    if action not in _VALID_RESERVATION_ACTIONS:
        return {
            "ok": False,
            "error": f"action must be one of {sorted(_VALID_RESERVATION_ACTIONS)}, got '{action}'",
        }
    if action == "book" and not experience_id:
        return {"ok": False, "error": "experience_id is required to book a reservation"}
    if action == "cancel" and not reservation_id:
        return {"ok": False, "error": "reservation_id is required to cancel a reservation"}

    with get_session(get_cultpass_engine()) as session:
        user = session.get(cultpass.User, external_user_id)
        if user is None:
            return {"ok": False, "error": f"No CultPass user found for id '{external_user_id}'"}

        if action == "book":
            if user.is_blocked:
                return {"ok": False, "error": "Account is blocked; cannot book a reservation"}

            experience = session.get(cultpass.Experience, experience_id)
            if experience is None:
                return {"ok": False, "error": f"No experience found for id '{experience_id}'"}
            if experience.slots_available <= 0:
                return {"ok": False, "error": f"Experience '{experience_id}' has no slots available"}

            reservation = cultpass.Reservation(
                reservation_id=str(uuid.uuid4())[:6],
                user_id=external_user_id,
                experience_id=experience_id,
                status="reserved",
            )
            experience.slots_available -= 1
            session.add(reservation)
            session.flush()

            return {
                "ok": True,
                "data": {
                    "reservation_id": reservation.reservation_id,
                    "status": reservation.status,
                    "experience_id": experience.experience_id,
                    "experience_title": experience.title,
                    "when": experience.when.isoformat() if experience.when else None,
                },
            }

        # action == "cancel"
        reservation = session.get(cultpass.Reservation, reservation_id)
        if reservation is None or reservation.user_id != external_user_id:
            return {
                "ok": False,
                "error": f"No reservation '{reservation_id}' found for user '{external_user_id}'",
            }
        if reservation.status == "cancelled":
            return {"ok": False, "error": f"Reservation '{reservation_id}' is already cancelled"}

        reservation.status = "cancelled"
        if reservation.experience is not None:
            reservation.experience.slots_available += 1
        session.flush()

        return {
            "ok": True,
            "data": {
                "reservation_id": reservation.reservation_id,
                "status": reservation.status,
                "experience_id": reservation.experience_id,
            },
        }


def search_experiences(query: str = "", upcoming_only: bool = True) -> dict[str, Any]:
    """Search CultPass experiences by keyword.

    Args:
        query: Case-insensitive substring matched against title, description,
            and location. Empty string matches everything.
        upcoming_only: If True (default), only return experiences scheduled
            in the future with at least one open slot.

    Returns:
        {"ok": True, "data": [{"experience_id", "title", "description",
        "location", "when", "slots_available", "is_premium"}, ...]}.
    """
    needle = (query or "").strip().lower()

    with get_session(get_cultpass_engine()) as session:
        experiences = session.query(cultpass.Experience).all()

        results = []
        for exp in experiences:
            if upcoming_only and (
                exp.when is None or exp.when < datetime.now() or exp.slots_available <= 0
            ):
                continue
            haystack = " ".join([exp.title or "", exp.description or "", exp.location or ""]).lower()
            if needle and needle not in haystack:
                continue
            results.append(
                {
                    "experience_id": exp.experience_id,
                    "title": exp.title,
                    "description": exp.description,
                    "location": exp.location,
                    "when": exp.when.isoformat() if exp.when else None,
                    "slots_available": exp.slots_available,
                    "is_premium": exp.is_premium,
                }
            )

        results.sort(key=lambda r: r["when"] or "")
        return {"ok": True, "data": results}
