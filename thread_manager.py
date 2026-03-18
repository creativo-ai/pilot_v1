"""
thread_manager.py
Manages per-agent per-user conversation threads persisted in Firestore.

Firestore collection: agent_threads
Document ID: {user_id}__{agent_name}
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

PROJECT_ID = "creativo-bf5c8"
db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")

COLLECTION = "agent_threads"


def _doc_id(user_id: str, agent_name: str) -> str:
    return f"{user_id}__{agent_name}"


# ── Load / Save thread ────────────────────────────────────────────────────────

def load_thread(user_id: str, agent_name: str) -> dict:
    """
    Returns the full thread document dict, or an empty default.
    Keys: messages, summary, collected_fields, last_active, status, brand_id
    """
    doc = db.collection(COLLECTION).document(_doc_id(user_id, agent_name)).get()
    if doc.exists:
        return doc.to_dict()
    return {
        "user_id": user_id,
        "agent_name": agent_name,
        "brand_id": None,
        "messages": [],
        "summary": "",
        "collected_fields": {},
        "last_active": None,
        "status": "active",       # active | finalized
        "created_at": None,
    }


def save_thread(user_id: str, agent_name: str, thread: dict) -> None:
    """Persist the full thread document, updating last_active timestamp."""
    now = datetime.now(timezone.utc)
    thread["last_active"] = now
    thread["status"] = "active"  # always reset to active on save — finalizer will re-run if needed
    if not thread.get("created_at"):
        thread["created_at"] = now
    thread["user_id"] = user_id
    thread["agent_name"] = agent_name

    db.collection(COLLECTION).document(
        _doc_id(user_id, agent_name)
    ).set(thread)


# ── Message helpers ───────────────────────────────────────────────────────────

def append_messages(thread: dict, user_input: str, assistant_response: str) -> dict:
    """Append a user+assistant turn to thread messages. Returns updated thread."""
    thread.setdefault("messages", [])
    thread["messages"].append({"role": "user", "content": user_input})
    thread["messages"].append({"role": "assistant", "content": assistant_response})
    return thread


def get_messages(thread: dict) -> list:
    return thread.get("messages", [])


# ── Summary ───────────────────────────────────────────────────────────────────

def update_summary(user_id: str, agent_name: str, summary: str) -> None:
    db.collection(COLLECTION).document(
        _doc_id(user_id, agent_name)
    ).update({"summary": summary})


def get_summary(user_id: str, agent_name: str) -> str:
    doc = db.collection(COLLECTION).document(_doc_id(user_id, agent_name)).get()
    return doc.to_dict().get("summary", "") if doc.exists else ""


# ── Onboarding-specific fields ────────────────────────────────────────────────

def update_collected_fields(user_id: str, agent_name: str, new_fields: dict) -> None:
    """Merge new extracted fields into collected_fields."""
    doc_ref = db.collection(COLLECTION).document(_doc_id(user_id, agent_name))
    doc = doc_ref.get()
    existing = doc.to_dict() if doc.exists else {}

    merged_fields = {**existing.get("collected_fields", {}), **new_fields}
    doc_ref.update({"collected_fields": merged_fields})


def get_collected_fields(user_id: str, agent_name: str) -> dict:
    doc = db.collection(COLLECTION).document(_doc_id(user_id, agent_name)).get()
    return doc.to_dict().get("collected_fields", {}) if doc.exists else {}


def mark_finalized(user_id: str, agent_name: str) -> None:
    db.collection(COLLECTION).document(
        _doc_id(user_id, agent_name)
    ).update({"status": "finalized"})


def touch_thread(user_id: str, agent_name: str) -> None:
    """
    Reset last_active to now without changing any other fields.
    Called by background brand captures to reset the 5-min inactivity timer
    so all brand info accumulates in one session before finalisation.
    Also re-activates a finalized thread if new info arrives.
    """
    now = datetime.now(timezone.utc)
    doc_ref = db.collection(COLLECTION).document(_doc_id(user_id, agent_name))
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.update({"last_active": now, "status": "active"})
    else:
        doc_ref.set({
            "user_id": user_id,
            "agent_name": agent_name,
            "brand_id": None,
            "messages": [],
            "summary": "",
            "collected_fields": {},
            "last_active": now,
            "status": "active",
            "created_at": now,
        })


# ── Inactivity check ──────────────────────────────────────────────────────────

def get_stale_onboarding_threads(inactivity_minutes: int = 5) -> list:
    """
    Returns all onboarding threads that have been inactive longer than
    inactivity_minutes and are not yet finalized.
    Used by brand_finalizer.py.
    """
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=inactivity_minutes)

    docs = (
        db.collection(COLLECTION)
        .where(filter=FieldFilter("agent_name", "==", "brand_onboarding"))
        .where(filter=FieldFilter("status", "==", "active"))
        .stream()
    )

    stale = []
    for doc in docs:
        data = doc.to_dict()
        last_active = data.get("last_active")
        if last_active and last_active < cutoff:
            stale.append(data)
    return stale


# ── Active agent detection ────────────────────────────────────────────────────

def get_last_active_agent(user_id: str) -> str | None:
    """
    Returns the name of the agent with the most recent last_active timestamp
    for this user. Used by main.py to resume the right agent thread.
    """
    docs = (
        db.collection(COLLECTION)
        .where(filter=FieldFilter("user_id", "==", user_id))
        .where(filter=FieldFilter("status", "==", "active"))
        .stream()
    )

    latest_agent = None
    latest_time = None
    for doc in docs:
        data = doc.to_dict()
        t = data.get("last_active")
        if t and (latest_time is None or t > latest_time):
            latest_time = t
            latest_agent = data.get("agent_name")

    return latest_agent

# ── Session reset ─────────────────────────────────────────────────────────────

def reset_user_behavior(user_id: str) -> list:
    """
    Clear all behavioral agent threads for this user.
    Keeps brand_onboarding thread intact (preserves brand data).
    Returns list of cleared agent names.
    """
    docs = (
        db.collection(COLLECTION)
        .where(filter=FieldFilter("user_id", "==", user_id))
        .stream()
    )
    cleared = []
    for doc in docs:
        data = doc.to_dict()
        agent_name = data.get("agent_name", "")
        if agent_name == "brand_onboarding":
            continue  # preserve brand data
        doc.reference.delete()
        cleared.append(agent_name)
    return cleared