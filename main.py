"""
main.py
Entry point for the Creativo AI Assistant.
Maintains a short conversation window so routing always has context.
"""

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

import thread_manager
import threading
from agent_router import find_best_agent, AGENT_REGISTRY
from brand_finalizer import start_listener
from debug_server import start as start_debug
from debug_hooks import log_user_input, log_agent_selected, log_agent_response

USER_ID  = "manar"
BRAND_ID = "mario"

# Short-term conversation window — last N turns kept in memory for routing context
# This is NOT agent history — it's just for the classifier to understand context
# Scoped per brand — cleared when brand changes so stale context never bleeds across brands
ROUTING_WINDOW = []
MAX_WINDOW = 6
_last_turn: dict = {}  # {user, response, agent} — full text, no truncation
_current_brand = BRAND_ID  # tracks active brand to detect switches
_routing_summary = None   # Gemini-generated summary of recent conversation
_last_media_action: str = ""  # persists last media action summary across turns (never overwritten by non-media turns)
_session_media_actioned: bool = False  # True after first media approval/rejection this session


def reset_session(user_id: str) -> None:
    """
    Clear all behavioral agent threads for this user (keeps brand_onboarding + brand doc).
    Also resets in-memory session state: routing summary, last turn, routing window.
    Call this at session start for returning users who need a clean slate.
    """
    global _routing_summary, _last_turn, ROUTING_WINDOW, _last_media_action
    from thread_manager import reset_user_behavior
    cleared = reset_user_behavior(user_id)
    _routing_summary = None
    _last_turn = {}
    _last_media_action = ""
    _session_media_actioned = False
    thread_manager.clear_last_media_action(user_id, BRAND_ID)
    ROUTING_WINDOW.clear()
    if cleared:
        print(f"[Session] Reset {len(cleared)} agent threads: {', '.join(cleared)}")
    else:
        print("[Session] No behavioral threads found to reset.")


def get_agent_by_name(name: str):
    for agent in AGENT_REGISTRY:
        if agent.name == name:
            return agent
    return None


def build_routing_context() -> str:
    """
    Return the current Gemini summary of recent conversation.
    Does NOT include _last_media_action — that is only for caption agent
    and would confuse orchestrator status checks.
    """
    if _routing_summary:
        return _routing_summary

    if not ROUTING_WINDOW:
        return None

    # Fallback: last 2 turns only, trimmed
    lines = []
    for turn in ROUTING_WINDOW[-2:]:
        lines.append(f"{turn['role'].upper()}: {turn['content'][:120]}")
    return "\n".join(lines)


def build_caption_context() -> str:
    """
    Routing context specifically for caption agent.
    Includes _last_media_action so caption knows which media was recently actioned.
    """
    base = build_routing_context() or ""
    media_note = f"Last media action: {_last_media_action}" if _last_media_action else ""
    if media_note:
        return (base + " | " + media_note) if base else media_note
    return base or None


def _update_routing_summary(user_input: str, response: str, agent_name: str):
    """
    After each turn, ask Gemini to update the rolling conversation summary.
    Media actions are extracted and stored persistently in _last_media_action
    so they survive non-media turns (e.g. "thank you") without being overwritten.
    """
    global _routing_summary, _last_media_action
    from llm_client import gemini_generate
    import re as _re

    # Always inject last known media action into summary so it persists across turns
    # (actual tracking now happens in main loop via agent._last_action_meta)
    recent = ""
    if _routing_summary:
        recent = f"Previous summary: {_routing_summary}\n\n"
    recent += f"Latest turn (agent: {agent_name}):\nUser: {user_input[:200]}\nAssistant: {response[:300]}"
    try:
        _routing_summary = gemini_generate(
            prompt=recent,
            system=(
                "You are summarizing a conversation for routing context. "
                "Write 2-3 sentences capturing what was just discussed and what the user is trying to do. "
                "If media actions are present (Approved/Rejected with IDs), always include them exactly as-is — e.g. 'Approved: image_003 | Rejected: video_001'. Never include Pending items. "
                "Always end with: 'Last active agent: <agent_name>.'"
            )
        )
    except Exception:
        _routing_summary = f"User: {user_input[:80]}. Last active agent: {agent_name}."


def _flush_on_exit():
    """On exit, write collected fields to brand doc — always, regardless of thread status."""
    try:
        from brand_finalizer import _finalize_thread
        thread = thread_manager.load_thread(USER_ID, "brand_onboarding")
        collected = thread.get("collected_fields", {})
        messages = thread.get("messages", [])
        if collected or messages:
            # Always finalize — even if already marked finalized, new fields may have been added
            _finalize_thread(thread, USER_ID, BRAND_ID)
            thread_manager.mark_finalized(USER_ID, "brand_onboarding")
            print("\n[Brand book saved]")
    except Exception as e:
        print(f"[Flush error] {e}")  # never block exit


def main():
    print("Welcome to Creativo AI Assistant. Type 'exit' to quit.")
    start_listener()
    start_debug()

    global ROUTING_WINDOW, _current_brand, _routing_summary, _last_media_action, _session_media_actioned

    # Restore last media action from Firestore so caption context survives restarts
    _last_media_action = thread_manager.load_last_media_action(USER_ID, BRAND_ID)
    if _last_media_action:
        print(f"[Session] Restored last media action: {_last_media_action}")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        log_user_input(user_input)

        if user_input.lower().strip() in ["exit", "quit"]:
            _flush_on_exit()
            print("👋 Goodbye!")
            break

        # Clear routing window if brand has changed — prevents stale context bleeding across brands
        if BRAND_ID != _current_brand:
            ROUTING_WINDOW.clear()
            _routing_summary = None
            _last_media_action = ""
            _session_media_actioned = False
            thread_manager.clear_last_media_action(USER_ID, BRAND_ID)
            _current_brand = BRAND_ID
            print(f"[Brand switched to: {BRAND_ID} — routing context cleared]")

        routing_context = build_routing_context()

        agent = find_best_agent(user_input, conversation_context=routing_context)

        if agent is None:
            print("AI: I'm not sure how to help with that. Could you rephrase?")
            continue

        print(f"\n[Active agent: {agent.name}]")

        # Snapshot current _last_turn BEFORE calling handle — this is what agents receive
        # (the previous turn's full response, not the current one being generated)
        last_turn_snapshot = dict(_last_turn)

        # Caption agent gets richer context including last media action
        # All other agents get clean routing context without media action history
        agent_routing_context = (
            build_caption_context() if agent.name == "caption"
            else routing_context
        )

        response = agent.handle(
            user_input=user_input,
            user_id=USER_ID,
            brand_id=BRAND_ID,
            routing_context=agent_routing_context,
            last_turn=last_turn_snapshot,
        )

        log_agent_selected(agent.name, routing_context=routing_context)
        log_agent_response(agent.name, response)
        print(f"AI: {response}")

        # Persist exact actioned IDs from tool calls — ground truth, no regex parsing
        if agent.name == "media_approval":
            _meta = getattr(agent, "_last_action_meta", None)
            if _meta and isinstance(_meta, dict):
                _parts = []
                if _meta.get("approved"): _parts.append("Approved: " + ", ".join(_meta["approved"]))
                if _meta.get("rejected"): _parts.append("Rejected: " + ", ".join(_meta["rejected"]))
                if _parts:
                    new_action = " | ".join(_parts)
                    if not _session_media_actioned:
                        # First approval this session — replace old session state entirely
                        _last_media_action = new_action
                        _session_media_actioned = True
                        print(f"[Session] New session — replaced media action: {_last_media_action}")
                    else:
                        # Same session — append new actions to existing ones
                        _last_media_action = (_last_media_action + " | " + new_action) if _last_media_action else new_action
                        print(f"[Session] Appended media action: {_last_media_action}")
                    try:
                        thread_manager.save_last_media_action(USER_ID, BRAND_ID, _last_media_action)
                    except Exception as _e:
                        print(f"[Session] Failed to persist media action: {_e}")

        # NOW store this turn so the NEXT agent gets it
        ROUTING_WINDOW.append({"role": "user",      "content": user_input})
        _last_turn["user"] = user_input
        _last_turn["response"] = response
        _last_turn["agent"] = agent.name
        ROUTING_WINDOW.append({"role": "assistant",  "content": response[:300]})

        # Keep window bounded
        if len(ROUTING_WINDOW) > MAX_WINDOW * 2:
            ROUTING_WINDOW.pop(0)
            ROUTING_WINDOW.pop(0)

        # Update Gemini summary in background — non-blocking
        threading.Thread(
            target=_update_routing_summary,
            args=(user_input, response, agent.name),
            daemon=True
        ).start()


if __name__ == "__main__":
    main()