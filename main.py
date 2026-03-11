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
from agent_router import find_best_agent, AGENT_REGISTRY
from brand_finalizer import start_listener
from debug_server import start as start_debug
from debug_hooks import log_user_input, log_agent_selected, log_agent_response

USER_ID  = "manar"
BRAND_ID = "creativo"

# Short-term conversation window — last N turns kept in memory for routing context
# This is NOT agent history — it's just for the classifier to understand context
# Scoped per brand — cleared when brand changes so stale context never bleeds across brands
ROUTING_WINDOW = []
MAX_WINDOW = 6
_last_turn: dict = {}  # {user, response, agent} — full text, no truncation  # last 3 exchanges (user + assistant pairs)
_current_brand = BRAND_ID  # tracks active brand to detect switches
_routing_summary = None   # Gemini-generated summary of recent conversation


def get_agent_by_name(name: str):
    for agent in AGENT_REGISTRY:
        if agent.name == name:
            return agent
    return None


def build_routing_context() -> str:
    """
    Return the current Gemini summary of recent conversation.
    Falls back to a short raw snippet if no summary yet.
    """
    if _routing_summary:
        return _routing_summary
    if not ROUTING_WINDOW:
        return None
    # Fallback: first 2 turns only, trimmed — avoids flooding classifier
    lines = []
    for turn in ROUTING_WINDOW[-2:]:
        lines.append(f"{turn['role'].upper()}: {turn['content'][:120]}")
    return "\n".join(lines)


def _update_routing_summary(user_input: str, response: str, agent_name: str):
    """
    After each turn, ask Gemini to update the rolling conversation summary.
    Always includes the last active agent so the classifier can route follow-ups correctly.
    """
    global _routing_summary
    from llm_client import gemini_generate
    recent = ""
    if _routing_summary:
        recent = f"Previous summary: {_routing_summary}\n\n"
    recent += f"Latest turn (agent: {agent_name}):\nUser: {user_input[:200]}\nAssistant: {response[:200]}"
    try:
        _routing_summary = gemini_generate(
            prompt=recent,
            system=(
                "You are summarizing a conversation for routing context. "
                "Write 2-3 sentences capturing what was just discussed and what the user is trying to do. "
                "Always end with: 'Last active agent: <agent_name>.'"
            )
        )
    except Exception:
        _routing_summary = f"User: {user_input[:80]}. Last active agent: {agent_name}."


def _flush_on_exit():
    """On exit, write collected fields to brand doc — always, regardless of thread status."""
    try:
        from brand_finalizer import _finalize_thread
        import thread_manager
        thread = thread_manager.load_thread(USER_ID, "brand_onboarding")
        collected = thread.get("collected_fields", {})
        messages = thread.get("messages", [])
        if collected or messages:
            # Always finalize — even if already marked finalized, new fields may have been added
            _finalize_thread(thread, USER_ID, BRAND_ID)
            thread_manager.mark_finalized(USER_ID, "brand_onboarding")
            print("\n[Brand book saved]")
    except Exception as e:
        pass  # never block exit


def main():
    print("Welcome to Creativo AI Assistant. Type 'exit' to quit.")
    start_listener()
    start_debug()

    last_agent_name = None  # track last agent used for continuity
    global ROUTING_WINDOW, _current_brand, _routing_summary

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
            last_agent_name = None
            _current_brand = BRAND_ID
            print(f"[Brand switched to: {BRAND_ID} — routing context cleared]")

        routing_context = build_routing_context()

        # Decide which agent to use:
        # 1. If last agent can still handle this message in context → reuse it
        # 2. Otherwise classify fresh using full routing context window
        agent = None

        # Always let Gemini decide routing — it has the full rolling summary as context
        # The routing summary captures what was just discussed so Gemini can correctly
        # route follow-ups like "ok approve first 2" or "yes let's add later"
        agent = find_best_agent(user_input, conversation_context=routing_context)

        if agent is None:
            print("AI: I'm not sure how to help with that. Could you rephrase?")
            continue

        print(f"\n[Active agent: {agent.name}]")

        # Snapshot current _last_turn BEFORE calling handle — this is what agents receive
        # (the previous turn's full response, not the current one being generated)
        last_turn_snapshot = dict(_last_turn)

        response = agent.handle(
            user_input=user_input,
            user_id=USER_ID,
            brand_id=BRAND_ID,
            routing_context=routing_context,
            last_turn=last_turn_snapshot,
        )

        log_agent_selected(agent.name, routing_context=routing_context)
        log_agent_response(agent.name, response)
        print(f"AI: {response}")

        # NOW store this turn so the NEXT agent gets it
        last_agent_name = agent.name
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
        import threading
        threading.Thread(
            target=_update_routing_summary,
            args=(user_input, response, agent.name),
            daemon=True
        ).start()


if __name__ == "__main__":
    main()