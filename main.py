"""
main.py
Entry point for the Creativo AI Assistant.
Determines the active agent per user, routes to it, and persists thread history.
"""

from dotenv import load_dotenv
import os
import thread_manager
from agent_router import find_best_agent, AGENT_REGISTRY
from brand_finalizer import start_listener


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"



USER_ID = "manar"
BRAND_ID = "creativo"


def get_agent_by_name(name: str):
    for agent in AGENT_REGISTRY:
        if agent.name == name:
            return agent
    return None


def main():
    print("Welcome to Creativo AI Assistant. Type 'exit' to quit.")
    start_listener()  # background daemon thread — stops when main exits

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # Determine which agent to use:
        # 1. Check if user has an active ongoing thread (resume it)
        # 2. Otherwise classify the new message to find the best agent
        active_agent_name = thread_manager.get_last_active_agent(USER_ID)
        agent = get_agent_by_name(active_agent_name) if active_agent_name else None

        # If last active agent is unlikely to handle this new message, reclassify
        if agent and not agent._can_handle(user_input):
            agent = None

        if agent is None:
            agent = find_best_agent(user_input)

        if agent is None:
            print("AI: I'm not sure how to help with that. Could you rephrase?")
            continue

        print(f"\n[Active agent: {agent.name}]")
        print("AI: ", end="", flush=True)

        response = agent.handle(
            user_input=user_input,
            user_id=USER_ID,
            brand_id=BRAND_ID,
        )

        print(response)


if __name__ == "__main__":
    main()