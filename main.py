from orchestrator import orchestrate
#from orchestrator_v2 import orchestrate
#from actions_registry import ACTION_SCHEMAS, AVAILABLE_ACTIONS
from llm_client import get_claude_client


print("Welcome to the AI Assistant. Type 'exit' to quit.")

conversation_history = []
user_id = "manar"
brand_id = "creativo"

while True:
    user_request = input("\nYou: ")
    if user_request.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    if not user_request.strip():
        continue

    conversation_history.append({"role": "user", "content": user_request})

    print("\nAI: ", end="", flush=True)

    response = orchestrate(
        user_input=user_request,
        conversation_history=conversation_history,
        user_id=user_id,
        brand_id=brand_id,
        stream=True
    )

    conversation_history.append({"role": "assistant", "content": response})