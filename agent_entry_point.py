"""
agent_entry_point.py
Test any agent directly as an entry point without going through main.py.
Usage: python agent_entry_point.py
"""
from agent_router import EmailAgent, CaptionAgent, MediaApprovalAgent, MediaSearchAgent, BrandOnboardingAgent, DocsAgent

USER_ID = "manar"
BRAND_ID = "maketing"

print("Available agents: email, caption, media_approval, media_search, brand_onboarding, docs")
agent_name = input("Agent to test: ").strip() or "email"

AGENTS = {
    "email": EmailAgent(),
    "caption": CaptionAgent(),
    "media_approval": MediaApprovalAgent(),
    "media_search": MediaSearchAgent(),
    "brand_onboarding": BrandOnboardingAgent(),
    "docs": DocsAgent(),
}

agent = AGENTS.get(agent_name, EmailAgent())
print(f"\nTesting {agent.name} agent. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input or user_input.lower() in ["exit", "quit"]:
        break

    response = agent.handle(
        user_input=user_input,
        user_id=USER_ID,
        brand_id=BRAND_ID,
    )
    print(f"\nAgent: {response}\n")