"""
agent_router.py
Central agent registry. Finds the best agent for a request using Gemini classification.
Each agent class wraps the corresponding function from agents.py.
"""

from agent_base import BaseAgent
from llm_client import gemini_generate


# ── Agent classes ─────────────────────────────────────────────────────────────

class EmailAgent(BaseAgent):
    name = "email"
    domain = "Writing, drafting, or creating email messages."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import write_email
        from llm_client import get_claude_client
        client = get_claude_client()
        return write_email(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class CaptionAgent(BaseAgent):
    name = "caption"
    domain = "Writing social media captions or posts for any platform."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import write_caption_agent
        from llm_client import get_claude_client
        client = get_claude_client()
        return write_caption_agent(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class TalkAgent(BaseAgent):
    name = "talk"
    domain = "General brand conversation, brand identity questions, or anything not covered by other agents."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import talk_agent
        from llm_client import get_claude_client
        client = get_claude_client()

        # Enrich history with routing context as a system seed if provided
        seeded_history = list(history)
        if routing_context:
            seeded_history = [{"role": "user", "content": f"[Context from previous conversation: {routing_context}]"},
                              {"role": "assistant", "content": "Understood, I'll keep that context in mind."}] + seeded_history

        return talk_agent(
            user_input=user_input,
            conversation_history=seeded_history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class MediaApprovalAgent(BaseAgent):
    name = "media_approval"
    domain = "Approving or rejecting images and videos. Listing pending media awaiting approval."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        # Delegate to orchestrator which handles multi-step approve/reject via tool calling
        from orchestrator import orchestrate
        return orchestrate(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id
        )


class MediaSearchAgent(BaseAgent):
    name = "media_search"
    domain = "Searching, listing, or filtering media history including images and videos."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import search_media_agent
        from llm_client import get_claude_client
        client = get_claude_client()
        return search_media_agent(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class BrandUpdateAgent(BaseAgent):
    name = "brand_update"
    domain = "Updating or changing brand fields: tone, mission, vision, values, colors, voice, audience."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import brand_update_agent
        from llm_client import get_claude_client
        client = get_claude_client()
        return brand_update_agent(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class DocsAgent(BaseAgent):
    name = "docs"
    domain = "Answering platform how-to questions: sign up, sign in, subscription, features."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import search_docs_agent
        from llm_client import get_claude_client
        client = get_claude_client()
        return search_docs_agent(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class BrandOnboardingAgent(BaseAgent):
    name = "brand_onboarding"
    domain = "Collecting brand information from scratch, onboarding a new brand, setting up brand identity."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from brand_onboarding_agent import BrandOnboardingAgent as _Agent
        return _Agent().run(
            user_input=user_input,
            history=history,
            user_id=user_id,
            brand_id=brand_id,
            routing_context=routing_context,
            **kwargs
        )


# ── Registry ──────────────────────────────────────────────────────────────────

AGENT_REGISTRY: list[BaseAgent] = [
    EmailAgent(),
    CaptionAgent(),
    MediaApprovalAgent(),
    MediaSearchAgent(),
    BrandUpdateAgent(),
    DocsAgent(),
    BrandOnboardingAgent(),
    TalkAgent(),       # last — catches anything unmatched
]


def find_best_agent(user_input: str, exclude: str = None) -> BaseAgent | None:
    """
    Use a single Gemini call to pick the best agent.
    Much cheaper than calling can_handle() on each agent individually.
    """
    agent_names = [a.name for a in AGENT_REGISTRY if a.name != exclude]

    answer = gemini_generate(
        prompt=user_input,
        system=(
            f"Return ONLY the single agent name that best fits this request. "
            f"Choose from: {', '.join(agent_names)}"
        )
    ).strip().lower()

    # Find matching agent
    for agent in AGENT_REGISTRY:
        if agent.name == answer and agent.name != exclude:
            return agent

    # Fallback: return TalkAgent if no match
    for agent in AGENT_REGISTRY:
        if agent.name == "talk":
            return agent

    return None
