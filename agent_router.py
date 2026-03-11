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

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
        from agents import write_email
        from llm_client import get_claude_client
        client = get_claude_client()

        seeded_history = list(history)
        if last_turn and last_turn.get("response"):
            seeded_history = [
                {"role": "user",      "content": last_turn.get("user", "")},
                {"role": "assistant", "content": last_turn["response"]}
            ] + seeded_history

        return write_email(
            user_input=user_input,
            conversation_history=seeded_history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class CaptionAgent(BaseAgent):
    name = "caption"
    domain = "Writing social media captions or posts for any platform."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
        from agents import write_caption_agent
        from llm_client import get_claude_client
        client = get_claude_client()

        # Seed the previous turn's full response into history
        # so the model sees exactly what was shown (media titles, IDs, statuses)
        seeded_history = list(history)
        if last_turn and last_turn.get("response"):
            seeded_history = [
                {"role": "user",      "content": last_turn.get("user", "")},
                {"role": "assistant", "content": last_turn["response"]}
            ] + seeded_history

        return write_caption_agent(
            user_input=user_input,
            conversation_history=seeded_history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


class TalkAgent(BaseAgent):
    name = "talk"
    domain = "Formatter and presenter for media search and media approval results only. Not used for conversation."

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

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
        # Delegate to orchestrator which handles multi-step approve/reject via tool calling
        from orchestrator import orchestrate
        return orchestrate(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id,
            routing_context=routing_context,
            last_turn=last_turn,
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
    domain = "All brand conversation: brand lookups, strategy, messaging, audience questions, expansion ideas, and new onboarding. Default agent for any brand or general conversation."

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


def find_best_agent(user_input: str, exclude: str = None, conversation_context: str = None) -> BaseAgent | None:
    """
    Use a single Gemini call to pick the best agent.
    Passes full agent descriptions and conversation context for accurate routing.
    """
    agents = [a for a in AGENT_REGISTRY if a.name != exclude]

    agent_descriptions = "\n".join([f"- {a.name}: {a.domain}" for a in agents])
    context_hint = f"\n\nRecent conversation:\n{conversation_context}" if conversation_context else ""

    from debug_hooks import log_routing_decision
    answer = gemini_generate(
        prompt=f"User message: {user_input}{context_hint}",
        system=(
            f"You are a routing classifier for a marketing AI assistant.\n"
            f"Return ONLY the single agent name that best fits the user message.\n\n"
            f"Agent options:\n{agent_descriptions}\n\n"
            f"Routing rules:\n"
            f"- 'docs': ONLY for app how-to — sign in, sign up, password reset, subscription steps\n"
            f"- 'talk': ONLY used as a formatter/presenter for media_search and media_approval results. Nothing else.\n"
            f"- 'email': any email writing, campaign, or newsletter request\n"
            f"- 'caption': writing social media posts or captions for any platform\n"
            f"- 'brand_onboarding': ALL brand conversation — lookups, strategy, messaging, audience, general chat, everything brand-related that is not a direct field update\n"
            f"- 'media_approval': approving or rejecting media items\n"
            f"- 'media_search': listing, searching, or viewing media\n"
            f"- 'brand_update': updating a specific brand field with an exact new value\n"
            f"- 'docs': app how-to questions only\n"
            f"- Use the conversation context to understand what the user is referring to and route accordingly\n"
            f"- When in doubt → 'brand_onboarding'\n"
            f"Return ONLY the agent name, nothing else."
        )
    ).strip().lower()

    log_routing_decision(user_input, answer)

    for agent in AGENT_REGISTRY:
        if agent.name == answer and agent.name != exclude:
            return agent

    # Fallback to talk
    for agent in AGENT_REGISTRY:
        if agent.name == "talk":
            return agent

    return None