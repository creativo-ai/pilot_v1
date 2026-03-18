"""
agent_router.py
Central agent registry. Finds the best agent for a request using Gemini classification.
Each agent class wraps the corresponding function from agents.py.
"""

from agent_base import BaseAgent
from llm_client import gemini_generate, get_claude_client


# ── Agent classes ─────────────────────────────────────────────────────────────

class EmailAgent(BaseAgent):
    name = "email"
    domain = "Writing, drafting, or creating email messages."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
        from agents import write_email
        client = get_claude_client()

        last_agent = last_turn.get("agent") if last_turn else None
        if last_turn and last_turn.get("response") and last_agent != "email":
            seeded_history = [
                {"role": "user",      "content": last_turn.get("user", "")},
                {"role": "assistant", "content": last_turn["response"]}
            ]
        else:
            seeded_history = list(history)

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
    domain = "Writing social media captions or posts for any platform. Also handles questions about captions — what caption was written, reviewing a caption, or retrieving a previously written caption."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
        from agents import write_caption_agent
        client = get_claude_client()

        last_agent = last_turn.get("agent") if last_turn else None

        # If arriving from a different agent, only use last_turn as context.
        # Do NOT mix in stale caption thread history — it causes the model to
        # pick up the last caption topic instead of the current one.
        if last_turn and last_turn.get("response") and last_agent != "caption":
            # Arriving from another agent — seed last_turn as context
            seeded_history = [
                {"role": "user",      "content": last_turn.get("user", "")},
                {"role": "assistant", "content": last_turn["response"]}
            ]
            # If coming from media agents, prepend full media list for richer context
            if routing_context and last_agent in ("media_approval", "media_search"):
                seeded_history = [
                    {"role": "user",      "content": f"[Media context: {routing_context}]"},
                    {"role": "assistant", "content": "Understood, I have the full media context."}
                ] + seeded_history
        elif last_turn and last_turn.get("response") and last_agent == "caption":
            seeded_history = list(history)
            if routing_context and ("image" in routing_context.lower() or "video" in routing_context.lower()):
                seeded_history = [
                    {"role": "user",      "content": f"[Media context: {routing_context}]"},
                    {"role": "assistant", "content": "Understood, I have the full media context."}
                ] + seeded_history
        else:
            seeded_history = list(history)

        # Build media_routing from last_turn if it was a media agent,
        # OR from routing_context if it contains a media action summary (persists across turns)
        media_routing = None
        if last_turn and last_turn.get("response") and last_agent in ("media_approval", "media_search"):
            media_routing = last_turn["response"]
        elif routing_context and any(k in routing_context for k in ("Approved:", "Rejected:", "Pending:", "media_approval", "media_search")):
            media_routing = routing_context

        # Prepend a brand reminder as the first assistant turn so the model
        # cannot mistake old brand names from history for the current brand
        from data_layer_vertexAI import get_brand_info as _gbi
        _bi = _gbi(brand_id) or {}
        _current_brand_name = (
            _bi.get("metadata", {}).get("brand_name") or
            _bi.get("brand_name") or
            brand_id
        )
        print(f"[Caption Router] brand_name injected: '{_current_brand_name}'")
        if _current_brand_name:
            # Append brand reminder as the LAST assistant message before user input
            # so it is the most recent context the model reads — highest priority
            seeded_history = seeded_history + [
                {"role": "user", "content": "[IMPORTANT] What brand name must I use in this caption?"},
                {"role": "assistant", "content": f"[IMPORTANT] I must use '{_current_brand_name}' — this is the current brand name from the brand profile. I will not use any other name from history."}
            ]

        return write_caption_agent(
            user_input=user_input,
            conversation_history=seeded_history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            routing_context=media_routing,
            **kwargs
        )


class TalkAgent(BaseAgent):
    name = "talk"
    domain = "ONLY used internally to format and present media search results and media approval confirmations. Never used for brand questions, brand updates, or general conversation."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import talk_agent
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
    domain = "Approving or rejecting images and videos by ID. Only for approve/reject actions — NOT for listing or searching media."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
        # Delegate to orchestrator which handles multi-step approve/reject via tool calling
        from orchestrator import orchestrate
        result = orchestrate(
            user_input=user_input,
            conversation_history=history,
            user_id=user_id,
            brand_id=brand_id,
            routing_context=routing_context,
            last_turn=last_turn,
        )
        # If orchestrator returns a dict with actioned IDs, store in kwargs for main.py
        if isinstance(result, dict):
            self._last_action_meta = result  # main.py reads this via agent._last_action_meta
            return result.get("response", "Done.")
        self._last_action_meta = None
        return result


class MediaSearchAgent(BaseAgent):
    name = "media_search"
    domain = "Searching, listing, or filtering media items (images and videos) — their status, platform, dates, and IDs. Does NOT handle captions or written content."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
        from agents import search_media_agent
        client = get_claude_client()

        seeded_history = list(history)
        if last_turn and last_turn.get("response"):
            seeded_history = [
                {"role": "user",      "content": last_turn.get("user", "")},
                {"role": "assistant", "content": last_turn["response"]}
            ] + seeded_history

        return search_media_agent(
            user_input=user_input,
            conversation_history=seeded_history,
            user_id=user_id,
            brand_id=brand_id,
            client=client,
            **kwargs
        )


# BrandUpdateAgent removed from routing — brand field updates are handled
# internally by brand_onboarding and flushed on exit or inactivity.


class DocsAgent(BaseAgent):
    name = "docs"
    domain = "Answering platform how-to questions: sign up, sign in, subscription, features."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, **kwargs):
        from agents import search_docs_agent
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
    domain = "Everything brand-related: building brand book, updating any brand field (name, mission, vision, values, tone, audience, colors), brand lookups, strategy, and general conversation. Use this for ANY request that mentions changing, updating, or setting a brand field. Default agent when nothing else matches."

    def run(self, user_input, history, user_id, brand_id, routing_context=None, last_turn=None, **kwargs):
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

            f"- Use the conversation context to understand what the user is referring to and route accordingly\n"
            f"- When in doubt → 'brand_onboarding'\n"
            f"Return ONLY the agent name, nothing else."
        )
    ).strip().lower()

    log_routing_decision(user_input, answer)

    for agent in AGENT_REGISTRY:
        if agent.name == answer and agent.name != exclude:
            return agent

    # Fallback to brand_onboarding (not talk — talk is only for media formatting)
    for agent in AGENT_REGISTRY:
        if agent.name == "brand_onboarding":
            return agent

    return None