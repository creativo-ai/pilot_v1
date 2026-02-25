"""
Central agent registry and router.
Finds the best agent for a given user input.
"""

from agent_base import BaseAgent
from agents import (
    write_email, talk_agent, list_pending_media,
    approve_media, reject_media, search_docs_agent,
    search_media_agent, brand_update_agent, write_caption_agent
)


class EmailAgent(BaseAgent):
    name = "email"
    domain = "Writing, drafting, or creating email messages for any audience or purpose."

    def run(self, user_input, conversation_history, user_id, brand_id, client=None, **kwargs):
        return write_email(user_input=user_input, conversation_history=conversation_history,
                           user_id=user_id, brand_id=brand_id, client=client, **kwargs)


class MediaApprovalAgent(BaseAgent):
    name = "media_approval"
    domain = "Approving or rejecting images and videos. Listing pending media awaiting approval."

    def run(self, user_input, conversation_history, user_id, brand_id, client=None, **kwargs):
        # Delegate to orchestrator which handles multi-step approve/reject
        from orchestrator import orchestrate
        return orchestrate(user_input=user_input, conversation_history=conversation_history,
                           user_id=user_id, brand_id=brand_id)


class MediaSearchAgent(BaseAgent):
    name = "media_search"
    domain = "Searching, listing, or retrieving media history including images and videos with their statuses."

    def run(self, user_input, conversation_history, user_id, brand_id, client=None, **kwargs):
        return search_media_agent(user_input=user_input, conversation_history=conversation_history,
                                  user_id=user_id, brand_id=brand_id, client=client, **kwargs)


class BrandUpdateAgent(BaseAgent):
    name = "brand_update"
    domain = "Updating, changing, or adding brand information such as tone, mission, vision, values, target audience, or brand voice."

    def run(self, user_input, conversation_history, user_id, brand_id, client=None, **kwargs):
        return brand_update_agent(user_input=user_input, conversation_history=conversation_history,
                                  user_id=user_id, brand_id=brand_id, client=client, **kwargs)


class CaptionAgent(BaseAgent):
    name = "caption"
    domain = "Writing social media posts, captions, or content for Instagram, LinkedIn, Facebook, TikTok, Twitter, or YouTube."

    def run(self, user_input, conversation_history, user_id, brand_id, client=None, **kwargs):
        return write_caption_agent(user_input=user_input, conversation_history=conversation_history,
                                   user_id=user_id, brand_id=brand_id, client=client, **kwargs)


class DocsAgent(BaseAgent):
    name = "docs"
    domain = "Answering how-to questions about the platform: sign up, sign in, subscription plans, account features."

    def run(self, user_input, conversation_history, user_id, brand_id, client=None, **kwargs):
        return search_docs_agent(user_input=user_input, conversation_history=conversation_history,
                                 user_id=user_id, brand_id=brand_id, client=client, **kwargs)


class TalkAgent(BaseAgent):
    name = "talk"
    domain = "General conversation, brand identity questions (mission, vision, values, tone), and anything not covered by other agents."

    def run(self, user_input, conversation_history, user_id, brand_id, client=None, **kwargs):
        return talk_agent(user_input=user_input, conversation_history=conversation_history,
                          user_id=user_id, brand_id=brand_id, client=client, **kwargs)


# Registry of all agents in priority order
AGENT_REGISTRY: list[BaseAgent] = [
    EmailAgent(),
    MediaApprovalAgent(),
    MediaSearchAgent(),
    BrandUpdateAgent(),
    CaptionAgent(),
    DocsAgent(),
    TalkAgent(),  
]


def find_best_agent(user_input: str, conversation_history: list, exclude: str = None) -> BaseAgent | None:
    """Find the first agent that can handle this request, skipping the excluded one."""
    for agent in AGENT_REGISTRY:
        if agent.name == exclude:
            continue
        if agent.can_handle(user_input, conversation_history):
            return agent
    return None
