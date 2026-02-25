"""
Base class for all agents in the multi-agent routing architecture.
Every agent can be an entry point and a router.
"""

from abc import ABC, abstractmethod
from llm_client import get_claude_client


ROUTING_PROMPT = """
You are a strict intent classifier.
Given a user message and conversation history, decide if this message belongs to the described agent's domain.
Return ONLY "yes" or "no". No explanation.

Agent domain: {domain}
"""


class BaseAgent(ABC):
    # Subclasses define their domain description for self-classification
    domain: str = ""
    name: str = ""

    def can_handle(self, user_input: str, conversation_history: list) -> bool:
        """Uses LLM to decide if this agent should handle the message."""
        client = get_claude_client()
        # Single call replaces 7 can_handle() calls
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system="Return ONLY the agent name that best fits this request: email, media_approval, media_search, brand_update, caption, docs, talk",
            messages=[{"role": "user", "content": user_input}],
            max_tokens=10
        )
        
        
        
        
        #response = client.messages.create(
        #    model="claude-opus-4-6",
        #    system=ROUTING_PROMPT.format(domain=self.domain),
        #    messages=[
        #        *[{"role": m["role"], "content": m["content"]} for m in conversation_history[-4:]],
        #        {"role": "user", "content": user_input}
        #    ],
        #    max_tokens=5
        #)
        answer = response.content[0].text.strip().lower()
        return answer == "yes"

    @abstractmethod
    def run(self, user_input: str, conversation_history: list, user_id: str, brand_id: str, client=None, **kwargs) -> str:
        """Execute the agent's main task."""
        pass

    def handle(self, user_input: str, conversation_history: list, user_id: str, brand_id: str, routing_depth: int = 0, **kwargs) -> str:
        """
        Entry point for any agent.
        Checks if it can handle → runs if yes → routes if no.
        """
        if routing_depth > 2:
            # Prevent infinite routing loops — fall back to orchestrator
            from orchestrator import orchestrate
            return orchestrate(user_input=user_input, conversation_history=conversation_history,
                               user_id=user_id, brand_id=brand_id)

        if self.can_handle(user_input, conversation_history):
            print(f"[Router] {self.name} is handling the request")
            return self.run(user_input, conversation_history, user_id, brand_id, **kwargs)
        else:
            print(f"[Router] {self.name} is routing away")
            return self._route(user_input, conversation_history, user_id, brand_id, routing_depth, **kwargs)

    def _route(self, user_input: str, conversation_history: list, user_id: str, brand_id: str, routing_depth: int, **kwargs) -> str:
        """Route to the best matching agent or fall back to orchestrator."""
        from agent_router import find_best_agent

        best_agent = find_best_agent(
            user_input=user_input,
            conversation_history=conversation_history,
            exclude=self.name
        )

        if best_agent:
            print(f"[Router] Routing to {best_agent.name}")
            return best_agent.handle(
                user_input=user_input,
                conversation_history=conversation_history,
                user_id=user_id,
                brand_id=brand_id,
                routing_depth=routing_depth + 1,
                **kwargs
            )

        # No agent matched — fall back to orchestrator
        print("[Router] No agent matched — falling back to orchestrator")
        from orchestrator import orchestrate
        return orchestrate(user_input=user_input, conversation_history=conversation_history,
                           user_id=user_id, brand_id=brand_id)