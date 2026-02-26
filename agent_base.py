"""
agent_base.py
Base class for all agents. Each agent:
- Has its own Firestore thread (history isolated per agent per user)
- Generates a summary when routing out (not passing full history)
- Can pull brand context from Vertex AI or structured fields from Firestore
"""

from abc import ABC, abstractmethod
from llm_client import get_claude_client, gemini_generate
import thread_manager
from agents.talk_agent import TalkAgent
from agent_router import find_best_agent






class BaseAgent(ABC):
    name: str = ""
    domain: str = ""

    # ── Main entry point ──────────────────────────────────────────────────────

    def handle(
        self,
        user_input: str,
        user_id: str,
        brand_id: str,
        routing_context: str = None,   # summary passed from routing agent
        **kwargs
    ) -> str:
        """
        Load this agent's thread, run, save thread, return response.
        routing_context: short summary from the agent that routed here.
        """
        thread = thread_manager.load_thread(user_id, self.name)
        thread["brand_id"] = brand_id

        history = thread_manager.get_messages(thread)

        # Classify: should this agent handle it, or route further?
        if not self._can_handle(user_input, routing_context):
            print(f"[Router] {self.name} routing away")
            return self._route(user_input, user_id, brand_id, thread, **kwargs)

        print(f"[Router] {self.name} handling request")
        response = self.run(
            user_input=user_input,
            history=history,
            user_id=user_id,
            brand_id=brand_id,
            routing_context=routing_context,
            **kwargs
        )

        # Persist updated thread
        thread = thread_manager.append_messages(thread, user_input, response)
        thread_manager.save_thread(user_id, self.name, thread)

        return response

    # ── Abstract: subclasses implement this ───────────────────────────────────

    @abstractmethod
    def run(
        self,
        user_input: str,
        history: list,
        user_id: str,
        brand_id: str,
        routing_context: str = None,
        **kwargs
    ) -> str:
        pass

    # ── Routing ───────────────────────────────────────────────────────────────

    def _can_handle(self, user_input: str, routing_context: str = None) -> bool:
        """Gemini 2.5 Flash classifies whether this agent should handle the request."""
        context_hint = f"\nContext from previous agent: {routing_context}" if routing_context else ""
        answer = gemini_generate(
            prompt=f"{user_input}{context_hint}",
            system=(
                "Return ONLY the single agent name that best fits this request. "
                "Choose from: email, caption, talk, media_approval, media_search, "
                "brand_update, docs, brand_onboarding"
            )
        ).strip().lower()
        return answer == self.name

    def _route(
        self,
        user_input: str,
        user_id: str,
        brand_id: str,
        thread: dict,
        routing_depth: int = 0,
        **kwargs
    ) -> str:
        """Generate summary of own thread, find best agent, route with summary."""
        if routing_depth > 2:
            # Safety: fall back to talk agent
            return TalkAgent().handle(user_input, user_id, brand_id)

        # Generate summary of this agent's conversation so far
        history = thread_manager.get_messages(thread)
        summary = self._summarize(history) if history else ""
        if summary:
            thread_manager.update_summary(user_id, self.name, summary)

        # Find best agent (excluding self)
        best = find_best_agent(user_input, exclude=self.name)

        if best:
            print(f"[Router] {self.name} → {best.name} | summary: {summary[:80]}...")
            return best.handle(
                user_input=user_input,
                user_id=user_id,
                brand_id=brand_id,
                routing_context=summary,
                routing_depth=routing_depth + 1,
                **kwargs
            )

        # No match — talk agent as final fallback
        return TalkAgent().handle(
            user_input=user_input,
            user_id=user_id,
            brand_id=brand_id,
            routing_context=summary,
        )

    def _summarize(self, history: list) -> str:
        """Generate a concise summary of a message history using Gemini."""
        if not history:
            return ""
        formatted = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-10:]
        )
        return gemini_generate(
            prompt=formatted,
            system=(
                "Summarize this conversation concisely in 2-4 sentences. "
                "Capture the user's intent, key decisions made, and any context "
                "another agent would need to continue helping them."
            )
        )