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


class BaseAgent(ABC):
    name: str = ""
    domain: str = ""

    # ── Main entry point ──────────────────────────────────────────────────────

    def handle(
        self,
        user_input: str,
        user_id: str,
        brand_id: str,
        routing_context: str = None,
        last_turn: dict = None,        # full {user, response, agent} of previous turn
        **kwargs
    ) -> str:
        """
        Load this agent's thread, run, save thread, return response.
        routing_context: short summary from the agent that routed here.
        """
        thread = thread_manager.load_thread(user_id, self.name)

        # If this thread was built for a different brand, clear its history
        # so the agent doesn't answer from the wrong brand's context
        if thread.get("brand_id") and thread["brand_id"] != brand_id:
            print(f"[Brand switch detected in {self.name} thread — clearing stale history]")
            thread["messages"] = []
            thread["collected_fields"] = {}
            thread["summary"] = ""

        thread["brand_id"] = brand_id
        history = thread_manager.get_messages(thread)

        routing_depth = kwargs.pop("routing_depth", 0)

        # If we were explicitly routed here (routing_context exists), trust the routing decision
        # Only re-classify if this is a fresh request from main.py
        if routing_context or routing_depth == 0 and self._can_handle(user_input, routing_context):
            pass  # proceed to run()
        elif routing_depth == 0 and not self._can_handle(user_input, routing_context):
            print(f"[Router] {self.name} routing away")
            return self._route(user_input, user_id, brand_id, thread, routing_depth=routing_depth, **kwargs)

        print(f"[Router] {self.name} handling request")
        response = self.run(
            user_input=user_input,
            history=history,
            user_id=user_id,
            brand_id=brand_id,
            routing_context=routing_context,
            last_turn=last_turn,
            thread=thread,
            **kwargs
        )

        # Persist updated thread — thread dict may have been mutated by run() (e.g. collected_fields)
        thread = thread_manager.append_messages(thread, user_input, response)
        thread_manager.save_thread(user_id, self.name, thread)

        # Passive brand learning:
        # If this agent asked the user a brand question and got an answer,
        # forward the exchange silently to onboarding in the background.
        # Only applies to non-onboarding agents — onboarding handles itself.
        if self.name not in ("brand_onboarding", "brand_update"):
            self._maybe_capture_brand_info(
                agent_question=response,
                user_answer=user_input,
                user_id=user_id,
                brand_id=brand_id,
                original_user_input=user_input,
            )

        return response

    def _maybe_capture_brand_info(
        self,
        agent_question: str,
        user_answer: str,
        user_id: str,
        brand_id: str,
        original_user_input: str,
    ):
        """
        Uses Gemini to decide whether this exchange contains brand info worth capturing.
        No hardcoded phrases — Gemini reads the actual context and judges intent.
        """
        try:
            from llm_client import gemini_generate
            decision = gemini_generate(
                prompt=(
                    f"Agent said: {agent_question[:300]}\n"
                    f"User replied: {user_answer[:300]}"
                ),
                system=(
                    "Decide whether the user reply contains brand information worth saving. "
                    "Answer only YES or NO.\n"
                    "YES if: the agent asked about the brand (identity, values, tone, audience, colors, "
                    "mission, tagline, offerings, personality, etc.) AND the user provided a brand attribute.\n"
                    "NO if: the user is making a task request, giving a conversational reply with no brand data, "
                    "or the agent was not asking a brand question."
                )
            )
            if "YES" in decision.strip().upper()[:5]:
                from agents import capture_brand_info_in_background
                capture_brand_info_in_background(
                    user_input=original_user_input,
                    agent_question=agent_question,
                    user_answer=user_answer,
                    user_id=user_id,
                    brand_id=brand_id,
                )
        except Exception as e:
            print(f"[Passive capture trigger failed: {e}]")


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
        """Gemini classifies whether this agent should handle the request, using conversation context."""
        from agent_router import find_best_agent
        best = find_best_agent(user_input, conversation_context=routing_context)
        return best is not None and best.name == self.name

    def _route(
        self,
        user_input: str,
        user_id: str,
        brand_id: str,
        thread: dict,
        routing_depth: int = 0,
        routing_context: str = None,
        **kwargs
    ) -> str:
        """Generate summary of own thread, find best agent, route with summary."""
        if routing_depth > 2:
            # Safety: fall back to talk agent directly, no more routing
            from agent_router import AGENT_REGISTRY
            talk = next((a for a in AGENT_REGISTRY if a.name == "talk"), None)
            if talk:
                return talk.run(
                    user_input=user_input,
                    history=thread_manager.get_messages(thread_manager.load_thread(user_id, "talk")),
                    user_id=user_id,
                    brand_id=brand_id,
                    routing_context=f"Fallback after routing depth exceeded. Original request: {user_input}"
                )
            return "I'm having trouble routing your request. Could you rephrase?"

        # Generate summary of this agent's conversation so far
        # Tag summary with brand_id so receiving agent knows the context scope
        history = thread_manager.get_messages(thread)
        brand_id_tag = f"[Brand: {thread.get('brand_id', 'unknown')}] "
        summary = (brand_id_tag + self._summarize(history)) if history else ""
        if summary:
            thread_manager.update_summary(user_id, self.name, summary)

        # Find best agent (excluding self), pass conversation context for accurate routing
        from agent_router import find_best_agent
        best = find_best_agent(user_input, exclude=self.name, conversation_context=summary or routing_context)

        if best:
            print(f"[Router] {self.name} → {best.name} | summary: {summary[:80]}...")
            # Pass routing_context so receiving agent trusts the routing and doesn't re-classify
            return best.handle(
                user_input=user_input,
                user_id=user_id,
                brand_id=brand_id,
                routing_context=summary or f"Routed from {self.name}. User said: {user_input}",
                routing_depth=routing_depth + 1,
            )

        # No match — talk agent as final fallback
        from agent_router import AGENT_REGISTRY
        talk = next((a for a in AGENT_REGISTRY if a.name == "talk"), None)
        if talk:
            return talk.handle(
                user_input=user_input,
                user_id=user_id,
                brand_id=brand_id,
                routing_context=summary,
                routing_depth=routing_depth + 1,
            )
        return "I'm not sure how to help with that. Could you rephrase?"

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
                "Summarize this conversation in 2-4 sentences for a different AI agent "
                "that will handle the next request. "
                "Rules: "
                "1. Include the user's intent, key decisions, and all relevant context. "
                "2. Mark resolved topics with [RESOLVED] — the next agent must NOT revisit them. "
                "3. If the last exchange was the user agreeing to do something (said yes/sure/ok), "
                "   describe WHAT they agreed to in full detail — the next agent has no other context. "
                "   Example: 'User agreed to expand target audience to include public figures [AGREED]. "
                "   Next agent should pick up this discussion and explore it.' "
                "4. Never summarize 'yes' as just 'yes' — always explain what was agreed to. "
                "The receiving agent should use this as context to continue naturally."
            )
        )