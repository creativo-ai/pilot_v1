"""
brand_onboarding_agent.py
Conversational agent that collects brand info through natural dialogue.
- Structured fields → tracked in thread's collected_fields (→ Firestore brands doc)
- Free-form info → tracked in thread's free_form_notes (→ Vertex AI brand context)
"""

from agent_base import BaseAgent
from llm_client import get_claude_client, gemini_generate
import thread_manager
import json

BRAND_SCHEMA_FIELDS = [
    "brand_name", "industry", "mission", "vision", "values",
    "tone", "brand_voice", "communication_style",
    "target_audience", "primary_goal", "default_email_signature"
]

ONBOARDING_SYSTEM_PROMPT = """
You are a friendly brand strategist onboarding a new client for Creativo.

Your goal is to learn about their brand through natural conversation — not an interview.
Ask one topic at a time. Be warm, curious, and professional.

Topics to cover (weave them in naturally, don't list them):
- Brand name and what they do
- Their industry
- Their mission and vision
- Core values
- Tone and brand voice (how do they want to sound?)
- Target audience
- Primary marketing goal
- Email sign-off preference

Also listen for and acknowledge any extra context they share:
- Competitors they mention
- Brand story or origin
- Market positioning
- Customer pain points
- Anything else that paints a picture of who they are

When you have enough on a topic, move to the next naturally.
Never output JSON. Never say "I'm collecting your brand info".
"""

EXTRACTION_SYSTEM_PROMPT = """
You are a data extraction assistant.
Given a conversation, extract brand information into two outputs:

1. structured_fields: a JSON object with any of these keys found in the conversation:
   brand_name, industry, mission, vision, values, tone, brand_voice,
   communication_style, target_audience, primary_goal, default_email_signature

2. free_form_notes: a plain text paragraph capturing anything relevant that
   doesn't fit the structured fields (competitors, brand story, positioning,
   audience nuances, origin story, etc.)

Return ONLY valid JSON in this exact format:
{
  "structured_fields": { ... },
  "free_form_notes": "..."
}
"""


class BrandOnboardingAgent(BaseAgent):
    name = "brand_onboarding"
    domain = "Collecting brand information, onboarding a new brand, or setting up brand identity from scratch."

    def run(
        self,
        user_input: str,
        history: list,
        user_id: str,
        brand_id: str,
        routing_context: str = None,
        **kwargs
    ) -> str:
        client = get_claude_client()

        # Load what we've collected so far
        thread = thread_manager.load_thread(user_id, self.name)
        collected = thread.get("collected_fields", {})
        missing = [f for f in BRAND_SCHEMA_FIELDS if f not in collected]

        # Build system prompt with current collection status
        fields_status = (
            f"\n\nFields collected so far: {', '.join(collected.keys()) or 'none'}."
            f"\nFields still needed: {', '.join(missing) or 'all collected — wrap up warmly'}."
        )

        # Seed context if routed from another agent
        seed = f"\n\nContext from previous conversation: {routing_context}" if routing_context else ""

        messages = list(history)
        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=ONBOARDING_SYSTEM_PROMPT + fields_status + seed,
            messages=messages,
            max_tokens=600
        )

        reply = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                reply = block.text.strip()
                break

        # Extract structured + free-form info from latest exchange
        self._extract_and_store(user_input, reply, user_id)

        return reply

    def _extract_and_store(self, user_input: str, assistant_reply: str, user_id: str):
        """
        After each turn, run Gemini extraction on the latest exchange
        and merge results into the thread's collected_fields + free_form_notes.
        """
        exchange = f"USER: {user_input}\nASSISTANT: {assistant_reply}"

        raw = gemini_generate(prompt=exchange, system=EXTRACTION_SYSTEM_PROMPT)

        try:
            parsed = json.loads(raw)
            structured = parsed.get("structured_fields", {})
            free_form = parsed.get("free_form_notes", "")

            if structured or free_form:
                thread_manager.update_collected_fields(
                    user_id=user_id,
                    agent_name=self.name,
                    new_fields=structured,
                    free_form=free_form if free_form else None
                )
        except Exception as e:
            print(f"⚠️ Extraction parse failed: {e} | raw: {raw[:100]}")
