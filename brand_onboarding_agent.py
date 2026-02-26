"""
brand_onboarding_agent.py

Long-running conversational agent that guides users through building their brand
from scratch, and handles future brand updates.

Design:
- Loads brand_book_template.json to know what fields to collect
- Tracks filled vs missing fields per section
- Saves structured fields to Firestore incrementally as confirmed
- Free-form info lives ONLY in conversation history
- Finalizer later summarizes the full conversation to extract free-form context
"""

import json
import os
from agent_base import BaseAgent
from llm_client import get_claude_client, gemini_generate
import thread_manager

# ── Load template once at module level ───────────────────────────────────────

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "brand_book_template.json")
with open(TEMPLATE_PATH, "r") as f:
    BRAND_TEMPLATE = json.load(f)

# Conversation sections — guide the agent through the template progressively
# Each section maps to template top-level keys, with a human-friendly name
ONBOARDING_SECTIONS = [
    ("brand_foundation",       "Brand Foundation",     "mission, vision, purpose, values, brand story, positioning, tagline"),
    ("brand_personality",      "Brand Personality",    "brand archetype, personality traits, voice, tone of voice, language do's and don'ts"),
    ("target_audience",        "Target Audience",      "primary audience demographics, psychographics, pain points, goals, buyer personas"),
    ("market_positioning",     "Market Positioning",   "industry, competitors, competitive advantages, pricing strategy, distribution"),
    ("brand_messaging",        "Brand Messaging",      "core messages, elevator pitch, key benefits, emotional hooks, CTAs"),
    ("visual_identity",        "Visual Identity",      "colors, typography, imagery style, design principles"),
    ("content_strategy",       "Content Strategy",     "content pillars, platform strategy, posting frequency, hashtags, SEO keywords"),
    ("social_media_guidelines","Social Media",         "bio templates, caption style, community tone, crisis communication"),
    ("email_branding",         "Email Branding",       "email signature, greeting style, closing style, newsletter structure"),
    ("product_or_service",     "Products & Services",  "offerings, flagship product, pricing tiers, features"),
    ("customer_experience",    "Customer Experience",  "customer journey, brand touchpoints, support tone"),
    ("future_vision",          "Future Vision",        "short-term goals, long-term goals, expansion plans"),
]

# Fields in metadata that get auto-filled, not asked
AUTO_FILL_SECTIONS = {"metadata", "legal_and_compliance", "internal_guidelines",
                      "performance_metrics", "attachments"}


ONBOARDING_SYSTEM_PROMPT = """
You are a warm, expert brand strategist named Alex, onboarding a new client for Creativo.

Your goal is to help the user build their complete brand book through natural conversation.
This is an ongoing session — you pick up exactly where you left off each time.

## Rules
- Ask about ONE topic at a time. Never bombard the user with multiple questions.
- Be conversational, warm, and encouraging — not clinical or form-like.
- When the user gives information, acknowledge it naturally before moving on.
- If the user's answer is vague, gently ask for more detail.
- If the user doesn't know something, suggest examples or offer to skip and return later.
- When updating an existing brand, acknowledge what already exists and ask what they want to change.
- Never say "I'm filling out a form" or "this is field X". Just have a natural conversation.
- Never output JSON to the user.

## Current Session State
Brand: {brand_name}
Mode: {mode}

## Progress
Completed sections: {completed_sections}
Current section: {current_section}
Missing fields in current section: {missing_fields}

## What to do next
{next_instruction}
"""

EXTRACTION_SYSTEM_PROMPT = """
You are a data extraction assistant for brand information.

Given a conversation exchange, extract any brand information mentioned and map it
to the brand book template structure below. Only extract information that was
clearly stated or confirmed by the user — do not infer or assume.

Return ONLY valid JSON matching the template structure (partial is fine).
Only include keys where you found actual values. Empty strings and empty arrays mean no data.
No explanation, no markdown fences.

Template structure:
{template_structure}
"""


class BrandOnboardingAgent(BaseAgent):
    name = "brand_onboarding"
    domain = "Building a brand from scratch, brand onboarding, setting up brand identity, or updating existing brand information."

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

        # Load current thread state
        thread = thread_manager.load_thread(user_id, self.name)
        collected = thread.get("collected_fields", {})

        # Determine mode: new onboarding or updating existing brand
        is_update = self._is_update_request(collected)
        mode = "Updating existing brand" if is_update else "New brand onboarding"

        # Figure out where we are in the conversation
        current_section_key, current_section_name, section_hint = self._get_current_section(collected)
        completed_sections = self._get_completed_sections(collected)
        missing_fields = self._get_missing_fields(current_section_key, collected)

        # Determine next instruction for the agent
        if not history and not collected:
            next_instruction = (
                "Start with a warm welcome. Introduce yourself as Alex from Creativo. "
                "Tell the user you'll help them build their brand book together. "
                "Ask for their brand name and what they do — keep it light and exciting."
            )
        elif not current_section_key:
            next_instruction = (
                "All sections are complete! Congratulate the user warmly. "
                "Tell them their brand book is ready and summarize what was covered. "
                "Ask if they'd like to review or adjust anything."
            )
        else:
            next_instruction = (
                f"You are currently in the '{current_section_name}' section. "
                f"Topics in this section: {section_hint}. "
                f"Missing fields: {missing_fields}. "
                f"Continue naturally from the last message. Ask about the next missing field."
            )

        brand_name = collected.get("metadata", {}).get("brand_name") or \
                     collected.get("brand_foundation", {}).get("mission", "")[:20] or \
                     "the brand"

        system = ONBOARDING_SYSTEM_PROMPT.format(
            brand_name=brand_name,
            mode=mode,
            completed_sections=", ".join(completed_sections) if completed_sections else "none yet",
            current_section=current_section_name or "Wrapping up",
            missing_fields=missing_fields or "all filled",
            next_instruction=next_instruction,
        )

        # Add routing context as a silent seed if coming from another agent
        messages = list(history)
        if routing_context and not history:
            messages = [
                {"role": "user", "content": f"[Previous context: {routing_context}]"},
                {"role": "assistant", "content": "Got it, I have context from our previous conversation."}
            ]

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=system,
            messages=messages,
            max_tokens=700
        )

        reply = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                reply = block.text.strip()
                break

        # Extract structured fields from this exchange and merge into collected
        self._extract_and_store(user_input, reply, collected, user_id, brand_id, thread)

        return reply

    # ── Field extraction ──────────────────────────────────────────────────────

    def _extract_and_store(
        self,
        user_input: str,
        assistant_reply: str,
        collected: dict,
        user_id: str,
        brand_id: str,
        thread: dict
    ):
        """
        Run Gemini extraction on the latest exchange.
        Merge structured fields into thread's collected_fields.
        Save to Firestore incrementally.
        Free-form info is NOT stored in any variable — it stays in history only.
        """
        exchange = f"USER: {user_input}\nASSISTANT: {assistant_reply}"

        # Give Gemini only the relevant template sections (not the full 153-field template)
        # to keep the extraction prompt focused
        relevant_template = {
            k: v for k, v in BRAND_TEMPLATE.items()
            if k not in AUTO_FILL_SECTIONS
        }

        raw = gemini_generate(
            prompt=exchange,
            system=EXTRACTION_SYSTEM_PROMPT.format(
                template_structure=json.dumps(relevant_template, indent=2)
            )
        )

        try:
            extracted = json.loads(raw)
        except Exception as e:
            print(f"⚠️  Extraction parse failed: {e} | raw: {raw[:120]}")
            return

        if not extracted:
            return

        # Deep merge extracted into collected
        merged = _deep_merge(collected, extracted)

        # Update thread in memory only — no Firestore write until session ends
        thread["collected_fields"] = merged
        thread_manager.update_collected_fields(
            user_id=user_id,
            agent_name=self.name,
            new_fields=merged
        )

    # ── Section tracking ──────────────────────────────────────────────────────

    def _get_current_section(self, collected: dict):
        """Return the first section that still has missing fields."""
        for section_key, section_name, section_hint in ONBOARDING_SECTIONS:
            if not self._is_section_complete(section_key, collected):
                return section_key, section_name, section_hint
        return None, None, None

    def _get_completed_sections(self, collected: dict) -> list:
        return [
            name for key, name, _ in ONBOARDING_SECTIONS
            if self._is_section_complete(key, collected)
        ]

    def _is_section_complete(self, section_key: str, collected: dict) -> bool:
        """A section is 'complete' if at least the core fields have values."""
        section_data = collected.get(section_key, {})
        if not section_data:
            return False
        # Check that at least 30% of leaf fields in this section are filled
        template_section = BRAND_TEMPLATE.get(section_key, {})
        total = _count_leaf_fields(template_section)
        filled = _count_filled_fields(section_data)
        return total > 0 and (filled / total) >= 0.3

    def _get_missing_fields(self, section_key: str, collected: dict) -> str:
        """Return a short description of what's still missing in the current section."""
        if not section_key:
            return ""
        template_section = BRAND_TEMPLATE.get(section_key, {})
        collected_section = collected.get(section_key, {})
        missing = _get_empty_keys(template_section, collected_section)
        return ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")

    def _is_update_request(self, collected: dict) -> bool:
        """True if the brand already has fields filled — this is an update, not a fresh onboarding."""
        return bool(collected)


# ── Utility functions ─────────────────────────────────────────────────────────

def _deep_merge(base: dict, updates: dict) -> dict:
    """Recursively merge updates into base, preserving existing values."""
    result = dict(base)
    for k, v in updates.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        elif k in result and isinstance(result[k], list) and isinstance(v, list):
            # Merge lists: add new items, avoid duplicates
            existing = result[k]
            for item in v:
                if item and item not in existing:
                    existing.append(item)
            result[k] = existing
        elif v not in (None, "", [], {}):
            result[k] = v
    return result



def _count_leaf_fields(obj) -> int:
    if isinstance(obj, dict):
        return sum(_count_leaf_fields(v) for v in obj.values())
    return 1


def _count_filled_fields(obj) -> int:
    if isinstance(obj, dict):
        return sum(_count_filled_fields(v) for v in obj.values())
    if isinstance(obj, list):
        return 1 if obj else 0
    return 1 if obj not in (None, "", [], {}) else 0


def _get_empty_keys(template_obj, collected_obj, depth=0) -> list:
    """Return list of unfilled field names (top 2 levels only for readability)."""
    missing = []
    if not isinstance(template_obj, dict):
        return missing
    for k, v in template_obj.items():
        collected_val = collected_obj.get(k) if isinstance(collected_obj, dict) else None
        if isinstance(v, dict) and depth < 1:
            missing.extend(_get_empty_keys(v, collected_val or {}, depth + 1))
        elif not collected_val:
            missing.append(k.replace("_", " "))
    return missing