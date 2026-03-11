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
from data_layer_vertexAI import get_brand_info, get_latest_brand_context
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
- When the user gives information, acknowledge it naturally then move to the next missing field.
- If the user's answer is vague, gently ask for more detail — but only once per field.
- If the user doesn't know something, suggest examples or offer to skip and return later.
- Never say "I'm filling out a form" or "this is field X". Just have a natural conversation.
- Never output JSON to the user.

## Context Awareness — Critical
- The "Brand Book — Already Documented" section above is your source of truth.
- NEVER ask about anything already in the brand book. It is documented. It is known.
- NEVER ask "what is your mission", "what does your brand do", "what is your brand name"
  if those fields exist in the brand book above.
- ALWAYS read history, routing context, AND the brand book before asking any question.
- Before asking anything — check: is it in the brand book? If yes, use it silently.
- Only ask about fields that are genuinely empty in both the brand book AND the conversation.

## Handling Strategy Conversations
- If the user discusses brand direction, new sectors, new client types, or expansion — this IS brand onboarding work. Engage with it naturally and capture it as brand context.
- Do NOT redirect strategy conversations elsewhere. Handle them here, ask follow-up questions, and save the insights.
- If the user says something like "we want to serve hospitals" — acknowledge it, explore it with one question, and note it as target audience or sector expansion.

## Responding to What Was Actually Said
- Always respond directly to the user's latest message first.
- Never introduce topics the user hasn't brought up.
- Do not assume the user is asking about something other than what they said.

## Brand Book — Already Documented (READ THIS BEFORE ASKING ANYTHING)
{saved_brand_text}

## Brand Context Summary
{brand_summary}

## Current Session State
Brand: {brand_name}
Mode: {mode}

## Already Known From Other Agents
{routing_context}

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
        thread: dict = None,
        **kwargs
    ) -> str:
        client = get_claude_client()

        # Load current thread state
        # Use thread passed from agent_base if available — avoids double-load and lost writes
        if thread is None:
            thread = thread_manager.load_thread(user_id, self.name)
        collected = thread.get("collected_fields", {})

        # Load existing brand data from Firestore — so agent knows what's already documented
        saved_brand = get_brand_info(brand_id) or {}
        brand_summary = get_latest_brand_context(brand_id) or ""

        from debug_hooks import log_brand_fields
        log_brand_fields(saved_brand)

        # Determine mode: new onboarding or updating existing brand
        is_update = self._is_update_request(collected) or bool(saved_brand)
        mode = "Updating existing brand" if is_update else "New brand onboarding"

        # Figure out where we are in the conversation
        current_section_key, current_section_name, section_hint = self._get_current_section(collected)
        completed_sections = self._get_completed_sections(collected)
        missing_fields = self._get_missing_fields(current_section_key, collected)

        # Determine next instruction for the agent
        if not history and not collected and not routing_context and not brand_id:
            next_instruction = (
                "Start with a warm welcome. Introduce yourself as Alex from Creativo. "
                "Tell the user you'll help them build their brand book together. "
                "Ask for their brand name and what they do — keep it light and exciting."
            )
        elif not history and not collected and (routing_context or brand_id):
            next_instruction = (
                f"The brand is already known: {brand_id}. "
                "Do NOT ask for the brand name — it is already set. "
                "Do NOT introduce yourself as if starting from scratch. "
                "Read the routing context below carefully — it describes exactly what the user "
                "was just discussing with another agent. "
                "Pick up that conversation naturally and continue from where they left off. "
                "If the routing context describes an audience expansion discussion, "
                "acknowledge what was said and move directly into exploring it. "
                "Never ask 'what is your brand name' or 'what does your business do'."
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

        routing_ctx_display = routing_context or "Nothing shared yet from other agents."

        # Format saved brand as readable flat text for the prompt
        # Works for both flat and nested brand docs
        def _flatten_for_display(d: dict, prefix="") -> list:
            lines = []
            for k, v in d.items():
                if k in ("metadata",):
                    continue
                label = (prefix + k).replace("_", " ").title()
                if isinstance(v, dict):
                    lines.extend(_flatten_for_display(v, prefix=k + "_"))
                elif isinstance(v, list):
                    if v:
                        lines.append(f"{label}: {', '.join(str(i) for i in v)}")
                elif v and v not in ("", {}, []):
                    lines.append(f"{label}: {v}")
            return lines

        if saved_brand:
            brand_lines = _flatten_for_display(saved_brand)
            saved_brand_text = "\n".join(brand_lines) if brand_lines else "No brand data saved yet."
        else:
            saved_brand_text = "No brand data saved yet."

        system = ONBOARDING_SYSTEM_PROMPT.format(
            brand_name=brand_name,
            mode=mode,
            saved_brand_text=saved_brand_text,
            brand_summary=brand_summary or "No summary available yet.",
            routing_context=routing_ctx_display,
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

        from debug_hooks import log_onboarding_state
        log_onboarding_state(
            mode=mode,
            section=current_section_name or "complete",
            missing=missing_fields or "none",
            collected=collected
        )

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
        Uses a flat extraction prompt — no nested template — to avoid Gemini returning {}.
        Merges into collected_fields and persists to Firestore.
        """
        exchange = f"USER: {user_input}\nASSISTANT: {assistant_reply}"

        # Use a flat field list instead of the full nested template
        # The nested template is too large — Gemini returns {} or truncates
        FLAT_FIELDS = """
brand_name, industry, mission, vision, values, tone, brand_voice,
communication_style, target_audience, primary_goal, default_email_signature,
tagline, brand_story, positioning, brand_archetype, personality_traits,
language_style, competitors, competitive_advantages, pricing_strategy,
core_messages, elevator_pitch, key_benefits, primary_color, secondary_color,
content_pillars, posting_frequency, hashtags, email_greeting, email_closing,
flagship_product, offerings, short_term_goals, long_term_goals, expansion_plans,
primary_audience_age, primary_audience_location, primary_audience_pain_points
"""

        print(f"[EXTRACT] Sending exchange to Gemini: {exchange[:200]}")
        raw = gemini_generate(
            prompt=exchange,
            system=(
                "Extract brand information that the USER explicitly stated about THEIR brand. "
                "Only extract what the USER said — never extract tone, voice, or personality "
                "from how the ASSISTANT speaks. "
                "For example: if the ASSISTANT says 'Hey, welcome!' that tells you nothing "
                "about the brand's tone — ignore it. "
                "Only extract when the USER explicitly describes their brand. "
                "Map extracted info to these flat field names:\n"
                + FLAT_FIELDS +
                "\nReturn ONLY valid JSON with the fields you found. "
                "Example: {\"brand_name\": \"Acme\", \"tone\": \"professional\"} "
                "If the user said nothing brand-related (e.g. just said hello), return {}. "
                "No explanation, no markdown fences, no nested objects."
            )
        )

        # Strip fences defensively
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        from debug_hooks import log_error
        try:
            extracted = json.loads(cleaned)
        except Exception as e:
            log_error(f"Extraction parse failed: {e}", context=raw[:120])
            print(f"⚠️  Extraction parse failed: {e} | raw: {raw[:120]}")
            return

        if not extracted or not isinstance(extracted, dict):
            log_error("Extraction returned empty", context=f"raw={raw[:80]}")
            return

        # Remove empty values before merging
        extracted = {k: v for k, v in extracted.items() if v not in (None, "", [], {})}
        if not extracted:
            log_error("Extraction all empty after filter", context=str(list(extracted.keys())))
            return

        print(f"✅ Extracted fields: {list(extracted.keys())} → {extracted}")

        # Deep merge extracted into collected
        merged = _deep_merge(collected, extracted)

        # Update thread in memory
        thread["collected_fields"] = merged

        # thread["collected_fields"] is now updated in memory.
        # agent_base.handle() will call save_thread() after run() returns,
        # which persists the entire thread including the updated collected_fields.
        print(f"✅ Extracted and merged fields: {list(merged.keys())}")

        from debug_hooks import log_onboarding_state
        log_onboarding_state(
            mode="extracting",
            section="field extraction",
            missing="",
            collected=merged
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