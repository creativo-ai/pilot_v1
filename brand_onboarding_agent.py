"""
brand_onboarding_agent.py

Conversational brand onboarding agent. Guides users through building their
brand book naturally — one topic at a time, no form-filling feel.

File uploads are handled via Claude's Files API for accurate native extraction.
Fields are extracted from conversation and saved incrementally to Firestore.
"""

import json
import os
import anthropic
from agent_base import BaseAgent
from llm_client import get_claude_client, gemini_generate
from data_layer_vertexAI import get_brand_info, get_latest_brand_context
import thread_manager

# ── Template ──────────────────────────────────────────────────────────────────

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "brand_book_template.json")
with open(TEMPLATE_PATH, "r") as f:
    BRAND_TEMPLATE = json.load(f)

# Section definitions — used only for completion tracking, NOT for scripting Claude
ONBOARDING_SECTIONS = [
    ("metadata",         "Brand Basics",      "brand name, industry, brand ID"),
    ("brand_core",       "Brand Core",        "vision, mission, values, USP, positioning, competitors"),
    ("brand_voice",      "Brand Voice",       "tone of voice, words to avoid, key messages, taglines"),
    ("target_audience",  "Target Audience",   "primary audience, demographics, pain points, communication style"),
    ("visual_identity",  "Visual Identity",   "typography, color palette"),
]

FLAT_FIELDS = (
    "brand_id, brand_name, version, status, created_at, updated_at, created_by, approved_by, notes, "
    "vision, mission, emotion, competitor, domain, usp_statement, name, market_positioning, core_values, "
    "tone_of_voice, words_to_avoid, key_messages, taglines_and_slogans, "
    "primary_audience, demographics_gender, demographics_age, demographics_location, demographics_income_level, "
    "demographics_interests, demographics_pain_points_behaviors, demographics_communication_style, "
    ", primary_colors, secondary_colors"
)

# ── System prompt — trust Claude's intelligence ───────────────────────────────

ONBOARDING_SYSTEM_PROMPT = """You are Lumio, a warm and expert brand strategist at Creativo.
Your job is to help clients build a complete brand book through natural conversation.

## Your personality
- Warm, encouraging, genuinely curious about the brand
- You ask ONE question at a time — never a list of questions
- You acknowledge what the user just said before moving forward
- You guide proactively: after each answer, you naturally move to the next topic
- You never say "now let's move to field X" — you transition like a real strategist would
- You never sound like a form or a checklist

## What you're building
A brand book covering: brand basics, core brand identity (vision/mission/values/USP), 
brand voice and tone, target audience, and visual identity guidelines.

## How to run the conversation
1. Read the brand book below carefully — NEVER ask about anything already there
2. Identify what's genuinely missing
3. Ask about the most important missing thing in a natural, engaging way
4. When the user answers, acknowledge it warmly, then move to the next gap
5. If an answer is vague, ask ONE follow-up — then move on
6. If the user doesn't know something, offer examples or suggest skipping

## On file uploads
When a file is provided, read it carefully and extract every piece of brand information.
Tell the user what you found ("I can see your brand name is X, your mission is Y..."),
confirm anything ambiguous, then continue filling gaps conversationally.
Only report what is explicitly in the file — never infer or assume.

## Critical rules
- "Our brand" / "we" / "our" = the CLIENT's brand, not Creativo
- Never output JSON or structured data to the user
- Never reference field names or template sections out loud
- If the user wants to update something already saved, handle it naturally here
- STRICT RULE: Do NOT ask the user about their logo or logo variations. They will upload logo files later. For visual identity, ask ONLY about colors and typography.
- STRICT RULE: NEVER use the phrase "brand book" or tell the user that you are "building" one. Frame the conversation completely naturally around getting to know them and their brand identity.
---

## Brand book — what's already documented
{saved_brand_text}

## Free-form brand context
{brand_summary}

## Completion snapshot
{completion_snapshot}

## Context from other agents
{routing_context}
"""


class BrandOnboardingAgent(BaseAgent):
    name = "brand_onboarding"
    domain = (
        "Everything brand-related: building brand book, updating any brand field "
        "(name, mission, vision, values, tone, audience, colors), brand lookups, "
        "strategy, file uploads for brand info, and general conversation. "
        "Default agent when nothing else matches."
    )

    def run(
        self,
        user_input: str,
        history: list,
        user_id: str,
        brand_id: str,
        routing_context: str = None,
        thread: dict = None,
        file_content: str = None,   # raw text fallback (legacy)
        file_name: str = None,
        file_id: str = None,        # Claude Files API file_id (preferred)
        file_media_type: str = None,
        **kwargs
    ) -> str:
        client = get_claude_client()

        if thread is None:
            thread = thread_manager.load_thread(user_id, self.name)
        collected = thread.get("collected_fields", {})

        saved_brand = get_brand_info(brand_id) or {}
        brand_summary = get_latest_brand_context(brand_id) or ""

        from debug_hooks import log_brand_fields
        log_brand_fields(saved_brand)

        # ── Build system prompt ───────────────────────────────────────────────
        saved_brand_text = _flatten_for_display(saved_brand) or "Nothing documented yet."
        completion_snapshot = self._completion_snapshot(collected)

        system = ONBOARDING_SYSTEM_PROMPT.format(
            saved_brand_text=saved_brand_text,
            brand_summary=brand_summary or "Not available yet.",
            completion_snapshot=completion_snapshot,
            routing_context=routing_context or "None.",
        )

        # ── Build messages ────────────────────────────────────────────────────
        messages = list(history)

        # Inject routing context as a synthetic turn if starting fresh
        if routing_context and not history:
            messages = [
                {"role": "user",      "content": f"[Context from previous session: {routing_context}]"},
                {"role": "assistant", "content": "Got it — I have context from our earlier conversation. Let me pick up from there."}
            ]

        # Build the current user turn content
        user_turn_content = self._build_user_turn(
            user_input=user_input,
            file_id=file_id,
            file_media_type=file_media_type,
            file_content=file_content,
            file_name=file_name,
        )

        messages.append({"role": "user", "content": user_turn_content})

        from debug_hooks import log_onboarding_state
        log_onboarding_state(
            mode="onboarding",
            section=self._get_current_section(collected)[1] or "complete",
            missing=self._get_missing_fields(self._get_current_section(collected)[0], collected),
            collected=collected
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=system,
            messages=messages,
            max_tokens=700,
        )

        reply = next(
            (b.text.strip() for b in response.content if getattr(b, "type", None) == "text"),
            ""
        )

        # Extract fields from this exchange and save
        self._extract_and_store(user_input, reply, collected, user_id, brand_id, thread)
        return reply

    # ── Files API — build user turn with optional file attachment ─────────────

    def _build_user_turn(
        self,
        user_input: str,
        file_id: str = None,
        file_media_type: str = None,
        file_content: str = None,
        file_name: str = None,
    ):
        """
        Build the user turn content. If a file_id is provided, attach it as a
        native document block via the Files API. Falls back to inline text
        if only raw file_content is available (legacy path).
        """
        # Simple text-only turn
        if not file_id and not file_content:
            return user_input

        content_blocks = []

        # Claude Files API — preferred path
        if file_id and file_media_type:
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id,
                },
            })

        # Legacy fallback: inline text content
        elif file_content:
            label = f"Uploaded file: {file_name}\n\n" if file_name else "Uploaded file:\n\n"
            content_blocks.append({
                "type": "text",
                "text": label + file_content[:12000],
            })

        # Always append the user's message text
        if user_input and user_input.strip():
            content_blocks.append({
                "type": "text",
                "text": user_input,
            })

        return content_blocks if len(content_blocks) > 1 else user_input

    # ── Conversation extraction ───────────────────────────────────────────────

    def _extract_and_store(
        self,
        user_input: str,
        assistant_reply: str,
        collected: dict,
        user_id: str,
        brand_id: str,
        thread: dict,
    ):
        """Extract brand fields from latest exchange and merge into thread."""
        exchange = f"USER: {user_input}\nASSISTANT: {assistant_reply}"

        raw = gemini_generate(
            prompt=exchange,
            system=(
                "Extract brand information from the exchange. "
                "1. If the user is answering a question, extract what the USER explicitly stated. "
                "2. If the user uploaded a file, the ASSISTANT will summarize what it found in the file. In this case, extract ALL the brand fields the ASSISTANT found. "
                "Map to these flat field names:\n" + FLAT_FIELDS + "\n"
                "Return ONLY valid JSON, e.g. {\"brand_name\": \"Acme\", \"tone_of_voice\": \"bold\"}. "
                "If nothing brand-related, return {}. No markdown, no explanation."
            )
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            extracted = json.loads(cleaned)
        except Exception as e:
            from debug_hooks import log_error
            log_error(f"Extraction parse failed: {e}", context=raw[:120])
            return

        if not isinstance(extracted, dict):
            return

        extracted = {k: v for k, v in extracted.items() if v not in (None, "", [], {})}
        if not extracted:
            return

        print(f"[Extract] {list(extracted.keys())} → {extracted}")
        thread["collected_fields"] = _deep_merge(collected, extracted)

    # ── Section tracking (for metadata/UI only) ───────────────────────────────

    def _get_current_section(self, collected: dict):
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
        section_data = collected.get(section_key, {})
        if not section_data:
            return False
        template_section = BRAND_TEMPLATE.get(section_key, {})
        total = _count_leaf_fields(template_section)
        filled = _count_filled_fields(section_data)
        return total > 0 and (filled / total) >= 0.3

    def _get_missing_fields(self, section_key: str, collected: dict) -> str:
        if not section_key:
            return ""
        template_section = BRAND_TEMPLATE.get(section_key, {})
        collected_section = collected.get(section_key, {})
        missing = _get_empty_keys(template_section, collected_section)
        return ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")

    def _completion_percentage(self, collected: dict) -> int:
        total = sum(
            _count_leaf_fields(BRAND_TEMPLATE.get(key, {}))
            for key, _, _ in ONBOARDING_SECTIONS
        )
        filled = sum(
            _count_filled_fields(collected.get(key, {}))
            for key, _, _ in ONBOARDING_SECTIONS
        )
        return round((filled / total) * 100) if total > 0 else 0

    def _completion_snapshot(self, collected: dict) -> str:
        """Human-readable completion summary injected into the system prompt."""
        lines = []
        for key, name, _ in ONBOARDING_SECTIONS:
            done = self._is_section_complete(key, collected)
            missing = self._get_missing_fields(key, collected)
            status = "complete" if done else f"missing: {missing or 'all fields'}"
            lines.append(f"- {name}: {status}")
        pct = self._completion_percentage(collected)
        lines.append(f"\nOverall: {pct}% complete")
        return "\n".join(lines)


# ── Files API helper — call this from chat_server.py before agent.run() ───────

def upload_file_to_claude(file_bytes: bytes, file_name: str, media_type: str) -> str:
    """
    Upload a file to Claude's Files API and return the file_id.
    Call this once when the user uploads a file, then pass file_id to agent.run().

    Supported media types: application/pdf, text/plain, text/html,
    application/vnd.openxmlformats-officedocument.wordprocessingml.document, image/*

    Returns file_id string, or raises on failure.
    """
    client = get_claude_client()
    import io
    response = client.beta.files.upload(
        file=(file_name, io.BytesIO(file_bytes), media_type),
    )
    print(f"[Files API] Uploaded '{file_name}' → file_id: {response.id}")
    return response.id


def delete_file_from_claude(file_id: str):
    """Delete a file from Claude's Files API after processing."""
    try:
        client = get_claude_client()
        client.beta.files.delete(file_id)
        print(f"[Files API] Deleted file_id: {file_id}")
    except Exception as e:
        print(f"[Files API] Delete failed for {file_id}: {e}")


# ── Utility functions ─────────────────────────────────────────────────────────

def _flatten_for_display(d: dict, prefix: str = "") -> str:
    lines = []
    for k, v in d.items():
        if k == "metadata":
            continue
        label = (prefix + k).replace("_", " ").title()
        if isinstance(v, dict):
            nested = _flatten_for_display(v, prefix=k + "_")
            if nested:
                lines.append(nested)
        elif isinstance(v, list) and v:
            lines.append(f"{label}: {', '.join(str(i) for i in v)}")
        elif v and v not in ("", {}, []):
            lines.append(f"{label}: {v}")
    return "\n".join(filter(None, lines))


def _deep_merge(base: dict, updates: dict) -> dict:
    result = dict(base)
    for k, v in updates.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        elif k in result and isinstance(result[k], list) and isinstance(v, list):
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