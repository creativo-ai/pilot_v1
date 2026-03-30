"""
Claude handles the conversation intelligence.
Code handles state and persistence only.
"""

import json, os, io
from agent_base import BaseAgent
from llm_client import get_claude_client
from data_layer_vertexAI import get_brand_info, get_latest_brand_context
import thread_manager

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "brand_book_template.json")
with open(TEMPLATE_PATH, "r") as f:
    BRAND_TEMPLATE = json.load(f)

ONBOARDING_SECTIONS = [
    ("metadata",        "Brand Basics",     "brand name, domain"),
    ("brand_core",      "Brand Core",       "vision, mission, emotion, competitors, USP, positioning, core values"),
    ("brand_voice",     "Brand Voice",      "tone of voice, words to avoid, key messages, taglines"),
    ("target_audience", "Target Audience",  "primary audience, demographics"),
]

# Fields intentionally excluded from chat — collected via UI pickers instead
VISUAL_FIELDS_SKIP = {
    "font_for_headings", "font_for_body_text",
    "primary_colors", "secondary_colors",
    "primary_color", "secondary_color",
    "logo_usage_icon_only_version", "logo_usage_full_logo",
    "logo_usage_black_white_variations",
}

SYSTEM_PROMPT = """You are Lumio, a warm brand onboarding specialist at Creativo.
Your job: help clients build their brand book through natural conversation.
You are NOT a strategist — your role ends once the brand book essentials are captured.

## How you work
- Ask ONE question at a time — never a list.
- Acknowledge what the user just said, and then IMMEDIATELY ask about the next missing field.
- STRICT RULE: You MUST always end your reply with a specific question asking for the next missing piece of information.
- Be warm and human, not clinical or form-like.
- Never mention field names, sections, or template structure out loud.
- Never output JSON to the user.
- NEVER announce that you are saving, extracting, or updating data. Do it silently.

## Tool Usage & Conversation Flow (CRITICAL)
When you use the `save_brand_fields` tool, you MUST include your full conversational reply AND the next question in your text response before or alongside the tool call.
Do NOT wait for a tool result to ask the next question. You must acknowledge the answer, silently call the tool, and explicitly ask the next question all in the SAME single response.

## What you do NOT ask about
STRICT RULE: DO NOT ask about fonts, colors, hex codes, or logos in the chat. 
If the user uploads a file containing this information, extract it silently to save it, but do NOT ask any follow-up questions about visual identity. 

## Returning clients
If the brand book already has content, open warmly:
- Greet them by brand name if available.
- Briefly highlight what's already documented (2–3 things, not a list) to show you remember them.
- Invite them to continue with a specific next gap. Do not start from scratch.

## Handling File Uploads
STRICT RULE: If the user uploads a file, you MUST do the following in your reply:
1. Summarize the key brand information you successfully gathered from the file in a brief, readable bulleted list.
2. End your reply by asking a specific question to fill in the next missing gap to continue the conversation.

## Completion & handoff
When all conversational sections are covered (brand basics, core, voice, audience), wrap up warmly and introduce Lucy.
Say something exactly like: "That wraps up everything I need from you! 🎉 From here, Lucy — our brand strategist — will take over. I am just here for onboarding, but she will use everything you've shared to shape your strategy and guide your brand forward."
Do NOT continue asking questions after this message.

## Brand book so far
{brand_book}

## Free-form brand context
{brand_context}

## Context from other agents
{routing_context}
"""

# ── Extraction tool — Claude fills this instead of Gemini ────────────────────

def _build_extract_tool() -> dict:
    """
    Build the extraction tool schema dynamically from FIELD_MAP.
    Single source of truth — add/remove fields in FIELD_MAP only.
    Type is inferred from the template default value:
      [] → array of strings
      "" → string
    """
    from brand_finalizer import FIELD_MAP
    import json as _json

    # Load template once to infer field types
    _template = _json.load(open(TEMPLATE_PATH))

    def _get_template_default(path: list):
        """Walk nested path in template and return the default value."""
        node = _template
        for key in path:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return None
            if node is None:
                return None
        return node

    properties = {}
    for flat_key, path in FIELD_MAP.items():
        default = _get_template_default(path)
        if isinstance(default, list):
            properties[flat_key] = {"type": "array", "items": {"type": "string"}}
        else:
            properties[flat_key] = {"type": "string"}

    return {
        "name": "save_brand_fields",
        "description": (
            "Extract all potential brand fields based on the user's explicitly stated facts or uploaded files. "
            "Extract direct values and clear logical implications, but DO NOT guess, infer wildly, or hallucinate information just to fill fields. "
            "Leave unknown fields null."
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": [],
        },
    }

# Built once at module load — reflects FIELD_MAP automatically
EXTRACT_TOOL = _build_extract_tool()


class BrandOnboardingAgent(BaseAgent):
    name = "brand_onboarding"
    domain = (
        "Everything brand-related: building brand book, updating brand fields, "
        "brand strategy, file uploads for brand info, general conversation. "
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
        file_id: str = None,
        file_media_type: str = None,
        file_content: str = None,  # legacy fallback
        file_name: str = None,
        **kwargs
    ) -> str:
        client = get_claude_client()

        if thread is None:
            thread = thread_manager.load_thread(user_id, self.name)

        # ── Context ───────────────────────────────────────────────────────────
        saved_brand    = get_brand_info(brand_id) or {}
        brand_context  = get_latest_brand_context(brand_id) or ""
        brand_book_txt = _flatten_for_display(saved_brand) or "Nothing documented yet."

        system = SYSTEM_PROMPT.format(
            brand_book=brand_book_txt,
            brand_context=brand_context or "Not available yet.",
            routing_context=routing_context or "None.",
        )

        # ── Returning user: inject a warm welcome on first turn if brand exists ──
        collected = thread.get("collected_fields", {})
        is_first_turn = not history and not thread.get("messages")
        has_existing_brand = bool(saved_brand) and any(
            v not in (None, "", [], {})
            for k, v in saved_brand.items()
            if k != "metadata"
        )

        if is_first_turn and has_existing_brand and not routing_context:
            brand_name = (
                saved_brand.get("metadata", {}).get("brand_name")
                or collected.get("brand_name")
                or "your brand"
            )
            # Build a quick highlights string from what's already saved
            highlights = _get_brand_highlights(saved_brand)
            welcome_injection = (
                f"[SYSTEM NOTE: This is a returning client for brand '{brand_name}'. "
                f"Their brand book already has: {highlights}. "
                f"Open warmly — acknowledge their progress, mention 1-2 specific things "
                f"already documented, then guide them naturally to the next gap. "
                f"Do not start from scratch.]"
            )
            history = [
                {"role": "user", "content": welcome_injection},
                {"role": "assistant", "content": f"Welcome back! Great to continue working on {brand_name}."},
            ]

        # ── Check if onboarding is complete — send handoff if so ─────────────
        if self._is_onboarding_complete(collected) and history:
            # Only send handoff once — check if it was already sent
            last_msgs = thread.get("messages", [])
            already_handed_off = any(
                "Lucy" in m.get("content", "") and m.get("role") == "assistant"
                for m in last_msgs[-4:]
            )
            if not already_handed_off:
                return (
                    "That wraps up everything I need from you! 🎉 "
                    "From here, **Lucy** — our brand strategist — will take over. "
                    "She'll use everything you've shared to help shape your strategy, "
                    "refine your positioning, and guide your brand forward. "
                    "You're in great hands!"
                )

        # ── Messages ──────────────────────────────────────────────────────────
        messages = list(history)

        if routing_context and not history:
            messages = [
                {"role": "user",      "content": f"[Context: {routing_context}]"},
                {"role": "assistant", "content": "Got it, picking up from there."},
            ]

        messages.append({
            "role": "user",
            "content": _build_user_content(user_input, file_id, file_media_type, file_content, file_name),
        })

        # ── Single Claude call — responds + extracts fields simultaneously ────
        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=system,
            messages=messages,
            max_tokens=700,
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "auto"},
        )

        # ── Parse response ────────────────────────────────────────────────────
        reply = ""
        extracted = {}

        for block in response.content:
            if block.type == "text":
                reply = block.text.strip()
            elif block.type == "tool_use" and block.name == "save_brand_fields":
                extracted = {
                    k: v for k, v in block.input.items()
                    if v not in (None, "", [], {})
                }

        if extracted:
            print(f"[Extract] {list(extracted.keys())}")
            thread["collected_fields"] = _deep_merge(
                thread.get("collected_fields", {}),
                extracted,
            )

        if not reply:
            reply = "Got it! Let's keep going. What else can you tell me about the brand?"

        return reply

    # ── UI metadata helpers (for chat_server.py progress bar) ─────────────────

    def _completion_percentage(self, collected: dict) -> int:
        from brand_finalizer import FIELD_MAP
        # Only count conversational fields (exclude visual UI fields)
        conversational_fields = [k for k in FIELD_MAP if k not in VISUAL_FIELDS_SKIP]
        total  = len(conversational_fields)
        filled = sum(1 for k in conversational_fields if collected.get(k) not in (None, "", [], {}))
        return round((filled / total) * 100) if total else 0

    def _get_current_section(self, collected: dict):
        for key, name, hint in ONBOARDING_SECTIONS:
            if not self._is_section_complete(key, collected):
                return key, name, hint
        return None, None, None

    def _is_section_complete(self, section_key: str, collected: dict) -> bool:
        # These are the mandatory fields before Lucy is allowed to take over
        section_fields = {
            "metadata":        ["brand_name"],
            "brand_core":      ["vision", "mission", "usp_statement", "core_values"],
            "brand_voice":     ["tone_of_voice", "key_messages"],
            "target_audience": ["primary_audience", "demographics_age"],
        }
        required = section_fields.get(section_key, [])
        if not required:
            return True
            
        filled = sum(1 for f in required if collected.get(f) not in (None, "", [], {}))
        
        # 🔒 STRICT CHECK: Must have ALL required fields filled to move forward
        return filled == len(required)
    

    def _is_onboarding_complete(self, collected: dict) -> bool:
        """True when all conversational sections have enough data for handoff."""
        return all(
            self._is_section_complete(key, collected)
            for key, _, _ in ONBOARDING_SECTIONS
        )


# ── Files API ──────────────────────────────────────────────────────────────────

def upload_file_to_claude(file_bytes: bytes, file_name: str, media_type: str) -> str:
    client = get_claude_client()
    response = client.beta.files.upload(
        file=(file_name, io.BytesIO(file_bytes), media_type),
    )
    print(f"[Files API] Uploaded '{file_name}' → {response.id}")
    return response.id


def delete_file_from_claude(file_id: str):
    try:
        get_claude_client().beta.files.delete(file_id)
        print(f"[Files API] Deleted {file_id}")
    except Exception as e:
        print(f"[Files API] Delete failed: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_brand_highlights(brand: dict) -> str:
    """Return a short comma-separated summary of what's already in the brand book."""
    highlights = []
    core = brand.get("brand_core", {})
    voice = brand.get("brand_voice", {})
    audience = brand.get("target_audience", {})
    meta = brand.get("metadata", {})

    if meta.get("brand_name"):
        highlights.append(f"brand name ({meta['brand_name']})")
    if core.get("mission"):
        highlights.append("mission")
    if core.get("vision"):
        highlights.append("vision")
    if core.get("usp_statement"):
        highlights.append("USP")
    if core.get("core_values"):
        highlights.append("core values")
    if voice.get("tone_of_voice"):
        highlights.append("tone of voice")
    if voice.get("key_messages"):
        highlights.append("key messages")
    if audience.get("primary_audience"):
        highlights.append("target audience")

    if not highlights:
        return "some initial information"
    return ", ".join(highlights[:4])  # cap at 4 to keep it concise


def _build_user_content(user_input: str, file_id: str, file_media_type: str, file_content: str, file_name: str):
    """Plain string for text-only turns, content block list when file is present."""
    if not file_id and not file_content:
        return user_input

    blocks = []

    if file_id and file_media_type:
        blocks.append({
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
        })
    elif file_content:
        label = f"Uploaded file: {file_name}\n\n" if file_name else ""
        blocks.append({"type": "text", "text": label + file_content[:12000]})

    if user_input and user_input.strip():
        blocks.append({"type": "text", "text": user_input})

    return blocks


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