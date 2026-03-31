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

## Missing Fields to Collect
{missing_fields}

STRICT RULE: Look at the Missing Fields list above. Your ONLY goal is to ask questions to fill those specific gaps. NEVER ask a question about a topic that is not on that list.

## Handling "I don't know" or Confusion
STRICT RULE: If the user says they don't know the answer or don't understand what a term means, you MUST:
1. Briefly and simply explain what the concept means in plain English.
2. Provide 2 or 3 tailored examples based on their specific industry to inspire them.
3. Ask if any of those examples resonate with them.

## What you do NOT ask about
STRICT RULE: DO NOT ask about fonts, colors, hex codes, or logos in the chat. 

## Returning clients
If the brand book already has content, open warmly, briefly highlight 2-3 things already documented to show you remember them, and invite them to continue with the next gap.

## Handling File Uploads
STRICT RULE: If the user uploads a file, you MUST summarize the key brand information you successfully gathered in a brief bulleted list, and end your reply by asking a specific question to fill the next missing gap.

## Completion & handoff
When all conversational sections are covered, wrap up warmly and introduce Lucy.
Say something exactly like: "That wraps up everything I need from you! 🎉 From here, Lucy — our brand strategist — will take over. I am just here for onboarding, but she will use everything you've shared to shape your strategy and guide your brand forward."
Do NOT continue asking questions after this message.

## Brand book so far
{brand_book}

## Free-form brand context
{brand_context}

## Context from other agents
{routing_context}
"""

class BrandOnboardingAgent(BaseAgent):
    name = "brand_onboarding"
    domain = (
        "Everything brand-related: building brand book, updating brand fields, "
        "brand strategy, file uploads for brand info, general conversation. "
        "Default agent when nothing else matches."
    )

    def _get_missing_fields(self, unified_state: dict) -> str:
        required = {
            "metadata":        ["brand_name"],
            "brand_core":      ["vision", "mission", "emotion", "competitor", "usp_statement", "market_positioning", "core_values"],
            "brand_voice":     ["tone_of_voice", "words_to_avoid", "key_messages", "taglines_and_slogans"],
            "target_audience": ["primary_audience", "demographics_age", "demographics_location", "demographics_pain_points_behaviors", "demographics_communication_style"],
        }
        missing = []
        for section, fields in required.items():
            for f in fields:
                if unified_state.get(f) in (None, "", [], {}):
                    missing.append(f.replace("_", " ").title())
        return ", ".join(missing) if missing else "None! Ready for handoff."


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
        file_content: str = None,  
        file_name: str = None,
        **kwargs
    ) -> str:
        client = get_claude_client()

        if thread is None:
            thread = thread_manager.load_thread(user_id, self.name)

        # ── 1. Load Data FIRST ────────────────────────────────────────────────
        saved_brand = get_brand_info(brand_id) or {}
        brand_context = get_latest_brand_context(brand_id) or ""
        collected = thread.get("collected_fields", {})

        # ── 2. Create Unified State ───────────────────────────────────────────
        unified_state = _get_unified_flat_state(saved_brand, collected)
        
        brand_book_txt = "\n".join(
            f"{k.replace('_', ' ').title()}: {', '.join(v) if isinstance(v, list) else v}"
            for k, v in unified_state.items()
        ) or "Nothing documented yet."

        system = SYSTEM_PROMPT.format(
            brand_book=brand_book_txt,
            brand_context=brand_context or "Not available yet.",
            routing_context=routing_context or "None.",
            missing_fields=self._get_missing_fields(unified_state) 
        )

        # ── 3. Returning User Logic ───────────────────────────────────────────
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

        # ── 4. Check Handoff (Using unified state!) ───────────────────────────
        if self._is_onboarding_complete(unified_state) and history:
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

        # ── 5. Build Message History ──────────────────────────────────────────
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

        # ── 6. Chat Call (Claude) ─────────────────────────────────────────────
        response = client.messages.create(
            model="claude-sonnet-4-6", 
            system=system,
            messages=messages,
            max_tokens=700,
        )

        reply = ""
        for block in response.content:
            if block.type == "text":
                reply += block.text
        reply = reply.strip()

        if not reply:
            reply = "Got it! Let's keep going. What else can you tell me about the brand?"

        # ── 7. Background Extraction (Gemini) ─────────────────────────────────
        try:
            from llm_client import gemini_generate
            import json as _json
            
            # STRICT KEYS ONLY - No aliases allowed to confuse the progress checker
            exact_keys = [
                "brand_name", "vision", "mission", "emotion", "competitor", "usp_statement",
                "market_positioning", "core_values", "tone_of_voice", "words_to_avoid",
                "key_messages", "taglines_and_slogans", "primary_audience", "demographics_age",
                "demographics_location", "demographics_pain_points_behaviors",
                "demographics_communication_style", "font_for_headings", "font_for_body_text",
                "primary_colors", "secondary_colors", "logo_usage_icon_only_version",
                "logo_usage_full_logo", "logo_usage_black_white_variations"
            ]
            
            gemini_system = f"""You are a highly accurate data extractor for a brand book.
            Analyze the conversation history provided and extract brand information into a JSON object.

            CRITICAL RULES:
            1. ONLY extract information explicitly stated by the USER, or explicitly CONFIRMED by the user (e.g., if AI suggests "Cheap" and User says "Yes").
            2. DO NOT extract AI examples or suggestions if the user has not confirmed them.
            3. Do NOT guess, infer, or hallucinate fields (e.g., do not invent an emotion just because you know the brand name).
            4. Use EXACTLY these JSON keys and no others: {', '.join(exact_keys)}.
            5. Return ONLY valid JSON. If nothing new/relevant was established, return {{}}.
            """
            
            # BUILD MULTI-TURN CONTEXT (So Gemini knows what "Yes" means)
            recent_history = messages[-4:] if len(messages) >= 4 else messages
            exchange_text = ""
            for m in recent_history:
                role = "AI" if m["role"] == "assistant" else "USER"
                content = m["content"]
                if isinstance(content, list):
                    content = " ".join([b.get("text", "") for b in content if b.get("type") == "text"])
                exchange_text += f"{role}: {content}\n"
            
            exchange_text += f"AI: {reply}\n"
            
            raw_json = gemini_generate(prompt=exchange_text, system=gemini_system)
            
            if raw_json:
                extracted = _json.loads(raw_json)
                extracted = {k: v for k, v in extracted.items() if v not in (None, "", [], {})}
                
                if extracted:
                    print(f"[Gemini Extract] {list(extracted.keys())}")
                    thread["collected_fields"] = _deep_merge(
                        thread.get("collected_fields", {}),
                        extracted,
                    )
        except Exception as e:
            print(f"[Gemini Extraction Error] {e}")

        return reply     

    # ── UI metadata helpers ───────────────────────────────────────────────────

    def _completion_percentage(self, collected: dict) -> int:
        from brand_finalizer import FIELD_MAP
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
        section_fields = {
            "metadata":        ["brand_name"],
            "brand_core":      ["vision", "mission", "emotion", "competitor", "usp_statement", "market_positioning", "core_values"],
            "brand_voice":     ["tone_of_voice", "words_to_avoid", "key_messages", "taglines_and_slogans"],
            "target_audience": ["primary_audience", "demographics_age", "demographics_location", "demographics_pain_points_behaviors", "demographics_communication_style"],
        }
        required = section_fields.get(section_key, [])
        if not required:
            return True
            
        filled = sum(1 for f in required if collected.get(f) not in (None, "", [], {}))
        return filled == len(required)
    
    def _is_onboarding_complete(self, collected: dict) -> bool:
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
    highlights = []
    core = brand.get("brand_core", {})
    voice = brand.get("brand_voice", {})
    audience = brand.get("target_audience", {})
    meta = brand.get("metadata", {})

    if meta.get("brand_name"): highlights.append(f"brand name ({meta['brand_name']})")
    if core.get("mission"): highlights.append("mission")
    if core.get("vision"): highlights.append("vision")
    if core.get("usp_statement"): highlights.append("USP")
    if core.get("core_values"): highlights.append("core values")
    if voice.get("tone_of_voice"): highlights.append("tone of voice")
    if voice.get("key_messages"): highlights.append("key messages")
    if audience.get("primary_audience"): highlights.append("target audience")

    if not highlights:
        return "some initial information"
    return ", ".join(highlights[:4])  

def _build_user_content(user_input: str, file_id: str, file_media_type: str, file_content: str, file_name: str):
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
            if nested: lines.append(nested)
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

def _get_unified_flat_state(saved_brand: dict, collected: dict) -> dict:
    from brand_finalizer import FIELD_MAP
    unified = {}
    
    if saved_brand:
        for flat_key, path in FIELD_MAP.items():
            node = saved_brand
            for step in path:
                if isinstance(node, dict):
                    node = node.get(step)
                else:
                    node = None
                    break
            if node not in (None, "", [], {}):
                unified[flat_key] = node
                
    for k, v in collected.items():
        if v not in (None, "", [], {}):
            unified[k] = v
            
    return unified