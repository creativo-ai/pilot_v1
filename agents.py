from llm_client import get_claude_client
from data_layer_vertexAI import (
    get_brand_info, get_media_history, update_media_status,
    search_documentation, get_brand_context, get_latest_brand_context,
    update_brand_info, search_media, get_media_by_id
)
import json

# ---------------
# Models
# ---------------
MODEL_SONNET = "claude-sonnet-4-6"        # email, caption, talk
MODEL_HAIKU = "claude-haiku-4-5-20251001"  # docs/media formatting


# ---------------
# Extracting values
# ---------------


def get_nested(data, *keys, default=""):
    """
    Safely extract a nested field from a dictionary.
    Example:
        get_nested(brand_info, "brand_personality", "tone_of_voice", "primary_tone")
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        else:
            return default
    return data if data else default

# ---------------
# Email Agent
# ---------------

EMAIL_SYSTEM_PROMPT = """
## Information priority (applies on conflict only)
1. Active Client Brand Context below → source of truth for brand name, tone, audience, values
2. Conversation history → continuity and context
When no conflict exists, combine all sources for richest output.

You are an expert email writer working inside Creativo, a marketing agency.
Creativo's team uses you to draft emails on behalf of the agency or its clients.

## Active Client Brand
{agency_context}

## Rules
- The brand context above is for the CLIENT brand currently active — use it for tone, audience, and voice.
- Write ONLY the email: subject line + body. No commentary before or after.
- Do NOT write "Here's your email:" or "Would you like changes?".
- Sign off using the brand signature provided.
- If writing on behalf of a client, use the client's voice and audience — not Creativo's.
- If writing for Creativo itself (e.g. agency outreach), use Creativo's brand voice.
- ONLY ask a question if the request has absolutely no topic, product, or subject at all.
- If you have any topic or product from the conversation, write the email immediately.
- Never ask the same question twice.

## If Brand Info Is Missing
{missing_note}
"""


def write_email(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    brand_chunks: list = None,
    missing_brand_info: str = None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    # Layer 1: Structured fields from Firestore brands doc (always loaded — fast, small)
    brand_info = get_brand_info(brand_id) or {}


    brand_name = get_nested(brand_info, "metadata", "brand_name")
    tone = get_nested(brand_info, "brand_personality", "tone_of_voice", "primary_tone")
    brand_voice = get_nested(brand_info, "brand_personality", "brand_voice_description")
    target_audience = get_nested(brand_info, "target_audience", "primary_audience", "demographics", "age_range")  # adjust if needed
    mission = get_nested(brand_info, "brand_foundation", "mission")
    values = ", ".join(get_nested(brand_info, "brand_foundation", "core_values", default=[]))
    communication_style = get_nested(brand_info, "brand_personality", "personality_traits")  # or whatever fits
    sign_off = get_nested(
        brand_info, "email_branding", "default_signature", 
        default=f"Best regards,\n{brand_name} Team"
    )


    # Layer 2: Brand context embedding summary (always loaded — narrative richness)
    brand_summary = get_latest_brand_context(brand_id) or ""

    # Layer 3: On-demand — only load full brand book fields if Layer 1+2 seem thin
    agency_context = f"""Brand Name: {brand_name}
Tone: {tone}
Brand Voice: {brand_voice}
Target Audience: {target_audience}
Mission: {mission}
Values: {values}
Communication Style: {communication_style}
Sign Off: {sign_off}
{("\nBrand Context Summary:\n" + brand_summary) if brand_summary else ""}
"""

    missing_note = (
        f"The following info was provided by the user and should be used: {missing_brand_info}"
        if missing_brand_info else
        "All brand info is available above."
    )

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    system = EMAIL_SYSTEM_PROMPT.format(
        brand_name=brand_name,
        agency_context=agency_context,
        missing_note=missing_note
    )

    response = client.messages.create(
        model=MODEL_SONNET,
        system=system,
        messages=messages,
        max_tokens=1000
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return "Could not generate email."


# ---------------
# Talk Agent
# ---------------

TALK_SYSTEM_PROMPT = """
You are a helpful internal assistant for Creativo, a marketing agency. You help Creativo's team members with questions about their clients' brands, content strategy, and general tasks. The active client brand context is provided below.

## Brand Info
{brand_context}

## Relevant Brand Book Sections
{brand_book}

## Background Context (from previous agent)
{routing_context}

## Rules
- Be warm and natural.
- Use the brand info above when answering brand-related questions.
- Never output JSON.
- Keep responses concise and conversational.
- If background context contains [RESOLVED] topics — do NOT ask about them or follow up on them.
  They are history. Focus only on the user's current message.
"""


def talk_agent(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    brand_chunks: list = None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    # Path A: structured fields from Firestore
    brand_info = get_brand_info(brand_id) or {}
    brand_context_str = f"""
        Brand Name: {brand_info.get("brand_name", "Unknown")}
        Tone: {brand_info.get("tone", "")}
        Mission: {brand_info.get("mission", "")}
        Vision: {brand_info.get("vision", "")}
        Values: {brand_info.get("values", "")}
        Target Audience: {brand_info.get("target_audience", "")}
        """
    # Path B: rich brand context from Vertex AI for nuanced questions
    ctx_chunks = brand_chunks or get_brand_context(brand_id, query=user_input, top_k=2)
    chunks_text = "\n\n---\n\n".join(
        [f"{c.get('title', c.get('chunk_id',''))}\n{c.get('content', '')}" for c in ctx_chunks]
    ) or "No additional brand context."

    routing_ctx = kwargs.get("routing_context", "") or ""
    system = TALK_SYSTEM_PROMPT.format(
        brand_name=brand_info.get("brand_name", "the brand"),
        brand_context=brand_context_str,
        brand_book=chunks_text,
        routing_context=routing_ctx if routing_ctx else "None."
    )

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model=MODEL_SONNET,
        system=system,
        messages=messages,
        max_tokens=600
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return "Sorry, I couldn't respond."


# ---------------
# Search Docs Agent
# ---------------

SEARCH_DOCS_SYSTEM_PROMPT = """
    You are a helpful assistant that answers app how-to questions using the provided documentation.
    - Answer based ONLY on the documentation chunks provided.
    - If the answer is not in the docs, say clearly: "I don't have documentation on that yet."
    - Be concise and direct.
    - Never output JSON.
    - Do NOT add commentary after your answer.
    - NEVER ask the user a clarifying question — if you cannot answer from the docs, say so and stop.
    - NEVER ask "Could you clarify what you mean?" or "Can you tell me more?" — not your job.

    MEDIA PLACEMENT RULES — follow exactly:
    - Media (images and videos) must appear INLINE directly after the step or sentence they illustrate.
    - NEVER group all media at the end of the response.
    - If a step says "click the Sign In button" and there is a screenshot of it, place the image immediately after that sentence.
    - If there is a walkthrough video, place it at the start before the steps.
    - Images → render as: ![image](url)
    - YouTube videos → render as: 🎬 [Watch Video](url)
    - CRITICAL: Include ALL media from the documentation. Never skip any.
    """


def search_docs_agent(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    query: str = None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    # Use query arg if provided by tool call, otherwise use user_input
    search_query = query or user_input

    docs = search_documentation(query=search_query, top_k=3)

    if not docs:
        return "I couldn't find relevant documentation for your question."

    def format_doc(d):
        text = f"Source: {d.get('filename', d.get('source', 'unknown'))}\n{d.get('content', d.get('text', ''))}"
        media_urls = d.get("media_urls", [])
        if media_urls:
            media_lines = []
            for url in media_urls:
                if "youtube.com/embed" in url:
                    media_lines.append(f"- Video: {url}")
                else:
                    media_lines.append(f"- Image: {url}")
            text += "\n\nMedia in this section:\n" + "\n".join(media_lines)
        return text

    docs_context = "\n\n---\n\n".join([format_doc(d) for d in docs])

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    # Inject media URLs explicitly into the user-facing message so Haiku never skips them
    media_summary = []
    for d in docs:
        for url in d.get("media_urls", []):
            if "youtube.com/embed" in url:
                media_summary.append(f"[Watch Video]({url})")
            else:
                media_summary.append(f"![image]({url})")

    media_block = (
        "\n\nMEDIA PLACEMENT INSTRUCTION: Place each of these media items INLINE "
        "directly after the step or sentence they illustrate — do NOT group them at the end:\n"
        + "\n".join(media_summary)
    ) if media_summary else ""

    response = client.messages.create(
        model=MODEL_HAIKU,
        system=SEARCH_DOCS_SYSTEM_PROMPT,
        messages=messages[:-1] + [
            {"role": "user", "content": f"{user_input}\n\nDocumentation:\n{docs_context}{media_block}"}
        ],
        max_tokens=800
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return "Sorry, I couldn't find an answer."


# ---------------
# Conversational Response Wrapper
# ---------------
CONVERSATIONAL_SYSTEM_PROMPT = """
You are a helpful internal assistant for a marketing agency.
Communicate the action result naturally and conversationally.
    - Stay strictly aligned with what actually happened.
    - Be concise and friendly.
    - Do NOT invent details or suggest external platform changes.
    - Do NOT add suggestions or next steps unless the result explicitly mentions them.
    - Never output raw JSON."""

def _conversational_response(client, user_input: str, conversation_history: list, action_result: str) -> str:
    if client is None:
        client = get_claude_client()

    # Do NOT pass conversation history — old messages cause the LLM to override
    # the action_result with stale status information from previous turns.
    # The action_result is the ONLY source of truth for what just happened.
    messages = [
        {
            "role": "user",
            "content": (
                f"The user requested: {user_input}\n\n"
                f"Action result: {action_result}\n\n"
                f"Communicate this result conversationally in 1-3 sentences. "
                f"Report EXACTLY what the action result says — do not change, reinterpret, "
                f"or contradict it under any circumstances."
            )
        }
    ]

    response = client.messages.create(
        model=MODEL_HAIKU,
        system=CONVERSATIONAL_SYSTEM_PROMPT,
        messages=messages,
        max_tokens=300
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return action_result  # fallback: return raw result if LLM fails

# ---------------
# Image Agents
# ---------------

def list_pending_media(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    media_items = get_media_history(user_id=user_id, brand_id=brand_id)
    pending_media = [img for img in media_items if img.get("status", "").lower() == "pending"]

    if not pending_media:
        action_result = "The user has no media pending approval."
    else:
        lines = "\n".join(
            [f"- ID {img.get('id')}: {img.get('description', 'No description')} | Platform: {img.get('platform', '')} | Type: {img.get('type', 'image')} | Date: {img.get('created_at', '')}"
             for img in pending_media]
        )
        action_result = f"Found {len(pending_media)} pending media items:\n{lines}"

    return _conversational_response(client, user_input, conversation_history, action_result)


def approve_media(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    media_ids = kwargs.get("media_ids", kwargs.get("image_ids", []))
    succeeded = []
    skipped = []
    not_found = []

    for mid in media_ids:
        doc = get_media_by_id(mid)
        if not doc:
            not_found.append(mid)
        elif doc.get("status") != "pending":
            skipped.append(f"{mid} (currently {doc.get('status', 'unknown')})")
        else:
            if update_media_status(mid, "approved"):
                succeeded.append(mid)
            else:
                not_found.append(mid)

    parts = []
    if succeeded: parts.append(f"Approved: {', '.join(succeeded)}")
    if skipped:   parts.append(f"Skipped (not pending): {', '.join(skipped)}")
    if not_found: parts.append(f"Not found: {', '.join(not_found)}")
    action_result = " | ".join(parts) or "No items were actioned."
    print(f"[approve_media] action_result: {action_result}")
    return _conversational_response(client, user_input, conversation_history, action_result)


def reject_media(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    media_ids = kwargs.get("media_ids", kwargs.get("image_ids", []))
    succeeded = []
    skipped = []
    not_found = []

    for mid in media_ids:
        doc = get_media_by_id(mid)
        if not doc:
            not_found.append(mid)
        elif doc.get("status") != "pending":
            skipped.append(f"{mid} (currently {doc.get('status', 'unknown')})")
        else:
            if update_media_status(mid, "rejected"):
                succeeded.append(mid)
            else:
                not_found.append(mid)

    parts = []
    if succeeded: parts.append(f"Rejected: {', '.join(succeeded)}")
    if skipped:   parts.append(f"Skipped (not pending): {', '.join(skipped)}")
    if not_found: parts.append(f"Not found: {', '.join(not_found)}")
    action_result = " | ".join(parts) or "No items were actioned."
    print(f"[reject_media] action_result: {action_result}")
    return _conversational_response(client, user_input, conversation_history, action_result)


# ---------------
# Media Search Agent
# ---------------

MEDIA_SEARCH_SYSTEM_PROMPT = """
You are an internal media library tool for Creativo, a marketing agency. The media shown belongs to the client brand currently active in the system.
Summarize the media results in a clear, friendly, conversational way.
- Group by type (images vs videos) if both are present.
- Include ID, description, platform, date, and status for each item.
- Use emojis for status: ✅ approved, ⏳ pending, ❌ rejected.
- If no results found, say so clearly.
- Never output raw JSON.
- Do NOT add suggestions or commentary after the list.
"""


def search_media_agent(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    status: str = None,
    media_type: str = None,
    platform: str = None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    results = search_media(
        user_id=user_id,
        brand_id=brand_id,
        status=status,
        media_type=media_type,
        platform=platform
    )

    if not results:
        return "No media found matching your criteria."

    media_context = json.dumps(results, indent=2)

    response = client.messages.create(
        model=MODEL_HAIKU,
        system=MEDIA_SEARCH_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"User asked: {user_input}\n\nMedia results:\n{media_context}"}
        ],
        max_tokens=800
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return "Could not retrieve media."


# ---------------
# Brand Update Agent
# ---------------



def brand_update_agent(
    user_input: str = None,
    conversation_history: list = None,
    user_id: str = None,
    brand_id: str = None,
    client=None,
    update_request: str = None,
    changes: dict = None,
    **kwargs
):
    """
    Dual-mode brand update:
    - Orchestrator tool call: receives user_input + update_request, extracts fields via Gemini.
    - Internal call (finalizer/flush): receives brand_id + changes dict directly.
    """
    # Internal call path (from flush_on_exit / finalizer)
    if changes is not None and brand_id and user_input is None:
        if not changes:
            return False
        update_brand_info(brand_id, changes)
        print(f"[BrandUpdate] Fields written to Firestore: {list(changes.keys())}")
        return True

    # Orchestrator call path — extract fields from user_input via Gemini
    if client is None:
        client = get_claude_client()

    from llm_client import gemini_generate
    import json as _json

    source = update_request or user_input or ""
    raw = gemini_generate(
        prompt=f"User request: {source}",
        system=(
            "Extract brand fields to update. Return ONLY valid JSON. No markdown.\n"
            "Extractable fields: brand_name, industry, mission, vision, values, tone, "
            "brand_voice, communication_style, target_audience, primary_goal, "
            "default_email_signature, tagline, positioning, brand_archetype.\n"
            "Example: {\"tone\": \"bold and direct\", \"mission\": \"new mission\"}\n"
            "If nothing clear to extract, return {}"
        )
    )

    try:
        extracted = _json.loads(raw.strip())
    except Exception:
        extracted = {}

    extracted = {k: v for k, v in extracted.items() if v not in (None, "", [], {})}

    if not extracted:
        return "I couldn\'t identify which brand fields to update. Could you be more specific? For example: \'change our tone to bold and direct\' or \'update our mission to...\'."

    update_brand_info(brand_id, extracted)
    fields = ", ".join(extracted.keys())
    print(f"[BrandUpdate] Orchestrator updated fields: {fields}")

    action_result = f"Successfully updated brand fields: {fields}."
    return _conversational_response(client, user_input or source, conversation_history or [], action_result)


# ---------------
# Caption Agent
# ---------------

CAPTION_SYSTEM_PROMPT = """
You are a social media writer for {agency_name}.
Write every caption immediately. Never ask clarifying questions unless the platform is missing.

## Priority order — always follow this
1. BRAND FACTS section below: Agency Name, Industry, Mission, Vision, Values, Tone, Audience
   → These are the source of truth. Always use them. They override everything else.
2. Additional Brand Context section: background narrative — use only to add depth, never to override facts above
3. Conversation history: for continuity only — never use it to determine brand name, industry, or audience

If history or brand context narrative contradicts the BRAND FACTS, ignore the contradiction and use BRAND FACTS.

## How to write
Read Industry and Audience from BRAND FACTS to understand what kind of content to write.
Write a post about the given topic in the brand's voice, for the brand's audience, using the brand name from BRAND FACTS.

## When Media Details are provided
If a "## Media Details (fetched from Firestore)" section is present in this prompt:
- It contains the exact media the user is referring to — use its description as the topic.
- Use the platform from the media details as the target platform.
- Do NOT ask what the media is about — the answer is already there.
- Write the caption specifically about that media item.

When platform is missing AND no media details provided → ask for platform once. That is the only question you may ask.

## Format
Return ONLY the caption. No preamble. Start directly with the content.
Always use the brand name from BRAND FACTS in the post or hashtags.

## Platform style
- Instagram: Hook + value + CTA, under 150 words, 3-5 hashtags, emojis welcome
- LinkedIn: Professional, insight-driven, 1-3 hashtags, minimal emojis
- Facebook: Conversational, community feel, medium length
- TikTok: Very short, punchy, 3-5 hashtags
- Twitter/X: Under 280 chars, bold, 1-2 hashtags
"""

def write_caption_agent(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    platform: str = None,
    topic: str = None,
    brand_chunks: list = None,
    missing_brand_info: str = None,
    routing_context: str = None,
    **kwargs
):
    if client is None:
        client = get_claude_client()

    brand_info = get_brand_info(brand_id) or {}
    brand_summary = get_latest_brand_context(brand_id) or ""

    print(f"[Caption] brand_id={brand_id}, metadata.brand_name={brand_info.get('metadata',{}).get('brand_name','?')}, summary_len={len(brand_summary)}")

    def _read(brand_info, *paths):
        """
        Try each path in order. A path is a list of keys to traverse.
        Falls back to checking flat keys directly on brand_info.
        Returns first non-empty string found.
        """
        for path in paths:
            if isinstance(path, str):
                path = [path]
            node = brand_info
            for key in path:
                if isinstance(node, dict):
                    node = node.get(key, {})
                else:
                    node = {}
                    break
            # Flatten result to string
            if isinstance(node, str) and node.strip():
                return node.strip()
            if isinstance(node, list) and node:
                flat = ", ".join(str(v) for v in node if v)
                if flat:
                    return flat
            if isinstance(node, dict) and node:
                # Try common string keys inside the dict
                for k in ("description", "primary_tone", "summary", "text", "statement"):
                    if isinstance(node.get(k), str) and node[k].strip():
                        return node[k].strip()
                # Last resort: join non-empty string values
                flat = ", ".join(str(v) for v in node.values() if isinstance(v, str) and v.strip())
                if flat:
                    return flat
        return ""

    # Read each field trying multiple possible paths (nested → flat fallback)
    brand_name    = _read(brand_info, ["metadata", "brand_name"], ["brand_name"])
    industry      = _read(brand_info, ["metadata", "industry"], ["industry"])
    mission       = _read(brand_info, ["brand_foundation", "mission"], ["mission"])
    vision        = _read(brand_info, ["brand_foundation", "vision"], ["vision"])
    values        = _read(brand_info, ["brand_foundation", "core_values"], ["values"], ["core_values"])
    tone          = _read(brand_info, ["brand_personality", "tone_of_voice", "primary_tone"],
                                      ["brand_personality", "tone_of_voice"],
                                      ["tone"])
    brand_voice   = _read(brand_info, ["brand_personality", "brand_voice_description"], ["brand_voice"])
    target_aud    = _read(brand_info, ["target_audience", "primary_audience", "description"],
                                      ["target_audience", "primary_audience"],
                                      ["target_audience"])
    tagline       = _read(brand_info, ["brand_foundation", "tagline"], ["tagline"])
    positioning   = _read(brand_info, ["brand_foundation", "positioning_statement"], ["positioning"])
    personality   = _read(brand_info, ["brand_personality", "personality_traits"], ["personality_traits"])
    content_pill  = _read(brand_info, ["content_strategy", "content_pillars"], ["content_pillars"])
    hashtags      = _read(brand_info, ["content_strategy", "hashtag_strategy"], ["hashtags"])

    # Build brand context — only include fields that have actual values
    parts = []
    if brand_name:       parts.append(f"Agency Name: {brand_name}")
    if industry:         parts.append(f"Industry: {industry}")
    if mission:          parts.append(f"Mission: {mission}")
    if vision:           parts.append(f"Vision: {vision}")
    if values:           parts.append(f"Values: {values}")
    if tone:             parts.append(f"Tone: {tone}")
    if brand_voice:      parts.append(f"Brand Voice: {brand_voice}")
    if personality:      parts.append(f"Personality: {personality}")
    if target_aud:       parts.append(f"Target Audience: {target_aud}")
    if tagline:          parts.append(f"Tagline: {tagline}")
    if positioning:      parts.append(f"Positioning: {positioning}")
    if content_pill:     parts.append(f"Content Pillars: {content_pill}")
    if hashtags:         parts.append(f"Hashtag Strategy: {hashtags}")
    if brand_summary:    parts.append(f"\nAdditional Brand Context (supplementary only — structured fields above take priority):\n{brand_summary}")
    if missing_brand_info: parts.append(f"Additional info: {missing_brand_info}")

    brand_context = "\n".join(parts) if parts else "No brand context available yet."
    print(f"[Caption] brand_context fields: {[p.split(':')[0] for p in parts]}")

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    # Resolve which media item(s) the user is referring to for caption writing.
    # Uses a single Gemini call to understand the reference from full context —
    # no hardcoded keyword matching. Returns a list of media IDs to fetch from Firestore.
    import json as _json
    media_detail_section = ""
    media_ids = []

    try:
        import thread_manager as _tm
        from data_layer_vertexAI import search_media as _sm
        from llm_client import gemini_generate

        # Fetch current media list from Firestore
        all_media = _sm(user_id=user_id, brand_id=brand_id)
        media_summary = _json.dumps([
            {"id": m.get("id"), "type": m.get("type"), "description": m.get("description", ""),
             "platform": m.get("platform", ""), "status": m.get("status", ""),
             "created_at": m.get("created_at", "")}
            for m in all_media
        ])

        # Build context for Gemini — session state + user request
        session_context = f"Last media actions: {routing_context}" if routing_context else "No recent media actions."

        resolved = gemini_generate(
            prompt=(
                f"User request: {user_input}\n\n"
                f"Session context: {session_context}\n\n"
                f"Available media:\n{media_summary}"
            ),
            system=(
                "You are resolving which media item(s) the user wants to write a caption for.\n"
                "Rules:\n"
                "- If the user mentions a specific ID (image_001 etc.), return that ID.\n"
                "- If the user says 'latest approved' or 'most recently approved', return the ID "
                "  from the session context that was most recently approved (not sorted by created_at).\n"
                "- If the user says 'last image/video', return the last item of that type from the media list.\n"
                "- If the user references something discussed in session context, use those IDs.\n"
                "- If no clear reference, return empty.\n"
                "Return ONLY a JSON array of media IDs, e.g. [\"image_001\"] or []. No explanation."
            )
        )

        try:
            resolved_ids = _json.loads(resolved.strip())
            if isinstance(resolved_ids, list):
                media_ids = [m for m in resolved_ids if isinstance(m, str)]
                print(f"[Caption] Gemini resolved media IDs: {media_ids}")
        except Exception:
            print(f"[Caption] Gemini resolution parse failed: {resolved}")

    except Exception as _e:
        print(f"[Caption] media resolution error: {_e}")

    if media_ids:
        try:
            lines = []
            for mid in media_ids:
                media = get_media_by_id(mid)
                if media:
                    lines.append(
                        f"- {mid}: {media.get('description', media.get('title', 'unknown'))} | "
                        f"Platform: {media.get('platform', 'unknown')} | "
                        f"Campaign: {media.get('campaign_name', 'n/a')}"
                    )
            if lines:
                media_detail_section = "\n\n## Media Details (fetched from Firestore)\n" + "\n".join(lines)
                print(f"[Caption] fetched media details for: {media_ids}")
        except Exception as e:
                print(f"[Caption] media fetch error: {e}")

    routing_section = (
        f"\n\n## Recent Context\n{routing_context}"
    ) if routing_context else ""

    # Fill agency_name placeholder in system prompt dynamically
    filled_prompt = CAPTION_SYSTEM_PROMPT.replace("{agency_name}", brand_name or "our agency")

    system = (
        filled_prompt
        + f"\n\n## BRAND FACTS (Firestore — source of truth on conflict)\n{brand_context}"
        + f"\n\nPlatform: {platform or 'infer from context or ask once'}"
        + (f"\nTopic: {topic}" if topic else "")
        + routing_section
        + media_detail_section  # always injected — even when routing_context is None
    )


    response = client.messages.create(
        model=MODEL_SONNET,
        system=system,
        messages=messages,
        max_tokens=600
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return "Could not generate caption."


# ─────────────────────────────────────────────────────────────────────────────
# PASSIVE BRAND LEARNING
# ─────────────────────────────────────────────────────────────────────────────

def capture_brand_info_in_background(
    user_input: str,
    agent_question: str,
    user_answer: str,
    user_id: str,
    brand_id: str
):
    """
    Called silently by any agent when it asks the user a brand question and gets an answer.
    Forwards the Q&A exchange to the onboarding agent in a background thread.
    The onboarding agent merges the info into the active session and resets the 5-min timer.
    The user sees nothing — their request was already completed before this runs.
    """
    import threading

    def _run():
        try:
            from brand_onboarding_agent import BrandOnboardingAgent
            import thread_manager

            agent = BrandOnboardingAgent()
            # Build a minimal exchange so the onboarding agent can extract the field
            exchange = (
                f"[Background capture — do not reply to the user, just extract and store] "
                f"Agent asked: {agent_question} "
                f"User answered: {user_answer}"
            )
            # Load onboarding thread and run extraction only — no reply needed
            thread = thread_manager.load_thread(user_id, "brand_onboarding")
            if thread.get("brand_id") and thread["brand_id"] != brand_id:
                # Different brand — clear stale state
                thread["messages"] = []
                thread["collected_fields"] = {}
                thread["summary"] = ""
            thread["brand_id"] = brand_id

            collected = thread.get("collected_fields", {})
            agent._extract_and_store(
                user_input=user_answer,
                assistant_reply=agent_question,
                collected=collected,
                user_id=user_id,
                brand_id=brand_id,
                thread=thread
            )
            # Touch the thread timestamp so the finalizer's 5-min timer resets
            thread_manager.touch_thread(user_id, "brand_onboarding")
            from debug_hooks import log_passive_capture
            log_passive_capture(
                question=agent_question[:120],
                answer=user_answer[:120],
                brand_id=brand_id
            )
        except Exception as e:
            from debug_hooks import log_error
            log_error(f"Background capture failed: {e}", context=user_answer[:80])

    t = threading.Thread(target=_run, daemon=True)
    t.start()