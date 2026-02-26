from llm_client import get_claude_client
from data_layer_vertexAI import get_brand_info, get_media_history, update_media_status, search_documentation, get_brand_context
import json
from data_layer_vertexAI import update_brand_info

# ---------------
# Models
# ---------------
MODEL_SONNET = "claude-sonnet-4-6"        # email, caption, talk
MODEL_HAIKU = "claude-haiku-4-5-20251001"  # docs/media formatting


# ---------------
# Email Agent
# ---------------

EMAIL_SYSTEM_PROMPT = """
    You are an expert email writer for a marketing agency.

    Rules:
    - Write ONLY the email content: subject line + body.
    - Do NOT add any explanation, commentary, highlights, or notes after the email.
    - Do NOT write things like "Here's your email:" or "Would you like changes?".
    - Use the brand tone and identity provided.
    - If critical info is missing, ask ONE clarifying question instead of writing the email.
    - Sign off using the brand signature if provided.
    """


def write_email(
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
    # Path B: rich contextual brand info from Vertex AI
    brand_ctx_chunks = brand_chunks or get_brand_context(brand_id, query=user_input, top_k=2)
    chunks_text = "\n\n---\n\n".join(
            [
                f"{c.get('title') or c.get('chunk_id', '')}\n{c.get('content', '')}"
                for c in brand_ctx_chunks
            ]
        ) or "No additional brand context."
    
    brand_context = f"""
        Brand Name: {brand_info.get("brand_name")}
        Tone: {brand_info.get("tone")}
        Brand Voice: {brand_info.get("brand_voice")}
        Sign Off: {brand_info.get("default_email_signature", "Best regards,\nCreativo Team")}

        Rich Brand Context:
        {chunks_text}
        """

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model=MODEL_SONNET,
        system=EMAIL_SYSTEM_PROMPT + "\n\n" + brand_context,
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
    You are a conversational AI assistant for {brand_name}.

    ## Brand Info
    {brand_context}

    ## Relevant Brand Book Sections
    {brand_book}

    - Be warm and natural.
    - Use the brand info above when answering brand-related questions.
    - Never output JSON.
    - Keep responses concise and conversational.
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

    system = TALK_SYSTEM_PROMPT.format(
        brand_name=brand_info.get("brand_name", "the brand"),
        brand_context=brand_context_str,
        brand_book=chunks_text
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
    You are a helpful assistant that answers questions using the provided documentation.
    - Answer based ONLY on the documentation chunks provided.
    - If the answer is not in the docs, say so clearly.
    - Be concise and direct.
    - Never output JSON.
    - Do NOT add commentary after your answer.
    - CRITICAL: If the documentation chunk contains a "Media in this section" block with URLs, you MUST include ALL of them in your response using the exact formats below. Never skip or omit media.
    - Images → render as: ![image](url)
    - YouTube videos → render as a labeled link: 🎬 [Watch Video](url)
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

    media_block = ("\n\nIMPORTANT — include these media links verbatim in your response:\n" + "\n".join(media_summary)) if media_summary else ""

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
CONVERSATIONAL_SYSTEM_PROMPT="""You are a warm, conversational AI assistant.
    Communicate the action result naturally and conversationally.
    - Stay strictly aligned with what actually happened.
    - Be concise and friendly.
    - Do NOT invent details.
    - Do NOT add suggestions or next steps unless the result explicitly mentions them.
    - Never output raw JSON."""

def _conversational_response(client, user_input: str, conversation_history: list, action_result: str) -> str:

    prompt = f"User asked: {user_input}\n\nAction completed: {action_result}\n\nCommunicate this result conversationally to the user."
    if client is None:
        client = get_claude_client()

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model=MODEL_HAIKU,
        system=CONVERSATIONAL_SYSTEM_PROMPT,
        messages=messages,
        max_tokens=600
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return "Sorry, I couldn't respond."

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

    media_ids = kwargs.get("image_ids", [])
    succeeded = [mid for mid in media_ids if update_media_status(mid, "approved")]
    failed = [mid for mid in media_ids if mid not in succeeded]
    action_result = f"Successfully approved {len(succeeded)} item(s): {', '.join(succeeded)}."
    if failed:
        action_result += f" The following IDs were not found: {', '.join(failed)}."
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

    media_ids = kwargs.get("image_ids", [])
    succeeded = [mid for mid in media_ids if update_media_status(mid, "rejected")]
    failed = [mid for mid in media_ids if mid not in succeeded]
    action_result = f"Successfully rejected {len(succeeded)} item(s): {', '.join(succeeded)}."
    if failed:
        action_result += f" The following IDs were not found: {', '.join(failed)}."
    return _conversational_response(client, user_input, conversation_history, action_result)


# ---------------
# Media Search Agent
# ---------------

MEDIA_SEARCH_SYSTEM_PROMPT = """
You are a media search assistant.
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

    from data_layer_vertexAI import search_media
    results = search_media(
        user_id=user_id,
        brand_id=brand_id,
        status=status,
        media_type=media_type,
        platform=platform
    )

    if not results:
        return "No media found matching your criteria."

    import json
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

BRAND_UPDATE_SYSTEM_PROMPT = """
    You are a brand information collection assistant.

    Your job:
    1. Analyze the user's message and conversation history.
    2. Extract any brand information the user wants to update or provide.
    3. Return ONLY a valid JSON object. No explanation, no markdown fences.

    Extractable fields:
    - brand_name, industry, mission, vision, values, tone, brand_voice,
    communication_style, target_audience, primary_goal, default_email_signature

    If no clear update is requested, return:
    {"action": "clarify", "message": "<your clarifying question>"}

    If updates are found, return:
    {"action": "update", "changes": { "<field>": "<value>", ... }}
    """


def brand_update_agent(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    client=None,
    **kwargs
):


    if client is None:
        client = get_claude_client()

    # Build conversation history for Claude
    messages = [{"role": m["role"], "content": m["content"]} for m in conversation_history]
    messages.append({"role": "user", "content": user_input})

    # Call Claude Haiku for structured JSON extraction
    response = client.messages.create(
        model=MODEL_HAIKU,
        system=BRAND_UPDATE_SYSTEM_PROMPT,
        messages=messages,
        max_tokens=500
    )

    raw = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw += block.text

    # Strip code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        return _conversational_response(
            client,
            user_input,
            conversation_history,
            "Could not understand the brand update request."
        )

    action = parsed.get("action")

    if action == "clarify":
        return parsed.get("message", "Could you clarify what you'd like to update?")

    if action == "update":
        changes = parsed.get("changes", {})
        if not changes:
            return _conversational_response(
                client,
                user_input,
                conversation_history,
                "No brand fields were found to update."
            )

        # Apply changes directly to Firestore brands doc
        update_brand_info(brand_id, changes)
        fields_updated = ", ".join(changes.keys())
        action_result = f"Successfully updated brand fields: {fields_updated}. Changes will sync to brand book vectors within 5 minutes."

        return _conversational_response(client, user_input, conversation_history, action_result)

    # Fallback
    return _conversational_response(
        client,
        user_input,
        conversation_history,
        "Could not process the brand update.")


# ---------------
# Caption Agent
# ---------------

CAPTION_SYSTEM_PROMPT = """
You are an expert social media content writer.

Platform guidelines:
- Instagram: Visual-first, aspirational, hook + body + CTA, under 150 words, 3-5 hashtags
- LinkedIn: Professional, thought leadership, 1-3 hashtags max
- Facebook: Conversational, community-focused, medium length
- TikTok: Fun, punchy, trend-aware, very short, 3-5 hashtags
- Twitter/X: Under 280 chars, witty or bold, 1-2 hashtags max
- YouTube: SEO-friendly description, detailed

CRITICAL RULES:
- Return ONLY the caption text itself. Nothing else.
- No intro like "Here's your caption:"
- No explanation after the caption
- No bullet points about what you did
- No "Would you like changes?" at the end
- Start directly with the caption content
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
    **kwargs
):
    if client is None:
        client = get_claude_client()

    brand_info = get_brand_info(brand_id) or {}
    chunks_text = "\n\n---\n\n".join(
        [f"{c.get('title', '')}\n{c.get('content', '')}" for c in (brand_chunks or [])]
    ) or "No additional brand context."

    brand_context = f"""
Brand Name: {brand_info.get("brand_name")}
Tone: {brand_info.get("tone")}
Brand Voice: {brand_info.get("brand_voice")}
Target Audience: {brand_info.get("target_audience")}

Relevant Brand Book Sections:
{chunks_text}
"""

    platform_note = f"Platform: {platform}" if platform else "Platform: not specified — infer from context."
    topic_note = f"Topic/Subject: {topic}" if topic else ""

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model=MODEL_SONNET,
        system=CAPTION_SYSTEM_PROMPT + f"\n\n## Brand Context\n{brand_context}\n\n{platform_note}\n{topic_note}",
        messages=messages,
        max_tokens=600
    )

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return "Could not generate caption."