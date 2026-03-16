from llm_client import get_claude_client
from data_layer_vertexAI import get_brand_info, get_media_history, update_media_status, search_documentation, get_brand_context, get_brand_field, get_latest_brand_context
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
    brand_name = brand_info.get("brand_name", "the agency")

    # Layer 2: Brand context embedding summary (always loaded — narrative richness)
    brand_summary = get_latest_brand_context(brand_id) or ""

    # Layer 3: On-demand — only load full brand book fields if Layer 1+2 seem thin
    # Check if tone or voice are missing from structured fields before fetching
    extra_fields = {}
    for field in ["tone", "brand_voice", "target_audience", "communication_style"]:
        if not brand_info.get(field):
            val = get_brand_field(brand_id, field)
            if val:
                extra_fields[field] = val

    agency_context = f"""Brand Name: {brand_name}
Tone: {brand_info.get("tone") or extra_fields.get("tone", "")}
Brand Voice: {brand_info.get("brand_voice") or extra_fields.get("brand_voice", "")}
Target Audience: {brand_info.get("target_audience") or extra_fields.get("target_audience", "")}
Mission: {brand_info.get("mission", "")}
Values: {brand_info.get("values", "")}
Communication Style: {brand_info.get("communication_style") or extra_fields.get("communication_style", "")}
Sign Off: {brand_info.get("default_email_signature", f"Best regards,\n{brand_name} Team")}
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
CONVERSATIONAL_SYSTEM_PROMPT="""You are a helpful internal assistant for Creativo, a marketing agency.
    Communicate the action result naturally and conversationally to the Creativo team member.
    - Stay strictly aligned with what actually happened.
    - Be concise and friendly.
    - Do NOT invent details or suggest external platform changes.
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

    media_ids = kwargs.get("media_ids", kwargs.get("image_ids", []))
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

    media_ids = kwargs.get("media_ids", kwargs.get("image_ids", []))
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



def brand_update_agent(brand_id: str, changes: dict) -> bool:
    """
    Internal function — called only by flush_on_exit and flush_on_inactivity.
    Writes collected brand field changes to Firestore using correct nested paths.
    NOT a user-facing agent. Never called directly from routing.

    Args:
        brand_id: Firestore brand document ID
        changes: dict of flat field keys and new values e.g. {"brand_name": "Zerox", "mission": "..."}
    Returns:
        True if successful
    """
    if not changes:
        return False
    update_brand_info(brand_id, changes)
    print(f"[BrandUpdate] Fields written to Firestore: {list(changes.keys())}")
    return True


# ---------------
# Caption Agent
# ---------------

CAPTION_SYSTEM_PROMPT = """
You are an expert social media content writer working inside Creativo, a marketing agency.
Creativo's team uses you to write captions for their clients' social media posts.

## Your Role
You write captions FOR Creativo's clients. Always write AS the client brand — in their voice, for their audience.
Use the client brand context below (tone, values, audience, personality) to shape every caption.

## Two Caption Types
1. Client's own content (product, campaign, announcement) → write in the CLIENT's voice directly
   Example client = burger restaurant: "Summer just got tastier 🍔☀️ Come try our new..."
2. Creativo showcasing work FOR a client → write as Creativo talking about the work
   Example: "We built a full summer campaign for our client in the F&B space 🎨"
   Only use this type if the user explicitly says they're posting about agency work.

Default to type 1 (client voice) unless told otherwise.

## When to Write Immediately
- Platform given → write now, no questions
- "Another post for [platform]" → platform is known, write now
- Client name or topic mentioned → write now, client name is enough
- If you have ANY of: platform + topic, platform + client, or platform + context → write immediately
- The client name is a nice detail but NOT required — if missing, write "our client" or infer from topic

## Platform Guidelines
- Instagram: Visual-first, hook + body + CTA, under 150 words, 3-5 hashtags
- LinkedIn: Professional, thought leadership, 1-3 hashtags max
- Facebook: Conversational, community-focused, medium length
- TikTok: Fun, punchy, trend-aware, very short, 3-5 hashtags
- Twitter/X: Under 280 chars, witty or bold, 1-2 hashtags max
- YouTube: SEO-friendly description, detailed

## Critical Rules
- Return ONLY the caption text. No intro, no explanation, no "Would you like changes?".
- Start directly with the caption content.
- Use Brand Context below — NEVER ask for brand name, tone, or voice.
- If platform is missing AND cannot be inferred → ask for it ONCE, that is your ONLY allowed question.
- Read full conversation history — use any platform, client, media title, or topic already mentioned.
- NEVER ask for client name — if missing, write "our client" and move on.
- If conversation history shows a media item (video/image title, ID, platform) → use it as the topic immediately.
- NEVER ask "what is in the video/image" if the title or context is already in the conversation history.
- A media title like "Brand intro reel for Instagram" is enough context — write the caption using it.
- When the user references "the approved video/image", "the rejected one", or similar — find the MOST RECENTLY mentioned media item matching that description in conversation history. Use that specific item, not an older one.
- When context shows a media item with a known platform (e.g. "LinkedIn Product Launch Graphic" on LinkedIn) — infer the platform from it. Never ask for the platform if it is already clear from the item's name or platform field in context.
- If asked to retrieve or review a caption previously written, search conversation history first. If found, return it exactly. If not found, say: "I don't have a saved caption for that — would you like me to write one now?" Never invent a caption you did not write in this conversation.
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

    # Layer 1: Structured fields from Firestore brands doc (always loaded)
    brand_info = get_brand_info(brand_id) or {}
    brand_name = brand_info.get("brand_name", "")

    # Layer 2: Brand context embedding summary (always loaded)
    brand_summary = get_latest_brand_context(brand_id) or ""

    # Layer 3: On-demand — fetch any missing fields from full brand JSON
    fetch_fields = ["tone", "brand_voice", "target_audience", "vision",
                    "content_strategy", "social_media_guidelines", "hashtags",
                    "content_pillars", "brand_archetype"]
    extra_fields = {}
    for field in fetch_fields:
        if not brand_info.get(field):
            val = get_brand_field(brand_id, field)
            if val:
                extra_fields[field] = val

    def _get(key):
        return brand_info.get(key) or extra_fields.get(key, "")

    brand_context_parts = [
        f"Brand Name: {brand_name}",
        f"Mission: {_get('mission')}",
        f"Vision: {_get('vision')}",
        f"Values: {_get('values')}",
        f"Target Audience: {_get('target_audience')}",
        f"Tone: {_get('tone')}",
        f"Brand Voice: {_get('brand_voice')}",
        f"Brand Archetype: {_get('brand_archetype')}",
        f"Content Pillars: {_get('content_pillars')}",
        f"Content Strategy: {_get('content_strategy')}",
        f"Social Media Style: {_get('social_media_guidelines')}",
        f"Hashtag Strategy: {_get('hashtags')}",
    ]
    # Only include lines that have actual values
    brand_context = "\n".join(line for line in brand_context_parts if not line.endswith(": ") and not line.endswith(": []") and not line.endswith(": None"))
    if brand_summary:
        brand_context += f"\n\nBrand Context Summary:\n{brand_summary}"
    if missing_brand_info:
        brand_context += f"\n\nAdditional info from user: {missing_brand_info}"

    platform_note = f"Platform: {platform}" if platform else "Platform: not specified — infer from context or ask once."
    topic_note = f"Topic/Subject: {topic}" if topic else ""

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    routing_section = (
        f"\n\n## Recent Context\n"
        f"The following was just discussed — use it to understand what media or topic the user is referring to:\n"
        f"{routing_context}"
    ) if routing_context else ""

    response = client.messages.create(
        model=MODEL_SONNET,
        system=CAPTION_SYSTEM_PROMPT + f"\n\n## Brand Context\n{brand_context}\n\n{platform_note}\n{topic_note}" + routing_section,
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