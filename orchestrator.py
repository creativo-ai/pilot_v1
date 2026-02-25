import json
from llm_client import get_claude_client
from data_layer_vertexAI import get_brand_info, search_brand_book
from agents import (
    write_email,
    talk_agent,
    list_pending_media,
    approve_media,
    reject_media,
    search_docs_agent,
    search_media_agent,
    brand_update_agent,
    write_caption_agent
)

# Agents that generate final content — return their result directly, skip Claude's rewrite
CONTENT_AGENTS = {"write_email", "write_caption", "search_docs"}

# Agents that perform DB actions — result passed back to Claude to summarize
ACTION_AGENTS = {"list_pending_media", "approve_media", "reject_media", "search_media", "update_brand"}

TOOLS = [
    {
        "name": "write_email",
        "description": "Write a brand-aligned email to a specific audience.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_goal": {"type": "string", "description": "The purpose of the email"},
                "audience": {"type": "string", "description": "Target audience of the email"}
            },
            "required": ["email_goal"]
        }
    },
    {
        "name": "list_pending_media",
        "description": "List media that are pending approval for the user.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "approve_media",
        "description": "Approve one or more media by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "media_ids": {"type": "array", "items": {"type": "string"}, "description": "List of media IDs to approve"}
            },
            "required": ["media_ids"]
        }
    },
    {
        "name": "reject_media",
        "description": "Reject one or more media by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "media_ids": {"type": "array", "items": {"type": "string"}, "description": "List of media IDs to reject"}
            },
            "required": ["media_ids"]
        }
    },
    {
        "name": "search_media",
        "description": "Search and retrieve all media (images and videos) for the user. Use when the user asks about their media history, current status of media, or wants to filter by status (pending/approved/rejected), type (image/video), or platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status: pending, approved, or rejected."},
                "media_type": {"type": "string", "description": "Filter by type: media or video."},
                "platform": {"type": "string", "description": "Filter by platform: Instagram, LinkedIn, TikTok, Facebook, YouTube, Website, Twitter."}
            },
            "required": []
        }
    },
    {
        "name": "write_caption",
        "description": "Write a social media post or caption for a specific platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "description": "Target platform: Instagram, LinkedIn, Facebook, TikTok, Twitter, YouTube"},
                "topic": {"type": "string", "description": "The topic or subject of the post"}
            },
            "required": ["platform", "topic"]
        }
    },
    {
        "name": "update_brand",
        "description": "Update or add brand information such as name, mission, vision, values, tone, voice, target audience, or any other brand identity fields.",
        "input_schema": {
            "type": "object",
            "properties": {
                "update_request": {"type": "string", "description": "The user's brand update request in their own words"}
            },
            "required": ["update_request"]
        }
    },
    {
        "name": "search_docs",
        "description": "Search documentation to answer how-to or platform usage questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's question"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_brand",
        "description": "Get the current brand information including colors, tone, mission, vision, values, and target audience. Use this whenever the user asks about brand settings, colors, tone, or any brand identity field.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

AVAILABLE_AGENTS = {
    "write_email": write_email,
    "list_pending_media": list_pending_media,
    "approve_media": approve_media,
    "reject_media": reject_media,
    "search_media": search_media_agent,
    "write_caption": write_caption_agent,
    "update_brand": brand_update_agent,
    "search_docs": search_docs_agent,
    "get_brand": None  # handled inline in orchestrate()
}


def build_system_prompt(brand_info: dict, brand_context_chunks: list) -> str:
    brand_identity = json.dumps({
        "brand_name": brand_info.get("brand_name"),
        "tone": brand_info.get("tone"),
        "mission": brand_info.get("mission"),
        "vision": brand_info.get("vision"),
        "values": brand_info.get("values"),
        "target_audience": brand_info.get("target_audience")
    }, indent=2)

    brand_book_context = "\n\n---\n\n".join(
        [f"## {c.get('title', '')}\n{c.get('content', '')}" for c in brand_context_chunks]
    ) if brand_context_chunks else "No additional brand book context retrieved."

    return f"""
        You are a helpful AI assistant for a marketing agency.

        ## Brand Identity (cached at session start)
        {brand_identity}

        ## Relevant Brand Book Sections
        {brand_book_context}

        ## Tool Usage Rules
        - ALWAYS use "get_brand" when the user asks about ANY brand setting: colors, tone, voice, mission, vision, values, target audience. NEVER answer brand questions from memory or from the cached brand identity above.
        - Use "list_pending_media" when the user asks specifically about pending media needing approval.
        - Use "update_brand" when the user wants to change or add any brand information.
        - Use "search_media" for media history, status, filtering by type/platform/status. ALWAYS use this tool, never answer from memory.
        - If the user wants to approve or reject media by position or description instead of ID, MUST call "search_media" first to get IDs, then approve/reject.
        - Use "approve_media" or "reject_media" to approve or reject media.
        - Use "write_caption" for social media posts or captions.
        - Use "write_email" to write or draft an email.
        - Use "search_docs" ONLY for platform how-to questions (sign up, sign in, subscription).
        """


def orchestrate(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    stream: bool = False,
    **kwargs
):
    client = get_claude_client()

    # Guard against empty input crashing embed_text
    if not user_input or not user_input.strip():
        return "I didn't catch that — could you say that again?"

    brand_info = get_brand_info(brand_id) or {}
    brand_chunks = search_brand_book(brand_id=brand_id, query=user_input.strip(), top_k=3)
    print(f"  Brand context: {len(brand_chunks)} chunk(s) retrieved from brand book")
    system_prompt = build_system_prompt(brand_info, brand_chunks)

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    MAX_STEPS = 5
    step = 0
    last_text_response = None

    while step < MAX_STEPS:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=system_prompt,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            max_tokens=1000
        )

        print(f"\nStep {step + 1} — Stop reason: {response.stop_reason}")
        step += 1

        # Claude finished — stream or return final text
        if response.stop_reason == "end_turn":
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    if stream:
                        return _stream_text(block.text.strip())
                    return block.text.strip()
            return last_text_response or "Done."

        # Claude wants to use a tool
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    print(f"  Tool called: {tool_name} | Args: {tool_args}")

                    # Handle get_brand inline — always fetch fresh from DB
                    if tool_name == "get_brand":
                        fresh_brand = get_brand_info(brand_id) or {}
                        result = json.dumps(fresh_brand, indent=2)
                        last_text_response = None  # let Claude format this response

                    else:
                        agent_fn = AVAILABLE_AGENTS.get(tool_name)
                        if agent_fn:
                            result = agent_fn(
                                user_input=user_input,
                                conversation_history=conversation_history,
                                user_id=user_id,
                                brand_id=brand_id,
                                client=client,
                                brand_chunks=brand_chunks,
                                **tool_args
                            )
                            last_text_response = result
                            # Content agents produce the final answer — return immediately
                            if tool_name in CONTENT_AGENTS:
                                if stream:
                                    return _stream_text(result)
                                return result
                        else:
                            result = f"Tool '{tool_name}' not found."

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

    return last_text_response or "Could not complete the request."


def _stream_text(text: str):
    """Simulate streaming by yielding the final text as a generator."""
    import time
    for char in text:
        print(char, end="", flush=True)
        time.sleep(0.01)
    print()
    return text