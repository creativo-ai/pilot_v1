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

# Agents that perform DB actions — no streaming, return instantly
ACTION_AGENTS = {"list_pending_media", "approve_media", "reject_media", "search_media", "update_brand"}

# Agents that generate text — support streaming
STREAMING_AGENTS = {"write_email", "write_caption", "search_docs", "talk"}

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
        "description": "Approve one or more media items by their exact ID. Only call this after confirming the item is pending via search_media.",
        "input_schema": {
            "type": "object",
            "properties": {
                "media_ids": {"type": "array", "items": {"type": "string"}, "description": "List of media IDs to approve — must be pending status only"}
            },
            "required": ["media_ids"]
        }
    },
    {
        "name": "reject_media",
        "description": "Reject one or more media items by their exact ID. Only call this after confirming the item is pending via search_media.",
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
                "media_type": {"type": "string", "description": "Filter by type: must be exactly 'image' or 'video'. Never use 'media'."},
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
    "search_docs": search_docs_agent
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
        You are an internal media management tool for Creativo, a marketing agency.
        You help Creativo's team members review, approve, reject, and search media assets belonging to their clients.
        The active client brand is shown in the brand identity below — all media belongs to that client.
        Be direct and efficient — you are an internal tool used by agency staff, not a customer-facing assistant.

        ## Brand Identity (always available)
        {brand_identity}

        ## Relevant Brand Book Sections
        {brand_book_context}

        ## Media Rules

        Only PENDING items can be approved or rejected. Status is final once set.

        When the user references media by position (first, second, last, etc.):
        - If the previous response already listed media items, resolve the position from THAT list directly — do not call search_media again.
        - Call search_media only if you do not already have the list in context.
        - After resolving the position, check the item's status from the list.
        - If the resolved item is NOT pending (already approved or rejected), do NOT act on it.
          Instead ask: "The last image (image_XXX - [title]) is already [status]. Did you mean the last PENDING image (image_YYY - [title])?"
          Wait for user confirmation before taking any action.
        - Only act immediately when the resolved item is clearly pending.

        When the user says "from the pending ones" or similar, re-resolve positions against only the pending subset.

        Always include the item title AND platform in your approval/rejection summary so other agents can use that context.

        If the user wants to use a rejected item on a different platform, explain that re-platforming requires uploading a new version.

        - Use search_docs only for platform how-to questions.
        """


def orchestrate(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    stream: bool = False,
    routing_context: str = None,
    last_turn: dict = None,
    **kwargs
):
    client = get_claude_client()

    # Guard against empty input crashing embed_text
    if not user_input or not user_input.strip():
        return "I didn't catch that — could you say that again?"

    brand_info = get_brand_info(brand_id) or {}
    brand_chunks = search_brand_book(brand_id=brand_id, query=user_input.strip(), top_k=3)
    system_prompt = build_system_prompt(brand_info, brand_chunks)

    # Inject routing context so agent knows what was just discussed (e.g. which media was listed)
    if last_turn and last_turn.get("response"):
        system_prompt += (
            "\n\n## Previous Turn\n"
            f"The user previously said: {last_turn.get('user', '')}\n"
            f"You (or a peer agent) responded:\n{last_turn['response']}\n\n"
            "Use this to resolve ALL references in the current message before calling any tool.\n"
            "Ordinal references ('first', 'second', 'last', 'third') refer to the numbered position "
            "in the list shown in the response above — NOT to pending-only position.\n"
            "CRITICAL: If the previous response already listed the media items, resolve ordinals "
            "from that list directly. Do NOT call search_media to re-fetch the list just to resolve ordinals.\n"
            "Only call search_media if you genuinely need data not present in the previous response."
        )
    elif routing_context:
        system_prompt += (
            "\n\n## Recent Conversation Context\n"
            + routing_context
        )

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

                    agent_fn = AVAILABLE_AGENTS.get(tool_name)
                    if agent_fn:
                        # Stream text agents on final call, not intermediate steps
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